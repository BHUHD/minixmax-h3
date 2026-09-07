"""HCCL process group for 8-die sequence parallel (container-only).

910C ``dist.barrier()`` / large AllReduce used to die in AICPU
``RunAicpuKfcResInitV2``. Small AIV AllReduce and ``broadcast`` work.
Gather is implemented as a broadcast ring (or ``all_gather_into_tensor`` when
that kernel accepts the tensor), not a 256K AllReduce storm.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from datetime import timedelta
from typing import Optional

import torch


@dataclass
class DistState:
    rank: int = 0
    local_rank: int = 0
    world_size: int = 1
    group: Optional[object] = None
    backend: str = "none"


_STATE = DistState()
# "into" | "broadcast" — chosen once during warmup.
_GATHER_MODE = os.environ.get("H3_HCCL_GATHER", "broadcast").strip() or "broadcast"
_GATHER_BUF: dict = {}


def state() -> DistState:
    return _STATE


def is_distributed() -> bool:
    return _STATE.world_size > 1


def _apply_hccl_env() -> None:
    """Container-only HCCL knobs. Does not touch host driver files."""
    os.environ.setdefault("HCCL_CONNECT_TIMEOUT", "1800")
    os.environ.setdefault("HCCL_WHITELIST_DISABLE", "1")
    os.environ.setdefault("HCCL_OP_EXPANSION_MODE", "AIV")
    os.environ.setdefault("HCCL_BUFFSIZE", "512")
    os.environ.setdefault("HCCL_HOST_SOCKET_PORT_RANGE", "auto")
    os.environ.setdefault("HCCL_NPU_SOCKET_PORT_RANGE", "auto")
    os.environ.setdefault("HCCL_INTRA_PCIE_ENABLE", "1")
    os.environ.setdefault("HCCL_INTRA_ROCE_ENABLE", "0")
    os.environ.setdefault("TASK_QUEUE_ENABLE", "1")


_ATTACHED = False


def attach_npu() -> None:
    """Serialize TSD/device open. Concurrent ``set_device`` races 507033 on 910C."""
    global _ATTACHED
    if _ATTACHED:
        return
    import fcntl
    import time
    from pathlib import Path

    lock = Path(os.environ.get("H3_NPU_OPEN_LOCK", "/workspace/out/npu_open.lock"))
    lock.parent.mkdir(parents=True, exist_ok=True)
    rank = os.environ.get("RANK", "?")
    vis = os.environ.get("ASCEND_RT_VISIBLE_DEVICES")
    try:
        rank_i = int(rank)
    except ValueError:
        rank_i = 0
    # Spawn is already staggered; a short extra offset keeps TSD from colliding.
    time.sleep(min(0.05 * rank_i, 0.8))
    with open(lock, "w") as fh:
        fcntl.flock(fh, fcntl.LOCK_EX)
        import torch_npu  # noqa: F401  # must be inside the lock; TSD open is not reentrant
        last_err: Optional[BaseException] = None
        for attempt in range(6):
            try:
                torch.npu.set_device(0)
                t = torch.zeros(8, device="npu:0", dtype=torch.float32)
                torch.npu.synchronize()
                del t
                last_err = None
                break
            except Exception as exc:  # noqa: BLE001
                last_err = exc
                print(f"[hccl] rank={rank} vis={vis} set_device retry {attempt+1}/6: {exc}", flush=True)
                time.sleep(1.5 * (attempt + 1))
        if last_err is not None:
            raise last_err
        time.sleep(0.04)
    _ATTACHED = True
    print(f"[hccl] rank={rank} vis={vis} device open ok", flush=True)


def _wait_peers(stage: str, world: int, rank: int, timeout: float | None = None) -> None:
    """Filesystem barrier so every rank reaches HCCL init together."""
    import time
    from pathlib import Path

    if timeout is None:
        timeout = float(os.environ.get("HCCL_CONNECT_TIMEOUT", "1800"))
        if stage == "attached":
            timeout = max(timeout, 1200.0)
    root = Path(os.environ.get("H3_SYNC_DIR", "/workspace/out/h3_sync")) / stage
    root.mkdir(parents=True, exist_ok=True)
    (root / str(rank)).write_text("ok")
    t0 = time.time()
    while time.time() - t0 < timeout:
        if sum(1 for _ in root.iterdir()) >= world:
            return
        time.sleep(0.02)
    raise RuntimeError(f"timeout waiting for {stage}: {sum(1 for _ in root.iterdir())}/{world}")


def init_hccl() -> DistState:
    """Init torch.distributed HCCL. Call after one-die pin, before DiT H2D."""
    import time

    t0 = time.time()
    attach_npu()
    print(f"[hccl] rank={os.environ.get('RANK')} attached in {time.time()-t0:.1f}s", flush=True)
    rank = int(os.environ.get("RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    _apply_hccl_env()

    if world_size <= 1:
        _STATE.rank = 0
        _STATE.local_rank = 0
        _STATE.world_size = 1
        _STATE.group = None
        _STATE.backend = "none"
        return _STATE

    import torch.distributed as dist

    if not dist.is_initialized():
        _wait_peers("attached", world_size, rank)
        t1 = time.time()
        print(f"[hccl] rank={rank} init_process_group world={world_size} (all attached)", flush=True)
        dist.init_process_group(
            backend="hccl",
            init_method="env://",
            rank=rank,
            world_size=world_size,
            device_id=torch.device("npu:0"),
            timeout=timedelta(seconds=int(os.environ.get("HCCL_CONNECT_TIMEOUT", "1800"))),
        )
        print(f"[hccl] rank={rank} process group ready in {time.time()-t1:.1f}s", flush=True)
    torch.npu.synchronize()
    _STATE.rank = dist.get_rank()
    _STATE.local_rank = 0
    _STATE.world_size = dist.get_world_size()
    _STATE.group = dist.group.WORLD
    _STATE.backend = "hccl"
    return _STATE


def allreduce_npu(t: torch.Tensor) -> torch.Tensor:
    if not is_distributed():
        return t
    import torch.distributed as dist

    x = t.contiguous()
    dist.all_reduce(x)
    return x


def _gather_into(local: torch.Tensor) -> torch.Tensor:
    import torch.distributed as dist

    x = local.contiguous()
    ws = _STATE.world_size
    key = (tuple(x.shape), str(x.dtype), ws)
    out = _GATHER_BUF.get(key)
    need = (ws, *x.shape)
    if out is None or tuple(out.shape) != need or out.dtype != x.dtype:
        out = torch.empty(need, device=x.device, dtype=x.dtype)
        _GATHER_BUF[key] = out
    dist.all_gather_into_tensor(out, x)
    return out.reshape(ws * x.shape[0], *x.shape[1:])


def _gather_broadcast(local: torch.Tensor) -> torch.Tensor:
    """One broadcast per rank into a reused packed buffer (no cat)."""
    import torch.distributed as dist

    x = local.contiguous()
    ws = _STATE.world_size
    rank = _STATE.rank
    key = (tuple(x.shape), str(x.dtype), ws)
    packed = _GATHER_BUF.get(key)
    need = (ws, *x.shape)
    if packed is None or tuple(packed.shape) != need or packed.dtype != x.dtype:
        packed = torch.empty(need, device=x.device, dtype=x.dtype)
        _GATHER_BUF[key] = packed
    packed[rank].copy_(x)
    for src in range(ws):
        dist.broadcast(packed[src], src=src)
    return packed.reshape(ws * x.shape[0], *x.shape[1:])


def _gather_list(local: torch.Tensor) -> torch.Tensor:
    """HCCL all_gather into views of a packed buffer (one collective)."""
    import torch.distributed as dist

    x = local.contiguous()
    ws = _STATE.world_size
    key = (tuple(x.shape), str(x.dtype), ws)
    packed = _GATHER_BUF.get(key)
    need = (ws, *x.shape)
    if packed is None or tuple(packed.shape) != need or packed.dtype != x.dtype:
        packed = torch.empty(need, device=x.device, dtype=x.dtype)
        _GATHER_BUF[key] = packed
    parts = [packed[i] for i in range(ws)]
    dist.all_gather(parts, x)
    return packed.reshape(ws * x.shape[0], *x.shape[1:])


def all_gather_seq(local: torch.Tensor) -> torch.Tensor:
    """Gather sequence-sharded [S_local, ...] into [S, ...] on every rank."""
    if not is_distributed():
        return local
    if _GATHER_MODE == "into":
        return _gather_into(local)
    if _GATHER_MODE == "list":
        return _gather_list(local)
    return _gather_broadcast(local)


def _gather_broadcast_axis1(local: torch.Tensor) -> torch.Tensor:
    """Gather [H, S_local, ...] → [H, S, ...] without transposing the full KV."""
    import torch.distributed as dist

    x = local.contiguous()
    ws = _STATE.world_size
    rank = _STATE.rank
    key = ("ax1", tuple(x.shape), str(x.dtype), ws)
    packed = _GATHER_BUF.get(key)
    need = (ws, *x.shape)
    if packed is None or tuple(packed.shape) != need or packed.dtype != x.dtype:
        packed = torch.empty(need, device=x.device, dtype=x.dtype)
        _GATHER_BUF[key] = packed
    packed[rank].copy_(x)
    for src in range(ws):
        dist.broadcast(packed[src], src=src)
    h, s_local = x.shape[0], x.shape[1]
    rest = x.shape[2:]
    merged_key = ("ax1m", h, s_local, rest, str(x.dtype), ws)
    merged = _GATHER_BUF.get(merged_key)
    out_shape = (h, ws * s_local, *rest)
    if merged is None or tuple(merged.shape) != out_shape or merged.dtype != x.dtype:
        merged = torch.empty(out_shape, device=x.device, dtype=x.dtype)
        _GATHER_BUF[merged_key] = merged
    merged.copy_(packed.permute(1, 0, 2, *range(3, packed.dim())).reshape(out_shape))
    return merged


def _gather_list_axis1(local: torch.Tensor) -> torch.Tensor:
    import torch.distributed as dist

    x = local.contiguous()
    ws = _STATE.world_size
    key = ("ax1l", tuple(x.shape), str(x.dtype), ws)
    packed = _GATHER_BUF.get(key)
    need = (ws, *x.shape)
    if packed is None or tuple(packed.shape) != need or packed.dtype != x.dtype:
        packed = torch.empty(need, device=x.device, dtype=x.dtype)
        _GATHER_BUF[key] = packed
    parts = [packed[i] for i in range(ws)]
    dist.all_gather(parts, x)
    h, s_local = x.shape[0], x.shape[1]
    rest = x.shape[2:]
    return packed.permute(1, 0, 2, *range(3, packed.dim())).reshape(h, ws * s_local, *rest)


def _gather_chunk_axis1(local: torch.Tensor) -> torch.Tensor:
    """Head-chunked list all_gather — keeps each collective under AICPU KFC limits."""
    chunk = int(os.environ.get("H3_HCCL_CHUNK_HEADS", "8") or "8")
    chunk = max(1, chunk)
    h = local.shape[0]
    if h <= chunk:
        return _gather_list_axis1(local)
    parts = [_gather_list_axis1(local[i : i + chunk]) for i in range(0, h, chunk)]
    return torch.cat(parts, dim=0)


def all_gather_kv(
    k: torch.Tensor, v: torch.Tensor, *, seq_dim: int = 0
) -> tuple[torch.Tensor, torch.Tensor]:
    """Single gather of concatenated K/V (one collective instead of two).

    seq_dim=0: [S_local, H, D] (legacy). seq_dim=1: [H, S_local, D] so FA can
    consume BNSD without copying the full gathered KV.
    """
    if not is_distributed():
        return k, v
    d = k.shape[-1]
    kv = torch.cat((k, v), dim=-1).contiguous()
    if seq_dim == 1:
        if _GATHER_MODE == "list":
            kv = _gather_list_axis1(kv)
        elif _GATHER_MODE == "chunk":
            kv = _gather_chunk_axis1(kv)
        else:
            kv = _gather_broadcast_axis1(kv)
    else:
        kv = all_gather_seq(kv)
    return kv[..., :d], kv[..., d:]


def warmup_collective() -> None:
    """Pick a working gather path while HBM is still empty."""
    global _GATHER_MODE
    if not is_distributed():
        return
    t = torch.ones(8, device="npu:0", dtype=torch.float32)
    t = allreduce_npu(t)
    torch.npu.synchronize()
    got = float(t[0].item())
    expect = float(_STATE.world_size)
    if abs(got - expect) > 1e-3:
        raise RuntimeError(f"HCCL warmup allreduce got {got}, expected {expect}")

    probe = torch.arange(2048 * 64, device="npu:0", dtype=torch.bfloat16).reshape(-1, 64)
    probe = probe + float(_STATE.rank)
    wanted = os.environ.get("H3_HCCL_GATHER", "broadcast").strip() or "broadcast"
    _GATHER_MODE = wanted if wanted in ("into", "list", "broadcast", "chunk") else "broadcast"
    g = all_gather_seq(probe)
    torch.npu.synchronize()
    expect_s = probe.shape[0] * _STATE.world_size
    if g.shape[0] != expect_s:
        raise RuntimeError(f"HCCL gather shape {tuple(g.shape)} expected seq={expect_s}")
    del g
    if _STATE.rank == 0:
        print(
            f"[hccl] warmup ok world={_STATE.world_size} "
            f"mode={os.environ.get('HCCL_OP_EXPANSION_MODE')} gather={_GATHER_MODE} "
            f"sum={got}",
            flush=True,
        )


def barrier() -> None:
    if not is_distributed():
        return
    t = torch.ones(1, device="npu:0", dtype=torch.float32)
    allreduce_npu(t)
    torch.npu.synchronize()


def broadcast_tensor(t: torch.Tensor, src: int = 0) -> torch.Tensor:
    if not is_distributed():
        return t
    import torch.distributed as dist

    dist.broadcast(t.contiguous(), src=src)
    return t
