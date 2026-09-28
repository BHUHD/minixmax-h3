"""H3 gather + Flash-Attention fusion (self-hosted, no vendor ticket).

Strategies (H3_GATHER_FA_MODE):
  fused         - reuse KV workspace + single infer_v2 FA (default)
  tiled         - workspace gather + seq-tiled FA with online LSE merge
  head_chunk    - workspace gather + per-head-chunk FA (8 heads/chunk)
  shard_lse     - 按 rank broadcast 本地 KV 片 + 本地 FA/LSE merge（无全量 gather）

Env:
  H3_GATHER_FA_TILE_SK   - seq tile for ``tiled`` mode (default 18720)
  H3_GATHER_FA_HEAD_CHUNK - heads per chunk for ``head_chunk`` (default 8)
"""
from __future__ import annotations

import os
from typing import Optional

import torch

from h3_npu.ops.fa_h3 import _infer_bnsd, _infer_v2_bnsd, _merge_lse
from h3_npu.runtime.dist import is_distributed, state as dist_state

_LOGGED = False


def _mode() -> str:
    return (os.environ.get("H3_GATHER_FA_MODE") or "fused").strip().lower()


def _log_once(msg: str) -> None:
    global _LOGGED
    if not _LOGGED:
        print(msg, flush=True)
        _LOGGED = True


def _fa_backend() -> str:
    return (os.environ.get("H3_FA_BACKEND") or "infer").strip().lower()


def _fa_single(qb: torch.Tensor, kb: torch.Tensor, vb: torch.Tensor, heads: int, scale: float) -> torch.Tensor:
    backend = _fa_backend()
    if backend == "ascend_c":
        from h3_npu.ops.gather_fa_ascend_c import gather_fa_native

        return gather_fa_native(qb, kb, vb, scale, num_heads=heads)
    try:
        return _infer_v2_bnsd(qb, kb, vb, heads, scale)
    except Exception:
        out, _ = _infer_bnsd(qb, kb, vb, heads, scale, lse=False)
        return out


def _fa_tiled_lse(
    qb: torch.Tensor,
    kb: torch.Tensor,
    vb: torch.Tensor,
    heads: int,
    scale: float,
    tile_sk: int,
) -> torch.Tensor:
    _b, _h, sk, _d = kb.shape
    tile_sk = max(8, min(tile_sk, sk))
    acc_o: Optional[torch.Tensor] = None
    acc_lse: Optional[torch.Tensor] = None
    for t0 in range(0, sk, tile_sk):
        t1 = min(t0 + tile_sk, sk)
        kb_t = kb[:, :, t0:t1, :].contiguous()
        vb_t = vb[:, :, t0:t1, :].contiguous()
        blk_o, blk_lse = _infer_bnsd(qb, kb_t, vb_t, heads, scale, lse=True)
        if blk_lse is None:
            return _fa_single(qb, kb, vb, heads, scale)
        if acc_o is None:
            acc_o, acc_lse = blk_o, blk_lse
        else:
            acc_o, acc_lse = _merge_lse(acc_o, acc_lse, blk_o, blk_lse)
    assert acc_o is not None
    return acc_o.to(qb.dtype)


def _fa_head_chunked(qb: torch.Tensor, kb: torch.Tensor, vb: torch.Tensor, heads: int, scale: float, chunk: int) -> torch.Tensor:
    chunk = max(16, chunk)  # FA kernel needs >=16 heads
    if chunk >= heads:
        return _fa_single(qb, kb, vb, heads, scale)
    outs = []
    for h0 in range(0, heads, chunk):
        h1 = min(h0 + chunk, heads)
        ch = h1 - h0
        outs.append(_fa_single(qb[:, h0:h1], kb[:, h0:h1], vb[:, h0:h1], ch, scale))
    return torch.cat(outs, dim=1)


def _fa_shard_broadcast_lse(
    qb: torch.Tensor,
    k_local: torch.Tensor,
    v_local: torch.Tensor,
    heads: int,
    scale: float,
) -> torch.Tensor:
    """按 rank 轮流 broadcast 本地 KV 片，本地 FA + LSE merge；避免全量 gather 与 P2P。

    k_local/v_local: [1,H,Sk_local,D]；qb: [1,H,Sq,D]（Sq 为本地 query 长度）。
    """
    import torch.distributed as dist

    st = dist_state()
    ws, rank = st.world_size, st.rank
    k_buf = torch.empty_like(k_local)
    v_buf = torch.empty_like(v_local)
    acc_o: Optional[torch.Tensor] = None
    acc_lse: Optional[torch.Tensor] = None
    for src in range(ws):
        if src == rank:
            k_buf.copy_(k_local)
            v_buf.copy_(v_local)
        dist.broadcast(k_buf, src=src)
        dist.broadcast(v_buf, src=src)
        blk_o, blk_lse = _infer_bnsd(qb, k_buf, v_buf, heads, scale, lse=True)
        if blk_lse is None:
            # 无 LSE 时退回全量 gather 路径由调用方处理
            raise RuntimeError("shard_lse requires softmax_lse_flag")
        if acc_o is None:
            acc_o, acc_lse = blk_o, blk_lse
        else:
            acc_o, acc_lse = _merge_lse(acc_o, acc_lse, blk_o, blk_lse)
    assert acc_o is not None
    return acc_o.to(qb.dtype)


def gather_fa_bnsd(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    *,
    scale: Optional[float] = None,
    gathered: bool = False,
) -> torch.Tensor:
    """[1,H,Sq,D] Q with [1,H,Sk,D] or [H,Sk_local,D] local K/V when SP."""
    if q.dim() == 3:
        q = q.unsqueeze(0)
    if k.dim() == 3:
        k = k.unsqueeze(0)
    if v.dim() == 3:
        v = v.unsqueeze(0)
    _b, heads, sq, d = q.shape
    scale = (d**-0.5) if scale is None else scale

    mode = _mode()
    # Phase F：分片 broadcast + LSE，跳过全量 all_gather_kv
    if (not gathered) and is_distributed() and mode in ("shard_lse", "broadcast_lse"):
        _log_once(
            f"[h3_gather_fa] shard_lse H={heads} Sq={sq} Sk_local={k.shape[2]} "
            f"ws={dist_state().world_size}"
        )
        try:
            return _fa_shard_broadcast_lse(q, k, v, heads, scale)
        except Exception as exc:
            _log_once(f"[h3_gather_fa] shard_lse failed ({exc}); fallback all_gather")

    if gathered or not is_distributed():
        kb, vb = k, v
    else:
        # [1,H,S,D] -> [H,S,D] for HCCL gather (reuses dist._GATHER_BUF)
        from h3_npu.runtime.dist import all_gather_kv

        kh, vh = k[0], v[0]
        kg, vg = all_gather_kv(kh, vh, seq_dim=1)
        kb = kg.unsqueeze(0).contiguous()
        vb = vg.unsqueeze(0).contiguous()
        _log_once(
            f"[h3_gather_fa] all_gather_kv mode={mode} "
            f"H={heads} Sq={sq} Sk={kb.shape[2]} ws={dist_state().world_size}"
        )

    if mode == "tiled":
        tile = int(os.environ.get("H3_GATHER_FA_TILE_SK", "18720") or "18720")
        return _fa_tiled_lse(q, kb, vb, heads, scale, tile)
    if mode == "head_chunk":
        chunk = int(os.environ.get("H3_GATHER_FA_HEAD_CHUNK", "8") or "8")
        return _fa_head_chunked(q, kb, vb, heads, scale, chunk)
    return _fa_single(q, kb, vb, heads, scale)


def sp_gather_fa(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    *,
    scale: Optional[float] = None,
) -> torch.Tensor:
    """Production entry from dit Attention: q/k/v are [H, S_local, D] BNSD."""
    return gather_fa_bnsd(q.unsqueeze(0), k, v, scale=scale, gathered=False)[0]


try:
    import torch_npu  # noqa: F401

    @torch.library.custom_op("h3::gather_fa_bnsd", mutates_args=())
    def gather_fa_custom_op(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, scale: float) -> torch.Tensor:
        return gather_fa_bnsd(q, k, v, scale=scale, gathered=False)

    @gather_fa_custom_op.register_fake
    def _gather_fa_fake(q, k, v, scale):
        return torch.empty_like(q)
except Exception:
    pass
