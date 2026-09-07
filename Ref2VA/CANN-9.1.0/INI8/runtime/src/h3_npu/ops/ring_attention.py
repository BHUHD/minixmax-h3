"""Ring (striped) attention for sequence-parallel DiT — no full KV gather.

Mathematically identical to local-Q + full-KV FA, but streams K/V blocks around
the HCCL group and merges with online softmax.  Avoids 16× broadcast per layer.
"""
from __future__ import annotations

import os
from typing import Optional

import torch
import torch.nn.functional as F

from h3_npu.ops.attention import _can_use_fa, _fused_infer_bnsd
from h3_npu.runtime.dist import is_distributed, state as dist_state


def _fa_or_sdpa(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    heads: int,
    scale: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return (out, block_max, block_sum) for online merge. q/k/v: [1,H,S,D]."""
    _b, h, sq, d = q.shape
    sk = k.shape[2]
    if q.device.type == "npu" and _can_use_fa(h, sq) and sk % 8 == 0:
        # FA path: recompute scores stats in fp32 for merge (block is small enough).
        qf = q.float()
        kf = k.float()
        vf = v.float()
        scores = torch.matmul(qf, kf.transpose(-2, -1)) * scale
        block_max = scores.amax(dim=-1, keepdim=True)
        exp_s = torch.exp(scores - block_max)
        block_sum = exp_s.sum(dim=-1, keepdim=True).clamp(min=1e-8)
        out = torch.matmul(exp_s, vf) / block_sum
        return out.to(q.dtype), block_max.squeeze(-1), block_sum.squeeze(-1)
    qn = q.transpose(1, 2)
    kn = k.transpose(1, 2)
    vn = v.transpose(1, 2)
    scores = torch.matmul(qn.float(), kn.float().transpose(-2, -1)) * scale
    block_max = scores.amax(dim=-1, keepdim=True)
    exp_s = torch.exp(scores - block_max)
    block_sum = exp_s.sum(dim=-1, keepdim=True).clamp(min=1e-8)
    out = torch.matmul(exp_s, vn.float()) / block_sum
    return out.to(q.dtype).transpose(1, 2), block_max.squeeze(-1), block_sum.squeeze(-1)


def _merge_block(
    acc_o: torch.Tensor,
    acc_m: torch.Tensor,
    acc_l: torch.Tensor,
    block_o: torch.Tensor,
    block_m: torch.Tensor,
    block_l: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Online softmax merge. Tensors [1,H,Sq] except acc_o/block_o [1,H,Sq,D]."""
    m_new = torch.maximum(acc_m, block_m)
    exp_old = torch.exp(acc_m - m_new)
    exp_blk = torch.exp(block_m - m_new)
    l_new = acc_l * exp_old + block_l * exp_blk
    o_new = acc_o * (exp_old / l_new).unsqueeze(-1) + block_o * (exp_blk / l_new).unsqueeze(-1)
    return o_new, m_new, l_new


def ring_attention_bnsd(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    *,
    scale: Optional[float] = None,
) -> torch.Tensor:
    """q/k/v local shards [1, H, S_local, D]. Returns same shape as q."""
    if not is_distributed():
        from h3_npu.ops.attention import fusion_attention_bnsd

        return fusion_attention_bnsd(q, k, v, scale=scale)

    import torch.distributed as dist

    st = dist_state()
    ws, rank = st.world_size, st.rank
    _b, h, _sq, d = q.shape
    scale = (d**-0.5) if scale is None else scale

    send_rank = (rank + 1) % ws
    recv_rank = (rank + ws - 1) % ws

    k_ring = k.contiguous()
    v_ring = v.contiguous()
    recv_k = torch.empty_like(k_ring)
    recv_v = torch.empty_like(v_ring)

    acc_o = torch.zeros_like(q)
    acc_m = torch.full((1, h, q.shape[2]), -1e4, device=q.device, dtype=torch.float32)
    acc_l = torch.zeros((1, h, q.shape[2]), device=q.device, dtype=torch.float32)

    ops: list = []
    for step in range(ws):
        if step > 0:
            for req in ops:
                req.wait()
            ops.clear()
            k_ring, recv_k = recv_k, k_ring
            v_ring, recv_v = recv_v, v_ring

        if step + 1 < ws:
            ops.append(dist.P2POp(dist.isend, k_ring, send_rank))
            ops.append(dist.P2POp(dist.isend, v_ring, send_rank))
            ops.append(dist.P2POp(dist.irecv, recv_k, recv_rank))
            ops.append(dist.P2POp(dist.irecv, recv_v, recv_rank))
            ops = dist.batch_isend_irecv(ops)

        block_o, block_m, block_l = _fa_or_sdpa(q, k_ring.unsqueeze(0) if k_ring.dim() == 3 else k_ring, v_ring.unsqueeze(0) if v_ring.dim() == 3 else v_ring, h, scale)
        if k_ring.dim() == 3:
            block_o = block_o  # already [1,H,S,D]
        acc_o, acc_m, acc_l = _merge_block(acc_o.float(), acc_m, acc_l, block_o.float(), block_m, block_l)
        acc_o = acc_o.to(q.dtype)

    for req in ops:
        req.wait()
    return acc_o


def ring_attention_bnsd_fused(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    *,
    scale: Optional[float] = None,
) -> torch.Tensor:
    """Fast path: use native FA per ring block when merge stats come from FA."""
    if not is_distributed():
        from h3_npu.ops.attention import fusion_attention_bnsd

        return fusion_attention_bnsd(q, k, v, scale=scale)

    import torch.distributed as dist

    st = dist_state()
    ws, rank = st.world_size, st.rank
    _b, h, sq, d = q.shape
    scale = (d**-0.5) if scale is None else scale

    send_rank = (rank + 1) % ws
    recv_rank = (rank + ws - 1) % ws
    k_ring = k.contiguous()
    v_ring = v.contiguous()
    recv_k = torch.empty_like(k_ring)
    recv_v = torch.empty_like(v_ring)

    acc_o = torch.zeros(1, h, sq, d, device=q.device, dtype=torch.float32)
    acc_m = torch.full((1, h, sq), -1e4, device=q.device, dtype=torch.float32)
    acc_l = torch.zeros((1, h, sq), device=q.device, dtype=torch.float32)

    ops: list = []
    use_fa = q.device.type == "npu" and _can_use_fa(h, sq)
    for step in range(ws):
        if step > 0:
            for req in ops:
                req.wait()
            ops.clear()
            k_ring, recv_k = recv_k, k_ring
            v_ring, recv_v = recv_v, v_ring

        if step + 1 < ws:
            ops = dist.batch_isend_irecv([
                dist.P2POp(dist.isend, k_ring, send_rank),
                dist.P2POp(dist.isend, v_ring, send_rank),
                dist.P2POp(dist.irecv, recv_k, recv_rank),
                dist.P2POp(dist.irecv, recv_v, recv_rank),
            ])

        qb = q
        kb = k_ring.unsqueeze(0) if k_ring.dim() == 3 else k_ring
        vb = v_ring.unsqueeze(0) if v_ring.dim() == 3 else v_ring
        if use_fa and kb.shape[2] % 8 == 0:
            block_o = _fused_infer_bnsd(qb, kb, vb, h, scale).float()
            qf, kf = qb.float(), kb.float()
            scores = torch.matmul(qf, kf.transpose(-2, -1)) * scale
            block_m = scores.amax(dim=-1)
            block_l = torch.exp(scores - block_m.unsqueeze(-1)).sum(dim=-1)
        else:
            block_o, block_m, block_l = _fa_or_sdpa(qb, kb, vb, h, scale)
            block_o = block_o.float()

        acc_o, acc_m, acc_l = _merge_block(acc_o, acc_m, acc_l, block_o, block_m, block_l)

    for req in ops:
        req.wait()
    return acc_o.to(q.dtype)


def maybe_ring_attention_bnsd(q, k, v, *, scale=None):
    if os.environ.get("H3_ATTN_RING", "0") == "1" and is_distributed():
        return ring_attention_bnsd_fused(q, k, v, scale=scale)
    from h3_npu.ops.attention import fusion_attention_bnsd

    return fusion_attention_bnsd(q, k, v, scale=scale)
