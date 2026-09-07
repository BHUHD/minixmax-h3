"""H3 DiT attention kernels on Ascend 910C (operator layer).

Targets 16-way 1080P: Q_local=9360, KV_full=149760, H=56, D=128.
Backends (H3_FA_BACKEND):
  infer       - npu_fused_infer_attention_score (default)
  infer_v2    - npu_fused_infer_attention_score_v2 BF16
  fusion      - npu_fusion_attention
  mindie      - MindIE-SD attention_forward runtime dispatch
  mindie_pfa  - npu_prompt_flash_attention via MindIE-SD
  mindie_fas  - fused_attn_score via MindIE-SD (ATB path)
  h3_gather_fa / h3_gf / gather_fa - self-hosted workspace gather + FA fusion
  ascend_c    - custom Ascend C H3GatherFlashAttention kernel (install custom_opp first)
  ring        - ring+LSE (experimental, HCCL P2P)
"""
from __future__ import annotations

import os
from typing import Optional

import torch

from h3_npu.ops.attention import _can_use_fa, _log

_LOGGED: set[str] = set()


def _get_backend() -> str:
    return os.environ.get("H3_FA_BACKEND", "infer").strip().lower() or "infer"


def _infer_bnsd(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    heads: int,
    scale: float,
    *,
    lse: bool = False,
) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
    import torch_npu

    kw = dict(
        num_heads=heads,
        input_layout="BNSD",
        scale=scale,
        pre_tokens=int(os.environ.get("H3_FA_PRE_TOKENS", "65536")),
        next_tokens=int(os.environ.get("H3_FA_NEXT_TOKENS", "65536")),
    )
    if lse:
        kw["softmax_lse_flag"] = True
    inner = os.environ.get("H3_FA_INNER_PRECISE")
    if inner is not None and inner != "":
        try:
            r = torch_npu.npu_fused_infer_attention_score(
                q, k, v, inner_precise=int(inner), **kw
            )
            return (r[0], r[1] if len(r) > 1 and r[1].numel() else None)
        except TypeError:
            pass
    r = torch_npu.npu_fused_infer_attention_score(q, k, v, **kw)
    out = r[0]
    lse_t = r[1] if lse and len(r) > 1 and r[1].numel() else None
    return out, lse_t


def _infer_v2_bnsd(q, k, v, heads: int, scale: float) -> torch.Tensor:
    import torch_npu

    kw = dict(
        num_query_heads=heads,
        num_key_value_heads=heads,
        input_layout="BNSD",
        softmax_scale=scale,
        pre_tokens=int(os.environ.get("H3_FA_PRE_TOKENS", "65536")),
        next_tokens=int(os.environ.get("H3_FA_NEXT_TOKENS", "65536")),
    )
    sparse = os.environ.get("H3_FA_V2_SPARSE_MODE")
    if sparse is not None and sparse != "":
        kw["sparse_mode"] = int(sparse)
    return torch_npu.npu_fused_infer_attention_score_v2(q, k, v, **kw)[0]


def _fusion_bnsd(q, k, v, heads: int, scale: float) -> torch.Tensor:
    import torch_npu

    sync = os.environ.get("H3_FA_FUSION_SYNC", "0") == "1"
    ip = int(os.environ.get("H3_FA_INNER_PRECISE", "0") or "0")
    return torch_npu.npu_fusion_attention(
        q,
        k,
        v,
        heads,
        "BNSD",
        pse=None,
        padding_mask=None,
        atten_mask=None,
        scale=scale,
        keep_prob=1.0,
        pre_tockens=int(os.environ.get("H3_FA_PRE_TOKENS", "2147483647")),
        next_tockens=int(os.environ.get("H3_FA_NEXT_TOKENS", "2147483647")),
        inner_precise=ip,
        sparse_mode=0,
        gen_mask_parallel=True,
        sync=sync,
    )[0]


def _mindie_bnsd(q, k, v, scale: float, *, op_type: Optional[str] = None) -> torch.Tensor:
    from mindiesd.layers.flash_attn.attention_forward import attention_forward

    opt_mode = os.environ.get("H3_MINDIE_FA_MODE", "runtime").strip() or "runtime"
    kwargs: dict = dict(
        scale=scale,
        fused=True,
        head_first=True,
        opt_mode=opt_mode,
        layout="BNSD",
    )
    if op_type:
        kwargs["op_type"] = op_type
    elif os.environ.get("H3_MINDIE_FA_OP"):
        kwargs["op_type"] = os.environ["H3_MINDIE_FA_OP"].strip()
    return attention_forward(q, k, v, **kwargs)


def _prompt_flash_bnsd(q, k, v, heads: int, scale: float) -> torch.Tensor:
    import torch_npu

    return torch_npu.npu_prompt_flash_attention(
        q,
        k,
        v,
        input_layout="BNSD",
        scale_value=scale,
        pre_tokens=int(os.environ.get("H3_FA_PRE_TOKENS", "2147483647")),
        next_tokens=int(os.environ.get("H3_FA_NEXT_TOKENS", "2147483647")),
        num_heads=heads,
    )


def _merge_lse(
    acc_o: torch.Tensor,
    acc_lse: torch.Tensor,
    blk_o: torch.Tensor,
    blk_lse: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Merge BNSD FA blocks via LSE (vLLM-Omni / FlashAttention stable form)."""
    import torch.nn.functional as F

    o = acc_o.float().transpose(1, 2)
    bo = blk_o.float().transpose(1, 2)
    lo = acc_lse.float().transpose(1, 2)
    ln = blk_lse.float().transpose(1, 2)
    o = o - torch.sigmoid(ln - lo) * (o - bo)
    lo = lo - F.logsigmoid(lo - ln)
    return o.transpose(1, 2), lo.transpose(1, 2)


def ring_attention_lse_bnsd(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    *,
    scale: Optional[float] = None,
) -> torch.Tensor:
    """Ring KV with hardware FA + softmax_lse_flag (no full KV gather)."""
    from h3_npu.runtime.dist import is_distributed, state as dist_state

    if not is_distributed():
        o, _ = _infer_bnsd(q, k, v, q.shape[1], (q.shape[-1] ** -0.5) if scale is None else scale)
        return o

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

    acc_o: Optional[torch.Tensor] = None
    acc_lse: Optional[torch.Tensor] = None
    ops: list = []

    for step in range(ws):
        if step > 0:
            for req in ops:
                req.wait()
            ops.clear()
            k_ring, recv_k = recv_k, k_ring
            v_ring, recv_v = recv_v, v_ring

        if step + 1 < ws:
            ops = dist.batch_isend_irecv(
                [
                    dist.P2POp(dist.isend, k_ring, send_rank),
                    dist.P2POp(dist.isend, v_ring, send_rank),
                    dist.P2POp(dist.irecv, recv_k, recv_rank),
                    dist.P2POp(dist.irecv, recv_v, recv_rank),
                ]
            )

        kb = k_ring.unsqueeze(0) if k_ring.dim() == 3 else k_ring
        vb = v_ring.unsqueeze(0) if v_ring.dim() == 3 else v_ring
        blk_o, blk_lse = _infer_bnsd(q, kb, vb, h, scale, lse=True)
        if blk_lse is None:
            raise RuntimeError("ring FA requires softmax_lse_flag support on this CANN build")
        if acc_o is None:
            acc_o, acc_lse = blk_o, blk_lse
        else:
            acc_o, acc_lse = _merge_lse(acc_o, acc_lse, blk_o, blk_lse)

    for req in ops:
        req.wait()
    return acc_o.to(q.dtype)


def h3_attention_bnsd(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    *,
    scale: Optional[float] = None,
    gathered: bool = False,
) -> torch.Tensor:
    """Entry for H3 DiT: q/k/v [1,H,S,D]. ``gathered=True`` means KV is already global."""
    if q.dim() != 4:
        raise ValueError("expected [1,H,S,D] BNSD")
    _b, h, s, d = q.shape
    s_kv = k.shape[2]
    scale = (d**-0.5) if scale is None else scale
    backend = _get_backend()
    tag = f"h3_h{h}_sq{s}_sk{s_kv}_{backend}"

    if not (q.device.type == "npu" and _can_use_fa(h, s) and s_kv % 8 == 0):
        import torch.nn.functional as F

        _log(tag, f"[h3_fa] SDPA fallback heads={h} q={s} kv={s_kv}")
        return F.scaled_dot_product_attention(q, k, v, scale=scale)

    if backend in ("h3_gather_fa", "h3_gf", "gather_fa"):
        from h3_npu.ops.gather_fa import gather_fa_bnsd

        out = gather_fa_bnsd(q, k, v, scale=scale, gathered=gathered)
        _log(tag, f"[h3_fa] h3_gather_fa mode={os.environ.get('H3_GATHER_FA_MODE', 'fused')}")
        return out

    if backend == "ascend_c":
        from h3_npu.ops.gather_fa_ascend_c import gather_fa_native

        out = gather_fa_native(q, k, v, scale)
        _log(tag, "[h3_fa] Ascend C H3GatherFlashAttention custom opp")
        return out

    if backend == "ring":
        if gathered:
            o, _ = _infer_bnsd(q, k, v, h, scale, lse=False)
        else:
            o = ring_attention_lse_bnsd(q, k, v, scale=scale)
        _log(tag, "[h3_fa] ring+LSE BNSD")
        return o

    if backend == "fusion":
        out = _fusion_bnsd(q, k, v, h, scale)
        _log(tag, "[h3_fa] npu_fusion_attention BNSD")
        return out

    if backend == "infer_v2":
        out = _infer_v2_bnsd(q, k, v, h, scale)
        _log(tag, "[h3_fa] npu_fused_infer_attention_score_v2 BNSD")
        return out

    if backend in ("mindie", "mindie_runtime"):
        out = _mindie_bnsd(q, k, v, scale)
        _log(tag, f"[h3_fa] MindIE-SD attention_forward mode={os.environ.get('H3_MINDIE_FA_MODE', 'runtime')}")
        return out

    if backend == "mindie_fas":
        out = _mindie_bnsd(q, k, v, scale, op_type="fused_attn_score")
        _log(tag, "[h3_fa] MindIE-SD fused_attn_score (ATB)")
        return out

    if backend == "mindie_pfa":
        try:
            out = _mindie_bnsd(q, k, v, scale, op_type="prompt_flash_attn")
        except Exception:
            out = _prompt_flash_bnsd(q, k, v, h, scale)
        _log(tag, "[h3_fa] MindIE-SD prompt_flash_attention")
        return out

    if backend == "pfa":
        out = _prompt_flash_bnsd(q, k, v, h, scale)
        _log(tag, "[h3_fa] npu_prompt_flash_attention BNSD")
        return out

    out, _ = _infer_bnsd(q, k, v, h, scale, lse=False)
    _log(tag, "[h3_fa] npu_fused_infer_attention_score BNSD")
    return out
