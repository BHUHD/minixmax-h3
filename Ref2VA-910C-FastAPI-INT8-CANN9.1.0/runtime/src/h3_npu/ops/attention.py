"""Attention: ACLNN fused infer, BNSD-native (no full-KV transpose).

910C FlashAttentionScore is silent-wrong when head_num is not a multiple of 8
(Ulysses splits 56 → 7). Do not cache a global FA-ok flag across head counts.

16-die 1080P: keep Q/K/V as [B, N, S, D] so FA does not copy ~4 GiB KV/layer.
"""
from __future__ import annotations

import os
from typing import Optional

import torch
import torch.nn.functional as F

_LOGGED: set[str] = set()
_INT8_KV: Optional[bool] = None


def _can_use_fa(heads: int, seq: int) -> bool:
    if os.environ.get("H3_NPU_FORCE_SDPA", "0") == "1":
        return False
    if os.environ.get("H3_NPU_FORCE_FA", "0") == "1":
        return True
    return heads >= 16 and heads % 8 == 0 and seq >= 32 and seq % 8 == 0


def _log(tag: str, msg: str) -> None:
    if tag not in _LOGGED:
        print(msg, flush=True)
        _LOGGED.add(tag)


def _fused_infer_bnsd(qb, kb, vb, heads: int, scale: float):
    import torch_npu

    kw = dict(
        num_heads=heads,
        input_layout="BNSD",
        scale=scale,
        pre_tokens=65536,
        next_tokens=65536,
    )
    inner = os.environ.get("H3_FA_INNER_PRECISE")
    if inner is not None and inner != "":
        try:
            return torch_npu.npu_fused_infer_attention_score(
                qb, kb, vb, inner_precise=int(inner), **kw
            )[0]
        except TypeError:
            pass
    return torch_npu.npu_fused_infer_attention_score(qb, kb, vb, **kw)[0]


def _try_int8_kv_bnsd(qb, kb, vb, heads: int, scale: float):
    """A3 path: BF16 Q + INT8 KV, pertoken dequant (quant_mode=1)."""
    global _INT8_KV
    if _INT8_KV is False or os.environ.get("H3_FA_INT8_KV", "0") != "1":
        return None
    try:
        import torch_npu

        d = kb.shape[-1]
        k2 = kb.reshape(-1, d).contiguous()
        v2 = vb.reshape(-1, d).contiguous()
        k_i8, k_sc = torch_npu.npu_dynamic_quant(k2)
        v_i8, v_sc = torch_npu.npu_dynamic_quant(v2)
        k_i8 = k_i8.view_as(kb)
        v_i8 = v_i8.view_as(vb)
        k_sc = k_sc.view(*kb.shape[:-1])
        v_sc = v_sc.view(*vb.shape[:-1])
        if k_sc.dtype not in (torch.float16, torch.bfloat16):
            k_sc = k_sc.to(torch.bfloat16)
            v_sc = v_sc.to(torch.bfloat16)
        out = torch_npu.npu_fused_infer_attention_score_v2(
            qb,
            k_i8,
            v_i8,
            num_query_heads=heads,
            num_key_value_heads=heads,
            input_layout="BNSD",
            softmax_scale=scale,
            key_quant_mode=1,
            value_quant_mode=1,
            dequant_scale_key=k_sc,
            dequant_scale_value=v_sc,
        )[0]
        if _INT8_KV is None:
            torch.npu.synchronize()
            _INT8_KV = True
            _log("int8kv", "[h3_npu] attention FA INT8-KV fused_infer_v2 mode=1")
        return out
    except Exception as exc:
        if _INT8_KV is None:
            _log("int8kv", f"[h3_npu] INT8-KV FA off ({exc})")
        _INT8_KV = False
        return None


def fusion_attention_bnsd(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    *,
    scale: Optional[float] = None,
) -> torch.Tensor:
    """q/k/v already [B, N, S, D]. Returns [B, N, S_q, D] (still BNSD)."""
    if q.dim() != 4:
        raise ValueError("expected 4D qkv [B, N, S, D]")
    _b, h, s, d = q.shape
    s_kv = k.shape[2]
    scale = (d**-0.5) if scale is None else scale
    tag = f"bnsd_h{h}_sq{s}_sk{s_kv}"
    use_fa = q.device.type == "npu" and _can_use_fa(h, s) and (s_kv % 8 == 0)
    if not use_fa:
        _log(tag, f"[h3_npu] attention SDPA BNSD heads={h} q={s} kv={s_kv}")
        return F.scaled_dot_product_attention(q, k, v, scale=scale)

    out = _try_int8_kv_bnsd(q, k, v, h, scale)
    if out is not None:
        _log(tag, f"[h3_npu] attention FA INT8-KV BNSD heads={h} q={s} kv={s_kv}")
        return out
    try:
        out = _fused_infer_bnsd(q, k, v, h, scale)
        _log(tag, f"[h3_npu] attention FA infer-BNSD-native heads={h} q={s} kv={s_kv}")
        return out
    except Exception as exc:
        _log(tag, f"[h3_npu] native BNSD FA fallback SDPA: {exc}")
        if max(s, s_kv) >= 2048:
            raise RuntimeError(f"FA failed q={s} kv={s_kv} heads={h}: {exc}") from exc
        return F.scaled_dot_product_attention(q, k, v, scale=scale)


def fusion_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    *,
    scale: Optional[float] = None,
    force_sdpa: bool = False,
) -> torch.Tensor:
    """
    q: [B, S_q, H, D]; k/v: [B, S_kv, H, D]. Returns [B, S_q, H, D].
    Prefer fusion_attention_bnsd when tensors are already BNSD.
    """
    if q.dim() != 4:
        raise ValueError("expected 4D qkv [B, S, H, D]")
    _b, s, h, d = q.shape
    s_kv = k.shape[1]
    scale = (d**-0.5) if scale is None else scale
    tag = f"h{h}_sq{s}_sk{s_kv}"
    use_fa = (not force_sdpa) and q.device.type == "npu" and _can_use_fa(h, s) and (s_kv % 8 == 0)

    if use_fa:
        try:
            qb = q.transpose(1, 2).contiguous()
            kb = k.transpose(1, 2).contiguous()
            vb = v.transpose(1, 2).contiguous()
            out = _try_int8_kv_bnsd(qb, kb, vb, h, scale)
            used = "INT8-KV"
            if out is None:
                out = _fused_infer_bnsd(qb, kb, vb, h, scale)
                used = "infer-BNSD"
            _log(tag, f"[h3_npu] attention FA {used} heads={h} q={s} kv={s_kv}")
            return out.transpose(1, 2).contiguous()
        except Exception as exc:  # noqa: BLE001
            _log(tag, f"[h3_npu] npu_fusion_attention fallback to SDPA ({tag}): {exc}")
            if max(s, s_kv) >= 2048 and os.environ.get("H3_NPU_FORCE_SDPA", "0") != "1":
                raise RuntimeError(
                    f"FA failed on large attention q={s} kv={s_kv} heads={h}; "
                    f"SDPA would OOM. Original: {exc}"
                ) from exc

    _log(tag, f"[h3_npu] attention SDPA heads={h} q={s} kv={s_kv} device={q.device.type}")
    qn = q.transpose(1, 2)
    kn = k.transpose(1, 2)
    vn = v.transpose(1, 2)
    out = F.scaled_dot_product_attention(qn, kn, vn, scale=scale)
    return out.transpose(1, 2).contiguous()
