"""Float8-safe cast for 910C: aclnnCast lacks DT_FLOAT8_E4M3FN.

Prefer integer bit-decode (NPU-capable) over `Tensor.to` through CPU.
"""
from __future__ import annotations

import torch


_FP8 = tuple(
    dt
    for dt in (
        getattr(torch, "float8_e4m3fn", None),
        getattr(torch, "float8_e5m2", None),
        getattr(torch, "float8_e8m0fnu", None),
    )
    if dt is not None
)


def _fp8_e4m3fn_to_f32(bits: torch.Tensor) -> torch.Tensor:
    """OCP E4M3FN → fp32 using only integer/float ops (runs on NPU)."""
    b = bits.to(torch.int32)
    sign = (b >> 7) & 1
    exp = (b >> 3) & 0xF
    mant = b & 0x7
    fmant = mant.to(torch.float32) / 8.0
    sub = fmant * (2.0**-6)
    two = torch.tensor(2.0, device=b.device, dtype=torch.float32)
    norm = (1.0 + fmant) * torch.pow(two, (exp - 7).to(torch.float32))
    val = torch.where(exp == 0, sub, norm)
    val = torch.where((exp == 0) & (mant == 0), torch.zeros_like(val), val)
    val = torch.where((exp == 15) & (mant == 7), torch.full_like(val, float("nan")), val)
    return torch.where(sign.bool(), -val, val)


def safe_cast_fp8(t: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
    """Cast float8 without aclnnCast. Bit-decode on the tensor's device when possible."""
    if t.dtype not in _FP8:
        return t.to(dtype=dtype)
    fp8_e4 = getattr(torch, "float8_e4m3fn", None)
    try:
        bits = t.view(torch.uint8)
        if t.dtype == fp8_e4:
            out = _fp8_e4m3fn_to_f32(bits)
            return out.to(dtype=dtype)
    except Exception:
        pass
    if t.device.type == "npu":
        return t.cpu().to(dtype=dtype).to(t.device)
    return t.to(dtype=dtype)
