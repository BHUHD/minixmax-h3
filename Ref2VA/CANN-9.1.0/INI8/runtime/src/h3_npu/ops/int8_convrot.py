"""INT8 ConvRot Linear on 910C Cube (ACLNN).

MindIE W8A8 path: pack weight [K, N] once (FRACTAL_NZ=29 when the runtime
accepts it), then ``npu_quant_matmul`` / ``npu_weight_quant_batchmatmul``.
Do **not** ``.t().contiguous()`` every forward — that copies ~20 GiB/step.
"""
from __future__ import annotations

import os
from typing import Optional

import torch
import torch.nn.functional as F

from .hadamard import rotate_activation, rotate_weight_back

_FORCE_DEQUANT = os.environ.get("H3_NPU_INT8_FORCE_DEQUANT", "0") == "1"
_PATH: Optional[str] = None
_NZ_OK: Optional[bool] = None
_W8A16_OK: Optional[bool] = None
_W8A8_OK: Optional[bool] = None


def pack_int8_weight_cube(weight_nk: torch.Tensor) -> tuple[torch.Tensor, str]:
    """[N, K] INT8 → Cube-ready [K, N] (NZ if possible). Call once after H2D."""
    global _NZ_OK
    kn = weight_nk.t().contiguous()
    if os.environ.get("H3_NPU_INT8_NZ", "0") != "1" or kn.device.type != "npu":
        return kn, "kn"
    if _NZ_OK is False:
        return kn, "kn"
    try:
        import torch_npu

        nz = torch_npu.npu_format_cast(kn, 29)
        _NZ_OK = True
        return nz, "nz29"
    except Exception as exc:  # noqa: BLE001
        if _NZ_OK is None:
            print(f"[h3_npu] npu_format_cast NZ skipped ({exc}); using ND [K,N]", flush=True)
        _NZ_OK = False
        return kn, "kn"


def _try_weight_quant_bmm(
    x: torch.Tensor,
    w_cube: torch.Tensor,
    scale: torch.Tensor,
    bias: Optional[torch.Tensor],
    out_dtype: torch.dtype,
) -> Optional[torch.Tensor]:
    """W8A16 Cube: BF16 act × INT8 weight. Skips per-token act quant."""
    global _W8A16_OK
    try:
        import torch_npu

        sc = scale.reshape(-1)
        # aclnnWeightQuantBatchMatmulV2 rejects DT_FLOAT scales (async).
        if sc.dtype not in (torch.float16, torch.bfloat16, torch.int64, torch.uint64):
            sc = sc.to(torch.bfloat16 if x.dtype == torch.bfloat16 else torch.float16)
        kw = {"bias": bias} if bias is not None else {}
        y = torch_npu.npu_weight_quant_batchmatmul(
            x.contiguous(),
            w_cube,
            sc.contiguous(),
            **kw,
        )
        if _W8A16_OK is None:
            torch.npu.synchronize()
        return y
    except Exception:
        return None


def _try_npu_quant_matmul(
    x_i8: torch.Tensor,
    w_cube: torch.Tensor,
    scale: torch.Tensor,
    pertoken_scale: torch.Tensor,
    bias: Optional[torch.Tensor],
    out_dtype: torch.dtype,
) -> Optional[torch.Tensor]:
    try:
        import torch_npu

        y = torch_npu.npu_quant_matmul(
            x_i8.contiguous(),
            w_cube,
            scale.reshape(-1).to(torch.float32).contiguous(),
            pertoken_scale=pertoken_scale.reshape(-1).to(torch.float32).contiguous(),
            bias=bias,
            output_dtype=out_dtype,
        )
        return y
    except Exception:
        return None


def _dynamic_quant_rows(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    try:
        import torch_npu

        y, scale = torch_npu.npu_dynamic_quant(x.contiguous())
        return y, scale.reshape(-1).to(torch.float32)
    except Exception:
        pass
    x2 = x.float()
    amax = x2.abs().amax(dim=-1).clamp(min=1e-12)
    scale = amax / 127.0
    y = (x2 / scale.unsqueeze(-1)).round().clamp(-128, 127).to(torch.int8)
    return y, scale


def int8_convrot_linear(
    x: torch.Tensor,
    weight_i8: torch.Tensor,
    weight_scale: torch.Tensor,
    bias: Optional[torch.Tensor] = None,
    *,
    convrot: bool = True,
    group_size: int = 256,
    out_dtype: torch.dtype = torch.bfloat16,
    weight_layout: str = "nk",
) -> torch.Tensor:
    """
    x: [..., K] bf16/fp16 on NPU
    weight_i8: [N, K] if layout=nk, Cube [K, N]/NZ if layout=kn|nz29
    weight_scale: [N] float32
    """
    global _PATH, _W8A16_OK, _W8A8_OK
    if x.device.type != "npu":
        raise RuntimeError("int8_convrot_linear expects NPU tensors")

    orig = x.shape
    x2 = x.reshape(-1, orig[-1]).to(dtype=out_dtype)
    w_scale = weight_scale.reshape(-1).to(device=x.device, dtype=torch.float32)
    b = bias.to(device=x.device, dtype=out_dtype) if bias is not None else None

    if convrot:
        x2 = rotate_activation(x2, group_size)

    packed = weight_layout in ("kn", "nz29")
    w_cube = weight_i8 if packed else None

    if not _FORCE_DEQUANT and packed:
        if _W8A16_OK is not False:
            y = _try_weight_quant_bmm(x2, w_cube, w_scale, b, out_dtype)
            if y is not None:
                if _PATH is None:
                    _PATH = f"w8a16_{weight_layout}"
                    print(f"[h3_npu] INT8 Cube path={_PATH}", flush=True)
                _W8A16_OK = True
                return y.reshape(*orig[:-1], w_scale.numel())
            _W8A16_OK = False

        if _W8A8_OK is not False:
            x_i8, x_scale = _dynamic_quant_rows(x2)
            y = _try_npu_quant_matmul(x_i8, w_cube, w_scale, x_scale, b, out_dtype)
            if y is not None:
                if _PATH is None:
                    _PATH = f"w8a8_{weight_layout}"
                    print(f"[h3_npu] INT8 Cube path={_PATH}", flush=True)
                _W8A8_OK = True
                return y.reshape(*orig[:-1], w_scale.numel())
            _W8A8_OK = False

    if not _FORCE_DEQUANT and not packed:
        x_i8, x_scale = _dynamic_quant_rows(x2)
        y = _try_npu_quant_matmul(
            x_i8,
            weight_i8.t().contiguous(),
            w_scale,
            x_scale,
            b,
            out_dtype,
        )
        if y is not None:
            if _PATH is None:
                _PATH = "w8a8_nk_transpose"
                print(f"[h3_npu] INT8 Cube path={_PATH} (pack weights to avoid this)", flush=True)
            return y.reshape(*orig[:-1], weight_i8.shape[0])

    if _PATH is None:
        _PATH = "bf16_dequant"
        print("[h3_npu] npu_quant_matmul unavailable, BF16 dequant fallback", flush=True)

    if packed:
        w_nk = weight_i8.t() if weight_i8.dim() == 2 else weight_i8
        try:
            w = w_nk.to(dtype=torch.float32) * w_scale.unsqueeze(-1)
        except Exception:
            w = weight_i8.to(dtype=torch.float32)
            if w.shape[0] != w_scale.numel():
                w = w.t()
            w = w * w_scale.unsqueeze(-1)
        if convrot:
            w = rotate_weight_back(w, group_size)
        w = w.to(dtype=out_dtype)
        x2 = x.reshape(-1, orig[-1]).to(dtype=out_dtype)
        y = F.linear(x2, w, b)
        return y.reshape(*orig[:-1], w.shape[0])

    w = weight_i8.to(device=x.device, dtype=torch.float32) * w_scale.unsqueeze(-1)
    if convrot:
        w = rotate_weight_back(w, group_size)
    w = w.to(dtype=out_dtype)
    x2 = x.reshape(-1, orig[-1]).to(dtype=out_dtype)
    y = F.linear(x2, w, b)
    return y.reshape(*orig[:-1], weight_i8.shape[0])
