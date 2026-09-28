"""NVFP4 weight dequant for Qwen3-VL TE on 910C.

Native NVFP4 matmul is unavailable; we:
  1) unpack uint8 → e2m1 nibble values (CPU-safe path; NPU embedding is flaky)
  2) cast float8 block scales via CPU (aclnnCast gap)
  3) dequant to BF16
  4) F.linear on NPU (CANN GEMM)

Optional: cache dequantized BF16 weights in QuantLinear after first call.
TorchAir / Ascend C: `nvfp4_gemm_stub` marks where a fused kernel would plug in.
"""
from __future__ import annotations

from typing import Optional

import torch
import torch.nn.functional as F

from .float8_cast import safe_cast_fp8

# FP4 e2m1 lookup (matches comfy_kitchen E2M1)
_E2M1 = [
    0.0,
    0.5,
    1.0,
    1.5,
    2.0,
    3.0,
    4.0,
    6.0,
    -0.0,
    -0.5,
    -1.0,
    -1.5,
    -2.0,
    -3.0,
    -4.0,
    -6.0,
]


def _ceil_div(a: int, b: int) -> int:
    return (a + b - 1) // b


def from_blocked(blocked_matrix: torch.Tensor, num_rows: int, num_cols: int) -> torch.Tensor:
    """反 swizzle cuBLAS tiled block-scale 布局，对齐 comfy_kitchen ``from_blocked``。

    NVFP4 的 ``weight_scale`` 以 (RoundUp(rows,128), RoundUp(cols,4)) 的
    SWIZZLE_32_4_4 顺序存放；反量化前必须还原成逻辑 (num_rows, num_cols)。
    """
    n_row_blocks = _ceil_div(num_rows, 128)
    n_col_blocks = _ceil_div(num_cols, 4)
    padded_rows = n_row_blocks * 128
    padded_cols = n_col_blocks * 4
    step1 = blocked_matrix.reshape(-1, 32, 16)
    step2 = step1.reshape(-1, 32, 4, 4).transpose(1, 2)
    step3 = step2.reshape(n_row_blocks, n_col_blocks, 4, 32, 4)
    step4 = step3.reshape(n_row_blocks, n_col_blocks, 128, 4)
    step5 = step4.permute(0, 2, 1, 3)
    unblocked = step5.reshape(padded_rows, padded_cols)
    return unblocked[:num_rows, :num_cols]


def _prepare_block_scales(
    block_scales: torch.Tensor,
    num_rows: int,
    num_blocks_per_row: int,
    device: torch.device | str,
) -> torch.Tensor:
    """float8 block scales → float32，并做 from_blocked 反 swizzle。"""
    bs = safe_cast_fp8(block_scales.detach().to(device), torch.float32)
    # 已是逻辑布局且未 padding 时，from_blocked 仍会按 tiled 重排；
    # 存盘格式与 Comfy 一致，必须始终 unswizzle。
    if bs.dim() == 1:
        bs = bs.view(-1)
        # 扁平 swizzle：先还原到 padded 2D
        n_row_blocks = _ceil_div(num_rows, 128)
        n_col_blocks = _ceil_div(num_blocks_per_row, 4)
        bs = bs.reshape(n_row_blocks * 128, n_col_blocks * 4)
    elif bs.dim() != 2:
        bs = bs.reshape(bs.shape[0], -1)
    return from_blocked(bs, num_rows, num_blocks_per_row)


def dequantize_nvfp4_host(
    qx: torch.Tensor,
    per_tensor_scale: torch.Tensor,
    block_scales: torch.Tensor,
    *,
    out_dtype: torch.dtype = torch.bfloat16,
    hi_first: bool = True,
    block_size: int = 16,
) -> torch.Tensor:
    """CPU LUT dequant（含 block-scale 反 swizzle），对齐 comfy_kitchen eager 路径。"""
    qx_cpu = qx.detach().to("cpu", dtype=torch.uint8).contiguous()
    lo = (qx_cpu & 0x0F).to(torch.long)
    hi = (qx_cpu >> 4).to(torch.long)
    idx = torch.stack([hi, lo] if hi_first else [lo, hi], dim=-1).reshape(*qx_cpu.shape[:-1], -1)
    lut = torch.tensor(_E2M1, dtype=torch.float32)
    out = lut[idx]
    n, k = out.shape
    if k % block_size != 0:
        raise ValueError(f"K={k} not divisible by block_size={block_size}")
    n_blocks = k // block_size
    out = out.reshape(n, n_blocks, block_size)
    bs = _prepare_block_scales(block_scales, n, n_blocks, "cpu")
    ts = safe_cast_fp8(per_tensor_scale.detach().to("cpu"), torch.float32).reshape(())
    out = out * (ts * bs).unsqueeze(-1)
    return out.reshape(n, k).to(dtype=out_dtype)


def _unpack_e2m1(qx: torch.Tensor, hi_first: bool) -> torch.Tensor:
    """uint8 packed nibbles → e2m1 values. one_hot@LUT avoids NPU embedding gaps."""
    q = qx.to(dtype=torch.int64)
    lo = q & 0x0F
    hi = (q >> 4) & 0x0F
    idx = torch.stack([hi, lo] if hi_first else [lo, hi], dim=-1).reshape(*qx.shape[:-1], -1)
    lut = torch.tensor(_E2M1, dtype=torch.float32, device=qx.device)
    oh = torch.nn.functional.one_hot(idx, num_classes=16).to(dtype=torch.float32)
    return (oh @ lut).reshape(*idx.shape)


def dequantize_nvfp4(
    qx: torch.Tensor,
    per_tensor_scale: torch.Tensor,
    block_scales: torch.Tensor,
    *,
    out_dtype: torch.dtype = torch.bfloat16,
    hi_first: bool = True,
    block_size: int = 16,
) -> torch.Tensor:
    """qx uint8 packed [N, K/2]; block_scales float8（swizzled）；返回 [N, K]。"""
    device = qx.device
    try:
        out = _unpack_e2m1(qx.detach().contiguous(), hi_first)
        n, k = out.shape
        if k % block_size != 0:
            raise ValueError(f"K={k} not divisible by block_size={block_size}")
        n_blocks = k // block_size
        out = out.reshape(n, n_blocks, block_size)
        # 反 swizzle 在 CPU 做更稳；再搬回 device
        bs = _prepare_block_scales(block_scales, n, n_blocks, "cpu").to(device=device)
        ts = safe_cast_fp8(per_tensor_scale.detach().to("cpu"), torch.float32).reshape(()).to(device)
        out = out * (ts * bs).unsqueeze(-1)
        return out.reshape(n, k).to(dtype=out_dtype)
    except Exception as exc:  # noqa: BLE001
        print(f"[h3_npu] NVFP4 device unpack fallback to CPU: {exc}", flush=True)
    return dequantize_nvfp4_host(
        qx, per_tensor_scale, block_scales, out_dtype=out_dtype, hi_first=hi_first, block_size=block_size
    ).to(device=device)


def nvfp4_linear(
    x: torch.Tensor,
    weight_u8: torch.Tensor,
    weight_scale: torch.Tensor,
    weight_scale_2: torch.Tensor,
    bias: Optional[torch.Tensor] = None,
    *,
    out_dtype: torch.dtype = torch.bfloat16,
    weight_cache: Optional[list] = None,
) -> torch.Tensor:
    """x [...,K] @ W[N,K]^T with NVFP4-packed W."""
    if weight_cache is not None and weight_cache and weight_cache[0] is not None:
        w = weight_cache[0]
    else:
        w = dequantize_nvfp4(weight_u8, weight_scale_2, weight_scale, out_dtype=out_dtype)
        if weight_cache is not None:
            weight_cache[0] = w
    w = w.to(device=x.device, dtype=out_dtype)
    b = bias.to(device=x.device, dtype=out_dtype) if bias is not None else None
    return F.linear(x.to(out_dtype), w, b)


def nvfp4_gemm_stub(*_args, **_kwargs):
    """Future Ascend C / TorchAir fused NVFP4 GEMM entry (not available on 910C yet)."""
    raise NotImplementedError(
        "Fused NVFP4 GEMM needs Ascend C custom op; use nvfp4_linear dequant path on 910C"
    )
