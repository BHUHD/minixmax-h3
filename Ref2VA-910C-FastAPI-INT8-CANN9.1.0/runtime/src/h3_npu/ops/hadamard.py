"""Regular Hadamard (ConvRot) as one Cube GEMM on grouped last-dim blocks."""
from __future__ import annotations

import math

import torch

_CACHE: dict[tuple, torch.Tensor] = {}


def build_hadamard(size: int, device: torch.device, dtype: torch.dtype = torch.float32) -> torch.Tensor:
    key = (size, str(device), str(dtype))
    if key in _CACHE:
        return _CACHE[key]
    if size < 4 or (size & (size - 1)) != 0 or math.log(size, 4) % 1 != 0:
        raise ValueError(f"Regular Hadamard size must be a power of 4, got {size}")
    cpu_key = (size, "cpu", str(dtype))
    if cpu_key not in _CACHE:
        h4 = torch.tensor(
            [[1, 1, 1, -1], [1, 1, -1, 1], [1, -1, 1, 1], [-1, 1, 1, 1]],
            dtype=dtype,
            device="cpu",
        )
        h = h4
        cur = 4
        while cur < size:
            h = torch.kron(h, h4)
            cur *= 4
        _CACHE[cpu_key] = h / (size**0.5)
    h = _CACHE[cpu_key].to(device=device, dtype=dtype)
    _CACHE[key] = h
    return h


def rotate_activation(x: torch.Tensor, group_size: int) -> torch.Tensor:
    """Online activation rotation: flatten groups into one Cube mm."""
    *lead, k = x.shape
    if k % group_size != 0:
        raise ValueError(f"K={k} not divisible by group_size={group_size}")
    h = build_hadamard(group_size, device=x.device, dtype=x.dtype)
    n = k // group_size
    # [M*n, g] @ [g, g]  — one GEMM instead of a tiny batched matmul.
    y = torch.mm(x.reshape(-1, group_size), h)
    return y.reshape(*lead, k)


def rotate_weight_back(weight: torch.Tensor, group_size: int) -> torch.Tensor:
    """Offline inverse: W = W_rot @ H."""
    out_f, in_f = weight.shape
    h = build_hadamard(group_size, device=weight.device, dtype=weight.dtype)
    n = in_f // group_size
    wr = torch.mm(weight.reshape(out_f * n, group_size), h)
    return wr.reshape(out_f, in_f)
