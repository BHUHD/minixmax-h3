"""Fused micro-kernels for DiT block hot paths (fewer launches / TorchAir targets)."""
from __future__ import annotations

import torch
import torch.nn.functional as F

from h3_npu.ops.rope import rms_rope_split_half


def _swiglu(x: torch.Tensor) -> torch.Tensor:
    if x.device.type == "npu":
        try:
            import torch_npu

            return torch_npu.npu_swiglu(x)
        except Exception:
            pass
    gate, up = x.chunk(2, dim=-1)
    return F.silu(gate) * up


def fused_qkv_rope(
    x: torch.Tensor,
    qkv_proj,
    q_norm,
    k_norm,
    rope_freqs: torch.Tensor | None,
    *,
    s0: int,
    s1: int,
    heads: int,
    head_dim: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """QKV linear + RMS + RoPE in one Python region (enables compile/fusion)."""
    s_local = x.shape[0]
    q, k, v = qkv_proj(x).split(heads * head_dim, dim=-1)
    q = q.view(s_local, heads, head_dim)
    k = k.view(s_local, heads, head_dim)
    v = v.view(s_local, heads, head_dim)
    if rope_freqs is not None:
        rot = rope_freqs.shape[-3] * 2
        local_fr = rope_freqs[:, s0:s1]
        qw = q_norm.weight.to(device=x.device)
        kw = k_norm.weight.to(device=x.device)
        q, k = rms_rope_split_half(
            q.unsqueeze(0),
            k.unsqueeze(0),
            local_fr,
            qw,
            kw,
            epsilon=q_norm.eps,
            rot_dim=rot,
        )
        q, k = q[0], k[0]
    else:
        q = q_norm(q)
        k = k_norm(k)
    return q, k, v


def fused_mlp_swiglu(x: torch.Tensor, mlp) -> torch.Tensor:
    """fc1 → SwiGLU → fc2 without extra module dispatch."""
    return mlp.fc2(_swiglu(mlp.fc1(x)))
