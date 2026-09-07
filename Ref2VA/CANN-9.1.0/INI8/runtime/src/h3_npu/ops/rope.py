"""RoPE + RMS helpers matching Comfy kitchen split-half (not interleaved pairs)."""
from __future__ import annotations

import torch


def rope_rotation_table(angles: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
    """[S, rot_dim] -> [1, S, 1, rot_dim/2, 2, 2]."""
    half = angles.shape[-1] // 2
    ang = angles[:, :half]
    c, s = torch.cos(ang), torch.sin(ang)
    table = torch.stack([c, -s, s, c], dim=-1).reshape(1, angles.shape[0], 1, half, 2, 2)
    return table.to(dtype)


def rms_norm(x: torch.Tensor, weight: torch.Tensor, eps: float) -> torch.Tensor:
    if x.device.type == "npu":
        try:
            import torch_npu

            return torch_npu.npu_rms_norm(x, weight.to(device=x.device, dtype=x.dtype), epsilon=eps)[0]
        except Exception:
            pass
    orig = x.dtype
    x32 = x.float()
    x32 = x32 * torch.rsqrt(x32.pow(2).mean(-1, keepdim=True) + eps)
    return (x32 * weight.float()).to(orig)


def apply_rope_split_half1(x: torch.Tensor, freqs_cis: torch.Tensor) -> torch.Tensor:
    """Split-half RoPE: pair first half of D with second half (Comfy kitchen).

    Interleaved ``reshape(..., half, 2)`` is the wrong convention and makes
    DiT tokens fail to mix, which decoded as an 8×8 mosaic.
    """
    t_ = x.reshape(*x.shape[:-1], 2, -1).movedim(-2, -1).unsqueeze(-2).to(dtype=freqs_cis.dtype)
    t_out = freqs_cis[..., 0] * t_[..., 0] + freqs_cis[..., 1] * t_[..., 1]
    return t_out.movedim(-1, -2).reshape(*x.shape).type_as(x)


def rms_rope_apply(
    x: torch.Tensor,
    freqs_cis: torch.Tensor,
    weight: torch.Tensor,
    epsilon: float,
    rot_dim: int,
) -> torch.Tensor:
    """RMS + split-half RoPE. x: [1, S, H, D]; freqs: [1, S, 1, rot_dim/2, 2, 2]."""
    x = rms_norm(x, weight, epsilon)
    if rot_dim and rot_dim != x.shape[-1]:
        return torch.cat(
            [apply_rope_split_half1(x[..., :rot_dim], freqs_cis), x[..., rot_dim:]],
            dim=-1,
        )
    return apply_rope_split_half1(x, freqs_cis)


def rms_rope_split_half(
    q: torch.Tensor,
    k: torch.Tensor,
    rope_freqs: torch.Tensor,
    q_weight: torch.Tensor,
    k_weight: torch.Tensor,
    *,
    epsilon: float,
    rot_dim: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """q/k: [1, S, H, D]; rope_freqs: [1, S, 1, rot_dim/2, 2, 2]."""
    q = rms_norm(q, q_weight, epsilon)
    k = rms_norm(k, k_weight, epsilon)

    def _rotate(x: torch.Tensor) -> torch.Tensor:
        if rot_dim and rot_dim != x.shape[-1]:
            return torch.cat(
                [apply_rope_split_half1(x[..., :rot_dim], rope_freqs), x[..., rot_dim:]],
                dim=-1,
            )
        return apply_rope_split_half1(x, rope_freqs)

    return _rotate(q), _rotate(k)


def apply_rope_split_half(
    q: torch.Tensor,
    k: torch.Tensor,
    rope_freqs: torch.Tensor,
    rot_dim: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Rotate q/k only (RMS already applied). q/k: [B, S, H, D]."""

    def _rotate(x: torch.Tensor) -> torch.Tensor:
        if rot_dim and rot_dim != x.shape[-1]:
            return torch.cat(
                [apply_rope_split_half1(x[..., :rot_dim], rope_freqs), x[..., rot_dim:]],
                dim=-1,
            )
        return apply_rope_split_half1(x, rope_freqs)

    return _rotate(q), _rotate(k)
