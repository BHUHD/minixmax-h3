"""Ulysses sequence-parallel all-to-all for DiT attention (HCCL).

Each rank holds S/world tokens and all heads. Before FA: gather sequence, split
heads. After FA: restore S-local / all-heads. 56 heads and padded S must divide
world_size (8).
"""
from __future__ import annotations

import torch

from h3_npu.runtime.dist import is_distributed, state


def all_to_all_heads(x: torch.Tensor, *, gather_seq: bool) -> torch.Tensor:
    """
    gather_seq=True:  [S_local, H, D] -> [S, H/world, D]
    gather_seq=False: [S, H/world, D] -> [S_local, H, D]
    """
    if not is_distributed():
        return x
    import torch.distributed as dist

    ws = state().world_size
    if gather_seq:
        s_local, heads, dim = x.shape
        if heads % ws != 0:
            raise RuntimeError(f"heads={heads} not divisible by world_size={ws}")
        h_local = heads // ws
        send = x.view(s_local, ws, h_local, dim).permute(1, 0, 2, 3).contiguous()
        recv = torch.empty_like(send)
        dist.all_to_all_single(recv, send)
        return recv.reshape(ws * s_local, h_local, dim)

    seq, h_local, dim = x.shape
    if seq % ws != 0:
        raise RuntimeError(f"seq={seq} not divisible by world_size={ws}")
    s_local = seq // ws
    send = x.view(ws, s_local, h_local, dim).contiguous()
    recv = torch.empty_like(send)
    dist.all_to_all_single(recv, send)
    return recv.permute(1, 0, 2, 3).reshape(s_local, ws * h_local, dim)


def split_range(seq_len: int) -> tuple[int, int]:
    ws = state().world_size
    rank = state().rank
    if seq_len % ws != 0:
        raise RuntimeError(f"seq_len={seq_len} not divisible by world_size={ws} (pad tokens)")
    chunk = seq_len // ws
    return rank * chunk, (rank + 1) * chunk


def clip_mod_segments(
    segments: list[tuple[int, int, int]], s0: int, s1: int
) -> list[tuple[int, int, int]]:
    out: list[tuple[int, int, int]] = []
    for a, b, row in segments:
        aa, bb = max(a, s0), min(b, s1)
        if aa < bb:
            out.append((aa - s0, bb - s0, row))
    return out
