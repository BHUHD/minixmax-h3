"""Fused gather + FA region for TorchAir / compile (reduce launch + HBM trips)."""
from __future__ import annotations

import os
from typing import Optional

import torch
import torch.nn as nn

from h3_npu.ops.gather_fa import gather_fa_bnsd


class GatherFABlock(nn.Module):
    """Wrap [H,S,D] local Q/K/V → gather KV → FA → [H,S_q,D] (BNSD)."""

    def __init__(self, heads: int, sequence_parallel: bool = True):
        super().__init__()
        self.heads = heads
        self.sequence_parallel = sequence_parallel

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        *,
        scale: Optional[float] = None,
    ) -> torch.Tensor:
        sp = self.sequence_parallel and is_distributed()
        qb = q.unsqueeze(0)
        kb = k.unsqueeze(0)
        vb = v.unsqueeze(0)
        if sp:
            out = gather_fa_bnsd(qb, k, v, scale=scale, gathered=False)
        else:
            out = gather_fa_bnsd(qb, kb, vb, scale=scale, gathered=True)
        return out[0]
