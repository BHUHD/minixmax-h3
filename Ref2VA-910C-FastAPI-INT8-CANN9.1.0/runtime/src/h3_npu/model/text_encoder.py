"""Minimal TE helper: INT8 embed + one NVFP4 layer path (full Qwen3-VL TBD).

Full 32B TE encoding can be wired later (layerwise NVFP4). Smoke validates the
910C-critical kernels used by every TE linear.
"""
from __future__ import annotations

from pathlib import Path

import torch

from h3_npu.load.safetensors_quant import load_comfy_tensors
from h3_npu.modules.linear import QuantEmbedding, QuantLinear


def load_embed_tokens(path: str | Path, device: torch.device) -> QuantEmbedding:
    tensors, metas = load_comfy_tensors(path)
    prefix = "model.embed_tokens"
    if prefix not in metas:
        raise KeyError("model.embed_tokens missing comfy_quant")
    emb = QuantEmbedding(1, 1)
    emb.set_quant(tensors[f"{prefix}.weight"], tensors[f"{prefix}.weight_scale"], device=device)
    return emb


def load_te_layer(path: str | Path, prefix: str, device: torch.device) -> QuantLinear:
    from h3_npu.load.safetensors_quant import load_nvfp4_linear_smoke

    return load_nvfp4_linear_smoke(path, layer_prefix=prefix, device=device)
