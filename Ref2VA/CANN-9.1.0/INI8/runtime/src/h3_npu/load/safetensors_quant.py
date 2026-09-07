"""Load Comfy-quant safetensors into h3_npu modules (DiT / TE)."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Optional

import os
import torch
import torch.nn as nn
from safetensors import safe_open

from h3_npu.modules.linear import Linear, QuantEmbedding, QuantLinear, decode_comfy_quant
from h3_npu.model.dit import MiniMaxH3DiT


def _resolve_module(root: nn.Module, dotted: str) -> tuple[nn.Module, str]:
    parts = dotted.split(".")
    parent = root
    for p in parts[:-1]:
        parent = getattr(parent, p)
    return parent, parts[-1]


def load_comfy_tensors(path: str | Path) -> tuple[dict[str, torch.Tensor], dict[str, dict[str, Any]]]:
    path = str(path)
    tensors: dict[str, torch.Tensor] = {}
    metas: dict[str, dict[str, Any]] = {}
    with safe_open(path, framework="pt") as f:
        for key in f.keys():
            if key.endswith(".comfy_quant"):
                prefix = key[: -len(".comfy_quant")]
                metas[prefix.rstrip(".")] = decode_comfy_quant(f.get_tensor(key))
            else:
                tensors[key] = f.get_tensor(key)
    return tensors, metas


def apply_quant_modules(
    model: nn.Module,
    tensors: dict[str, torch.Tensor],
    metas: dict[str, dict[str, Any]],
    *,
    device: torch.device,
    compute_dtype: torch.dtype = torch.bfloat16,
) -> dict[str, torch.Tensor]:
    """Replace Linear children covered by comfy_quant; return remaining dense state_dict."""
    consumed: set[str] = set()
    for prefix, meta in sorted(metas.items()):
        w_key = f"{prefix}.weight"
        if w_key not in tensors:
            continue
        parent, attr = _resolve_module(model, prefix)
        child = getattr(parent, attr)
        if not isinstance(child, (Linear, QuantLinear, nn.Linear)):
            # Embedding path
            if attr == "embed_tokens" or isinstance(child, (nn.Embedding, QuantEmbedding)):
                qe = QuantEmbedding(1, 1, compute_dtype=compute_dtype)
                qe.set_quant(tensors[w_key], tensors[f"{prefix}.weight_scale"], device=device)
                setattr(parent, attr, qe)
                consumed.update({w_key, f"{prefix}.weight_scale", f"{prefix}.comfy_quant"})
                continue
            continue

        bias = tensors.get(f"{prefix}.bias")
        scale = tensors.get(f"{prefix}.weight_scale")
        scale2 = tensors.get(f"{prefix}.weight_scale_2")
        pqs = tensors.get(f"{prefix}.pre_quant_scale")
        ql = QuantLinear.from_quant_tensors(
            tensors[w_key],
            meta=meta,
            weight_scale=scale,
            weight_scale_2=scale2,
            bias=bias,
            pre_quant_scale=pqs,
            device=device,
            compute_dtype=compute_dtype,
        )
        setattr(parent, attr, ql)
        consumed.update(
            {
                w_key,
                f"{prefix}.weight_scale",
                f"{prefix}.weight_scale_2",
                f"{prefix}.bias",
                f"{prefix}.pre_quant_scale",
                f"{prefix}.comfy_quant",
            }
        )

    remaining = {k: v for k, v in tensors.items() if k not in consumed and not k.endswith(".comfy_quant")}
    return remaining


def _move_dense_only(model: nn.Module, device: torch.device) -> None:
    """Move non-quant parameters/buffers; leave QuantLinear packs on device."""
    for module in model.modules():
        if isinstance(module, (QuantLinear, QuantEmbedding)):
            continue
        for name, param in list(module.named_parameters(recurse=False)):
            if param is None:
                continue
            module._parameters[name] = nn.Parameter(param.detach().to(device), requires_grad=False)
        for name, buf in list(module.named_buffers(recurse=False)):
            if buf is None:
                continue
            module._buffers[name] = buf.detach().to(device)


def load_pruned_dit(
    path: str | Path,
    *,
    device: torch.device,
    dtype: torch.dtype = torch.bfloat16,
    map_dense_to_device: bool = True,
) -> MiniMaxH3DiT:
    """Build pruned DiT and load INT8 ConvRot + dense weights.

    Structure is created on ``meta`` to avoid allocating BF16 shells for 200
    INT8 layers (keeps host/cgroup RAM under the 512G container budget).
    """
    tensors, metas = load_comfy_tensors(path)
    with torch.device("meta"):
        model = MiniMaxH3DiT(
            time_embed_dim=8,
            adaln_curve_grid=1025,
            dtype=dtype,
            device="meta",
        )
    # Keep INT8 packs on CPU when layerwise offload is enabled (move per block).
    import os

    pack_device = torch.device("cpu") if os.environ.get("H3_NPU_LAYERWISE_OFFLOAD", "0") == "1" else device
    remaining = apply_quant_modules(model, tensors, metas, device=pack_device, compute_dtype=dtype)
    # Materialize dense tensors from safetensors (CPU) onto real storages.
    missing, unexpected = model.load_state_dict(remaining, strict=False, assign=True)
    missing = [
        m
        for m in missing
        if not (
            m.endswith(".weight")
            and any(x in m for x in (".qkv_proj.", ".out_proj.", ".fc1.", ".fc2."))
        )
    ]
    if unexpected:
        print(f"[h3_npu] unexpected keys (first 10): {unexpected[:10]}", flush=True)
    if missing:
        print(f"[h3_npu] missing keys (first 10): {missing[:10]}", flush=True)
    layerwise = os.environ.get("H3_NPU_LAYERWISE_OFFLOAD", "0") == "1"
    if map_dense_to_device:
        _move_dense_only(model, device)
    if layerwise:
        # Activations stay on NPU; transformer blocks stream from host RAM (≤512G).
        model.blocks.to("cpu")
        model.final_layer.to("cpu")
        print("[h3_npu] layerwise offload: blocks/final_layer on CPU", flush=True)
    model.eval()
    return model


def load_pruned_dit_streaming(
    path: str | Path,
    *,
    device: torch.device,
    dtype: torch.dtype = torch.bfloat16,
) -> MiniMaxH3DiT:
    """Load on CPU (fast mmap), then move packs to NPU HBM. No layerwise offload."""
    path = str(path)
    layerwise = os.environ.get("H3_NPU_LAYERWISE_OFFLOAD", "0") == "1"
    pack_device = torch.device("cpu")
    print(f"[h3_npu] reading safetensors on CPU {path}", flush=True)
    tensors, metas = load_comfy_tensors(path)
    print(f"[h3_npu] CPU tensors={len(tensors)} quant={len(metas)}", flush=True)
    with torch.device("meta"):
        model = MiniMaxH3DiT(
            time_embed_dim=8,
            adaln_curve_grid=1025,
            dtype=dtype,
            device="meta",
        )
    remaining = apply_quant_modules(model, tensors, metas, device=pack_device, compute_dtype=dtype)
    del tensors
    import gc

    gc.collect()
    missing, unexpected = model.load_state_dict(remaining, strict=False, assign=True)
    del remaining
    missing = [
        m
        for m in missing
        if not (
            m.endswith(".weight")
            and any(x in m for x in (".qkv_proj.", ".out_proj.", ".fc1.", ".fc2."))
        )
    ]
    if unexpected:
        print(f"[h3_npu] unexpected keys (first 10): {unexpected[:10]}", flush=True)
    if missing:
        print(f"[h3_npu] missing keys (first 10): {missing[:10]}", flush=True)
    if layerwise:
        _move_dense_only(model, device)
        model.blocks.to("cpu")
        model.final_layer.to("cpu")
        print("[h3_npu] layerwise offload: blocks/final_layer on CPU", flush=True)
    else:
        print(f"[h3_npu] H2D DiT blocks → {device} (resident, no offload)", flush=True)
        if device.type == "npu":
            from h3_npu.runtime.h2d import dit_blocks_to_npu, warmup_npu

            warmup_npu(device)
            dit_blocks_to_npu(model, device)
        else:
            model.to(device)
    model.eval()
    return model


def load_nvfp4_linear_smoke(
    path: str | Path,
    layer_prefix: str = "model.layers.0.mlp.gate_proj",
    *,
    device: torch.device,
    dtype: torch.dtype = torch.bfloat16,
) -> QuantLinear:
    """Load a single TE NVFP4 linear for ACLNN/dequant path validation."""
    tensors, metas = load_comfy_tensors(path)
    if layer_prefix not in metas:
        raise KeyError(f"{layer_prefix} not in quant metas")
    meta = metas[layer_prefix]
    return QuantLinear.from_quant_tensors(
        tensors[f"{layer_prefix}.weight"],
        meta=meta,
        weight_scale=tensors.get(f"{layer_prefix}.weight_scale"),
        weight_scale_2=tensors.get(f"{layer_prefix}.weight_scale_2"),
        bias=tensors.get(f"{layer_prefix}.bias"),
        pre_quant_scale=tensors.get(f"{layer_prefix}.pre_quant_scale"),
        device=device,
        compute_dtype=dtype,
    )
