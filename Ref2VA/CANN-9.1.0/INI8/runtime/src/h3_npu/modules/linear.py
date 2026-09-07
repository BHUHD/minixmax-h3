"""NPU-aware Linear / Embedding / RMSNorm covering Comfy quant formats."""
from __future__ import annotations

import json
from typing import Any, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from h3_npu.ops.int8_convrot import int8_convrot_linear
from h3_npu.ops.nvfp4 import nvfp4_linear


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-5, dtype=None, device=None, **_):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim, dtype=dtype or torch.bfloat16, device=device))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.device.type == "npu":
            try:
                import torch_npu

                return torch_npu.npu_rms_norm(x, self.weight.to(device=x.device, dtype=x.dtype), epsilon=self.eps)[0]
            except Exception:
                pass
        orig = x.dtype
        x32 = x.float()
        x32 = x32 * torch.rsqrt(x32.pow(2).mean(-1, keepdim=True) + self.eps)
        return (x32 * self.weight.float()).to(orig)


class Linear(nn.Module):
    """Dense Linear matching Comfy `operations.Linear` ctor kwargs."""

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        dtype=None,
        device=None,
        **_,
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        dt = dtype or torch.bfloat16
        self.weight = nn.Parameter(
            torch.empty(out_features, in_features, dtype=dt, device=device), requires_grad=False
        )
        if bias:
            self.bias = nn.Parameter(torch.empty(out_features, dtype=dt, device=device), requires_grad=False)
        else:
            self.register_parameter("bias", None)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        w = self.weight.to(dtype=x.dtype, device=x.device)
        b = self.bias.to(dtype=x.dtype, device=x.device) if self.bias is not None else None
        return F.linear(x, w, b)


class QuantLinear(nn.Module):
    """INT8 ConvRot / NVFP4 / dense Linear — fills 910C quant gaps via ACLNN."""

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        dtype=None,
        device=None,
        **_,
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.quant_format: Optional[str] = None
        self.convrot = False
        self.group_size = 256
        self._compute_dtype = dtype or torch.bfloat16
        # Materialized on load; keep empty dense shell for structure.
        self.weight = nn.Parameter(
            torch.empty(out_features, in_features, dtype=self._compute_dtype, device=device),
            requires_grad=False,
        )
        if bias:
            self.bias = nn.Parameter(
                torch.empty(out_features, dtype=self._compute_dtype, device=device),
                requires_grad=False,
            )
        else:
            self.register_parameter("bias", None)
        self.register_buffer("weight_i8", None, persistent=False)
        self.register_buffer("weight_u8", None, persistent=False)
        self.register_buffer("weight_scale", None, persistent=False)
        self.register_buffer("weight_scale_2", None, persistent=False)
        self.register_buffer("pre_quant_scale", None, persistent=False)
        self.weight_layout = "nk"
        # Lazy BF16 cache for NVFP4 (list so forward can mutate without rebinding)
        self._nvfp4_cache: list = [None]

    @classmethod
    def from_quant_tensors(
        cls,
        weight: torch.Tensor,
        *,
        meta: dict[str, Any],
        weight_scale: Optional[torch.Tensor] = None,
        weight_scale_2: Optional[torch.Tensor] = None,
        bias: Optional[torch.Tensor] = None,
        pre_quant_scale: Optional[torch.Tensor] = None,
        device=None,
        compute_dtype=torch.bfloat16,
    ) -> "QuantLinear":
        fmt = meta.get("format")
        if fmt == "int8_tensorwise":
            out_f, in_f = weight.shape
        elif fmt == "nvfp4":
            out_f, packed_k = weight.shape
            in_f = packed_k * 2
        else:
            out_f, in_f = weight.shape
        m = cls(in_f, out_f, bias=bias is not None, dtype=compute_dtype, device=device)
        m.set_quant(
            weight,
            meta=meta,
            weight_scale=weight_scale,
            weight_scale_2=weight_scale_2,
            bias=bias,
            pre_quant_scale=pre_quant_scale,
            device=device,
        )
        return m

    def set_quant(
        self,
        weight: torch.Tensor,
        *,
        meta: dict[str, Any],
        weight_scale: Optional[torch.Tensor] = None,
        weight_scale_2: Optional[torch.Tensor] = None,
        bias: Optional[torch.Tensor] = None,
        pre_quant_scale: Optional[torch.Tensor] = None,
        device=None,
    ) -> None:
        self.quant_format = meta.get("format")
        self.convrot = bool(meta.get("convrot", False))
        self.group_size = int(meta.get("convrot_groupsize", 256))
        # Drop dense shell weight from autograd / state to save RAM.
        self.weight = nn.Parameter(torch.empty(0, device=device), requires_grad=False)
        if self.quant_format == "int8_tensorwise":
            self.weight_i8 = weight.to(device=device, dtype=torch.int8).contiguous()
            self.weight_scale = weight_scale.to(device=device, dtype=torch.float32).contiguous()
            self.out_features, self.in_features = self.weight_i8.shape
        elif self.quant_format == "nvfp4":
            self.weight_u8 = weight.to(device=device, dtype=torch.uint8).contiguous()
            self.weight_scale = weight_scale.to(device=device).contiguous()
            self.weight_scale_2 = weight_scale_2.to(device=device).contiguous()
            self.out_features = self.weight_u8.shape[0]
            self.in_features = self.weight_u8.shape[1] * 2
        else:
            raise ValueError(f"unsupported quant format: {self.quant_format}")
        if bias is not None:
            self.bias = nn.Parameter(bias.to(device=device), requires_grad=False)
        if pre_quant_scale is not None:
            self.pre_quant_scale = pre_quant_scale.to(device=device)

    def prepare_cube(self) -> None:
        """Pack INT8 [N,K] → Cube [K,N]/NZ once. Must run after the pack is on NPU."""
        if self.quant_format != "int8_tensorwise" or self.weight_i8 is None:
            return
        if self.weight_layout != "nk":
            return
        from h3_npu.ops.int8_convrot import pack_int8_weight_cube

        packed, layout = pack_int8_weight_cube(self.weight_i8)
        self.weight_i8 = packed
        self.weight_layout = layout

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.pre_quant_scale is not None:
            x = x * self.pre_quant_scale.to(device=x.device, dtype=x.dtype)
        out_dtype = x.dtype if x.dtype in (torch.float16, torch.bfloat16) else self._compute_dtype
        if self.quant_format == "int8_tensorwise":
            return int8_convrot_linear(
                x,
                self.weight_i8,
                self.weight_scale,
                self.bias,
                convrot=self.convrot,
                group_size=self.group_size,
                out_dtype=out_dtype,
                weight_layout=self.weight_layout,
            )
        if self.quant_format == "nvfp4":
            return nvfp4_linear(
                x,
                self.weight_u8,
                self.weight_scale,
                self.weight_scale_2,
                self.bias,
                out_dtype=out_dtype,
                weight_cache=self._nvfp4_cache,
            )
        w = self.weight.to(dtype=out_dtype, device=x.device)
        b = self.bias.to(dtype=out_dtype, device=x.device) if self.bias is not None else None
        return F.linear(x.to(out_dtype), w, b)


class QuantEmbedding(nn.Module):
    """INT8 tensorwise embedding (TE embed_tokens)."""

    def __init__(self, num_embeddings: int, embedding_dim: int, compute_dtype=torch.bfloat16):
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.compute_dtype = compute_dtype
        self.register_buffer("weight_i8", torch.empty(0, dtype=torch.int8), persistent=False)
        self.register_buffer("weight_scale", torch.empty(0, dtype=torch.float32), persistent=False)

    def set_quant(self, weight: torch.Tensor, weight_scale: torch.Tensor, device=None) -> None:
        # Dequant once into BF16 so gather is a normal NPU embedding, not INT8 indexing.
        w = weight.to(device=device, dtype=torch.int8)
        s = weight_scale.to(device=device, dtype=torch.float32).reshape(-1, 1)
        table = (w.float() * s).to(self.compute_dtype)
        self.weight_i8 = w
        self.weight_scale = s
        self.register_buffer("weight_bf16", table, persistent=False)
        self.num_embeddings, self.embedding_dim = table.shape

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        table = getattr(self, "weight_bf16", None)
        if table is not None and table.numel() > 0:
            return F.embedding(input_ids, table.to(device=input_ids.device))
        rows = self.weight_i8[input_ids]
        scales = self.weight_scale[input_ids]
        return (rows.float() * scales).to(self.compute_dtype)


def decode_comfy_quant(blob: torch.Tensor) -> dict[str, Any]:
    raw = bytes(blob.detach().cpu().numpy().tobytes()).decode("utf-8")
    return json.loads(raw)


class NpuOps:
    """Drop-in for Comfy `operations` namespace."""

    Linear = Linear
    RMSNorm = RMSNorm
