"""Qwen3-VL-32B text encoder (first 50 layers) on NPU — no ComfyUI.

NVFP4 weights are dequantized on CPU once per layer, then BF16 GEMM / SDPA run
on the 910C. Layerwise so 32B BF16 never co-resides with the DiT.
"""
from __future__ import annotations

import math
import os
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from safetensors import safe_open

from h3_npu.modules.linear import RMSNorm, decode_comfy_quant
from h3_npu.ops.nvfp4 import dequantize_nvfp4_host


HIDDEN = 5120
INTERMEDIATE = 25600
N_HEADS = 64
N_KV = 8
HEAD_DIM = 128
N_LAYERS = 50
ROPE_THETA = 5_000_000.0
RMS_EPS = 1e-6

_TOKENIZER_DIR = Path(__file__).resolve().parents[1] / "tokenizer" / "qwen25"


def tokenize_prompt(text: str) -> list[int]:
    from transformers import Qwen2Tokenizer

    tok = Qwen2Tokenizer.from_pretrained(str(_TOKENIZER_DIR), local_files_only=True)
    ids = tok(text, add_special_tokens=False)["input_ids"]
    if not ids:
        ids = [int(tok.pad_token_id or 151643)]
    return [int(i) for i in ids]


def format_ref2va_prompt(prompt: str, n_images: int) -> str:
    """Comfy ref2va presentation: ``<Picture i>: `` labels then the user prompt.

    Vision blocks are spliced after each label in Comfy; this path keeps the
    labels as text so the DiT tag runs stay aligned when Qwen vision is skipped.
    """
    return format_ref2va_prompt_multimodal(
        prompt,
        n_pictures=n_images,
        video_has_audio=[],
        n_standalone_audios=0,
    )


def format_ref2va_prompt_multimodal(
    prompt: str,
    *,
    n_pictures: int = 0,
    video_has_audio: list[bool] | None = None,
    video_with_audio: int | None = None,
    n_standalone_audios: int = 0,
) -> str:
    """按 Comfy / vLLM-Omni 顺序注入 presentation 标签，再追加用户 prompt。

    顺序：images → videos（有音轨则 ``<Audio j>:`` 紧挨在 ``<Video k>:`` 前）
    → standalone audios。正文里可引用同序号标签做职责绑定；正文出现标签
    **不会**跳过前缀注入（避免角色描述写了 ``<Picture 1>`` 导致 presentation 丢失）。

    若 prompt 已以完整 presentation 开头（``<Picture 1>:`` / ``<Video 1>:`` /
    ``<Audio 1>:``），则原样返回。
    """
    stripped = (prompt or "").lstrip()
    if (
        stripped.startswith("<Picture 1>:")
        or stripped.startswith("<Video 1>:")
        or stripped.startswith("<Audio 1>:")
    ):
        return prompt

    # 兼容旧调用：只传 video_with_audio 计数时，视为前 N 个视频带音轨
    if video_has_audio is None:
        n_va = int(video_with_audio or 0)
        video_has_audio = [True] * n_va
    else:
        video_has_audio = [bool(x) for x in video_has_audio]

    parts: list[str] = []
    for i in range(1, int(n_pictures) + 1):
        parts.append(f"<Picture {i}>: ")
    audio_idx = 1
    for k, has_audio in enumerate(video_has_audio, start=1):
        if has_audio:
            parts.append(f"<Audio {audio_idx}>: ")
            audio_idx += 1
        parts.append(f"<Video {k}>: ")
    for _ in range(int(n_standalone_audios)):
        parts.append(f"<Audio {audio_idx}>: ")
        audio_idx += 1
    return "".join(parts) + (prompt or "")


def _precompute_rope(seq_len: int, device, dtype):
    theta_num = torch.arange(0, HEAD_DIM, 2, device=device).float()
    inv_freq = 1.0 / (ROPE_THETA ** (theta_num / HEAD_DIM))
    pos = torch.arange(seq_len, device=device).float()
    freqs = torch.outer(pos, inv_freq)
    emb = torch.cat((freqs, freqs), dim=-1)
    cos = emb.cos().to(dtype)[None, None, :, :]
    sin = emb.sin().to(dtype)[None, None, :, :]
    nsin = -sin[..., HEAD_DIM // 2 :]
    sin_half = sin[..., : HEAD_DIM // 2]
    return cos, sin_half, nsin


def _apply_rope(xq, xk, cos, sin_half, nsin):
    orig = xq.dtype
    q = xq.float()
    k = xk.float()
    cos_f = cos.float()
    qe = q * cos_f
    split = qe.shape[-1] // 2
    qe[..., :split] = qe[..., :split] + q[..., split:] * nsin.float()
    qe[..., split:] = qe[..., split:] + q[..., :split] * sin_half.float()
    ke = k * cos_f
    ke[..., :split] = ke[..., :split] + k[..., split:] * nsin.float()
    ke[..., split:] = ke[..., split:] + k[..., :split] * sin_half.float()
    return qe.to(orig), ke.to(orig)


class _Dense(nn.Module):
    def __init__(self, weight: torch.Tensor, pre_quant_scale=None, bias=None):
        super().__init__()
        self.weight = nn.Parameter(weight, requires_grad=False)
        self.register_buffer("pre_quant_scale", pre_quant_scale, persistent=False)
        self.bias = None if bias is None else nn.Parameter(bias, requires_grad=False)

    def forward(self, x):
        if self.pre_quant_scale is not None:
            x = x * self.pre_quant_scale.to(device=x.device, dtype=x.dtype)
        return F.linear(x, self.weight.to(dtype=x.dtype), None if self.bias is None else self.bias.to(dtype=x.dtype))


def _load_nvfp4_dense(f, prefix: str, device, dtype) -> _Dense:
    meta = decode_comfy_quant(f.get_tensor(prefix + ".comfy_quant"))
    if meta.get("format") != "nvfp4":
        raise ValueError(f"{prefix} format {meta.get('format')}")
    w = dequantize_nvfp4_host(
        f.get_tensor(prefix + ".weight"),
        f.get_tensor(prefix + ".weight_scale_2"),
        f.get_tensor(prefix + ".weight_scale"),
        out_dtype=dtype,
    ).to(device)
    pqs = None
    pk = prefix + ".pre_quant_scale"
    try:
        pqs = f.get_tensor(pk).to(device=device, dtype=dtype)
    except Exception:
        pqs = None
    bias = None
    try:
        bias = f.get_tensor(prefix + ".bias").to(device=device, dtype=dtype)
    except Exception:
        bias = None
    return _Dense(w, pre_quant_scale=pqs, bias=bias)


class Qwen3Block(nn.Module):
    def __init__(self, f, layer_i: int, device, dtype):
        super().__init__()
        p = f"model.layers.{layer_i}"
        self.input_layernorm = RMSNorm(HIDDEN, eps=RMS_EPS, dtype=dtype, device=device)
        self.input_layernorm.weight = nn.Parameter(
            f.get_tensor(f"{p}.input_layernorm.weight").to(device=device, dtype=dtype), requires_grad=False
        )
        self.post_attention_layernorm = RMSNorm(HIDDEN, eps=RMS_EPS, dtype=dtype, device=device)
        self.post_attention_layernorm.weight = nn.Parameter(
            f.get_tensor(f"{p}.post_attention_layernorm.weight").to(device=device, dtype=dtype),
            requires_grad=False,
        )
        self.q_proj = _load_nvfp4_dense(f, f"{p}.self_attn.q_proj", device, dtype)
        self.k_proj = _load_nvfp4_dense(f, f"{p}.self_attn.k_proj", device, dtype)
        self.v_proj = _load_nvfp4_dense(f, f"{p}.self_attn.v_proj", device, dtype)
        self.o_proj = _load_nvfp4_dense(f, f"{p}.self_attn.o_proj", device, dtype)
        self.gate_proj = _load_nvfp4_dense(f, f"{p}.mlp.gate_proj", device, dtype)
        self.up_proj = _load_nvfp4_dense(f, f"{p}.mlp.up_proj", device, dtype)
        self.down_proj = _load_nvfp4_dense(f, f"{p}.mlp.down_proj", device, dtype)
        self.q_norm = RMSNorm(HEAD_DIM, eps=RMS_EPS, dtype=dtype, device=device)
        self.q_norm.weight = nn.Parameter(
            f.get_tensor(f"{p}.self_attn.q_norm.weight").to(device=device, dtype=dtype), requires_grad=False
        )
        self.k_norm = RMSNorm(HEAD_DIM, eps=RMS_EPS, dtype=dtype, device=device)
        self.k_norm.weight = nn.Parameter(
            f.get_tensor(f"{p}.self_attn.k_norm.weight").to(device=device, dtype=dtype), requires_grad=False
        )

    def forward(self, x, rope):
        residual = x
        h = self.input_layernorm(x)
        b, s, _ = h.shape
        q = self.q_proj(h).view(b, s, N_HEADS, HEAD_DIM).transpose(1, 2)
        k = self.k_proj(h).view(b, s, N_KV, HEAD_DIM).transpose(1, 2)
        v = self.v_proj(h).view(b, s, N_KV, HEAD_DIM).transpose(1, 2)
        q = self.q_norm(q)
        k = self.k_norm(k)
        q, k = _apply_rope(q, k, *rope)
        nrep = N_HEADS // N_KV
        k = k.repeat_interleave(nrep, dim=1)
        v = v.repeat_interleave(nrep, dim=1)
        attn = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        h = self.o_proj(attn.transpose(1, 2).reshape(b, s, N_HEADS * HEAD_DIM))
        x = residual + h
        residual = x
        h = self.post_attention_layernorm(x)
        h = self.down_proj(F.silu(self.gate_proj(h)) * self.up_proj(h))
        return residual + h


def _load_embed(f, device, dtype) -> torch.Tensor:
    """INT8 embed_tokens → BF16 table on NPU."""
    w = f.get_tensor("model.embed_tokens.weight").to(torch.int8)
    scale = f.get_tensor("model.embed_tokens.weight_scale").to(torch.float32).reshape(-1, 1)
    return (w.float() * scale).to(dtype=dtype, device=device)


def _layer_range(rank: int, world: int, n_layers: int = N_LAYERS) -> tuple[int, int]:
    """Split TE layers. Rank0 can take none so it encodes the ref image in parallel."""
    skip0 = os.environ.get("H3_TE_RANK0_ENCODE", "1") == "1" and world > 1
    if skip0:
        if rank == 0:
            return 0, 0
        rank, world = rank - 1, world - 1
    base, rem = divmod(n_layers, max(world, 1))
    if rank < rem:
        start = rank * (base + 1)
        return start, start + base + 1
    start = rem * (base + 1) + (rank - rem) * base
    return start, start + base


@torch.inference_mode()
def encode_text_npu(
    te_path: str | Path,
    prompt: str,
    *,
    device: torch.device,
    dtype: torch.dtype = torch.bfloat16,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Returns context [1, S, 5120] and token tags [S] (1 = text) on ``device``.

    8-die: each rank dequants a contiguous layer slice in parallel, then the
    activation is pipelined with tiny broadcasts (prompt is tens of tokens).
    """
    from h3_npu.runtime.dist import broadcast_tensor, is_distributed, state as dist_state

    rank = dist_state().rank if is_distributed() else 0
    world = dist_state().world_size if is_distributed() else 1
    t0 = __import__("time").time()
    # Tokenizer is not safe under 8 concurrent from_pretrained; rank0 only.
    if rank == 0:
        ids = tokenize_prompt(prompt)
        print(f"[te] prompt tokens={len(ids)} {prompt[:80]!r}", flush=True)
        ntok_t = torch.tensor([len(ids)], device=device, dtype=torch.int32)
        idt = torch.tensor(ids, device=device, dtype=torch.int32)
    else:
        ntok_t = torch.zeros(1, device=device, dtype=torch.int32)
        idt = None
    if is_distributed():
        broadcast_tensor(ntok_t, src=0)
        n = int(ntok_t.item())
        if rank != 0:
            idt = torch.empty(n, device=device, dtype=torch.int32)
        broadcast_tensor(idt, src=0)
    ids = [int(x) for x in idt.tolist()]
    rope = None
    with safe_open(str(te_path), framework="pt") as f:
        if rank == 0:
            table = _load_embed(f, device, dtype)
            input_ids = torch.tensor(ids, device=device, dtype=torch.long).unsqueeze(0)
            x = F.embedding(input_ids, table)
            slen = torch.tensor([x.shape[1]], device=device, dtype=torch.int32)
            del table, input_ids
        else:
            slen = torch.zeros(1, device=device, dtype=torch.int32)
        if is_distributed():
            broadcast_tensor(slen, src=0)
        seq = int(slen.item())
        if rank != 0:
            x = torch.empty(1, seq, HIDDEN, device=device, dtype=dtype)
        if is_distributed():
            broadcast_tensor(x, src=0)
        rope = _precompute_rope(seq, device, dtype)
        start, end = _layer_range(rank, world)
        print(f"[te] rank={rank} layers {start}..{end - 1} ({end - start} / {N_LAYERS})", flush=True)
        if world == 1:
            for i in range(N_LAYERS):
                block = Qwen3Block(f, i, device, dtype)
                x = block(x, rope)
                del block
                if (i + 1) % 10 == 0 or i == 0:
                    print(f"[te] layer {i + 1}/{N_LAYERS}", flush=True)
                if torch.npu.is_available():
                    torch.npu.empty_cache()
        else:
            blocks = [Qwen3Block(f, i, device, dtype) for i in range(start, end)]
            if torch.npu.is_available():
                torch.npu.empty_cache()
            for owner in range(world):
                o0, o1 = _layer_range(owner, world)
                if rank == owner:
                    for bi, block in enumerate(blocks):
                        x = block(x, rope)
                        if (o0 + bi + 1) % 10 == 0 or o0 + bi == 0 or o0 + bi + 1 == N_LAYERS:
                            print(f"[te] layer {o0 + bi + 1}/{N_LAYERS}", flush=True)
                    del blocks
                    blocks = []
                    if torch.npu.is_available():
                        torch.npu.empty_cache()
                broadcast_tensor(x, src=owner)
    print(f"[te] encoded in {__import__('time').time() - t0:.1f}s |x|={float(x.float().norm()):.4f}", flush=True)
    tags = torch.ones(seq, device=device, dtype=torch.long)
    return x.contiguous(), tags
