"""MiniMax-H3 pruned Ref2VA DiT (Comfy graph) — no ComfyUI imports.

Checkpoint: minimax_h3_ref2va_pruned_int8_convrot.safetensors
  - adaln_t_table [1025, 8], time_embed_dim=8
  - blocks.*.{attn,mlp}.* INT8 ConvRot via QuantLinear
"""
from __future__ import annotations

import math
import os
import time
from typing import Any, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from h3_npu.modules.linear import Linear, NpuOps, RMSNorm
from h3_npu.ops.attention import fusion_attention, fusion_attention_bnsd
from h3_npu.ops.fa_h3 import h3_attention_bnsd
from h3_npu.ops.fused_block import fused_mlp_swiglu, fused_qkv_rope
from h3_npu.ops.rope import rms_rope_apply, rope_rotation_table
from h3_npu.runtime.streams import maybe_sync, npu_stream, record_event, wait_event
from h3_npu.ops.ulysses import clip_mod_segments, split_range
from h3_npu.runtime.dist import all_gather_kv, all_gather_seq, is_distributed, state as dist_state

FRAME_PER_TOKEN = (1, 4, 4, 4, 4)
FRAME_RESCALE = 5.0 / 3.0
VISUAL_COND_TIMESTEP = 0.999
AUDIO_COND_TIMESTEP = 1.0


def time_shift_sigma(sigma, from_shift, to_shift):
    base = sigma / (from_shift + sigma * (1.0 - from_shift))
    return to_shift * base / (1.0 + (to_shift - 1.0) * base)


def time_shift_slope(sigma, from_shift, to_shift):
    base = sigma / (from_shift + sigma * (1.0 - from_shift))
    return (to_shift * (1.0 + (from_shift - 1.0) * base) ** 2) / (
        from_shift * (1.0 + (to_shift - 1.0) * base) ** 2
    )


def pad_to_patch_size(latent: torch.Tensor, patch_size=(1, 2, 2)) -> torch.Tensor:
    pt, ph, pw = patch_size
    t, h, w = latent.shape[2], latent.shape[3], latent.shape[4]
    pad_t = (pt - t % pt) % pt
    pad_h = (ph - h % ph) % ph
    pad_w = (pw - w % pw) % pw
    if pad_t or pad_h or pad_w:
        latent = F.pad(latent, (0, pad_w, 0, pad_h, 0, pad_t))
    return latent


def patchify_video(latent, patch_size=(1, 2, 2)):
    b, c, t_full, h_full, w_full = latent.shape
    pt, ph, pw = patch_size
    t, h, w = t_full // pt, h_full // ph, w_full // pw
    x = latent.reshape(b, c, t, pt, h, ph, w, pw)
    x = torch.einsum("nctrhpwq->nthwcrpq", x)
    return x.reshape(b * t * h * w, c * pt * ph * pw)


def unpatchify_video(rows, t, h, w, c=24, patch_size=(1, 2, 2)):
    pt, ph, pw = patch_size
    x = rows.reshape(-1, t, h, w, c, pt, ph, pw)
    x = torch.einsum("nthwcrpq->nctrhpwq", x)
    return x.reshape(-1, c, t * pt, h * ph, w * pw)


def pack_audio(latent):
    b, c, ch, t = latent.shape
    return latent[0].permute(1, 2, 0).reshape(ch * t, c)


def unpack_audio(rows, ch=2):
    t = rows.shape[0] // ch
    return rows.reshape(ch, t, rows.shape[-1]).permute(2, 0, 1).unsqueeze(0)


def _axis_from_sqrt_area(dim, patch, sqrt_area):
    ratio = dim / sqrt_area
    n = dim // patch
    return (torch.arange(n, dtype=torch.float64) * (ratio / n) + (1.0 - ratio) / 2.0) * 32.0


def _frame_grid(h, w):
    area = math.sqrt(h * w)
    hh, ww = torch.meshgrid(
        _axis_from_sqrt_area(h, 2, area), _axis_from_sqrt_area(w, 2, area), indexing="ij"
    )
    return torch.stack([hh.reshape(-1), ww.reshape(-1)], dim=-1), _axis_from_sqrt_area(w, 2, area)


def _video_t_spans(n):
    return [FRAME_RESCALE * FRAME_PER_TOKEN[k % 5] for k in range(n)]


def _video_t_grid(n, origin):
    spans = torch.tensor(_video_t_spans(n), dtype=torch.float64)
    return float(origin) + torch.cat([torch.zeros(1, dtype=torch.float64), spans[:-1].cumsum(0)])


def _audio_grid(cursor, t, w_low, w_high):
    g = torch.zeros(t * 2, 3, dtype=torch.float64)
    g[:, 0] = (cursor + torch.arange(t, dtype=torch.float64)).repeat(2)
    g[:t, 2] = w_low
    g[t:, 2] = w_high
    return g


def _video_grid(vt, frame, cursor):
    g = torch.empty(vt, frame.shape[0], 3, dtype=torch.float64)
    g[:, :, 0] = _video_t_grid(vt, cursor)[:, None]
    g[:, :, 1:] = frame[None]
    return g.reshape(-1, 3)


def _swiglu(x: torch.Tensor) -> torch.Tensor:
    if x.device.type == "npu":
        try:
            import torch_npu

            return torch_npu.npu_swiglu(x)
        except Exception:
            pass
    gate, up = x.chunk(2, dim=-1)
    return F.silu(gate) * up


_PROF_LEFT = int(os.environ.get("H3_PROFILE_LAYER", "0") or "0")


def _fa_sync_flags() -> tuple[bool, bool]:
    return (
        os.environ.get("H3_FA_PRE_GATHER_SYNC", "0") == "1",
        os.environ.get("H3_FA_PRE_FA_SYNC", "0") == "1",
    )


def _use_fused_qkv_rope() -> bool:
    return os.environ.get("H3_FUSED_QKV_ROPE", "0") == "1"


def _use_fused_mlp() -> bool:
    return os.environ.get("H3_FUSED_MLP", "0") == "1"


def _layer_overlap() -> bool:
    if os.environ.get("H3_NPU_TORCHAIR", "0") == "1":
        return False
    return os.environ.get("H3_LAYER_OVERLAP", "0") == "1"


class Attention(nn.Module):
    def __init__(self, hidden, heads, head_dim, eps, dtype=None, device=None, operations=None, sequence_parallel=True):
        super().__init__()
        operations = operations or NpuOps
        self.heads = heads
        self.head_dim = head_dim
        self.sequence_parallel = sequence_parallel
        inner = heads * head_dim
        self.qkv_proj = operations.Linear(hidden, inner * 3, bias=False, dtype=dtype, device=device)
        self.q_norm = operations.RMSNorm(head_dim, eps=eps, dtype=dtype, device=device)
        self.k_norm = operations.RMSNorm(head_dim, eps=eps, dtype=dtype, device=device)
        self.out_proj = operations.Linear(inner, hidden, bias=False, dtype=dtype, device=device)

    def _qkv_rope(self, x, rope_freqs, s0: int, s1: int):
        s_local = x.shape[0]
        if _use_fused_qkv_rope():
            q, k, v = fused_qkv_rope(
                x,
                self.qkv_proj,
                self.q_norm,
                self.k_norm,
                rope_freqs,
                s0=s0,
                s1=s1,
                heads=self.heads,
                head_dim=self.head_dim,
            )
        else:
            q, k, v = self.qkv_proj(x).split(self.heads * self.head_dim, dim=-1)
            q = q.view(s_local, self.heads, self.head_dim)
            k = k.view(s_local, self.heads, self.head_dim)
            v = v.view(s_local, self.heads, self.head_dim)
            if rope_freqs is not None:
                rot = rope_freqs.shape[-3] * 2
                local_fr = rope_freqs[:, s0:s1]
                qw = self.q_norm.weight.to(device=x.device)
                kw = self.k_norm.weight.to(device=x.device)
                q = rms_rope_apply(q.unsqueeze(0), local_fr, qw, self.q_norm.eps, rot)[0]
                k = rms_rope_apply(k.unsqueeze(0), local_fr, kw, self.k_norm.eps, rot)[0]
            else:
                q = self.q_norm(q)
                k = self.k_norm(k)
        return q, k, v

    def _gather_fa(self, q, k, v, *, comm_stream=None):
        pre_g, pre_f = _fa_sync_flags()
        sp = self.sequence_parallel and is_distributed()
        kv_len = k.shape[1]
        backend = (os.environ.get("H3_FA_BACKEND") or "infer").strip().lower()
        use_h3_gf = backend in ("h3_gather_fa", "h3_gf", "gather_fa", "ascend_c") or os.environ.get(
            "H3_FUSED_GATHER_FA", "0"
        ) == "1"
        if sp and use_h3_gf:
            from h3_npu.ops.gather_fa import sp_gather_fa

            maybe_sync(pre_g)
            scale = self.head_dim**-0.5
            out = sp_gather_fa(q, k, v, scale=scale)
            maybe_sync(pre_f)
            return out.unsqueeze(0), kv_len * dist_state().world_size
        if sp and os.environ.get("H3_ATTN_RING", "0") == "1":
            from h3_npu.ops.ring_attention import maybe_ring_attention_bnsd

            maybe_sync(pre_g)
            out = maybe_ring_attention_bnsd(q.unsqueeze(0), k.unsqueeze(0), v.unsqueeze(0))
            return out, kv_len
        if sp:
            use_async = _layer_overlap() or os.environ.get("H3_FA_ASYNC_GATHER", "0") == "1"
            if not use_async:
                maybe_sync(pre_g)
                k_g, v_g = all_gather_kv(k, v, seq_dim=1)
                k_g = k_g.contiguous()
                v_g = v_g.contiguous()
                maybe_sync(pre_f)
                out = h3_attention_bnsd(q.unsqueeze(0), k_g.unsqueeze(0), v_g.unsqueeze(0), gathered=True)
                kv_len = k_g.shape[1]
                return out, kv_len
            stream_comm = comm_stream or npu_stream("comm")
            stream_fa = npu_stream("fa")
            maybe_sync(pre_g)
            with torch.npu.stream(stream_comm):
                k_g, v_g = all_gather_kv(k, v, seq_dim=1)
                k_g = k_g.contiguous()
                v_g = v_g.contiguous()
                gather_evt = record_event(stream_comm)
            wait_event(gather_evt, stream_fa)
            with torch.npu.stream(stream_fa):
                maybe_sync(pre_f)
                out = h3_attention_bnsd(q.unsqueeze(0), k_g.unsqueeze(0), v_g.unsqueeze(0), gathered=True)
            kv_len = k_g.shape[1]
            return out, kv_len
        out = fusion_attention_bnsd(q.unsqueeze(0), k.unsqueeze(0), v.unsqueeze(0))
        return out, kv_len

    def forward(self, x, rope_freqs=None):
        global _PROF_LEFT
        prof = _PROF_LEFT > 0 and x.device.type == "npu" and x.shape[0] >= 1024
        if prof:
            torch.npu.synchronize()
            t0 = time.time()
        s_local = x.shape[0]
        sp = self.sequence_parallel and is_distributed()
        if sp:
            s0, s1 = split_range(rope_freqs.shape[1] if rope_freqs is not None else s_local * dist_state().world_size)
        else:
            s0, s1 = 0, s_local
        q, k, v = self._qkv_rope(x, rope_freqs, s0, s1)
        if prof:
            torch.npu.synchronize()
            t_qkv = time.time()
            t_rope = t_qkv
        q = q.permute(1, 0, 2).contiguous()
        k = k.permute(1, 0, 2).contiguous()
        v = v.permute(1, 0, 2).contiguous()
        if prof:
            t_g = t_rope
        comm_stream = npu_stream("comm") if _layer_overlap() else None
        out, kv_len = self._gather_fa(q, k, v, comm_stream=comm_stream)
        if prof:
            torch.npu.synchronize()
            t_fa = time.time()
            t_g = t_rope  # gather+fa bucketed when overlap splits streams
        out = out[0].permute(1, 0, 2).reshape(s_local, self.heads * self.head_dim)
        out = self.out_proj(out)
        if prof:
            torch.npu.synchronize()
            _PROF_LEFT -= 1
            print(
                f"[prof] attn qkv={t_qkv-t0:.3f}s rope={t_rope-t_qkv:.3f}s "
                f"gather={t_g-t_rope:.3f}s fa={t_fa-t_g:.3f}s o={time.time()-t_fa:.3f}s "
                f"q={s_local} kv={kv_len}",
                flush=True,
            )
        return out

class MLP(nn.Module):
    def __init__(self, hidden, ffn, dtype=None, device=None, operations=None):
        super().__init__()
        operations = operations or NpuOps
        self.fc1 = operations.Linear(hidden, ffn * 2, bias=False, dtype=dtype, device=device)
        self.fc2 = operations.Linear(ffn, hidden, bias=False, dtype=dtype, device=device)

    def forward(self, x):
        if _use_fused_mlp():
            return fused_mlp_swiglu(x, self)
        return self.fc2(_swiglu(self.fc1(x)))


class AdalnProj(nn.Module):
    def __init__(
        self,
        t_dim,
        hidden,
        expand,
        modalities,
        apply_silu=True,
        dtype=None,
        device=None,
        operations=None,
    ):
        super().__init__()
        operations = operations or NpuOps
        self.expand = expand
        self.modalities = modalities
        self.hidden = hidden
        self.apply_silu = apply_silu
        self.linear = operations.Linear(
            t_dim, expand * hidden * modalities, bias=True, dtype=dtype, device=device
        )

    def forward(self, t_emb):
        x = self.linear(F.silu(t_emb) if self.apply_silu else t_emb)
        x = x.view(x.shape[0] * self.modalities, self.expand * self.hidden)
        return x.chunk(self.expand, dim=-1)


def _mod_scale_shift(h, shift, scale, segments):
    for a, b, row in segments:
        h[a:b].mul_(1.0 + scale[row].to(h.dtype)).add_(shift[row].to(h.dtype))
    return h


def _mod_gate(x, gate, other, segments):
    for a, b, row in segments:
        x[a:b].addcmul_(other[a:b], gate[row].to(x.dtype))
    return x


class RefinerBlock(nn.Module):
    def __init__(self, hidden, heads, head_dim, ffn, eps, qk_eps, dtype=None, device=None, operations=None):
        super().__init__()
        operations = operations or NpuOps
        self.norm1 = operations.RMSNorm(hidden, eps=eps, dtype=dtype, device=device)
        self.norm2 = operations.RMSNorm(hidden, eps=eps, dtype=dtype, device=device)
        self.attn = Attention(
            hidden,
            heads,
            head_dim,
            qk_eps,
            dtype=dtype,
            device=device,
            operations=operations,
            sequence_parallel=False,
        )
        self.mlp = MLP(hidden, ffn, dtype=dtype, device=device, operations=operations)

    def forward(self, x):
        x = self.attn(self.norm1(x)).add_(x)
        return self.mlp(self.norm2(x)).add_(x)


class TokenRefiner(nn.Module):
    def __init__(
        self,
        num_layers,
        hidden,
        heads,
        head_dim,
        ffn,
        eps,
        qk_eps,
        final_eps,
        dtype=None,
        device=None,
        operations=None,
    ):
        super().__init__()
        self.blocks = nn.ModuleList(
            [
                RefinerBlock(
                    hidden, heads, head_dim, ffn, eps, qk_eps, dtype=dtype, device=device, operations=operations
                )
                for _ in range(num_layers)
            ]
        )
        operations = operations or NpuOps
        self.final_norm = operations.RMSNorm(hidden, eps=final_eps, dtype=dtype, device=device)

    def forward(self, x):
        for block in self.blocks:
            x = block(x)
        return self.final_norm(x)


class DiTBlock(nn.Module):
    def __init__(
        self,
        hidden,
        heads,
        head_dim,
        ffn,
        t_dim,
        eps,
        qk_eps,
        apply_silu=True,
        adaln_dtype=None,
        dtype=None,
        device=None,
        operations=None,
    ):
        super().__init__()
        operations = operations or NpuOps
        self.norm1 = operations.RMSNorm(hidden, eps=eps, dtype=dtype, device=device)
        self.norm2 = operations.RMSNorm(hidden, eps=eps, dtype=dtype, device=device)
        self.attn = Attention(hidden, heads, head_dim, qk_eps, dtype=dtype, device=device, operations=operations)
        self.mlp = MLP(hidden, ffn, dtype=dtype, device=device, operations=operations)
        self.adaln_proj = AdalnProj(
            t_dim,
            hidden,
            6,
            3,
            apply_silu=apply_silu,
            dtype=adaln_dtype if adaln_dtype is not None else dtype,
            device=device,
            operations=operations,
        )

    def forward(self, x, t_emb, mod_segments, rope_freqs):
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.adaln_proj(t_emb)
        h = _mod_scale_shift(self.norm1(x), shift_msa, scale_msa, mod_segments)
        x = _mod_gate(x, gate_msa, self.attn(h, rope_freqs=rope_freqs), mod_segments)
        h = _mod_scale_shift(self.norm2(x), shift_mlp, scale_mlp, mod_segments)
        return _mod_gate(x, gate_mlp, self.mlp(h), mod_segments)

    def forward_attn(self, x, t_emb, mod_segments, rope_freqs):
        shift_msa, scale_msa, gate_msa, _, _, _ = self.adaln_proj(t_emb)
        h = _mod_scale_shift(self.norm1(x), shift_msa, scale_msa, mod_segments)
        attn_out = self.attn(h, rope_freqs=rope_freqs)
        return _mod_gate(x, gate_msa, attn_out, mod_segments)

    def forward_mlp(self, x, t_emb, mod_segments):
        _, _, _, shift_mlp, scale_mlp, gate_mlp = self.adaln_proj(t_emb)
        h = _mod_scale_shift(self.norm2(x), shift_mlp, scale_mlp, mod_segments)
        return _mod_gate(x, gate_mlp, self.mlp(h), mod_segments)


def _run_dit_blocks(blocks, h, t_emb, mod_segments, rope_freqs):
    """50 DiT blocks; optional dual-stream overlap (MLP || next gather+FA)."""
    if not _layer_overlap():
        for block in blocks:
            h = block(h, t_emb, mod_segments, rope_freqs)
        return h

    stream_mlp = npu_stream("mlp")
    stream_attn = npu_stream("attn")
    mlp_done: Optional[object] = None
    for block in blocks:
        if mlp_done is not None:
            stream_attn.wait_event(mlp_done)
        with torch.npu.stream(stream_attn):
            h = block.forward_attn(h, t_emb, mod_segments, rope_freqs)
            attn_done = record_event(stream_attn)
        with torch.npu.stream(stream_mlp):
            stream_mlp.wait_event(attn_done)
            h = block.forward_mlp(h, t_emb, mod_segments)
            mlp_done = record_event(stream_mlp)
    torch.npu.current_stream().wait_stream(stream_mlp)
    return h


class FinalLayer(nn.Module):
    def __init__(
        self,
        hidden,
        t_dim,
        video_dim,
        audio_dim,
        eps,
        apply_silu=True,
        adaln_dtype=None,
        dtype=None,
        device=None,
        operations=None,
    ):
        super().__init__()
        operations = operations or NpuOps
        self.norm = operations.RMSNorm(hidden, eps=eps, dtype=dtype, device=device)
        self.adaln_proj = AdalnProj(
            t_dim,
            hidden,
            2,
            1,
            apply_silu=apply_silu,
            dtype=adaln_dtype if adaln_dtype is not None else dtype,
            device=device,
            operations=operations,
        )
        self.video_out = operations.Linear(hidden, video_dim, bias=True, dtype=torch.float32, device=device)
        self.audio_out = operations.Linear(hidden, audio_dim, bias=True, dtype=torch.float32, device=device)

    def forward(self, x, t_emb, video_seg, audio_seg):
        shift, scale = self.adaln_proj(t_emb)
        va, vb, vrow = video_seg
        aa, ab, arow = audio_seg
        hv = (self.norm(x[va:vb]) * (1.0 + scale[vrow]) + shift[vrow]).to(torch.float32)
        ha = (self.norm(x[aa:ab]) * (1.0 + scale[arow]) + shift[arow]).to(torch.float32)
        return self.video_out(hv), self.audio_out(ha)


class PackedLayout:
    def __init__(self, text_len, latent_t, latent_h, latent_w, audio_t, keyframes=None, refs=None, frame_count=None):
        frame, w_grid = _frame_grid(latent_h, latent_w)
        frame_rows = frame.shape[0]
        segments = [("text", text_len)]
        g = torch.zeros(text_len, 3, dtype=torch.float64)
        g[:, 0] = torch.arange(text_len, dtype=torch.float64)
        pos = [g]
        img_pos, img_update = [], []
        audio_pos, audio_update = [], []
        cursor = text_len
        row = text_len

        if keyframes:
            for kf in keyframes:
                pixel_index = kf["resolved_frame_index"]
                if pixel_index == 0:
                    cond_t = float(text_len)
                elif frame_count is not None and pixel_index == frame_count - 1:
                    cond_t = float(text_len) + sum(_video_t_spans(latent_t)) - FRAME_RESCALE
                else:
                    raise ValueError("only first/last keyframe anchors are supported")
                g = torch.empty(frame_rows, 3, dtype=torch.float64)
                g[:, 0] = cond_t
                g[:, 1:] = frame
                segments.append(("cond", frame_rows))
                pos.append(g)
                img_pos.append(torch.arange(row, row + frame_rows))
                img_update.append(torch.zeros(frame_rows, dtype=torch.bool))
                row += frame_rows

        target_audio_w = (float(w_grid[0]), float(w_grid[-1]))
        if refs:
            cursor = float(text_len)
            for blk in refs:
                kind = blk["kind"]
                if kind == "image":
                    r_frame, _ = _frame_grid(blk["latent_h"], blk["latent_w"])
                    n = r_frame.shape[0]
                    g = torch.empty(n, 3, dtype=torch.float64)
                    g[:, 0] = cursor
                    g[:, 1:] = r_frame
                    segments.append(("ref_img", n))
                    pos.append(g)
                    img_pos.append(torch.arange(row, row + n))
                    img_update.append(torch.zeros(n, dtype=torch.bool))
                    row += n
                    cursor += 1.0
                elif kind == "audio":
                    rt = blk["ref_audio_t"]
                    if rt > 0:
                        segments.append(("ref_audio", rt * 2))
                        pos.append(_audio_grid(cursor, rt, *target_audio_w))
                        audio_pos.append(torch.arange(row, row + rt * 2))
                        audio_update.append(torch.zeros(rt * 2, dtype=torch.bool))
                        row += rt * 2
                    cursor += float(rt)
                elif kind in ("video", "video_audio"):
                    rt = blk["ref_audio_t"]
                    vt = blk["latent_t"]
                    r_frame, r_w_grid = _frame_grid(blk["latent_h"], blk["latent_w"])
                    if rt > 0:
                        segments.append(("ref_audio", rt * 2))
                        pos.append(_audio_grid(cursor, rt, float(r_w_grid[0]), float(r_w_grid[-1])))
                        audio_pos.append(torch.arange(row, row + rt * 2))
                        audio_update.append(torch.zeros(rt * 2, dtype=torch.bool))
                        row += rt * 2
                    n = vt * r_frame.shape[0]
                    segments.append(("ref_img", n))
                    pos.append(_video_grid(vt, r_frame, cursor))
                    img_pos.append(torch.arange(row, row + n))
                    img_update.append(torch.zeros(n, dtype=torch.bool))
                    row += n
                    cursor += max(float(rt), sum(_video_t_spans(vt)))

        segments.append(("audio", audio_t * 2))
        pos.append(_audio_grid(cursor, audio_t, *target_audio_w))
        audio_pos.append(torch.arange(row, row + audio_t * 2))
        audio_update.append(torch.ones(audio_t * 2, dtype=torch.bool))
        row += audio_t * 2

        n_video = latent_t * frame_rows
        segments.append(("video", n_video))
        pos.append(_video_grid(latent_t, frame, cursor))
        img_pos.append(torch.arange(row, row + n_video))
        img_update.append(torch.ones(n_video, dtype=torch.bool))
        row += n_video

        self.seq_len = row
        self.position_ids = torch.cat(pos)
        self.img_pos = torch.cat(img_pos) if img_pos else torch.zeros(0, dtype=torch.long)
        self.img_update = torch.cat(img_update) if img_update else torch.zeros(0, dtype=torch.bool)
        self.audio_pos = torch.cat(audio_pos) if audio_pos else torch.zeros(0, dtype=torch.long)
        self.audio_update = torch.cat(audio_update) if audio_update else torch.zeros(0, dtype=torch.bool)
        self.signature = (text_len, latent_t, latent_h, latent_w, audio_t)
        seg_abs = []
        off = 0
        for kind, n in segments:
            seg_abs.append((off, off + n, kind))
            off += n
        self.segments = seg_abs


class MiniMaxH3DiT(nn.Module):
    """Pruned Comfy Ref2VA DiT (adaln curves)."""

    def __init__(
        self,
        hidden_size=5376,
        num_layers=50,
        token_refiner_num_layers=2,
        num_attention_heads=56,
        attention_head_dim=128,
        ffn_hidden_size=14336,
        latents_dim=24,
        audio_latents_dim=32,
        patch_size=(1, 2, 2),
        text_dim=5120,
        time_embed_dim=8,
        rope_inv_freq_len=16,
        norm_eps=1e-5,
        qk_norm_eps=1e-5,
        final_norm_eps=1e-5,
        sigma_shift_video=12.0,
        sigma_shift_audio=3.0,
        adaln_curve_grid=1025,
        dtype=None,
        device=None,
        operations=None,
        **_kwargs,
    ):
        super().__init__()
        operations = operations or NpuOps
        self.dtype = dtype or torch.bfloat16
        self.hidden_size = hidden_size
        self.patch_size = tuple(patch_size)
        self.latents_dim = latents_dim
        self.audio_latents_dim = audio_latents_dim
        self.sigma_shift_video = sigma_shift_video
        self.sigma_shift_audio = sigma_shift_audio
        self.use_adaln_curves = adaln_curve_grid is not None
        curve = {
            "apply_silu": not self.use_adaln_curves,
            "adaln_dtype": torch.float16 if self.use_adaln_curves else self.dtype,
        }
        video_patch_dim = latents_dim * self.patch_size[0] * self.patch_size[1] * self.patch_size[2]

        self.video_patch_proj = operations.Linear(
            video_patch_dim, hidden_size, bias=True, dtype=torch.float32, device=device
        )
        self.audio_patch_proj = operations.Linear(
            audio_latents_dim, hidden_size, bias=True, dtype=torch.float32, device=device
        )
        self.condition_proj = operations.Linear(
            text_dim, hidden_size, bias=True, dtype=self.dtype, device=device
        )
        if self.use_adaln_curves:
            self.register_buffer(
                "adaln_t_table", torch.empty(adaln_curve_grid, time_embed_dim, dtype=torch.float32)
            )
        self.rope = nn.Module()
        self.rope.register_buffer("inv_freq", torch.empty(rope_inv_freq_len, dtype=torch.float32))
        self.token_refiner = TokenRefiner(
            token_refiner_num_layers,
            hidden_size,
            num_attention_heads,
            attention_head_dim,
            ffn_hidden_size,
            norm_eps,
            qk_norm_eps,
            final_norm_eps,
            dtype=self.dtype,
            device=device,
            operations=operations,
        )
        self.blocks = nn.ModuleList(
            [
                DiTBlock(
                    hidden_size,
                    num_attention_heads,
                    attention_head_dim,
                    ffn_hidden_size,
                    time_embed_dim,
                    norm_eps,
                    qk_norm_eps,
                    **curve,
                    dtype=self.dtype,
                    device=device,
                    operations=operations,
                )
                for _ in range(num_layers)
            ]
        )
        self.final_layer = FinalLayer(
            hidden_size,
            time_embed_dim,
            video_patch_dim,
            audio_latents_dim,
            final_norm_eps,
            **curve,
            dtype=self.dtype,
            device=device,
            operations=operations,
        )

    def _cond_video_rows(self, payload, device):
        rows = []
        aug = payload.get("visual_cond_noise_aug", VISUAL_COND_TIMESTEP)
        seed = int(payload.get("seed", 0))
        for z in payload.get("cond_video_latents") or []:
            r = patchify_video(z.to(torch.float32), self.patch_size)
            if aug < 1.0:
                gen = torch.Generator("cpu").manual_seed(seed)
                noise = torch.randn(r.shape, generator=gen, dtype=torch.float32)
                r = aug * r + (1.0 - aug) * noise.to(r.device)
            rows.append(r.to(device))
        return torch.cat(rows, dim=0) if rows else None

    def _cond_audio_rows(self, payload, device):
        rows = []
        aug = payload.get("audio_cond_noise_aug", AUDIO_COND_TIMESTEP)
        seed = int(payload.get("seed", 0)) + 1
        for z in payload.get("cond_audio_latents") or []:
            r = pack_audio(z.to(torch.float32))
            if aug < 1.0:
                gen = torch.Generator("cpu").manual_seed(seed)
                noise = torch.randn(r.shape, generator=gen, dtype=torch.float32)
                r = aug * r + (1.0 - aug) * noise.to(r.device)
            rows.append(r.to(device))
        return torch.cat(rows, dim=0) if rows else None

    def rope_freqs(self, position_ids, device):
        pos = position_ids.to(torch.float32).to(device)
        inv = self.rope.inv_freq.to(device=device)
        per_axis = pos.unsqueeze(-1) * inv.view(1, 1, -1)
        t_f, h_f, w_f = per_axis.unbind(dim=1)
        half = torch.cat((t_f, h_f, w_f), dim=-1)
        return torch.cat((half, half), dim=-1)

    def forward(
        self,
        x,
        timestep,
        context,
        minimax_payload: Optional[dict[str, Any]] = None,
        **_kwargs,
    ):
        video_x, audio_x = x[0], x[1]
        orig_t, orig_h, orig_w = video_x.shape[2], video_x.shape[3], video_x.shape[4]
        video_x = pad_to_patch_size(video_x, self.patch_size)
        if video_x.shape[0] != 1:
            raise ValueError("MiniMax H3 supports batch size 1")
        payload = minimax_payload or {}
        device = video_x.device
        dtype = context.dtype

        latent_t, lat_h, lat_w = video_x.shape[2], video_x.shape[3], video_x.shape[4]
        audio_t = audio_x.shape[-1]
        text_len = context.shape[1]
        layout = payload.get("layout")
        if layout is None or layout.signature != (text_len, latent_t, lat_h, lat_w, audio_t):
            layout = PackedLayout(
                text_len,
                latent_t,
                lat_h,
                lat_w,
                audio_t,
                keyframes=payload.get("keyframes"),
                refs=payload.get("refs"),
                frame_count=payload.get("frame_count"),
            )

        shift_v = float(self.sigma_shift_video)
        shift_a = float(self.sigma_shift_audio)
        sigma_v = (timestep.flatten()[0] / 1000.0).float().clamp(min=1e-6)
        t_v = float(1.0 - sigma_v)
        t_a = float(1.0 - time_shift_sigma(sigma_v, shift_v, shift_a))

        vis_aug = float(payload.get("visual_cond_noise_aug", VISUAL_COND_TIMESTEP))
        aud_aug = float(payload.get("audio_cond_noise_aug", AUDIO_COND_TIMESTEP))
        has_vis_cond = any(k in ("cond", "ref_img") for _, _, k in layout.segments)
        has_aud_cond = any(k == "ref_audio" for _, _, k in layout.segments)
        seg_t = {
            "text": t_v,
            "video": t_v,
            "audio": t_a,
            "cond": max(t_v, vis_aug),
            "ref_img": max(t_v, vis_aug),
            "ref_audio": max(t_a, aud_aug),
        }
        unique_t = sorted(
            {t_v, t_a}
            | ({seg_t["cond"]} if has_vis_cond else set())
            | ({seg_t["ref_audio"]} if has_aud_cond else set())
        )
        t_row = {t: i for i, t in enumerate(unique_t)}
        seg_tag = {"text": 1, "video": 0, "audio": 2, "cond": 0, "ref_img": 0, "ref_audio": 2}

        text_tags = payload.get("text_token_tags")
        mod_segments = []
        for a, b, kind in layout.segments:
            row_base = t_row[seg_t[kind]] * 3
            if kind == "text" and text_tags is not None:
                tags = text_tags.view(-1).tolist()
                run_start = 0
                for i in range(1, b - a + 1):
                    if i == b - a or tags[i] != tags[run_start]:
                        mod_segments.append((a + run_start, a + i, row_base + int(tags[run_start])))
                        run_start = i
            else:
                mod_segments.append((a, b, row_base + seg_tag[kind]))

        img_update = layout.img_update.to(device)
        audio_update = layout.audio_update.to(device)
        video_rows = patchify_video(video_x.to(torch.float32), self.patch_size)
        audio_rows = pack_audio(audio_x.to(torch.float32))
        cond_video_rows = self._cond_video_rows(payload, device)
        cond_audio_rows = self._cond_audio_rows(payload, device)

        all_video_rows = video_rows
        if cond_video_rows is not None:
            n_cond = int((~img_update).sum().item())
            if cond_video_rows.shape[0] != n_cond:
                raise RuntimeError(
                    f"ref visual tokens {cond_video_rows.shape[0]} != layout cond slots {n_cond}"
                )
            all_video_rows = torch.empty(img_update.shape[0], video_rows.shape[1], dtype=torch.float32, device=device)
            all_video_rows[~img_update] = cond_video_rows
            all_video_rows[img_update] = video_rows
        all_audio_rows = audio_rows
        if cond_audio_rows is not None:
            all_audio_rows = torch.empty(audio_update.shape[0], audio_rows.shape[1], dtype=torch.float32, device=device)
            all_audio_rows[~audio_update] = cond_audio_rows
            all_audio_rows[audio_update] = audio_rows

        video_embed = self.video_patch_proj(all_video_rows).to(dtype)
        audio_embed = self.audio_patch_proj(all_audio_rows).to(dtype)
        text_states = context[0]
        if text_states.shape[-1] != self.hidden_size:
            text_states = self.token_refiner(self.condition_proj(text_states))

        h = torch.empty(layout.seq_len, self.hidden_size, dtype=dtype, device=device)
        voff = aoff = 0
        for a, b, kind in layout.segments:
            n = b - a
            if kind == "text":
                h[a:b] = text_states
            elif kind in ("cond", "ref_img", "video"):
                h[a:b] = video_embed[voff : voff + n]
                voff += n
            else:
                h[a:b] = audio_embed[aoff : aoff + n]
                aoff += n

        t_vals = torch.tensor(unique_t, dtype=torch.float32, device=device)
        if self.use_adaln_curves:
            table = self.adaln_t_table.to(device=device)
            pos = t_vals.clamp(0.0, 1.0) * (table.shape[0] - 1)
            i0 = pos.floor().long().clamp(max=table.shape[0] - 2)
            t_emb = torch.lerp(table[i0], table[i0 + 1], (pos - i0).unsqueeze(1))
        else:
            raise RuntimeError("non-curve time embedder not packaged; use pruned checkpoint")

        layerwise = os.environ.get("H3_NPU_LAYERWISE_OFFLOAD", "0") == "1"
        pad_n = 0
        if not layerwise:
            ws = dist_state().world_size if is_distributed() else 1
            # Local Q length must be %8 for FA/SDPA alignment: S % (8*world) == 0.
            align = 8 * max(ws, 1)
            pad_n = (align - h.shape[0] % align) % align
            if pad_n:
                h = torch.cat(
                    [h, torch.zeros(pad_n, h.shape[1], dtype=h.dtype, device=h.device)], dim=0
                )
                extra_pos = layout.position_ids[-1:].repeat(pad_n, 1)
                pos_ids = torch.cat([layout.position_ids, extra_pos], dim=0)
            else:
                pos_ids = layout.position_ids
        else:
            pos_ids = layout.position_ids

        if dist_state().rank == 0 and not getattr(self, "_logged_seq", False):
            ws = dist_state().world_size if is_distributed() else 1
            print(
                f"[dit] seq={h.shape[0]} pad={pad_n} world={ws} "
                f"local={h.shape[0] // max(ws, 1)} hidden={h.shape[1]}",
                flush=True,
            )
            self._logged_seq = True

        if getattr(self, "_rope_cache_key", None) != (h.shape[0], str(dtype)):
            self._rope_cache = rope_rotation_table(self.rope_freqs(pos_ids, device), dtype)
            self._rope_cache_key = (h.shape[0], str(dtype))
        rope_freqs = self._rope_cache
        if is_distributed() and not layerwise:
            s0, s1 = split_range(h.shape[0])
            h = h[s0:s1]
            mod_segments = clip_mod_segments(mod_segments, s0, s1)
        if layerwise:
            for block in self.blocks:
                block.to(device)
                h = block(h, t_emb, mod_segments, rope_freqs)
                block.to("cpu")
                if torch.npu.is_available():
                    torch.npu.empty_cache()
        else:
            h = _run_dit_blocks(self.blocks, h, t_emb, mod_segments, rope_freqs)

        if is_distributed() and not layerwise:
            h = all_gather_seq(h)
            if pad_n:
                h = h[:-pad_n]
        if layerwise:
            self.final_layer.to(device)
        video_seg = next((a, b, t_row[seg_t["video"]]) for a, b, k in layout.segments if k == "video")
        audio_seg = next((a, b, t_row[seg_t["audio"]]) for a, b, k in layout.segments if k == "audio")
        v, a = self.final_layer(h, t_emb, video_seg, audio_seg)
        if layerwise:
            self.final_layer.to("cpu")
        video_out = unpatchify_video(v, latent_t, lat_h // 2, lat_w // 2, self.latents_dim, self.patch_size)
        video_out = video_out[:, :, :orig_t, :orig_h, :orig_w]
        audio_out = unpack_audio(a)
        slope_a = time_shift_slope(sigma_v, shift_v, shift_a).to(audio_out.dtype)
        return [-video_out.to(video_x.dtype), (-slope_a) * audio_out.to(audio_x.dtype)]
