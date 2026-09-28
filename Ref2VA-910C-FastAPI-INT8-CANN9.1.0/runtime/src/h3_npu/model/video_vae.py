"""MiniMax-H3 video VAE decoder on torch_npu (no ComfyUI).

Port of Comfy ``ldm/minimax/vae.py`` decode path: ViT3D + temporal chunking.
Encoder is omitted; generate only needs decode.
"""
from __future__ import annotations

import math
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

from h3_npu.ops.attention import fusion_attention
from h3_npu.ops.rope import apply_rope_split_half, rms_norm


IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

LATENTS_MEAN = [
    0.858090341091156, -0.9606591463088989, 1.0661640167236328, -0.5090325474739075,
    -0.2727581858634949, -1.3675414323806763, -0.2553254961967468, -0.26907554268836975,
    -0.5376840829849243, -0.0464097298681736, 0.6657370328903198, 0.19690127670764923,
    -0.5460608005523682, -0.4035342037677765, -0.23683024942874908, 0.25928452610969543,
    -0.30133944749832153, 0.211341992020607, -1.1206848621368408, 0.3581933379173279,
    -0.04225143790245056, 0.2604829967021942, 0.22864092886447906, 0.7056031823158264,
]
LATENTS_STD = [
    1.2223774194717407, 1.2767263650894165, 1.68317747116088865, 1.7549455165863037,
    1.5636216402053833, 2.194143533706665, 0.96531379222869875, 1.05698859691619875,
    0.841948926448822, 0.7729952931404114, 1.8955937623977661, 0.946841835975647,
    0.7996809482574463, 0.44988900423049925, 0.7197399735450745, 0.69362932443618775,
    2.961095094680786, 2.7694199085235595, 3.0496184825897215, 2.1088054180145265,
    3.276226282119751, 3.1627357006073, 2.28168129920959475, 2.6127843856811525,
]


def _create_token_ids(patch_dims, device, dtype):
    coords_list = []
    for dim_size in patch_dims:
        coords = torch.arange(0.5, dim_size, dtype=dtype, device=device)
        coords = coords / dim_size
        coords = 2.0 * coords - 1.0
        coords_list.append(coords)
    coords = torch.stack(torch.meshgrid(*coords_list, indexing="ij"), dim=-1)
    return coords.flatten(0, len(patch_dims) - 1).unsqueeze(0)


class RotaryEmbeddingND(nn.Module):
    def __init__(self, dim, rotary_base=100.0, n_dim=3):
        super().__init__()
        self.n_dim = n_dim
        self.angle_scale = 2.0 * math.pi
        inv_freq = 1 / rotary_base ** torch.arange(0, 1, 2 * n_dim / dim, dtype=torch.float32)
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    def forward(self, img_ids):
        angles = (
            self.angle_scale
            * img_ids[:, :, :, None].float()
            * self.inv_freq.to(img_ids.device)[None, None, None, :]
        )
        angles = angles.flatten(2, 3)
        c, s = torch.cos(angles), torch.sin(angles)
        table = torch.stack([c, -s, s, c], dim=-1).reshape(*angles.shape[:2], 1, angles.shape[-1], 2, 2)
        return table.to(img_ids.dtype)


class _RMS(nn.Module):
    def __init__(self, dim, eps=1e-5, affine=True):
        super().__init__()
        self.eps = eps
        if affine:
            self.weight = nn.Parameter(torch.ones(dim))
        else:
            self.register_parameter("weight", None)

    def forward(self, x):
        w = self.weight if self.weight is not None else torch.ones(x.shape[-1], device=x.device, dtype=x.dtype)
        return rms_norm(x, w, self.eps)


class FeedForward(nn.Module):
    def __init__(self, dim, mult=4, bias=True):
        super().__init__()
        inner = dim * mult
        self.w1 = nn.Linear(dim, inner * 2, bias=bias)
        self.w2 = nn.Linear(inner, dim, bias=bias)

    def forward(self, x):
        gate, y = self.w1(x).chunk(2, dim=-1)
        return self.w2(F.silu(gate) * y)


class VaeAttention(nn.Module):
    def __init__(self, heads, dim_head, bias=True, eps=1e-5):
        super().__init__()
        self.dim_head = dim_head
        self.heads = heads
        inner = dim_head * heads
        self.norm_q = _RMS(dim_head, eps=eps, affine=False)
        self.norm_k = _RMS(dim_head, eps=eps, affine=False)
        self.to_qkv = nn.Linear(inner, inner * 3, bias=bias)
        self.to_out = nn.Linear(inner, inner, bias=bias)

    def forward(self, x, rotary_pos_emb=None):
        b, s, _ = x.shape
        qkv = self.to_qkv(x).view(b, s, self.heads, 3 * self.dim_head)
        q, k, v = torch.chunk(qkv, 3, dim=-1)
        q = self.norm_q(q)
        k = self.norm_k(k)
        if rotary_pos_emb is not None:
            rot = rotary_pos_emb.shape[-3] * 2
            q[..., :rot], k[..., :rot] = apply_rope_split_half(
                q[..., :rot], k[..., :rot], rotary_pos_emb, rot
            )
        out = fusion_attention(q, k, v)
        out = out.reshape(b, s, self.heads * self.dim_head).nan_to_num_(0.0)
        return self.to_out(out)


class TransformerBlock(nn.Module):
    def __init__(self, heads, dim_head, bias=True, eps=1e-5):
        super().__init__()
        dim = heads * dim_head
        self.norm1 = _RMS(dim, eps=eps, affine=True)
        self.attn = VaeAttention(heads=heads, dim_head=dim_head, bias=bias, eps=eps)
        self.scale1 = nn.Parameter(torch.empty(dim))
        self.norm2 = _RMS(dim, eps=eps, affine=True)
        self.ff = FeedForward(dim=dim, bias=bias)
        self.scale2 = nn.Parameter(torch.empty(dim))

    def forward(self, x, rotary_pos_emb=None):
        x = x + self.attn(self.norm1(x), rotary_pos_emb) * self.scale1.to(dtype=x.dtype)
        return x + self.ff(self.norm2(x)) * self.scale2.to(dtype=x.dtype)


class ViT3DDecoder(nn.Module):
    def __init__(
        self,
        patch_size=16,
        patch_size_t=4,
        in_channels=24,
        out_channels=3,
        num_layers=36,
        heads=32,
        dim_head=64,
        rope_theta=100.0,
        rope_dim_ratio=0.75,
        bias=True,
        eps=1e-5,
        num_register_tokens=4,
    ):
        super().__init__()
        dim = heads * dim_head
        self.patch_size = patch_size
        self.patch_size_t = patch_size_t
        self.out_channels = out_channels
        self.num_register_tokens = num_register_tokens
        self.pos_embed = RotaryEmbeddingND(int(dim_head * rope_dim_ratio), rope_theta, n_dim=3)
        self.x_embedder = nn.Linear(in_channels, dim)
        self.register_tokens = nn.Parameter(torch.empty(1, num_register_tokens, dim))
        self.register_buffer("mask_token", torch.empty(1, 1, dim))
        self.transformer_blocks = nn.ModuleList(
            [TransformerBlock(heads=heads, dim_head=dim_head, bias=bias, eps=eps) for _ in range(num_layers)]
        )
        self.norm_out = nn.LayerNorm(dim, eps=eps)
        self.proj_out = nn.Linear(dim, out_channels * patch_size_t * patch_size * patch_size)

    def forward(self, x):
        b, _c, lt, lh, lw = x.shape
        h = self.x_embedder(x.flatten(2).transpose(1, 2))
        n_patch = h.shape[1]
        n_suffix = 1 + self.num_register_tokens
        h = torch.cat(
            [h, self.register_tokens.to(dtype=h.dtype, device=h.device).expand(b, -1, -1), torch.zeros_like(h[:, 0:1, :])],
            dim=1,
        )
        img_ids = _create_token_ids((lt, lh, lw), x.device, x.dtype).expand(b, -1, -1)
        suffix_ids = torch.zeros((b, n_suffix, 3), device=x.device, dtype=img_ids.dtype)
        rotary = self.pos_embed(torch.cat([img_ids, suffix_ids], dim=1))
        for block in self.transformer_blocks:
            h = block(h, rotary)
        output = self.proj_out(self.norm_out(h))[:, :n_patch, :]
        output = output.view(b, lt, lh, lw, self.out_channels, self.patch_size_t, self.patch_size, self.patch_size)
        output = output.permute(0, 4, 1, 5, 2, 6, 3, 7).contiguous()
        return output.reshape(
            b,
            self.out_channels,
            lt * self.patch_size_t,
            lh * self.patch_size,
            lw * self.patch_size,
        )


class CausalConv3d(nn.Conv3d):
    """Reflect spatial pad + causal (front-zero) temporal pad. T=1 uses last tap."""

    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0):
        super().__init__(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=0)
        self.causal_padding = (padding, padding, padding) if isinstance(padding, int) else tuple(padding)

    def forward(self, x):
        pt, ph, pw = self.causal_padding
        if pt or ph or pw:
            spatial = (pw, pw, ph, ph, 0, 0)
            try:
                x = F.pad(x, spatial, mode="reflect")
            except RuntimeError:
                x = F.pad(x, spatial, mode="replicate")
            if x.shape[2] == 1:
                w = self.weight[:, :, -x.shape[2] :, :, :]
                return F.conv3d(x, w, self.bias, self.stride, 0, self.dilation, self.groups)
            x = F.pad(x, (0, 0, 0, 0, pt * 2, 0), mode="constant")
        return F.conv3d(x, self.weight, self.bias, self.stride, 0, self.dilation, self.groups)


class TemporalIsolatedGroupNorm(nn.GroupNorm):
    def forward(self, x):
        if x.dim() == 5:
            b, c, t, h, w = x.shape
            x = x.permute(0, 2, 1, 3, 4).contiguous().view(b * t, c, 1, h, w)
            x = super().forward(x)
            return x.view(b, t, c, h, w).permute(0, 2, 1, 3, 4).contiguous()
        return super().forward(x)


def group_norm_3d(num_channels):
    return TemporalIsolatedGroupNorm(num_groups=32, num_channels=num_channels, eps=1e-6, affine=True)


class Downsample3D(nn.Module):
    def __init__(self, in_channels, out_channels, time_stride=1, space_stride=2):
        super().__init__()
        self.space_stride = space_stride
        self.conv = CausalConv3d(
            in_channels,
            out_channels,
            kernel_size=3,
            padding=(1, 0, 0),
            stride=(time_stride, space_stride, space_stride),
        )

    def forward(self, x):
        if self.space_stride == 2:
            try:
                x = F.pad(x, (0, 1, 0, 1, 0, 0), mode="reflect")
            except RuntimeError:
                x = F.pad(x, (0, 1, 0, 1, 0, 0), mode="replicate")
        return self.conv(x)


class ResnetBlock3D(nn.Module):
    def __init__(self, in_channels, out_channels=None):
        super().__init__()
        self.in_channels = in_channels
        out_channels = in_channels if out_channels is None else out_channels
        self.out_channels = out_channels
        self.norm1 = group_norm_3d(in_channels)
        self.norm2 = group_norm_3d(out_channels)
        self.conv1 = CausalConv3d(in_channels, out_channels, kernel_size=3, padding=1)
        self.conv2 = CausalConv3d(out_channels, out_channels, kernel_size=3, padding=1)
        if in_channels != out_channels:
            self.nin_shortcut = CausalConv3d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        h = self.conv1(F.silu(self.norm1(x), inplace=True))
        h = self.conv2(F.silu(self.norm2(h), inplace=True))
        if self.in_channels != self.out_channels:
            x = self.nin_shortcut(x)
        return h.add_(x)


class EncoderFCN3D(nn.Module):
    def __init__(self, ch, ch_mult, space_down, time_down, num_res_blocks, in_channels, z_channels, double_z=True):
        super().__init__()
        self.num_levels = len(ch_mult)
        if isinstance(num_res_blocks, int):
            num_res_blocks = [num_res_blocks] * self.num_levels
        self.num_res_blocks = num_res_blocks
        block_mid = [ch * ch_mult[i] for i in range(self.num_levels)]
        block_in = [block_mid[0]] + block_mid[:-1]
        block_out = block_mid
        self.conv_in = CausalConv3d(in_channels, block_in[0], kernel_size=3, padding=1)
        self.down = nn.ModuleList()
        for i_level in range(self.num_levels):
            down = nn.Module()
            down.block = nn.ModuleList()
            for i in range(self.num_res_blocks[i_level]):
                down.block.append(
                    ResnetBlock3D(
                        in_channels=block_in[i_level] if i == 0 else block_mid[i_level],
                        out_channels=block_mid[i_level],
                    )
                )
            if space_down[i_level] * time_down[i_level] > 1:
                down.downsample = Downsample3D(
                    block_mid[i_level],
                    block_out[i_level],
                    time_stride=time_down[i_level],
                    space_stride=space_down[i_level],
                )
            self.down.append(down)
        self.norm_out = group_norm_3d(block_out[-1])
        self.conv_out = CausalConv3d(
            block_out[-1],
            2 * z_channels if double_z else z_channels,
            kernel_size=3,
            padding=1,
        )

    def forward(self, x):
        h = self.conv_in(x)
        for i_level in range(self.num_levels):
            for i_block in range(self.num_res_blocks[i_level]):
                h = self.down[i_level].block[i_block](h)
            if hasattr(self.down[i_level], "downsample"):
                h = self.down[i_level].downsample(h)
        h = F.silu(self.norm_out(h))
        return self.conv_out(h)


class MiniMaxH3VideoVAE(nn.Module):
    def __init__(
        self,
        in_channels=3,
        out_ch=3,
        ch=128,
        embed_dim=24,
        z_channels=24,
        ch_mult=(1, 2, 2, 4, 4, 8),
        num_res_blocks=2,
        space_down=(2, 2, 2, 2, 1, 1),
        time_down=(1, 2, 2, 1, 1, 1),
        clip_length=17,
        token_drop=3,
        tile_size=256,
        tile_overlap_min=64,
        tiling=True,
        include_encoder=False,
    ):
        super().__init__()
        self.vae_ratio = int(math.prod(space_down))
        self.vae_ratio_t = int(math.prod(time_down))
        self.clip_length = clip_length
        self.token_drop = token_drop
        self.frame_pre_padding = (-clip_length) % self.vae_ratio_t
        self.tokens_chunk_size = math.ceil(clip_length / self.vae_ratio_t)
        self.token_overlap = (-token_drop) % self.tokens_chunk_size
        self.frame_overlap = max(self.token_overlap * self.vae_ratio_t - self.frame_pre_padding, 0)
        self.tiling = tiling
        self.tile_size = tile_size
        self.tile_overlap_min = tile_overlap_min
        self.include_encoder = include_encoder
        if include_encoder:
            self.encoder = EncoderFCN3D(
                ch=ch,
                ch_mult=list(ch_mult),
                space_down=list(space_down),
                time_down=list(time_down),
                num_res_blocks=num_res_blocks,
                in_channels=in_channels,
                z_channels=z_channels,
                double_z=True,
            )
            self.quant_conv = nn.Conv3d(z_channels * 2, 2 * embed_dim, 1)
        self.post_quant_conv = nn.Conv3d(embed_dim, z_channels, 1)
        self.decoder = ViT3DDecoder(
            patch_size=self.vae_ratio,
            patch_size_t=self.vae_ratio_t,
            in_channels=z_channels,
            out_channels=out_ch,
        )
        self.register_buffer("latents_mean", torch.tensor(LATENTS_MEAN))
        self.register_buffer("latents_std", torch.tensor(LATENTS_STD))
        self.register_buffer("pixel_mean", torch.tensor(IMAGENET_MEAN).view(1, 3, 1, 1, 1), persistent=False)
        self.register_buffer("pixel_std", torch.tensor(IMAGENET_STD).view(1, 3, 1, 1, 1), persistent=False)

    def _encode_moments(self, x):
        if not self.include_encoder:
            raise RuntimeError("VAE encoder weights were not loaded")
        return self.quant_conv(self.encoder(x))

    def _adaptive_encode(self, x):
        if self.tiling:
            return self.tiled_encode(x)
        return self._encode_moments(x)

    def tiled_encode(self, x):
        height, width = x.shape[-2], x.shape[-1]
        y_idx, y_len, y_overlap = self.split_tiles(height)
        x_idx, x_len, x_overlap = self.split_tiles(width)
        rows = []
        for i_pos, i_len in zip(y_idx, y_len):
            row = []
            for j_pos, j_len in zip(x_idx, x_len):
                tile = x[..., i_pos : i_pos + i_len, j_pos : j_pos + j_len]
                row.append(self._encode_moments(tile))
            rows.append(row)
        latent_y_overlap = [o // self.vae_ratio for o in y_overlap]
        latent_x_overlap = [o // self.vae_ratio for o in x_overlap]
        result_rows = []
        for i, row in enumerate(rows):
            result_row = []
            for j, tile in enumerate(row):
                if i > 0:
                    tile = self.blend(rows[i - 1][j], tile, latent_y_overlap[i - 1], dim=-2)
                if j > 0:
                    tile = self.blend(row[j - 1], tile, latent_x_overlap[j - 1], dim=-1)
                if i < len(rows) - 1:
                    tile = tile[..., :-latent_y_overlap[i], :]
                if j < len(row) - 1:
                    tile = tile[..., :, :-latent_x_overlap[j]]
                result_row.append(tile)
            result_rows.append(torch.cat(result_row, dim=-1))
        return torch.cat(result_rows, dim=-2)

    def encode_temporal(self, x):
        if x.shape[2] % self.clip_length != 0:
            pad_size = (-x.shape[2]) % self.clip_length
            pad_frames = x[:, :, -1:].repeat(1, 1, pad_size, 1, 1)
            x = torch.cat([x, pad_frames], dim=2)
        z_list = []
        for i in range(x.shape[2] // self.clip_length):
            clip_x = x[:, :, i * self.clip_length : (i + 1) * self.clip_length, :, :]
            z_list.append(self._adaptive_encode(clip_x))
        z = torch.cat(z_list, dim=2)
        if self.token_drop > 0:
            z = z[:, :, : -self.token_drop]
        return z

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """x: [B, 3, T, H, W] in [-1, 1] → normalized latents [B, 24, T_lat, H/16, W/16]."""
        if x.ndim == 4:
            x = x.unsqueeze(2)
        x = x.add(1.0).mul_(0.5).sub_(self.pixel_mean.to(x)).div_(self.pixel_std.to(x))
        if x.shape[2] == 1:
            moments = self._adaptive_encode(x)
            moments = moments[:, :, -1:, :, :]
        else:
            moments = self.encode_temporal(x)
        mean = torch.chunk(moments.float(), 2, dim=1)[0]
        latents_mean = self.latents_mean.view(1, -1, 1, 1, 1).to(mean)
        latents_std = self.latents_std.view(1, -1, 1, 1, 1).to(mean)
        return (mean - latents_mean) / latents_std

    def _decode_pixels(self, z):
        return self.decoder(self.post_quant_conv(z))

    def split_tiles(self, input_len):
        tile_size = self.tile_size
        if tile_size >= input_len:
            return [0], [input_len], []
        n = math.ceil(input_len / tile_size)
        while True:
            overlaps = [self.tile_overlap_min] * (n - 1)
            remaining = tile_size * n - sum(overlaps) - input_len
            if remaining < 0:
                n += 1
            else:
                break
        remaining_units = remaining // self.vae_ratio
        for i in range(remaining_units):
            overlaps[i % (n - 1)] += self.vae_ratio
        tile_start_idx = [0]
        for i in range(n - 1):
            tile_start_idx.append(tile_start_idx[-1] + tile_size - overlaps[i])
        return tile_start_idx, [tile_size] * n, overlaps

    def blend(self, a, b, blend_extent, dim):
        blend_extent = min(a.shape[dim], b.shape[dim], blend_extent)
        positions = torch.arange(blend_extent, device=b.device, dtype=b.dtype)
        weight_a = 1 - positions / blend_extent
        weight_b = positions / blend_extent
        shape = [1] * a.ndim
        shape[dim] = blend_extent
        weight_a = weight_a.view(shape)
        weight_b = weight_b.view(shape)
        slice_a = [slice(None)] * a.ndim
        slice_a[dim] = slice(-blend_extent, None)
        slice_b = [slice(None)] * b.ndim
        slice_b[dim] = slice(0, blend_extent)
        blended = a[tuple(slice_a)] * weight_a + b[tuple(slice_b)] * weight_b
        if blend_extent < b.shape[dim]:
            slice_b_rest = [slice(None)] * b.ndim
            slice_b_rest[dim] = slice(blend_extent, None)
            return torch.cat([blended, b[tuple(slice_b_rest)]], dim=dim)
        return blended

    def tiled_decode(self, z):
        height, width = z.shape[-2] * self.vae_ratio, z.shape[-1] * self.vae_ratio
        y_idx, y_len, y_overlap = self.split_tiles(height)
        x_idx, x_len, x_overlap = self.split_tiles(width)
        canvas = None
        row_tails = []
        out_y = 0
        for i, (i_pos, i_len) in enumerate(zip(y_idx, y_len)):
            zi, zl = i_pos // self.vae_ratio, i_len // self.vae_ratio
            left_tail = None
            new_tails = []
            out_x = 0
            for j, (j_pos, j_len) in enumerate(zip(x_idx, x_len)):
                zj, zw = j_pos // self.vae_ratio, j_len // self.vae_ratio
                tile = self._decode_pixels(z[..., zi : zi + zl, zj : zj + zw])
                if i < len(y_idx) - 1:
                    new_tails.append(tile[..., -y_overlap[i] :, :].clone())
                next_left = tile[..., :, -x_overlap[j] :].clone() if j < len(x_idx) - 1 else None
                if i > 0:
                    tile = self.blend(row_tails[j], tile, y_overlap[i - 1], dim=-2)
                if j > 0:
                    tile = self.blend(left_tail, tile, x_overlap[j - 1], dim=-1)
                left_tail = next_left
                if i < len(y_idx) - 1:
                    tile = tile[..., :-y_overlap[i], :]
                if j < len(x_idx) - 1:
                    tile = tile[..., :, :-x_overlap[j]]
                if canvas is None:
                    canvas = torch.empty(*tile.shape[:-2], height, width, dtype=tile.dtype, device=tile.device)
                canvas[..., out_y : out_y + tile.shape[-2], out_x : out_x + tile.shape[-1]].copy_(tile)
                out_x += tile.shape[-1]
            row_tails = new_tails
            out_y += tile.shape[-2]
        return canvas

    def _adaptive_decode(self, z):
        if self.tiling:
            return self.tiled_decode(z)
        return self._decode_pixels(z)

    def _decode_temporal_pad_frames(self, z_len, pad_tokens):
        if pad_tokens <= 0:
            return 0
        intra_tail = self.clip_length % self.vae_ratio_t
        if intra_tail == 0:
            return pad_tokens * self.vae_ratio_t
        z_len_before_pad = z_len - pad_tokens
        return sum(
            (intra_tail if (z_len_before_pad + k) % self.tokens_chunk_size == 0 else self.vae_ratio_t)
            for k in range(pad_tokens)
        )

    def _decode_temporal_frame_plan(self, z_len, num_chunks, pad_tokens):
        chunk_dec = self.tokens_chunk_size * self.vae_ratio_t
        split_count = int(self.token_drop > 0) + 1
        total_frames = 0
        final_overlap_frames = 0
        for i in range(num_chunks):
            t_start_idx = i * self.tokens_chunk_size
            t_end_idx = t_start_idx + self.tokens_chunk_size + self.token_overlap
            clip_token_len = max(0, min(t_end_idx, z_len) - min(t_start_idx, z_len))
            clip_frame_len = clip_token_len * self.vae_ratio_t
            for j in range(split_count):
                f_start_idx = j * chunk_dec
                f_end_idx = min(f_start_idx + chunk_dec, clip_frame_len)
                chunk_frames = max(0, f_end_idx - f_start_idx - self.frame_pre_padding)
                if j == 0:
                    total_frames += chunk_frames
                else:
                    final_overlap_frames = chunk_frames
        total_frames += final_overlap_frames
        return total_frames - self._decode_temporal_pad_frames(z_len, pad_tokens)

    def decode_temporal(self, z):
        chunk_dec = self.tokens_chunk_size * self.vae_ratio_t
        split_count = int(self.token_drop > 0) + 1
        pseudo_total_tokens = z.shape[2] + self.token_drop
        pad_tokens = 0
        remainder = pseudo_total_tokens % self.tokens_chunk_size
        if remainder != 0:
            pad_tokens = self.tokens_chunk_size - remainder
            pseudo_total_tokens += pad_tokens
        num_chunks = pseudo_total_tokens // self.tokens_chunk_size - int(self.token_drop > 0)
        if num_chunks < 1:
            pad_tokens += self.tokens_chunk_size
            num_chunks += 1
        if pad_tokens > 0:
            z = torch.cat([z, z[:, :, -1:, :, :].repeat(1, 1, pad_tokens, 1, 1)], dim=2)
        output_frames = self._decode_temporal_frame_plan(z.shape[2], num_chunks, pad_tokens)
        dec = None
        dec_overlap = None
        write_pos = 0

        def write_part(part):
            nonlocal dec, write_pos
            part_frames = part.shape[2]
            if part_frames <= 0:
                return
            if dec is None:
                out_shape = list(part.shape)
                out_shape[2] = output_frames
                dec = torch.empty(out_shape, dtype=part.dtype, device=part.device)
            copy_frames = min(part_frames, max(0, dec.shape[2] - write_pos))
            if copy_frames > 0:
                dec[:, :, write_pos : write_pos + copy_frames].copy_(part[:, :, :copy_frames])
                write_pos += copy_frames

        for i in range(num_chunks):
            t_start = i * self.tokens_chunk_size
            t_end = t_start + self.tokens_chunk_size + self.token_overlap
            clip_dec = self._adaptive_decode(z[:, :, t_start:t_end])
            for j in range(split_count):
                f0 = j * chunk_dec
                f1 = min(f0 + chunk_dec, clip_dec.shape[2])
                chunk = clip_dec[:, :, f0:f1, :, :][:, :, self.frame_pre_padding :, :, :]
                if j == 0:
                    if dec_overlap is not None:
                        chunk = self.blend(dec_overlap, chunk, self.frame_overlap, dim=-3)
                        dec_overlap = None
                    write_part(chunk)
                else:
                    dec_overlap = chunk.contiguous()
            if i == num_chunks - 1 and dec_overlap is not None:
                write_part(dec_overlap)
                dec_overlap = None
        return dec

    def decode_temporal_parallel(self, z):
        """Temporal chunks split across HCCL ranks, then stitch on every rank."""
        from h3_npu.runtime.dist import broadcast_tensor, is_distributed, state as dist_state

        if not is_distributed() or dist_state().world_size <= 1:
            return self.decode_temporal(z)

        chunk_dec = self.tokens_chunk_size * self.vae_ratio_t
        split_count = int(self.token_drop > 0) + 1
        pseudo_total_tokens = z.shape[2] + self.token_drop
        pad_tokens = 0
        remainder = pseudo_total_tokens % self.tokens_chunk_size
        if remainder != 0:
            pad_tokens = self.tokens_chunk_size - remainder
            pseudo_total_tokens += pad_tokens
        num_chunks = pseudo_total_tokens // self.tokens_chunk_size - int(self.token_drop > 0)
        if num_chunks < 1:
            pad_tokens += self.tokens_chunk_size
            num_chunks += 1
        if pad_tokens > 0:
            z = torch.cat([z, z[:, :, -1:, :, :].repeat(1, 1, pad_tokens, 1, 1)], dim=2)
        output_frames = self._decode_temporal_frame_plan(z.shape[2], num_chunks, pad_tokens)
        rank = dist_state().rank
        ws = dist_state().world_size
        out_dtype = self.post_quant_conv.weight.dtype
        local = {}
        for i in range(num_chunks):
            if i % ws != rank:
                continue
            t_start = i * self.tokens_chunk_size
            t_end = t_start + self.tokens_chunk_size + self.token_overlap
            local[i] = self._adaptive_decode(z[:, :, t_start:t_end]).contiguous()
            print(f"[vae] rank={rank} chunk {i + 1}/{num_chunks} {tuple(local[i].shape)}", flush=True)

        clips = []
        for i in range(num_chunks):
            src = i % ws
            if rank == src:
                c = local[i]
                meta = torch.tensor(list(c.shape), device=z.device, dtype=torch.int32)
            else:
                meta = torch.zeros(5, device=z.device, dtype=torch.int32)
            broadcast_tensor(meta, src=src)
            shape = tuple(int(v) for v in meta.tolist())
            if rank != src:
                c = torch.empty(shape, device=z.device, dtype=out_dtype)
            broadcast_tensor(c, src=src)
            clips.append(c)
        del local

        dec = None
        dec_overlap = None
        write_pos = 0

        def write_part(part):
            nonlocal dec, write_pos
            part_frames = part.shape[2]
            if part_frames <= 0:
                return
            if dec is None:
                out_shape = list(part.shape)
                out_shape[2] = output_frames
                dec = torch.empty(out_shape, dtype=part.dtype, device=part.device)
            copy_frames = min(part_frames, max(0, dec.shape[2] - write_pos))
            if copy_frames > 0:
                dec[:, :, write_pos : write_pos + copy_frames].copy_(part[:, :, :copy_frames])
                write_pos += copy_frames

        for i in range(num_chunks):
            clip_dec = clips[i]
            for j in range(split_count):
                f0 = j * chunk_dec
                f1 = min(f0 + chunk_dec, clip_dec.shape[2])
                chunk = clip_dec[:, :, f0:f1, :, :][:, :, self.frame_pre_padding :, :, :]
                if j == 0:
                    if dec_overlap is not None:
                        chunk = self.blend(dec_overlap, chunk, self.frame_overlap, dim=-3)
                        dec_overlap = None
                    write_part(chunk)
                else:
                    dec_overlap = chunk.contiguous()
            if i == num_chunks - 1 and dec_overlap is not None:
                write_part(dec_overlap)
                dec_overlap = None
        return dec

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        wdtype = self.post_quant_conv.weight.dtype
        z = z.to(device=self.post_quant_conv.weight.device, dtype=wdtype)
        latents_mean = self.latents_mean.view(1, -1, 1, 1, 1).to(device=z.device, dtype=wdtype)
        latents_std = self.latents_std.view(1, -1, 1, 1, 1).to(device=z.device, dtype=wdtype)
        z = z * latents_std + latents_mean
        if z.shape[2] == 1:
            dec = self._adaptive_decode(z)[:, :, -1:, :, :]
        else:
            dec = self.decode_temporal_parallel(z)
        dec = dec.float()
        dec.mul_(self.pixel_std.to(dec)).add_(self.pixel_mean.to(dec)).clamp_(0.0, 1.0).mul_(2.0).sub_(1.0)
        return dec


def load_video_vae(
    path: str | Path,
    *,
    device: torch.device,
    dtype: torch.dtype = torch.float16,
    include_encoder: bool = False,
) -> MiniMaxH3VideoVAE:
    from safetensors import safe_open

    model = MiniMaxH3VideoVAE(include_encoder=include_encoder)
    sd: dict[str, torch.Tensor] = {}
    with safe_open(str(path), framework="pt") as f:
        for key in f.keys():
            if not include_encoder and (key.startswith("encoder.") or key.startswith("quant_conv.")):
                continue
            sd[key] = f.get_tensor(key).contiguous().clone()
    missing, unexpected = model.load_state_dict(sd, strict=False)
    del sd
    miss = [m for m in missing if include_encoder or (not m.startswith("encoder") and not m.startswith("quant_conv"))]
    if unexpected[:8]:
        print(f"[h3_npu] vae unexpected (first): {unexpected[:8]}", flush=True)
    if miss[:8]:
        print(f"[h3_npu] vae missing (first): {miss[:8]}", flush=True)
    model.to(device=device, dtype=dtype)
    model.eval()
    return model
