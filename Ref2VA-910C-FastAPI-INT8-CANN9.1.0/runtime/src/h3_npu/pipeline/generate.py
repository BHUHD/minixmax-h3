"""8-die generate: INT8 ConvRot DiT + FP16 VAE → mp4 (no ComfyUI).

Ref2VA / FL2VA 使用不同 DiT 权重（``H3_PARTITION``）；TE/VAE 共用。
Ref2VA: ``[text | ref | audio | video]``（无参考时退化为 ``[text | audio | video]``）。
"""
from __future__ import annotations

import argparse
import json
import math
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Optional

import torch

_SRC = Path(__file__).resolve().parents[2]
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from h3_npu.load.safetensors_quant import load_pruned_dit_streaming
from h3_npu.pipeline.partition import (
    PARTITION_REF2VA,
    get_serve_partition,
    resolve_dit_path,
    resolve_generation_mode,
)
from h3_npu.model.audio_vae import load_audio_vae
from h3_npu.model.qwen3_te import format_ref2va_prompt_multimodal
from h3_npu.model.video_vae import load_video_vae
from h3_npu.runtime.device import empty_cache, sync
from h3_npu.runtime.dist import barrier, broadcast_tensor, init_hccl
from h3_npu.runtime.metrics import MetricsLog


DEFAULT_MODELS = Path(os.environ.get("H3_MODELS", "/models/h3_quant"))
DEFAULT_REF = Path(os.environ.get("H3_REF_IMAGE", "/workspace/assets/ref_golden_retriever.png"))
VAE_SPATIAL = 16
FPS = 24.0
AUDIO_LATENT_FPS = 40.0
AUDIO_SAMPLE_RATE = 32000
FLOW_SHIFT = 12.0
CANVAS_MULTIPLE = 32
DEFAULT_PROMPT = (
    "Use <Picture 1> as the character. A golden retriever running along a sunny "
    "beach, waves in the background, cinematic lighting, 24fps"
)


def _match_ref_hw(src_w: int, src_h: int, gen_w: int, gen_h: int) -> tuple[int, int]:
    """Comfy ``ref_image_size=match``: downscale only to the generation pixel area."""
    scale = min(1.0, math.sqrt((gen_w * gen_h) / max(src_w * src_h, 1)))
    tw = max(CANVAS_MULTIPLE, round(src_w * scale / CANVAS_MULTIPLE) * CANVAS_MULTIPLE)
    th = max(CANVAS_MULTIPLE, round(src_h * scale / CANVAS_MULTIPLE) * CANVAS_MULTIPLE)
    return tw, th


def _cover_crop_hw(src_w: int, src_h: int, gen_w: int, gen_h: int) -> tuple[int, int, int, int]:
    """Scale to cover the canvas, then center-crop. Aligns ref RoPE with video tokens."""
    scale = max(gen_w / max(src_w, 1), gen_h / max(src_h, 1))
    rw = max(gen_w, int(round(src_w * scale)))
    rh = max(gen_h, int(round(src_h * scale)))
    return rw, rh, gen_w, gen_h


def _load_ref_n11(path: Path, tw: int, th: int, crop_w: int | None = None, crop_h: int | None = None) -> torch.Tensor:
    """PIL RGB → [1, 3, 1, H, W] in [-1, 1]. Optional center crop after resize."""
    from PIL import Image

    im = Image.open(path).convert("RGB").resize((tw, th), Image.Resampling.LANCZOS)
    if crop_w is not None and crop_h is not None and (crop_w != tw or crop_h != th):
        left = max(0, (tw - crop_w) // 2)
        top = max(0, (th - crop_h) // 2)
        im = im.crop((left, top, left + crop_w, top + crop_h))
    arr = __import__("numpy").asarray(im).astype("float32") / 255.0
    x = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0).unsqueeze(2)
    return x.mul(2.0).sub(1.0)


def _save_n11_preview(path: Path, video: torch.Tensor) -> None:
    x = video[0].detach().float().clamp(-1, 1)
    x = ((x + 1.0) * 127.5).round().to(torch.uint8).permute(1, 2, 3, 0).cpu().numpy()
    from PIL import Image

    Image.fromarray(x[0]).save(path)


def _align_ref_frame_count(n: int) -> int:
    """参考视频帧数对齐到 17k+5 网格（与 Comfy 一致）。"""
    n = max(5, int(n))
    while n % 17 != 5:
        n -= 1
    return max(5, n)


def _adapt_ref_video_canvas(vw: int, vh: int, gen_w: int, gen_h: int) -> tuple[int, int]:
    """参考视频画布：768 短边 + 面积上限，与 Comfy ``adapt_canvas`` 对齐。"""
    ratio = vw / max(vh, 1)
    base = 768
    max_pixels = 768 * 1344
    if ratio >= 1.0:
        nom_w, nom_h = base * ratio, float(base)
    else:
        nom_w, nom_h = float(base), base / ratio
    if nom_w * nom_h > max_pixels:
        s = math.sqrt(max_pixels / (nom_w * nom_h))
        nom_w, nom_h = nom_w * s, nom_h * s
    cw = max(CANVAS_MULTIPLE, round(nom_w / CANVAS_MULTIPLE) * CANVAS_MULTIPLE)
    ch = max(CANVAS_MULTIPLE, round(nom_h / CANVAS_MULTIPLE) * CANVAS_MULTIPLE)
    if vw * vh < cw * ch:
        cw = max(CANVAS_MULTIPLE, round(vw / CANVAS_MULTIPLE) * CANVAS_MULTIPLE)
        ch = max(CANVAS_MULTIPLE, round(vh / CANVAS_MULTIPLE) * CANVAS_MULTIPLE)
    return cw, ch


def _ffprobe_video_size(path: Path) -> tuple[int, int]:
    proc = subprocess.run(
        [
            "ffprobe",
            "-v",
            "error",
            "-select_streams",
            "v:0",
            "-show_entries",
            "stream=width,height",
            "-of",
            "json",
            str(path),
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    if proc.returncode != 0:
        raise RuntimeError(f"ffprobe failed for {path}: {proc.stderr[-200:]}")
    data = json.loads(proc.stdout)
    streams = data.get("streams") or []
    if not streams:
        raise RuntimeError(f"no video stream in {path}")
    return int(streams[0]["width"]), int(streams[0]["height"])


def _load_ref_video_n11(path: Path, gen_w: int, gen_h: int) -> torch.Tensor:
    """mp4/视频 → [1,3,T,H,W] in [-1,1] @ 24fps。"""
    vw, vh = _ffprobe_video_size(path)
    tw, th = _adapt_ref_video_canvas(vw, vh, gen_w, gen_h)
    cmd = [
        "ffmpeg",
        "-y",
        "-i",
        str(path),
        "-vf",
        f"scale={tw}:{th},fps=24",
        "-f",
        "rawvideo",
        "-pix_fmt",
        "rgb24",
        "pipe:1",
    ]
    proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False)
    if proc.returncode != 0:
        raise RuntimeError(f"ffmpeg decode {path}: {proc.stderr[-300:].decode('utf-8', 'ignore')}")
    raw = proc.stdout
    frame_bytes = tw * th * 3
    if frame_bytes <= 0 or len(raw) < frame_bytes:
        raise RuntimeError(f"empty video decode {path}")
    n_frames = len(raw) // frame_bytes
    n_frames = _align_ref_frame_count(n_frames)
    arr = __import__("numpy").frombuffer(raw[: n_frames * frame_bytes], dtype="uint8").reshape(
        n_frames, th, tw, 3
    )
    x = torch.from_numpy(arr.copy()).permute(0, 3, 1, 2).unsqueeze(0).float().div_(127.5).sub_(1.0)
    return x.permute(0, 2, 1, 3, 4).contiguous()


def _load_audio_stereo_32k(path: Path) -> torch.Tensor:
    """任意音频 → [1,2,L] float32 @ 32kHz（平面声道，对齐 Comfy waveform）。

    ffmpeg ``f32le`` 默认是 **交错** LRLR…；必须先 ``reshape(L,2).T``，
    不能 ``reshape(1,2,L)``（会把左右声道搅进同一维，参考音频条件变噪声）。
    """
    cmd = [
        "ffmpeg",
        "-y",
        "-i",
        str(path),
        "-ac",
        "2",
        "-ar",
        str(AUDIO_SAMPLE_RATE),
        "-f",
        "f32le",
        "pipe:1",
    ]
    proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False)
    if proc.returncode != 0:
        raise RuntimeError(f"ffmpeg audio {path}: {proc.stderr[-300:].decode('utf-8', 'ignore')}")
    if not proc.stdout:
        raise RuntimeError(f"empty audio decode {path}")
    interleaved = __import__("numpy").frombuffer(proc.stdout, dtype="float32")
    n = interleaved.size // 2
    # [L,R,L,R,...] → [1, 2, L] 平面（ch0=L, ch1=R）
    planar = interleaved[: n * 2].reshape(n, 2).T[None, ...].copy()
    return torch.from_numpy(planar)



def _try_extract_video_audio(path: Path) -> torch.Tensor | None:
    """尝试从视频容器提取音轨；无音轨则返回 None。"""
    probe = subprocess.run(
        [
            "ffprobe",
            "-v",
            "error",
            "-select_streams",
            "a:0",
            "-show_entries",
            "stream=codec_type",
            "-of",
            "csv=p=0",
            str(path),
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    if probe.returncode != 0 or "audio" not in (probe.stdout or ""):
        return None
    try:
        return _load_audio_stereo_32k(path)
    except Exception as exc:  # noqa: BLE001
        print(f"[generate] video audio extract skip {path.name}: {exc}", flush=True)
        return None


def _encode_audio_ref_latent(audio_vae, waveform: torch.Tensor) -> tuple[torch.Tensor, int]:
    """波形 [1,2,L] → 归一化 audio latent 与 ref_audio_t。"""
    with torch.inference_mode():
        z = audio_vae.encode(waveform.to(dtype=torch.float32))
    ref_audio_t = int(z.shape[-1])
    return z.detach().float().contiguous(), ref_audio_t


def _broadcast_ref2va_payload(device, rank, refs_meta, cond_videos, cond_audios):
    """广播 Ref2VA 参考布局 + video/audio 条件 latent。"""
    if rank == 0:
        meta_bytes = json.dumps(refs_meta).encode("utf-8")
        n_b = torch.tensor([len(meta_bytes)], device=device, dtype=torch.int32)
    else:
        n_b = torch.zeros(1, device=device, dtype=torch.int32)
    broadcast_tensor(n_b)
    nb = int(n_b.item())
    if rank == 0:
        buf = torch.tensor(list(meta_bytes), device=device, dtype=torch.uint8)
    else:
        buf = torch.empty(max(nb, 1), device=device, dtype=torch.uint8)
    if nb > 0:
        broadcast_tensor(buf)
    if rank != 0:
        refs_meta = json.loads(bytes(buf[:nb].cpu().tolist()).decode("utf-8"))

    out_v, out_a = [], []
    for target_list, source in ((out_v, cond_videos or []), (out_a, cond_audios or [])):
        if rank == 0:
            n = torch.tensor([len(source)], device=device, dtype=torch.int32)
        else:
            n = torch.zeros(1, device=device, dtype=torch.int32)
        broadcast_tensor(n)
        for _ in range(int(n.item())):
            if rank == 0:
                z = source[len(target_list)].float().contiguous()
                sh = list(z.shape)
                ndim = torch.tensor([len(sh)], device=device, dtype=torch.int32)
            else:
                ndim = torch.zeros(1, device=device, dtype=torch.int32)
            broadcast_tensor(ndim)
            nd = int(ndim.item())
            if rank == 0:
                meta = torch.tensor(sh, device=device, dtype=torch.int32)
            else:
                meta = torch.zeros(nd, device=device, dtype=torch.int32)
            broadcast_tensor(meta)
            shape = tuple(int(x) for x in meta.tolist())
            if rank != 0:
                z = torch.empty(shape, device=device, dtype=torch.float32)
            broadcast_tensor(z)
            target_list.append(z)
    return refs_meta, out_v, out_a


def _encode_ref2va_conditions(
    vae_path: Path,
    audio_vae_path: Path,
    image_paths: list[Path],
    video_paths: list[Path],
    audio_paths: list[Path],
    gen_w: int,
    gen_h: int,
    device,
    preview: Path | None,
) -> tuple[list[dict], list[torch.Tensor], list[torch.Tensor], int]:
    """按 Comfy 顺序编码 images → videos(+内嵌音轨) → standalone audios。"""
    vae = load_video_vae(vae_path, device=device, dtype=torch.float16, include_encoder=True)
    audio_vae = load_audio_vae(audio_vae_path, device=torch.device("cpu"), dtype=torch.float32)

    refs_meta: list[dict] = []
    cond_videos: list[torch.Tensor] = []
    cond_audios: list[torch.Tensor] = []
    video_with_audio = 0

    # 参考图尺寸：对齐 Comfy ``ref_image_size=match``（只缩小、保比例，不 cover 裁切）
    for i, p in enumerate(image_paths):
        from PIL import Image

        with Image.open(p) as im:
            src_w, src_h = im.size
        tw, th = _match_ref_hw(src_w, src_h, gen_w, gen_h)
        pixels = _load_ref_n11(p, tw, th).to(device=device, dtype=torch.float16)
        z = vae.encode(pixels)
        sync()
        print(
            f"[generate] ref_img{i} {p.name} {src_w}x{src_h} → {tw}x{th} (match) latent={tuple(z.shape)}",
            flush=True,
        )
        if i == 0 and preview is not None and os.environ.get("H3_SKIP_REF_ROUNDTRIP", "1") != "1":
            _save_n11_preview(preview, vae.decode(z))
        cond_videos.append(z.detach().float().contiguous())
        refs_meta.append({"kind": "image", "latent_h": int(z.shape[3]), "latent_w": int(z.shape[4])})

    for i, p in enumerate(video_paths):
        pixels = _load_ref_video_n11(p, gen_w, gen_h).to(device=device, dtype=torch.float16)
        z = vae.encode(pixels)
        sync()
        ref_audio_t = 0
        soundtrack = _try_extract_video_audio(p)
        if soundtrack is not None:
            az, ref_audio_t = _encode_audio_ref_latent(audio_vae, soundtrack)
            cond_audios.append(az.to(device=device))
            video_with_audio += 1
            print(
                f"[generate] ref_vid{i} {p.name} frames={pixels.shape[2]} latent={tuple(z.shape)} "
                f"+audio_t={ref_audio_t}",
                flush=True,
            )
        else:
            print(
                f"[generate] ref_vid{i} {p.name} frames={pixels.shape[2]} latent={tuple(z.shape)} (no audio track)",
                flush=True,
            )
        kind = "video_audio" if ref_audio_t else "video"
        cond_videos.append(z.detach().float().contiguous())
        refs_meta.append(
            {
                "kind": kind,
                "latent_t": int(z.shape[2]),
                "latent_h": int(z.shape[3]),
                "latent_w": int(z.shape[4]),
                "ref_audio_t": ref_audio_t,
            }
        )

    for i, p in enumerate(audio_paths):
        wav = _load_audio_stereo_32k(p)
        az, ref_audio_t = _encode_audio_ref_latent(audio_vae, wav)
        cond_audios.append(az.to(device=device))
        refs_meta.append({"kind": "audio", "ref_audio_t": ref_audio_t})
        print(f"[generate] ref_aud{i} {p.name} ref_audio_t={ref_audio_t}", flush=True)

    del vae, audio_vae
    empty_cache()
    return refs_meta, cond_videos, cond_audios, video_with_audio


def _broadcast_ref_payload(device, rank, refs=None, conds=None):
    """兼容旧接口：仅 image refs。"""
    refs_meta = [{"kind": "image", "latent_h": int(z.shape[3]), "latent_w": int(z.shape[4])} for z in (conds or [])]
    r, v, _a = _broadcast_ref2va_payload(device, rank, refs_meta, conds or [], [])
    return r, v


def _video_has_audio(path: Path) -> bool:
    probe = subprocess.run(
        [
            "ffprobe",
            "-v",
            "error",
            "-select_streams",
            "a:0",
            "-show_entries",
            "stream=codec_type",
            "-of",
            "csv=p=0",
            str(path),
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    return probe.returncode == 0 and "audio" in (probe.stdout or "")


def _ref_cache_token(path: str) -> str:
    """用文件身份做缓存键，避免 /workspace/jobs/<id>/... 路径导致 serve 缓存 miss。"""
    p = Path(path)
    try:
        st = p.stat()
        return f"{p.name}:{st.st_size}:{st.st_mtime_ns}"
    except OSError:
        return str(path)


def _latent_hw(height: int, width: int) -> tuple[int, int]:
    lh = max(2, height // VAE_SPATIAL)
    lw = max(2, width // VAE_SPATIAL)
    lh += lh % 2
    lw += lw % 2
    return lh, lw


def _align_frame_count(n: int) -> int:
    n = max(5, int(n))
    while n % 17 != 5:
        n += 1
    return n


def _video_latent_t(frame_count: int) -> int:
    return 2 if frame_count <= 5 else ((frame_count - 5) // 17) * 5 + 2


def _temporal_shape(seconds: float) -> tuple[int, int, int]:
    """Official MiniMax H3 17k+5 frame grid (Comfy EmptyMiniMaxH3LatentAV)."""
    frame_count = _align_frame_count(max(5, int(round(seconds * FPS))))
    duration = frame_count / FPS
    return frame_count, _video_latent_t(frame_count), int(round(duration * AUDIO_LATENT_FPS))


def _flow_sigmas(steps: int, device, shift: float = FLOW_SHIFT) -> torch.Tensor:
    t = torch.linspace(1.0, 0.0, steps + 1, device=device, dtype=torch.float32)
    return shift * t / (1.0 + (shift - 1.0) * t)


def _const_denoised(x: torch.Tensor, model_out: torch.Tensor, sigma: torch.Tensor) -> torch.Tensor:
    return x - model_out.to(dtype=x.dtype) * sigma.to(dtype=x.dtype)


def _res_multistep_update(
    x: torch.Tensor,
    denoised: torch.Tensor,
    sigma: torch.Tensor,
    sigma_next: torch.Tensor,
    old_denoised: Optional[torch.Tensor],
    old_sigma_down: Optional[torch.Tensor],
    sigma_prev: Optional[torch.Tensor],
) -> torch.Tensor:
    """Comfy ``sample_res_multistep`` with eta=0 (BasicGuider / CONST flow)."""
    if float(sigma_next) == 0.0 or old_denoised is None or old_sigma_down is None or sigma_prev is None:
        d = (x - denoised) / sigma.clamp(min=1e-8).to(dtype=x.dtype)
        return x + d * (sigma_next - sigma).to(dtype=x.dtype)
    t = torch.log(sigma.clamp(min=1e-8)).neg()
    t_old = torch.log(old_sigma_down.clamp(min=1e-8)).neg()
    t_next = torch.log(sigma_next.clamp(min=1e-8)).neg()
    t_prev = torch.log(sigma_prev.clamp(min=1e-8)).neg()
    h = t_next - t
    c2 = (t_prev - t_old) / h
    phi1 = torch.expm1(-h) / (-h)
    phi2 = (phi1 - 1.0) / (-h)
    b1 = torch.nan_to_num(phi1 - phi2 / c2, nan=0.0)
    b2 = torch.nan_to_num(phi2 / c2, nan=0.0)
    return torch.exp(-h).to(dtype=x.dtype) * x + (h * (b1 * denoised + b2 * old_denoised)).to(dtype=x.dtype)


def assert_npu_resident(module: torch.nn.Module, *, allow_cpu: bool = False) -> None:
    cpu = []
    for name, t in list(module.named_parameters()) + list(module.named_buffers()):
        if t is None or t.numel() == 0:
            continue
        if t.device.type != "npu":
            cpu.append((name, str(t.device), tuple(t.shape)))
    if cpu and not allow_cpu:
        preview = cpu[:8]
        raise RuntimeError(f"weights not resident on NPU HBM (first): {preview}")


def _normalize_waveform(waveform: torch.Tensor) -> torch.Tensor:
    """对齐 Comfy ``VAEDecodeAudio`` 的响度归一化：按 batch 通道标准差缩放。"""
    # waveform [B, C, L]
    std = torch.std(waveform, dim=[1, 2], keepdim=True) * 5.0
    std = torch.clamp(std, min=1.0)
    return waveform / std


def write_mp4(
    path: Path,
    video: torch.Tensor,
    fps: float = FPS,
    *,
    audio: torch.Tensor | None = None,
    sample_rate: int = AUDIO_SAMPLE_RATE,
) -> None:
    """video [1,3,T,H,W] in [-1,1] → mp4；可选 stereo audio [B,2,L] 或 [2,L] 混入。

    Rank-0 only. Accepts CPU or NPU tensors; streams frames to ffmpeg stdin.
    """
    import tempfile
    import wave

    path.parent.mkdir(parents=True, exist_ok=True)
    x = video[0].detach()
    if x.device.type != "cpu":
        x = x.float().cpu()
    else:
        x = x.float()
    x = x.clamp(-1, 1)
    frames = ((x + 1.0) * 127.5).round().to(torch.uint8).permute(1, 2, 3, 0).contiguous()
    t, h, w, _c = frames.shape

    wav_path: Path | None = None
    if audio is not None and os.environ.get("H3_SKIP_AUDIO", "0") != "1":
        aw = audio.detach()
        if aw.device.type != "cpu":
            aw = aw.float().cpu()
        else:
            aw = aw.float()
        if aw.ndim == 2:
            aw = aw.unsqueeze(0)
        if aw.ndim != 3 or aw.shape[1] < 1:
            raise ValueError(f"unexpected audio shape {tuple(aw.shape)}")
        # 单声道则复制为立体声；多声道截到 2
        if aw.shape[1] == 1:
            aw = aw.repeat(1, 2, 1)
        aw = aw[:, :2].contiguous()
        aw = _normalize_waveform(aw).clamp(-1, 1)
        # 对齐视频时长（样本数）
        target_n = max(1, int(round(t / float(fps) * sample_rate)))
        cur_n = int(aw.shape[-1])
        if cur_n < target_n:
            aw = torch.nn.functional.pad(aw, (0, target_n - cur_n))
        elif cur_n > target_n:
            aw = aw[..., :target_n]
        pcm = (aw[0].transpose(0, 1).contiguous() * 32767.0).round().to(torch.int16).numpy()
        fd, wav_name = tempfile.mkstemp(suffix=".wav", prefix="h3_audio_")
        os.close(fd)
        wav_path = Path(wav_name)
        with wave.open(str(wav_path), "wb") as wf:
            wf.setnchannels(2)
            wf.setsampwidth(2)
            wf.setframerate(int(sample_rate))
            wf.writeframes(pcm.tobytes())
        print(
            f"[generate] audio wav {wav_path.name} sr={sample_rate} samples={target_n} "
            f"dur={target_n / sample_rate:.3f}s",
            flush=True,
        )

    # 有音频：先写无声视频再 mux；无音频：直接 -an
    video_only = path if wav_path is None else path.with_suffix(".video_only.mp4")
    cmd = [
        "ffmpeg",
        "-y",
        "-f",
        "rawvideo",
        "-vcodec",
        "rawvideo",
        "-s",
        f"{w}x{h}",
        "-pix_fmt",
        "rgb24",
        "-r",
        str(fps),
        "-i",
        "-",
        "-an",
        "-c:v",
        "libx264",
        "-preset",
        "ultrafast",
        "-crf",
        "23",
        "-pix_fmt",
        "yuv420p",
        "-movflags",
        "+faststart",
        str(video_only),
    ]
    try:
        proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
        assert proc.stdin is not None
        preview = None
        for i in range(t):
            plane = frames[i].numpy()
            if i == 0:
                preview = plane
            proc.stdin.write(plane.tobytes())
        proc.stdin.close()
        err = proc.stderr.read() if proc.stderr else b""
        rc = proc.wait()
        if rc != 0:
            raise RuntimeError(err[-400:].decode("utf-8", "ignore"))
        if wav_path is not None:
            mux = [
                "ffmpeg",
                "-y",
                "-i",
                str(video_only),
                "-i",
                str(wav_path),
                "-c:v",
                "copy",
                "-c:a",
                "aac",
                "-b:a",
                "128k",  # 对齐 Demo Comfy SaveVideo 默认 AAC 码率
                "-shortest",
                "-movflags",
                "+faststart",
                str(path),
            ]
            m = subprocess.run(mux, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, check=False)
            if m.returncode != 0:
                raise RuntimeError(m.stderr[-400:].decode("utf-8", "ignore"))
            try:
                video_only.unlink(missing_ok=True)
            except Exception:
                pass
            try:
                wav_path.unlink(missing_ok=True)
            except Exception:
                pass
            print(
                f"[generate] wrote {path} ({t} frames {w}x{h} @ {fps}fps + aac {sample_rate}Hz)",
                flush=True,
            )
        else:
            print(f"[generate] wrote {path} ({t} frames {w}x{h} @ {fps}fps, silent)", flush=True)
        if preview is not None and os.environ.get("H3_SKIP_PREVIEW", "0") != "1":
            try:
                from PIL import Image

                Image.fromarray(preview).save(path.with_suffix(".png"))
                print(f"[generate] preview {path.with_suffix('.png')}", flush=True)
            except Exception as exc:  # noqa: BLE001
                print(f"[generate] png preview skipped: {exc}", flush=True)
        return
    except (FileNotFoundError, RuntimeError, OSError) as exc:
        if wav_path is not None:
            try:
                wav_path.unlink(missing_ok=True)
            except Exception:
                pass
        print(f"[generate] ffmpeg failed ({exc}); writing PNG sequence", flush=True)
    try:
        from PIL import Image
    except ImportError as exc:
        npy = path.with_suffix(".npy")
        __import__("numpy").save(npy, frames.numpy())
        raise RuntimeError(f"cannot write video (no ffmpeg/PIL); saved {npy}") from exc
    seq = path.with_suffix("")
    seq.mkdir(parents=True, exist_ok=True)
    for i in range(t):
        Image.fromarray(frames[i].numpy()).save(seq / f"frame_{i:04d}.png")
    print(f"[generate] wrote {seq}/frame_*.png", flush=True)


def _align_canvas(h: int, w: int) -> tuple[int, int]:
    h = max(CANVAS_MULTIPLE, int(round(h / CANVAS_MULTIPLE)) * CANVAS_MULTIPLE)
    w = max(CANVAS_MULTIPLE, int(round(w / CANVAS_MULTIPLE)) * CANVAS_MULTIPLE)
    return h, w


def _parse_progressive(steps: int, final_h: int, final_w: int) -> list[tuple[int, int, int]]:
    """Return [(height, width, n_steps), ...] covering exactly ``steps``.

    Env ``H3_PROGRESSIVE`` examples:
      - empty / ``0``: single stage at final resolution
      - ``960x544:12,1920x1088:8`` (WxH:steps)
      - ``auto``: half-res structure + full-res refine for 20-step 1080P
    """
    raw = (os.environ.get("H3_PROGRESSIVE") or "").strip()
    if not raw or raw in ("0", "off", "none"):
        return [(final_h, final_w, steps)]
    if raw.lower() == "auto":
        if steps >= 20 and final_w >= 1920 and final_h >= 1080:
            # Native 1080P FA is ~0.30s/layer (15s/step attention alone) — 20
            # full-res steps cannot fit in 5 minutes.  Run the DiT at ~2/3
            # spatial size (FA∝S²) then pixel-upsample VAE output to 1080P.
            # Latent-space progressive refine is available via an explicit
            # H3_PROGRESSIVE=WxH:n,... schedule plus H3_UPSAMPLE_NOISE.
            mid_h, mid_w = _align_canvas((final_h * 2) // 3, (final_w * 2) // 3)
            return [(mid_h, mid_w, steps)]
        return [(final_h, final_w, steps)]
    stages: list[tuple[int, int, int]] = []
    for part in raw.split(","):
        part = part.strip()
        if not part:
            continue
        res, n_s = part.rsplit(":", 1)
        w_s, h_s = res.lower().split("x")
        h, w = _align_canvas(int(h_s), int(w_s))
        stages.append((h, w, int(n_s)))
    total = sum(n for _h, _w, n in stages)
    if total != steps:
        raise ValueError(f"H3_PROGRESSIVE steps sum={total} != --steps {steps}")
    if stages[-1][0] != final_h or stages[-1][1] != final_w:
        print(
            f"[generate] WARN progressive final stage {stages[-1][1]}x{stages[-1][0]} "
            f"!= canvas {final_w}x{final_h}",
            flush=True,
        )
    return stages


def _upsample_video_latent(z: torch.Tensor, nh: int, nw: int) -> torch.Tensor:
    """Spatial upsample of DiT video latent [B,C,T,H,W]."""
    if z.shape[-2] == nh and z.shape[-1] == nw:
        return z
    return torch.nn.functional.interpolate(
        z.float(), size=(z.shape[2], nh, nw), mode="trilinear", align_corners=False
    ).to(dtype=z.dtype)


def _upsample_latent_for_stage(
    z: torch.Tensor,
    nh: int,
    nw: int,
    *,
    sigma: torch.Tensor,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    """Upsample then reinject HF noise so full-res refine sees the right σ-level.

    Pure trilinear leaves missing high-frequency modes as smooth zeros; the DiT
    then blows up. Scale fresh noise by the current flow sigma (CONST).
    """
    up = _upsample_video_latent(z, nh, nw)
    strength = float(os.environ.get("H3_UPSAMPLE_NOISE", "1.0"))
    if strength <= 0.0:
        return up
    noise = torch.randn(
        up.shape, dtype=torch.float32, device="cpu", generator=generator
    ).to(device=up.device, dtype=up.dtype)
    return up + noise * sigma.to(device=up.device, dtype=up.dtype) * strength


def _parse_args(argv=None):
    p = argparse.ArgumentParser(description="H3 quant 8-NPU generate (torch_npu, no ComfyUI)")
    p.add_argument("--models", type=Path, default=DEFAULT_MODELS)
    p.add_argument("--out", type=Path, default=Path("/workspace/out/h3_quant_generate.mp4"))
    p.add_argument("--seconds", type=float, default=10.0)
    p.add_argument("--height", type=int, default=1088)
    p.add_argument("--width", type=int, default=1920)
    p.add_argument("--steps", type=int, default=20)
    p.add_argument("--text-len", type=int, default=32)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--prompt", type=str, default=os.environ.get("H3_PROMPT", DEFAULT_PROMPT))
    p.add_argument(
        "--ref-image",
        action="append",
        default=None,
        help="Reference image for Ref2VA (repeatable). Default: assets/ref_golden_retriever.png",
    )
    p.add_argument(
        "--ref-video",
        action="append",
        default=None,
        help="Reference video for Ref2VA (repeatable, mp4).",
    )
    p.add_argument(
        "--ref-audio",
        action="append",
        default=None,
        help="Standalone reference audio for Ref2VA (repeatable, wav/mp3).",
    )
    p.add_argument("--shift", type=float, default=FLOW_SHIFT)
    p.add_argument(
        "--sampler",
        type=str,
        default=os.environ.get("H3_SAMPLER", "res_multistep"),
        choices=("euler", "res_multistep"),
    )
    return p.parse_args(argv)


def load_dit_to_npu(argv=None):
    """Load INT8 DiT onto logical npu:0 (one phy die already pinned per rank)."""
    import gc

    os.environ.setdefault("PYTORCH_NPU_ALLOC_CONF", "expandable_segments:False")
    import torch_npu  # noqa: F401

    args = _parse_args(argv)
    if args.ref_image is None:
        args.ref_image = []
    if args.ref_video is None:
        args.ref_video = []
    if args.ref_audio is None:
        args.ref_audio = []
    args.height = max(CANVAS_MULTIPLE, int(round(args.height / CANVAS_MULTIPLE)) * CANVAS_MULTIPLE)
    args.width = max(CANVAS_MULTIPLE, int(round(args.width / CANVAS_MULTIPLE)) * CANVAS_MULTIPLE)
    os.environ["H3_NPU_LAYERWISE_OFFLOAD"] = "0"
    rank = int(os.environ.get("RANK", "0"))
    torch.npu.set_device(0)
    device = torch.device("npu:0")
    dit_path = resolve_dit_path(args.models, get_serve_partition())
    if rank == 0 and not dit_path.is_file():
        raise FileNotFoundError(
            f"DiT checkpoint missing for partition={get_serve_partition()}: {dit_path}"
        )
    t_load = time.time()
    print(
        f"[generate] rank {rank} loading DiT ({get_serve_partition()}) "
        f"{dit_path.name} → {device} vis={os.environ.get('ASCEND_RT_VISIBLE_DEVICES')}",
        flush=True,
    )
    model = load_pruned_dit_streaming(dit_path, device=device, dtype=torch.bfloat16)
    from h3_npu.runtime.accel import maybe_torchair_compile

    if os.environ.get("H3_NPU_TORCHAIR", "0") == "1":
        attn_only = os.environ.get("H3_TORCHAIR_ATTN_ONLY", "0") == "1"
        mlp_only = os.environ.get("H3_TORCHAIR_MLP_ONLY", "0") == "1"
        for i, block in enumerate(model.blocks):
            if mlp_only:
                block.mlp = maybe_torchair_compile(block.mlp, dynamic=False)
            elif attn_only:
                block.attn = maybe_torchair_compile(block.attn, dynamic=False)
            else:
                model.blocks[i] = maybe_torchair_compile(block, dynamic=False)
        if rank == 0:
            if mlp_only:
                mode = "mlp-only"
            elif attn_only:
                mode = "attn-only"
            else:
                mode = "full-block"
            print(f"[generate] TorchAir compile ({mode}) on {len(model.blocks)} DiT blocks", flush=True)
    gc.collect()
    assert_npu_resident(model, allow_cpu=False)
    print(f"[generate] rank {rank} DiT on HBM in {time.time() - t_load:.1f}s", flush=True)
    args._load_s = time.time() - t_load
    return model, device, args


@torch.inference_mode()
def run_generate(model, device, args) -> None:
    t_job = float(os.environ.get("H3_JOB_T0") or time.time())
    dist = init_hccl()
    torch.manual_seed(args.seed + dist.rank)
    metrics = MetricsLog()
    metrics.add(
        "start",
        device=device,
        extra={"world_size": dist.world_size, "backend": dist.backend, "load_s": round(getattr(args, "_load_s", 0), 2)},
    )
    if dist.rank == 0:
        print(
            f"[generate] world={dist.world_size} device={device} "
            f"{args.width}x{args.height} {args.seconds}s steps={args.steps} "
            f"shift={args.shift} sampler={args.sampler}",
            flush=True,
        )
    metrics.add("dit_loaded", device=device)
    barrier()

    ref_image_paths: list[Path] = [Path(raw) for raw in (args.ref_image or [])]
    ref_video_paths: list[Path] = [Path(raw) for raw in (args.ref_video or [])]
    ref_audio_paths: list[Path] = [Path(raw) for raw in (args.ref_audio or [])]
    if not ref_video_paths and os.environ.get("H3_REF_VIDEOS"):
        ref_video_paths = [Path(p) for p in json.loads(os.environ["H3_REF_VIDEOS"])]
    if not ref_audio_paths and os.environ.get("H3_REF_AUDIOS"):
        ref_audio_paths = [Path(p) for p in json.loads(os.environ["H3_REF_AUDIOS"])]

    partition = get_serve_partition()
    task_raw = os.environ.get("H3_TASK")

    # 先按实际上传参考解析模式，避免 API 空列表被默认金毛图误判为 ref2va
    try:
        mode = resolve_generation_mode(
            partition,
            n_images=len(ref_image_paths),
            n_videos=len(ref_video_paths),
            n_audios=len(ref_audio_paths),
            task=task_raw,
        )
    except ValueError as exc:
        raise RuntimeError(str(exc)) from exc
    is_t2va = mode == "t2va"

    # CLI / ref2va 预热：仅非 t2va 且无参考时回退默认图
    if not is_t2va:
        allow_empty_cli = os.environ.get("H3_ALLOW_EMPTY_REFS", "0") == "1"
        if (
            not ref_image_paths
            and not ref_video_paths
            and not ref_audio_paths
            and not allow_empty_cli
            and partition == PARTITION_REF2VA
        ):
            env_ref = os.environ.get("H3_REF_IMAGE")
            candidate = Path(env_ref) if env_ref else DEFAULT_REF
            if candidate.is_file():
                ref_image_paths = [candidate]

    if is_t2va:
        ref_image_paths = []
        ref_video_paths = []
        ref_audio_paths = []

    vae_path = args.models / "vae" / "minimax_h3_video_vae_fp16.safetensors"
    audio_vae_path = args.models / "vae" / "minimax_h3_audio_vae_fp32.safetensors"
    video_has_audio = [_video_has_audio(p) for p in ref_video_paths]
    video_with_audio = sum(1 for x in video_has_audio if x)
    prompt = (args.prompt or "").strip()
    if not prompt:
        if is_t2va:
            raise RuntimeError("t2va 需要非空 prompt")
        prompt = DEFAULT_PROMPT
    # t2va：Comfy ReferenceToVideo 零参考 — 纯用户正文，不注入 Picture/Video/Audio 标签
    te_prompt = prompt if is_t2va else format_ref2va_prompt_multimodal(
        prompt,
        n_pictures=len(ref_image_paths),
        video_has_audio=video_has_audio,
        n_standalone_audios=len(ref_audio_paths),
    )
    if dist.rank == 0:
        print(
            f"[generate] partition={partition} mode={mode} "
            f"ref inputs images={len(ref_image_paths)} videos={len(ref_video_paths)} "
            f"audios={len(ref_audio_paths)} video_with_audio={video_with_audio}",
            flush=True,
        )
        head = te_prompt[:240].replace("\n", "\\n")
        print(f"[generate] presentation+prompt head: {head}...", flush=True)
    from h3_npu.model.qwen3_te import encode_text_npu

    te_path = args.models / "text_encoders" / "qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors"
    # Prewarm uses steps=0 but must still cache refs for every progressive stage.
    plan_steps = args.steps if args.steps > 0 else int(os.environ.get("H3_SERVE_PLAN_STEPS", "20"))
    cache_stages = _parse_progressive(plan_steps, args.height, args.width)
    run_stages = cache_stages if args.steps <= 0 else _parse_progressive(args.steps, args.height, args.width)
    res_keys: list[tuple[int, int]] = []
    for h, w, _n in cache_stages:
        if (h, w) not in res_keys:
            res_keys.append((h, w))
    cond_key = (
        te_prompt,
        tuple(_ref_cache_token(str(p)) for p in ref_image_paths),
        tuple(_ref_cache_token(str(p)) for p in ref_video_paths),
        tuple(_ref_cache_token(str(p)) for p in ref_audio_paths),
        tuple(res_keys),
    )
    cached = getattr(args, "_cond_cache", None)
    ref_by_res: dict[tuple[int, int], tuple[list, list, list]] = {}
    # t2va prompt 每单不同，禁止复用 resident TE 缓存（预热短 prompt 会污染长 prompt 任务）
    if is_t2va:
        cached = None
    if cached is not None and cached[0] == cond_key:
        context, tags, ref_by_res = cached[1]
        ntok = int(context.shape[1])
        if dist.rank == 0:
            print("[generate] reuse cached TE + multi-res ref latents (resident serve)", flush=True)
        metrics.add("te_encoded", device=device, extra={"text_len": ntok, "cached": 1})
        metrics.add(
            "refs_encoded",
            device=device,
            extra={"n_refs": len(next(iter(ref_by_res.values()))[0]) if ref_by_res else 0, "cached": 1},
        )
    else:
        unique_res = []
        for h, w, _n in cache_stages:
            if (h, w) not in unique_res:
                unique_res.append((h, w))

        sequential = os.environ.get("H3_SERVE", "0") == "1" or os.environ.get("H3_TE_REF_SEQUENTIAL", "0") == "1"
        enc_thread = None
        enc_box: dict = {}
        enc_err: list = []

        def _enc_job():
            try:
                out = {}
                for hi, wi in unique_res:
                    if is_t2va:
                        out[(hi, wi)] = ([], [], [])
                    else:
                        meta, cv, ca, _vwa = _encode_ref2va_conditions(
                            vae_path,
                            audio_vae_path,
                            ref_image_paths,
                            ref_video_paths,
                            ref_audio_paths,
                            wi,
                            hi,
                            device,
                            Path(args.out).with_name("h3_ref_roundtrip.png") if (hi, wi) == unique_res[0] else None,
                        )
                        out[(hi, wi)] = (meta, cv, ca)
                enc_box["refs"] = out
            except Exception as exc:  # noqa: BLE001
                enc_err.append(exc)

        if (
            not is_t2va
            and not sequential
            and dist.rank == 0
            and dist.world_size > 1
            and os.environ.get("H3_TE_RANK0_ENCODE", "1") == "1"
        ):
            import threading

            enc_thread = threading.Thread(target=_enc_job, daemon=True)
            enc_thread.start()

        context, tags = encode_text_npu(te_path, te_prompt, device=device, dtype=torch.bfloat16)
        tags = tags.to(dtype=torch.int32)
        ntok = int(context.shape[1])
        empty_cache()
        metrics.add("te_encoded", device=device, extra={"text_len": ntok})

        if enc_thread is not None:
            enc_thread.join()
            if enc_err:
                raise enc_err[0]
            ref_by_res = enc_box["refs"]
        else:
            ref_by_res = {}
            if dist.rank == 0:
                t_ref = time.time()
                for hi, wi in unique_res:
                    if is_t2va:
                        ref_by_res[(hi, wi)] = ([], [], [])
                    else:
                        meta, cv, ca, _vwa = _encode_ref2va_conditions(
                            vae_path,
                            audio_vae_path,
                            ref_image_paths,
                            ref_video_paths,
                            ref_audio_paths,
                            wi,
                            hi,
                            device,
                            Path(args.out).with_name("h3_ref_roundtrip.png") if (hi, wi) == unique_res[0] else None,
                        )
                        ref_by_res[(hi, wi)] = (meta, cv, ca)
                if is_t2va:
                    print(
                        f"[generate] t2va: skip ref encode for {len(unique_res)} resolution(s)",
                        flush=True,
                    )
                else:
                    print(
                        f"[generate] encoded refs for {len(unique_res)} resolution(s) in {time.time() - t_ref:.1f}s"
                        f"{' (sequential after TE)' if sequential else ''}",
                        flush=True,
                    )
                empty_cache()
        synced: dict[tuple[int, int], tuple[list, list, list]] = {}
        for hi, wi in unique_res:
            r0, cv0, ca0 = (None, None, None) if dist.rank != 0 else ref_by_res[(hi, wi)]
            r0, cv0, ca0 = _broadcast_ref2va_payload(device, dist.rank, r0, cv0, ca0)
            synced[(hi, wi)] = (r0, cv0, ca0)
        ref_by_res = synced
        metrics.add("refs_encoded", device=device, extra={"n_refs": len(next(iter(ref_by_res.values()))[0])})
        if not is_t2va:
            args._cond_cache = (cond_key, (context, tags, ref_by_res))
        elif dist.rank == 0:
            print("[generate] t2va: skip resident TE cache (prompt-specific encode)", flush=True)
    barrier()
    if args.steps <= 0:
        if dist.rank == 0:
            print("[generate] prewarm TE+ref only (no DiT)", flush=True)
        return

    if dist.rank == 0:
        desc = ", ".join(f"{w}x{h}:{n}" for h, w, n in run_stages)
        print(f"[generate] stages [{desc}] (FA∝S²)", flush=True)

    frame_count, lt, audio_t = _temporal_shape(args.seconds)
    # 对齐 Comfy ``prepare_noise``：``generator = torch.manual_seed(seed)`` 后
    # 对 NestedTensor 按 video→audio 顺序各抽一次 CPU float32 randn。
    g = torch.manual_seed(args.seed)
    h0, w0, _ = run_stages[0]
    lh, lw = _latent_hw(h0, w0)
    video = torch.randn(
        1, 24, lt, lh, lw, dtype=torch.float32, layout=torch.strided, generator=g, device="cpu"
    ).to(device=device, dtype=torch.bfloat16)
    audio = torch.randn(
        1, 32, 2, audio_t, dtype=torch.float32, layout=torch.strided, generator=g, device="cpu"
    ).to(device=device, dtype=torch.bfloat16)
    if dist.rank == 0:
        print(
            f"[generate] frames={frame_count} start_latent=24x{lt}x{lh}x{lw} audio_t={audio_t} "
            f"text_len={ntok}",
            flush=True,
        )

    sigmas = _flow_sigmas(args.steps, device, shift=args.shift)
    step_times: list[float] = []
    skip_warmup = os.environ.get("H3_SKIP_WARMUP", "1") == "1"
    if not skip_warmup:
        refs_meta, cond_v, cond_a = ref_by_res[(h0, w0)]
        payload = {
            "text_token_tags": tags.long(),
            "refs": refs_meta,
            "cond_video_latents": cond_v,
            "cond_audio_latents": cond_a,
            "seed": args.seed,
        }
        ts = (sigmas[0] * 1000.0).view(1)
        _ = model([video, audio], ts, context, minimax_payload=payload)
        sync()
        empty_cache()
        metrics.add("warmup_done", device=device)
    elif dist.rank == 0:
        print("[generate] skip warmup forward (first step is the cold kernel)", flush=True)

    t_ready = time.time()
    if dist.rank == 0:
        print(
            f"[generate] request-ready te+ref done in {t_ready - t_job:.1f}s "
            f"(dit+vae is the remaining task response)",
            flush=True,
        )
        try:
            (Path(args.out).parent / "progress.json").write_text(
                json.dumps(
                    {
                        "phase": "encode",
                        "step": 0,
                        "steps_total": int(args.steps),
                        "percent": 5,
                    }
                )
            )
        except Exception:
            pass

    old_dv = old_da = old_sigma_down = None
    # Always sync for wall-clock truth; async queue made "7s" steps look fake.
    step_sync = os.environ.get("H3_STEP_SYNC", "1") == "1"
    step_stats = os.environ.get("H3_STEP_STATS", "0") == "1"
    global_i = 0
    for stage_idx, (stage_h, stage_w, stage_n) in enumerate(run_stages):
        lh, lw = _latent_hw(stage_h, stage_w)
        if video.shape[-2] != lh or video.shape[-1] != lw:
            sigma_here = sigmas[global_i]
            if dist.rank == 0:
                print(
                    f"[generate] upsample latent {tuple(video.shape)} → 24x{lt}x{lh}x{lw} "
                    f"for stage {stage_w}x{stage_h} sigma={float(sigma_here):.4f} "
                    f"noise={os.environ.get('H3_UPSAMPLE_NOISE', '1.0')}",
                    flush=True,
                )
            g_up = torch.Generator(device="cpu")
            g_up.manual_seed(args.seed + 17 + stage_idx)
            video = _upsample_latent_for_stage(
                video, lh, lw, sigma=sigma_here, generator=g_up
            )
            old_dv = old_da = old_sigma_down = None
            # Force RoPE / seq log rebuild for new token count.
            if hasattr(model, "_rope_cache_key"):
                model._rope_cache_key = None
            if hasattr(model, "_logged_seq"):
                model._logged_seq = False
        refs_meta, cond_v, cond_a = ref_by_res[(stage_h, stage_w)]
        payload = {
            "text_token_tags": tags.long(),
            "refs": refs_meta,
            "cond_video_latents": cond_v,
            "cond_audio_latents": cond_a,
            "seed": args.seed,
        }
        for _ in range(stage_n):
            sigma, sigma_next = sigmas[global_i], sigmas[global_i + 1]
            ts = (sigma * 1000.0).view(1)
            if step_sync:
                sync()
            t1 = time.time()
            out = model([video, audio], ts, context, minimax_payload=payload)
            sync()
            dt = time.time() - t1
            step_times.append(dt)
            vel_v, vel_a = out[0].to(video.dtype), out[1].to(audio.dtype)
            if args.sampler == "euler":
                video = video + (sigma_next - sigma).to(video.dtype) * vel_v
                audio = audio + (sigma_next - sigma).to(audio.dtype) * vel_a
            else:
                den_v = _const_denoised(video, vel_v, sigma)
                den_a = _const_denoised(audio, vel_a, sigma)
                sigma_prev = sigmas[global_i - 1] if global_i > 0 else None
                video = _res_multistep_update(video, den_v, sigma, sigma_next, old_dv, old_sigma_down, sigma_prev)
                audio = _res_multistep_update(audio, den_a, sigma, sigma_next, old_da, old_sigma_down, sigma_prev)
                old_dv, old_da, old_sigma_down = den_v, den_a, sigma_next
            rec = metrics.add(
                f"step_{global_i+1}",
                device=device,
                extra={
                    "step_s": round(dt, 3),
                    "sigma": float(sigma),
                    "sigma_next": float(sigma_next),
                    "stage": f"{stage_w}x{stage_h}",
                },
            )
            if dist.rank == 0:
                extra = ""
                if step_stats:
                    extra = f" v_mean={float(vel_v.float().mean()):.4f} v_std={float(vel_v.float().std()):.4f}"
                print(
                    f"[generate] step {global_i+1}/{args.steps} {dt:.3f}s "
                    f"{stage_w}x{stage_h}{extra} "
                    f"npu_alloc={rec['npu_mem_mb']['allocated']}MB rss={rec['host_rss_mb']}MB",
                    flush=True,
                )
                # 供 FastAPI / Gateway 实时进度查询
                try:
                    prog_path = Path(args.out).parent / "progress.json"
                    prog_path.write_text(
                        json.dumps(
                            {
                                "phase": "dit",
                                "step": global_i + 1,
                                "steps_total": int(args.steps),
                                "percent": int((global_i + 1) / max(args.steps, 1) * 90),
                                "step_s": round(dt, 3),
                                "stage": f"{stage_w}x{stage_h}",
                            }
                        )
                    )
                except Exception:
                    pass
            global_i += 1

    metrics.add(
        "dit_done",
        device=device,
        extra={
            "avg_step_s": round(sum(step_times) / max(len(step_times), 1), 3),
            "min_step_s": round(min(step_times), 3) if step_times else None,
            "max_step_s": round(max(step_times), 3) if step_times else None,
            "wall_s": round(time.time() - t_job, 1),
            "request_s": round(time.time() - t_ready, 1),
            "progressive": ",".join(f"{w}x{h}:{n}" for h, w, n in run_stages),
        },
    )
    if dist.rank == 0:
        avg = sum(step_times) / max(len(step_times), 1)
        print(
            f"[generate] dit_done avg_step={avg:.3f}s n={len(step_times)} "
            f"wall={time.time()-t_job:.1f}s request={time.time()-t_ready:.1f}s target=300s",
            flush=True,
        )

    t_vae = time.time()
    if os.environ.get("H3_SKIP_VAE", "0") == "1":
        if dist.rank == 0:
            print("[generate] H3_SKIP_VAE=1 — skip decode/write (warmup)", flush=True)
        del video, audio
        empty_cache()
        barrier()
        return

    if dist.rank == 0:
        try:
            (Path(args.out).parent / "progress.json").write_text(
                json.dumps(
                    {
                        "phase": "vae",
                        "step": int(args.steps),
                        "steps_total": int(args.steps),
                        "percent": 92,
                    }
                )
            )
        except Exception:
            pass
    # 先把 audio latent 拷到 CPU，避免与 video VAE 抢 NPU 显存
    audio_cpu = audio.detach().float().cpu()
    del audio
    empty_cache()

    vae = load_video_vae(vae_path, device=device, dtype=torch.float16)
    assert_npu_resident(vae, allow_cpu=False)
    if dist.rank == 0 and os.environ.get("H3_STEP_STATS", "0") == "1":
        vf = video.float()
        af = audio_cpu
        print(
            f"[generate] latent video mean={float(vf.mean()):.4f} std={float(vf.std()):.4f} "
            f"audio mean={float(af.mean()):.4f} std={float(af.std()):.4f}",
            flush=True,
        )
    pixels = vae.decode(video)
    sync()
    metrics.add("vae_decoded", device=device, extra={"vae_s": round(time.time() - t_vae, 2)})
    del vae, video
    empty_cache()

    waveform = None
    if os.environ.get("H3_SKIP_AUDIO", "0") != "1":
        if dist.rank == 0:
            try:
                (Path(args.out).parent / "progress.json").write_text(
                    json.dumps(
                        {
                            "phase": "audio",
                            "step": int(args.steps),
                            "steps_total": int(args.steps),
                            "percent": 96,
                        }
                    )
                )
            except Exception:
                pass
            t_audio = time.time()
            audio_vae_path = DEFAULT_MODELS / "vae" / "minimax_h3_audio_vae_fp32.safetensors"
            # Audio VAE 约 0.6GB FP32：默认 CPU 解码，避开 DiT resident 后的 HBM 压力
            audio_vae = load_audio_vae(audio_vae_path, device=torch.device("cpu"), dtype=torch.float32)
            with torch.inference_mode():
                waveform = audio_vae.decode(audio_cpu.to(dtype=torch.float32))
            del audio_vae
            metrics.add(
                "audio_decoded",
                device=device,
                extra={"audio_s": round(time.time() - t_audio, 2), "wave": list(waveform.shape)},
            )
            print(
                f"[generate] audio decoded shape={tuple(waveform.shape)} "
                f"sr={AUDIO_SAMPLE_RATE} in {time.time() - t_audio:.2f}s",
                flush=True,
            )
    del audio_cpu

    if dist.rank == 0:
        # Pull pixels to host before the (optional) 1080P upsample + ffmpeg so
        # we do not keep a 6 GiB NPU tensor alive across the final collective.
        pix = pixels.detach().float().cpu()
        del pixels
        empty_cache()
        if pix.shape[-2] != args.height or pix.shape[-1] != args.width:
            print(
                f"[generate] pixel upsample {tuple(pix.shape)} → "
                f"1x3x{pix.shape[2]}x{args.height}x{args.width} (CPU)",
                flush=True,
            )
            pix = torch.nn.functional.interpolate(
                pix,
                size=(pix.shape[2], args.height, args.width),
                mode="trilinear",
                align_corners=False,
            )
        try:
            (Path(args.out).parent / "progress.json").write_text(
                json.dumps(
                    {
                        "phase": "mux",
                        "step": int(args.steps),
                        "steps_total": int(args.steps),
                        "percent": 98,
                    }
                )
            )
        except Exception:
            pass
        write_mp4(args.out, pix, fps=FPS, audio=waveform, sample_rate=AUDIO_SAMPLE_RATE)
        metrics.add("video_written", device=device, extra={"out": str(args.out)})
        metrics.dump(args.out.with_suffix(".metrics.json"))
        print(f"[generate] metrics → {args.out.with_suffix('.metrics.json')}", flush=True)
        del pix, waveform
    else:
        del pixels
        empty_cache()
    # Serve mode: ranks rendezvous via done.* files — avoid a late HCCL barrier
    # while rank0 is stuck in ffmpeg (previously tripped SDMA 507035).
    if os.environ.get("H3_SERVE", "0") != "1":
        barrier()
    if dist.rank == 0:
        wall = time.time() - t_job
        req = time.time() - t_ready
        print(
            f"[generate] done wall={wall:.1f}s request={req:.1f}s target=300s "
            f"{'OK' if req <= 300 else 'OVER'}",
            flush=True,
        )


def generate(argv=None) -> None:
    model, device, args = load_dit_to_npu(argv)
    run_generate(model, device, args)


def main(argv=None):
    generate(argv)


if __name__ == "__main__":
    main()
