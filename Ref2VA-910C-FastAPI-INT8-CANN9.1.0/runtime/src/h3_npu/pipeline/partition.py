"""MiniMax-H3 生成模式：Ref2VA 带参考 / 无参考纯文本。

Comfy ``MiniMaxH3ReferenceToVideo`` 允许 ref_images/videos/audios 全空：
仍加载 Ref2VA DiT，序列退化为 ``[text | audio | video]``，不注入 presentation 标签。

``H3_PARTITION`` 仅决定加载哪套 DiT checkpoint（ref2va / fl2va），
无参考 t2va 在 ref2va 与 fl2va 分区上行为一致（空 ref、纯 prompt）。
"""
from __future__ import annotations

import os
from pathlib import Path

PARTITION_REF2VA = "ref2va"
PARTITION_FL2VA = "fl2va"

DIT_CHECKPOINTS: dict[str, str] = {
    PARTITION_REF2VA: "minimax_h3_ref2va_pruned_int8_convrot.safetensors",
    PARTITION_FL2VA: "minimax_h3_fl2va_pruned_int8_convrot.safetensors",
}

T2VA_TASKS = frozenset({"t2va", "text", "text2video", "txt2vid"})
REF2VA_TASKS = frozenset({"ref2va", "r2v", "reference", "ref2video"})


def get_serve_partition() -> str:
    """当前 serve 加载的分区（容器 env ``H3_PARTITION``）。"""
    raw = (os.environ.get("H3_PARTITION") or PARTITION_REF2VA).strip().lower()
    if raw in (PARTITION_REF2VA, PARTITION_FL2VA):
        return raw
    return PARTITION_REF2VA


def resolve_dit_path(models_dir: Path, partition: str | None = None) -> Path:
    """按分区解析 DiT checkpoint；``H3_DIT_CHECKPOINT`` 可覆盖文件名或相对路径。"""
    override = os.environ.get("H3_DIT_CHECKPOINT")
    if override:
        p = Path(override)
        if p.is_absolute():
            return p
        cand = models_dir / override
        if cand.is_file():
            return cand
        return models_dir / "diffusion_models" / override
    part = partition or get_serve_partition()
    name = DIT_CHECKPOINTS[part]
    return models_dir / "diffusion_models" / name


def has_any_refs(n_images: int, n_videos: int, n_audios: int) -> bool:
    return (n_images + n_videos + n_audios) > 0


def normalize_task(task: str | None) -> str:
    """归一化为 auto | t2va | ref2va。"""
    t = (task or os.environ.get("H3_TASK") or "auto").strip().lower()
    if t in ("auto", ""):
        return "auto"
    if t in T2VA_TASKS:
        return "t2va"
    if t in REF2VA_TASKS:
        return "ref2va"
    return t


def resolve_generation_mode(
    partition: str,
    *,
    n_images: int,
    n_videos: int,
    n_audios: int,
    task: str | None = None,
) -> str:
    """返回 ``t2va``（无参考纯文本）或 ``ref2va``（带参考）。

    Ref2VA 权重下无参考 t2va 与 Comfy ReferenceToVideo 零参考路径对齐。
    """
    t = normalize_task(task)
    refs = has_any_refs(n_images, n_videos, n_audios)

    if t == "t2va":
        if refs:
            raise ValueError("t2va 任务不能上传参考图/视频/音频")
        return "t2va"

    if t == "ref2va":
        if not refs:
            raise ValueError("ref2va 任务至少需要一路参考（图/视频/音频）")
        return "ref2va"

    # auto：有参考 → ref2va；无参考 → t2va（Ref2VA / FL2VA DiT 均可）
    if refs:
        return "ref2va"
    return "t2va"
