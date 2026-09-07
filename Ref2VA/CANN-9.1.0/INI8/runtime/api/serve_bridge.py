"""与 h3_serve 邮箱协议对接（真实 NPU serve 模式）。"""
from __future__ import annotations

import json
import time
from pathlib import Path

from .storage import TaskRecord


def serve_ready(serve_dir: Path) -> bool:
    return (serve_dir / "READY").is_file()


def submit_to_mailbox(
    serve_dir: Path,
    rec: TaskRecord,
    *,
    ref_images: list[str],
    ref_videos: list[str],
    ref_audios: list[str],
    output_path: Path,
) -> None:
    """向 resident serve 写入 job.json。"""
    serve_dir.mkdir(parents=True, exist_ok=True)
    done = serve_dir / "DONE"
    if done.exists():
        done.unlink()

    job = {
        "id": rec.task_id,
        "steps": int(rec.params.get("steps", 20)),
        "seconds": float(rec.params.get("duration", 10)),
        "height": int(rec.params.get("height", 1088)),
        "width": int(rec.params.get("width", 1920)),
        "out": str(output_path),
        "prompt": rec.prompt,
    }
    if rec.params.get("seed") is not None:
        job["seed"] = int(rec.params["seed"])
    task = str(rec.params.get("task") or "auto").strip().lower()
    if task and task != "auto":
        job["task"] = task
    # 始终写入列表（可为 []），避免 resident serve 沿用上一单的 video/audio
    job["ref_images"] = list(ref_images)
    job["ref_videos"] = list(ref_videos)
    job["ref_audios"] = list(ref_audios)

    (serve_dir / "job.json").write_text(json.dumps(job, indent=2))


def wait_mailbox_done(
    serve_dir: Path,
    task_id: str,
    *,
    timeout: float,
    poll: float = 0.5,
) -> bool:
    """等待 serve 写 DONE 且 id 匹配。"""
    t0 = time.time()
    done = serve_dir / "DONE"
    while time.time() - t0 < timeout:
        if done.is_file() and done.read_text().strip() == task_id:
            return True
        time.sleep(poll)
    return False


def read_progress_file(task_dir: Path) -> dict | None:
    """若 pipeline 写了 progress.json 则读取。"""
    p = task_dir / "progress.json"
    if p.is_file():
        try:
            return json.loads(p.read_text())
        except Exception:
            return None
    return None
