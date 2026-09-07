"""后台任务执行：mock 模拟 / 真实 mailbox。"""
from __future__ import annotations

import struct
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

from .config import Settings
from .serve_bridge import read_progress_file, serve_ready, submit_to_mailbox, wait_mailbox_done
from .storage import JobStorage, TaskRecord


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _minimal_mp4_bytes() -> bytes:
    """生成最小可识别 mp4 头（测试/模拟用）。"""
    # ftyp box: 仅用于模拟测试，非真实可播放长视频
    body = b"isomiso2mp41"
    size = 8 + len(body)
    return struct.pack(">I", size) + b"ftyp" + body


class TaskWorker:
    """串行任务执行器。"""

    def __init__(
        self,
        settings: Settings,
        storage: JobStorage,
        *,
        on_state_change: Callable[[], None] | None = None,
    ) -> None:
        self.settings = settings
        self.storage = storage
        self._lock = threading.Lock()
        self._current_task_id: str | None = None
        self._thread: threading.Thread | None = None
        self._on_state_change = on_state_change
        self._npu_ready = False
        self._ready_thread: threading.Thread | None = None

    def start_ready_watch(self) -> None:
        if self.settings.mock_mode:
            self._npu_ready = True
            return

        def _watch() -> None:
            while True:
                if serve_ready(self.settings.serve_dir):
                    self._npu_ready = True
                    self._notify()
                    return
                time.sleep(1.0)

        self._ready_thread = threading.Thread(target=_watch, daemon=True, name="npu-ready-watch")
        self._ready_thread.start()

    @property
    def npu_serve_ready(self) -> bool:
        return self._npu_ready

    @property
    def busy(self) -> bool:
        with self._lock:
            return self._current_task_id is not None

    @property
    def current_task_id(self) -> str | None:
        with self._lock:
            return self._current_task_id

    def _notify(self) -> None:
        if self._on_state_change:
            self._on_state_change()

    def submit(
        self,
        rec: TaskRecord,
        *,
        ref_image_paths: list[str],
        ref_video_paths: list[str],
        ref_audio_paths: list[str],
    ) -> None:
        with self._lock:
            if self._current_task_id is not None:
                raise RuntimeError("worker busy")
            self._current_task_id = rec.task_id

        self._thread = threading.Thread(
            target=self._run_task,
            args=(rec, ref_image_paths, ref_video_paths, ref_audio_paths),
            daemon=True,
            name=f"task-{rec.task_id[:8]}",
        )
        self._thread.start()

    def _run_task(
        self,
        rec: TaskRecord,
        ref_image_paths: list[str],
        ref_video_paths: list[str],
        ref_audio_paths: list[str],
    ) -> None:
        try:
            if self.settings.mock_mode:
                self._run_mock(rec)
            else:
                self._run_real(rec, ref_image_paths, ref_video_paths, ref_audio_paths)
        except Exception as exc:
            rec.status = "failed"
            rec.error = str(exc)
            rec.finished_at = _utc_now()
            self.storage.save_task(rec)
        finally:
            with self._lock:
                self._current_task_id = None
            self._notify()

    def _update_progress(
        self,
        rec: TaskRecord,
        *,
        phase: str,
        step: int,
        steps_total: int,
    ) -> None:
        percent = int(step / steps_total * 100) if steps_total else 0
        rec.progress = {
            "phase": phase,
            "step": step,
            "steps_total": steps_total,
            "percent": min(percent, 99) if rec.status == "running" else percent,
        }
        self.storage.save_task(rec)
        (self.storage.task_dir(rec.task_id) / "progress.json").write_text(
            __import__("json").dumps(rec.progress)
        )
        self._notify()

    def _run_mock(self, rec: TaskRecord) -> None:
        if self.settings.mock_fail_task_id == rec.task_id:
            raise RuntimeError("mock failure injected")
        if rec.prompt.strip() == "__mock_fail__":
            raise RuntimeError("mock failure via prompt")

        steps = int(rec.params.get("steps", 20))
        delay = self.settings.mock_step_delay

        self._update_progress(rec, phase="encode", step=0, steps_total=steps)
        time.sleep(delay)

        for i in range(1, steps + 1):
            phase = "dit" if i < steps else "vae"
            self._update_progress(rec, phase=phase, step=i, steps_total=steps)
            time.sleep(delay)

        self._update_progress(rec, phase="mux", step=steps, steps_total=steps)
        out = self.storage.output_path(rec.task_id)
        out.write_bytes(_minimal_mp4_bytes())

        rec.status = "succeeded"
        rec.output_ready = True
        rec.output_size = out.stat().st_size
        rec.progress["phase"] = "done"
        rec.progress["percent"] = 100
        rec.finished_at = _utc_now()
        self.storage.save_task(rec)
        self._export_output(rec.task_id, out)

    def _run_real(
        self,
        rec: TaskRecord,
        ref_image_paths: list[str],
        ref_video_paths: list[str],
        ref_audio_paths: list[str],
    ) -> None:
        if not self._npu_ready:
            raise RuntimeError("npu serve not ready")

        out = self.storage.output_path(rec.task_id)
        self._update_progress(rec, phase="queued", step=0, steps_total=int(rec.params.get("steps", 20)))

        submit_to_mailbox(
            self.settings.serve_dir,
            rec,
            ref_images=ref_image_paths,
            ref_videos=ref_video_paths,
            ref_audios=ref_audio_paths,
            output_path=out,
        )

        steps_total = int(rec.params.get("steps", 20))
        t0 = time.time()
        while time.time() - t0 < self.settings.task_timeout:
            # serve 进程崩溃：READY 被删或 mailbox 目录无存活信号
            if not (self.settings.serve_dir / "READY").is_file():
                raise RuntimeError("npu serve crashed (READY missing)")

            prog = read_progress_file(self.storage.task_dir(rec.task_id))
            if prog:
                rec.progress = prog
                self.storage.save_task(rec)
                self._notify()

            failed_flag = self.settings.serve_dir / "FAILED"
            if failed_flag.is_file() and failed_flag.read_text().strip() == rec.task_id:
                raise RuntimeError("inference failed (serve FAILED)")

            if wait_mailbox_done(
                self.settings.serve_dir,
                rec.task_id,
                timeout=0.5,
                poll=0.1,
            ):
                if out.is_file():
                    rec.status = "succeeded"
                    rec.output_ready = True
                    rec.output_size = out.stat().st_size
                    rec.progress = {
                        "phase": "done",
                        "step": steps_total,
                        "steps_total": steps_total,
                        "percent": 100,
                    }
                    rec.finished_at = _utc_now()
                    self.storage.save_task(rec)
                    # 可选：同步到宿主机挂载的输出目录（H3_OUTPUT_DIR）
                    self._export_output(rec.task_id, out)
                    return
                raise RuntimeError("DONE received but output.mp4 missing")

            time.sleep(0.5)

        raise TimeoutError(f"task timed out after {self.settings.task_timeout}s")

    def _export_output(self, task_id: str, src: Path) -> None:
        """若设置了 H3_OUTPUT_DIR（compose 挂载），复制成品 mp4 方便宿主机取用。"""
        import os
        import shutil

        out_dir = Path(os.environ.get("H3_OUTPUT_DIR", "/workspace/output"))
        try:
            out_dir.mkdir(parents=True, exist_ok=True)
            dest = out_dir / f"{task_id}.mp4"
            shutil.copy2(src, dest)
            print(f"[api] exported video -> {dest}", flush=True)
        except Exception as exc:
            print(f"[api] export skip: {exc}", flush=True)
