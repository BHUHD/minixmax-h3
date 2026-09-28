"""任务 TTL 定时清理（默认保留 7 天；容器重启时由 startup_cleanup 全清）。"""
from __future__ import annotations

import json
import logging
import shutil
import threading
import time
from datetime import datetime, timezone
from pathlib import Path

from .storage import JobStorage, TaskRecord

log = logging.getLogger("h3.api.cleanup")


def _parse_ts(ts: str | None) -> datetime | None:
    if not ts:
        return None
    try:
        return datetime.fromisoformat(ts.replace("Z", "+00:00"))
    except Exception:
        return None


class TaskRetentionCleaner:
    """后台扫描并删除超过保留期的已完成/失败任务。"""

    def __init__(
        self,
        storage: JobStorage,
        *,
        retention_days: float = 7.0,
        interval_sec: float = 3600.0,
    ) -> None:
        self.storage = storage
        self.retention_sec = retention_days * 86400.0
        self.interval_sec = interval_sec
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._purged: set[str] = set()

    def start(self) -> None:
        if self._thread and self._thread.is_alive():
            return
        self._thread = threading.Thread(target=self._loop, daemon=True, name="task-retention")
        self._thread.start()
        # 启动后立即扫一遍
        self.run_once()

    def stop(self) -> None:
        self._stop.set()

    def _loop(self) -> None:
        while not self._stop.wait(self.interval_sec):
            self.run_once()

    def is_purged(self, task_id: str) -> bool:
        return task_id in self._purged

    def run_once(self) -> list[str]:
        if self.retention_sec <= 0:
            return []
        now = datetime.now(timezone.utc)
        removed: list[str] = []
        if not self.storage.job_root.is_dir():
            return removed

        for child in self.storage.job_root.iterdir():
            if not child.is_dir() or child.name.startswith("."):
                continue
            meta = child / "meta.json"
            if not meta.is_file():
                continue
            try:
                rec = TaskRecord.from_dict(json.loads(meta.read_text()))
            except Exception:
                continue
            if rec.status not in ("succeeded", "failed"):
                continue
            finished = _parse_ts(rec.finished_at)
            if finished is None:
                continue
            age = (now - finished.astimezone(timezone.utc)).total_seconds()
            if age <= self.retention_sec:
                continue
            tid = rec.task_id
            shutil.rmtree(child, ignore_errors=True)
            self.storage._tasks.pop(tid, None)
            self._purged.add(tid)
            removed.append(tid)
            log.info("purged expired task %s age=%.0fs", tid, age)

        if removed:
            self.storage._write_instance_file()
        return removed
