"""任务目录、元数据与重启 tombstone 管理。"""
from __future__ import annotations

import json
import shutil
import time
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass
class TaskRecord:
    """单个任务的持久化记录。"""

    task_id: str
    instance_id: str
    status: str = "running"
    prompt: str = ""
    params: dict[str, Any] = field(default_factory=dict)
    input_counts: dict[str, int] = field(default_factory=dict)
    progress: dict[str, Any] = field(default_factory=lambda: {
        "phase": "accepted",
        "step": 0,
        "steps_total": 0,
        "percent": 0,
    })
    error: str | None = None
    started_at: str | None = None
    finished_at: str | None = None
    output_ready: bool = False
    output_size: int | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> TaskRecord:
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})


class JobStorage:
    """管理 /workspace/jobs 下的任务与重启 tombstone。"""

    TOMBSTONE_NAME = ".tombstone"
    INSTANCE_NAME = ".instance.json"

    def __init__(self, job_root: Path) -> None:
        self.job_root = job_root
        self.tombstone_dir = job_root / self.TOMBSTONE_NAME
        self.instance_id = str(uuid.uuid4())
        self.started_at = time.time()
        self._tasks: dict[str, TaskRecord] = {}

    def startup_cleanup(self) -> None:
        """启动时：归档上一实例任务 ID，再清空任务缓存与 mp4。"""
        self.job_root.mkdir(parents=True, exist_ok=True)
        self.tombstone_dir.mkdir(parents=True, exist_ok=True)

        prev_instance_file = self.job_root / self.INSTANCE_NAME
        prev_task_ids: list[str] = []
        prev_instance_id: str | None = None
        if prev_instance_file.is_file():
            try:
                prev = json.loads(prev_instance_file.read_text())
                prev_instance_id = prev.get("instance_id")
                prev_task_ids = list(prev.get("task_ids", []))
            except Exception:
                pass

        if prev_instance_id and prev_task_ids:
            tomb = self.tombstone_dir / f"{prev_instance_id}.json"
            tomb.write_text(
                json.dumps(
                    {
                        "instance_id": prev_instance_id,
                        "task_ids": prev_task_ids,
                        "archived_at": _utc_now(),
                    },
                    indent=2,
                )
            )

        # 清除任务目录与本机 mp4（保留 tombstone）
        for child in self.job_root.iterdir():
            if child.name == self.TOMBSTONE_NAME:
                continue
            if child.is_dir():
                shutil.rmtree(child, ignore_errors=True)
            elif child.suffix.lower() == ".mp4":
                child.unlink(missing_ok=True)
            elif child.name != self.INSTANCE_NAME:
                child.unlink(missing_ok=True)

        self._tasks.clear()
        self._write_instance_file()

    def _write_instance_file(self) -> None:
        path = self.job_root / self.INSTANCE_NAME
        path.write_text(
            json.dumps(
                {
                    "instance_id": self.instance_id,
                    "task_ids": sorted(self._tasks.keys()),
                    "started_at": _utc_now(),
                },
                indent=2,
            )
        )

    def task_dir(self, task_id: str) -> Path:
        return self.job_root / task_id

    def input_dir(self, task_id: str) -> Path:
        return self.task_dir(task_id) / "input"

    def output_path(self, task_id: str) -> Path:
        return self.task_dir(task_id) / "output.mp4"

    def meta_path(self, task_id: str) -> Path:
        return self.task_dir(task_id) / "meta.json"

    def create_task(
        self,
        *,
        prompt: str,
        params: dict[str, Any],
        input_counts: dict[str, int],
    ) -> TaskRecord:
        task_id = str(uuid.uuid4())
        rec = TaskRecord(
            task_id=task_id,
            instance_id=self.instance_id,
            prompt=prompt,
            params=params,
            input_counts=input_counts,
            started_at=_utc_now(),
            progress={
                "phase": "accepted",
                "step": 0,
                "steps_total": int(params.get("steps", 20)),
                "percent": 0,
            },
        )
        tdir = self.task_dir(task_id)
        tdir.mkdir(parents=True, exist_ok=True)
        self.input_dir(task_id).mkdir(parents=True, exist_ok=True)
        self.save_task(rec)
        self._tasks[task_id] = rec
        self._write_instance_file()
        return rec

    def save_task(self, rec: TaskRecord) -> None:
        meta = self.meta_path(rec.task_id)
        meta.parent.mkdir(parents=True, exist_ok=True)
        meta.write_text(json.dumps(rec.to_dict(), indent=2))
        self._tasks[rec.task_id] = rec

    def get_task(self, task_id: str) -> TaskRecord | None:
        if task_id in self._tasks:
            return self._tasks[task_id]
        meta = self.meta_path(task_id)
        if meta.is_file():
            rec = TaskRecord.from_dict(json.loads(meta.read_text()))
            self._tasks[task_id] = rec
            return rec
        return None

    def is_tombstoned(self, task_id: str) -> bool:
        """任务是否属于已重启的上一个实例。"""
        for tomb in self.tombstone_dir.glob("*.json"):
            try:
                data = json.loads(tomb.read_text())
                if task_id in data.get("task_ids", []):
                    return True
            except Exception:
                continue
        return False

    def uptime_sec(self) -> float:
        return time.time() - self.started_at
