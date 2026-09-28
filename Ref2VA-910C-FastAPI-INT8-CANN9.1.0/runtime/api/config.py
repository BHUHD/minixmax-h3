"""运行时配置（环境变量）。"""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Settings:
    """容器 Worker 配置。"""

    job_root: Path
    serve_dir: Path
    mock_mode: bool
    mock_step_delay: float
    mock_fail_task_id: str | None
    task_timeout: float
    host: str
    port: int
    version: str
    task_retention_days: float
    cleanup_interval_sec: float
    h3_partition: str

    @classmethod
    def from_env(cls) -> Settings:
        job_root = Path(os.environ.get("H3_JOB_ROOT", "/workspace/jobs"))
        return cls(
            job_root=job_root,
            serve_dir=Path(os.environ.get("H3_SERVE_DIR", "/workspace/out/h3_serve")),
            mock_mode=os.environ.get("H3_API_MOCK", "0") == "1",
            mock_step_delay=float(os.environ.get("H3_MOCK_STEP_DELAY", "0.05")),
            mock_fail_task_id=os.environ.get("H3_MOCK_FAIL_TASK_ID") or None,
            task_timeout=float(os.environ.get("H3_TASK_TIMEOUT", "1800")),
            host=os.environ.get("H3_API_HOST", "0.0.0.0"),
            port=int(os.environ.get("H3_API_PORT", "8080")),
            version=os.environ.get("H3_API_VERSION", "0.1.0"),
            task_retention_days=float(os.environ.get("H3_TASK_RETENTION_DAYS", "7")),
            cleanup_interval_sec=float(os.environ.get("H3_TASK_CLEANUP_INTERVAL", "3600")),
            h3_partition=os.environ.get("H3_PARTITION", "ref2va").strip().lower(),
        )


def get_settings() -> Settings:
    return Settings.from_env()
