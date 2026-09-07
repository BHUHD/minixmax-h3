"""API 响应模型。"""
from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field


TaskStatus = Literal["running", "succeeded", "failed"]


class ProgressInfo(BaseModel):
    phase: str = "accepted"
    step: int = 0
    steps_total: int = 0
    percent: int = 0


class OutputInfo(BaseModel):
    ready: bool = False
    size_bytes: int | None = None
    filename: str = "output.mp4"


class TaskInfo(BaseModel):
    task_id: str
    instance_id: str
    status: TaskStatus
    progress: ProgressInfo
    output: OutputInfo
    error: str | None = None
    started_at: str | None = None
    finished_at: str | None = None
    input_counts: dict[str, int] = Field(default_factory=dict)


class StatusResponse(BaseModel):
    service: Literal["starting", "ready", "degraded", "error"]
    npu_serve_ready: bool
    busy: bool
    current_task_id: str | None
    instance_id: str
    uptime_sec: float
    version: str
    mock_mode: bool
    partition: str = "ref2va"


class TaskAcceptedResponse(BaseModel):
    task_id: str
    instance_id: str
    status: Literal["running"] = "running"


class BusyResponse(BaseModel):
    error: str = "busy"
    current_task_id: str


class ErrorDetail(BaseModel):
    detail: str
    reason: str | None = None
