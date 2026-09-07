"""FastAPI 应用与路由。"""
from __future__ import annotations

from pathlib import Path
from typing import Annotated

from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse

from .cleanup import TaskRetentionCleaner
from .config import Settings, get_settings
from .models import (
    BusyResponse,
    ErrorDetail,
    OutputInfo,
    ProgressInfo,
    StatusResponse,
    TaskAcceptedResponse,
    TaskInfo,
)
from .storage import JobStorage
from .worker import TaskWorker


def create_app(settings: Settings | None = None) -> FastAPI:
    cfg = settings or get_settings()

    app = FastAPI(
        title="MiniMax-H3 Ref2VA Worker",
        version=cfg.version,
        description="内网 Worker API，供 API Gateway 调用",
    )
    app.state.settings = cfg
    app.state.storage = None
    app.state.worker = None
    app.state.cleaner = None

    @app.on_event("startup")
    async def _startup() -> None:
        storage = JobStorage(cfg.job_root)
        storage.startup_cleanup()  # 重启：清空全部任务与 mp4
        worker = TaskWorker(cfg, storage)
        worker.start_ready_watch()
        cleaner = TaskRetentionCleaner(
            storage,
            retention_days=cfg.task_retention_days,
            interval_sec=cfg.cleanup_interval_sec,
        )
        cleaner.start()
        app.state.storage = storage
        app.state.worker = worker
        app.state.cleaner = cleaner

    @app.on_event("shutdown")
    async def _shutdown() -> None:
        cleaner = app.state.cleaner
        if cleaner:
            cleaner.stop()

    def _require_state(request: Request) -> tuple[JobStorage, TaskWorker, Settings]:
        storage = request.app.state.storage
        worker = request.app.state.worker
        settings_obj = request.app.state.settings
        if storage is None or worker is None:
            raise HTTPException(status_code=503, detail="worker not initialized")
        return storage, worker, settings_obj

    @app.get("/health")
    async def health() -> dict[str, str]:
        return {"status": "ok"}

    @app.get("/v1/status", response_model=StatusResponse)
    async def status(request: Request) -> StatusResponse:
        storage, worker, settings_obj = _require_state(request)
        svc = "ready" if worker.npu_serve_ready else "starting"
        if not worker.npu_serve_ready and settings_obj.mock_mode:
            svc = "ready"
        return StatusResponse(
            service=svc,
            npu_serve_ready=worker.npu_serve_ready,
            busy=worker.busy,
            current_task_id=worker.current_task_id,
            instance_id=storage.instance_id,
            uptime_sec=round(storage.uptime_sec(), 1),
            version=settings_obj.version,
            mock_mode=settings_obj.mock_mode,
            partition=settings_obj.h3_partition,
        )

    @app.post("/v1/tasks", response_model=TaskAcceptedResponse, status_code=202)
    async def create_task(
        request: Request,
        prompt: Annotated[str, Form()],
        width: Annotated[int, Form()] = 1920,
        height: Annotated[int, Form()] = 1088,
        duration: Annotated[float, Form()] = 10.0,
        steps: Annotated[int, Form()] = 20,
        seed: Annotated[int | None, Form()] = None,
        task: Annotated[str, Form()] = "auto",
        ref_images: Annotated[list[UploadFile], File()] = [],
        ref_videos: Annotated[list[UploadFile], File()] = [],
        ref_audios: Annotated[list[UploadFile], File()] = [],
    ):
        storage, worker, settings_obj = _require_state(request)

        if not worker.npu_serve_ready:
            raise HTTPException(
                status_code=503,
                detail=ErrorDetail(detail="npu serve not ready", reason="not_ready").model_dump(),
            )
        if worker.busy:
            raise HTTPException(
                status_code=409,
                detail=BusyResponse(current_task_id=worker.current_task_id or "").model_dump(),
            )

        params = {
            "width": width,
            "height": height,
            "duration": duration,
            "steps": steps,
            "task": task.strip().lower() if task else "auto",
        }
        if seed is not None:
            params["seed"] = seed

        # 提交前校验 task / partition / 参考组合
        import sys

        sys.path.insert(0, "/workspace/src")
        from h3_npu.pipeline.partition import resolve_generation_mode

        n_img, n_vid, n_aud = len(ref_images), len(ref_videos), len(ref_audios)
        try:
            resolve_generation_mode(
                settings_obj.h3_partition,
                n_images=n_img,
                n_videos=n_vid,
                n_audios=n_aud,
                task=task,
            )
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

        input_counts = {
            "ref_images": len(ref_images),
            "ref_videos": len(ref_videos),
            "ref_audios": len(ref_audios),
        }

        rec = storage.create_task(prompt=prompt, params=params, input_counts=input_counts)
        inp = storage.input_dir(rec.task_id)

        async def _save_files(files: list[UploadFile], subdir: str) -> list[str]:
            paths: list[str] = []
            dest_root = inp / subdir
            dest_root.mkdir(parents=True, exist_ok=True)
            for idx, uf in enumerate(files):
                name = uf.filename or f"{subdir}_{idx}"
                dest = dest_root / name
                content = await uf.read()
                dest.write_bytes(content)
                paths.append(str(dest))
            return paths

        ref_image_paths = await _save_files(ref_images, "images")
        ref_video_paths = await _save_files(ref_videos, "videos")
        ref_audio_paths = await _save_files(ref_audios, "audios")

        rec.progress["phase"] = "accepted"
        storage.save_task(rec)

        try:
            worker.submit(
                rec,
                ref_image_paths=ref_image_paths,
                ref_video_paths=ref_video_paths,
                ref_audio_paths=ref_audio_paths,
            )
        except RuntimeError:
            raise HTTPException(
                status_code=409,
                detail=BusyResponse(current_task_id=worker.current_task_id or "").model_dump(),
            )

        return TaskAcceptedResponse(task_id=rec.task_id, instance_id=storage.instance_id)

    @app.get("/v1/tasks/{task_id}", response_model=TaskInfo)
    async def get_task(task_id: str, request: Request) -> TaskInfo:
        storage, _, _ = _require_state(request)
        cleaner = request.app.state.cleaner
        if cleaner and cleaner.is_purged(task_id):
            raise HTTPException(
                status_code=410,
                detail=ErrorDetail(
                    detail="task expired and purged",
                    reason="expired",
                ).model_dump(),
            )
        rec = storage.get_task(task_id)
        if rec is None:
            if storage.is_tombstoned(task_id):
                raise HTTPException(
                    status_code=410,
                    detail=ErrorDetail(
                        detail="task lost due to container restart",
                        reason="container_restarted",
                    ).model_dump(),
                )
            raise HTTPException(status_code=404, detail="task not found")

        return TaskInfo(
            task_id=rec.task_id,
            instance_id=rec.instance_id,
            status=rec.status,  # type: ignore[arg-type]
            progress=ProgressInfo(**rec.progress),
            output=OutputInfo(
                ready=rec.output_ready,
                size_bytes=rec.output_size,
            ),
            error=rec.error,
            started_at=rec.started_at,
            finished_at=rec.finished_at,
            input_counts=rec.input_counts,
        )

    @app.get("/v1/tasks/{task_id}/video")
    async def download_video(task_id: str, request: Request):
        storage, _, _ = _require_state(request)
        cleaner = request.app.state.cleaner
        if cleaner and cleaner.is_purged(task_id):
            raise HTTPException(
                status_code=410,
                detail=ErrorDetail(
                    detail="task expired and purged",
                    reason="expired",
                ).model_dump(),
            )
        rec = storage.get_task(task_id)
        if rec is None:
            if storage.is_tombstoned(task_id):
                raise HTTPException(
                    status_code=410,
                    detail=ErrorDetail(
                        detail="task lost due to container restart",
                        reason="container_restarted",
                    ).model_dump(),
                )
            raise HTTPException(status_code=404, detail="task not found")
        if rec.status != "succeeded" or not rec.output_ready:
            raise HTTPException(status_code=409, detail="video not ready")
        out = storage.output_path(task_id)
        if not out.is_file():
            raise HTTPException(status_code=404, detail="video file missing")
        return FileResponse(
            path=str(out),
            media_type="video/mp4",
            filename="output.mp4",
        )

    return app


# 模块级 app（容器 uvicorn 默认加载）
app = create_app()
