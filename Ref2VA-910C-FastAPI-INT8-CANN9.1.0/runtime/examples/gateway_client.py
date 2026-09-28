#!/usr/bin/env python3
"""
API Gateway 调用 Worker 容器示例。

用法:
  python3 gateway_client.py --base http://127.0.0.1:8080 \\
      --prompt "your prompt" \\
      --ref-image img1.png --ref-image img2.png \\
      --out /tmp/result.mp4

流程: 查闲 → 提交 → 轮询 → 下载 → 保存本地
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import httpx


class WorkerRestarted(Exception):
    """容器 instance_id 变化或任务 410，表示 Worker 已重启。"""


class GatewayClient:
    """面向 Worker 容器的 Gateway 侧客户端。"""

    def __init__(self, base_url: str, *, poll_interval: float = 1.0, timeout: float = 1800.0):
        self.base = base_url.rstrip("/")
        self.poll_interval = poll_interval
        self.timeout = timeout
        self._known_instance_id: str | None = None

    def _url(self, path: str) -> str:
        return f"{self.base}{path}"

    def get_status(self) -> dict:
        r = httpx.get(self._url("/v1/status"), timeout=10.0)
        r.raise_for_status()
        data = r.json()
        iid = data.get("instance_id")
        if self._known_instance_id and iid and iid != self._known_instance_id:
            raise WorkerRestarted(
                f"instance_id changed: {self._known_instance_id} -> {iid}"
            )
        if iid:
            self._known_instance_id = iid
        return data

    def wait_idle(self, *, max_wait: float = 600.0) -> dict:
        """等待 Worker 就绪且空闲。"""
        t0 = time.time()
        while time.time() - t0 < max_wait:
            st = self.get_status()
            if st.get("service") == "ready" and st.get("npu_serve_ready") and not st.get("busy"):
                return st
            time.sleep(self.poll_interval)
        raise TimeoutError("worker not idle in time")

    def submit(
        self,
        *,
        prompt: str,
        ref_images: list[Path] | None = None,
        ref_videos: list[Path] | None = None,
        ref_audios: list[Path] | None = None,
        width: int = 1920,
        height: int = 1088,
        duration: float = 10.0,
        steps: int = 20,
        seed: int | None = None,
    ) -> dict:
        """提交生成任务，返回 202 body。"""
        self.wait_idle()
        files = []
        for p in ref_images or []:
            files.append(("ref_images", (p.name, p.read_bytes(), "application/octet-stream")))
        for p in ref_videos or []:
            files.append(("ref_videos", (p.name, p.read_bytes(), "application/octet-stream")))
        for p in ref_audios or []:
            files.append(("ref_audios", (p.name, p.read_bytes(), "application/octet-stream")))

        data = {
            "prompt": prompt,
            "width": str(width),
            "height": str(height),
            "duration": str(duration),
            "steps": str(steps),
        }
        if seed is not None:
            data["seed"] = str(seed)

        r = httpx.post(self._url("/v1/tasks"), data=data, files=files, timeout=120.0)
        if r.status_code == 409:
            raise RuntimeError(f"worker busy: {r.json()}")
        if r.status_code == 503:
            raise RuntimeError(f"worker not ready: {r.text}")
        r.raise_for_status()
        body = r.json()
        # 记录提交时的 instance_id，后续轮询用于检测重启
        self._known_instance_id = body.get("instance_id", self._known_instance_id)
        return body

    def get_task(self, task_id: str) -> dict:
        r = httpx.get(self._url(f"/v1/tasks/{task_id}"), timeout=10.0)
        if r.status_code == 410:
            raise WorkerRestarted(r.json().get("detail", {}).get("reason", "container_restarted"))
        if r.status_code == 404:
            raise WorkerRestarted("task_not_found_maybe_restarted")
        r.raise_for_status()
        data = r.json()
        # 轮询时也校验 instance_id
        iid = data.get("instance_id")
        if self._known_instance_id and iid and iid != self._known_instance_id:
            raise WorkerRestarted("instance_id mismatch during poll")
        return data

    def wait_task(self, task_id: str) -> dict:
        """轮询至 succeeded / failed。"""
        t0 = time.time()
        while time.time() - t0 < self.timeout:
            try:
                info = self.get_task(task_id)
            except WorkerRestarted:
                raise
            st = info.get("status")
            if st in ("succeeded", "failed"):
                return info
            time.sleep(self.poll_interval)
        raise TimeoutError(f"task {task_id} timed out")

    def download_video(self, task_id: str, dest: Path) -> Path:
        r = httpx.get(self._url(f"/v1/tasks/{task_id}/video"), timeout=120.0)
        if r.status_code == 410:
            raise WorkerRestarted("video gone after restart")
        r.raise_for_status()
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(r.content)
        return dest

    def generate_and_download(
        self,
        dest: Path,
        *,
        prompt: str,
        ref_images: list[Path] | None = None,
        ref_videos: list[Path] | None = None,
        ref_audios: list[Path] | None = None,
        **kwargs,
    ) -> dict:
        """完整 Gateway 流程。"""
        accepted = self.submit(
            prompt=prompt,
            ref_images=ref_images,
            ref_videos=ref_videos,
            ref_audios=ref_audios,
            **kwargs,
        )
        task_id = accepted["task_id"]
        print(f"[gateway] submitted task_id={task_id}", flush=True)

        final = self.wait_task(task_id)
        if final["status"] != "succeeded":
            raise RuntimeError(f"task failed: {final.get('error')}")

        path = self.download_video(task_id, dest)
        print(f"[gateway] saved {path} ({path.stat().st_size} bytes)", flush=True)
        return {"task_id": task_id, "output": str(path), "task": final}


def main() -> int:
    ap = argparse.ArgumentParser(description="Gateway -> Worker 调用示例")
    ap.add_argument("--base", default="http://127.0.0.1:8080", help="Worker base URL")
    ap.add_argument("--prompt", required=True)
    ap.add_argument("--ref-image", action="append", default=[], dest="ref_images")
    ap.add_argument("--ref-video", action="append", default=[], dest="ref_videos")
    ap.add_argument("--ref-audio", action="append", default=[], dest="ref_audios")
    ap.add_argument("--width", type=int, default=1920)
    ap.add_argument("--height", type=int, default=1088)
    ap.add_argument("--duration", type=float, default=10.0)
    ap.add_argument("--steps", type=int, default=20)
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--out", type=Path, default=Path("gateway_output.mp4"))
    ap.add_argument("--status-only", action="store_true", help="仅查询 /v1/status")
    args = ap.parse_args()

    client = GatewayClient(args.base)
    if args.status_only:
        print(json.dumps(client.get_status(), indent=2, ensure_ascii=False))
        return 0

    try:
        result = client.generate_and_download(
            args.out,
            prompt=args.prompt,
            ref_images=[Path(p) for p in args.ref_images],
            ref_videos=[Path(p) for p in args.ref_videos],
            ref_audios=[Path(p) for p in args.ref_audios],
            width=args.width,
            height=args.height,
            duration=args.duration,
            steps=args.steps,
            seed=args.seed,
        )
        print(json.dumps(result, indent=2, ensure_ascii=False, default=str))
        return 0
    except WorkerRestarted as e:
        print(f"[gateway] worker restarted: {e}", file=sys.stderr)
        return 2
    except Exception as e:
        print(f"[gateway] error: {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
