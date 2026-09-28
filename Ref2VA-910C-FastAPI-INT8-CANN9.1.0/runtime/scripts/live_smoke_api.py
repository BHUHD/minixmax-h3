#!/usr/bin/env python3
"""对运行中的 FastAPI 做 live HTTP 冒烟（mock 模式）。"""
from __future__ import annotations

import io
import json
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import httpx

ROOT = Path(__file__).resolve().parents[1]
PORT = int(os.environ.get("H3_API_PORT", "18081"))


def main() -> int:
    tmp = Path(tempfile.mkdtemp())
    env = os.environ.copy()
    env.update(
        {
            "H3_JOB_ROOT": str(tmp / "jobs"),
            "H3_API_MOCK": "1",
            "H3_API_PORT": str(PORT),
            "H3_API_HOST": "127.0.0.1",
            "H3_MOCK_STEP_DELAY": "0.02",
            "PYTHONPATH": str(ROOT),
        }
    )
    proc = subprocess.Popen(
        [sys.executable, str(ROOT / "scripts" / "run_api.py")],
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    base = f"http://127.0.0.1:{PORT}"
    try:
        for _ in range(50):
            try:
                r = httpx.get(f"{base}/health", timeout=1.0)
                if r.status_code == 200:
                    break
            except Exception:
                pass
            time.sleep(0.1)
        else:
            print("server not up", file=sys.stderr)
            return 1

        st = httpx.get(f"{base}/v1/status").json()
        assert st["service"] == "ready" and not st["busy"]

        files = []
        for i in range(9):
            files.append(("ref_images", (f"img_{i}.png", io.BytesIO(b"png"), "image/png")))
        for i in range(3):
            files.append(("ref_videos", (f"vid_{i}.mp4", io.BytesIO(b"vid"), "video/mp4")))
            files.append(("ref_audios", (f"aud_{i}.wav", io.BytesIO(b"aud"), "audio/wav")))

        r = httpx.post(
            f"{base}/v1/tasks",
            data={"prompt": "live smoke", "steps": "3"},
            files=files,
            timeout=30.0,
        )
        assert r.status_code == 202, r.text
        task_id = r.json()["task_id"]

        final = None
        for _ in range(100):
            q = httpx.get(f"{base}/v1/tasks/{task_id}").json()
            if q["status"] != "running":
                final = q
                break
            time.sleep(0.03)
        assert final and final["status"] == "succeeded"

        vid = httpx.get(f"{base}/v1/tasks/{task_id}/video")
        assert vid.status_code == 200 and len(vid.content) > 0

        busy = httpx.post(f"{base}/v1/tasks", data={"prompt": "x", "steps": "10"})
        # 若上一任务已结束应为 202；若在跑应为 409 —— 这里应已结束
        assert busy.status_code in (202, 409)

        print(json.dumps({"live_smoke": "ok", "task_id": task_id, "bytes": len(vid.content)}, indent=2))
        return 0
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            proc.kill()


if __name__ == "__main__":
    raise SystemExit(main())
