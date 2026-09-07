#!/usr/bin/env python3
"""
真实 NPU 容器 API 全功能实测。

前提：容器已通过 docker compose 启动，FastAPI 在 H3_API_PORT（默认 8080）。

用法:
  python3 tests/test_api_e2e_npu.py
  H3_WORKER_BASE=http://127.0.0.1:8080 python3 tests/test_api_e2e_npu.py
"""
from __future__ import annotations

import io
import json
import os
import subprocess
import sys
import tempfile
import time
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path

import httpx

RUNTIME = Path(__file__).resolve().parents[1]
BASE = os.environ.get("H3_WORKER_BASE", "http://127.0.0.1:8080")
CONTAINER = os.environ.get("H3_CONTAINER", "minimax-h3-ref2va-int8-npu16")
# 真实推理耗时长：功能测用 2 step；完整生成单独 1 项
FUNC_STEPS = int(os.environ.get("H3_NPU_TEST_STEPS", "2"))
FULL_STEPS = int(os.environ.get("H3_NPU_FULL_STEPS", "20"))
POLL = float(os.environ.get("H3_NPU_POLL", "5.0"))
TASK_TIMEOUT = float(os.environ.get("H3_NPU_TASK_TIMEOUT", "7200"))


def _fake_files(prefix: str, n: int, suffix: str, content: bytes = b"x") -> list[tuple]:
    return [
        (prefix, (f"{prefix}_{i}{suffix}", io.BytesIO(content * (i + 1)), "application/octet-stream"))
        for i in range(n)
    ]


def _ref_image_path() -> Path:
    p = RUNTIME.parent / "debug" / "input" / "ref_golden_retriever.png"
    if p.is_file():
        return p
    # fallback 1x1 png
    tmp = Path(tempfile.gettempdir()) / "h3_ref_test.png"
    tmp.write_bytes(
        bytes.fromhex(
            "89504e470d0a1a0a0000000d49484452000000010000000108060000001f15c489"
            "0000000a49444154789c6300010000050001000d0a2db40000000049454e44ae426082"
        )
    )
    return tmp


class NPURealBase(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.client = httpx.Client(base_url=BASE, timeout=120.0)
        # 等待 API + NPU serve 就绪
        t0 = time.time()
        while time.time() - t0 < 1800:
            try:
                h = cls.client.get("/health")
                if h.status_code != 200:
                    time.sleep(5)
                    continue
                s = cls.client.get("/v1/status").json()
                if s.get("npu_serve_ready") and s.get("service") == "ready":
                    cls.instance_id = s["instance_id"]
                    print(f"\n[NPU] ready instance_id={cls.instance_id[:8]}... mock={s.get('mock_mode')}")
                    return
            except Exception:
                pass
            time.sleep(5)
        raise unittest.SkipTest("NPU worker not ready within 1800s")

    @classmethod
    def tearDownClass(cls):
        cls.client.close()

    def _wait_task(self, task_id: str, *, timeout: float = TASK_TIMEOUT) -> dict:
        t0 = time.time()
        while time.time() - t0 < timeout:
            r = self.client.get(f"/v1/tasks/{task_id}")
            self.assertIn(r.status_code, (200, 410))
            if r.status_code == 410:
                self.fail(f"task {task_id} lost: {r.json()}")
            data = r.json()
            if data["status"] in ("succeeded", "failed"):
                return data
            time.sleep(POLL)
        self.fail(f"task {task_id} timeout after {timeout}s")

    def _submit_minimal(self, prompt: str = "npu api test", steps: int | None = None) -> str:
        steps = steps or FUNC_STEPS
        ref = _ref_image_path()
        files = [("ref_images", (ref.name, ref.read_bytes(), "image/png"))]
        r = self.client.post(
            "/v1/tasks",
            data={
                "prompt": prompt if "<Picture" in prompt else f"Use <Picture 1> as the character. {prompt}",
                "steps": str(steps),
                "width": "768",
                "height": "768",
                "duration": "2",
            },
            files=files,
        )
        self.assertEqual(r.status_code, 202, r.text)
        return r.json()["task_id"]


class Test01Health(NPURealBase):
    def test_health(self):
        self.assertEqual(self.client.get("/health").json()["status"], "ok")
        print("  [PASS] GET /health")


class Test02Status(NPURealBase):
    def test_status(self):
        s = self.client.get("/v1/status").json()
        self.assertTrue(s["npu_serve_ready"])
        self.assertFalse(s["mock_mode"])
        self.assertIn("instance_id", s)
        print(f"  [PASS] GET /v1/status busy={s['busy']}")


class Test03Multimodal(NPURealBase):
    def test_submit_multimodal(self):
        # 真实 NPU：参考图用 golden；视频/音频仅验证 API 多文件受理（pipeline 可忽略）
        ref = _ref_image_path()
        img_bytes = ref.read_bytes()
        files = []
        for i in range(9):
            files.append(("ref_images", (f"ref_{i}.png", img_bytes, "image/png")))
        files += _fake_files("ref_videos", 3, ".mp4", b"v")
        files += _fake_files("ref_audios", 3, ".wav", b"a")
        r = self.client.post(
            "/v1/tasks",
            data={
                "prompt": "Use <Picture 1> as the character. A golden retriever on beach",
                "steps": str(FUNC_STEPS),
                "width": "768",
                "height": "768",
                "duration": "2",
            },
            files=files,
        )
        self.assertEqual(r.status_code, 202, r.text)
        task_id = r.json()["task_id"]
        final = self._wait_task(task_id)
        self.assertEqual(final["status"], "succeeded", final.get("error"))
        self.assertEqual(final["input_counts"]["ref_images"], 9)
        self.assertEqual(final["input_counts"]["ref_videos"], 3)
        self.assertEqual(final["input_counts"]["ref_audios"], 3)
        print(f"  [PASS] multimodal submit+done {task_id[:8]}...")


class Test04ProgressDownload(NPURealBase):
    def test_progress_and_download(self):
        task_id = self._submit_minimal("progress dl test")
        saw_running = False
        t0 = time.time()
        while time.time() - t0 < TASK_TIMEOUT:
            q = self.client.get(f"/v1/tasks/{task_id}").json()
            if q["status"] == "running":
                saw_running = True
            if q["status"] == "succeeded":
                break
            time.sleep(POLL)
        self.assertTrue(saw_running)
        vid = self.client.get(f"/v1/tasks/{task_id}/video")
        self.assertEqual(vid.status_code, 200)
        self.assertGreater(len(vid.content), 1000)
        print(f"  [PASS] progress+download {len(vid.content)} bytes")


class Test05Busy409(NPURealBase):
    def test_busy_reject(self):
        # 等空闲后再测串行拒单
        t0 = time.time()
        while time.time() - t0 < 600:
            if not self.client.get("/v1/status").json().get("busy"):
                break
            time.sleep(2)
        ref = _ref_image_path()
        files = [("ref_images", (ref.name, ref.read_bytes(), "image/png"))]
        r1 = self.client.post(
            "/v1/tasks",
            data={
                "prompt": "Use <Picture 1>. busy test",
                "steps": str(max(FUNC_STEPS, 4)),
                "width": "768",
                "height": "768",
                "duration": "2",
            },
            files=files,
        )
        self.assertEqual(r1.status_code, 202, r1.text)
        time.sleep(1)
        r2 = self.client.post("/v1/tasks", data={"prompt": "second", "steps": "1"})
        self.assertEqual(r2.status_code, 409, r2.text)
        self.assertEqual(r2.json()["detail"]["error"], "busy")
        print("  [PASS] busy -> 409")
        self._wait_task(r1.json()["task_id"])


class Test06NotFound404(NPURealBase):
    def test_unknown_404(self):
        self.assertEqual(
            self.client.get("/v1/tasks/00000000-0000-0000-0000-000000000000").status_code,
            404,
        )
        print("  [PASS] unknown -> 404")


class Test07Restart410(NPURealBase):
    def test_restart_410(self):
        task_id = self._submit_minimal("before restart")
        self._wait_task(task_id)
        old_instance = self.instance_id

        subprocess.run(["docker", "restart", CONTAINER], check=True, capture_output=True)
        # 等待重启后就绪
        t0 = time.time()
        new_instance = None
        while time.time() - t0 < 1800:
            try:
                s = httpx.get(f"{BASE}/v1/status", timeout=10.0).json()
                if s.get("npu_serve_ready"):
                    new_instance = s["instance_id"]
                    if new_instance != old_instance:
                        break
            except Exception:
                pass
            time.sleep(10)
        self.assertIsNotNone(new_instance)
        self.assertNotEqual(old_instance, new_instance)

        r = httpx.get(f"{BASE}/v1/tasks/{task_id}", timeout=10.0)
        self.assertEqual(r.status_code, 410)
        self.assertEqual(r.json()["detail"]["reason"], "container_restarted")
        print("  [PASS] restart -> 410 container_restarted")


class Test08Retention7d(NPURealBase):
    def test_retention_config_and_purge(self):
        # 检查容器 env
        out = subprocess.run(
            ["docker", "exec", CONTAINER, "printenv", "H3_TASK_RETENTION_DAYS"],
            capture_output=True,
            text=True,
        )
        self.assertEqual(out.stdout.strip(), "7")
        print("  [PASS] H3_TASK_RETENTION_DAYS=7")

        # 在容器内模拟过期任务并触发 cleanup
        script = r"""
import json, os, sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
sys.path.insert(0, '/workspace/api')
sys.path.insert(0, '/workspace')
from api.storage import JobStorage, TaskRecord
from api.cleanup import TaskRetentionCleaner

root = Path(os.environ.get('H3_JOB_ROOT', '/workspace/jobs'))
storage = JobStorage(root)
storage.instance_id = 'ttl-test'
tid = 'ttl-expire-test-id-0001'
tdir = root / tid
tdir.mkdir(parents=True, exist_ok=True)
old = (datetime.now(timezone.utc) - timedelta(days=8)).isoformat()
rec = TaskRecord(
    task_id=tid, instance_id='ttl-test', status='succeeded',
    finished_at=old, output_ready=True, output_size=100,
)
storage.save_task(rec)
(tdir / 'output.mp4').write_bytes(b'mp4')
cleaner = TaskRetentionCleaner(storage, retention_days=7, interval_sec=9999)
purged = cleaner.run_once()
print(json.dumps({'purged': purged}))
"""
        out2 = subprocess.run(
            ["docker", "exec", CONTAINER, "python3", "-c", script],
            capture_output=True,
            text=True,
        )
        self.assertEqual(out2.returncode, 0, out2.stderr + out2.stdout)
        result = json.loads(out2.stdout.strip().splitlines()[-1])
        self.assertIn("ttl-expire-test-id-0001", result["purged"])
        print("  [PASS] TTL purge expired task (>7d)")


class Test09GatewayClient(NPURealBase):
    def test_gateway_flow(self):
        sys.path.insert(0, str(RUNTIME / "examples"))
        from gateway_client import GatewayClient  # noqa: E402

        out = Path(tempfile.gettempdir()) / "h3_npu_gateway_out.mp4"
        gc = GatewayClient(BASE, poll_interval=POLL, timeout=TASK_TIMEOUT)
        gc.wait_idle(max_wait=1800)
        ref = _ref_image_path()
        result = gc.generate_and_download(
            out,
            prompt="Use <Picture 1> as reference. A golden retriever on beach",
            ref_images=[ref],
            steps=FUNC_STEPS,
            width=768,
            height=768,
            duration=2.0,
        )
        self.assertTrue(out.is_file())
        self.assertGreater(out.stat().st_size, 1000)
        self.assertEqual(result["task"]["status"], "succeeded")
        print(f"  [PASS] gateway_client -> {out} ({out.stat().st_size} bytes)")


class Test10FullGeneration(NPURealBase):
    def test_full_1080p_steps20(self):
        """可选完整生成（默认 20 step 1080P，耗时长）。"""
        if os.environ.get("H3_SKIP_FULL_GEN") == "1":
            self.skipTest("H3_SKIP_FULL_GEN=1")
        ref = _ref_image_path()
        files = [("ref_images", (ref.name, ref.read_bytes(), "image/png"))]
        r = self.client.post(
            "/v1/tasks",
            data={
                "prompt": "Use <Picture 1>. Golden retriever on sunny beach, cinematic",
                "steps": str(FULL_STEPS),
                "width": "1920",
                "height": "1088",
                "duration": "10",
            },
            files=files,
        )
        self.assertEqual(r.status_code, 202)
        task_id = r.json()["task_id"]
        print(f"  [INFO] full gen task_id={task_id} steps={FULL_STEPS} ...")
        final = self._wait_task(task_id, timeout=TASK_TIMEOUT)
        self.assertEqual(final["status"], "succeeded")
        vid = self.client.get(f"/v1/tasks/{task_id}/video")
        self.assertGreater(len(vid.content), 10000)
        print(f"  [PASS] full 1080p gen {len(vid.content)} bytes")


def main() -> None:
    print(f"\n=== NPU Real E2E base={BASE} container={CONTAINER} ===")
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    mod = sys.modules[__name__]
    for name in [
        "Test01Health",
        "Test02Status",
        "Test03Multimodal",
        "Test04ProgressDownload",
        "Test05Busy409",
        "Test06NotFound404",
        "Test08Retention7d",
        "Test09GatewayClient",
        "Test10FullGeneration",
        "Test07Restart410",
    ]:
        suite.addTests(loader.loadTestsFromName(name, module=mod))
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    if not result.wasSuccessful():
        raise SystemExit(1)
    print("\n=== ALL NPU REAL E2E PASSED ===")


if __name__ == "__main__":
    main()
