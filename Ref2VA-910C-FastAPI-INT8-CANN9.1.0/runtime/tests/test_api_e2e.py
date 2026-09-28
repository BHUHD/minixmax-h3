#!/usr/bin/env python3
"""
FastAPI Worker 全功能实测（逐项 HTTP 调用，非仅冒烟）。

模式:
  dual_mock  H3_API_MOCK=0 + mock_serve（完整 mailbox 路径，无 NPU）
  api_mock   H3_API_MOCK=1（仅 FastAPI mock worker）

运行:
  cd Ref2VA/CANN-9.1.0/INI8/runtime
  python3 tests/test_api_e2e.py
  python3 tests/test_api_e2e.py --mode dual_mock
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
from pathlib import Path

import httpx

RUNTIME = Path(__file__).resolve().parents[1]
PORT = int(os.environ.get("H3_E2E_PORT", "18082"))


def _fake_files(prefix: str, n: int, suffix: str, content: bytes = b"x") -> list[tuple]:
    return [
        (prefix, (f"{prefix}_{i}{suffix}", io.BytesIO(content * (i + 1)), "application/octet-stream"))
        for i in range(n)
    ]


class WorkerServer:
    """启动双进程 entrypoint 或单 API 进程。"""

    def __init__(self, tmp: Path, *, mode: str = "dual_mock"):
        self.tmp = tmp
        self.mode = mode
        self.base = f"http://127.0.0.1:{PORT}"
        self.proc: subprocess.Popen | None = None
        self.env = os.environ.copy()
        self.env.update(
            {
                "WORKSPACE_ROOT": str(RUNTIME),
                "H3_JOB_ROOT": str(tmp / "jobs"),
                "H3_SERVE_DIR": str(tmp / "serve"),
                "H3_API_PORT": str(PORT),
                "H3_API_HOST": "127.0.0.1",
                "H3_MOCK_STEP_DELAY": "0.03",
                "H3_MOCK_SERVE_DELAY": "0.03",
                "PYTHONPATH": f"{RUNTIME}:{RUNTIME / 'api'}",
            }
        )
        if mode == "dual_mock":
            self.env["H3_API_MOCK"] = "0"
            self.env["H3_MOCK_SERVE"] = "1"
        else:
            self.env["H3_API_MOCK"] = "1"
            self.env["H3_MOCK_SERVE"] = "0"
        self.env.setdefault("WORKSPACE_ROOT", str(RUNTIME))

    def __enter__(self):
        cmd = ["bash", str(RUNTIME / "scripts" / "entrypoint_serve_api.sh")]
        if self.mode == "api_mock":
            cmd = [sys.executable, str(RUNTIME / "scripts" / "run_api.py")]
        self.proc = subprocess.Popen(
            cmd,
            env=self.env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
        )
        for _ in range(80):
            try:
                if httpx.get(f"{self.base}/health", timeout=1.0).status_code == 200:
                    st = httpx.get(f"{self.base}/v1/status", timeout=1.0).json()
                    if self.mode == "api_mock" or st.get("npu_serve_ready"):
                        return self
            except Exception:
                pass
            time.sleep(0.15)
        self._dump_log()
        raise RuntimeError("server failed to start")

    def __exit__(self, *_):
        if self.proc:
            self.proc.terminate()
            try:
                self.proc.wait(timeout=8)
            except subprocess.TimeoutExpired:
                self.proc.kill()

    def _dump_log(self) -> None:
        if self.proc and self.proc.stdout:
            print(self.proc.stdout.read(), file=sys.stderr)


class E2EBase(unittest.TestCase):
    mode = "dual_mock"

    def setUp(self):
        self._td = tempfile.TemporaryDirectory()
        self.tmp = Path(self._td.name)
        self.server = WorkerServer(self.tmp, mode=self.mode)
        self.server.__enter__()
        self.base = self.server.base
        self.client = httpx.Client(base_url=self.base, timeout=30.0)

    def tearDown(self):
        self.client.close()
        self.server.__exit__()
        self._td.cleanup()


class TestE2EHealth(E2EBase):
    def test_01_health(self):
        r = self.client.get("/health")
        self.assertEqual(r.status_code, 200)
        self.assertEqual(r.json()["status"], "ok")
        print("  [PASS] GET /health")


class TestE2EStatus(E2EBase):
    def test_02_status_ready_idle(self):
        s = self.client.get("/v1/status").json()
        self.assertEqual(s["service"], "ready")
        self.assertTrue(s["npu_serve_ready"])
        self.assertFalse(s["busy"])
        self.assertIsNotNone(s["instance_id"])
        print(f"  [PASS] GET /v1/status instance_id={s['instance_id'][:8]}...")


class TestE2ESubmitMultimodal(E2EBase):
    def test_03_submit_multimodal(self):
        files = []
        files += _fake_files("ref_images", 9, ".png")
        files += _fake_files("ref_videos", 3, ".mp4", b"v")
        files += _fake_files("ref_audios", 3, ".wav", b"a")
        r = self.client.post(
            "/v1/tasks",
            data={"prompt": "e2e multimodal", "steps": "4", "width": "1920", "height": "1088"},
            files=files,
        )
        self.assertEqual(r.status_code, 202)
        body = r.json()
        self.assertIn("task_id", body)
        self._task_id = body["task_id"]
        st = self.client.get("/v1/status").json()
        self.assertTrue(st["busy"])
        print(f"  [PASS] POST /v1/tasks -> {self._task_id}")


class TestE2EProgress(E2EBase):
    def test_04_progress_running_then_done(self):
        r = self.client.post("/v1/tasks", data={"prompt": "progress test", "steps": "5"})
        task_id = r.json()["task_id"]
        saw_running = False
        final = None
        for _ in range(150):
            q = self.client.get(f"/v1/tasks/{task_id}").json()
            if q["status"] == "running":
                saw_running = True
                self.assertIn("progress", q)
            if q["status"] in ("succeeded", "failed"):
                final = q
                break
            time.sleep(0.04)
        self.assertTrue(saw_running, "should see running")
        self.assertEqual(final["status"], "succeeded")
        self.assertTrue(final["output"]["ready"])
        print(f"  [PASS] GET /v1/tasks/{{id}} progress -> succeeded")


class TestE2EDownload(E2EBase):
    def test_05_download_video(self):
        task_id = self.client.post("/v1/tasks", data={"prompt": "dl", "steps": "2"}).json()["task_id"]
        for _ in range(100):
            if self.client.get(f"/v1/tasks/{task_id}").json()["status"] != "running":
                break
            time.sleep(0.04)
        vid = self.client.get(f"/v1/tasks/{task_id}/video")
        self.assertEqual(vid.status_code, 200)
        self.assertEqual(vid.headers.get("content-type"), "video/mp4")
        self.assertGreater(len(vid.content), 0)
        print(f"  [PASS] GET /v1/tasks/{{id}}/video ({len(vid.content)} bytes)")


class TestE2ESerialBusy(E2EBase):
    def test_06_busy_409(self):
        self.client.post("/v1/tasks", data={"prompt": "slow", "steps": "30"})
        time.sleep(0.05)
        r = self.client.post("/v1/tasks", data={"prompt": "second", "steps": "1"})
        self.assertEqual(r.status_code, 409)
        self.assertEqual(r.json()["detail"]["error"], "busy")
        print("  [PASS] POST while busy -> 409")


class TestE2EFailure(E2EBase):
    def test_07_task_failed(self):
        task_id = self.client.post(
            "/v1/tasks", data={"prompt": "__mock_fail__", "steps": "2"}
        ).json()["task_id"]
        final = None
        for _ in range(80):
            q = self.client.get(f"/v1/tasks/{task_id}").json()
            if q["status"] == "failed":
                final = q
                break
            time.sleep(0.04)
        self.assertIsNotNone(final)
        self.assertIn("error", final)
        self.assertEqual(self.client.get(f"/v1/tasks/{task_id}/video").status_code, 409)
        print(f"  [PASS] failed task error={final['error'][:40]}...")


class TestE2ENotFound(E2EBase):
    def test_08_unknown_404(self):
        r = self.client.get("/v1/tasks/00000000-0000-0000-0000-000000000000")
        self.assertEqual(r.status_code, 404)
        print("  [PASS] unknown task -> 404")


class TestE2ERestart410(E2EBase):
    def test_09_restart_410(self):
        # 第一次运行：提交并记住 task_id
        r1 = self.client.post("/v1/tasks", data={"prompt": "before restart", "steps": "2"})
        old_id = r1.json()["task_id"]
        old_instance = self.client.get("/v1/status").json()["instance_id"]
        for _ in range(80):
            if self.client.get(f"/v1/tasks/{old_id}").json()["status"] != "running":
                break
            time.sleep(0.04)

        # 模拟重启：停服再起
        self.client.close()
        self.server.__exit__()
        self.server = WorkerServer(self.tmp, mode=self.mode)
        self.server.__enter__()
        self.client = httpx.Client(base_url=self.server.base, timeout=30.0)

        new_instance = self.client.get("/v1/status").json()["instance_id"]
        self.assertNotEqual(old_instance, new_instance)

        r = self.client.get(f"/v1/tasks/{old_id}")
        self.assertEqual(r.status_code, 410)
        self.assertEqual(r.json()["detail"]["reason"], "container_restarted")
        print("  [PASS] after restart old task -> 410 container_restarted")


class TestE2EStartupCleanup(E2EBase):
    def test_10_startup_cleanup(self):
        jobs = self.tmp / "jobs"
        # 重启后 orphan 应被清
        self.client.close()
        self.server.__exit__()
        (jobs / "orphan.mp4").write_bytes(b"x")
        self.server = WorkerServer(self.tmp, mode=self.mode)
        self.server.__enter__()
        self.assertFalse((jobs / "orphan.mp4").exists())
        print("  [PASS] startup clears orphan mp4")


class TestE2EGatewayClient(E2EBase):
    def test_11_gateway_client_flow(self):
        sys.path.insert(0, str(RUNTIME / "examples"))
        from gateway_client import GatewayClient  # noqa: E402

        tmp_out = self.tmp / "gw_out.mp4"
        # 准备参考图
        imgs = []
        for i in range(3):
            p = self.tmp / f"ref_{i}.png"
            p.write_bytes(b"png")
            imgs.append(p)

        gc = GatewayClient(self.base, poll_interval=0.05, timeout=60.0)
        st = gc.wait_idle(max_wait=30.0)
        self.assertFalse(st["busy"])

        result = gc.generate_and_download(
            tmp_out,
            prompt="gateway client e2e",
            ref_images=imgs,
            steps=3,
        )
        self.assertTrue(tmp_out.is_file())
        self.assertGreater(tmp_out.stat().st_size, 0)
        self.assertEqual(result["task"]["status"], "succeeded")
        print(f"  [PASS] gateway_client.py full flow -> {tmp_out}")


class TestE2EApiMockMode(TestE2EHealth, TestE2EStatus, TestE2EDownload):
    mode = "api_mock"


class TestE2ENotReady503(unittest.TestCase):
    def test_12_not_ready_503(self):
        with tempfile.TemporaryDirectory() as td:
            tmp = Path(td)
            env = os.environ.copy()
            env.update(
                {
                    "H3_JOB_ROOT": str(tmp / "jobs"),
                    "H3_SERVE_DIR": str(tmp / "serve"),
                    "H3_API_PORT": str(PORT + 1),
                    "H3_API_HOST": "127.0.0.1",
                    "H3_API_MOCK": "0",
                    "H3_MOCK_SERVE": "0",
                    "PYTHONPATH": f"{RUNTIME}:{RUNTIME / 'api'}",
                }
            )
            proc = subprocess.Popen(
                [sys.executable, str(RUNTIME / "scripts" / "run_api.py")],
                env=env,
            )
            base = f"http://127.0.0.1:{PORT + 1}"
            try:
                for _ in range(30):
                    try:
                        if httpx.get(f"{base}/health", timeout=1.0).status_code == 200:
                            break
                    except Exception:
                        pass
                    time.sleep(0.1)
                r = httpx.post(f"{base}/v1/tasks", data={"prompt": "x", "steps": "1"})
                self.assertEqual(r.status_code, 503)
                print("  [PASS] real mode without serve -> 503")
            finally:
                proc.terminate()
                proc.wait(timeout=5)


def run(mode: str = "dual_mock") -> None:
    E2EBase.mode = mode

    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    mod = sys.modules[__name__]
    for name in [
        "TestE2EHealth",
        "TestE2EStatus",
        "TestE2ESubmitMultimodal",
        "TestE2EProgress",
        "TestE2EDownload",
        "TestE2ESerialBusy",
        "TestE2EFailure",
        "TestE2ENotFound",
        "TestE2ERestart410",
        "TestE2EStartupCleanup",
        "TestE2EGatewayClient",
        "TestE2ENotReady503",
    ]:
        suite.addTests(loader.loadTestsFromName(name, module=mod))

    print(f"\n=== E2E 实测 mode={mode} ===")
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    if not result.wasSuccessful():
        raise SystemExit(1)
    print("\n=== ALL E2E TESTS PASSED ===")


if __name__ == "__main__":
    mode = "dual_mock"
    if len(sys.argv) > 1 and sys.argv[1] == "--mode":
        mode = sys.argv[2]
    # 设置 E2EBase 默认 mode
    E2EBase.mode = mode
    run(mode)
