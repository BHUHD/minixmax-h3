#!/usr/bin/env python3
"""FastAPI Worker 全量模拟测试（无需 NPU / 无需重建镜像）。"""
from __future__ import annotations

import io
import json
import sys
import tempfile
import time
import unittest
from datetime import datetime, timezone
from pathlib import Path

RUNTIME_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RUNTIME_ROOT))

from fastapi.testclient import TestClient  # noqa: E402

from api.app import create_app  # noqa: E402
from api.config import Settings  # noqa: E402
from api.storage import JobStorage  # noqa: E402


def _settings_for(tmp: Path, *, mock: bool = True, step_delay: float = 0.01) -> Settings:
    return Settings(
        job_root=tmp / "jobs",
        serve_dir=tmp / "serve",
        mock_mode=mock,
        mock_step_delay=step_delay,
        mock_fail_task_id=None,
        task_timeout=30.0,
        host="127.0.0.1",
        port=8080,
        version="test",
        task_retention_days=7.0,
        cleanup_interval_sec=3600.0,
    )


def _client_ctx(tmp: Path, *, mock: bool = True, step_delay: float = 0.01) -> TestClient:
    return TestClient(create_app(_settings_for(tmp, mock=mock, step_delay=step_delay)))


def _fake_files(prefix: str, count: int, *, suffix: str, content: bytes = b"x") -> list[tuple]:
    return [
        (prefix, (f"{prefix}_{i}{suffix}", io.BytesIO(content * (i + 1)), "application/octet-stream"))
        for i in range(count)
    ]


class TestHealthAndStatus(unittest.TestCase):
    def test_health_and_status_idle(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            with _client_ctx(Path(td)) as client:
                self.assertEqual(client.get("/health").status_code, 200)
                s = client.get("/v1/status").json()
                self.assertEqual(s["service"], "ready")
                self.assertTrue(s["npu_serve_ready"])
                self.assertFalse(s["busy"])
                self.assertIsNone(s["current_task_id"])
                self.assertTrue(s["mock_mode"])


class TestSubmitAndDownload(unittest.TestCase):
    def test_gateway_flow_multimodal(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            with _client_ctx(Path(td)) as client:
                instance_id = client.get("/v1/status").json()["instance_id"]

                files = []
                files += _fake_files("ref_images", 9, suffix=".png")
                files += _fake_files("ref_videos", 3, suffix=".mp4", content=b"v")
                files += _fake_files("ref_audios", 3, suffix=".wav", content=b"a")

                data = {
                    "prompt": "test prompt",
                    "width": "1920",
                    "height": "1088",
                    "duration": "10",
                    "steps": "5",
                    "seed": "42",
                }
                r = client.post("/v1/tasks", data=data, files=files)
                self.assertEqual(r.status_code, 202, r.text)
                task_id = r.json()["task_id"]
                self.assertEqual(r.json()["instance_id"], instance_id)

                self.assertTrue(client.get("/v1/status").json()["busy"])

                final = None
                for _ in range(200):
                    q = client.get(f"/v1/tasks/{task_id}").json()
                    if q["status"] in ("succeeded", "failed"):
                        final = q
                        break
                    time.sleep(0.02)
                self.assertEqual(final["status"], "succeeded")
                self.assertTrue(final["output"]["ready"])
                self.assertEqual(final["input_counts"]["ref_images"], 9)
                self.assertEqual(final["input_counts"]["ref_videos"], 3)
                self.assertEqual(final["input_counts"]["ref_audios"], 3)

                vid = client.get(f"/v1/tasks/{task_id}/video")
                self.assertEqual(vid.status_code, 200)
                self.assertEqual(vid.headers["content-type"], "video/mp4")
                self.assertGreater(len(vid.content), 0)

                self.assertFalse(client.get("/v1/status").json()["busy"])


class TestSerialReject(unittest.TestCase):
    def test_busy_409(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            with _client_ctx(Path(td), step_delay=0.5) as client:
                r1 = client.post("/v1/tasks", data={"prompt": "slow", "steps": "20"})
                self.assertEqual(r1.status_code, 202)
                task_id = r1.json()["task_id"]

                r2 = client.post("/v1/tasks", data={"prompt": "second", "steps": "1"})
                self.assertEqual(r2.status_code, 409)
                self.assertEqual(r2.json()["detail"]["error"], "busy")
                self.assertEqual(r2.json()["detail"]["current_task_id"], task_id)


class TestFailure(unittest.TestCase):
    def test_mock_failure(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            with _client_ctx(Path(td)) as client:
                r = client.post("/v1/tasks", data={"prompt": "__mock_fail__", "steps": "2"})
                self.assertEqual(r.status_code, 202)
                task_id = r.json()["task_id"]

                final = None
                for _ in range(100):
                    q = client.get(f"/v1/tasks/{task_id}").json()
                    if q["status"] == "failed":
                        final = q
                        break
                    time.sleep(0.02)
                self.assertEqual(final["status"], "failed")
                self.assertIn("mock failure", final["error"])
                self.assertEqual(client.get(f"/v1/tasks/{task_id}/video").status_code, 409)


class TestNotFoundAndRestart(unittest.TestCase):
    def test_unknown_task_404(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            with _client_ctx(Path(td)) as client:
                self.assertEqual(
                    client.get("/v1/tasks/00000000-0000-0000-0000-000000000000").status_code,
                    404,
                )

    def test_restart_tombstone_410(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            job_root = root / "jobs"

            s1 = JobStorage(job_root)
            s1.startup_cleanup()
            rec = s1.create_task(
                prompt="before restart",
                params={"steps": 1},
                input_counts={"ref_images": 0, "ref_videos": 0, "ref_audios": 0},
            )
            old_task_id = rec.task_id

            s2 = JobStorage(job_root)
            s2.startup_cleanup()
            self.assertTrue(s2.is_tombstoned(old_task_id))

            with TestClient(create_app(_settings_for(root))) as client:
                r = client.get(f"/v1/tasks/{old_task_id}")
                self.assertEqual(r.status_code, 410)
                self.assertEqual(r.json()["detail"]["reason"], "container_restarted")


class TestStartupCleanup(unittest.TestCase):
    def test_cleanup_removes_mp4_and_tasks(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td) / "jobs"
            root.mkdir()
            (root / "orphan.mp4").write_bytes(b"mp4")
            task_dir = root / "old-task"
            task_dir.mkdir()
            (task_dir / "output.mp4").write_bytes(b"old")
            (root / ".instance.json").write_text(
                json.dumps({"instance_id": "prev", "task_ids": ["old-task-id"]})
            )

            s = JobStorage(root)
            s.startup_cleanup()

            self.assertFalse((root / "orphan.mp4").exists())
            self.assertFalse(task_dir.exists())
            self.assertTrue((root / ".tombstone" / "prev.json").exists())


class TestNotReady503(unittest.TestCase):
    def test_real_mode_not_ready(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            with _client_ctx(Path(td), mock=False) as client:
                self.assertEqual(client.post("/v1/tasks", data={"prompt": "x", "steps": "1"}).status_code, 503)


class TestInputPersistence(unittest.TestCase):
    def test_input_files_on_disk(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            tmp = Path(td)
            with _client_ctx(tmp) as client:
                files = _fake_files("ref_images", 2, suffix=".jpg")
                task_id = client.post(
                    "/v1/tasks",
                    data={"prompt": "disk", "steps": "1"},
                    files=files,
                ).json()["task_id"]
                for _ in range(50):
                    if client.get(f"/v1/tasks/{task_id}").json()["status"] != "running":
                        break
                    time.sleep(0.02)
                inp = tmp / "jobs" / task_id / "input" / "images"
                self.assertTrue(inp.is_dir())
                self.assertEqual(len(list(inp.iterdir())), 2)


class TestRetentionTTL(unittest.TestCase):
    def test_ttl_purge_7d(self):
        with tempfile.TemporaryDirectory() as td:
            from api.cleanup import TaskRetentionCleaner
            from api.storage import JobStorage, TaskRecord
            from datetime import timedelta

            root = Path(td) / "jobs"
            storage = JobStorage(root)
            storage.startup_cleanup()
            tid = "expire-me"
            tdir = root / tid
            tdir.mkdir(parents=True)
            old = (datetime.now(timezone.utc) - timedelta(days=8)).isoformat()
            rec = TaskRecord(
                task_id=tid,
                instance_id=storage.instance_id,
                status="succeeded",
                finished_at=old,
                output_ready=True,
            )
            storage.save_task(rec)
            (tdir / "output.mp4").write_bytes(b"mp4")
            cleaner = TaskRetentionCleaner(storage, retention_days=7, interval_sec=999)
            purged = cleaner.run_once()
            self.assertIn(tid, purged)
            self.assertFalse(tdir.exists())
            print("  [PASS] TTL purge >7d")


def run_tests() -> None:
    loader = unittest.TestLoader()
    suite = loader.loadTestsFromModule(sys.modules[__name__])
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    if not result.wasSuccessful():
        raise SystemExit(1)
    print("\n=== ALL SIMULATION TESTS PASSED ===")


if __name__ == "__main__":
    run_tests()
