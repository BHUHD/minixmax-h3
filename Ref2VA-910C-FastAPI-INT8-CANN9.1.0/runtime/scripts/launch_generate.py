#!/usr/bin/env python3
"""Spawn 8 generate ranks without importing torch in the parent.

torchrun / `python -c 'import torch_npu'` with 8 visible dies initializes every
chip in the parent, then workers H2D-deadlock on davinci_manager. This launcher
only uses stdlib and sets one ASCEND_RT_VISIBLE_DEVICES per child.
"""
from __future__ import annotations

import os
import signal
import subprocess
import sys
import time


def main() -> None:
    phy = [p.strip() for p in os.environ.get("H3_PHY_DEVICES", "0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15").split(",") if p.strip()]
    if len(phy) < 1:
        raise SystemExit("H3_PHY_DEVICES is empty")
    n = int(os.environ.get("H3_NPROC", str(len(phy))))
    phy = phy[:n]
    n = len(phy)
    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    os.environ.setdefault("MASTER_PORT", "29521")

    src = "/workspace/src"
    py = os.environ.get("PYTHONPATH", "")
    worker = os.environ.get("H3_WORKER_MODULE", "h3_npu.pipeline.generate_worker")
    extra = [sys.executable, "-u", "-m", worker, *sys.argv[1:]]

    procs: list[subprocess.Popen] = []

    def _kill_all(*_):
        for p in procs:
            if p.poll() is None:
                p.terminate()
        time.sleep(2)
        for p in procs:
            if p.poll() is None:
                p.kill()

    signal.signal(signal.SIGTERM, _kill_all)
    signal.signal(signal.SIGINT, _kill_all)

    from pathlib import Path

    sync = Path("/workspace/out/h3_sync")
    if sync.exists():
        import shutil

        shutil.rmtree(sync, ignore_errors=True)
    sync.mkdir(parents=True, exist_ok=True)

    # A resident group is deliberately long-lived, but its filesystem mailbox
    # is not.  Leaving a completed job.json/DONE behind lets a freshly started
    # server accidentally execute yesterday's request before it reports READY.
    # The directory is a launcher-owned, explicitly scoped control plane (not
    # an output directory), so it is safe to recreate for each service start.
    if os.environ.get("H3_SERVE", "0") == "1":
        serve = Path(os.environ.get("H3_SERVE_DIR", "/workspace/out/h3_serve"))
        if serve.exists():
            import shutil

            shutil.rmtree(serve)
        serve.mkdir(parents=True, exist_ok=True)

    job_t0 = str(time.time())
    stagger = float(os.environ.get("H3_SPAWN_STAGGER", "0.08"))
    print(f"[launch] spawning {n} ranks on phy {','.join(phy)} stagger={stagger:.2f}s (no torch in parent)", flush=True)
    for i, die in enumerate(phy):
        if i and stagger > 0:
            time.sleep(stagger)
        env = os.environ.copy()
        env["RANK"] = str(i)
        env["LOCAL_RANK"] = str(i)
        env["WORLD_SIZE"] = str(n)
        env["H3_JOB_T0"] = job_t0
        env["ASCEND_RT_VISIBLE_DEVICES"] = die
        env["ASCEND_VISIBLE_DEVICES"] = die
        env["ASCEND_DEVICE_ID"] = "0"
        env["PYTHONPATH"] = f"{src}:{py}" if py else src
        env["PYTHONUNBUFFERED"] = "1"
        procs.append(subprocess.Popen(extra, env=env))

    codes = [p.wait() for p in procs]
    bad = [c for c in codes if c != 0]
    if bad:
        _kill_all()
        raise SystemExit(max(abs(c) for c in codes))


if __name__ == "__main__":
    main()
