#!/usr/bin/env python3
"""Submit a generate job to a resident H3 serve group (stdlib only)."""
from __future__ import annotations

import json
import os
import time
from pathlib import Path


def main() -> None:
    # 容器内默认邮箱；宿主机调试时也可通过 H3_SERVE_DIR 覆盖
    root = Path(os.environ.get("H3_SERVE_DIR", "/workspace/out/h3_serve"))
    root.mkdir(parents=True, exist_ok=True)
    ready = root / "READY"
    t0 = time.time()
    timeout = float(os.environ.get("H3_SUBMIT_WAIT_READY", "1800"))
    print(f"[submit] waiting for {ready}", flush=True)
    while time.time() - t0 < timeout:
        if ready.is_file():
            break
        time.sleep(0.5)
    else:
        raise SystemExit("serve not READY")
    jid = os.environ.get("H3_JOB_ID") or str(int(time.time() * 1000))
    job = {
        "id": jid,
        # Eight real DiT evaluations are the validated latency tier for the
        # resident 16-die deployment.  Keep the one-shot compose job at 20
        # steps for its quality baseline; callers can always request it here
        # with H3_STEPS=20.
        "steps": int(os.environ.get("H3_STEPS", "8")),
        "seconds": float(os.environ.get("H3_SECONDS", "10")),
        "height": int(os.environ.get("H3_HEIGHT", "1088")),
        "width": int(os.environ.get("H3_WIDTH", "1920")),
        "out": os.environ.get("H3_OUT", "/workspace/out/h3_quant_generate.mp4"),
    }
    if os.environ.get("H3_PROMPT"):
        job["prompt"] = os.environ["H3_PROMPT"]
    done = root / "DONE"
    if done.exists():
        done.unlink()
    (root / "job.json").write_text(json.dumps(job))
    print(f"[submit] job {jid} {job}", flush=True)
    t1 = time.time()
    while time.time() - t1 < float(os.environ.get("H3_SUBMIT_TIMEOUT", "1800")):
        if done.is_file() and done.read_text().strip() == jid:
            print(f"[submit] DONE {jid} in {time.time()-t1:.1f}s", flush=True)
            return
        time.sleep(0.5)
    raise SystemExit(f"timeout waiting for job {jid}")


if __name__ == "__main__":
    main()
