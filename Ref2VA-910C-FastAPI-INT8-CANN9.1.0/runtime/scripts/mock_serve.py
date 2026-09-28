#!/usr/bin/env python3
"""模拟 h3_serve 邮箱（无 NPU），用于双进程 entrypoint 联调实测。"""
from __future__ import annotations

import json
import os
import struct
import time
from pathlib import Path


def _minimal_mp4() -> bytes:
    body = b"isomiso2mp41"
    return struct.pack(">I", 8 + len(body)) + b"ftyp" + body


def main() -> None:
    serve_dir = Path(os.environ.get("H3_SERVE_DIR", "/workspace/out/h3_serve"))
    delay = float(os.environ.get("H3_MOCK_SERVE_DELAY", "0.05"))
    fail_prompt = os.environ.get("H3_MOCK_SERVE_FAIL_PROMPT", "__mock_fail__")

    serve_dir.mkdir(parents=True, exist_ok=True)
    for p in serve_dir.glob("*"):
        if p.is_file():
            p.unlink()

    # 模拟 16 rank ready
    for i in range(int(os.environ.get("H3_NPROC", "16"))):
        (serve_dir / f"ready.{i}").write_text("ok")
    (serve_dir / "READY").write_text("ok")
    print(f"[mock_serve] READY at {serve_dir}", flush=True)

    last = ""
    while True:
        jobp = serve_dir / "job.json"
        if not jobp.is_file():
            time.sleep(0.1)
            continue
        try:
            job = json.loads(jobp.read_text())
        except Exception:
            time.sleep(0.1)
            continue
        jid = str(job.get("id") or "")
        if not jid or jid == last:
            time.sleep(0.1)
            continue
        last = jid
        print(f"[mock_serve] job {jid}", flush=True)

        steps = int(job.get("steps", 20))
        out = Path(job.get("out", serve_dir / "out.mp4"))
        out.parent.mkdir(parents=True, exist_ok=True)

        if str(job.get("prompt", "")).strip() == fail_prompt:
            print(f"[mock_serve] fail job {jid}", flush=True)
            (serve_dir / "FAILED").write_text(jid)
            time.sleep(delay)
            continue

        for i in range(1, steps + 1):
            prog = {
                "phase": "dit" if i < steps else "vae",
                "step": i,
                "steps_total": steps,
                "percent": int(i / steps * 100),
            }
            (out.parent / "progress.json").write_text(json.dumps(prog))
            time.sleep(delay)

        out.write_bytes(_minimal_mp4())
        nproc = int(os.environ.get("H3_NPROC", "16"))
        for r in range(nproc):
            (serve_dir / f"done.{r}").write_text(jid)
        (serve_dir / "DONE").write_text(jid)
        print(f"[mock_serve] DONE {jid} -> {out}", flush=True)


if __name__ == "__main__":
    main()
