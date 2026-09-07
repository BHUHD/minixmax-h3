#!/usr/bin/env python3
"""提交 10s 1080P 生成，并实时打印进度（供人工盯盘提效）。

用法:
  python3 scripts/run_1080p10s_live.py
  python3 scripts/run_1080p10s_live.py --base http://127.0.0.1:8080 --steps 20
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import httpx

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_REF = ROOT / "debug" / "input" / "ref_golden_retriever.png"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="http://127.0.0.1:8080")
    ap.add_argument("--steps", type=int, default=20)
    ap.add_argument("--width", type=int, default=1920)
    ap.add_argument("--height", type=int, default=1088)
    ap.add_argument("--duration", type=float, default=10.0)
    ap.add_argument("--poll", type=float, default=2.0)
    ap.add_argument("--out", type=Path, default=Path("/tmp/h3_1080p10s_live.mp4"))
    ap.add_argument("--ref", type=Path, default=DEFAULT_REF)
    ap.add_argument(
        "--prompt",
        default="Use <Picture 1> as the character. A golden retriever running along a sunny beach, waves in the background, cinematic lighting",
    )
    args = ap.parse_args()
    base = args.base.rstrip("/")

    print(f"[live] wait idle @ {base}", flush=True)
    t0 = time.time()
    while True:
        st = httpx.get(f"{base}/v1/status", timeout=10.0).json()
        if st.get("npu_serve_ready") and not st.get("busy"):
            print(f"[live] ready instance={st['instance_id'][:8]}... uptime={st['uptime_sec']}s", flush=True)
            break
        if time.time() - t0 > 1800:
            print("[live] timeout waiting ready/idle", file=sys.stderr)
            return 1
        print(f"[live] waiting ready={st.get('npu_serve_ready')} busy={st.get('busy')}", flush=True)
        time.sleep(5)

    if not args.ref.is_file():
        print(f"[live] missing ref: {args.ref}", file=sys.stderr)
        return 1

    files = [("ref_images", (args.ref.name, args.ref.read_bytes(), "image/png"))]
    data = {
        "prompt": args.prompt,
        "steps": str(args.steps),
        "width": str(args.width),
        "height": str(args.height),
        "duration": str(args.duration),
    }
    print(f"[live] submit {args.width}x{args.height} {args.duration}s steps={args.steps}", flush=True)
    r = httpx.post(f"{base}/v1/tasks", data=data, files=files, timeout=120.0)
    if r.status_code != 202:
        print(f"[live] submit failed {r.status_code}: {r.text}", file=sys.stderr)
        return 1
    task_id = r.json()["task_id"]
    print(f"[live] task_id={task_id}", flush=True)

    last_line = ""
    t_job = time.time()
    while True:
        info = httpx.get(f"{base}/v1/tasks/{task_id}", timeout=10.0).json()
        prog = info.get("progress") or {}
        line = (
            f"[progress] status={info['status']} phase={prog.get('phase')} "
            f"step={prog.get('step')}/{prog.get('steps_total')} "
            f"percent={prog.get('percent')} "
            f"elapsed={time.time()-t_job:.0f}s"
        )
        if line != last_line:
            print(line, flush=True)
            last_line = line
        if info["status"] == "succeeded":
            break
        if info["status"] == "failed":
            print(f"[live] FAILED: {info.get('error')}", file=sys.stderr)
            return 2
        time.sleep(args.poll)

    vid = httpx.get(f"{base}/v1/tasks/{task_id}/video", timeout=120.0)
    if vid.status_code != 200:
        print(f"[live] download failed {vid.status_code}", file=sys.stderr)
        return 3
    args.out.write_bytes(vid.content)
    print(
        f"[live] DONE saved {args.out} ({len(vid.content)} bytes) "
        f"wall={time.time()-t_job:.1f}s",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
