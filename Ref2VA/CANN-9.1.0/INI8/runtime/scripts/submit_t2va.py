#!/usr/bin/env python3
"""提交无参考纯文生视频（Ref2VA t2va）到 serve（默认 8080）。"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import httpx


def main() -> int:
    ap = argparse.ArgumentParser(description="MiniMax-H3 Ref2VA text-only smoke test")
    ap.add_argument("--base", default="http://127.0.0.1:8080", help="API base URL")
    ap.add_argument("--prompt", required=True)
    ap.add_argument("--width", type=int, default=1920)
    ap.add_argument("--height", type=int, default=1088)
    ap.add_argument("--duration", type=float, default=10.0)
    ap.add_argument("--steps", type=int, default=20)
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--out", type=Path, default=Path("tmp/t2va_output.mp4"))
    args = ap.parse_args()

    base = args.base.rstrip("/")
    st = httpx.get(f"{base}/v1/status", timeout=10.0)
    st.raise_for_status()
    info = st.json()
    print(f"[t2va] status partition={info.get('partition')} ready={info.get('npu_serve_ready')}", flush=True)
    if not info.get("npu_serve_ready"):
        print("[t2va] serve 未 READY", file=sys.stderr)
        return 2

    data = {
        "prompt": args.prompt,
        "task": "t2va",
        "width": str(args.width),
        "height": str(args.height),
        "duration": str(args.duration),
        "steps": str(args.steps),
    }
    if args.seed is not None:
        data["seed"] = str(args.seed)

    r = httpx.post(f"{base}/v1/tasks", data=data, timeout=120.0)
    if r.status_code >= 400:
        print(r.text, file=sys.stderr)
        r.raise_for_status()
    task_id = r.json()["task_id"]
    print(f"[t2va] task_id={task_id}", flush=True)

    while True:
        q = httpx.get(f"{base}/v1/tasks/{task_id}", timeout=15.0).json()
        prog = q.get("progress") or {}
        print(
            f"[t2va] {q.get('status')} phase={prog.get('phase')} "
            f"step={prog.get('step')}/{prog.get('steps_total')} pct={prog.get('percent')}",
            flush=True,
        )
        if q.get("status") == "succeeded":
            break
        if q.get("status") == "failed":
            print(q.get("error"), file=sys.stderr)
            return 1
        time.sleep(15)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    vid = httpx.get(f"{base}/v1/tasks/{task_id}/video", timeout=300.0)
    vid.raise_for_status()
    args.out.write_bytes(vid.content)
    print(f"[t2va] saved {args.out} ({len(vid.content)} bytes)", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
