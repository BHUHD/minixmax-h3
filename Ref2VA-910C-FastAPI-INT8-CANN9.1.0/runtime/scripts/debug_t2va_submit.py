#!/usr/bin/env python3
"""无参考 t2va 对齐调试：提交任务并轮询结果。"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import httpx


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="http://127.0.0.1:8080")
    ap.add_argument("--prompt-file", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--label", default="t2va")
    ap.add_argument("--width", type=int, default=768)
    ap.add_argument("--height", type=int, default=544)
    ap.add_argument("--duration", type=float, default=5.0)
    ap.add_argument("--steps", type=int, default=20)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    prompt = args.prompt_file.read_text(encoding="utf-8").strip()
    base = args.base.rstrip("/")
    st = httpx.get(f"{base}/v1/status", timeout=10.0).json()
    print(f"[{args.label}] partition={st.get('partition')} ready={st.get('npu_serve_ready')}", flush=True)
    if not st.get("npu_serve_ready"):
        print("serve not ready", file=sys.stderr)
        return 2

    r = httpx.post(
        f"{base}/v1/tasks",
        data={
            "prompt": prompt,
            "task": "t2va",
            "width": str(args.width),
            "height": str(args.height),
            "duration": str(args.duration),
            "steps": str(args.steps),
            "seed": str(args.seed),
        },
        timeout=120.0,
    )
    r.raise_for_status()
    tid = r.json()["task_id"]
    print(f"[{args.label}] task_id={tid}", flush=True)

    while True:
        q = httpx.get(f"{base}/v1/tasks/{tid}", timeout=15.0).json()
        p = q.get("progress") or {}
        print(
            f"[{args.label}] {q['status']} {p.get('phase')} step={p.get('step')}/{p.get('steps_total')}",
            flush=True,
        )
        if q["status"] == "succeeded":
            break
        if q["status"] == "failed":
            print(q.get("error"), file=sys.stderr)
            return 1
        time.sleep(20)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    vid = httpx.get(f"{base}/v1/tasks/{tid}/video", timeout=300.0)
    vid.raise_for_status()
    args.out.write_bytes(vid.content)
    print(f"[{args.label}] saved {args.out} ({len(vid.content)} bytes)", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
