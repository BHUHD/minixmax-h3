#!/usr/bin/env python3
"""宿主机入口：向 INT8 16-die serve 投递生成任务（docker compose exec）。

在 Ref2VA/CANN-9.1.0/INI8 目录下：

  docker compose -f docker-compose.yml up -d
  # 等待 debug/out/h3_serve/READY
  ./scripts/submit_generate.py
"""
from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CNAME = os.environ.get("CNAME", "minimax-h3-ref2va-int8-npu16")
SERVICE = os.environ.get("H3_COMPOSE_SERVICE", "ref2va-int8-npu16")
COMPOSE = ROOT / "docker-compose.yml"
READY = ROOT / "debug" / "out" / "h3_serve" / "READY"

DEFAULT_PROMPT = (
    "Use <Picture 1> as the character. A golden retriever running along a sunny beach, "
    "waves in the background, cinematic lighting"
)


def main() -> int:
    # 容器是否在跑
    ps = subprocess.run(
        ["docker", "ps", "--format", "{{.Names}}"],
        check=False,
        capture_output=True,
        text=True,
    )
    names = set(ps.stdout.splitlines())
    if CNAME not in names:
        print(f"容器未运行: {CNAME}", file=sys.stderr)
        print("请先执行: docker compose -f docker-compose.yml up -d", file=sys.stderr)
        return 1
    if not READY.is_file():
        print(f"serve 尚未 READY（等待 {READY}）", file=sys.stderr)
        print(f"可查看: docker logs -f {CNAME}", file=sys.stderr)
        return 1

    env_pass = {
        "H3_STEPS": os.environ.get("H3_STEPS", "20"),
        "H3_SECONDS": os.environ.get("H3_SECONDS", "10"),
        "H3_HEIGHT": os.environ.get("H3_HEIGHT", "1088"),
        "H3_WIDTH": os.environ.get("H3_WIDTH", "1920"),
        "H3_PROGRESSIVE": os.environ.get("H3_PROGRESSIVE", "0"),
        "H3_FA_BACKEND": os.environ.get("H3_FA_BACKEND", "infer_v2"),
        "H3_OUT": os.environ.get("H3_OUT", "/workspace/out/h3_quant_generate.mp4"),
        "H3_PROMPT": os.environ.get("H3_PROMPT", DEFAULT_PROMPT),
        "H3_SUBMIT_TIMEOUT": os.environ.get("H3_SUBMIT_TIMEOUT", "1800"),
        "H3_SERVE_DIR": "/workspace/out/h3_serve",
    }
    cmd = [
        "docker",
        "compose",
        "-f",
        str(COMPOSE),
        "exec",
        "-T",
    ]
    for k, v in env_pass.items():
        cmd.extend(["-e", f"{k}={v}"])
    cmd.extend([SERVICE, "python", "/workspace/scripts/submit_generate.py"])
    print("[host] ", " ".join(cmd[:6]), "... submit_generate.py", flush=True)
    return subprocess.call(cmd, cwd=str(ROOT))


if __name__ == "__main__":
    raise SystemExit(main())
