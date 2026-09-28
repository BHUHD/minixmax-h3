#!/usr/bin/env python3
"""遍历分辨率 × 时长 × Sol-tau 组合，单张参考图实测，并写结果到 output/results.jsonl。

默认矩阵（steps=20，seed=42，一张参考图）:
  分辨率: 1344×768 (768P), 1920×1088 (1080P)
  时长:   124帧(~5.17s), 192帧(~8.0s), 243帧(~10.125s)
  tau:    常数1.0 / 折中1.5 / 加速2.0

用法:
  python scripts/run_matrix_test.py
  python scripts/run_matrix_test.py --only-res 768P --only-dur 5s
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(HERE))

from api_client import (  # noqa: E402
    generate,
    load_workflow,
    rename_output_to_cn,
    wait_ready,
)

DEFAULT_WF = (
    ROOT
    / "assets"
    / "comfyui-api-workflow-minimax_h3_fl2va_1080p10s_sol_tau_const.json"
)
OUT_DIR = ROOT / "output"
LOG = OUT_DIR / "results.jsonl"

# 分辨率档位（宽×高须为 32 的倍数）
RESOLUTIONS = {
    "768P": (1344, 768),
    "1080P": (1920, 1088),
}

# 时长：帧数满足 H3 align (n % 17 == 5)
DURATIONS = {
    "5s": {"frames": 124, "approx_s": 5.17},
    "8s": {"frames": 192, "approx_s": 8.0},
    "10s": {"frames": 243, "approx_s": 10.125},
}

# Sol-Attn scheduled tau 预设
TAU_PRESETS = {
    "tau常数1.0": {
        "tau_low": 1.0,
        "tau_high": 1.0,
        "head_steps": 0,
        "tail_steps": 0,
        "label": "1.0-1.0_h0t0",
    },
    "tau折中1.5": {
        "tau_low": 1.0,
        "tau_high": 1.5,
        "head_steps": 1,
        "tail_steps": 1,
        "label": "1.0-1.5_h1t1",
    },
    "tau加速2.0": {
        "tau_low": 1.0,
        "tau_high": 2.0,
        "head_steps": 3,
        "tail_steps": 2,
        "label": "1.0-2.0_h3t2",
    },
}


def cn_filename(res_name: str, dur_name: str, tau_name: str, wall_s: float) -> str:
    return f"单参考图_{res_name}_{dur_name}_{tau_name}_耗时{int(round(wall_s))}秒.mp4"


def already_done(key: str) -> dict | None:
    if not LOG.is_file():
        return None
    last = None
    for line in LOG.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        rec = json.loads(line)
        if rec.get("key") == key and rec.get("status") == "success":
            last = rec
    return last


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="http://127.0.0.1:8188")
    ap.add_argument("--workflow", type=Path, default=DEFAULT_WF)
    ap.add_argument(
        "--image",
        default="input.png",
        help="input/ 下文件名（readme：先 cp ./assets/input.png ./input/）",
    )
    ap.add_argument("--steps", type=int, default=20)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--timeout", type=float, default=3600)
    ap.add_argument("--only-res", nargs="*", default=None, help="如 768P 1080P")
    ap.add_argument("--only-dur", nargs="*", default=None, help="如 5s 8s 10s")
    ap.add_argument("--only-tau", nargs="*", default=None, help="如 tau常数1.0")
    ap.add_argument("--force", action="store_true", help="忽略已成功记录，重跑")
    args = ap.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    wf = load_workflow(args.workflow)
    wait_ready(args.base)
    print("ready", args.base, flush=True)

    res_list = args.only_res or list(RESOLUTIONS)
    dur_list = args.only_dur or list(DURATIONS)
    tau_list = args.only_tau or list(TAU_PRESETS)

    cases = []
    for rn in res_list:
        for dn in dur_list:
            for tn in tau_list:
                cases.append((rn, dn, tn))

    print(f"total cases: {len(cases)}", flush=True)
    summary = []
    for i, (rn, dn, tn) in enumerate(cases, 1):
        key = f"{rn}|{dn}|{tn}"
        if not args.force:
            prev = already_done(key)
            if prev:
                print(f"[{i}/{len(cases)}] skip {key} wall={prev.get('wall_s')}", flush=True)
                summary.append(prev)
                continue

        w, h = RESOLUTIONS[rn]
        dur = DURATIONS[dn]
        tau = TAU_PRESETS[tn]
        prefix = f"matrix/{rn}_{dn}_{tau['label']}"
        print(
            f"[{i}/{len(cases)}] run {key} {w}x{h} frames={dur['frames']} "
            f"tau={tau['label']} steps={args.steps}",
            flush=True,
        )
        t_submit = time.strftime("%Y-%m-%d %H:%M:%S")
        try:
            result = generate(
                args.base,
                wf,
                image=args.image,
                width=w,
                height=h,
                frames=dur["frames"],
                steps=args.steps,
                seed=args.seed,
                tau_low=tau["tau_low"],
                tau_high=tau["tau_high"],
                head_steps=tau["head_steps"],
                tail_steps=tau["tail_steps"],
                filename_prefix=prefix,
                timeout=args.timeout,
                free_first=True,
                client_id="matrix_test",
            )
        except Exception as e:
            rec = {
                "key": key,
                "status": "error",
                "error": str(e),
                "submitted_at": t_submit,
                "resolution": rn,
                "duration": dn,
                "tau": tn,
            }
            with LOG.open("a", encoding="utf-8") as f:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            print("FAIL", rec, flush=True)
            summary.append(rec)
            continue

        wall = result["wall_s"]
        cn = cn_filename(rn, dn, tn, wall)
        dest = rename_output_to_cn(result.get("outputs") or [], OUT_DIR, cn)
        rec = {
            "key": key,
            "status": result["status"],
            "completed": result["completed"],
            "wall_s": wall,
            "submitted_at": t_submit,
            "resolution": rn,
            "width": w,
            "height": h,
            "duration": dn,
            "frames": dur["frames"],
            "approx_duration_s": dur["approx_s"],
            "tau": tn,
            "tau_params": {
                "tau_low": tau["tau_low"],
                "tau_high": tau["tau_high"],
                "head_steps": tau["head_steps"],
                "tail_steps": tau["tail_steps"],
            },
            "steps": args.steps,
            "seed": args.seed,
            "image": args.image,
            "prompt_id": result["prompt_id"],
            "outputs": result.get("outputs"),
            "cn_video": str(dest) if dest else None,
        }
        with LOG.open("a", encoding="utf-8") as f:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        print(f"[done] {key} wall={wall}s -> {dest}", flush=True)
        summary.append(rec)
        if result.get("status") != "success" and not result.get("completed"):
            print("ABORT", rec, flush=True)
            break

    summary_path = OUT_DIR / "results_summary.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print("summary", summary_path, flush=True)
    ok = all(r.get("status") == "success" for r in summary)
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
