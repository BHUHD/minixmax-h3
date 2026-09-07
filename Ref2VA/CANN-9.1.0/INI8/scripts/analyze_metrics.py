#!/usr/bin/env python3
"""从 h3_npu metrics.json 提取各模块/逐步耗时，写出 report。"""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--metrics", required=True)
    ap.add_argument("--out-dir", required=True)
    args = ap.parse_args()
    raw = json.loads(Path(args.metrics).read_text())
    if not isinstance(raw, list):
        raise SystemExit(f"unexpected metrics type: {type(raw)}")

    by_tag: dict[str, dict] = {}
    steps: list[dict] = []
    for e in raw:
        if not isinstance(e, dict):
            continue
        tag = e.get("tag")
        if not tag:
            continue
        by_tag[tag] = e
        if tag.startswith("step_") or ("step_s" in e and tag != "dit_done"):
            if "step_s" in e:
                steps.append(e)

    dit = by_tag.get("dit_done", {})
    written = by_tag.get("video_written", {})
    te = by_tag.get("te_encoded", {})
    refs = by_tag.get("refs_encoded", {})
    vae = by_tag.get("vae_decoded", {}) or by_tag.get("vae_done", {})

    step_s = [float(s["step_s"]) for s in steps if "step_s" in s]
    avg = sum(step_s) / len(step_s) if step_s else None

    # 模块墙钟：用 elapsed_s 差分（若有）
    def el(tag: str) -> float | None:
        e = by_tag.get(tag)
        if not e:
            return None
        v = e.get("elapsed_s")
        return float(v) if v is not None else None

    timeline = []
    for tag in (
        "start",
        "dit_loaded",
        "te_encoded",
        "refs_encoded",
        "dit_done",
        "vae_decoded",
        "vae_done",
        "video_written",
    ):
        if tag in by_tag:
            timeline.append((tag, el(tag), by_tag[tag]))

    report = {
        "source": str(args.metrics),
        "n_steps": len(step_s),
        "avg_step_s": avg,
        "min_step_s": min(step_s) if step_s else None,
        "max_step_s": max(step_s) if step_s else None,
        "dit_request_s": dit.get("request_s") or dit.get("wall_s"),
        "dit_elapsed_s": dit.get("elapsed_s"),
        "te_cached": te.get("cached"),
        "te_elapsed_s": te.get("elapsed_s"),
        "refs_elapsed_s": refs.get("elapsed_s"),
        "vae_s": vae.get("vae_s") or vae.get("decode_s") or vae.get("elapsed_s"),
        "e2e_elapsed_s": written.get("elapsed_s"),
        "out": written.get("out"),
        "steps": [
            {
                "tag": s.get("tag"),
                "step_s": s.get("step_s"),
                "elapsed_s": s.get("elapsed_s"),
            }
            for s in steps
        ],
        "modules": {},
    }

    # 模块耗时汇总（优先显式字段）
    modules = {
        "text_encoder_s": te.get("elapsed_s") if not te.get("cached") else 0.0,
        "ref_encode_s": refs.get("elapsed_s") if not refs.get("cached") else 0.0,
        "dit_20step_s": dit.get("request_s") or dit.get("wall_s"),
        "vae_decode_s": vae.get("vae_s") or vae.get("decode_s"),
        "e2e_wall_s": written.get("elapsed_s"),
    }
    # 写片 ≈ e2e - dit - vae（粗算）
    try:
        e2e = float(modules["e2e_wall_s"] or 0)
        dit_s = float(modules["dit_20step_s"] or 0)
        vae_s = float(modules["vae_decode_s"] or 0)
        modules["write_mp4_approx_s"] = round(max(0.0, e2e - dit_s - vae_s), 3)
    except Exception:
        modules["write_mp4_approx_s"] = None
    report["modules"] = modules

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "metrics" / "timings.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n"
    )

    md = []
    md.append("# INT8 16die 1080P 10s 分阶段耗时\n")
    md.append(f"- 来源：`{args.metrics}`\n")
    md.append("\n## 模块汇总\n\n")
    md.append("| 模块 | 耗时 (s) |\n|------|----------|\n")
    md.append(f"| Text Encoder | {modules.get('text_encoder_s')} |\n")
    md.append(f"| Ref encode | {modules.get('ref_encode_s')} |\n")
    md.append(f"| DiT 20 step | {modules.get('dit_20step_s')} |\n")
    md.append(f"| VAE decode | {modules.get('vae_decode_s')} |\n")
    md.append(f"| 写 mp4（粗算） | {modules.get('write_mp4_approx_s')} |\n")
    md.append(f"| **E2E 墙钟** | **{modules.get('e2e_wall_s')}** |\n")
    md.append("\n## DiT 逐步\n\n")
    md.append(f"- n={len(step_s)} avg={avg} min={report['min_step_s']} max={report['max_step_s']}\n\n")
    md.append("| step | step_s | elapsed_s |\n|------|--------|-----------|\n")
    for s in steps:
        md.append(f"| {s.get('tag')} | {s.get('step_s')} | {s.get('elapsed_s')} |\n")
    (out_dir / "report.md").write_text("".join(md))
    print("".join(md))


if __name__ == "__main__":
    main()
