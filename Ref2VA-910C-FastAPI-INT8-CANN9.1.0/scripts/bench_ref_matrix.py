#!/usr/bin/env python3
"""NPU Ref2VA 参考矩阵基准：4 种输入 × 分辨率，采集精细耗时。

用法（仓库根或 INI8 目录均可）：
  python3 Ref2VA/CANN-9.1.0/INI8/scripts/bench_ref_matrix.py
  python3 .../bench_ref_matrix.py --only 1080p
  python3 .../bench_ref_matrix.py --only 768p --cases t2va,img1
"""
from __future__ import annotations

import argparse
import json
import re
import subprocess
import time
from pathlib import Path

import httpx

ROOT = Path(__file__).resolve().parents[4]  # MiniMax-H3
REFS = ROOT / "tmp" / "xianxia_refs"
OUT_ROOT = ROOT / "tmp" / "bench_npu_v1.1"
CONTAINER = "minimax-h3-ref2va-int8-npu16"
BASE = "http://127.0.0.1:8080"
COMPOSE_FILE = ROOT / "Ref2VA" / "CANN-9.1.0" / "INI8" / "docker-compose-v1.1.yml"


def restart_npu_serve(timeout: float = 1800.0) -> None:
    """1080P 连续任务易碎片 OOM：重启容器清 HBM。"""
    print("[bench] restarting NPU container to reclaim HBM…", flush=True)
    subprocess.check_call(
        ["docker", "compose", "-f", str(COMPOSE_FILE), "down"],
        cwd=str(ROOT),
    )
    subprocess.check_call(
        ["docker", "compose", "-f", str(COMPOSE_FILE), "up", "-d"],
        cwd=str(ROOT),
    )
    t0 = time.time()
    while time.time() - t0 < timeout:
        try:
            st = httpx.get(f"{BASE}/v1/status", timeout=5).json()
            if st.get("npu_serve_ready") and not st.get("busy"):
                print(f"[bench] READY after restart ({time.time()-t0:.0f}s)", flush=True)
                return
        except Exception:
            pass
        time.sleep(10)
    raise TimeoutError("NPU not ready after restart")

IMGS6 = [
    REFS / "img01_daoist_male.png",
    REFS / "img02_swordswoman.png",
    REFS / "img03_elder.png",
    REFS / "img04_mountain_gate.png",
    REFS / "img05_pavilion.png",
    REFS / "img06_night_sky.png",
]
VIDS3 = [
    REFS / "vid01_daoist_with_audio.mp4",
    REFS / "vid02_swordswoman.mp4",
    REFS / "vid03_mountain.mp4",
]
AUDS3 = [
    REFS / "aud01_male_line.wav",
    REFS / "aud02_female_line.wav",
    REFS / "aud03_bgm_guqin.wav",
]

PROMPT_T2VA = (REFS / "prompt_t2va_structured.txt").read_text(encoding="utf-8").strip()
PROMPT_FULL = (REFS / "prompt.txt").read_text(encoding="utf-8").strip()
PROMPT_IMG1 = (
    "中国3D国漫仙侠，电影级光影。以 <Picture 1> 为白衣青袍男道士外貌服饰。"
    "云海仙门石阶夜色，灵气薄雾。道士沉声普通话：「此劫已至，你可愿同我共闯剑冢？」"
    "口型同步，人声清晰。禁止现代都市与真人写实。"
)
PROMPT_IMG2 = (
    "中国3D国漫仙侠，电影级光影。"
    "以 <Picture 1> 为白衣青袍男道士外貌服饰；以 <Picture 2> 为白衣女剑修外貌服饰。"
    "两人面对面站立，云海仙门夜色。男道士：「此劫已至，你可愿同我共闯剑冢？」"
    "女剑修：「道友莫虑，我愿与君并肩破阵。」口型同步，对白清晰。"
)

CASES = {
    "t2va": {
        "label": "文生视频(无参考)",
        "task": "t2va",
        "prompt": PROMPT_T2VA,
        "images": [],
        "videos": [],
        "audios": [],
    },
    "img1": {
        "label": "参考图1张",
        "task": "ref2va",
        "prompt": PROMPT_IMG1,
        "images": [IMGS6[0]],
        "videos": [],
        "audios": [],
    },
    "img2": {
        "label": "参考图2张",
        "task": "ref2va",
        "prompt": PROMPT_IMG2,
        "images": [IMGS6[0], IMGS6[1]],
        "videos": [],
        "audios": [],
    },
    "full6x3x3": {
        "label": "参考图6+视频3+音频3",
        "task": "ref2va",
        "prompt": PROMPT_FULL,
        "images": IMGS6,
        "videos": VIDS3,
        "audios": AUDS3,
    },
}

RESOLUTIONS = {
    "1080p": {"width": 1920, "height": 1088, "tag": "1080p"},
    "768p": {"width": 1344, "height": 768, "tag": "768p"},
}


def wait_free(cli: httpx.Client, timeout: float = 7200.0) -> dict:
    t0 = time.time()
    while time.time() - t0 < timeout:
        st = cli.get(f"{BASE}/v1/status", timeout=10).json()
        if st.get("npu_serve_ready") and not st.get("busy"):
            return st
        time.sleep(3)
    raise TimeoutError("NPU not free")


def summarize_metrics(records: list[dict]) -> dict:
    """从 metrics.json 提取精细阶段耗时（秒）。"""
    by = {r["tag"]: r for r in records if isinstance(r, dict) and r.get("tag")}
    steps = [r for r in records if str(r.get("tag", "")).startswith("step_") and "step_s" in r]
    step_s = [float(r["step_s"]) for r in steps]

    def el(tag: str) -> float | None:
        r = by.get(tag)
        return float(r["elapsed_s"]) if r and r.get("elapsed_s") is not None else None

    dit_loaded = el("dit_loaded") or 0.0
    te_e = el("te_encoded")
    refs_e = el("refs_encoded")
    dit = by.get("dit_done") or {}
    vae = by.get("vae_decoded") or {}
    audio = by.get("audio_decoded") or {}
    written = by.get("video_written") or {}

    te_cached = bool((by.get("te_encoded") or {}).get("cached"))
    refs_cached = bool((by.get("refs_encoded") or {}).get("cached"))
    n_refs = int((by.get("refs_encoded") or {}).get("n_refs") or 0)

    # TE / Ref：用 elapsed 差分；并行时 refs_e≈te_e → refs≈0；t2va n_refs=0 强制 0
    if te_cached:
        te_s = 0.0
    elif te_e is not None:
        te_s = max(0.0, te_e - dit_loaded)
    else:
        te_s = None

    if refs_cached or n_refs <= 0:
        refs_s = 0.0
    elif refs_e is not None and te_e is not None:
        refs_s = max(0.0, refs_e - te_e)
    elif refs_e is not None:
        refs_s = max(0.0, refs_e - dit_loaded)
    else:
        refs_s = 0.0

    dit_sum = sum(step_s) if step_s else None
    dit_request = dit.get("request_s")
    if dit_request is not None:
        dit_request = float(dit_request)
    vae_s = float(vae["vae_s"]) if vae.get("vae_s") is not None else None
    audio_s = float(audio["audio_s"]) if audio.get("audio_s") is not None else None
    e2e = float(written["elapsed_s"]) if written.get("elapsed_s") is not None else el("video_written")

    # mux/写片 ≈ e2e - (encode准备到dit前) - dit - vae - audio
    encode_prep = None
    if te_e is not None or refs_e is not None:
        encode_prep = max(te_e or 0.0, refs_e or 0.0) - dit_loaded
    mux_s = None
    if e2e is not None and dit.get("elapsed_s") is not None and vae_s is not None:
        after_dit = float(dit["elapsed_s"])
        # video_written.elapsed - vae start ≈ vae+audio+mux；vae 起点 ≈ dit.elapsed
        post = e2e - after_dit
        mux_s = max(0.0, post - (vae_s or 0.0) - (audio_s or 0.0))

    return {
        "n_steps": len(step_s),
        "dit_step_sum_s": round(dit_sum, 3) if dit_sum is not None else None,
        "dit_avg_step_s": round(sum(step_s) / len(step_s), 3) if step_s else None,
        "dit_min_step_s": round(min(step_s), 3) if step_s else None,
        "dit_max_step_s": round(max(step_s), 3) if step_s else None,
        "dit_request_s": dit_request,  # DiT 采样墙钟（含 sync）
        "te_s": round(te_s, 3) if te_s is not None else None,
        "te_cached": te_cached,
        "refs_encode_s": round(refs_s, 3) if refs_s is not None else None,
        "refs_cached": refs_cached,
        "encode_prep_s": round(encode_prep, 3) if encode_prep is not None else None,
        "vae_decode_s": round(vae_s, 3) if vae_s is not None else None,
        "audio_decode_s": round(audio_s, 3) if audio_s is not None else None,
        "mux_write_s": round(mux_s, 3) if mux_s is not None else None,
        "pipeline_wall_s": round(e2e, 3) if e2e is not None else None,
        "dit_wall_s": float(dit["wall_s"]) if dit.get("wall_s") is not None else None,
        "step_s_list": [round(x, 3) for x in step_s],
    }


def parse_logs_for_task(task_id: str) -> dict:
    """从容器日志补充 TE/Ref/dit_done 打印耗时。"""
    try:
        raw = subprocess.check_output(
            ["docker", "logs", CONTAINER],
            stderr=subprocess.STDOUT,
            text=True,
            errors="ignore",
        )
    except Exception as exc:
        return {"error": str(exc)}

    # 取该 task DONE 之前的一段
    idx = raw.rfind(task_id)
    if idx < 0:
        chunk = raw[-80000:]
    else:
        chunk = raw[max(0, idx - 60000) : idx + 2000]

    out: dict = {}
    m = re.findall(r"\[te\] encoded in ([0-9.]+)s", chunk)
    if m:
        out["te_log_s"] = float(m[-1])
    m = re.findall(r"encoded refs for .* in ([0-9.]+)s", chunk)
    if m:
        out["refs_log_s"] = float(m[-1])
    m = re.findall(r"request-ready te\+ref done in ([0-9.]+)s", chunk)
    if m:
        out["ready_log_s"] = float(m[-1])
    m = re.findall(
        r"dit_done avg_step=([0-9.]+)s n=(\d+) wall=([0-9.]+)s request=([0-9.]+)s",
        chunk,
    )
    if m:
        avg, n, wall, req = m[-1]
        out["dit_avg_log_s"] = float(avg)
        out["dit_n_log"] = int(n)
        out["dit_wall_log_s"] = float(wall)
        out["dit_request_log_s"] = float(req)
    m = re.findall(r"audio decoded .* in ([0-9.]+)s", chunk)
    if m:
        out["audio_log_s"] = float(m[-1])
    m = re.findall(r"\[generate\] done wall=([0-9.]+)s request=([0-9.]+)s", chunk)
    if m:
        out["done_wall_log_s"] = float(m[-1][0])
        out["done_request_log_s"] = float(m[-1][1])
    return out


def docker_cp_metrics(task_id: str, dest: Path) -> Path | None:
    src = f"{CONTAINER}:/workspace/jobs/{task_id}/output.metrics.json"
    dest.parent.mkdir(parents=True, exist_ok=True)
    try:
        subprocess.check_call(["docker", "cp", src, str(dest)], stdout=subprocess.DEVNULL)
        return dest
    except subprocess.CalledProcessError:
        # 有时写在 out 旁的其它路径
        alt = f"{CONTAINER}:/workspace/jobs/{task_id}/"
        try:
            listing = subprocess.check_output(["docker", "exec", CONTAINER, "ls", "-la", f"/workspace/jobs/{task_id}"], text=True)
            print(f"[{task_id[:8]}] job dir:\n{listing}", flush=True)
        except Exception:
            pass
        print(f"[{task_id[:8]}] metrics missing at {src}", flush=True)
        return None


def run_one(
    cli: httpx.Client,
    *,
    case_id: str,
    res_id: str,
    seed: int,
    steps: int,
    duration: float,
) -> dict:
    case = CASES[case_id]
    res = RESOLUTIONS[res_id]
    label = f"{res['tag']}_{case_id}"
    out_dir = OUT_ROOT / label
    out_dir.mkdir(parents=True, exist_ok=True)
    video_out = out_dir / "output.mp4"

    wait_free(cli)
    handles = []
    files = []
    try:
        for p in case["images"]:
            h = open(p, "rb")
            handles.append(h)
            files.append(("ref_images", (p.name, h, "image/png")))
        for p in case["videos"]:
            h = open(p, "rb")
            handles.append(h)
            files.append(("ref_videos", (p.name, h, "video/mp4")))
        for p in case["audios"]:
            h = open(p, "rb")
            handles.append(h)
            files.append(("ref_audios", (p.name, h, "audio/wav")))
        data = {
            "prompt": case["prompt"],
            "task": case["task"],
            "width": str(res["width"]),
            "height": str(res["height"]),
            "duration": str(duration),
            "steps": str(steps),
            "seed": str(seed),
        }
        t_submit = time.time()
        r = cli.post(f"{BASE}/v1/tasks", data=data, files=files, timeout=300.0)
        print(f"[{label}] submit {r.status_code} {r.text[:200]}", flush=True)
        r.raise_for_status()
        task_id = r.json()["task_id"]
    finally:
        for h in handles:
            h.close()

    while True:
        q = cli.get(f"{BASE}/v1/tasks/{task_id}", timeout=30).json()
        prog = q.get("progress") or {}
        print(
            f"[{label}] {q.get('status')} phase={prog.get('phase')} "
            f"step={prog.get('step')}/{prog.get('steps_total')} pct={prog.get('percent')}",
            flush=True,
        )
        if q.get("status") == "succeeded":
            break
        if q.get("status") == "failed":
            raise RuntimeError(f"{label} failed: {q.get('error')}")
        time.sleep(10)

    t_done = time.time()
    client_wall = t_done - t_submit
    vid = cli.get(f"{BASE}/v1/tasks/{task_id}/video", timeout=600)
    vid.raise_for_status()
    video_out.write_bytes(vid.content)
    t_dl = time.time()

    metrics_path = out_dir / "output.metrics.json"
    got = docker_cp_metrics(task_id, metrics_path)
    timing = {}
    if got and got.is_file():
        timing = summarize_metrics(json.loads(got.read_text()))
    log_timing = parse_logs_for_task(task_id)

    # 合并：日志补洞
    if timing.get("te_s") is None and log_timing.get("te_log_s") is not None:
        timing["te_s"] = log_timing["te_log_s"]
    if (not timing.get("refs_encode_s")) and log_timing.get("refs_log_s") is not None:
        timing["refs_encode_s"] = log_timing["refs_log_s"]
    if timing.get("dit_request_s") is None and log_timing.get("dit_request_log_s") is not None:
        timing["dit_request_s"] = log_timing["dit_request_log_s"]
    if timing.get("audio_decode_s") is None and log_timing.get("audio_log_s") is not None:
        timing["audio_decode_s"] = log_timing["audio_log_s"]

    result = {
        "case_id": case_id,
        "label": case["label"],
        "resolution": f"{res['width']}x{res['height']}",
        "res_tag": res["tag"],
        "task": case["task"],
        "n_images": len(case["images"]),
        "n_videos": len(case["videos"]),
        "n_audios": len(case["audios"]),
        "seed": seed,
        "steps": steps,
        "duration": duration,
        "task_id": task_id,
        "client_wall_s": round(client_wall, 3),
        "download_s": round(t_dl - t_done, 3),
        "output_mp4": str(video_out),
        "output_bytes": video_out.stat().st_size,
        "timing": timing,
        "log_timing": log_timing,
    }
    (out_dir / "result.json").write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n")
    print(f"[{label}] DONE client_wall={client_wall:.1f}s timing={json.dumps(timing, ensure_ascii=False)}", flush=True)
    return result


def write_summary(results: list[dict]) -> Path:
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    summary_path = OUT_ROOT / "summary.json"
    summary_path.write_text(json.dumps(results, indent=2, ensure_ascii=False) + "\n")

    md = ["# NPU Ref2VA 推理耗时统计（v1.1）\n\n"]
    md.append("镜像：`minimax-h3:ref2va-cann9.1.0-int8-v1.1`；16 die；`H3_PROGRESSIVE=0`；steps=20；duration=10s；seed=42。\n\n")
    md.append("| 分辨率 | 场景 | 总时间(客户端)s | TE s | Ref编码 s | DiT采样 s | DiT均步 s | VAE解码 s | Audio解码 s | Mux/写片 s | Pipeline墙钟 s |\n")
    md.append("|--------|------|----------------:|-----:|----------:|----------:|----------:|----------:|------------:|-----------:|---------------:|\n")
    for r in results:
        t = r.get("timing") or {}
        md.append(
            f"| {r['resolution']} | {r['label']} | {r['client_wall_s']} | "
            f"{t.get('te_s')} | {t.get('refs_encode_s')} | {t.get('dit_request_s')} | "
            f"{t.get('dit_avg_step_s')} | {t.get('vae_decode_s')} | {t.get('audio_decode_s')} | "
            f"{t.get('mux_write_s')} | {t.get('pipeline_wall_s')} |\n"
        )
    md.append("\n说明：\n")
    md.append("- **总时间(客户端)**：`POST /v1/tasks` 到任务 `succeeded`（不含视频下载）。\n")
    md.append("- **TE**：Text Encoder（Qwen3-VL NVFP4）编码墙钟。\n")
    md.append("- **Ref编码**：参考图/视频/音频 VAE 条件 latent 编码；文生为 0。\n")
    md.append("- **DiT采样**：20 step 扩散采样墙钟（`dit_done.request_s`）。\n")
    md.append("- **VAE/Audio**：video VAE decode / audio VAE decode。\n")
    md.append("- **Mux/写片**：ffmpeg 封装估算。\n")
    md.append("- **Pipeline墙钟**：serve 内 metrics 从 job start 到写完 mp4。\n")
    md_path = OUT_ROOT / "summary.md"
    md_path.write_text("".join(md), encoding="utf-8")
    print("".join(md), flush=True)
    return md_path


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", choices=["1080p", "768p", "all"], default="all")
    ap.add_argument("--cases", default="t2va,img1,img2,full6x3x3")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--steps", type=int, default=20)
    ap.add_argument("--duration", type=float, default=10.0)
    args = ap.parse_args()

    case_ids = [c.strip() for c in args.cases.split(",") if c.strip()]
    for c in case_ids:
        if c not in CASES:
            raise SystemExit(f"unknown case {c}")

    res_ids = list(RESOLUTIONS) if args.only == "all" else [args.only]
    cli = httpx.Client(timeout=60.0)
    st = wait_free(cli)
    print(f"[bench] ready partition={st.get('partition')} instance={st.get('instance_id')}", flush=True)

    results: list[dict] = []
    # 若已有部分结果，续跑时合并
    prev = OUT_ROOT / "summary.json"
    if prev.is_file():
        try:
            results = json.loads(prev.read_text())
        except Exception:
            results = []

    def already(res_id: str, case_id: str) -> bool:
        tag = RESOLUTIONS[res_id]["tag"]
        return any(r.get("res_tag") == tag and r.get("case_id") == case_id and r.get("timing") for r in results)

    # 先 768P（稳），再 1080P（每案重启清碎片）
    ordered = [r for r in ("768p", "1080p") if r in res_ids]
    ordered += [r for r in res_ids if r not in ordered]

    for res_id in ordered:
        for case_id in case_ids:
            if already(res_id, case_id):
                print(f"[bench] skip existing {res_id}_{case_id}", flush=True)
                continue
            results = [
                r
                for r in results
                if not (r.get("res_tag") == RESOLUTIONS[res_id]["tag"] and r.get("case_id") == case_id)
            ]
            # 1080P：每个用例前重启，避免上一单 VAE/碎片导致 OOM
            if res_id == "1080p":
                restart_npu_serve()

            try:
                results.append(
                    run_one(
                        cli,
                        case_id=case_id,
                        res_id=res_id,
                        seed=args.seed,
                        steps=args.steps,
                        duration=args.duration,
                    )
                )
            except Exception as exc:
                print(f"[bench] {res_id}_{case_id} error: {exc}", flush=True)
                if res_id == "1080p" or "OOM" in str(exc) or "FAILED" in str(exc):
                    print("[bench] retry once after restart…", flush=True)
                    restart_npu_serve()
                    results.append(
                        run_one(
                            cli,
                            case_id=case_id,
                            res_id=res_id,
                            seed=args.seed,
                            steps=args.steps,
                            duration=args.duration,
                        )
                    )
                else:
                    raise
            write_summary(results)

    write_summary(results)
    print(f"[bench] ALL DONE → {OUT_ROOT}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
