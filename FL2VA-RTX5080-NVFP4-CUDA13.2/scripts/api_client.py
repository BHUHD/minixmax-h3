#!/usr/bin/env python3
"""MiniMax-H3 FL2VA · ComfyUI 标准 API 客户端（单张参考图）。

依赖: 仅标准库。服务需已启动（默认 http://127.0.0.1:8188）。

示例:
  cp ./assets/input.png ./input/
  python scripts/api_client.py \\
    --image input.png \\
    --width 1920 --height 1088 --frames 243 --steps 20 \\
    --tau-low 1.0 --tau-high 1.0 --head-steps 0 --tail-steps 0 \\
    --seed 42 \\
    --prefix demo_1080p10s \\
    --cn-name 单参考图_1080P_10s_tau常数1.0.mp4
"""

from __future__ import annotations

import argparse
import copy
import json
import shutil
import time
import uuid
import urllib.error
import urllib.request
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
DEFAULT_WF = (
    ROOT
    / "assets"
    / "comfyui-api-workflow-minimax_h3_fl2va_1080p10s_sol_tau_const.json"
)


def http_json(url: str, data=None, timeout: float = 60):
    body = None if data is None else json.dumps(data).encode()
    req = urllib.request.Request(
        url,
        data=body,
        headers={"Content-Type": "application/json"} if body else {},
        method="POST" if body is not None else "GET",
    )
    with urllib.request.urlopen(req, timeout=timeout) as r:
        raw = r.read()
        return json.loads(raw) if raw else {}


def wait_ready(base: str, timeout: float = 300) -> None:
    t0 = time.time()
    while time.time() - t0 < timeout:
        try:
            http_json(f"{base}/system_stats", timeout=5)
            return
        except Exception:
            time.sleep(2)
    raise RuntimeError(f"ComfyUI not ready at {base}")


def free_vram(base: str) -> None:
    try:
        http_json(f"{base}/free", {"unload_models": True, "free_memory": True})
    except Exception:
        pass
    time.sleep(2)


def load_workflow(path: Path) -> dict:
    raw = json.loads(path.read_text(encoding="utf-8"))
    return {
        k: copy.deepcopy(v)
        for k, v in raw.items()
        if k != "_meta" and isinstance(v, dict) and "class_type" in v
    }


def build_prompt(
    workflow: dict,
    *,
    image: str,
    prompt_text: str | None,
    width: int,
    height: int,
    frames: int,
    steps: int,
    seed: int,
    tau_low: float,
    tau_high: float,
    head_steps: int,
    tail_steps: int,
    sink_conditioning: str,
    filename_prefix: str,
    fps: float,
) -> dict:
    p = copy.deepcopy(workflow)
    p["114"]["inputs"]["image"] = image
    if prompt_text is not None:
        p["104"]["inputs"]["prompt"] = prompt_text
    # 单张参考图：去掉可能残留的 last_frame
    p["104"]["inputs"].pop("last_frame", None)
    p["w"]["inputs"]["value"] = int(width)
    p["h"]["inputs"]["value"] = int(height)
    p["107"]["inputs"]["value"] = int(frames)
    p["9"]["inputs"]["steps"] = int(steps)
    p["15"]["inputs"]["noise_seed"] = int(seed)
    p["90"]["inputs"]["fps"] = float(fps)
    p["92"]["inputs"]["filename_prefix"] = filename_prefix
    bsa = p["bsa"]["inputs"]
    bsa["selection"] = "Sol-Attn (scheduled tau)"
    bsa["selection.tau_low"] = float(tau_low)
    bsa["selection.tau_high"] = float(tau_high)
    bsa["selection.head_steps"] = int(head_steps)
    bsa["selection.tail_steps"] = int(tail_steps)
    bsa["sink_conditioning"] = sink_conditioning
    return p


def wait_done(base: str, prompt_id: str, timeout: float) -> dict:
    t0 = time.perf_counter()
    while time.perf_counter() - t0 < timeout:
        try:
            hist = http_json(f"{base}/history/{prompt_id}", timeout=30)
        except urllib.error.HTTPError:
            hist = {}
        except Exception:
            time.sleep(3)
            continue
        item = hist.get(prompt_id)
        if item and (
            item.get("status", {}).get("completed")
            or item.get("status", {}).get("status_str") in ("success", "error")
        ):
            item["_wall_s"] = time.perf_counter() - t0
            return item
        time.sleep(3)
    raise TimeoutError(prompt_id)


def collect_outputs(item: dict) -> list[dict]:
    outs = []
    for o in item.get("outputs", {}).values():
        for key in ("gifs", "videos", "images"):
            for f in o.get(key, []) or []:
                outs.append(dict(f))
    return outs


def generate(
    base: str,
    workflow: dict,
    *,
    image: str,
    prompt_text: str | None = None,
    width: int = 1920,
    height: int = 1088,
    frames: int = 243,
    steps: int = 20,
    seed: int = 42,
    tau_low: float = 1.0,
    tau_high: float = 1.0,
    head_steps: int = 0,
    tail_steps: int = 0,
    sink_conditioning: str = "exact_kv_and_rows",
    filename_prefix: str = "fl2va",
    fps: float = 24.0,
    timeout: float = 2400,
    free_first: bool = True,
    client_id: str | None = None,
) -> dict:
    """提交一次生成，返回结果字典（含 wall_s、outputs、status）。"""
    wait_ready(base)
    if free_first:
        free_vram(base)
    prompt = build_prompt(
        workflow,
        image=image,
        prompt_text=prompt_text,
        width=width,
        height=height,
        frames=frames,
        steps=steps,
        seed=seed,
        tau_low=tau_low,
        tau_high=tau_high,
        head_steps=head_steps,
        tail_steps=tail_steps,
        sink_conditioning=sink_conditioning,
        filename_prefix=filename_prefix,
        fps=fps,
    )
    pid = str(uuid.uuid4())
    cid = client_id or str(uuid.uuid4())
    http_json(
        f"{base}/prompt",
        {"prompt": prompt, "client_id": cid, "prompt_id": pid},
        timeout=120,
    )
    item = wait_done(base, pid, timeout=timeout)
    st = item.get("status", {})
    return {
        "prompt_id": pid,
        "status": st.get("status_str"),
        "completed": bool(st.get("completed")),
        "wall_s": round(item.get("_wall_s", 0), 1),
        "outputs": collect_outputs(item),
        "params": {
            "image": image,
            "width": width,
            "height": height,
            "frames": frames,
            "steps": steps,
            "seed": seed,
            "tau_low": tau_low,
            "tau_high": tau_high,
            "head_steps": head_steps,
            "tail_steps": tail_steps,
            "sink_conditioning": sink_conditioning,
            "filename_prefix": filename_prefix,
            "fps": fps,
        },
    }


def rename_output_to_cn(out_files: list[dict], output_dir: Path, cn_name: str) -> Path | None:
    """把 ComfyUI 写出的首个 mp4 复制/重命名为中文文件名。"""
    if not out_files:
        return None
    f = out_files[0]
    src = output_dir / (f.get("subfolder") or "") / f["filename"]
    if not src.is_file():
        # 有时 subfolder 为空、文件直接在 output/
        src = output_dir / f["filename"]
    if not src.is_file():
        return None
    dest = output_dir / cn_name
    if dest.resolve() != src.resolve():
        shutil.copy2(src, dest)
    return dest


def main() -> int:
    ap = argparse.ArgumentParser(description="MiniMax-H3 FL2VA ComfyUI API 客户端")
    ap.add_argument("--base", default="http://127.0.0.1:8188")
    ap.add_argument("--workflow", type=Path, default=DEFAULT_WF)
    ap.add_argument(
        "--image",
        default="input.png",
        help="input/ 下文件名（readme：先 cp ./assets/input.png ./input/）",
    )
    ap.add_argument("--prompt-file", type=Path, default=None, help="可选：覆盖工作流中的文本提示")
    ap.add_argument("--width", type=int, default=1920)
    ap.add_argument("--height", type=int, default=1088)
    ap.add_argument("--frames", type=int, default=243, help="帧数；需满足 n%%17==5")
    ap.add_argument("--steps", type=int, default=20)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--fps", type=float, default=24.0)
    ap.add_argument("--tau-low", type=float, default=1.0)
    ap.add_argument("--tau-high", type=float, default=1.0)
    ap.add_argument("--head-steps", type=int, default=0)
    ap.add_argument("--tail-steps", type=int, default=0)
    ap.add_argument(
        "--sink",
        default="exact_kv_and_rows",
        choices=["exact_kv", "exact_kv_and_rows", "off"],
        help="Sol-Attn sink_conditioning",
    )
    ap.add_argument("--prefix", default="fl2va_api")
    ap.add_argument("--cn-name", default=None, help="完成后复制为中文文件名（写到 output/）")
    ap.add_argument("--timeout", type=float, default=2400)
    ap.add_argument("--no-free", action="store_true")
    args = ap.parse_args()

    prompt_text = None
    if args.prompt_file:
        prompt_text = args.prompt_file.read_text(encoding="utf-8")

    wf = load_workflow(args.workflow)
    print("waiting", args.base)
    result = generate(
        args.base,
        wf,
        image=args.image,
        prompt_text=prompt_text,
        width=args.width,
        height=args.height,
        frames=args.frames,
        steps=args.steps,
        seed=args.seed,
        tau_low=args.tau_low,
        tau_high=args.tau_high,
        head_steps=args.head_steps,
        tail_steps=args.tail_steps,
        sink_conditioning=args.sink,
        filename_prefix=args.prefix,
        fps=args.fps,
        timeout=args.timeout,
        free_first=not args.no_free,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))
    if args.cn_name and result.get("outputs"):
        dest = rename_output_to_cn(result["outputs"], ROOT / "output", args.cn_name)
        print("cn_video", dest)
    ok = result.get("status") == "success" or result.get("completed")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
