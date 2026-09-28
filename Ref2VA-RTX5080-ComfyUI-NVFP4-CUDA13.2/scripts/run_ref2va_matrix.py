#!/usr/bin/env python3
"""Ref2VA NVFP4 调试矩阵：4 种输入组合 × 768P/1080P × 20step。

记录总耗时与主要模块墙钟时间（基于 ComfyUI websocket executing 事件）。
"""
from __future__ import annotations

import argparse
import copy
import json
import threading
import time
import uuid
import urllib.error
import urllib.request
from collections import defaultdict
from pathlib import Path

try:
    import websocket  # websocket-client
except ImportError:
    websocket = None

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
ASSET = ROOT / "assets" / "test_input"
WF_BASE = ROOT / "workflows" / "minimax_h3_ref2va_1080p10s_sol_tau_const.json"
RESULTS = ROOT / "output" / "matrix_results.jsonl"
SUMMARY = ROOT / "output" / "matrix_summary.json"

RESOLUTIONS = {
    "768P": (1344, 768),
    "1080P": (1920, 1088),
}

# logical module buckets
MODULE_MAP = {
    "CLIPLoader": "text_encoder_load",
    "UNETLoader": "unet_load",
    "VAELoader": "vae_load",
    "LoadImage": "load_inputs",
    "LoadAudio": "load_inputs",
    "LoadVideo": "load_inputs",
    "GetVideoComponents": "load_inputs",
    "MiniMaxH3ReferenceToVideo": "ref2va_encode",
    "BlockSparseAttention": "sol_attn_patch",
    "BasicScheduler": "schedule",
    "KSamplerSelect": "schedule",
    "BasicGuider": "schedule",
    "RandomNoise": "schedule",
    "SamplerCustomAdvanced": "sampler",
    "VAEDecode": "vae_decode",
    "VAEDecodeAudio": "vae_decode",
    "CreateVideo": "mux_save",
    "SaveVideo": "mux_save",
    "PrimitiveInt": "misc",
    "PrimitiveFloat": "misc",
    "PrimitiveStringMultiline": "misc",
    "ComfyMathExpression": "misc",
}


def http_json(url, data=None, timeout=120):
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


def wait_ready(base, timeout=300):
    t0 = time.time()
    while time.time() - t0 < timeout:
        try:
            http_json(f"{base}/system_stats", timeout=5)
            return
        except Exception:
            time.sleep(2)
    raise RuntimeError(f"not ready: {base}")


def free_vram(base):
    try:
        http_json(f"{base}/free", {"unload_models": True, "free_memory": True})
    except Exception:
        pass
    time.sleep(3)


def upload_file(base: str, path: Path, subfolder: str = "test_input") -> str:
    """Upload via /upload/image (also accepts audio/video). Returns input-relative path."""
    boundary = "----ComfyBoundary" + uuid.uuid4().hex
    filename = f"{subfolder}/{path.name}"
    file_bytes = path.read_bytes()
    parts = []
    parts.append(
        f"--{boundary}\r\n"
        f'Content-Disposition: form-data; name="image"; filename="{filename}"\r\n'
        f"Content-Type: application/octet-stream\r\n\r\n".encode()
        + file_bytes
        + b"\r\n"
    )
    parts.append(
        f"--{boundary}\r\n"
        f'Content-Disposition: form-data; name="overwrite"\r\n\r\ntrue\r\n'.encode()
    )
    parts.append(
        f"--{boundary}\r\n"
        f'Content-Disposition: form-data; name="type"\r\n\r\ninput\r\n'.encode()
    )
    parts.append(f"--{boundary}--\r\n".encode())
    body = b"".join(parts)
    req = urllib.request.Request(
        f"{base}/upload/image",
        data=body,
        headers={"Content-Type": f"multipart/form-data; boundary={boundary}"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=120) as r:
        resp = json.loads(r.read())
    # prefer name which may include subfolder
    name = resp.get("name") or path.name
    sub = resp.get("subfolder") or ""
    if sub:
        return f"{sub}/{name}" if not name.startswith(sub) else name
    return name


def ensure_uploads(base: str, use_docker_paths: bool = True) -> dict:
    """Return input-relative path map. Default assumes assets already docker-cp'd."""
    paths = {}
    if use_docker_paths:
        for p in sorted((ASSET / "images").glob("*.png")):
            paths[f"img:{p.name}"] = f"test_input/images/{p.name}"
        for p in sorted((ASSET / "audios").glob("*.wav")):
            paths[f"aud:{p.name}"] = f"test_input/audios/{p.name}"
        for p in sorted((ASSET / "videos").glob("ref_video_*.mp4")):
            if "_src" in p.name:
                continue
            # LoadVideo lists files at input root
            paths[f"vid:{p.name}"] = p.name
        return paths
    for p in sorted((ASSET / "images").glob("*.png")):
        paths[f"img:{p.name}"] = upload_file(base, p)
    for p in sorted((ASSET / "audios").glob("*.wav")):
        paths[f"aud:{p.name}"] = upload_file(base, p)
    for p in sorted((ASSET / "videos").glob("ref_video_*.mp4")):
        if "_src" in p.name:
            continue
        paths[f"vid:{p.name}"] = upload_file(base, p)
    return paths

def base_workflow() -> dict:
    raw = json.loads(WF_BASE.read_text(encoding="utf-8"))
    return {k: copy.deepcopy(v) for k, v in raw.items() if k != "_meta" and isinstance(v, dict) and "class_type" in v}


def strip_hunt_specific(wf: dict) -> dict:
    """Remove fixed hunt img/aud nodes; rebuild per case."""
    drop = [k for k in list(wf) if k.startswith("img") or k.startswith("aud") or k.startswith("vid")]
    for k in drop:
        del wf[k]
    # clear ref wiring on 136
    m = wf["136"]["inputs"]
    for k in list(m):
        if k.startswith("ref_images") or k.startswith("ref_audios") or k.startswith("ref_videos") or k.startswith("ref_video_audios"):
            del m[k]
    return wf


def build_case(case: str, width: int, height: int, steps: int, seed: int, prefix: str, paths: dict) -> dict:
    wf = strip_hunt_specific(base_workflow())
    wf["w"]["inputs"]["value"] = width
    wf["h"]["inputs"]["value"] = height
    wf["143"]["inputs"]["value"] = steps
    wf["132"]["inputs"]["value"] = 10.0
    wf["129"]["inputs"]["noise_seed"] = seed
    wf["92"]["inputs"]["filename_prefix"] = prefix

    # Sol-Attn already in base
    m = wf["136"]["inputs"]

    if case in ("full6x3x3", "img6_aud3_vid1", "img1"):
        # images
        if case == "img1":
            imgs = [("00_首帧.png", "img0")]
            prompt = (ASSET / "prompt_1img.txt").read_text(encoding="utf-8")
        else:
            imgs = [
                ("00_首帧.png", "img0"),
                ("01_陆清川-男主.png", "img1"),
                ("02_玄霜真人-女师傅.png", "img2"),
                ("03_苏晚晴-女徒弟.png", "img3"),
                ("04_叶芷若-女徒弟.png", "img4"),
                ("05_陈砚舟-男徒弟.png", "img5"),
            ]
            prompt = (ASSET / "prompt.txt").read_text(encoding="utf-8")

        for i, (name, nid) in enumerate(imgs):
            key = f"img:{name}"
            wf[nid] = {"class_type": "LoadImage", "inputs": {"image": paths[key]}}
            m[f"ref_images.ref_image_{i}"] = [nid, 0]

        if case in ("full6x3x3", "img6_aud3_vid1"):
            auds = [
                ("01_陆清川-男主.wav", "aud0"),
                ("03_苏晚晴-女徒弟.wav", "aud1"),
                ("04_叶芷若-女徒弟.wav", "aud2"),
            ]
            for i, (name, nid) in enumerate(auds):
                wf[nid] = {"class_type": "LoadAudio", "inputs": {"audio": paths[f"aud:{name}"]}}
                m[f"ref_audios.ref_audio_{i}"] = [nid, 0]

            vids = ["ref_video_1.mp4", "ref_video_2.mp4", "ref_video_3.mp4"]
            if case == "img6_aud3_vid1":
                vids = vids[:1]
            for i, name in enumerate(vids):
                # LoadVideo expects basename listed by server; use uploaded name basename
                uploaded = paths[f"vid:{name}"]
                # prefer basename for LoadVideo combo
                file_arg = Path(uploaded).name
                vload = f"vidload{i}"
                vcomp = f"vidcomp{i}"
                wf[vload] = {"class_type": "LoadVideo", "inputs": {"file": file_arg}}
                wf[vcomp] = {"class_type": "GetVideoComponents", "inputs": {"video": [vload, 0]}}
                m[f"ref_videos.ref_video_{i}"] = [vcomp, 0]
    else:
        # text only — MiniMaxH3ReferenceToVideo with no refs
        prompt = (ASSET / "prompt_text_only.txt").read_text(encoding="utf-8")

    wf["138"]["inputs"]["value"] = prompt
    return wf


class NodeTimer:
    def __init__(self, prompt_id: str):
        self.prompt_id = prompt_id
        self.node_seconds = defaultdict(float)
        self.class_of = {}
        self._current = None
        self._t0 = None
        self.done = threading.Event()
        self.error = None

    def on_message(self, message: dict, workflow: dict):
        typ = message.get("type")
        data = message.get("data") or {}
        if data.get("prompt_id") and data.get("prompt_id") != self.prompt_id:
            return
        if typ == "executing":
            # close previous
            if self._current is not None and self._t0 is not None:
                self.node_seconds[self._current] += time.time() - self._t0
            nid = data.get("node")
            if nid is None:
                # execution finished
                self._current = None
                self._t0 = None
                self.done.set()
                return
            self._current = str(nid)
            self._t0 = time.time()
            node = workflow.get(str(nid)) or workflow.get(nid)
            if node:
                self.class_of[str(nid)] = node.get("class_type", "?")
        elif typ == "execution_error":
            self.error = data
            self.done.set()

    def module_breakdown(self) -> dict:
        by_mod = defaultdict(float)
        by_class = defaultdict(float)
        for nid, sec in self.node_seconds.items():
            cls = self.class_of.get(nid, "?")
            by_class[cls] += sec
            by_mod[MODULE_MAP.get(cls, f"other:{cls}")] += sec
        return {
            "by_module_seconds": {k: round(v, 2) for k, v in sorted(by_mod.items(), key=lambda x: -x[1])},
            "by_class_seconds": {k: round(v, 2) for k, v in sorted(by_class.items(), key=lambda x: -x[1])},
        }


def run_one(base: str, case: str, res_name: str, paths: dict, steps: int = 20, seed: int = 20260928) -> dict:
    w, h = RESOLUTIONS[res_name]
    prefix = f"matrix/{case}_{res_name}_{steps}step"
    wf = build_case(case, w, h, steps, seed, prefix, paths)
    client_id = str(uuid.uuid4())
    timer = NodeTimer("")

    ws_holder = {"ws": None}

    def ws_thread():
        if websocket is None:
            return
        url = base.replace("http://", "ws://").replace("https://", "wss://") + f"/ws?clientId={client_id}"
        ws = websocket.WebSocketApp(
            url,
            on_message=lambda _ws, msg: timer.on_message(json.loads(msg), wf),
        )
        ws_holder["ws"] = ws
        ws.run_forever()

    th = threading.Thread(target=ws_thread, daemon=True)
    th.start()
    time.sleep(0.5)

    free_vram(base)
    t_submit = time.time()
    resp = http_json(f"{base}/prompt", {"prompt": wf, "client_id": client_id})
    prompt_id = resp["prompt_id"]
    timer.prompt_id = prompt_id
    if resp.get("node_errors"):
        return {"case": case, "resolution": res_name, "status": "submit_error", "node_errors": resp["node_errors"]}

    # poll history as backup
    status = "unknown"
    while time.time() - t_submit < 3600:
        if timer.done.wait(timeout=5):
            break
        try:
            hist = http_json(f"{base}/history/{prompt_id}", timeout=30)
        except Exception:
            continue
        if prompt_id in hist:
            st = hist[prompt_id].get("status") or {}
            status = st.get("status_str", "done")
            timer.done.set()
            break

    elapsed = time.time() - t_submit
    hist = {}
    try:
        hist = http_json(f"{base}/history/{prompt_id}", timeout=30).get(prompt_id) or {}
    except Exception:
        pass
    st = hist.get("status") or {}
    status = st.get("status_str") or ("error" if timer.error else status)
    out_mp4 = None
    for nid, out in (hist.get("outputs") or {}).items():
        for key in ("videos", "images", "gifs"):
            for item in out.get(key) or []:
                fn = item.get("filename")
                if fn and fn.endswith(".mp4"):
                    out_mp4 = f"{item.get('subfolder','')}/{fn}".strip("/")

    if ws_holder["ws"] is not None:
        try:
            ws_holder["ws"].close()
        except Exception:
            pass

    breakdown = timer.module_breakdown()
    # wall-clock total is authoritative; sampler usually dominates
    result = {
        "case": case,
        "resolution": res_name,
        "width": w,
        "height": h,
        "steps": steps,
        "prompt_id": prompt_id,
        "status": status,
        "elapsed_seconds": round(elapsed, 2),
        "output": out_mp4,
        "error": timer.error,
        **breakdown,
    }
    return result


CASES = [
    ("text_only", "prompt-only"),
    ("img1", "prompt+1图"),
    ("img6_aud3_vid1", "prompt+6图+3音+1视频"),
    ("full6x3x3", "prompt+6图+3音+3视频"),
]

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="http://127.0.0.1:8188")
    ap.add_argument("--steps", type=int, default=20)
    ap.add_argument("--only", default="", help="comma cases and/or resolutions, e.g. img1,768P")
    args = ap.parse_args()

    wait_ready(args.base)
    print("uploading assets...", flush=True)
    paths = ensure_uploads(args.base)
    print("paths", json.dumps(paths, ensure_ascii=False, indent=2), flush=True)

    # verify LoadVideo sees files
    try:
        info = http_json(f"{args.base}/object_info/LoadVideo")
        opts = info["LoadVideo"]["input"]["required"]["file"][1].get("options") or []
        print("LoadVideo options sample:", opts[:20], flush=True)
    except Exception as e:
        print("LoadVideo info err", e, flush=True)

    RESULTS.parent.mkdir(parents=True, exist_ok=True)
    if RESULTS.exists():
        RESULTS.unlink()

    filters = {x.strip() for x in args.only.split(",") if x.strip()}
    all_results = []
    for case, label in CASES:
        for res_name in ("768P", "1080P"):
            if filters:
                if case not in filters and res_name not in filters and f"{case}_{res_name}" not in filters:
                    # allow filter by label keywords
                    if not any(f in case or f in res_name for f in filters):
                        continue
            print(f"\n=== RUN {label} / {res_name} ===", flush=True)
            try:
                r = run_one(args.base, case, res_name, paths, steps=args.steps)
            except Exception as e:
                r = {"case": case, "resolution": res_name, "status": "exception", "error": str(e), "elapsed_seconds": None}
            r["label"] = label
            print(json.dumps(r, ensure_ascii=False), flush=True)
            with RESULTS.open("a", encoding="utf-8") as f:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
            all_results.append(r)

    SUMMARY.write_text(json.dumps(all_results, ensure_ascii=False, indent=2), encoding="utf-8")
    print("\nSUMMARY ->", SUMMARY, flush=True)


if __name__ == "__main__":
    main()
