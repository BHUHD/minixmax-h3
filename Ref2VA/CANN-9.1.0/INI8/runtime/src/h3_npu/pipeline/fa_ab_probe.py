"""A/B microbench for all H3 FA backends (single die, 1080P shapes).

Usage (container):
  ASCEND_RT_VISIBLE_DEVICES=0 python /workspace/src/h3_npu/pipeline/fa_ab_probe.py
"""
from __future__ import annotations

import os
import sys
import time

os.environ["ASCEND_RT_VISIBLE_DEVICES"] = os.environ.get("H3_PROBE_DIE", "0")
os.environ["ASCEND_VISIBLE_DEVICES"] = os.environ["ASCEND_RT_VISIBLE_DEVICES"]
os.environ.setdefault("PYTORCH_NPU_ALLOC_CONF", "expandable_segments:False")
sys.path.insert(0, "/workspace/src")

import torch

BACKENDS = [
    "infer",
    "infer_v2",
    "fusion",
    "mindie_fas",
    "mindie_pfa",
    "mindie",
    "pfa",
]


def _sync():
    torch.npu.synchronize()


def _bench(fn, n=5, warmup=2):
    for _ in range(warmup):
        fn()
    _sync()
    t0 = time.time()
    for _ in range(n):
        fn()
    _sync()
    return (time.time() - t0) / n


def _run_backend(name: str, q, k, v, h, scale) -> tuple[float, str]:
    os.environ["H3_FA_BACKEND"] = name
    from importlib import reload
    import h3_npu.ops.fa_h3 as fa

    reload(fa)

    try:
        dt = _bench(lambda: fa.h3_attention_bnsd(q, k, v, gathered=True))
        return dt, "OK"
    except Exception as exc:
        msg = str(exc).split("\n")[0][:180]
        return -1.0, f"FAIL {msg}"


def main():
    import torch_npu  # noqa: F401

    torch.npu.set_device(0)
    device = torch.device("npu:0")
    h, d = 56, 128
    sq = int(os.environ.get("H3_PROBE_SQ", "9360"))
    sk = int(os.environ.get("H3_PROBE_SK", "149760"))
    scale = d**-0.5
    n = int(os.environ.get("H3_PROBE_ITERS", "5"))
    print(f"[fa-ab] die={os.environ['ASCEND_RT_VISIBLE_DEVICES']} q={sq} kv={sk} h={h}", flush=True)

    q = torch.randn(1, h, sq, d, device=device, dtype=torch.bfloat16)
    k = torch.randn(1, h, sk, d, device=device, dtype=torch.bfloat16)
    v = torch.randn(1, h, sk, d, device=device, dtype=torch.bfloat16)
    _sync()

    # Reference
    os.environ["H3_FA_BACKEND"] = "infer"
    from h3_npu.ops.fa_h3 import h3_attention_bnsd

    ref = h3_attention_bnsd(q, k, v, gathered=True).float()
    _sync()

    rows: list[tuple[str, float, float, str]] = []
    for name in BACKENDS:
        if name not in os.environ.get("H3_FA_AB_LIST", ",".join(BACKENDS)).split(","):
            continue
        dt, status = _run_backend(name, q, k, v, h, scale)
        rel = 0.0
        if status == "OK":
            os.environ["H3_FA_BACKEND"] = name
            from importlib import reload
            import h3_npu.ops.fa_h3 as fa

            reload(fa)
            out = fa.h3_attention_bnsd(q, k, v, gathered=True).float()
            rel = float((out - ref).abs().mean() / ref.abs().mean().clamp(min=1e-6))
        rows.append((name, dt, rel, status))
        print(f"[fa-ab] {name:14s} {dt:7.4f}s  rel={rel:.6f}  {status}", flush=True)

    best = min((r for r in rows if r[1] > 0), key=lambda x: x[1], default=None)
    if best:
        base = next((r[1] for r in rows if r[0] == "infer" and r[1] > 0), best[1])
        speedup = (base / best[1] - 1.0) * 100.0 if best[1] > 0 else 0.0
        print(f"[fa-ab] best={best[0]} {best[1]:.4f}s vs infer {base:.4f}s ({speedup:+.1f}%)", flush=True)
    print("[fa-ab] done", flush=True)


if __name__ == "__main__":
    main()
