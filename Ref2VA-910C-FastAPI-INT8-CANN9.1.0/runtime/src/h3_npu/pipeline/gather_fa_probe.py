"""Probe h3_gather_fa vs infer_v2 on single die (gathered KV, 1080P shapes)."""
from __future__ import annotations

import os
import sys
import time

os.environ.setdefault("ASCEND_RT_VISIBLE_DEVICES", "0")
sys.path.insert(0, "/workspace/src")

import torch

from h3_npu.ops.fa_h3 import h3_attention_bnsd


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


def main() -> None:
    torch.npu.set_device(0)
    dev = torch.device("npu:0")
    h, d, sq, sk = 56, 128, 9360, 149760
    q = torch.randn(1, h, sq, d, device=dev, dtype=torch.bfloat16)
    k = torch.randn(1, h, sk, d, device=dev, dtype=torch.bfloat16)
    v = torch.randn(1, h, sk, d, device=dev, dtype=torch.bfloat16)
    print(f"[gf-ab] die=0 q={sq} kv={sk} h={h}", flush=True)

    os.environ["H3_FA_BACKEND"] = "infer_v2"
    t0 = _bench(lambda: h3_attention_bnsd(q, k, v, gathered=True))
    print(f"[gf-ab] infer_v2        {t0:.4f}s  baseline", flush=True)

    from importlib import reload
    import h3_npu.ops.gather_fa as gf

    for mode in ("fused", "tiled", "head_chunk"):
        os.environ["H3_FA_BACKEND"] = "h3_gather_fa"
        os.environ["H3_GATHER_FA_MODE"] = mode
        reload(gf)
        t = _bench(lambda: gf.gather_fa_bnsd(q, k, v, gathered=True))
        rel = (t / t0 - 1.0) * 100.0
        print(f"[gf-ab] h3_gather_fa/{mode:<10} {t:.4f}s  rel={rel:+.1f}%", flush=True)
    print("[gf-ab] done", flush=True)


if __name__ == "__main__":
    main()
