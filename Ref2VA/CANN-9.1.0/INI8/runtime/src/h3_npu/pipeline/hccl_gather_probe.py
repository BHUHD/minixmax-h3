"""Sweep HCCL gather sizes. Parent: H3_WORKER_MODULE=h3_npu.pipeline.hccl_gather_probe."""
from __future__ import annotations

import os
import sys
import time
from pathlib import Path

sys.path.insert(0, "/workspace/src")
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from h3_npu.pipeline.generate_worker import pin_rank_die  # noqa: F401

pin_rank_die()


def main():
    from h3_npu.runtime.dist import init_hccl, state, _gather_broadcast, _gather_list

    st = init_hccl()
    import torch
    import torch.distributed as dist

    rank = st.rank
    elems = [
        64 * 1024,
        256 * 1024,
        1 * 1024 * 1024,
        4 * 1024 * 1024,
        16 * 1024 * 1024,
        32 * 1024 * 1024,
        9360 * 56 * 256,  # real KV cat bf16 elements (~268Mi elements? 9360*56*256=134M)
    ]
    # 9360*56*256 = 134,184,960 elems * 2 bytes ≈ 256MB
    for n in elems:
        x = torch.ones(n, device="npu:0", dtype=torch.bfloat16) * (rank + 1)
        torch.npu.synchronize()
        for name, fn in (("bcast", _gather_broadcast), ("list", _gather_list)):
            try:
                t0 = time.time()
                y = fn(x.view(-1, 1))
                torch.npu.synchronize()
                dt = time.time() - t0
                if rank == 0:
                    print(f"[gather-probe] {name} n={n} ({n*2/1e6:.1f}MB) {dt:.3f}s shape={tuple(y.shape)}", flush=True)
                del y
            except Exception as exc:
                if rank == 0:
                    print(f"[gather-probe] {name} n={n} FAIL {str(exc)[:180]}", flush=True)
                return
        del x
        torch.npu.empty_cache()
    if rank == 0:
        print("[gather-probe] done", flush=True)


if __name__ == "__main__":
    main()
