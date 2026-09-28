"""8-die HCCL allreduce probe: no DiT, one phy die per rank."""
from __future__ import annotations

import os
import sys
from pathlib import Path

sys.path.insert(0, "/workspace/src")
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))


def pin_rank_die() -> None:
    local_rank = int(os.environ.get("LOCAL_RANK", os.environ.get("RANK", "0")))
    phy = [p.strip() for p in os.environ.get("H3_PHY_DEVICES", "1,3,5,7,9,11,13,15").split(",") if p.strip()]
    die = phy[local_rank]
    os.environ["ASCEND_RT_VISIBLE_DEVICES"] = die
    os.environ["ASCEND_VISIBLE_DEVICES"] = die
    os.environ["ASCEND_DEVICE_ID"] = "0"
    print(f"[hccl-probe] rank={os.environ.get('RANK')} vis={die}", flush=True)


pin_rank_die()


def main() -> None:
    from h3_npu.runtime.dist import init_hccl, warmup_collective, barrier, allreduce_npu
    import torch
    import torch_npu  # noqa: F401

    torch.npu.set_device(0)
    st = init_hccl()
    warmup_collective()
    barrier()
    x = torch.arange(8, device="npu:0", dtype=torch.float32) + st.rank
    y = allreduce_npu(x)
    torch.npu.synchronize()
    print(
        f"[hccl-probe] rank={st.rank} world={st.world_size} vis={os.environ.get('ASCEND_RT_VISIBLE_DEVICES')} "
        f"sum0={float(y[0]):.1f} ok",
        flush=True,
    )
    barrier()
    if st.rank == 0:
        print("[hccl-probe] all ranks passed", flush=True)


if __name__ == "__main__":
    main()
