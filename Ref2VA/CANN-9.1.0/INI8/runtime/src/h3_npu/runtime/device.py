"""Device / CANN env helpers (container-only; no host config mutation)."""
from __future__ import annotations

import os

import torch


def init_npu(device_id: int = 0) -> torch.device:
    """Select one die and return torch.device('npu:0') after ASCEND_RT_VISIBLE_DEVICES remap."""
    import torch_npu  # noqa: F401

    if not torch.npu.is_available():
        raise RuntimeError("torch.npu is not available inside the container")
    torch.npu.set_device(device_id)
    # Prefer expandable segments inside the container cgroup.
    os.environ.setdefault("PYTORCH_NPU_ALLOC_CONF", "expandable_segments:True")
    return torch.device(f"npu:{device_id}")


def empty_cache() -> None:
    if torch.npu.is_available():
        torch.npu.empty_cache()


def sync() -> None:
    if torch.npu.is_available():
        torch.npu.synchronize()
