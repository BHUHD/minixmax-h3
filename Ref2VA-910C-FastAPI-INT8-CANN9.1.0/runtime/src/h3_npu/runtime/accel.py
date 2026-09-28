"""Optional TorchAir compile / Ascend C / HCCL hooks (910C)."""
from __future__ import annotations

import os
from typing import Optional

import torch.nn as nn


def maybe_torchair_compile(module: nn.Module, *, dynamic: bool = True) -> nn.Module:
    """Wrap with TorchAir NPU backend when H3_NPU_TORCHAIR=1."""
    if os.environ.get("H3_NPU_TORCHAIR", "0") != "1":
        return module
    try:
        import torch
        import torchair as tng
        from torchair.configs.compiler_config import CompilerConfig

        config = CompilerConfig()
        backend = tng.get_npu_backend(compiler_config=config)
        return torch.compile(module, backend=backend, dynamic=dynamic)
    except Exception as exc:  # noqa: BLE001
        print(f"[h3_npu] TorchAir compile skipped: {exc}", flush=True)
        return module


def hccl_world_info() -> dict:
    """Read HCCL / torch.distributed env without mutating host config."""
    return {
        "RANK": os.environ.get("RANK"),
        "WORLD_SIZE": os.environ.get("WORLD_SIZE"),
        "LOCAL_RANK": os.environ.get("LOCAL_RANK"),
        "HCCL_BUFFSIZE": os.environ.get("HCCL_BUFFSIZE"),
    }


def ascend_c_convrot_available() -> bool:
    """True when libh3_gather_fa.so (or future ConvRot kernel) is on disk."""
    from h3_npu.ops.gather_fa_ascend_c import gather_fa_ascend_c_available

    return gather_fa_ascend_c_available()
