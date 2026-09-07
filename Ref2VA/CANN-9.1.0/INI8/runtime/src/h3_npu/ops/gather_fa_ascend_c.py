"""Thin re-export: load aclnn integration from standalone module (OpCommand-safe)."""
from __future__ import annotations

import importlib.util
from pathlib import Path
from typing import Any

_ACLNN: Any = None


def _aclnn_mod():
    global _ACLNN
    if _ACLNN is None:
        p = Path(__file__).resolve().parents[2] / "h3_gather_fa_aclnn.py"
        spec = importlib.util.spec_from_file_location("h3_gather_fa_aclnn", p)
        if not spec or not spec.loader:
            raise RuntimeError(f"cannot load {p}")
        _ACLNN = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(_ACLNN)
    return _ACLNN


def gather_fa_ascend_c_available() -> bool:
    return _aclnn_mod().gather_fa_ascend_c_available()


def gather_fa_native(*args, **kwargs):
    return _aclnn_mod().gather_fa_native(*args, **kwargs)


def _as_nd_fp16(tensor):
    return _aclnn_mod()._as_nd_fp16(tensor)
