"""H3GatherFlashAttention aclnn/pybind integration (standalone — avoid h3_npu.ops import path)."""
from __future__ import annotations

import importlib.util
import os
from pathlib import Path
from typing import Optional


def _load_torch_ext():
    candidates = [
        os.environ.get("H3_GATHER_FA_TORCH_SO", ""),
        "/tmp/h3_torch_build/h3_gather_fa_torch.cpython-312-aarch64-linux-gnu.so",
        "/tmp/h3_gather_fa_torch.cpython-312-aarch64-linux-gnu.so",
    ]
    for raw in candidates:
        if not raw:
            continue
        p = Path(raw)
        if not p.is_file():
            continue
        spec = importlib.util.spec_from_file_location("h3_gather_fa_torch", p)
        if spec and spec.loader:
            mod = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(mod)
            return mod
    return None


def gather_fa_ascend_c_available() -> bool:
    return _load_torch_ext() is not None


def _ensure_custom_opp() -> None:
    if not os.environ.get("ASCEND_CUSTOM_OPP_PATH", ""):
        raise RuntimeError(
            "ASCEND_CUSTOM_OPP_PATH not set; install custom_opp and "
            "source /usr/local/Ascend/custom_opp/vendors/customize/bin/set_env.bash"
        )


def _as_nd_fp16(tensor):
    import torch

    if tensor.dtype != torch.float16:
        tensor = tensor.to(torch.float16)
    return tensor.contiguous()


def _infer_v2_fa(q, k, v, heads: int, scale: float):
    from h3_npu.ops.fa_h3 import _infer_v2_bnsd

    return _infer_v2_bnsd(q, k, v, heads, scale)


def _resolve_fa_engine() -> str:
    eng = (os.environ.get("H3_ASCEND_C_FA_ENGINE") or "").strip().lower()
    if eng:
        return eng
    if os.environ.get("H3_FA_BACKEND", "").strip().lower() == "ascend_c":
        # 默认走 custom；失败由 gather_fa_native 回落 infer_v2 并打日志
        return "custom"
    return "infer_v2"


def gather_fa_native(
    q,
    k,
    v,
    scale: float,
    *,
    num_heads: Optional[int] = None,
    world_size: Optional[int] = None,
):
    import torch
    import torch_npu

    if q.dim() == 3:
        q, k, v = q.unsqueeze(0), k.unsqueeze(0), v.unsqueeze(0)
    heads = num_heads if num_heads is not None else int(q.shape[1])

    engine = _resolve_fa_engine()
    if engine in ("infer_v2", "infer", "v2"):
        return _infer_v2_fa(q, k, v, heads, float(scale))

    backend = os.environ.get("H3_GATHER_FA_INVOKE", "pybind").strip().lower()
    if backend in ("om", "atc", "mdl"):
        from h3_npu.ops.ascend_c_om import gather_fa_om

        return gather_fa_om(q, k, v, scale, num_heads=num_heads, world_size=world_size)

    try:
        _ensure_custom_opp()
        ext = _load_torch_ext()
        if ext is None:
            raise RuntimeError(
                "h3_gather_fa_torch.so not built; run scripts/build_h3_gather_fa_torch.sh"
            )
        ws = world_size if world_size is not None else int(os.environ.get("H3_FA_WORLD_SIZE", "16"))
        heads_attr = 5 if heads == 4 else heads
        out = ext.forward(
            _as_nd_fp16(q),
            _as_nd_fp16(k),
            _as_nd_fp16(v),
            int(heads_attr),
            float(scale),
            int(ws),
        )
        torch.npu.synchronize()
        if q.dim() == 3 and out.dim() == 4:
            out = out.squeeze(0)
        return out
    except Exception as exc:
        if engine == "custom" or os.environ.get("H3_ASCEND_C_NO_FALLBACK", "0") == "1":
            raise
        if not getattr(gather_fa_native, "_warned_fallback", False):
            print(f"[h3_gather_fa] custom opp failed ({exc}); falling back to infer_v2", flush=True)
            gather_fa_native._warned_fallback = True  # type: ignore[attr-defined]
        return _infer_v2_fa(q, k, v, heads, float(scale))
