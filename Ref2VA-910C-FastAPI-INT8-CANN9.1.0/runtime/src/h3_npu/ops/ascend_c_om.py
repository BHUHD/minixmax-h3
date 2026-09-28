"""Execute pre-compiled H3GatherFlashAttention OM (ATC singleop) on NPU."""
from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

_ACL_INITED = False
_OM_CACHE: dict[str, tuple] = {}

ACL_MEMCPY_DEVICE_TO_DEVICE = 3


def _ensure_acl(device: int = 0) -> None:
    global _ACL_INITED
    if _ACL_INITED:
        return
    import acl

    ret = acl.init()
    if ret not in (0, 100002):  # 100002: already initialized
        raise RuntimeError(f"acl.init failed ret={ret}")
    ret = acl.rt.set_device(device)
    if ret != 0:
        raise RuntimeError(f"acl.rt.set_device failed ret={ret}")
    _ACL_INITED = True


def _pick_om(q_shape, k_shape) -> Path:
    root = Path(os.environ.get("H3_GATHER_FA_OM_DIR", "/tmp/h3_atc_out"))
    if q_shape == (1, 56, 9360, 128) and k_shape == (1, 56, 149760, 128):
        pat = "0_H3GatherFlashAttention_1_2_1_56_9360_128_*_56_9360_128.om"
    elif q_shape == (1, 4, 32, 128) and k_shape == (1, 4, 128, 128):
        pat = "0_H3GatherFlashAttention_1_2_1_4_32_128_*_4_32_128.om"
    else:
        raise ValueError(f"no OM for q={q_shape} k={k_shape}")
    hits = sorted(root.glob(pat))
    if not hits:
        raise FileNotFoundError(f"OM not found under {root} pattern {pat}; run scripts/build_h3_gather_fa_om.sh")
    return hits[0]


def _load_model(om: Path):
    key = str(om.resolve())
    if key in _OM_CACHE:
        return _OM_CACHE[key]
    import acl
    import acl.mdl as mdl

    model_id, ret = mdl.load_from_file(str(om))
    if ret != 0:
        raise RuntimeError(f"mdl.load_from_file failed ret={ret}: {acl.get_recent_err_msg()}")
    desc = mdl.create_desc()
    ret = mdl.get_desc(desc, model_id)
    if ret != 0:
        raise RuntimeError(f"mdl.get_desc failed ret={ret}")
    _OM_CACHE[key] = (model_id, desc)
    return model_id, desc


def gather_fa_om(q, k, v, scale: float, *, num_heads: Optional[int] = None, world_size: Optional[int] = None):
    """Run ATC-compiled OM; bypasses TBE runtime JIT."""
    import acl
    import acl.mdl as mdl
    import torch

    if q.dim() == 3:
        q, k, v = q.unsqueeze(0), k.unsqueeze(0), v.unsqueeze(0)
    if q.dtype != torch.float16:
        q, k, v = q.to(torch.float16), k.to(torch.float16), v.to(torch.float16)
    q, k, v = q.contiguous(), k.contiguous(), v.contiguous()

    _ensure_acl(int(str(q.device).split(":")[-1]) if "npu" in str(q.device) else 0)
    om = _pick_om(tuple(q.shape), tuple(k.shape))
    model_id, _desc = _load_model(om)
    out = torch.empty_like(q)

    in_ds = mdl.create_dataset()
    out_ds = mdl.create_dataset()
    in_bufs = []
    out_bufs = []
    try:
        for tensor in (q, k, v):
            size = tensor.numel() * tensor.element_size()
            buf = acl.create_data_buffer(int(tensor.data_ptr()), size)
            in_bufs.append(buf)
            mdl.add_dataset_buffer(in_ds, buf)
        out_size = out.numel() * out.element_size()
        out_buf = acl.create_data_buffer(int(out.data_ptr()), out_size)
        out_bufs.append(out_buf)
        mdl.add_dataset_buffer(out_ds, out_buf)

        ret = mdl.execute(model_id, in_ds, out_ds)
        if ret != 0:
            raise RuntimeError(f"mdl.execute failed ret={ret}: {acl.get_recent_err_msg()}")
        torch.npu.synchronize()
        return out[0] if out.dim() == 4 and q.dim() == 4 else out
    finally:
        for buf in in_bufs + out_bufs:
            acl.destroy_data_buffer(buf)
        mdl.destroy_dataset(in_ds)
        mdl.destroy_dataset(out_ds)
