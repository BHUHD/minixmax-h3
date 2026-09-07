"""Compare NPU FA vs SDPA and INT8 quant-matmul vs BF16 dequant."""
from __future__ import annotations

import os
import sys

sys.path.insert(0, "/workspace/src")
os.environ.setdefault("ASCEND_RT_VISIBLE_DEVICES", "1")
os.environ.setdefault("PYTORCH_NPU_ALLOC_CONF", "expandable_segments:False")

import torch
import torch.nn.functional as F

from h3_npu.ops.int8_convrot import int8_convrot_linear
from h3_npu.runtime.h2d import warmup_npu


def _sdpa(q, k, v, scale):
    qn, kn, vn = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)
    return F.scaled_dot_product_attention(qn, kn, vn, scale=scale).transpose(1, 2)


def probe_attention(device):
    os.environ["H3_NPU_FORCE_SDPA"] = "0"
    os.environ["H3_NPU_FORCE_FA"] = "1"
    from h3_npu.ops import attention as attn_mod

    attn_mod._LOGGED.clear()
    import torch_npu  # noqa: F401

    for heads, seq in ((7, 128), (32, 128), (56, 128)):
        b, d = 1, 128
        scale = d**-0.5
        q = torch.randn(b, seq, heads, d, device=device, dtype=torch.bfloat16)
        k = torch.randn(b, seq, heads, d, device=device, dtype=torch.bfloat16)
        v = torch.randn(b, seq, heads, d, device=device, dtype=torch.bfloat16)
        ref = _sdpa(q, k, v, scale)
        try:
            qn = q.transpose(1, 2).contiguous()
            kn = k.transpose(1, 2).contiguous()
            vn = v.transpose(1, 2).contiguous()
            fa = torch_npu.npu_fusion_attention(
                qn, kn, vn, heads, input_layout="BNSD", scale=scale,
                keep_prob=1.0, pre_tockens=65536, next_tockens=65536, sparse_mode=0,
            )[0].transpose(1, 2)
            torch.npu.synchronize()
            err = (fa.float() - ref.float()).abs().mean().item()
            rel = err / (ref.float().abs().mean().item() + 1e-6)
            print(f"[probe] FA vs SDPA heads={heads} seq={seq} mae={err:.5f} rel={rel:.5f}", flush=True)
        except Exception as exc:  # noqa: BLE001
            print(f"[probe] FA failed heads={heads}: {exc}", flush=True)


def probe_int8(device):
    m, k, n = 32, 256, 128
    x = torch.randn(m, k, device=device, dtype=torch.bfloat16)
    w = torch.randint(-8, 9, (n, k), device=device, dtype=torch.int8)
    scale = torch.full((n,), 0.02, device=device, dtype=torch.float32)
    y_q = int8_convrot_linear(x, w, scale, convrot=False, out_dtype=torch.bfloat16)
    w_bf = (w.float() * scale.unsqueeze(-1)).to(torch.bfloat16)
    y_d = F.linear(x, w_bf)
    err = (y_q.float() - y_d.float()).abs().mean().item()
    rel = err / (y_d.float().abs().mean().item() + 1e-6)
    print(f"[probe] INT8 quant_matmul vs dequant mae={err:.5f} rel={rel:.5f} "
          f"yq_std={float(y_q.float().std()):.4f} yd_std={float(y_d.float().std()):.4f}", flush=True)


def main():
    import torch_npu  # noqa: F401

    torch.npu.set_device(0)
    device = torch.device("npu:0")
    warmup_npu(device)
    probe_int8(device)
    probe_attention(device)
    print("[probe] done", flush=True)


if __name__ == "__main__":
    main()
