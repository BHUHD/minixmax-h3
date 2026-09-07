"""Single-die FA microbench: layout copies vs native BNSD vs INT8-KV.

Pin one die before torch_npu. Shapes match 16-way 1080P local Q.
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


def _infer_bnsd(q, k, v, h, scale, **extra):
    import torch_npu

    return torch_npu.npu_fused_infer_attention_score(
        q,
        k,
        v,
        num_heads=h,
        input_layout="BNSD",
        scale=scale,
        pre_tokens=65536,
        next_tokens=65536,
        **extra,
    )[0]


def main():
    import torch_npu  # noqa: F401

    torch.npu.set_device(0)
    device = torch.device("npu:0")
    h, d = 56, 128
    sq = int(os.environ.get("H3_PROBE_SQ", "9360"))
    sk = int(os.environ.get("H3_PROBE_SK", "149760"))
    scale = d**-0.5
    print(f"[fa-probe] die={os.environ['ASCEND_RT_VISIBLE_DEVICES']} q={sq} kv={sk} h={h} d={d}", flush=True)

    q_bsnd = torch.randn(1, sq, h, d, device=device, dtype=torch.bfloat16)
    k_bsnd = torch.randn(1, sk, h, d, device=device, dtype=torch.bfloat16)
    v_bsnd = torch.randn(1, sk, h, d, device=device, dtype=torch.bfloat16)
    q_bnsd = q_bsnd.transpose(1, 2).contiguous()
    k_bnsd = k_bsnd.transpose(1, 2).contiguous()
    v_bnsd = v_bsnd.transpose(1, 2).contiguous()
    _sync()

    def cur():
        qb = q_bsnd.transpose(1, 2).contiguous()
        kb = k_bsnd.transpose(1, 2).contiguous()
        vb = v_bsnd.transpose(1, 2).contiguous()
        out = _infer_bnsd(qb, kb, vb, h, scale)
        return out.transpose(1, 2).contiguous()

    def native():
        return _infer_bnsd(q_bnsd, k_bnsd, v_bnsd, h, scale)

    def native_ip1():
        try:
            return _infer_bnsd(q_bnsd, k_bnsd, v_bnsd, h, scale, inner_precise=1)
        except TypeError:
            return _infer_bnsd(q_bnsd, k_bnsd, v_bnsd, h, scale)

    dt_cur = _bench(cur)
    dt_nat = _bench(native)
    dt_ip1 = _bench(native_ip1)
    print(f"[fa-probe] current(BSND->BNSD copy+infer) {dt_cur:.4f}s", flush=True)
    print(f"[fa-probe] native BNSD infer             {dt_nat:.4f}s", flush=True)
    print(f"[fa-probe] native BNSD inner_precise=1   {dt_ip1:.4f}s", flush=True)

    # INT8 KV + BF16 Q, pertoken scales (A3 prefill mode 1).
    try:
        k2 = k_bnsd.reshape(-1, d).contiguous()
        v2 = v_bnsd.reshape(-1, d).contiguous()
        k_i8, k_sc = torch_npu.npu_dynamic_quant(k2)
        v_i8, v_sc = torch_npu.npu_dynamic_quant(v2)
        k_i8 = k_i8.view(1, h, sk, d)
        v_i8 = v_i8.view(1, h, sk, d)
        k_sc = k_sc.view(1, h, sk)
        v_sc = v_sc.view(1, h, sk)
        _sync()
        print(
            f"[fa-probe] dynamic_quant ok k={k_i8.dtype} scale={k_sc.dtype} {tuple(k_sc.shape)}",
            flush=True,
        )
    except Exception as exc:
        print(f"[fa-probe] dynamic_quant failed: {exc}", flush=True)
        k_i8 = None

    def _try_v2(**kw):
        return torch_npu.npu_fused_infer_attention_score_v2(
            q_bnsd,
            k_i8,
            v_i8,
            num_query_heads=h,
            num_key_value_heads=h,
            input_layout="BNSD",
            softmax_scale=scale,
            **kw,
        )[0]

    if k_i8 is not None:
        combos = [
            ("v2 mode1 bf16scale", dict(
                key_quant_mode=1, value_quant_mode=1,
                dequant_scale_key=k_sc.to(torch.bfloat16),
                dequant_scale_value=v_sc.to(torch.bfloat16),
            )),
            ("v2 mode1 fp32scale", dict(
                key_quant_mode=1, value_quant_mode=1,
                dequant_scale_key=k_sc.float(),
                dequant_scale_value=v_sc.float(),
            )),
            ("v2 mode0 perchannel", dict(
                key_quant_mode=0, value_quant_mode=0,
                dequant_scale_key=k_sc.to(torch.bfloat16).mean(-1, keepdim=True),
                dequant_scale_value=v_sc.to(torch.bfloat16).mean(-1, keepdim=True),
            )),
        ]
        for name, kw in combos:
            try:
                def run(kw=kw):
                    return _try_v2(**kw)

                dt = _bench(run, n=3, warmup=1)
                print(f"[fa-probe] INT8-KV {name} {dt:.4f}s OK", flush=True)
            except Exception as exc:
                msg = str(exc).split("\n")[0][:220]
                print(f"[fa-probe] INT8-KV {name} FAIL {msg}", flush=True)

        # v1 antiquant path
        try:
            def run_v1():
                return torch_npu.npu_fused_infer_attention_score(
                    q_bnsd,
                    k_i8,
                    v_i8,
                    num_heads=h,
                    input_layout="BNSD",
                    scale=scale,
                    pre_tokens=65536,
                    next_tokens=65536,
                    antiquant_mode=1,
                    key_antiquant_scale=k_sc.float(),
                    value_antiquant_scale=v_sc.float(),
                )[0]

            dt = _bench(run_v1, n=3, warmup=1)
            print(f"[fa-probe] INT8-KV v1 antiquant_mode1 {dt:.4f}s OK", flush=True)
        except Exception as exc:
            msg = str(exc).split("\n")[0][:220]
            print(f"[fa-probe] INT8-KV v1 FAIL {msg}", flush=True)

    print("[fa-probe] done", flush=True)


if __name__ == "__main__":
    main()
