"""Per-rank generate: pin one phy die, HCCL first, then parallel DiT H2D."""
from __future__ import annotations

import os
import sys
from pathlib import Path

sys.path.insert(0, "/workspace/src")
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))


def pin_rank_die() -> None:
    """Each rank must see exactly one 910C die before any torch import."""
    local_rank = int(os.environ.get("LOCAL_RANK", os.environ.get("RANK", "0")))
    phy = [p.strip() for p in os.environ.get("H3_PHY_DEVICES", "0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15").split(",") if p.strip()]
    if local_rank >= len(phy):
        raise SystemExit(f"LOCAL_RANK={local_rank} but H3_PHY_DEVICES has {len(phy)} entries")
    die = phy[local_rank]
    os.environ["ASCEND_RT_VISIBLE_DEVICES"] = die
    os.environ["ASCEND_VISIBLE_DEVICES"] = die
    os.environ["ASCEND_DEVICE_ID"] = "0"
    os.environ["PYTORCH_NPU_ALLOC_CONF"] = os.environ.get(
        "PYTORCH_NPU_ALLOC_CONF", "expandable_segments:False"
    )
    print(
        f"[generate] rank={os.environ.get('RANK', '?')} local={local_rank} "
        f"ASCEND_RT_VISIBLE_DEVICES={die} (logical npu:0)",
        flush=True,
    )


pin_rank_die()


def _serve_loop(model, device, args) -> None:
    """Keep DiT + HCCL resident; run generate when job.json appears."""
    import gc
    import json
    import time
    from pathlib import Path

    from h3_npu.pipeline.generate import run_generate
    from h3_npu.runtime.device import empty_cache
    from h3_npu.runtime.dist import state as dist_state

    st = dist_state()
    root = Path(os.environ.get("H3_SERVE_DIR", "/workspace/out/h3_serve"))
    root.mkdir(parents=True, exist_ok=True)
    (root / f"ready.{st.rank}").write_text("ok")
    if st.rank == 0:
        t0 = time.time()
        while time.time() - t0 < 600:
            if sum(1 for _ in root.glob("ready.*")) >= st.world_size:
                break
            time.sleep(0.05)
        (root / "READY").write_text("ok")
        print("[serve] READY (DiT resident, HCCL up) waiting for job.json", flush=True)
    last = ""
    while True:
        jobp = root / "job.json"
        if not jobp.is_file():
            time.sleep(0.15)
            continue
        try:
            job = json.loads(jobp.read_text())
        except Exception:
            time.sleep(0.1)
            continue
        jid = str(job.get("id") or "")
        if not jid or jid == last:
            time.sleep(0.15)
            continue
        (root / f"ack.{st.rank}").write_text(jid)
        t0 = time.time()
        while time.time() - t0 < 60:
            n = sum(1 for p in root.glob("ack.*") if p.read_text().strip() == jid)
            if n >= st.world_size:
                break
            time.sleep(0.05)
        last = jid
        if job.get("cmd") == "stop":
            if st.rank == 0:
                print("[serve] stop", flush=True)
            return
        if "steps" in job:
            args.steps = int(job["steps"])
        if "seconds" in job:
            args.seconds = float(job["seconds"])
        if "height" in job:
            args.height = int(job["height"])
        if "width" in job:
            args.width = int(job["width"])
        if "out" in job:
            args.out = Path(job["out"])
        if "prompt" in job:
            args.prompt = str(job["prompt"])
        if "seed" in job:
            args.seed = int(job["seed"])
        # 每单必须重置多模态路径，避免上一单的 video/audio 残留到「仅图片」任务
        if "ref_images" in job:
            args.ref_image = [str(p) for p in job["ref_images"]]
        elif "ref_image" in job:
            args.ref_image = [str(job["ref_image"])]
        else:
            args.ref_image = []
        if "ref_videos" in job:
            args.ref_video = [str(p) for p in job["ref_videos"]]
        else:
            args.ref_video = []
        if "ref_audios" in job:
            args.ref_audio = [str(p) for p in job["ref_audios"]]
        else:
            args.ref_audio = []
        # 显式清空时同步去掉环境回退，防止 generate.py 再读 H3_REF_* 旧值
        if "task" in job:
            task_val = str(job.get("task") or "auto").strip().lower()
            if task_val and task_val != "auto":
                os.environ["H3_TASK"] = task_val
            else:
                os.environ.pop("H3_TASK", None)
        if not args.ref_video:
            os.environ.pop("H3_REF_VIDEOS", None)
        if not args.ref_audio:
            os.environ.pop("H3_REF_AUDIOS", None)
        os.environ.pop("H3_ALLOW_EMPTY_REFS", None)

        from h3_npu.pipeline.partition import get_serve_partition, resolve_generation_mode

        partition = get_serve_partition()
        try:
            mode = resolve_generation_mode(
                partition,
                n_images=len(args.ref_image),
                n_videos=len(args.ref_video),
                n_audios=len(args.ref_audio),
                task=job.get("task"),
            )
        except ValueError as exc:
            if st.rank == 0:
                print(f"[serve] REJECT {jid}: {exc}", flush=True)
                (root / "FAILED").write_text(jid)
            continue
        if mode == "t2va":
            os.environ["H3_TASK"] = "t2va"
            args.ref_image = []
            args.ref_video = []
            args.ref_audio = []
            os.environ.pop("H3_REF_VIDEOS", None)
            os.environ.pop("H3_REF_AUDIOS", None)
            # 避免上一单 ref2va 的 TE 缓存被 t2va 误用
            if hasattr(args, "_cond_cache"):
                delattr(args, "_cond_cache")
        if isinstance(job.get("env"), dict):
            for k, v in job["env"].items():
                os.environ[str(k)] = str(v)
        os.environ["H3_JOB_T0"] = str(time.time())
        # 任务间清碎片，避免 768→1080P 时 OOM（实测 59GiB 占满仅余 2MiB）
        gc.collect()
        empty_cache()
        if st.rank == 0:
            print(f"[serve] job {jid} mode={mode} steps={args.steps} {args.width}x{args.height}", flush=True)
            failed = root / "FAILED"
            if failed.exists():
                failed.unlink()
        try:
            run_generate(model, device, args)
        except Exception as exc:
            if st.rank == 0:
                (root / "FAILED").write_text(jid)
                print(f"[serve] FAILED {jid}: {exc}", flush=True)
            # 让其它 rank 也看到失败信号后继续等下一单
            t0 = time.time()
            while time.time() - t0 < 30:
                if (root / "FAILED").is_file() and (root / "FAILED").read_text().strip() == jid:
                    break
                time.sleep(0.05)
            gc.collect()
            empty_cache()
            continue
        (root / f"done.{st.rank}").write_text(jid)
        if st.rank == 0:
            t0 = time.time()
            while time.time() - t0 < 60:
                n = sum(1 for p in root.glob("done.*") if p.read_text().strip() == jid)
                if n >= st.world_size:
                    break
                time.sleep(0.05)
            (root / "DONE").write_text(jid)
            print(f"[serve] DONE {jid}", flush=True)
        gc.collect()
        empty_cache()


def main():
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    need = int(os.environ.get("H3_REQUIRE_WORLD", os.environ.get("WORLD_SIZE", "1")))
    if world_size < need:
        print(f"[generate] FAIL: need {need} ranks, WORLD_SIZE={world_size}", flush=True)
        raise SystemExit(2)

    # HCCL first, while HBM is empty. Loading 20 GiB then hitting the first
    # collective was the RunAicpuKfcResInitV2 path.
    rank = os.environ.get("RANK", "?")
    print(f"[generate] rank {rank} init HCCL (before DiT H2D)", flush=True)
    from h3_npu.runtime.dist import init_hccl, warmup_collective

    init_hccl()
    warmup_collective()
    if os.environ.get("H3_HCCL_ONLY", "0") == "1":
        print(f"[generate] rank {rank} HCCL-only done", flush=True)
        return

    print(f"[generate] rank {rank} loading DiT (parallel H2D)", flush=True)
    from h3_npu.pipeline.generate import load_dit_to_npu, run_generate

    model, device, args = load_dit_to_npu()
    print(f"[generate] rank {rank} DiT H2D done", flush=True)
    if os.environ.get("H3_SERVE", "0") == "1":
        if os.environ.get("H3_SERVE_PREWARM", "1") == "1":
            saved_steps = args.steps
            args.steps = 0
            # serve 预热：t2va 任务为主时用空 ref 预热，避免 DiT layout 与金毛图 ref 不一致
            from h3_npu.pipeline.partition import get_serve_partition, PARTITION_FL2VA

            prewarm_t2va = (
                get_serve_partition() == PARTITION_FL2VA
                or os.environ.get("H3_PREWARM_T2VA", "1") == "1"
            )
            if prewarm_t2va:
                os.environ["H3_TASK"] = "t2va"
                args.ref_image = []
                args.ref_video = []
                args.ref_audio = []
                if not (args.prompt or "").strip() or "<Picture" in (args.prompt or ""):
                    args.prompt = "A cinematic establishing shot, soft daylight, 24fps"
            # steps=0 still builds noise and skips the loop; caches TE+ref.
            os.environ["H3_JOB_T0"] = str(__import__("time").time())
            try:
                run_generate(model, device, args)
            finally:
                args.steps = saved_steps
            if str(rank) in ("0",):
                print("[serve] prewarm TE+ref cached", flush=True)
            if os.environ.get("H3_SERVE_DIT_WARMUP", "1") == "1":
                args.steps = 1
                os.environ["H3_JOB_T0"] = str(__import__("time").time())
                # warmup 只编译 DiT kernel，跳过 VAE，避免 1080P decode 占满 HBM
                os.environ["H3_SKIP_VAE"] = "1"
                if str(rank) in ("0",):
                    print("[serve] prewarm native DiT 1 step (skip VAE)", flush=True)
                try:
                    run_generate(model, device, args)
                finally:
                    os.environ["H3_SKIP_VAE"] = "0"
                    args.steps = saved_steps
                import gc
                from h3_npu.runtime.device import empty_cache
                gc.collect()
                empty_cache()
                if str(rank) in ("0",):
                    print("[serve] prewarm DiT done", flush=True)
        _serve_loop(model, device, args)
        return
    run_generate(model, device, args)


if __name__ == "__main__":
    main()
