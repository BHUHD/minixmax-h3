# MiniMax-H3 Ref2VA INT8（CANN 9.1.0）

NPU 16-die 常驻推理：HTTP 提交 → 串行出片。架构细节见：[docs/自研优化代码架构.md](docs/自研优化代码架构.md)。

## 生成效果

[生成效果](https://www.bilibili.com/video/BV1LJbw6uEtc/) 类似如下：

![1080p_img2](./assets/1080p_img2.png)

## 硬件 / 驱动 / 固件 / CANN


| 项       | 要求                                           |
| ------- | -------------------------------------------- |
| 服务器     | Huawei Atlas 800I A3（或同规格）                   |
| NPU     | 8× Ascend 910C → **16 die**（每 die ≈64GB HBM） |
| 驱动 HDK  | **26.1.1**                                   |
| 固件      | **9.0.0.9.220**                              |
| 容器 CANN | **9.1.0**（镜像内；勿再挂宿主机 CANN toolkit）           |
| 主机内存    | 建议 ≥ 512GB                                   |
| Docker  | 可访问 `/dev/davinci0`…`15`                     |


确认：`npu-smi info` 可见 16 设备；`/usr/local/Ascend/driver` 与 `firmware` 已安装。

## 启动

准备ComfyUI版本的Ref2VA权重，放到本项目的 `models/ComfyUI/INI8/Ref2VA/` 目录内 ：

> models/ComfyUI/INI8/Ref2VA/
> ├── diffusion_models
> │   └── [minimax_h3_ref2va_pruned_int8_convrot.safetensors](https://huggingface.co/Comfy-Org/MiniMax-H3/resolve/main/diffusion_models/minimax_h3_ref2va_pruned_int8_convrot.safetensors)
> ├── text_encoders
> │   └── [qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors](https://huggingface.co/Comfy-Org/MiniMax-H3/resolve/main/text_encoders/qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors)
> └──vae
>     ├── [minimax_h3_audio_vae_fp32.safetensors](https://huggingface.co/Comfy-Org/MiniMax-H3/resolve/main/vae/minimax_h3_audio_vae_fp32.safetensors)
>     └── [minimax_h3_video_vae_fp16.safetensors](https://huggingface.co/Comfy-Org/MiniMax-H3/resolve/main/vae/minimax_h3_video_vae_fp16.safetensors)

执行下面命令下载镜像：

```bash
docker pull nuvic/minimax-h3:ref2va-cann9.1.0-int8-v1.1
```

在仓库根目录内执行启动命令：

```bash

# 如果要停止，则执行：
#   docker compose -f Ref2VA/CANN-9.1.0/INI8/docker-compose-v1.1.yml down
docker compose -f Ref2VA/CANN-9.1.0/INI8/docker-compose-v1.1.yml up -d


# 首次就绪约 5–10 分钟，执行下面命令测试服务是否准备就绪：
curl -sf http://127.0.0.1:8080/v1/status   # npu_serve_ready=true
```

提交示例测试：

```bash
# 文生
curl -F prompt='仙侠动漫' -F task=t2va -F width=1344 -F height=768 -F duration=5 -F steps=20 \
  http://127.0.0.1:8080/v1/tasks

# 带参考图
curl -F prompt='以 <Picture 1> 为角色…' -F task=ref2va -F 'ref_images=@ref.png' \
  -F width=1920 -F height=1088 -F duration=10 -F steps=20 http://127.0.0.1:8080/v1/tasks
```


## 推理耗时（v1.1 · 16 die）

条件：`H3_PROGRESSIVE=0`，steps=20，duration=**10s**，seed=42。  
**总时间** = 提交到 `succeeded`（不含下载）。DiT = 20 step 采样墙钟；Audio ≈ CPU Audio-VAE decode（~123s，与分辨率几乎无关）。


| 分辨率       | 场景       | 总时间 s | TE s | Ref编码 s | DiT采样 s | DiT均步 s | VAE s | Audio s | Mux/写片 s |
| --------- | -------- | ----- | ---- | ------- | ------- | ------- | ----- | ------- | -------- |
| 1920×1088 | 文生视频     | 630.2 | 17.9 | 0.0     | 447.6   | 22.37   | 22.6  | 124.2   | 15.4     |
| 1920×1088 | 参考图 1    | 650.3 | 19.3 | 15.3    | 451.9   | 22.59   | 22.1  | 124.5   | 15.1     |
| 1920×1088 | 参考图 2    | 670.3 | 23.5 | 16.7    | 451.8   | 22.58   | 22.8  | 132.8   | 15.5     |
| 1920×1088 | 图6+视3+音3 | 900.4 | 21.0 | 68.6    | 646.6   | 32.32   | 23.3  | 124.9   | 15.1     |
| 1344×768  | 文生视频     | 320.1 | 25.5 | 0.0     | 130.9   | 6.54    | 25.6  | 123.2   | 8.0      |
| 1344×768  | 参考图 1    | 340.1 | 25.1 | 16.4    | 136.4   | 6.81    | 23.2  | 123.6   | 7.8      |
| 1344×768  | 参考图 2    | 330.2 | 24.5 | 16.5    | 138.8   | 6.94    | 18.9  | 123.1   | 7.6      |
| 1344×768  | 图6+视3+音3 | 500.2 | 21.6 | 69.3    | 249.5   | 12.47   | 19.1  | 122.8   | 8.2      |


复现：`python3 Ref2VA/CANN-9.1.0/INI8/scripts/bench_ref_matrix.py --only all` , 执行完成后，输出结果在：`tmp/bench_npu_v1.1/summary.md`。

说明：1080P 峰值显存紧，连续多单建议隔单重启服务以免碎片 OOM；多参考会拉长序列，DiT 单步明显变慢（FA∝S²）。

## 文档


| 文档                                                                 | 内容                   |
| ------------------------------------------------------------------ | -------------------- |
| [docs/自研优化代码架构.md](docs/自研优化代码架构.md)                               | 调用链、量化/FA/分布式、如何改与优化 |
| [docs/INT8_v1.1.md](docs/INT8_v1.1.md)                             | v1.1 修复说明            |
| [docs/INT8_16die_deploy_guide.md](docs/INT8_16die_deploy_guide.md) | 部署验证细节               |


