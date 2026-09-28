# 在 RTX5080 (NVFP4 CUDA13.2) 上部署 MiniMax H3 Ref2VA

`编制：魏文应`  `时间：2026-09-28`

---

## 概述

视频生成，实测总耗时：

| #    | 输入组合                   | 分辨率            | 状态    | 总耗时 (s) | 总耗时 (min) | 成片                                                         |
| ---- | -------------------------- | ----------------- | ------- | ---------: | -----------: | ------------------------------------------------------------ |
| 1    | prompt + 6图 + 3音 + 3视频 | 768P (1344×768)   | success |    1052.21 |           18 | [视频1.mp4](https://www.bilibili.com/video/BV1Eoa361EtY/)    |
| 1    | prompt + 6图 + 3音 + 3视频 | 1080P (1920×1088) | **OOM** |      81.05 |            — | —                                                            |
| 2    | prompt + 6图 + 3音 + 1视频 | 768P              | success |     637.75 |           11 | [视频2.mp4](https://www.bilibili.com/video/BV1Eoa361EqV/)    |
| 2    | prompt + 6图 + 3音 + 1视频 | 1080P             | success |    1449.40 |           24 | [视频3.mp4](https://www.bilibili.com/video/BV1Eoa361Eiq/)    |
| 3    | prompt + 1图               | 768P              | success |     368.01 |            6 | [视频3.mp4](https://www.bilibili.com/video/BV17oa361Eos/)    |
| 3    | prompt + 1图               | 1080P             | success |    1019.87 |           17 | [视频4.mp4](https://www.bilibili.com/video/BV1Eoa361EVx/)    |
| 4    | prompt-only                | 768P              | success |     340.85 |            6 | [视频5.mp4](https://www.bilibili.com/video/BV1Eoa361Eqa/?vd_source=3864c9f21aada39c764f51eff1ef53c6) |
| 4    | prompt-only                | 1080P             | success |     926.70 |           16 | [视频6.mp4](https://www.bilibili.com/video/BV1Eoa361EEE/)    |

> **OOM 说明**：组合 1 @1080P（6 图 + 3 音 + 3 段参考视频）在 `SamplerCustomAdvanced` 报 `torch.OutOfMemoryError`（需再申请约 5.16 GiB，当时仅余约 2.29 GiB）。同组合 @768P 可跑通。16GB 卡上满参考建议优先 768P，或减少参考视频段数/分辨率。

## 部署

安装依赖：

```bash
pip install modelscope  # 用于模型下载
```

下载本项目：

```bash
git clone https://github.com/BHUHD/minixmax-h3.git
cd ./minixmax-h3/
```

在本项目根目录中，[从 modelscope 下载模型](https://modelscope.cn/models/weiwenying/nuvic-minimax-h3)：

```bash
# 确保包含 Ref2VA 权重：
# 	minimax_h3_ref2va_pruned_nvfp4.safetensors
# 	qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors
# 	minimax_h3_video_vae_fp16.safetensors
# 	minimax_h3_audio_vae_fp32.safetensors
modelscope download --model weiwenying/nuvic-minimax-h3 --include 'ComfyUI/nvfp4/*' --local_dir ./models
```

拉取镜像：

```bash
docker pull nuvic/minimax-h3:ref2va-rtx5080-comfyui-nvfp4-cuda13.2-v1.0.0
```

启动：

```bash
cd ./Ref2VA-RTX5080-NVFP4-CUDA13.2
mkdir -p output
docker compose up -d
docker compose logs -f   # 健康检查通过后可 Ctrl+C
```

确认就绪：

```bash
curl -fsS http://127.0.0.1:8188/system_stats
```

浏览器打开 `http://127.0.0.1:8188`。停止：

```bash
docker compose down
```

> Tip: 物理机仅挂载两类目录：
>
> | 宿主机路径 | 容器路径 | 模式 | 说明 |
> |---|---|---|---|
> | `../models/ComfyUI/nvfp4` | `/app/ComfyUI/models` | 只读 | 权重（modelscope 下载） |
> | `./output` | `/app/ComfyUI/output` | 读写 | 成片输出 |
>
> `input` / `temp` / `user` 留在容器内。上传参考图/音请用 ComfyUI `/upload/image` 等接口。

## API 调用说明（标准 ComfyUI）

参考生视频工作流：[minimax_h3_ref2va_1080p10s_sol_tau_const.json](Ref2VA-RTX5080-ComfyUI-NVFP4-CUDA13.2/assets/workflows/minimax_h3_ref2va_1080p10s_sol_tau_const.json)。

### 常用接口

| 方法 | 路径 | 作用 |
|---|---|---|
| GET | `/system_stats` | 健康检查 / 设备信息 |
| POST | `/prompt` | 提交工作流 |
| GET | `/history/{prompt_id}` | 查询任务状态与输出 |
| POST | `/upload/image` | 上传参考图到容器内 `input/` |
| POST | `/free` | 卸载模型、释放显存 |
| GET | `/view?filename=...&subfolder=...&type=output` | 下载成片 |

### 提交体结构

```http
POST /prompt
Content-Type: application/json
```

```json
{
  "prompt": { "...节点图..." },
  "client_id": "任意字符串"
}
```

### 关键节点（Ref2VA）

| 参数 | 节点 | 字段 | 说明 |
|---|---|---|---|
| 参考图 | `img*` LoadImage | `image` | 容器 `input/` 下路径 |
| 参考音 | `aud*` LoadAudio | `audio` | 容器 `input/` 下路径 |
| 提示词 | `138` PrimitiveStringMultiline | `value` | 使用 `<Picture i>` / `<Audio j>` / `<Video k>` |
| Ref2VA | `136` MiniMaxH3ReferenceToVideo | — | 参考图/音/视频接线 |
| 宽/高 | `w` / `h` | `value` | 默认 `1920` / `1088` |
| 时长 | `132` PrimitiveFloat | `value` | 秒，默认 `10` |
| 步数 | `143` PrimitiveInt | `value` | 默认 `20` |
| Sol-Attn | `bsa` BlockSparseAttention | `selection.tau_*` | 默认常数 `1.0` |
| 种子 | `129` RandomNoise | `noise_seed` | 可复现 |
| 输出前缀 | `92` SaveVideo | `filename_prefix` | 相对 `output/` |

参考素材上限（模型约束）：图 ≤9、视频 ≤3、独立音频 ≤3；单段音频约 2–15s，总音频 ≤15s。

### 支持的分辨率 / 时长（与 H3 一致）

| 项 | 说明 |
|---|---|
| 分辨率 | 常用 `864×480`、`1344×768`、`1920×1088`（32 对齐） |
| 帧数 | `n % 17 == 5`；约 `124≈5s`、`192≈8s`、`243≈10s` @24fps |
| 采样 | `res_multistep` + `simple`；推荐 20 step |

## 附录

### 镜像说明

| 项       | 值                                                           |
| -------- | ------------------------------------------------------------ |
| 镜像     | `nuvic/minimax-h3:ref2va-rtx5080-cuda13.2-nvfp4-v1.0.0`      |
| 基座     | `nuvic/minimax-h3:fl2va-cuda13.2-nvfp4-v1.0.1`               |
| UNET     | `minimax_h3_ref2va_pruned_nvfp4.safetensors`                 |
| 启动参数 | `--disable-cuda-malloc` + `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` |
| Sol-Attn | API 工作流默认 `tau_low=tau_high=1.0`                        |

### 实测

| 条目 | 说明                                                         |
| ---- | ------------------------------------------------------------ |
| 镜像 | `nuvic/minimax-h3:ref2va-rtx5080-cuda13.2-nvfp4-v1.0.0`      |
| 设备 | RTX 5080 16GB · CUDA 13.2 · Sol-Attn `tau=1.0`   10s / 20 step |

测试输入在 `assets/test_input/`：

| 类型 | 内容 |
|---|---|
| 6 张图 | `images/00_首帧.png` + 五角色图（陆清川/玄霜真人/苏晚晴/叶芷若/陈砚舟） |
| 3 路音 | `audios/` 陆清川、苏晚晴、叶芷若 |
| 3 段视频 | 由 `1080p10s20step_hunt_v2/out.mp4` 切成约 3.3s×3，运行时用 **864×480** 版（`ref_video_*.mp4`）；原 1080 切段保留为 `*_src1080.mp4` |
| Prompt | `prompt.txt`（多参考）/ `prompt_1img.txt` / `prompt_text_only.txt` |

墙钟按 ComfyUI websocket `executing` 事件累计；**总耗时**以提交→history 完成为准（含 free/排队间隙时可能略大于模块之和）。模块耗时明细（秒）：

| 组合 | 分辨率 | sampler | vae_decode | ref2va_encode | mux_save | vae_load | load_inputs | text_encoder_load | unet_load | **总耗时** |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| prompt-only | 768P | 282.81 | 45.27 | 7.52 | 2.74 | 1.89 | — | 0.45 | 0.13 | **340.85** |
| prompt-only | 1080P | 818.19 | 94.93 | 7.28 | 4.76 | 0.95 | — | 0.41 | 0.16 | **926.70** |
| prompt+1图 | 768P | 307.65 | 45.13 | 10.56 | 2.95 | 1.01 | 0.08 | 0.41 | 0.13 | **368.01** |
| prompt+1图 | 1080P | 906.86 | 94.98 | 11.40 | 4.93 | 1.02 | 0.08 | 0.44 | 0.14 | **1019.87** |
| prompt+6图+3音+1视频 | 768P | 551.55 | 45.06 | 36.21 | 2.84 | 0.94 | 0.54 | 0.42 | 0.14 | **637.75** |
| prompt+6图+3音+1视频 | 1080P | 1307.23 | 94.80 | 40.49 | 4.77 | 0.91 | 0.53 | 0.41 | 0.20 | **1449.40** |
| prompt+6图+3音+3视频 | 768P | 930.74 | 45.09 | 70.42 | 2.93 | 1.27 | 1.14 | 0.44 | 0.14 | **1052.21** |
| prompt+6图+3音+3视频 | 1080P | 2.52* | — | 75.85 | — | 0.91 | 1.16 | 0.42 | 0.13 | **OOM (81s)** |

模块含义：

| 模块 | 主要节点 |
|---|---|
| `sampler` | `SamplerCustomAdvanced`（含 Sol-Attn 采样步） |
| `vae_decode` | `VAEDecode` + `VAEDecodeAudio` |
| `ref2va_encode` | `MiniMaxH3ReferenceToVideo`（文本/参考图视频音频编码与 conditioning） |
| `mux_save` | `CreateVideo` + `SaveVideo` |
| `vae_load` / `unet_load` / `text_encoder_load` | 对应 Loader（冷启动或 free 后会更高） |
| `load_inputs` | `LoadImage` / `LoadAudio` / `LoadVideo` / `GetVideoComponents` |

结论摘要：

- 镜像可正常拉起并完成多数 Ref2VA 组合；**耗时瓶颈几乎全在 sampler**（约占总时间 80%–90%）。
- 参考越多，`ref2va_encode` 与 sampler 同步变长（3 段视频 @768P 编码约 70s，1 段约 36s，纯文本约 7s）。
- 1080P 相对 768P：同组合总耗时约 **2.3×–2.7×**。
- 16GB 上 **6图+3音+3视频@1080P 会 OOM**；可用 768P，或减到 1 段参考视频（1080P 已验证成功，约 24 分钟）。
