# INT8 · 16 die · CANN 9.1.0 部署与验证指南

本指南保证：在 **相同硬件/驱动栈** 的机器上，仅依赖本仓库
`MiniMax-H3`（含本目录运行时）+ 手动下载的 `models/ComfyUI/INI8` 量化权重，
即可复现 INT8 16-die Ref2VA，**不依赖** 已删除或其它机器上的 `minimax-quant` 工程。

工作目录：

```bash
cd Ref2VA/CANN-9.1.0/INI8
```

---

## 1. 依赖边界（克隆后需要什么）

| 依赖 | 是否在本仓库 | 说明 |
|------|--------------|------|
| `runtime/src` + `runtime/scripts` | ✅ 已内置 | `h3_npu` 与 serve/submit |
| `scripts/entrypoint.sh` / `submit_generate.py` | ✅ 已内置 | 容器入口与宿主机投递 |
| `debug/input/ref_golden_retriever.png` | ✅ 已内置 | 默认参考图 |
| `models/ComfyUI/INI8/Ref2VA/*` | ❌ 需手动下载 | 大权重，见 §3 |
| Docker 镜像 `minimax-h3:ref2va-cann9.1.0-bf16` | ❌ 需加载/构建 | 见 §3 |
| 宿主机 HDK / 固件 / `/dev/davinci*` | ❌ 机器环境 | 见 §2 |

`docker-compose.yml` 与 `docker-compose-test.yml` **均只挂载本目录 `./runtime`**，不再引用任何外部源码路径。

---

## 2. 已验证软硬件栈

| 项 | 版本 / 规格 |
|----|-------------|
| 服务器 | Huawei **Atlas 800I A3**（`BC83AMDBI02-7280Z`） |
| NPU | **8× Ascend 910C**（`Ascend910` / 芯片 `9362`） |
| Die | 每卡 2 die → **16 die**；每 die **64GB HBM** |
| 驱动 HDK | **26.1.1**（`package_version=26.1.1`） |
| 固件 | **9.0.0.9.220**（`Atlas-A3-hdk-npu-firmware-9.0.0.9.220`） |
| 容器 CANN | **9.1.0**（镜像内；**勿**再挂宿主机 CANN toolkit） |
| 镜像 | `minimax-h3:ref2va-cann9.1.0-bf16` |
| FA | `H3_FA_BACKEND=infer_v2` |

### 16 die 含义

- `/dev/davinci0`…`15` 各为一颗独立 die（独立算力 + HBM）。
- 本 compose **一次生成占用全部 16 die**（序列并行 + HCCL gather），不是 16 路独立 API。
- 只用 8 die 时，另外 8 die 的算力与显存会空闲。

---

## 3. 克隆后准备（仅两项）

### 3.1 镜像

```bash
docker images | grep 'minimax-h3:ref2va-cann9.1.0-bf16'
# 若本地仅有旧 tag：
docker tag minimax-h3-quant-cann910:v1 minimax-h3:ref2va-cann9.1.0-bf16
```

### 3.2 量化模型

放到（相对本目录）：

```text
../../../models/ComfyUI/INI8/Ref2VA/
├── diffusion_models/minimax_h3_ref2va_pruned_int8_convrot.safetensors
├── text_encoders/qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors
└── vae/
    ├── minimax_h3_video_vae_fp16.safetensors
    └── minimax_h3_audio_vae_fp32.safetensors
```

文件名必须一致。详见仓库内 `models/ComfyUI/INI8/README.md`。

宿主机还需：HDK 26.1.1 + 固件 9.0.0.9.220、`npu-smi` 可见 16 die、主机内存建议 ≥ 512GB。

---

## 4. 验证步骤

```bash
cd Ref2VA/CANN-9.1.0/INI8

# 启动常驻 serve（权重 H2D + HCCL + prewarm，数分钟）
docker compose -f docker-compose.yml up -d

# 等待 READY 文件
ls debug/out/h3_serve/READY
# 或：docker logs -f minimax-h3-ref2va-int8-npu16

# 投递 10s / 1920×1088 / 20 step（与历史 hdk2611 同参）
./scripts/submit_generate.py
```

等价投递：

```bash
docker compose -f docker-compose.yml exec ref2va-int8-npu16 \
  env H3_STEPS=20 H3_SECONDS=10 H3_HEIGHT=1088 H3_WIDTH=1920 \
      H3_OUT=/workspace/out/h3_quant_generate.mp4 \
      H3_FA_BACKEND=infer_v2 \
    python /workspace/scripts/submit_generate.py
```

| 默认参数 | 值 |
|----------|-----|
| 时长 / 分辨率 / 步数 | 10s / 1920×1088 / 20 |
| Prompt | golden retriever on sunny beach + `<Picture 1>` |
| 参考图 | `debug/input/ref_golden_retriever.png` |
| 输出 | `debug/out/h3_quant_generate.mp4` + `.metrics.json` |

停止：`docker compose -f docker-compose.yml down`

> 本路径是 **文件邮箱**（`job.json` / `READY` / `DONE`），**不是** HTTP API。

---

## 5. 参考实测（同栈）

| 轮次 | DiT 20step | avg step | VAE | E2E |
|------|------------|----------|-----|-----|
| hdk2611（2026-08-23） | 441.5s | 22.07s | 23.0s | **478.6s** |
| 本仓库复测（2026-08-28） | 450.0s | 22.49s | 22.1s | **486.9s** |

复测报告：`debug/out/bench_hdk2611_replay_20260828_164133/report.md`（若保留）。  
目标 ≤300s 尚未达到；本目录用于栈与量化路径正确性复现。

---

## 6. 目录说明

```text
INI8/
├── docker-compose.yml          # 主入口（自包含）
├── docker-compose-test.yml     # 备用（同样自包含，容器名 -test）
├── runtime/                    # 随仓库的 h3_npu + serve 脚本
├── scripts/                    # entrypoint / 宿主机 submit / 辅助脚本
├── debug/input|out             # 参考图与输出
├── docs/INT8_16die_deploy_guide.md   # 本文件
└── README.md                   # 占位（另有他用）
```

---

## 7. 排障

1. **无 READY**：`docker logs minimax-h3-ref2va-int8-npu16`；检查模型文件名与 NPU 占用。  
2. **缺模型**：对照 `models/ComfyUI/INI8/README.md`。  
3. **抢卡**：先 stop 其它 `minimax-h3-*` 容器。  
4. **勿挂宿主机 CANN** 进容器（与镜像 torch_npu ABI 冲突）。  
5. **镜像缺失**：先 `docker load` / 构建并打上 `minimax-h3:ref2va-cann9.1.0-bf16`。
