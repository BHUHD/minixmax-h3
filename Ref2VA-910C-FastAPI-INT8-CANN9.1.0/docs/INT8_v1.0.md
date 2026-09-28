# MiniMax-H3 Ref2VA INT8 镜像 v1.0 说明

镜像：`minimax-h3:ref2va-cann9.1.0-int8-v1.0`  
基座：`minimax-h3:ref2va-cann9.1.0-int8-base`（CANN 9.1.0 + torch_npu）

---

## 1. 一键启动

在仓库根目录 `MiniMax-H3/`：

```bash
# 确认模型已在 models/ComfyUI/INI8/Ref2VA/
docker compose -f Ref2VA/CANN-9.1.0/INI8/docker-compose-v1.0.yml up -d

# 等待 NPU serve + API 就绪（约 5–8 分钟）
curl -sf http://127.0.0.1:8080/v1/status | python3 -m json.tool
```

停止：

```bash
docker compose -f Ref2VA/CANN-9.1.0/INI8/docker-compose-v1.0.yml down
```

---

## 2. 挂载约定

| 路径 | 是否必须 | 说明 |
|------|----------|------|
| `models/ComfyUI/INI8/Ref2VA` → `/models/h3_quant` | **必须** | INT8 量化权重 |
| `$HOME/minimax-h3/output` → `/workspace/output` | 可选 | compose 中**默认注释**；解开后成品 mp4 同步到宿主机 |
| Ascend driver / firmware / `davinci*` | 必须 | 机器 NPU 栈 |

代码、脚本、FastAPI、默认参考图均已打进镜像，**不再挂载**仓库 `runtime/`。

开启视频落盘（编辑 `docker-compose-v1.0.yml`，取消注释）：

```yaml
- ${H3_OUTPUT_HOST:-${HOME}/minimax-h3/output}:/workspace/output
```

任务成功后会复制为：`/workspace/output/<task_id>.mp4`。

---

## 3. 镜像内文件布局

```text
/
├── opt/h3/
│   └── entrypoint.sh              # 入口 → dual serve+API
├── workspace/                     # WORKDIR / WORKSPACE_ROOT
│   ├── src/h3_npu/                # INT8 推理运行时
│   │   ├── pipeline/              # generate / generate_worker
│   │   ├── model/                 # DiT / TE / VAE
│   │   ├── ops/                   # FA / gather 等
│   │   ├── load/                  # safetensors 加载
│   │   └── runtime/               # HCCL / device / metrics
│   ├── api/                       # FastAPI Worker
│   │   ├── app.py                 # /health /v1/status /v1/tasks*
│   │   ├── worker.py              # 串行任务 + 邮箱对接
│   │   ├── storage.py             # jobs / tombstone / 重启清理
│   │   ├── cleanup.py             # 7 天 TTL
│   │   ├── serve_bridge.py
│   │   ├── config.py / models.py
│   │   └── requirements.txt
│   ├── scripts/
│   │   ├── entrypoint_serve_api.sh  # 双进程：NPU serve + uvicorn
│   │   ├── run_serve.sh / run_generate.sh / launch_generate.py
│   │   ├── run_api.py / run_api.sh
│   │   ├── mock_serve.py            # 无 NPU 联调
│   │   ├── run_1080p10s_live.py     # 1080P 实时进度脚本
│   │   ├── submit_generate.py
│   │   └── hdk/cann_env.sh
│   ├── examples/gateway_client.py   # Gateway 调用示例
│   ├── tests/                       # 模拟 / E2E / NPU 实测
│   ├── assets/                      # 默认参考图 ref_golden_retriever.png
│   ├── jobs/                        # 任务工作区（重启清空；含 output.mp4）
│   ├── output/                      # 可选导出目录（compose 挂载后可见宿主机）
│   └── out/h3_serve/                # 16-die 文件邮箱（READY / job.json / DONE）
├── models/h3_quant/                 # 【外挂】量化权重
└── usr/local/Ascend/                # 镜像内 CANN 9.1.0（勿再挂宿主机 toolkit）
```

### 调试常用路径

| 目的 | 路径 |
|------|------|
| 看 API 日志 | `docker logs -f minimax-h3-ref2va-int8-npu16` |
| 当前任务元数据 | `/workspace/jobs/<task_id>/meta.json` |
| 步进进度 | `/workspace/jobs/<task_id>/progress.json` |
| 成品视频（容器内） | `/workspace/jobs/<task_id>/output.mp4` |
| 成品视频（若挂载 output） | `/workspace/output/<task_id>.mp4` |
| serve 邮箱 | `/workspace/out/h3_serve/{READY,job.json,DONE,FAILED}` |
| 改推理代码后验证 | 临时挂载覆盖 `/workspace/src`（生产镜像勿依赖） |

---

## 4. API 摘要

| 方法 | 路径 | 说明 |
|------|------|------|
| GET | `/health` | 进程存活 |
| GET | `/v1/status` | 就绪 / 闲忙 / `instance_id` |
| POST | `/v1/tasks` | 提交生成（multipart） |
| GET | `/v1/tasks/{id}` | 进度与终态 |
| GET | `/v1/tasks/{id}/video` | 下载 mp4 |

串行：忙时 409；未就绪 503；容器重启后旧任务 410。

---

## 5. 构建镜像

```bash
cd Ref2VA/CANN-9.1.0/INI8
docker build -t minimax-h3:ref2va-cann9.1.0-int8-v1.0 -f Dockerfile.v1.0 .
```

需本地已有 `minimax-h3:ref2va-cann9.1.0-int8-base`。

---

## 6. 实测脚本（宿主机）

```bash
cd Ref2VA/CANN-9.1.0/INI8/runtime
# 等 /v1/status 中 npu_serve_ready=true 后：
python3 tests/test_api_e2e_npu.py
# 或 1080P 盯进度：
python3 scripts/run_1080p10s_live.py --steps 20
```
