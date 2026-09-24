# 在RTX5080(NVFP4 CUDA13.2)上部署MiniMax H3 FL2VA

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

在本项目根目录中，[从modelscope下载模型](https://modelscope.cn/models/weiwenying/nuvic-minimax-h3)：

```bash
modelscope download  --model weiwenying/nuvic-minimax-h3  --include 'ComfyUI/nvfp4/*'  --local_dir ./models
```

下载模型后，手动下载镜像：

```bash
docker pull nuvic/minimax-h3:fl2va-cuda13.2-nvfp4-v1.0.1
```

进入本目录：

```bash
cd ./FL2VA-RTX5080-NVFP4-CUDA13.2
mkdir -p input output
docker compose up -d
docker compose logs -f   # 健康检查通过后可 Ctrl+C
```

确认就绪：

```bash
curl -fsS http://127.0.0.1:8188/system_stats
```

浏览器打开 `http://127.0.0.1:8188` 可使用 UI。如需停止：

```bash
docker compose down
```

> Tis: 物理机挂载了目录：
>
> | 宿主机路径                | 容器路径              | 模式 | 说明                    |
> | ------------------------- | --------------------- | ---- | ----------------------- |
> | `../models/ComfyUI/nvfp4` | `/app/ComfyUI/models` | 只读 | 权重（modelscope 下载） |
> | `./input`                 | `/app/ComfyUI/input`  | 读写 | 输入                    |
> | `./output`                | `/app/ComfyUI/output` | 读写 | 输出                    |
>

## API 调用说明（标准 ComfyUI）

本服务即标准 ComfyUI HTTP API。图生视频工作流：`workflows/minimax_h3_fl2va_1080p10s_sol_tau_const.json`（提交前去掉顶层 `_meta`）。

### 常用接口

| 方法 | 路径 | 作用 |
|---|---|---|
| GET | `/system_stats` | 健康检查 / 设备信息 |
| POST | `/prompt` | 提交工作流 |
| GET | `/history/{prompt_id}` | 查询任务状态与输出 |
| POST | `/upload/image` | 上传参考图到 `input/` |
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
  "client_id": "任意字符串",
  "prompt_id": "可选-UUID"
}
```

`prompt` 为工作流节点字典。改参数只需改对应节点的 `inputs`。

### 可设置参数一览

| 参数 | 节点 | 字段 | 默认 | 说明 |
|---|---|---|---|---|
| 参考图 | `114` LoadImage | `image` | `testcase_1080p10s20step.png` | `input/` 下文件名；本部署只测**一张**参考图（`first_frame`） |
| 文本提示 | `104` MiniMaxH3ImageToVideo | `prompt` | 工作流内置中文提示 | 描述动作/对白/运镜 |
| 宽 | `w` PrimitiveInt | `value` | `1920` | 像素宽，建议 32 对齐 |
| 高 | `h` PrimitiveInt | `value` | `1088` | 像素高，建议 32 对齐 |
| 帧数（时长） | `107` PrimitiveInt | `value` | `243` | 见下方支持时长；H3 会按 `n%17==5` 对齐 |
| 采样步数 | `9` BasicScheduler | `steps` | `20` | 质量/耗时主旋钮 |
| 种子 | `15` RandomNoise | `noise_seed` | `0` | 可复现 |
| 帧率 | `90` CreateVideo | `fps` | `24` | 输出 fps |
| 输出文件前缀 | `92` SaveVideo | `filename_prefix` | 见工作流 | 相对 `output/` |
| Sol 选择 | `bsa` BlockSparseAttention | `selection` | `Sol-Attn (scheduled tau)` | 推荐保持 scheduled tau |
| tau_low | `bsa` | `selection.tau_low` | `1.0` | 头/尾步更密的 tau |
| tau_high | `bsa` | `selection.tau_high` | `1.0` | 中间步 tau；越大越稀疏越快 |
| head_steps | `bsa` | `selection.head_steps` | `0` | 前几步用 tau_low |
| tail_steps | `bsa` | `selection.tail_steps` | `0` | 后几步用 tau_low |
| sink | `bsa` | `sink_conditioning` | `exact_kv_and_rows` | 可选 `exact_kv` / `exact_kv_and_rows` / `off` |
| 采样器 | `17` KSamplerSelect | `sampler_name` | `res_multistep` | 一般不改 |
| 调度器 | `9` BasicScheduler | `scheduler` | `simple` | 一般不改 |

单张参考图时：只设置 `104.inputs.first_frame`（由节点 `114` 连入），**不要**设置 `last_frame`。

### 支持的分辨率

| 项 | 官方规格 |
|---|---|
| 宽高比 | `21:9`、`16:9`、`4:3`、`1:1`、`3:4`、`9:16`；图生视频（首/尾帧）可由输入图自适应（`adaptive`） |
| 帧率 / 音频 | 24 FPS；32 kHz 立体声 |
| 时长 | 4～15 秒 （RTX 5080 GPU上，768P/1080P只支持时长最长10s） |

典型分辨率：

| 宽高比 | 768P（短边≈768） |
|---|---|
| 21:9 | 1536×672 |
| 16:9 | **1344×768** |
| 4:3 | 1024×768 |
| 1:1 | 768×768 |
| 3:4 | 768×1024 |
| 9:16 | 768×1344 |

### Sol-tau 常用预设

| 名称 | tau_low | tau_high | head | tail | 倾向 |
|---|---|---|---|---|---|
| 常数 1.0（质量默认） | 1.0 | 1.0 | 0 | 0 | 最稳、最慢 |
| 折中 1.5 | 1.0 | 1.5 | 1 | 1 | 速度/质量折中 |
| 加速 2.0 | 1.0 | 2.0 | 3 | 2 | 更快，细结构风险上升 |

### API测试：Python

本目录已提供客户端与矩阵测试：

```bash
cd ./FL2VA-RTX5080-NVFP4-CUDA13.2/

cp ./assets/input.png ./input/

# 单次生成
python scripts/api_client.py \
  --image input.png \
  --width 1920 --height 1088 --frames 243 --steps 20 \
  --tau-low 1.0 --tau-high 1.0 --head-steps 0 --tail-steps 0 \
  --seed 42 \
  --prefix demo_1080p10s \
  --cn-name 单参考图_1080P_10s_tau常数1.0.mp4

# 遍历 分辨率×时长×tau（18 组），中文命名输出到 output/
python scripts/run_matrix_test.py
```

`api_client.py` 参数与上表一一对应；`--cn-name` 会在完成后把成片复制为中文文件名。最小自写示例：

```python
import copy, json, uuid, urllib.request
from pathlib import Path

BASE = "http://127.0.0.1:8188"
wf = json.loads(Path("./assets/comfyui-api-workflow-minimax_h3_fl2va_1080p10s_sol_tau_const.json").read_text())
prompt = {k: copy.deepcopy(v) for k, v in wf.items() if k != "_meta"}
prompt["114"]["inputs"]["image"] = "input.png"
prompt["w"]["inputs"]["value"] = 1344
prompt["h"]["inputs"]["value"] = 768
prompt["107"]["inputs"]["value"] = 124
prompt["9"]["inputs"]["steps"] = 20
prompt["15"]["inputs"]["noise_seed"] = 42
prompt["92"]["inputs"]["filename_prefix"] = "api_demo"
# Sol-tau
bsa = prompt["bsa"]["inputs"]
bsa["selection"] = "Sol-Attn (scheduled tau)"
bsa["selection.tau_low"] = 1.0
bsa["selection.tau_high"] = 1.0
bsa["selection.head_steps"] = 0
bsa["selection.tail_steps"] = 0

body = json.dumps({"prompt": prompt, "client_id": "demo", "prompt_id": str(uuid.uuid4())}).encode()
req = urllib.request.Request(f"{BASE}/prompt", data=body, headers={"Content-Type": "application/json"})
print(urllib.request.urlopen(req).read().decode())
```

轮询：`GET /history/{prompt_id}`，直到 `status.status_str` 为 `success` / `error`。成片路径在 `outputs` 里，或直接看宿主机 `output/`。

## 附录

### 实测效果

#### 测试环境

| 项 | 值 |
|---|---|
| GPU | RTX 5080 16GB |
| 镜像 | `nuvic/minimax-h3:fl2va-cuda13.2-nvfp4-v1.0.1` |
| steps | 20 |
| seed | 42 |
| fps | 24 |
| 参考图 | `./assets/input.png` |
| sink | `exact_kv_and_rows` |
| 实测日期 | 2026-09-23 |

- **默认质量档**（1080P × 10s × tau常数1.0）：约 **1030s（~17.2min）**，与调试基线一致。
- **同分辨率下**，tau加速2.0 相对常数1.0 约快 **10%–25%**；折中1.5 介于两者之间。
- **768P × 10s × 常数1.0** 约 **378s（~6.3min）**，约为 1080P 同配置的约 37%。
- 分辨率、时长、tau 三组参数均可经标准 `/prompt` API 正常覆盖设置。

#### 耗时汇总表

| 分辨率 | 时长 | 帧数 | $τ$ 预设 | 耗时(s) | 耗时(m) | 中文成片 |
|---|---|---:|---|---:|---:|---|
| 768P | 5s | 124 | tau常数1.0 | 168 | 2.8 | [单参考图_768P_5s_tau常数1.0_耗时168秒.mp4](bilibili.com/video/BV1gDhf6wEnU/) |
| 768P | 5s | 124 | tau折中1.5 | 159 | 2.7 | [单参考图_768P_5s_tau折中1.5_耗时159秒.mp4](bilibili.com/video/BV1Qeaw6zEAr/) |
| 768P | 5s | 124 | tau加速2.0 | 153 | 2.6 | [单参考图_768P_5s_tau加速2.0_耗时153秒.mp4](bilibili.com/video/BV1Qeaw6zEFn/) |
| 768P | 8s | 192 | tau常数1.0 | 285 | 4.8 | [单参考图_768P_8s_tau常数1.0_耗时285秒.mp4](bilibili.com/video/BV1QYaw6gEj4/) |
| 768P | 8s | 192 | tau折中1.5 | 255 | 4.3 | [单参考图_768P_8s_tau折中1.5_耗时255秒.mp4](https://www.bilibili.com/video/BV1Deaw6zEXR/) |
| 768P | 8s | 192 | tau加速2.0 | 246 | 4.1 | [单参考图_768P_8s_tau加速2.0_耗时246秒.mp4](bilibili.com/video/BV1QYaw6gEje/) |
| 768P | 10s | 243 | tau常数1.0 | 378 | 6.3 | [单参考图_768P_10s_tau常数1.0_耗时378秒.mp4](bilibili.com/video/BV1heaw66E2b/) |
| 768P | 10s | 243 | tau折中1.5 | 339 | 5.7 | [单参考图_768P_10s_tau折中1.5_耗时339秒.mp4](bilibili.com/video/BV1Qeaw6zE3f/) |
| 768P | 10s | 243 | tau加速2.0 | 321 | 5.4 | [单参考图_768P_10s_tau加速2.0_耗时321秒.mp4](https://www.bilibili.com/video/BV13eaw66Ehd/) |
| 1080P | 5s | 124 | tau常数1.0 | 423 | 7.1 | [单参考图_1080P_5s_tau常数1.0_耗时423秒.mp4](https://www.bilibili.com/video/BV1Qeaw6zEKh/) |
| 1080P | 5s | 124 | tau折中1.5 | 372 | 6.2 | [单参考图_1080P_5s_tau折中1.5_耗时372秒.mp4](bilibili.com/video/BV1Qeaw6zE3f/) |
| 1080P | 5s | 124 | tau加速2.0 | 351 | 5.9 | [单参考图_1080P_5s_tau加速2.0_耗时351秒.mp4](https://www.bilibili.com/video/BV1Deaw6zEUV/) |
| 1080P | 8s | 192 | tau常数1.0 | 735 | 12.3 | [单参考图_1080P_8s_tau常数1.0_耗时735秒.mp4](bilibili.com/video/BV1Qeaw6zEcy/) |
| 1080P | 8s | 192 | tau折中1.5 | 618 | 10.3 | [单参考图_1080P_8s_tau折中1.5_耗时618秒.mp4](https://www.bilibili.com/video/BV13eaw66ExE/) |
| 1080P | 8s | 192 | tau加速2.0 | 573 | 9.6 | [单参考图_1080P_8s_tau加速2.0_耗时573秒.mp4](https://www.bilibili.com/video/BV1Qeaw6zEGA/) |
| 1080P | 10s | 243 | tau常数1.0 | 1029 | 17.2 | [单参考图_1080P_10s_tau常数1.0_耗时1030秒.mp4](bilibili.com/video/BV1Qeaw6zEGu/) |
| 1080P | 10s | 243 | tau折中1.5 | 843 | 14.1 | [单参考图_1080P_10s_tau折中1.5_耗时843秒.mp4](https://www.bilibili.com/video/BV1Qeaw6zEAe/) |
| 1080P | 10s | 243 | tau加速2.0 | 771 | 12.9 | [单参考图_1080P_10s_tau加速2.0_耗时771秒.mp4](bilibili.com/video/BV1gDhf6wEUU/) |

#### 如果要复现

```bash
cd FL2VA-RTX5080-NVFP4-CUDA13.2
docker compose up -d
python scripts/run_matrix_test.py
ls -lh output/单参考图_*.mp4
cat output/results_summary.json
```
