# MiniMax-H3 Ref2VA INT8 v1.1

镜像：`minimax-h3:ref2va-cann9.1.0-int8-v1.1`  
基座：`minimax-h3:ref2va-cann9.1.0-int8-v1.0`（再覆盖干净 `runtime/`，非 `docker commit`）

## 相对 v1.0 的修复

1. **TE NVFP4 `from_blocked`**：block-scale 反 swizzle，对齐 Comfy `comfy_kitchen`；修复零参考 t2va 文本语义跑偏。
2. **初始噪声**：CPU `float32` `randn` 再 cast 到 bf16，对齐 Comfy `prepare_noise`；声画与同 seed GPU 更一致。
3. **AAC**：mux 码率 `128k`，对齐 Demo SaveVideo。
4. **t2va**：Ref2VA 分区支持 `task=t2va` / `auto` 无参考；禁止误用 TE 缓存与默认参考图。

## 构建

```bash
cd Ref2VA/CANN-9.1.0/INI8
docker build -t minimax-h3:ref2va-cann9.1.0-int8-v1.1 -f Dockerfile.v1.1 .
```

## 启动

```bash
docker compose -f Ref2VA/CANN-9.1.0/INI8/docker-compose-v1.1.yml up -d
```
