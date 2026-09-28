# MiniMax-H3 Ref2VA · RTX5080 · NVFP4 · CUDA13.2
# 目标标签: nuvic/minimax-h3:ref2va-rtx5080-cuda13.2-nvfp4-v1.0.0
# 基于 FL2VA 运行时镜像，仅替换 Ref2VA 工作流与入口种子逻辑。
# 模型权重 / 输出视频不打进镜像，运行时 volume 挂载。

FROM nuvic/minimax-h3:fl2va-cuda13.2-nvfp4-v1.0.1

LABEL org.opencontainers.image.title="MiniMax-H3 Ref2VA NVFP4" \
      org.opencontainers.image.version="v1.0.0" \
      org.opencontainers.image.description="ComfyUI MiniMax-H3 Ref2VA + Sol-Attn NVFP4 for RTX5080 CUDA13.2"

# API + UI 工作流
COPY workflows/ /app/ComfyUI/workflows/
COPY bundled_workflows/ /opt/comfyui/bundled_workflows/
COPY entrypoint.sh /usr/local/bin/comfyui-entrypoint.sh

RUN chmod +x /usr/local/bin/comfyui-entrypoint.sh \
    && mkdir -p /app/ComfyUI/user/default/workflows \
    && if [ -f "/opt/comfyui/bundled_workflows/MiniMax-H3 Ref2VA 1080P10s Sol-Attn.json" ]; then \
         cp -a "/opt/comfyui/bundled_workflows/MiniMax-H3 Ref2VA 1080P10s Sol-Attn.json" \
           "/app/ComfyUI/user/default/workflows/MiniMax-H3 Ref2VA 1080P10s Sol-Attn.json"; \
       fi \
    && find /app/ComfyUI -type d -name '__pycache__' -prune -exec rm -rf {} + 2>/dev/null || true \
    && rm -rf /tmp/* /var/tmp/* /root/.cache 2>/dev/null || true

ENTRYPOINT ["/usr/local/bin/comfyui-entrypoint.sh"]
