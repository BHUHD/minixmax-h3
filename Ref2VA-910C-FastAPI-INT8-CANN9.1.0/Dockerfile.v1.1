# MiniMax-H3 Ref2VA INT8 API 服务镜像 v1.1
# 基于 v1.0，用宿主机干净 runtime 覆盖 bake 代码（勿 docker commit 运行中容器）
#
# 相对 v1.0 主要修复：
#   - TE NVFP4 block-scale from_blocked（零参考 t2va 文本对齐）
#   - 初始噪声对齐 Comfy：CPU float32 randn → bf16
#   - AAC 128k；Ref2VA 分区支持 task=t2va / auto 无参考
#
# 构建（在本目录 INI8/）:
#   docker build -t minimax-h3:ref2va-cann9.1.0-int8-v1.1 -f Dockerfile.v1.1 .
FROM minimax-h3:ref2va-cann9.1.0-int8-v1.0

LABEL description="MiniMax-H3 Ref2VA INT8 16-die + FastAPI Worker (CANN 9.1.0) v1.1"
LABEL version="1.1"
LABEL ascend.cann="9.1.0"
LABEL h3.changelog="nvfp4 from_blocked; fp32 noise; aac128k; t2va zero-ref"

# 仅覆盖运行时代码；模型仍外挂。不拷贝 jobs/out/缓存等运行残留。
COPY runtime/src /workspace/src
COPY runtime/api /workspace/api
COPY runtime/scripts /workspace/scripts
COPY runtime/examples /workspace/examples
COPY runtime/tests /workspace/tests
COPY debug/input /workspace/assets
COPY scripts/entrypoint.sh /opt/h3/entrypoint.sh

WORKDIR /workspace
RUN chmod +x /opt/h3/entrypoint.sh \
      /workspace/scripts/*.sh \
      /workspace/scripts/*.py \
    && find /workspace -type d -name __pycache__ -prune -exec rm -rf {} + 2>/dev/null || true \
    && find /workspace -type f -name '*.pyc' -delete 2>/dev/null || true

ENTRYPOINT ["/opt/h3/entrypoint.sh"]
