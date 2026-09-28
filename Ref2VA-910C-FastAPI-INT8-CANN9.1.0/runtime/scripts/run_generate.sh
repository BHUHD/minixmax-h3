#!/usr/bin/env bash
# 8-die Ulysses generate inside the container. Does not modify host Ascend config.
# Parent must NOT import torch_npu (that inits all visible dies and deadlocks H2D).
set -euo pipefail
export PYTHONPATH="/workspace/src:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1
export PYTORCH_NPU_ALLOC_CONF="${PYTORCH_NPU_ALLOC_CONF:-expandable_segments:False}"
export H3_MODELS="${H3_MODELS:-/models/h3_quant}"
export H3_NPU_LAYERWISE_OFFLOAD=0
export HCCL_CONNECT_TIMEOUT="${HCCL_CONNECT_TIMEOUT:-1800}"
export HCCL_WHITELIST_DISABLE="${HCCL_WHITELIST_DISABLE:-1}"
export HCCL_OP_EXPANSION_MODE="${HCCL_OP_EXPANSION_MODE:-AIV}"
export HCCL_BUFFSIZE="${HCCL_BUFFSIZE:-1024}"
export HCCL_HOST_SOCKET_PORT_RANGE="${HCCL_HOST_SOCKET_PORT_RANGE:-auto}"
export HCCL_NPU_SOCKET_PORT_RANGE="${HCCL_NPU_SOCKET_PORT_RANGE:-auto}"
export HCCL_INTRA_PCIE_ENABLE="${HCCL_INTRA_PCIE_ENABLE:-1}"
export HCCL_INTRA_ROCE_ENABLE="${HCCL_INTRA_ROCE_ENABLE:-0}"
export TASK_QUEUE_ENABLE="${TASK_QUEUE_ENABLE:-2}"
export H3_PHY_DEVICES="${H3_PHY_DEVICES:-0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15}"
export H3_NPROC="${H3_NPROC:-16}"
export H3_SKIP_WARMUP="${H3_SKIP_WARMUP:-1}"

echo "[entrypoint] torch_npu H3 quant ${H3_NPROC}-NPU generate (stdlib launch, no parent torch)"
exec python -u /workspace/scripts/launch_generate.py "$@"
