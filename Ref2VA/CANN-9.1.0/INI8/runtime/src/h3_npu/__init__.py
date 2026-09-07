"""H3 quantized Ref2VA on Ascend 910C via torch_npu / CANN — no ComfyUI.

Stack intent (aligned with Ascend video-gen guidance):
  CANN / ACLNN → ATB → Ascend C stubs → TorchAir hooks → HCCL (optional SP)
  Primary path: torch_npu kernels (npu_quant_matmul, npu_dynamic_quant,
  npu_fusion_attention) with safe fallbacks when 910C lacks an op.

Models (container): /models/h3_quant
  ← host: MiniMax-H3/models/ComfyUI/INI8/Ref2VA
"""

__version__ = "0.1.0"
