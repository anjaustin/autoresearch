# Soft-chip: ternary matmul kernels (AVX2 CPU + Vulkan iGPU) for BitNet b1.58
# Includes FP32 LM head patch for CPUs without AMX/VNNI
from softchip.torch_ternary import (
    patch_model,
    patch_lm_head_fp32,
    unpatch_model,
    unpatch_lm_head,
)
