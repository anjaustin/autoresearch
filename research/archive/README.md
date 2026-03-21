# Archive

This directory contains code and logs from completed or shelved research tracks.
Nothing here is deleted — it is preserved for reference.

## softchip/

Ghost engine and older kernel versions:

- `ghost_engine.c` / `ghost_engine.so` — Pure C/AVX2 batched inference engine for PRNG weight regeneration. Correct and debugged but ~200x slower than the real-weights path on Ryzen. Would be competitive on hardware with fast parallel PRNG (GPU).
- `ghost_engine_wrapper.py` — ctypes bridge for the ghost engine.
- `ghost_matmul.c`, `ghost_matmul_lut.c` — AVX2 PRNG weight matmul kernels (scalar and LUT variants). Used when `USE_GHOST=True`.
- `mtp18_matmul.c` — Native Multi-Trit Floating Point (MTP18) kernel. Explored in Pass 10; slower than ternary on x86 (no native base-3 SIMD).
- `ternary_matmul.c` — v1 scalar kernel prototype (221ms, 0.24x speedup — kept for history).
- `ternary_matmul_v2.c` — v2 LUT+AVX2 kernel (13.2ms, 4.1x). Superseded by v3's smart threading.
- `build_engine.sh` — Builds ghost_engine.so. Superseded by `softchip/build_kernels.sh`.
- `test_ghost_kernel.py`, `test_mtp18_vs_ghost.py` — Tests for archived kernels.

## vulkan/

Vulkan iGPU backend (Vega 7):

- `vk_backend.c` — Vulkan compute backend shared library.
- `vk_ternary*.c` — Dispatch harnesses (v2 proof-of-concept, v3 optimized).
- `*.comp` / `*.spv` — GLSL compute shaders and compiled SPIR-V.
- `test_vk_backend.py`, `test_vk_model.py`, `bench_vk_model.py` — Tests and benchmarks.

The Vulkan path achieves 0.30ms/layer (vs 1.6ms CPU) but Vulkan submit overhead (~1ms/dispatch on RADV) makes it slower end-to-end than CPU for 210 dispatches. The shader optimization work (XOR+AND bit trick, full unrolling, LDS sizing) transfers directly to CUDA where submit overhead is ~10μs.

See FINDINGS.md Pass 4 for full analysis.

## logs/

Historical training run logs from experimental iterations. `grpo_final.log` (current run) and `grpo_log.jsonl` (canonical JSONL) remain in the repo root.

## scripts/

- `math_validation.py` — Earlier version of the serial vs parallel coupling test. Superseded by `math_validation_v2.py`.
- `test_ghost_model.py`, `test_ghost_training.py` — Tests for the GhostWeight training path.
- `test_crash.py` — Crash reproduction script for a now-fixed bug.
- `test_gmoe_logic.py` — Proof-of-concept for Ghost Mixture of Experts routing. Shelved pending GhostWeight validation.
