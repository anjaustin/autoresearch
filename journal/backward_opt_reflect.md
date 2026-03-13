# Reflect: Backward Pass Optimization — Decision

## Assessment

This is one of the most clear-cut optimization decisions in the project. The data is unambiguous:

| Metric | BF16 (current) | FP32 (Option A) | Ratio |
|--------|----------------|-----------------|-------|
| LM head backward | ~9,000ms | ~90ms | 100x |
| LM head forward+backward | ~18,000ms | ~190ms | 95x |
| Full backward pass | 19,500ms | ~1,679ms | 11.6x |
| Full training iteration | ~20,700ms | ~2,879ms | 7.2x |
| Memory cost | 0.66 GB | 1.31 GB | +0.66 GB |

The root cause is hardware-specific: MKL BF16 GEMM on Zen 3 (no AMX/VNNI) uses an extremely slow fallback path. This won't be an issue on the Thor (Blackwell GPU has native BF16 tensor cores), but for development on this machine, FP32 is the correct choice.

## Decision

**Implement Option A: FP32 LM head with custom autograd function.**

Implementation plan:
1. Add `FP32LMHeadFunction` to `softchip/torch_ternary.py`
2. Add `patch_lm_head_fp32(model)` function that replaces the lm_head forward with the custom function
3. Call it from training scripts after `patch_autobitlinear(model)`
4. Validate numerical accuracy against original
5. Benchmark end-to-end

## Risk Assessment

- **Numerical drift**: FP32 is MORE precise than BF16 for this matmul. No risk of degradation.
- **Weight tying**: The FP32 copy is read-only. The original BF16 weight (shared with embedding) remains the source of truth. No tying issues.
- **Thor portability**: On Thor, BF16 will be fast natively. The FP32 patch can simply be skipped (or kept — FP32 is still fine on GPU). No portability risk.
