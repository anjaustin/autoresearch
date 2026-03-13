# Synthesize: Backward Pass Optimization — FP32 LM Head Results

## Implementation

Added to `softchip/torch_ternary.py`:
- `FP32LMHeadFunction` — custom autograd function that casts BF16↔FP32 at boundaries, does all matmul in FP32
- `patch_lm_head_fp32(model)` — patches model's `lm_head` with FP32 weight copy and custom forward
- `unpatch_lm_head(model)` — restores original forward
- `unpatch_model()` now also unpatches the LM head

## Validation Results

| Test | Result |
|------|--------|
| Forward accuracy (logit RMSE) | 0.00006 — essentially identical |
| Loss difference | 0.00000000 — exact match |
| Max logit difference | 0.031 — within BF16 quantization noise |
| Gradient flow | PASS — gradients reach decoder layers through FP32 LM head |
| Weight gradient | None (correctly frozen) |

## Benchmark Results — Full Stack (soft-chip + FP32 LM head)

| Metric | Before (Pass 4) | After (Pass 5) | Speedup |
|--------|-----------------|-----------------|---------|
| Forward | 1,328ms | 1,345ms | ~same |
| Backward | 19,500ms | **1,065ms** | **18.3x** |
| Total iteration | ~20,700ms | **2,410ms** | **8.6x** |

### Backward Breakdown (profiled)

| Operation | Time | % | Notes |
|-----------|------|---|-------|
| TernaryMatmulBackward (210 layers) | 755ms | 67.9% | Ternary add/sub/skip, unchanged |
| aten::mm (FP32 LM head backward) | 200ms | 18.1% | Was 18,000ms in BF16 — **90x faster** |
| Other (fill, copy, attention, etc.) | 156ms | 14.0% | Small ops |
| **Total** | **1,111ms** | 100% | |

### Memory Cost

- FP32 LM head weight: 1,313 MB (+657 MB over BF16)
- Total model memory: ~5.2 GB → ~5.9 GB
- Available RAM: 64 GB — negligible impact

## Cumulative Optimization Journey

| Pass | What | Forward | Backward | Total |
|------|------|---------|----------|-------|
| Baseline (stock PyTorch) | All BF16 autograd | ~6,200ms | ~82,900ms | ~89,100ms |
| Pass 3 (CPU soft-chip forward) | AVX2 ternary forward | 1,328ms | 82,900ms | ~84,200ms |
| Pass 4 (ternary backward) | Ternary backward kernel | 1,328ms | 19,500ms | ~20,800ms |
| **Pass 5 (FP32 LM head)** | **FP32 cast fix** | **1,345ms** | **1,065ms** | **2,410ms** |

**Total speedup from baseline: 37x** (89,100ms → 2,410ms)

## Key Insight

The root cause was platform-specific: MKL's BF16 GEMM on Zen 3 (no AMX/VNNI) is 32-90x slower than FP32 for the same computation. This won't affect the Thor (Blackwell has native BF16 tensor cores). The fix is simple, portable, and numerically superior (FP32 is more precise than BF16).

## What This Enables

With 2.4s per training iteration:
- **25 iterations/minute** (was 2.9/minute before)
- **A 100-step GRPO training run takes ~4 minutes** (was 34 minutes)
- Fast enough for interactive experimentation

The backward is now well-balanced: 755ms ternary layers + 200ms LM head + 110ms other. No single operation dominates by more than 4x. Further optimization would require either:
1. Parallelizing ternary backward across layers (currently sequential)
2. Moving to Thor hardware (native BF16 + GPU parallelism)

## Next Steps

1. Write GRPO training loop — the full stack is now fast enough
2. Implement reward functions for RL
3. Run first TinyLoRA + GRPO experiments on BitNet b1.58
