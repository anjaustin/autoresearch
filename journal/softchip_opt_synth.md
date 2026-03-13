# Synthesis: Soft-Chip Optimization Opportunities

## Decision

Implement the **minimal viable optimization sequence** and then shift focus to PyTorch integration and numerical validation. Do NOT deep-dive into tiling or PSHUFB -- those are over-investment in a CPU path that will be superseded by GPU on Thor.

## Action Plan (Ordered)

### Step 1: Fix malloc in hot loop (5 min)
Replace `aligned_alloc`/`free` inside the OpenMP parallel region with thread-local pre-allocated buffers. This is a correctness fix (heap contention under threads) not just optimization.

### Step 2: Parallelize out_features loop (30 min)
Move the OpenMP parallelism from the batch dimension to a 2D decomposition:
- Outer: batch rows (M)
- Inner: chunks of out_features (N), divided across threads

For M=1 (autoregressive generation), this gives 6x parallelism over the current 1x. For M=19, it gives better load balancing. Use `#pragma omp parallel for collapse(2)` or manual 2D tiling.

### Step 3: Numerical validation against AutoBitLinear (1-2 hours)
Write a Python test that:
1. Loads one `AutoBitLinear` layer from the BitNet model
2. Extracts its BF16 weights and computes a forward pass through PyTorch
3. Packs the same weights via our C kernel and computes the same forward pass
4. Compares outputs element-by-element (tolerance: 1e-3 relative, 1e-5 absolute)

This is the hard gate for integration. If the outputs don't match, diagnose whether the divergence is in weight quantization, activation quantization, or accumulation order.

### Step 4: PyTorch C extension wrapper (2-3 hours)
Build `softchip/torch_ternary.py` that:
1. Compiles the C kernel via `torch.utils.cpp_extension.load()`
2. Defines a `TernaryMatmulFunction(torch.autograd.Function)` with:
   - `forward`: pack weights (cached), call C kernel, return tensor
   - `backward`: fall back to PyTorch's standard backward (STE path)
3. Provides a `patch_model(model)` function that replaces `AutoBitLinear.forward` with the soft-chip path
4. Handles weight prepacking at model load time (pack once, cache packed weights)

### Step 5: Full 30-layer benchmark (30 min)
After integration, benchmark the full model forward pass (19 tokens through all 30 layers) comparing:
- Stock PyTorch `AutoBitLinear` path (~12s)
- Soft-chip patched path (projected ~3s)

Report actual speedup and update FINDINGS.md.

## What We're NOT Doing (and Why)

| Opportunity | Why skip |
|---|---|
| Cache tiling (8 output rows) | High complexity, moderate gain, superseded by GPU on Thor |
| PSHUFB register decode | Low ROI vs LUT, adds instruction complexity, LUT fits L1 |
| AVX-512 port | Ryzen 5 5675U doesn't have AVX-512 |
| Multi-layer fusion | Requires understanding full transformer dataflow; premature |
| INT8 activation pipeline | Already implemented (matching BitNet's ActQuant) |

## Key Insight

The soft-chip's strategic value is as a **development accelerator for the Ryzen**, not as a production kernel. It lets us iterate 4x faster on the forward path during CPU development. The real performance jump comes from deploying to Thor's GPU. Therefore: get to "good enough" on CPU (Steps 1-5 above), then pivot to Thor deployment.

## Metrics for Success

| Metric | Target |
|---|---|
| Out-features parallelism for M=1 | 3-5x speedup over current single-thread |
| Numerical match vs AutoBitLinear | <1e-3 relative error per element |
| Full 30-layer forward pass | <4s (3x+ speedup over PyTorch's 12s) |
| PyTorch integration working | Forward pass drop-in replacement, backward untouched |

## Updated Next Steps (Post This LMM Pass)

1. Implement Steps 1-2 (kernel fixes) → benchmark
2. Implement Step 3 (numerical validation) → pass/fail gate
3. Implement Steps 4-5 (PyTorch integration + full benchmark)
4. Update FINDINGS.md with results
5. Commit and push
6. Begin Thor deployment planning (separate LMM pass)
