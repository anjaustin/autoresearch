# Reflect: Parallelizing Ternary Backward — Decision

## Assessment

Option A (OpenMP over N within the kernel) is the clear winner:

| Option | Kernel speedup | Total backward | Complexity | Risk |
|--------|---------------|---------------|-----------|------|
| A: OpenMP over N | 5.7x (548→95ms) | ~302ms | Low (~40 LOC) | Low |
| B: Python threads | 1.4x (548→398ms) | ~605ms | Medium | Medium |
| C: A+B combined | 4.1x (548→135ms) | ~342ms | High | High |
| D: Fused C group | ~2x | ~480ms | High | Medium |
| E: Custom backward graph | 5.7x + no overhead | ~295ms | Very high | High |

Option A is the right move. The 207ms autograd overhead will become the new bottleneck after this, but addressing that (Option E) is a much larger undertaking that should be deferred until we actually need sub-300ms iteration times.

## Decision: Implement Option A

### Implementation Plan

1. **Modify `ternary_matmul_backward()` in `ternary_matmul_v3.c`:**
   - Change the M=1 serial path to use `#pragma omp parallel for reduction`
   - Each thread gets a private `grad_input` buffer (stack-allocated or malloc'd)
   - After the parallel region, sum all thread buffers into the output `grad_input`
   - Keep the M≥6 path unchanged (it already parallelizes over batch)

2. **Threshold tuning:**
   - The forward kernel uses M<6 for serial. For backward over N, the threshold is different.
   - Need to find the crossover where OpenMP overhead < parallel gain.
   - For N=640 (k/v_proj), parallel over N may not help (only 0.40ms serial).
   - For N≥2560, parallel should always win.
   - Test threshold: `N >= 2048` for OpenMP, serial below.

3. **Validation:**
   - Compare OpenMP backward output against serial backward output (should be bit-exact since FP32 addition is commutative... actually, NOT bit-exact due to different accumulation order. Need NRMSE < threshold.)
   - Run `test_backward.py` to verify.

4. **Benchmark:**
   - Per-shape kernel timing
   - End-to-end backward pass
   - Full training iteration

## Risk Assessment

- **Numerical precision:** OpenMP changes the order of floating-point additions. The result won't be bit-exact vs serial. But NRMSE should be < 1e-5 (same as existing tolerance). This is acceptable for training.
- **Thread overhead for small N:** k/v_proj at 640x2560 — only 640 rows to parallelize over. With 6 threads, that's ~107 rows/thread. Marginal gain vs overhead. Use a threshold.
- **Forward kernel interaction:** The forward kernel also uses OpenMP for M≥6. Both kernels share the same thread pool. No conflict since forward and backward don't overlap.
