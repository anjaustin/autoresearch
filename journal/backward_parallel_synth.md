# Synthesize: Parallelizing Ternary Backward — Negative Result

## What Was Attempted

OpenMP parallelization of the ternary backward kernel's inner loop over N (out_features), using per-thread static buffers + AVX2 reduction. Three variants tested:
1. malloc/free per call: 3.0x kernel speedup
2. Thread-local storage (TLS): 1.4x kernel speedup (critical section bottleneck)
3. Static buffer pool (final): 3.4x kernel speedup

## Kernel-Level Results (Isolated)

| Shape | Serial (ms) | Parallel (ms) | Speedup |
|-------|-------------|---------------|---------|
| q/o_proj (2560x2560) | 2.09 | 0.48 | 4.4x |
| k/v_proj (640x2560) | 0.40 | 0.31 | 1.3x |
| gate/up (6912x2560) | 4.18 | 1.25 | 3.3x |
| down_proj (2560x6912) | 4.91 | 1.32 | 3.7x |
| **Total (210 layers)** | **548ms** | **162ms** | **3.4x** |

Numerical validation passed (NRMSE < 2e-6) for all shapes.

## End-to-End Results (Within Autograd)

| Metric | Serial | Parallel | Change |
|--------|--------|----------|--------|
| Kernel time | 548ms | 162ms | -386ms |
| Autograd overhead | ~517ms | ~998ms | +481ms |
| **Total backward** | **~1,065ms** | **~1,160ms** | **+95ms (9% WORSE)** |

The 386ms kernel speedup was **more than negated** by a 481ms increase in system overhead.

## Root Cause Analysis

The regression is caused by **OpenMP/MKL thread pool contention**:

1. Our backward kernel spawns OpenMP threads (12) for each of the ~150 large-N calls
2. Between our kernel calls, PyTorch runs attention backward, layernorm, etc., which use MKL with its own thread pool (12 threads)
3. The rapid alternation (our OpenMP → MKL → our OpenMP → MKL × 150) causes:
   - Thread pool thrashing (both pools compete for 6 physical cores)
   - L2/L3 cache pollution (each pool's working set evicts the other's)
   - Increased kernel scheduling overhead

This is the same class of problem discovered in LMM Pass 3: on this CPU, OpenMP threading overhead dominates for small workloads. The M=1 backward per-call time (0.4–5ms) is in the zone where thread management cost exceeds the parallelism benefit in a multi-framework context.

## Variants Tested

| Configuration | End-to-End Backward |
|--------------|-------------------|
| Serial (production) | ~1,065ms |
| Parallel, OMP_NUM_THREADS=12 | ~1,160ms |
| Parallel, OMP_NUM_THREADS=6 | ~1,148ms |
| Parallel, OMP_NUM_THREADS=3 | ~1,239ms |
| Parallel, OMP_WAIT_POLICY=passive | ~1,550ms |
| Parallel, threshold N≥4096 only | ~1,563ms |

None of the parallel configurations beat serial end-to-end.

## Decision: Revert to Serial

The serial backward kernel was restored as the production path. The kernel code retains a comment documenting the parallelization attempt and explaining why serial wins.

## Key Lesson

**Micro-benchmark speedups don't translate to system-level speedups when thread pools compete.** The kernel in isolation showed 3.4x, but the system showed -9%. This is particularly acute on CPUs with moderate core counts (6C/12T) running multiple threading frameworks (OpenMP + MKL) in rapid alternation.

This contention would likely NOT occur on the Thor, where:
1. CUDA kernels don't use CPU thread pools
2. GPU parallelism is implicit (no fork/join overhead)
3. The CPU is free to run Python/autograd without competition

## What Would Actually Help

The 548ms kernel time is well-optimized (AVX2 ternary add/sub/skip, smart threading). The remaining optimization opportunities are:
1. **Bypass autograd** — the ~517ms overhead is from Python/autograd traversal. A fused backward function that handles the entire decoder layer (not just the matmul) would eliminate per-op dispatch overhead. This is a much larger engineering effort.
2. **Move to GPU** — on the Thor, native BF16 matmul will handle both forward and backward efficiently, making the CPU kernel unnecessary.
3. **torch.compile** — could potentially fuse autograd operations and reduce overhead. Not tested.
