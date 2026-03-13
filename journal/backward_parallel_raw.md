# Raw Thoughts: Parallelizing Ternary Backward Across Layers

## Stream of Consciousness

After the FP32 LM head fix, the backward pass takes 1,065ms. The profiler shows TernaryMatmulBackward at 755ms (71%). The LM head is 200ms (19%). "Other" is ~110ms. The ternary backward is now the single largest piece.

The 755ms comes from 210 sequential calls to `ternary_matmul_backward()`. Each call computes `grad_input = W^T @ grad_output` for one AutoBitLinear layer using the accumulate-scatter approach: iterate over N weight rows, scatter-add into grad_input.

Per-shape kernel timings (M=1, isolated):

| Shape | Serial (ms) | Count | Total (ms) |
|-------|-------------|-------|------------|
| 2560x2560 (q/o) | 2.09 | 60 | 125.6 |
| 640x2560 (k/v) | 0.40 | 60 | 24.1 |
| 6912x2560 (gate/up) | 4.18 | 60 | 250.8 |
| 2560x6912 (down) | 4.91 | 30 | 147.4 |
| **TOTAL** | | **210** | **548.0** |

The 548ms kernel-only vs 755ms profiled means ~207ms is Python/autograd overhead per iteration.

## Can we parallelize across layers?

No. Transformer backward is sequential across decoder layers because of residual connections: each layer's grad_output depends on the layer above's grad_input.

Within a single decoder layer, the backward dependency graph has TWO parallelizable groups:
1. **gate_proj + up_proj**: Both receive grad from the SiLU*up split, both write to independent grad_input buffers that get summed afterward
2. **q_proj + k_proj + v_proj**: All receive grad from attention backward, all write to independent grad_input buffers that get summed

Testing Python-level parallelism (ThreadPoolExecutor):
- gate+up parallel: 10.52ms → 6.36ms (1.65x)
- q+k+v parallel: 3.88ms → 2.84ms (1.37x)

Disappointing — thread dispatch overhead eats into gains. Each kernel is only 0.4-5ms, so even microseconds of overhead matter.

## Can we parallelize WITHIN each kernel call?

Yes! The current M=1 backward uses a serial path. The inner loop iterates over N (out_features) rows, scatter-adding each row's contribution into grad_input. We could parallelize over N with per-thread buffers + reduction.

Problem: all threads write to the SAME grad_input array → race condition.
Solution: per-thread buffers (each thread has its own grad_input), then sum at the end.

Cost analysis for 6912x2560 with 6 threads:
- Each thread processes 6912/6 = 1152 rows
- Compute: ~4.18/6 = 0.70ms per thread
- Per-thread buffer: 2560 * 4 = 10 KB (fits L1 easily)
- Reduction: sum 6 arrays of 2560 floats = 60 KB
  - 6 * 2560 adds with AVX2 = 6 * 320 * 1 cycle = 1920 cycles = ~1 μs
- Fork/join: OpenMP hot threads ~20 μs
- Expected: ~0.72ms per call (5.8x speedup)

Projection for full model with 6-thread OpenMP backward:
- 30 layers × ~3.18ms per layer = **95ms** (vs 548ms sequential = **5.7x**)

## Could we combine BOTH approaches?

OpenMP within kernel + Python thread pool for independent layers.
Problem: 6 OpenMP threads per kernel × 2 parallel kernels = 12 threads on 6 cores = oversubscribed.
Would need to coordinate: 3 threads per kernel × 2 parallel = 6 total.
But 3-thread kernel is only ~3x speedup (losing 2x from halved thread count).
Net gain of 4.1x vs 5.7x for OpenMP-only.

The simpler OpenMP-only approach (option A) gets most of the gain with less complexity.

## Memory bandwidth check

6912x2560 packed weights = 4.2 MB per layer.
With 6 threads reading simultaneously: 4.2 MB / 0.7ms = 6 GB/s effective.
System bandwidth: ~51 GB/s. We're at 12% utilization.
NOT bandwidth-limited even with 6 threads. Good.

6 threads × 10 KB per-thread buffer = 60 KB total temporary allocation per call.
Trivial. Could use stack allocation or thread-local storage.

## The autograd overhead question

The 207ms of Python/autograd overhead (755ms total - 548ms kernel) is ~28% of the backward.
If we 5.7x the kernel time (548→95ms), the overhead becomes:
- 95ms kernel + 207ms overhead = 302ms
- The overhead is now **69%** of the backward
- Further kernel optimization has diminishing returns without addressing overhead

Options for reducing overhead:
1. Fuse multiple backward calls into a single C call (batch dispatch)
2. Use torch.compile or custom backward that bypasses autograd
3. Accept it — 302ms is already fast

## Bottom line

The highest-impact, lowest-complexity move is: **add OpenMP parallelism over N within `ternary_matmul_backward()` for M=1, using per-thread buffers + reduction.**

Expected improvement: 548ms → ~95ms kernel time, total backward ~302ms.
Total training iteration: ~1200ms forward + ~302ms backward + ~200ms LM head = ~1700ms (vs 2410ms current).
