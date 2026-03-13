# Nodes of Interest: Soft-Chip Optimization Opportunities

## Node 1: Out-Features Parallelization
The inner loop over `out_features` (N=2560) is serial within each batch row. OpenMP currently parallelizes only the batch dimension (M=19). For autoregressive generation (M=1), this means ZERO parallelism -- the entire computation runs on one core. Moving the `#pragma omp parallel for` to the out_features loop (or using a 2D decomposition) would give 6x scaling for M=1 and better load balancing for all batch sizes.
**Why it matters:** Autoregressive generation (M=1) is the primary use case for RL rollouts. Single-core performance makes the soft-chip useless for the autoresearch loop's bottleneck.

## Node 2: Cache Tiling (Multiple Output Rows)
Currently each `ternary_dot_v2` call reads 640 bytes of packed weights and 10KB of activations. The activations are reloaded from L1 for every output row. If we process 8 output rows simultaneously, we load the activation vector once and apply it to 8 different weight rows, getting 8x data reuse. Working set: 8 × 640 = 5KB weights + 10KB activations + 8KB LUTs = 23KB, comfortably in 32KB L1d.
**Why it matters:** This converts memory-bound dot products into compute-bound tiled operations, potentially doubling throughput.

## Node 3: Memory Allocation in Hot Loop
`aligned_alloc` and `free` are called inside the OpenMP parallel for loop for every batch row. These are system calls that can contend on the heap allocator's mutex under threading. The buffer is always the same size (in_features × 4 bytes = 10KB). Should be pre-allocated once and reused per-thread.
**Why it matters:** Easy fix, removes unnecessary syscall overhead.

## Node 4: PSHUFB Register-Based Decode
Replace the memory LUT with `vpshufb` (byte shuffle) to decode 2-bit packed weights entirely in registers. Pack 4 ternary values per byte, use shift+mask to isolate them, and `vpshufb` to expand to 32-bit masks. This eliminates the 8KB LUT from L1 pressure and removes memory loads from the critical path. llama.cpp uses this technique for quantized matmul.
**Why it matters:** Frees L1 capacity for weight/activation data, removes load-use latency from the inner loop. But complex to implement and the LUT already fits L1 with room to spare -- diminishing returns.

## Node 5: Numerical Validation
We haven't verified that the soft-chip kernel produces the same output as PyTorch's `AutoBitLinear`. Differences could come from: (a) weight quantization (our absmean vs BitNet's WeightQuant), (b) activation quantization (our symmetric INT8 vs BitNet's ActQuant), (c) floating-point ordering (SIMD horizontal sum vs sequential accumulation). A numerical validation test comparing per-element outputs is required before any PyTorch integration.
**Why it matters:** If the kernel doesn't match, the model will behave differently with the soft-chip, potentially invalidating training results. This BLOCKS all integration work.

## Node 6: PyTorch C Extension Integration
To use the kernel in the actual training loop: (a) prepack all 210 layers' weights at model load, (b) create a `torch.autograd.Function` subclass that calls our C kernel for forward and defers to PyTorch for backward (STE), (c) monkey-patch or replace `AutoBitLinear.forward`. The `torch.utils.cpp_extension` system can JIT-compile the C code. Alternatively, build a shared library and use `ctypes`.
**Why it matters:** The kernel is useless for training without PyTorch integration. This is the critical path to actual speedup.

## Node 7: L2/L3 Bandwidth Bound
Packed weights per layer: 1.6MB. L2 per core: 512KB. Weights don't fit in L2 for a single core. We're streaming from L3 (16MB, shared). L3 bandwidth on Zen 3: ~30-50 GB/s aggregate. At 1.6MB per batch row and 19 rows, we read ~30MB of weights total. At 40 GB/s L3 bandwidth, that's 0.75ms just for weight reads. The measured 13.2ms means we're ~18x slower than the memory bound, so we're compute-bound or instruction-bound, not memory-bound. Tiling and out_features parallelism should help.
**Why it matters:** We're leaving most of the memory bandwidth on the table. Better parallelism and tiling will get us closer to the bandwidth limit.

## Node 8: Full 30-Layer Forward Pass
One layer = 7 AutoBitLinear calls (q/k/v/o_proj + gate/up/down_proj) with varying dimensions (2560×2560 for q/o, 640×2560 for k/v, 6912×2560 for gate/up, 2560×6912 for down). Total per layer is much more than one 2560×2560 benchmark. The full model is 30 layers × 7 calls = 210 matmuls. Need to benchmark the actual total, not extrapolate from one size. The larger 6912-width layers may have different cache behavior.
**Why it matters:** Our 4.1x speedup is for one specific size. The real speedup for the full model could be higher or lower.

## Node 9: Thor Irrelevance
On the Jetson AGX Thor (Blackwell GPU, 96 Tensor Cores, FP4 support), GPU matmul will be orders of magnitude faster than any CPU kernel. The soft-chip is a CPU-only optimization. Its value is: (a) making the Ryzen 5 usable for development iteration, (b) potentially useful on Thor's ARM cores for CPU-side work, (c) demonstrating the ternary-exploit technique. It does NOT block the Thor deployment path.
**Why it matters:** Don't over-invest in CPU optimization. The soft-chip should be "good enough" on CPU, not perfect.

## Tensions Summary
- **Node 1 vs Node 9:** Out-features parallelism is essential for CPU utility, but don't over-engineer for a platform we're leaving
- **Node 2 vs Node 4:** Tiling gives data reuse, PSHUFB gives instruction efficiency -- tiling is higher ROI
- **Node 5 vs Node 6:** Numerical validation BLOCKS integration -- must be done first
- **Node 7 vs Node 1:** We're compute-bound, so parallelism > memory optimization
- **Node 8 vs Node 9:** Full model benchmarking is needed but don't goldplate the CPU path

## Dependencies
- Node 5 (numerical validation) blocks Node 6 (PyTorch integration)
- Node 1 (out_features parallelism) is independent -- can do now
- Node 3 (malloc fix) is trivial -- do first
- Node 2 (tiling) depends on understanding Node 7 (bandwidth analysis)
- Node 6 (PyTorch integration) is the critical deliverable
