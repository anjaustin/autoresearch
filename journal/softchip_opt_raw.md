# Raw Thoughts: Soft-Chip Optimization Opportunities

## Stream of Consciousness

We have a 4.1x speedup over PyTorch for one layer's forward pass. The question is: where does the remaining time go, and how much faster can we push it?

Let me think about what's actually happening in the 13.2ms per call for M=19, K=2560, N=2560.

The outer loop is over batch (M=19 rows) and out_features (N=2560 columns). That's 19 × 2560 = 48,640 dot products, each of length 2560. Total work: 19 × 2560 × 2560 = ~125 million add/compare operations. At 13.2ms that's 125M / 0.0132s ≈ 9.5 billion ops/sec, reported as 18.9 GFLOP/s equivalent. The Ryzen 5 5675U (Zen 3) can do 8 FP32 ops per AVX2 FMA per cycle at 4.4 GHz × 6 cores = ~211 GFLOP/s peak FP32. So we're at about 9% of peak. Where's the other 91%?

Several things to consider:

1. **OpenMP threading model.** The `#pragma omp parallel for` is on the batch dimension (M=19). With 6 cores and 19 rows, load balancing is uneven (3 cores get 4 rows, 3 get 3). But the real issue: 19 tasks across 12 threads (with HT) means each thread does 1-2 rows. Thread creation/synchronization overhead for ~0.7ms of work per thread is significant. For M=1 (single-token generation), the entire computation is serial -- no parallelism at all.

2. **The inner loop structure.** For each of the 2560 output features, we call `ternary_dot_v2` which iterates over 2560 input features in chunks of 8. That's 320 iterations of the inner loop. Each iteration does: 2 byte loads (LUT index), 2 × 128-bit LUT load, 2 × 256-bit insert to build 256-bit registers, 1 × XOR, 1 × AND, 1 × ADD, plus the 256-bit activation load. That's roughly 8-10 instructions per 8 elements. For 2560 elements: ~3200-4000 instructions per dot product.

3. **Memory access pattern.** The packed weight matrix is 2560 × (2560/4) = 2560 × 640 = 1,638,400 bytes ≈ 1.6MB. The L2 cache is 512KB per core (total 3MB shared). Wait -- lscpu says 3 MiB across 6 instances, so that's 512KB per core. The packed weights DON'T fit in a single core's L2. They fit in L3 (16MB). So we're running at L3 bandwidth, not L2.

   Actually wait. The access pattern matters. For each batch row, we iterate over all 2560 output rows of the weight matrix. The activations for that batch row (2560 × 4 bytes = 10KB) easily fit L1. The LUTs (8KB) fit L1. But we're streaming through 1.6MB of packed weights per batch row. At 19 batch rows, we read 19 × 1.6MB = 30.4MB of weight data total. If the weights stay in L3 after the first batch row, subsequent rows hit L3 cache (~30-50 GB/s per core on Zen 3).

4. **The out_features loop is serial within each batch row.** This is the biggest missed opportunity. Each of the 2560 dot products within a row is independent -- they could be parallelized across threads. Currently only the batch dimension (M=19) is parallelized. For M=1 (autoregressive generation), this means zero parallelism.

5. **Activation quantization overhead.** For each batch row, we do a full pass over 2560 elements to find max abs, then another full pass to quantize. That's 2 × 2560 × 4 = 20KB of reads. Small compared to the 1.6MB weight scan, but it's wasted work if we precompute the quantized activations.

6. **aligned_alloc per batch row.** We're calling aligned_alloc/free inside the hot loop for the quantized activations buffer. This is a system call on every batch row. Should be pre-allocated.

7. **The LUT approach fundamentally processes 4 weights per lookup.** With 256-bit AVX2 registers holding 8 floats, we need 2 lookups per 8 elements. An alternative: process 16 elements per iteration with 4 LUT lookups, using two 256-bit accumulators. This reduces loop overhead by 2x. Or: use a 16-bit LUT (65K entries) to process 8 elements per lookup, but that's 64K × 16 bytes = 1MB per LUT which blows L1.

8. **PSHUFB-based decode.** Instead of a memory LUT, use `vpshufb` (byte shuffle) as a 4-bit → 4-byte LUT in registers. For 2-bit weights packed 4 per byte, we could use shift+mask to isolate nibbles, then `vpshufb` to decode. This keeps everything in registers -- no memory LUT, no L1 pressure. This is the technique used by llama.cpp for its quantized matmul. The challenge: each 2-bit code needs to expand to a full 32-bit mask, so the in-register LUT approach needs careful design.

9. **Tiling for cache.** Instead of computing one output element at a time (one 2560-element dot product), we could tile: compute a block of output elements simultaneously, reusing the activation data loaded into registers. For example, process 8 output rows at once: load 8 bytes of packed weights (one from each of 8 weight rows), decode, and accumulate 8 dot products in parallel. This reads the activation vector once instead of 8 times, giving 8x data reuse.

   For N=2560 output features, processing 8 at a time means 320 tiles. Each tile streams over K=2560 input elements. The packed weights for 8 rows is 8 × 640 = 5120 bytes -- easily fits L1 alongside the 10KB activation vector and 8KB LUT.

10. **For Thor (Blackwell GPU), none of this matters.** Tensor Cores on Blackwell with FP4 support will dominate. The soft-chip is specifically a CPU-path optimization for the Ryzen (and potentially the Thor's ARM cores). On GPU we'd use cuBLAS or a custom CUDA kernel.

11. **The backward pass problem.** STE means backprop uses BF16 master weights, not ternary. The soft-chip only helps forward pass. For training, the backward pass (208s) is the bottleneck, and it doesn't benefit from ternary tricks. So the soft-chip's value is primarily for inference/rollout generation in the RL loop, not for gradient computation.

12. **PyTorch C extension integration.** To actually USE the soft-chip in the training loop, we need to replace `AutoBitLinear`'s forward path with our kernel. This means: (a) prepack all 210 layers' weights at model load time, (b) register a custom `torch.autograd.Function` that calls our C kernel for forward and falls back to PyTorch for backward, (c) handle the weight scale tracking. The `torch.utils.cpp_extension` system can JIT-compile our C code.

13. **Numerical accuracy.** We haven't validated that our kernel produces the same output as PyTorch's `AutoBitLinear`. The weight packing uses `roundf(w * scale)` which is absmean quantization matching BitNet's `WeightQuant`, and the activation quantization matches `ActQuant`, but floating-point ordering differences (SIMD reduction vs sequential) could cause small divergences. Need a numerical validation test.

## Questions Arising
- What's the actual L2 vs L3 latency profile during the dot product?
- Can PSHUFB-based decode eliminate the LUT entirely?
- How much speedup does tiling (8 output rows at once) give?
- What's the overhead of the aligned_alloc calls?
- For M=1 (autoregressive), how bad is single-threaded performance?
- Does the kernel numerically match AutoBitLinear's output?
- What's the minimum viable PyTorch integration (custom Function vs full module replacement)?

## First Instincts
- The out_features loop parallelization and tiling are the two biggest wins
- PSHUFB decode is elegant but complex; LUT approach is good enough for now
- Numerical validation should come BEFORE any PyTorch integration
- The aligned_alloc in the hot loop is a bug, fix immediately
- For autoregressive generation (M=1), threading the out_features loop is essential
- Cache tiling is the "right" optimization but also the most complex to implement
