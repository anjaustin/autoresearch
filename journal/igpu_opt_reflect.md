# Reflect: iGPU Vulkan Shader Optimization

## What Surprised Me

The analysis reveals we're at 1.3% of peak FP32, which is much worse than the CPU's 9% of peak. The GPU should be dramatically more efficient for this workload (embarrassingly parallel, regular access pattern). The fact that it's not suggests the bottleneck is either memory latency hiding (occupancy) or instruction-level inefficiency in the shader, not fundamental compute limits.

The most unexpected insight is Node 7 (memory coalescing). I initially thought coalescing wouldn't matter because we're reading from LDS for activations. But the WEIGHT reads from global memory (VRAM) are uncoalesced: consecutive threads in a wavefront read from different weight rows at the same in_features offset. This means a 64-thread wavefront issues 64 reads to addresses stride=640 bytes apart. On a 64-byte cache line, that's 64 cache lines per wavefront access = 4 KB per read. With 160 loop iterations, that's 160 × 4 KB = 640 KB of cache line traffic per wavefront. The entire weight matrix is 1.6 MB, so we're reading ~40% of it per wavefront of 64 threads. This is terrible.

A transposed weight layout where consecutive output rows are stored with their same-position weights adjacent would turn these 64 scattered reads into 64 consecutive reads from a single cache line -- 100x less cache pressure.

But transposing changes the packing format and requires the Vulkan harness to pack differently. That's a significant change.

## Where I Was Wrong

I initially assumed the XOR+AND bit trick would be the biggest win. On reflection, it saves maybe 1-2 instructions per weight on GPU, where the cost is dominated by memory latency, not instruction count. The GPU is memory-bound, not compute-bound.

The real wins are:
1. **Memory coalescing** (Node 7) -- potentially 10-50x better effective bandwidth
2. **Occupancy** (Node 3) -- better latency hiding via more wavefronts
3. **Instruction reduction** (Nodes 1+2) -- marginal, but cumulative

This is the opposite of the CPU analysis where we were compute-bound. The GPU has abundant compute but is starved for data.

## What the Tensions Reveal

The core tension is Node 6 vs Node 7 vs the current model:

**Current model:** Each thread = one output element. Weights are row-major. Access pattern is uncoalesced. Simple to implement, but memory-inefficient.

**Cooperative model (Node 6):** Each wavefront = one output element. Threads cooperate on a dot product. Weight access is coalesced within a row. But parallelism drops from 2560 to 40 workgroups (2560/64), which is only 5.7 per CU -- borderline for hiding latency.

**Transposed model (Node 7):** Keep current threading but transpose weight layout so consecutive threads access consecutive memory. Coalesced access without reducing parallelism. But requires transposed packing.

The transposed model (Node 7) seems strictly superior: coalesced access AND high parallelism. The complexity is in the weight packing, not the shader.

## The Memory Bandwidth Analysis

At 51 GB/s shared DDR4 and 1.6 MB of weights per layer:
- Theoretical minimum read time: 1.6 MB / 51 GB/s = 31 us
- With perfect coalescing and L2 cache: much less (weights fit ~1.5x in L2)
- Current (uncoalesced): probably ~10x amplification = 310 us effective
- Our measured kernel time: ~600 us -- of which ~300 us could be memory stalls

With coalesced access (transposed layout):
- Read time drops to ~31-60 us (L2 cached after first wavefront)
- Kernel time could drop to ~100-200 us = **3-6x improvement**

## Revised Priority Ordering

1. **Transposed weight layout (Node 7)** -- biggest win, fixes memory coalescing. ~1 hour.
2. **LDS right-sizing (Node 3)** -- improves occupancy. Requires specialization constants. ~30 min.
3. **XOR+AND bit trick (Node 1)** -- reduces instruction count. ~15 min.
4. **vec4 processing (Node 2)** -- reduces loop overhead. Combines with Node 1. ~30 min.
5. **Batched command buffer (Node 5)** -- eliminates submit overhead. ~1 hour.
6. **FP16 Rapid Packed Math (Node 4)** -- investigate feasibility. High risk.
7. **Cooperative dot product (Node 6)** -- defer. Transposed layout gives coalescing without the rewrite.

Items 1-4 should be done as a single optimized shader v3. Item 5 is infrastructure. Items 6-7 are future work.

## Target Performance

With Nodes 1-4 implemented:
- Per-layer kernel time: 0.15-0.25 ms (from 0.61 ms, 2.5-4x improvement)
- Vulkan submit overhead: amortized to ~0.06 ms/layer with batched command buffer
- Full model forward (M=1): 45-65 ms (from ~140 ms projected, another 2-3x)
- vs CPU soft-chip (910 ms): **14-20x speedup**
- vs PyTorch stock (4200 ms): **65-93x speedup**

This would make 200-token rollout generation take ~10-13 seconds, down from 3 minutes on CPU or 14 minutes on stock PyTorch. That's fast enough for meaningful overnight experiment loops even on this laptop.
