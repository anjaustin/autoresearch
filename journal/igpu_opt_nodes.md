# Nodes of Interest: iGPU Vulkan Shader Optimization

## Node 1: XOR+AND Bit Trick (Port from CPU)
Replace `float nz * sign * activation` with bitwise `(act_bits ^ sign_mask) & nz_mask` using `floatBitsToUint`/`uintBitsToFloat`. This is exactly the CPU kernel's technique. Eliminates int-to-float conversions and float multiplies from the hot path. Reduces per-weight operations to: 1 shift, 1 NEG, 2 ANDs, 1 XOR, 1 float add -- all single-cycle on Vega VALU.
**Why it matters:** Directly reduces instruction count in the innermost loop. Estimated 1.5-2x speedup from instruction throughput alone.

## Node 2: vec4 Vectorized Processing
Process 4 weights + 4 activations per loop iteration using GLSL vec4 types. Decode 8 bits (4 × 2-bit codes) into a uvec4 of codes, build vec4 masks, and do vec4 multiply-accumulate. Reduces inner loop iterations from 160 to 40 (4x fewer). On Vega, vec4 ops compile to 4 VALU instructions (not a single wide op), but the loop overhead (branch, increment, compare) is amortized 4x.
**Why it matters:** Combined with Node 1, this reduces total instruction count by ~4x. Inner loop becomes: 4 shifts, 4 NEGs, 8 ANDs, 4 XORs, 4 float adds = 24 ops per 4 weights vs current ~20 ops per 4 weights (5 per weight × 4). Marginal per-weight, but the loop overhead reduction matters.

## Node 3: Shared Memory Right-Sizing
The current shader declares `shared float s_act[6912]` (27 KB) regardless of actual in_features. For 2560×2560 layers, only 10 KB is needed. This wastes 17 KB of LDS per workgroup, limiting occupancy to 2 workgroups per CU (54 KB / 64 KB). If we size to 10 KB, we could fit 6 workgroups per CU, giving 24 wavefronts -- 60% occupancy vs current 20%.
**Why it matters:** Higher occupancy lets the GPU better hide memory latency by switching between wavefronts while one is stalled on a memory read. This is critical because our weight reads go through L2 → VRAM (shared DDR4).
**How:** Use Vulkan specialization constants to set shared memory array size at pipeline creation time. Create separate pipelines for each layer shape (2560, 6912).

## Node 4: FP16 Rapid Packed Math
Vega's Rapid Packed Math processes two FP16 operations in one FP32 clock cycle, doubling throughput to 3.23 TFLOPS. Activations could be stored in LDS as float16 (halving LDS usage to 5 KB for 2560 elements) and accumulated in FP16 pairs. BitNet already quantizes activations to INT8, so FP16 precision is more than sufficient.
**Why it matters:** 2x throughput if supported. But requires `VK_KHR_shader_float16_int8` extension which may not be available on Vega via RADV.
**Risk:** RADV may not expose FP16 storage/compute for Vega APUs. Even if exposed, the compiler may not emit packed math instructions.

## Node 5: Batched Command Buffer (All 210 Layers)
Currently we batch 7 dispatches per submission (one decoder layer). We could batch all 210 dispatches (30 decoder layers × 7) into a single command buffer, eliminating all per-layer submission overhead. For different layer sizes, we'd use different pipeline bindings within the same command buffer.
**Why it matters:** Eliminates 30 × 0.43 ms = 12.9 ms of submit overhead from the full forward pass. With 210 dispatches at 0.61 ms each = 128 ms kernel time, the overhead is 10% of total -- worth eliminating.

## Node 6: Cooperative Dot Product (Reduction Model)
Instead of one thread per output, have a wavefront (64 threads) cooperate on one dot product. Each thread processes 2560/64 = 40 elements, then subgroup reduction combines partial sums. Better weight cache behavior (one row per wavefront vs one row per thread).
**Why it matters:** Improves L2 cache utilization -- a single weight row (640 bytes) fits entirely in cache for the cooperative access pattern. Currently, 64 threads in a wavefront access 64 different rows simultaneously (40 KB of weight data per wavefront).
**Risk:** Reduces parallelism from 2560 independent outputs to 2560/64 = 40 cooperative groups. With 7 CUs and 40 groups, occupancy is low. Need more workgroups.

## Node 7: Weight Memory Layout
Current layout: row-major, each row contiguous. For the "one output per thread" model, consecutive threads access consecutive rows, which means consecutive global memory addresses are stride = packed_row_bytes apart. This is bad for coalescing (threads in a wavefront should access consecutive addresses).
**Better:** Transpose the weight layout so that the same in_features position across multiple output rows is contiguous. Then consecutive threads reading the same position of different rows would access consecutive memory.
**Why it matters:** GPU memory coalescing can deliver 10-50x better effective bandwidth than uncoalesced access.

## Tensions Summary
- **Node 1 vs Node 2:** Bit trick and vec4 are complementary, not competing. Do both.
- **Node 3 vs Node 4:** Right-sizing LDS vs FP16 storage both improve occupancy. FP16 is higher risk.
- **Node 5 vs Node 6:** Batched submission is low-effort. Cooperative dot product is a rewrite.
- **Node 6 vs Node 7:** Cooperative model wants one-row-at-a-time access. Transposed layout wants many-rows-same-position access. These conflict -- must choose one threading model.

## Dependencies
- Node 1 (bit trick) is independent -- can implement now
- Node 2 (vec4) is independent, combines with Node 1
- Node 3 (LDS sizing) requires Vulkan specialization constants -- moderate effort
- Node 4 (FP16) requires checking RADV extension support -- investigate first
- Node 5 (batched command) requires multi-pipeline command buffer -- moderate effort
- Node 6 (cooperative) is a major rewrite -- defer unless Nodes 1-3 are insufficient
- Node 7 (transposed layout) depends on choosing Node 6 -- defer
