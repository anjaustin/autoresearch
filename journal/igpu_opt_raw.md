# Raw Thoughts: iGPU Vulkan Shader Optimization for Vega 7

## Stream of Consciousness

We have a naive shader doing 0.61 ms/layer (amortized, 2560x2560, M=1) on the Vega 7 iGPU with 7 CUs. That's 21.6 GFLOP/s equivalent. The GPU has 1,610 GFLOP/s peak FP32. We're at 1.3% of peak. There's enormous headroom.

Let me think about what the current shader is actually doing per invocation:

Each invocation computes one output element -- a 2560-element dot product. It reads 160 uint32s of packed weights (2560/16 = 160 words) and 2560 floats of activation from shared memory. The inner loop processes 16 weights per uint32, doing 16 shift+mask+int-to-float+multiply+add operations per iteration. That's 160 iterations of 16 operations each.

On Vega, each CU has 4 SIMD-16 units. A wavefront is 64 threads. With 256 threads per workgroup and 7 CUs, we dispatch ceil(2560/256) = 10 workgroups. Only 7 can run simultaneously (one per CU). So we need 2 waves of workgroups. Each CU processes one workgroup of 256 threads = 4 wavefronts.

**Problem 1: Occupancy.** Each workgroup uses shared memory for the activation array: 6912 * 4 = 27,648 bytes. Vega has 64 KB LDS per CU. So we can fit 2 workgroups per CU (54 KB out of 64 KB). With 256 threads per workgroup = 4 wavefronts, that's 8 wavefronts per CU. Vega supports up to 40 wavefronts per CU (10 per SIMD unit), so we're at 8/40 = 20% wavefront occupancy. This limits the GPU's ability to hide memory latency.

We can improve occupancy by:
- Reducing shared memory per workgroup (use only what's needed: 2560*4 = 10 KB for the common case)
- Or increasing workgroup count by reducing workgroup size

Actually, the `shared float s_act[6912]` declaration reserves 27 KB even when in_features = 2560 (only needs 10 KB). The Vulkan spec doesn't support dynamic shared memory sizing like CUDA's `extern __shared__`. But we could use a specialization constant to set the shared memory size at pipeline creation time. Or just declare `shared float s_act[2560]` for the common case and create separate pipelines for different layer sizes.

**Problem 2: Scalar inner loop.** Each thread processes weights one at a time. Vega's SIMD-16 units execute 16 threads in lockstep. All 16 threads are doing the same operation (shift+mask+add) on different weight rows. This is efficient in terms of SIMD utilization -- no divergence. But the per-element work is: 1 shift, 1 AND, 1 int-to-float, 1 multiply, 1 FMA. That's 5 instructions per weight, and 16 weights per uint32 = 80 instructions per loop iteration. With 160 iterations, that's 12,800 instructions per invocation.

At 1800 MHz, one SIMD-16 instruction takes 4 cycles. So 12,800 instructions = 12,800 * 4 = 51,200 cycles per wavefront. At 1800 MHz, that's 28.4 us per wavefront. With 4 wavefronts per workgroup: 113.6 us. With 10 workgroups across 7 CUs (2 waves): ~2 * 113.6 us / (7/10 overlap) ≈ ~325 us compute time.

But our measured time is 610 us, roughly 2x the compute estimate. The gap is likely from:
- Memory latency for weight reads (packed_w from VRAM, ~100-200ns per cache miss)
- Insufficient occupancy to hide that latency
- Vulkan dispatch overhead (even in batched mode)

**Problem 3: Weight reads.** Each invocation reads 160 uint32 = 640 bytes of packed weights from global memory. With 2560 invocations, that's 2560 * 640 = 1.6 MB of weight reads total. At ~40-50 GB/s effective VRAM bandwidth (shared DDR4), reading 1.6 MB takes ~32-40 us. But the reads are scattered across the weight matrix (each thread reads a different row), so L2 cache helps. The entire packed weight matrix is 1.6 MB which fits in Vega's ~1 MB L2 with some spillage. After the first workgroup warms the cache, subsequent workgroups may get partial L2 hits.

The activation reads are from LDS (shared memory) -- essentially free (1 cycle latency).

**Optimization 1: vec4 loads.** Process 4 activations at a time using vec4 types. Instead of loading s_act[idx] one float at a time, load vec4 chunks. This quadruples the arithmetic intensity per memory access.

But wait -- the weights are 2-bit packed, not floating point. We'd need to decode 4 weights simultaneously to a vec4 mask. We can: extract 8 bits (4 codes) from the packed word, convert each 2-bit code to a float mask component, then multiply the activation vec4 by the mask vec4. This halves the number of inner loop iterations (from processing 1 at a time to 4 at a time).

Actually, even better: process 4 weights into a vec4 sign/mask, then do one vec4 multiply-accumulate:
```glsl
uvec4 codes = uvec4((word >> 0) & 3, (word >> 2) & 3, (word >> 4) & 3, (word >> 6) & 3);
vec4 nz = vec4(codes & 1u);
vec4 signs = vec4(1.0) - 2.0 * vec4(codes >> 1u);
vec4 act4 = vec4(s_act[base_k], s_act[base_k+1], s_act[base_k+2], s_act[base_k+3]);
acc4 += act4 * nz * signs;
```

This reduces loop iterations from 160 to 40 (processing 4 at a time from 16 per uint, so 4 groups of 4 per uint, 160/4=40). The vec4 ops should compile to 4-wide VALU instructions.

**Optimization 2: Multiple accumulations per thread.** Instead of one output per invocation, each invocation could compute 2 or 4 outputs. This increases register usage but doubles/quadruples the weight data reuse from L2 cache. However, we're already using the "one output per thread" model which maps well to the wavefront model.

Actually, the better direction is the opposite: have multiple threads cooperate on a SINGLE output, then reduce. This is the classic "reduction" pattern. Each invocation processes a portion of the 2560-element dot product, then a subgroup reduction (shuffle-add) combines partial sums. With wavefront size 64, we could have 64 threads each process 2560/64 = 40 elements, then reduce. This gives much better memory coalescing for the weight reads (all threads in a wavefront read adjacent portions of the same weight row).

Wait, that changes the threading model completely. Currently: 2560 invocations, each doing a full 2560-element dot product. Alternative: 2560 * 64 = 163,840 invocations, each doing 40 elements, then reduce per wavefront. This would be 2560 workgroups of 64, or 640 workgroups of 256. Way too many workgroups.

The standard efficient approach for GPU matrix-vector multiply (GEMV) at M=1:
- Each workgroup computes one or a few output elements
- Within the workgroup, threads cooperate on the dot product via parallel reduction
- LDS is used to store partial sums for the intra-workgroup reduction

For a 2560-element dot product with 256 threads per workgroup:
- Each thread processes 2560/256 = 10 elements
- Then 256-thread reduction via shared memory to get the final sum
- This gives excellent memory coalescing: consecutive threads read consecutive weight values

This is fundamentally different from the current shader. Let me think about which approach is better.

Current: 256 invocations per workgroup, each doing a full 2560-element dot product independently. All 256 threads read the SAME activation vector from LDS but DIFFERENT weight rows. Weight reads are coalesced within a wavefront because consecutive threads read consecutive weight rows (consecutive memory addresses in the packed weight buffer).

Alternative (reduction): 256 invocations per workgroup, cooperating on a SINGLE dot product. All 256 threads read DIFFERENT chunks of the SAME weight row and the SAME activation vector. Needs reduction at the end.

The current approach has better parallelism (256 independent outputs vs 1 output with reduction), but the alternative has better cache behavior for weights (all threads read from a single row which is only 640 bytes, easily cacheable).

For our case (2560 outputs, 2560 input features), the current approach seems reasonable. The bottleneck isn't the parallelism pattern -- it's the per-element instruction count in the inner loop.

**Optimization 3: Bit manipulation for branchless sign.** Instead of:
```
float nz = float(code & 1u);
float sign = 1.0 - 2.0 * float(code >> 1u);
acc += s_act[idx] * nz * sign;
```

We can use `uintBitsToFloat` to directly construct the sign-flipped value:
```
// If code = 01 (+1): XOR activation with 0 (no sign flip)
// If code = 11 (-1): XOR activation with 0x80000000 (sign flip)  
// If code = 00 (zero): AND with 0 (zero out)
uint sign_mask = (code & 2u) << 30;  // 0 or 0x80000000
uint nz_mask = (code & 1u) != 0u ? 0xFFFFFFFFu : 0u;
uint act_bits = floatBitsToUint(s_act[idx]);
acc += uintBitsToFloat((act_bits ^ sign_mask) & nz_mask);
```

This is exactly the CPU's XOR+AND trick, but in shader language! It eliminates the int-to-float conversions and multiplies, reducing it to 2 shifts, 2 ANDs, 1 XOR, and 1 float add per weight. That's 6 ops vs the current 5, but the ops are cheaper (integer bitwise vs float multiply). On Vega, bitwise ops execute on SALU (scalar ALU) or VALU and take 1 cycle, while float multiply takes 4 cycles.

Wait -- uintBitsToFloat and floatBitsToUint are free on GPU (reinterpret cast, no instruction emitted). So the actual instruction count is: 2 shifts, 2 ANDs, 1 XOR, 1 comparison (for nz_mask), 1 float add = 7 instructions. That's worse than the current path if the float multiply compiles to a single FMA.

Actually, the better approach: use the bit trick but avoid the comparison. Since code & 1 is either 0 or 1, we can generate the mask by negation:
```
uint nz_mask = -(code & 1u);  // 0x00000000 or 0xFFFFFFFF
```
That's a single NEG instruction. So: 1 shift, 2 ANDs, 1 NEG, 1 XOR, 1 float add = 6 instructions, all cheap.

**Optimization 4: Subgroup operations.** Vega supports subgroups (wavefronts) of size 64. We can use `subgroupAdd()` for reductions if we switch to the cooperative dot product model. But as discussed, the current independent-output model may be better.

**Optimization 5: Use float16 (FP16) with Rapid Packed Math.** Vega's Rapid Packed Math processes two FP16 ops in one FP32 slot, giving 3.23 TFLOPS. Our activations could be stored as float16 in LDS (halving LDS usage) and the accumulation done in FP16 pairs. Since BitNet already quantizes activations to INT8, FP16 precision is more than sufficient. This would double the effective throughput.

However: GLSL doesn't natively support FP16 in Vulkan without the `VK_KHR_shader_float16_int8` extension. Need to check if RADV supports it for Vega.

## Questions Arising
- Does RADV support VK_KHR_shader_float16_int8 on Vega 7?
- What's the actual L2 cache size on Vega 7 APU (1 MB or less)?
- Is there a way to do dynamic shared memory sizing in Vulkan (specialization constants)?
- How does RADV compile the branchless decode -- is it already optimal?
- What's the actual wavefront occupancy we're achieving?
- Can we batch all 210 dispatches in one command buffer submission?

## First Instincts
- The XOR+AND bit trick is the highest-ROI single change -- same logic as CPU kernel
- vec4 processing is the second-highest -- 4x reduction in loop iterations
- Reducing shared memory from 6912 to 2560 elements via specialization constants improves occupancy
- FP16 Rapid Packed Math is the "moonshot" -- 2x throughput if supported
- Cooperative dot product (reduction model) is a bigger rewrite with uncertain payoff
- Profile before optimizing: need to determine if we're memory-bound or compute-bound
