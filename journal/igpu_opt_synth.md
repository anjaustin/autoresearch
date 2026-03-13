# Synthesis: iGPU Vulkan Shader Optimization for Vega 7

## Decision

Implement the optimized shader as a **single v3 rewrite** combining transposed weight layout, XOR+AND bit trick, vec4 processing, and LDS right-sizing. Then wire it into Python and batch all 210 dispatches into one command buffer. Do NOT pursue FP16 Rapid Packed Math or the cooperative dot product model — the transposed layout solves the coalescing problem without a threading model rewrite, and FP16 support on Vega via RADV is uncertain.

The REFLECT phase revealed that the GPU is **memory-bound, not compute-bound** (opposite of the CPU). The #1 bottleneck is uncoalesced weight reads: consecutive wavefront threads access different rows at stride=640 bytes, turning every read into 64 scattered cache lines. Fixing this via transposed layout is the dominant optimization. Everything else is secondary.

## Action Plan (Ordered)

### Step 1: Transposed Weight Packing (1 hour)

Modify the Vulkan harness (`vk_ternary.c`) to pack weights in **transposed layout**: instead of row-major (each output row contiguous), store weights so that the same `in_features` position across consecutive output rows is contiguous in memory.

**Current layout** (row-major):
```
out_row_0: [w(0,0), w(0,1), ..., w(0,2559)]   // packed into 160 uint32s
out_row_1: [w(1,0), w(1,1), ..., w(1,2559)]
...
```

**Transposed layout** (column-major in 2-bit packing):
```
position_0: [w(0,0), w(1,0), w(2,0), ..., w(15,0)]   // 16 output rows packed into 1 uint32
position_1: [w(0,1), w(1,1), w(2,1), ..., w(15,1)]
...
```

With this layout, consecutive threads (processing consecutive output rows) read from consecutive memory addresses at each loop iteration. A 64-thread wavefront reads 64 consecutive uint32s = 256 bytes = 4 cache lines. This is **16x fewer cache lines** than the current 64 scattered reads.

The packing groups 16 output rows into each uint32 at each in_features position. For 2560 outputs, that's 160 groups of 16. For 2560 in_features, that's 160 × 2560 = 409,600 uint32s = 1.6 MB (same total size, different layout).

**Validation:** After packing, verify that unpacking the transposed layout recovers the same ternary matrix as the original row-major layout. Element-by-element comparison, zero tolerance.

### Step 2: Update Shader for Transposed Layout + Bit Trick + vec4 (1.5 hours)

Rewrite the compute shader (`ternary_matmul.comp` → new v3 shader) with:

**a) Transposed weight indexing.** Each thread `t` (computing output row `t`) reads from position `group_base + (t % 16)` within each packed uint32. The thread's 2-bit code is at bit position `(t % 16) * 2` within each word.

**b) XOR+AND bit trick.** Replace float multiply with bitwise ops:
```glsl
uint code = (word >> bit_offset) & 3u;
uint nz_mask = -(code & 1u);           // 0x00000000 or 0xFFFFFFFF
uint sign_mask = (code & 2u) << 30;    // 0x00000000 or 0x80000000
uint act_bits = floatBitsToUint(s_act[k]);
acc += uintBitsToFloat((act_bits ^ sign_mask) & nz_mask);
```
This eliminates int-to-float conversion and float multiply from the hot path. All ops are single-cycle VALU.

**c) vec4 processing.** Process 4 in_features positions per loop iteration:
```glsl
// Load 4 activation values
vec4 a = vec4(s_act[k], s_act[k+1], s_act[k+2], s_act[k+3]);

// Extract 4 codes from 4 consecutive words (transposed layout)
// Build vec4 masks, apply XOR+AND trick per component
// Accumulate: acc4 += masked_a;
```
This reduces loop iterations from 2560 to 640 (processing 4 per iteration). The loop overhead (branch, increment) is amortized 4x.

**d) Combine accumulator.** Final `acc = acc4.x + acc4.y + acc4.z + acc4.w` after the loop (single `dot(acc4, vec4(1.0))` instruction).

### Step 3: LDS Right-Sizing via Specialization Constants (30 min)

Add a Vulkan specialization constant for `in_features`:
```glsl
layout(constant_id = 0) const uint SPEC_IN_FEATURES = 2560;
shared float s_act[SPEC_IN_FEATURES];
```

Create two pipeline variants at initialization:
- `pipeline_2560`: for q/k/v/o_proj layers (in_features = 2560) → 10 KB LDS → 6 workgroups/CU possible → 60% occupancy
- `pipeline_6912`: for gate/up_proj layers (in_features = 6912) → 27 KB LDS → 2 workgroups/CU → 20% occupancy

The k/v_proj layers (out_features = 640) also benefit: smaller dispatch, faster completion.

### Step 4: Benchmark Optimized Shader (30 min)

Benchmark each layer type individually and as a full 7-layer decoder block:
- 2560×2560 (q/o_proj): target 0.15 ms (from 0.61 ms)
- 640×2560 (k/v_proj): target 0.05 ms
- 6912×2560 (gate/up_proj): target 0.40 ms
- 2560×6912 (down_proj): target 0.40 ms

Report: per-layer time, total decoder block time, effective bandwidth utilization, % of peak.

**Hard gate:** If the transposed layout doesn't achieve at least 2x improvement on 2560×2560, something is wrong with the coalescing analysis. Debug before proceeding.

### Step 5: Batched Command Buffer — All 210 Dispatches (1 hour)

Record a single `VkCommandBuffer` with all 210 dispatches for the full 30-layer forward pass:
1. For each decoder layer (30 layers):
   - Bind appropriate pipeline (2560 or 6912 variant)
   - Bind weight buffer + push constants for each of 7 sublayers
   - `vkCmdDispatch()` for each sublayer
   - Insert `vkCmdPipelineBarrier()` between layers (output of one layer = input to next... actually, the activation buffer is managed by Python between layers, so barriers may not be needed within a single matmul dispatch)

Actually — we can't batch all 210 dispatches into one submission because the **Python-side operations** (LayerNorm, residual add, attention mask/softmax, SiLU gating) happen between matmuls. The GPU only accelerates the `AutoBitLinear` matmuls, not the full transformer block.

**Revised approach:** Batch the 7 matmuls within each decoder layer into one command buffer submission. This eliminates 6 of the 7 per-layer submit overheads. With 30 layers × 1 submit each = 30 submissions vs current 30 × 7 = 210 submissions.

Wait — the 7 matmuls within a layer also have Python ops between them (the attention mechanism uses q/k/v outputs before o_proj, and the FFN uses gate/up before down). So even within a layer, we can only batch:
- **q + k + v** (3 independent matmuls, all take the same input) → 1 submission
- **o_proj** (depends on attention output) → 1 submission  
- **gate + up** (2 independent matmuls, same input) → 1 submission
- **down** (depends on gate*up) → 1 submission

That's 4 submissions per layer × 30 = 120 total, saving ~0.43ms × 90 = ~39 ms over 210 individual submissions. Significant.

### Step 6: Python/PyTorch Integration (2 hours)

Extend `softchip/torch_ternary.py` to support a Vulkan backend:
1. At `patch_model()` time, pack all 210 layers' weights in transposed format and upload to a single Vulkan buffer (497 MB)
2. `TernaryMatmulFunction.forward()` dispatches to Vulkan when available, CPU otherwise
3. Activation data transferred via `torch.Tensor` → Vulkan buffer copy (small: 10-27 KB per dispatch)
4. Output read back from Vulkan buffer → `torch.Tensor`

The CPU↔GPU transfer overhead for activations is tiny (10 KB at ~20 GB/s = 0.5 us). The overhead is in the Vulkan submission, which we minimize via batching (Step 5).

**Alternative:** Use a ctypes/cffi wrapper around `vk_ternary.c` rather than building a torch C extension. Simpler, avoids torch compilation complexity.

## What We're NOT Doing (and Why)

| Opportunity | Why skip |
|---|---|
| FP16 Rapid Packed Math (Node 4) | Uncertain RADV support on Vega APU. High risk, complex shader changes. Investigate later if needed. |
| Cooperative dot product (Node 6) | Transposed layout (Node 7) gives coalesced access without rewriting the threading model. Cooperative model also reduces parallelism to ~40 workgroups. |
| Multi-output per thread | Increases register pressure, complicates transposed layout indexing. The current one-output-per-thread model is clean and sufficient. |
| Full transformer on GPU | LayerNorm, softmax, SiLU need separate shaders. Over-investment for a Vega 7 that won't be the production target. |
| Async CPU+GPU overlap | Shared DDR4 bus means bandwidth contention. CPU and GPU competing on the same 51 GB/s would hurt both. Sequential is cleaner. |

## Key Insight

The iGPU shader optimization is a **development accelerator for the Ryzen**, same as the CPU soft-chip was. The Vega 7 won't be the production target (Thor will), but getting rollout generation down to ~10-13 seconds for 200 tokens makes overnight experiment loops viable on this laptop. That's the strategic goal: **unblock GRPO training** even before Thor arrives.

The most important lesson from this LMM pass: GPU optimization is about **memory access patterns**, not instruction count. The XOR+AND bit trick that was the #1 CPU optimization is a minor win on GPU. Transposed weight layout (a data structure change, not a compute change) is the dominant optimization. Always identify the actual bottleneck before optimizing.

## Metrics for Success

| Metric | Target | Current |
|---|---|---|
| Per-layer kernel time (2560×2560) | 0.15-0.25 ms | 0.61 ms |
| Full model forward (M=1) | 45-65 ms | ~140 ms (projected) |
| vs CPU soft-chip M=1 | 14-20x faster | 4.6x faster (CPU vs PyTorch) |
| vs stock PyTorch M=1 | 65-93x faster | — |
| 200-token rollout | 10-13 seconds | ~3 min (CPU), ~14 min (PyTorch) |
| Weight buffer fits VRAM | <512 MB | 497 MB (unchanged) |

## Updated Next Steps (Post This LMM Pass)

1. **Implement Steps 1-2** (transposed packing + v3 shader) → benchmark against hard gate
2. **Implement Step 3** (specialization constants) → benchmark occupancy improvement
3. **Implement Step 4** (full benchmark suite) → update FINDINGS.md
4. **Implement Steps 5-6** (batched submission + Python integration)
5. **End-to-end validation**: full model forward pass through Vulkan, compare output logits to CPU soft-chip and stock PyTorch
6. **200-token rollout benchmark** → confirm 10-13 second target
7. **Begin GRPO training loop design** (separate LMM pass)
