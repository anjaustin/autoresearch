# Raw Thoughts: Backward Pass Optimization — LM Head BF16 Bottleneck

## Stream of Consciousness

We had the ternary backward kernel working beautifully — 4.3x speedup on the 210 AutoBitLinear layers, bringing those from 82.9s to ~760ms total. But the full backward was still 19.5s. Where was the rest going?

Profiled with `torch.autograd.profiler.profile()` and the breakdown was shocking:

```
aten::mm:                        18.011s (92.6%) — 2 calls, ~9s each
TernaryMatmulFunctionBackward:      760ms (3.9%) — 210 calls
EmbeddingBackward:                  546ms (2.8%) — 1 call
aten::add:                          190ms (1.0%)
aten::fill_:                        164ms (0.8%)
```

92.6% of the backward is TWO matmul calls. These are from the LM head — `nn.Linear(2560, 128256)`. It's the only non-AutoBitLinear matrix operation in the model. It wasn't patched by our ternary backward because it's NOT ternary — it's a standard dense BF16 linear layer.

The LM head weight is [128256, 2560] — vocabulary size × hidden dimension. It's weight-tied with the embedding layer (`tie_word_embeddings: true` in config). The lm_head.weight IS embed_tokens.weight (same Python object, same data_ptr).

**Critical architectural fact**: Gradients MUST flow through the LM head to reach the TinyLoRA adapters. The forward path is: embeddings → 30 decoder layers (with AutoBitLinear + TinyLoRA) → final RMSNorm → LM head → logits → loss. Backward reverses this: loss → LM head backward → decoder layers → adapters. There's no bypass.

**The weight is frozen** (requires_grad=False). So autograd only needs to compute `grad_input = grad_output @ W`, not `grad_weight`. That's [1, seq_len, 128256] @ [128256, 2560] → [1, seq_len, 2560]. For autoregressive M=1, this is a vector-matrix multiply.

But even that takes ~3.3s per call. Why?

**Root cause: MKL BF16 GEMM on Ryzen (no AMX/no VNNI) is catastrophically slow.**

Benchmarked the same matmul in different dtypes:
- FP32: **84.2ms** 
- BF16: **2705ms** (32x slower!)
- BF16→FP32 conversion then FP32 matmul: **508ms** (conversion overhead)

The Ryzen 5 PRO 5675U has AVX2 but not AMX or VNNI instructions. MKL's BF16 GEMM on this hardware likely falls back to scalar BF16↔FP32 conversion per element inside the inner loop, destroying any memory locality and vectorization benefit.

**The fix is obvious**: Cast the LM head weight to FP32 once at initialization, then use a custom autograd function that does all computation in FP32 with BF16↔FP32 casts at the boundary.

Prototype `FP32LinearBackward`:
- Forward: cast input BF16→FP32, matmul in FP32, cast output FP32→BF16
- Backward: cast grad_output BF16→FP32, matmul in FP32, cast grad_input FP32→BF16
- Result: **190ms** forward+backward (vs ~9000ms original per direction)
- Speedup on this layer: **17.4x**

Memory cost: LM head in FP32 = 1.31 GB (vs 0.66 GB in BF16). Extra 0.66 GB. We have 64 GB RAM — completely irrelevant.

Projected full backward: 18,011ms → 190ms for LM head, rest unchanged = **1,679ms total** (11.6x speedup).
Projected full training iteration: forward 1,200ms + backward 1,679ms = **2,879ms** (vs 20,700ms current, 7.2x overall).
