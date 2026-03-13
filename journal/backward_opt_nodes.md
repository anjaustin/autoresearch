# Nodes: Backward Pass Optimization — LM Head Options

## The Problem

The LM head (`nn.Linear(2560, 128256)`, BF16, weight-tied with embeddings) accounts for 92.6% of backward pass time (18.0s of 19.5s). MKL's BF16 GEMM on Ryzen without AMX/VNNI is 32x slower than FP32.

## Options Considered

### Option A: Cast LM head weight to FP32, custom autograd function
- **Approach**: Pre-convert weight to FP32 at init. Custom `FP32LMHeadFunction` does BF16↔FP32 casts at boundary, all math in FP32.
- **Measured**: 190ms forward+backward (vs 9000ms+ original)
- **Memory**: +0.66 GB (trivial with 64 GB RAM)
- **Complexity**: Low — one new autograd Function, one patch call
- **Accuracy**: Potentially BETTER — FP32 matmul has higher precision than BF16

### Option B: Skip LM head backward entirely (gradient-free approach)
- **Approach**: Detach hidden states before LM head, use REINFORCE/policy gradient instead of backprop through logits.
- **Problem**: Loses the dense gradient signal from cross-entropy. REINFORCE has much higher variance. Would require fundamentally different training loop.
- **Complexity**: High — completely different optimization strategy

### Option C: Gradient checkpointing on LM head
- **Approach**: Don't store activations for LM head, recompute in backward.
- **Problem**: Doesn't help — the bottleneck is the matmul computation itself, not memory. Recomputing would make it WORSE (adds forward recompute on top of backward).

### Option D: Approximate gradient via random projection
- **Approach**: Project the 128256-dim gradient to lower dimension before multiplying by W.
- **Problem**: Loses gradient information. Unclear convergence properties. Research territory.

### Option E: Chunked BF16→FP32 matmul (without pre-converting weight)
- **Approach**: Process the matmul in chunks, converting each weight chunk to FP32 on the fly.
- **Measured**: ~508ms for full convert+matmul. Worse than Option A (190ms) due to repeated conversion overhead.

## Clear Winner: Option A

Option A is faster, simpler, and more numerically accurate. The memory cost is negligible. No other option comes close.
