# Nodes of Interest: BitNet b1.58 + TinyLoRA -- CPU-First Local Validation

## Node 1: Two Runtimes Problem
bitnet.cpp gives ~35 tok/s on CPU (ternary kernels). PyTorch through transformers gives standard 2B model speed (~1-3s per forward pass, maybe 0.5-1 tok/s generation). GRPO needs fast rollouts (bitnet.cpp territory) AND gradient computation (PyTorch territory). These are different executables with different model representations.
**Why it matters:** For local proof-of-concept, this complexity may be unnecessary. For production on Thor, it's probably essential.

## Node 2: LoRA + BitLinear Compatibility
BitNet uses `BitLinear` layers where continuous master weights are quantized to {-1, 0, +1} via absmean quantization during the forward pass. Standard PEFT LoRA adds a bypass: output = base(x) + BA(x). The bypass operates on the full-precision input x, independently of what base() does internally. This should work, but BitNet's custom code + `trust_remote_code=True` may not be recognized by PEFT's target module auto-detection.
**Why it matters:** If PEFT can't target BitLinear layers automatically, we need a manual adapter injection -- not hard, but a potential blocker.

## Node 3: Transformers Version Compatibility
The BitNet model card specifies a particular transformers commit (`096f25ae...`). We have 4.52.0.dev0. The model uses `custom_code` with its own modeling files on HuggingFace. Dev versions of transformers may or may not be compatible.
**Why it matters:** A version mismatch could block loading the model entirely. This is the first thing to test.

## Node 4: CPU Forward Pass Speed Through PyTorch
A 2B parameter model in BF16 on a Ryzen 5 (6 cores) through PyTorch: expect ~1-3 seconds per forward pass for a single sequence. For generation (autoregressive), this means ~1-3 seconds per token. Generating 200 tokens of chain-of-thought: ~200-600 seconds per completion.
**Why it matters:** This makes full GRPO training impractical on CPU through PyTorch. But a SINGLE forward+backward pass to prove gradient flow is fine (~10 seconds).

## Node 5: SFT vs GRPO for Local Validation
TinyLoRA's key finding: RL (GRPO) enables extreme parameter reduction; SFT needs 100-1000x more params. But for local validation, we only need to prove that (a) adapters attach, (b) gradients flow, (c) adapter weights update. SFT on a tiny batch proves all three without the complexity of reward computation and rollout generation.
**Tension with the full project:** We're deferring the GRPO validation, which is the actual mechanism. But SFT is sufficient to prove the BitNet + LoRA compatibility hypothesis.

## Node 6: The Novel Research Question
No one has applied TinyLoRA to a natively ternary model. The base weights are {-1, 0, +1}. The LoRA adapter is continuous-valued (BF16). The interaction is: output = quantize_ternary(W_master) @ x + B @ A @ x. The adapter is adding continuous corrections to ternary computations. Does this work better or worse than adding corrections to FP16 weights? This is genuinely unknown.
**Why it matters:** This is the intellectual core of the project. Local validation proves the mechanics work. Thor proves it works at scale.

## Node 7: Memory Profile is Trivial
BF16 master weights: ~4GB. TinyLoRA adapters: negligible. Optimizer states for 13-1000 params: negligible. One batch of activations for a short sequence: maybe 1-2GB. Total: ~6GB out of 64GB available. Memory is not a constraint at all. We could even load multiple copies of the model for parallel experiments.
**Why it matters:** This means we can focus entirely on compute speed and compatibility, not memory.

## Node 8: The Test Should Be Tiny and Decisive
The local test isn't training. It's a series of binary pass/fail checks:
1. Does the model load? (pass/fail)
2. Does PEFT attach? (pass/fail)
3. Does generation work with adapters? (pass/fail)
4. Do gradients flow to adapter params? (pass/fail)
5. Do adapter weights change after an update? (pass/fail)

Each check takes minutes, not hours. Total wall time: ~30-60 minutes including model download.

## Node 9: BF16 vs Packed Ternary Weights
Microsoft provides two weight variants: packed ternary (`bitnet-b1.58-2B-4T`, 0.4GB) for inference, and BF16 master weights (`bitnet-b1.58-2B-4T-bf16`, ~4GB) for training. We must use the BF16 variant for PyTorch training since we need continuous weights for gradient computation. The packed ternary variant is for bitnet.cpp inference only.
**Why it matters:** Must download the right model variant. The BF16 weights are 10x larger but still tiny by modern standards.

## Node 10: Path from Local Validation to Thor Deployment
Local proves: BitNet + LoRA works mechanically.
Thor adds: GRPO training at speed, autoresearch loop, overnight autonomous runs, bitnet.cpp for fast rollouts.
The code written locally should be reusable on Thor with minimal changes (add GPU support, swap in GRPO, add the loop).
**Why it matters:** Local work isn't throwaway -- it's the foundation.

## Tensions Summary
- **Node 1 vs Node 5:** Two-runtime complexity vs. simple SFT proof-of-concept -- defer the complexity
- **Node 2 vs Node 3:** PEFT compatibility depends on transformers version -- test sequentially
- **Node 4 vs Node 8:** CPU speed is bad for training but fine for pass/fail validation
- **Node 5 vs Node 6:** SFT proves mechanics but not the RL-specific insight -- accept this for V0
- **Node 9 vs Node 1:** BF16 for training, ternary for inference -- two representations of same model

## Dependencies
- Node 3 (transformers compat) blocks everything
- Node 2 (PEFT + BitLinear) blocks Nodes 5 and 6
- Node 9 (correct weights) blocks Node 2
- Node 8 (test design) is independent -- can be written now
