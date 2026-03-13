# Reflections: BitNet b1.58 + TinyLoRA -- CPU-First Local Validation

## Core Insight

**The local test is not about training. It's about proving a single question: can a continuous-valued LoRA adapter attach to and produce gradients through a natively ternary model?**

Everything else -- GRPO, autoresearch loop, overnight runs, speed optimization -- is downstream of this binary answer. If yes, we have a novel research direction and a clear path to Thor deployment. If no, we learn something interesting about the limits of LoRA on extreme quantization.

## Resolved Tensions

### Tension 1: Two-runtime complexity (Node 1) vs. simple proof-of-concept (Node 5)
**Resolution:** Pure PyTorch for the local test. Period. bitnet.cpp is a speed optimization for rollout generation -- it's irrelevant to proving adapter compatibility. We accept slow inference (~1-3s/token) because we only need to generate a few tokens to validate the pipeline. Speed matters on Thor, not here.

### Tension 2: SFT (simple, proves mechanics) vs. GRPO (the actual method)
**Resolution:** Use SFT for the local proof-of-concept. Specifically: a single supervised gradient step on a single GSM8K example. This proves:
- Model loads ✓
- Adapter attaches ✓
- Forward pass works with adapter ✓
- Loss computes ✓
- Backward pass produces adapter gradients ✓
- Optimizer step changes adapter weights ✓

GRPO adds reward computation and rollout sampling on top of this. If the base mechanics work, GRPO will work. Test the foundation first.

### Tension 3: PEFT auto-detection vs. BitLinear custom layers (Node 2 vs Node 3)
**Resolution:** Test PEFT first. If it fails to auto-detect BitLinear modules, fall back to manual adapter injection:
```python
# Manual injection approach if PEFT fails:
for name, module in model.named_modules():
    if isinstance(module, BitLinear):
        # Wrap with custom LoRA bypass
```
This is straightforward because LoRA is architecturally simple -- it's just two small matrices added in parallel to an existing linear layer. The custom code path is at most 30 lines.

### Tension 4: BF16 training weights vs. ternary inference weights (Node 9)
**Resolution:** Use `microsoft/bitnet-b1.58-2B-4T-bf16` exclusively for the local test. This is the training-intended variant. The ternary packing is an inference optimization we don't need yet. The BF16 weights still get quantized to ternary internally during the forward pass (that's how BitNet works), so we're testing the real architecture.

## Challenged Assumptions

### Assumption: "We need PEFT library for LoRA"
**Challenge:** PEFT is convenient but not necessary. LoRA is simple enough to implement manually in ~50 lines. If PEFT's auto-detection breaks on BitLinear, a manual implementation avoids the dependency entirely. TinyLoRA's sub-rank-1 reparameterization needs custom code anyway (PEFT doesn't support rank < 1). So we may end up writing custom adapter code regardless.

**New plan:** Skip PEFT. Write a minimal TinyLoRA adapter from scratch. It's cleaner, avoids dependency issues, and directly implements the paper's reparameterization.

### Assumption: "The backward pass through a frozen 2B model on CPU will be prohibitively slow"
**Challenge:** For a SINGLE backward pass on a short sequence (say 50 tokens), this is probably 5-15 seconds on the Ryzen 5. For validating gradient flow, that's fine. We're not training for accuracy; we're proving the pipeline works. One forward + one backward = under 30 seconds. Acceptable.

### Assumption: "We need GSM8K for the local test"
**Challenge:** We don't even need a real benchmark. For gradient flow validation, any text works. We can use a trivial prompt like "2 + 2 = " and compute cross-entropy loss against "4". The local test is about mechanics, not benchmark scores.

### Assumption: "This is purely a stepping stone to Thor"
**Challenge:** Actually, if bitnet.cpp works well on this Ryzen 5 for inference, and we solve the two-runtime problem, this 64GB CPU machine could potentially run the full autoresearch loop itself -- just slower than Thor. BitNet was designed for CPU inference. A Ryzen 5 with 64GB is a legitimate deployment target, not just a test bench.

## What I Now Understand

The local test decomposes into exactly 5 sequential steps, each taking 5-15 minutes:

1. **Install + Load:** Download BF16 weights, verify model loads in PyTorch
2. **Inspect:** Print model architecture, identify target layers for adapter injection
3. **Attach:** Inject TinyLoRA adapter (custom, not PEFT), verify forward pass still works
4. **Gradient:** Compute loss on a trivial example, backward pass, verify adapter.grad is non-None
5. **Update:** Optimizer step, verify adapter weights changed, verify model output changed

If step 3 or 4 fails, we've discovered something important about BitNet + LoRA compatibility. If all pass, we have a green light for the full system.

## The Key Leverage Point

**Writing TinyLoRA from scratch instead of depending on PEFT.** This:
- Eliminates the PEFT + BitLinear compatibility risk entirely
- Gives us direct control over the adapter architecture
- Implements the paper's sub-rank-1 reparameterization (which PEFT can't do)
- Produces reusable code that works identically on Thor
- Is only ~50-80 lines of PyTorch

The custom adapter is the highest-value code to write locally.
