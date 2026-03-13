# Synthesis: BitNet b1.58 + TinyLoRA -- CPU-First Local Validation

## Objective

Prove that a continuous-valued TinyLoRA adapter can attach to, forward through, and receive gradients from a natively ternary BitNet b1.58 model on CPU. This is a pass/fail validation that gates the full TinyMoE-Search project on Thor.

---

## Architecture (Local Test)

```
┌──────────────────────────────────────────────────────────┐
│                 LOCAL VALIDATION PIPELINE                  │
├──────────────────────────────────────────────────────────┤
│                                                           │
│  Step 1: LOAD                                             │
│  └─ Download microsoft/bitnet-b1.58-2B-4T-bf16           │
│  └─ Load into PyTorch on CPU (BF16, ~4GB)                │
│  └─ Verify: model generates text                          │
│                                                           │
│  Step 2: INSPECT                                          │
│  └─ Print model.named_modules()                           │
│  └─ Identify BitLinear layers and their dimensions        │
│  └─ Choose target layers for adapter injection            │
│                                                           │
│  Step 3: ATTACH                                           │
│  └─ Implement TinyLoRA adapter (~50 lines)               │
│  └─ Inject into target layers                             │
│  └─ Freeze all base model params                          │
│  └─ Verify: model still generates coherent text           │
│                                                           │
│  Step 4: GRADIENT                                         │
│  └─ Compute cross-entropy loss on trivial example         │
│  └─ loss.backward()                                       │
│  └─ Verify: adapter.grad is not None                      │
│  └─ Verify: base model grads are None (frozen)            │
│                                                           │
│  Step 5: UPDATE                                           │
│  └─ optimizer.step() on adapter params only               │
│  └─ Verify: adapter weights changed                       │
│  └─ Verify: model output changed (even slightly)          │
│                                                           │
│  RESULT: PASS or FAIL at each step                        │
└──────────────────────────────────────────────────────────┘
```

---

## Key Decisions

### 1. Skip PEFT, write TinyLoRA from scratch
Because PEFT may not recognize BitLinear layers, and TinyLoRA's sub-rank-1 reparameterization requires custom code anyway. ~50-80 lines of clean PyTorch. This code carries forward to Thor unchanged.

### 2. Use BF16 master weights, not packed ternary
Because gradient computation requires continuous weights. `microsoft/bitnet-b1.58-2B-4T-bf16` is the training variant. ~4GB download.

### 3. Pure PyTorch, no bitnet.cpp
Because the local test is about compatibility, not speed. bitnet.cpp is for fast inference deployment. We'll add it on Thor for GRPO rollout generation.

### 4. SFT-style gradient test, not GRPO
Because a single supervised backward pass proves gradient flow with zero infrastructure. GRPO adds reward design, rollout batching, and advantage estimation -- all orthogonal to the core compatibility question.

### 5. Trivial test data, not GSM8K
Because for mechanics validation, any input/output pair works. Use "2 + 2 = " -> "4" or similar. Zero data pipeline overhead.

---

## Implementation Spec

### Prerequisites (5 minutes)

```bash
pip install peft trl   # Install even if not used yet -- we'll want them later
```

### File: `test_bitnet_tinylora.py`

This single script runs all 5 validation steps sequentially, printing PASS/FAIL for each.

```python
"""
BitNet b1.58 + TinyLoRA Compatibility Validation
Tests whether a continuous LoRA adapter can produce gradients
through a natively ternary model.
"""

# === STEP 1: LOAD ===
# Download and load microsoft/bitnet-b1.58-2B-4T-bf16
# Verify generation works

# === STEP 2: INSPECT ===
# Print module names and types
# Identify BitLinear (or equivalent) layers
# Record layer dimensions for adapter sizing

# === STEP 3: ATTACH ===
# TinyLoRA implementation:
#
# class TinyLoRA(nn.Module):
#     """Sub-rank-1 LoRA adapter following the TinyLoRA paper.
#     
#     Instead of W + BA (rank-r), uses W + s * (u @ v^T)
#     where u, v are fixed random vectors and s is a learned scalar.
#     This gives 1 trainable parameter per adapted layer.
#     """
#     def __init__(self, base_layer, seed=42):
#         super().__init__()
#         self.base_layer = base_layer
#         out_features, in_features = base_layer.weight.shape
#         gen = torch.Generator().manual_seed(seed)
#         self.register_buffer('u', torch.randn(out_features, 1, generator=gen))
#         self.register_buffer('v', torch.randn(1, in_features, generator=gen))
#         self.scale = nn.Parameter(torch.zeros(1))  # 1 trainable param
#
#     def forward(self, x):
#         base_out = self.base_layer(x)
#         adapter_out = (x @ self.v.T) @ self.u.T * self.scale
#         return base_out + adapter_out
#
# Inject into target layers, freeze base model

# === STEP 4: GRADIENT ===
# Forward pass on trivial input
# Compute cross-entropy loss
# backward()
# Check adapter.scale.grad is not None
# Check base params have no grad

# === STEP 5: UPDATE ===
# optimizer.step()
# Verify adapter.scale value changed
# Forward pass again, verify output logits differ
```

### Expected Outputs

For each step, print:
```
[STEP 1] Model load .............. PASS (4.2GB, 32 layers)
[STEP 2] Architecture inspect .... PASS (found 24 BitLinear layers)
[STEP 3] TinyLoRA attach ......... PASS (24 params, generation coherent)
[STEP 4] Gradient flow ........... PASS (adapter grads: non-zero, base grads: None)
[STEP 5] Weight update ........... PASS (scale: 0.000 -> 0.003, output changed)

ALL STEPS PASSED: BitNet + TinyLoRA is viable.
```

Or for failure:
```
[STEP 3] TinyLoRA attach ......... FAIL (RuntimeError: BitLinear does not support ...)

BLOCKED AT STEP 3: Investigate BitLinear forward pass compatibility.
```

---

## Time Budget

| Step | Action | Estimated Time |
|------|--------|---------------|
| 0 | Install deps | 2 min |
| 1 | Download model + load | 10-15 min (download ~4GB) |
| 2 | Inspect architecture | 1 min |
| 3 | Write + attach adapter | 10 min |
| 4 | Gradient test | 2 min (one forward+backward) |
| 5 | Update test | 2 min |
| **Total** | | **~30 minutes** |

---

## Success Criteria

- [ ] BitNet BF16 model loads on this machine (CPU, 64GB RAM)
- [ ] Model architecture is inspectable, BitLinear layers identified
- [ ] Custom TinyLoRA adapter attaches without breaking forward pass
- [ ] Gradient flows to adapter parameters (grad is not None and non-zero)
- [ ] Base model parameters remain frozen (no grad)
- [ ] Optimizer step changes adapter weights
- [ ] Model output changes after adapter update

**All 7 checks must pass to proceed to Thor deployment.**

---

## Failure Modes and Fallbacks

| Failure | Likely Cause | Fallback |
|---------|-------------|----------|
| Model won't load | Transformers version mismatch | Install the specific commit from model card |
| CUDA error on load | Model defaults to GPU | Set `device_map="cpu"` explicitly |
| BitLinear has no `.weight` attribute | Custom implementation | Inspect source, find weight storage |
| Forward pass crashes with adapter | Dimension mismatch or dtype issue | Check layer I/O shapes, match dtypes |
| Gradients are None | Computation graph detached | Ensure `requires_grad=True` on adapter params, no `.detach()` in forward |
| Gradients are zero | LoRA bypass doesn't affect loss | Increase adapter scale init, check loss computation |

---

## What This Unlocks

If all steps pass:
1. **Immediate:** We have a working TinyLoRA adapter for BitNet b1.58 -- likely the first ever
2. **Next:** Port to Thor, add GRPO, add autoresearch loop
3. **Novel contribution:** TinyLoRA on ternary weights is unexplored territory
4. **Paper-worthy:** If sub-100 parameter adapters improve GSM8K on a ternary 2B model via RL

If any step fails:
1. **Valuable:** We've identified a specific incompatibility between LoRA and ternary quantization
2. **Debuggable:** Each step is isolated, so we know exactly what broke
3. **Publishable:** "LoRA doesn't work on natively ternary models because X" is also an interesting finding

---

## One-Sentence Summary

A 30-minute, 5-step validation script that answers: can 1 parameter steer a 2-billion-parameter ternary brain?
