# Nodes: GRPO Training Loop for TinyLoRA on BitNet b1.58

## Decision Points Extracted from RAW

### Node 1: RL Algorithm Variant
**Question:** Full GRPO (with clipping + KL + importance sampling) vs simplified REINFORCE with group advantages?

Options:
- A) Full GRPO with clipping and KL penalty
- B) REINFORCE with group-normalized advantages, no clipping, no KL
- C) REINFORCE with group advantages + KL penalty, no clipping

Tension: Full GRPO is more stable but harder to implement correctly. With only 210 parameters, instability risk is low. Clipping requires tracking π_old which adds complexity.

### Node 2: Rollout Strategy
**Question:** How to generate completions from the adapted model?

Options:
- A) Soft-chip autoregressive generation (~580ms/token, ~60s per 100-token completion)
- B) bitnet.cpp for generation (35 tok/s, ~3s per completion, but no adapter support)
- C) Stock PyTorch autoregressive (even slower, ~3s/token without soft-chip)

Tension: Soft-chip is 20x faster than stock PyTorch but 20x slower than bitnet.cpp. bitnet.cpp can't include adapters without C++ modification.

### Node 3: Group Size (G)
**Question:** How many completions per prompt?

Options:
- A) G=2 — fastest, but binary rewards often give zero signal (both right or both wrong)
- B) G=4 — good balance, ~4 min rollouts per prompt
- C) G=8 — best signal, ~8 min rollouts per prompt

Tension: Larger G gives better advantage estimates but costs proportionally more in rollout time.

### Node 4: Which Layers to Adapt
**Question:** How many of the 210 AutoBitLinear layers get TinyLoRA adapters?

Options:
- A) All 210 layers (210 trainable scalars)
- B) All 30 q_proj layers (30 scalars)
- C) One type per layer, 30 layers (30 scalars)
- D) Start with 4 (as in validation), expand later

Tension: More layers = more flexibility for RL to find signal, but more adapter overhead in forward pass. Though adapter compute is negligible (rank-1 matmul).

### Node 5: KL Regularization
**Question:** Include KL divergence penalty between π_θ and π_ref?

Options:
- A) Yes, compute KL via two forward passes (adapted vs base)
- B) Yes, approximate KL from adapter scale magnitudes
- C) No, skip KL in V1 — adapters are inherently bounded

Tension: KL prevents degenerate solutions but requires extra forward pass. 210 scalars near zero can't deviate far anyway.

### Node 6: Prompt Format
**Question:** How to prompt a base model for GSM8K?

Options:
- A) Zero-shot: "Question: ... Answer:"
- B) 2-shot: include 2 worked examples before the target question
- C) 4-shot: include 4 worked examples (standard GSM8K eval format)

Tension: More shots improve base model accuracy (good for signal) but increase sequence length (slower generation). Each few-shot example adds ~100 tokens.

### Node 7: Generation Parameters
**Question:** Temperature, max tokens, sampling strategy?

Options:
- A) temperature=0.7, max_tokens=256, top-p=0.9
- B) temperature=1.0, max_tokens=256, no top-p
- C) temperature=0.5, max_tokens=150, top-p=0.95

Tension: Higher temperature = more diversity = more mixed reward signals, but also more nonsense. Lower temperature = higher quality but less exploration.

### Node 8: Update Granularity
**Question:** Update after each prompt or batch multiple prompts?

Options:
- A) Update after each prompt (B=1): simplest, immediate gradient signal
- B) Accumulate over B=4 prompts before updating: smoother gradients
- C) Accumulate over B=8 prompts: best gradient estimate

Tension: With 210 parameters, even a single prompt's signal might be sufficient. Accumulating introduces delay. But single-prompt updates are noisy.

### Node 9: Learning Rate
**Question:** What learning rate for Adam on 210 scalar parameters with REINFORCE gradients?

Options:
- A) lr=0.01 (conservative)
- B) lr=0.1 (aggressive, as in validation test)
- C) lr=0.001 (very conservative)
- D) Schedule: start at 0.1, decay to 0.001

Tension: REINFORCE gradients are noisy. Too high = instability. Too low = no learning in feasible time. Validation showed gradients ~1e-5, so lr needs to compensate.

### Node 10: Pre-flight Validation
**Question:** What to verify before starting the training loop?

Must-haves:
- Base model accuracy on GSM8K (need non-zero for signal)
- Answer extraction works correctly
- Generation produces parseable output
- Full forward+backward pipeline timing confirmation
- Memory usage under the generation + training workload
