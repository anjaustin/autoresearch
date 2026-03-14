# Reflect: GRPO Training Loop for TinyLoRA on BitNet b1.58

## Decisions

### Node 1: RL Algorithm → Option C (REINFORCE + group advantages + KL penalty)

Rationale: Start with REINFORCE (simplest correct algorithm) with group-normalized advantages (the GRPO insight). Add KL penalty because it's cheap (zero-out scales, one extra forward pass) and prevents degenerate adapter growth. Skip clipping — with 210 scalars and moderate learning rate, catastrophic updates are unlikely. If training is unstable, add clipping as a patch.

The loss function:
```
L = -Σ_j Σ_t [A_j * log π_θ(y_j,t | y_j,<t, q)] + β * KL_approx
```

Where A_j is the group-normalized advantage for completion j.

### Node 2: Rollout Strategy → Option A (soft-chip autoregressive)

Rationale: We built the soft-chip specifically for this. 580ms/token is slow but workable. bitnet.cpp would require C++ adapter integration — engineering effort better spent on getting GRPO working. Stock PyTorch is 3x slower (no soft-chip). We use what we have.

Key implementation detail: for generation, we need token-by-token autoregressive loop using the patched model. The soft-chip handles the AutoBitLinear forward, the TinyLoRA adapters add their corrections, and we sample from the output distribution.

### Node 3: Group Size → Option B (G=4)

Rationale: G=2 with binary rewards has a high chance of zero signal (both right or both wrong). With BitNet's ~29% GSM8K accuracy, P(both wrong) ≈ 0.50, P(both right) ≈ 0.08, P(mixed) ≈ 0.42. So 58% of prompts give no signal.

With G=4: P(all wrong) ≈ 0.25, P(all right) ≈ 0.007, P(at least mixed) ≈ 0.74. Much better signal rate.

Cost: 4 × 60s = 240s rollouts per prompt. Acceptable for overnight runs.

### Node 4: Layers to Adapt → Option A (all 210 layers)

Rationale: Adapter forward pass is negligible (rank-1 matmul: `scale * (x @ v^T) @ u^T`). Zero reason to restrict which layers get adapters. Let RL discover which layers matter — this is the whole point of autoresearch. Post-training analysis of which scales moved most will be scientifically interesting.

210 trainable parameters is still absurdly few. The parameter ratio is 1:11.5M.

### Node 5: KL Regularization → Option A (two forward passes)

Rationale: KL is important even with tiny adapters. The approach: after generating completions, forward the full sequences through the model twice — once with adapters active (get π_θ log-probs), once with all scales zeroed (get π_ref log-probs). The KL is:

```
KL ≈ Σ_t [log π_θ(y_t | ...) - log π_ref(y_t | ...)]
```

Cost: one extra forward pass (~1.3s × 4 sequences = 5.2s). This is <3% of the rollout time. Worth it.

β = 0.01 initially. Tune if KL grows too fast or too slow.

### Node 6: Prompt Format → Option B (2-shot)

Rationale: Base model needs format guidance. 4-shot adds ~400 tokens to every prompt, making rollouts longer and slower. 2-shot (200 extra tokens) is a good compromise — establishes format with minimal overhead.

Use two diverse GSM8K examples: one simple (addition), one with multiplication. Format:
```
Solve the following math problems step by step.

Q: [example 1 question]
A: [example 1 solution]. The answer is [number].

Q: [example 2 question]  
A: [example 2 solution]. The answer is [number].

Q: [target question]
A:
```

### Node 7: Generation Parameters → Option A (temperature=0.7, max_tokens=256, top-p=0.9)

Rationale: Standard settings for diverse but coherent generation. Temperature 0.7 gives enough randomness for mixed outcomes within a group while keeping quality. Top-p 0.9 cuts off the long tail. Max 256 tokens is generous for GSM8K (most solutions are <150 tokens).

Stop conditions: stop at newline after "The answer is" or at max tokens.

### Node 8: Update Granularity → Option A (B=1, update per prompt)

Rationale: With 210 parameters and noisy REINFORCE gradients, faster iteration is better. Each prompt takes ~4 minutes — batching would mean waiting 16+ minutes between updates. Single-prompt updates give immediate feedback.

The noise in REINFORCE gradients is addressed by:
1. Group normalization (G=4 completions)
2. Adam optimizer (momentum smooths noise)
3. 210 parameters can only change so much per step

### Node 9: Learning Rate → Option D (schedule: 0.01 → 0.001)

Rationale: Validation showed gradients ~1e-5. With REINFORCE, gradients are scaled by advantages (±1 normalized), so effective gradients will be O(log-prob × advantage) ≈ O(1-10). Adam normalizes by second moment, so the raw gradient scale matters less.

Start at lr=0.01. If training loss oscillates wildly, reduce. If scales aren't moving, increase. Cosine decay to 0.001 over training. This is conservative but safe for a first run.

### Node 10: Pre-flight Validation → YES, comprehensive

Before training:
1. Verify base model generates parseable GSM8K answers (at least some)
2. Test answer extraction on 10 examples manually
3. Time a full rollout (4 completions for 1 prompt)
4. Time a full training step (forward + backward + optimizer)
5. Measure peak memory
6. Verify adapter gradients are non-zero after one step

This is the pre-flight checklist. No training without it.

## Cross-Cutting Concerns

### Checkpointing
Save only the 210 adapter scales + step number + optimizer state. Total: ~2KB per checkpoint. Save every 10 steps and after each evaluation.

### Evaluation
Every 25 steps, evaluate on 50 GSM8K test problems (greedy, no sampling). Report accuracy. This takes ~50 × 60s = 50 min (one prompt at a time). So evaluate sparingly.

Alternative: every 25 steps, evaluate on just 20 test problems (20 min). More frequent signal, less precise.

### Logging
Log to console and to a JSON lines file:
- step, wall_time, prompt_idx
- rewards (list of G values)
- mean_advantage, loss
- KL_divergence
- all 210 adapter scales (for tracking which layers move)
- eval accuracy (when evaluated)

### Stopping Criteria
- Time budget (8 hours overnight)
- Or 500 steps (whichever comes first)
- Early stop if eval accuracy stops improving for 100 steps

## Risk Assessment

| Risk | Probability | Mitigation |
|------|------------|------------|
| No gradient signal (model always right or wrong) | Low (29% base accuracy → 74% chance of mixed group) | Temperature tuning, curriculum |
| Gradients too noisy for 210 params | Medium | Adam momentum, group normalization |
| Generation produces unparseable output | Medium | Robust answer extraction, format enforcement |
| Memory issues during long sequences | Low (64GB is abundant) | Monitor, reduce max_tokens if needed |
| Training takes too long for meaningful results | Medium | Accept partial results, checkpoint frequently |
| Reward hacking | Very Low (210 params can't memorize) | KL penalty, monitor qualitative outputs |
