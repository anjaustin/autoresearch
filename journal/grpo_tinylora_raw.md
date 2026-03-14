# Raw Thoughts: GRPO Training Loop for TinyLoRA on BitNet b1.58

## Stream of Consciousness

We've built the engine. Six passes of optimization: soft-chip kernels, Vulkan exploration, backward pass fixes, the FP32 LM head breakthrough. A 2.4-second training iteration where stock PyTorch took 92 seconds. Now we need to actually train something.

GRPO -- Group Relative Policy Optimization. The key insight from TinyLoRA (Morris et al.): RL is what makes extreme parameter reduction work. SFT needs 1000x more parameters to reach the same performance. With 4-210 scalar parameters, we need every bit of signal efficiency that RL provides.

### What is GRPO exactly?

GRPO is a simplified RLHF variant from DeepSeek-R1. For each prompt:
1. Generate G completions from the current policy (the model with TinyLoRA adapters)
2. Score each completion with a reward function (binary for GSM8K: correct/incorrect)
3. Compute advantages within the group (normalize rewards within the batch)
4. Update the policy to increase probability of high-advantage completions

The key simplification vs PPO: no critic network. Advantages are computed from group statistics, not from a learned value function. This is perfect for us -- we can barely afford one forward/backward pass, let alone maintaining a separate critic.

### The Two-Runtime Problem (revisited)

Pass 2 RAW identified this: bitnet.cpp does inference at 35 tok/s, our soft-chip does ~1.7 tok/s through PyTorch. For GRPO we need:
- **Rollout generation**: generate G completions per prompt. This is autoregressive, token-by-token. At 580ms/token through soft-chip, a 200-token completion takes 116 seconds. G=4 completions = 464 seconds per prompt. That's 7.7 minutes for ONE prompt's rollouts.
- **Policy gradient computation**: forward+backward on the selected sequences. This is our optimized 2.4s path (for short sequences). For full rollout sequences (~200 tokens), it'll be proportionally longer but still manageable.

The rollout bottleneck is severe. Options:
1. **Accept it**: ~8 minutes per prompt. Run overnight. If we do 50 prompts that's ~7 hours. Viable for overnight autoresearch.
2. **Use bitnet.cpp for rollouts**: Fast (35 tok/s → 6s per 200-token completion) but requires bridging C++ inference with Python training. The adapter weights need to be "baked in" somehow.
3. **Shorter completions**: GSM8K solutions are often 50-100 tokens, not 200. That cuts rollout time significantly.
4. **Smaller group size**: G=2 instead of G=4. Noisier gradients but faster.
5. **Hybrid**: Use bitnet.cpp for the frozen model's forward pass, then correct with adapter in Python.

Wait -- option 5 doesn't work. The adapter changes the model's output distribution, so rollouts MUST go through the adapted model. You can't generate from the base model and then pretend the adapted model generated those sequences.

Actually, let me reconsider. In GRPO, we generate from the current policy π_θ. Our policy is base_model + TinyLoRA adapters. The adapters are tiny perturbations (scales initialized at 0, currently near 0 after few updates). Early in training, the adapted model is essentially the base model. So we COULD generate from bitnet.cpp (≈ base model) early on, and only switch to full adapted generation later. But this introduces bias and isn't principled.

Actually the right way to think about this: can we run bitnet.cpp with the TinyLoRA correction? The adapter adds `scale * (x @ v^T) @ u^T` at each adapted layer. That's a rank-1 correction. We could modify bitnet.cpp to include this... but that's a significant C++ engineering effort and takes us off the Python-first path.

### The Practical Decision

Let's just accept the slow rollouts for V1. Here's the math:

- Completions: ~100 tokens average for GSM8K
- Per-token: 580ms through soft-chip
- Per completion: 58 seconds
- Group size G=2: 116 seconds per prompt (two rollouts)
- Gradient update: ~2.4 seconds (amortized, depends on sequence length)
- Total per prompt: ~120 seconds = 2 minutes

For an overnight run (8 hours = 480 minutes):
- ~240 prompts processed
- ~480 rollouts generated
- ~240 gradient updates

That's actually not terrible for a proof-of-concept. GSM8K has 7,473 training examples. We'd get through ~3% in one night. But we're training 4-210 scalar parameters -- they might converge fast. TinyLoRA paper shows convergence in hundreds of steps, not thousands.

### Reward Function

GSM8K is clean: each problem has a numerical answer after "####". Binary reward:
- R = 1 if the model's answer matches the ground truth number
- R = 0 otherwise

Answer extraction: parse the last number in the completion. Need to handle edge cases (no number generated, wrong format, etc.).

### GRPO Algorithm Specifics

Standard GRPO for a group of G completions for prompt q:

```
For each prompt q_i:
  Generate y_1, ..., y_G ~ π_θ(·|q_i)
  Compute rewards r_1, ..., r_G
  Compute advantages: A_j = (r_j - mean(r)) / (std(r) + ε)
  
For the policy update:
  L = -E[min(ρ * A, clip(ρ, 1-ε, 1+ε) * A)] + β * KL(π_θ || π_ref)
  where ρ = π_θ(y|q) / π_old(y|q)
```

But wait -- with binary rewards and G=2, the advantages are very simple:
- If both correct or both wrong: A = 0 for both (no gradient signal)
- If one correct, one wrong: correct gets A=+1 (normalized), wrong gets A=-1

This means we only get gradient signal when the group has MIXED outcomes. If the model always gets it right or always wrong, no learning happens. This is actually a known issue with small group sizes on binary rewards.

Solutions:
1. Larger G (G=4 or G=8) -- but slower
2. Curriculum: pick problems at the boundary of the model's ability
3. Temperature sampling to increase diversity
4. Accept that many prompts give no signal and iterate faster

For V1, let's use G=4 with temperature=0.7. This gives a reasonable chance of mixed outcomes.

### Which Layers to Adapt?

The validation test adapted 4 q_proj layers. But the model has 210 AutoBitLinear layers across 7 types × 30 decoder layers. Options:
- **All 210**: 210 trainable parameters. Still tiny. But gradient computation through all adapted layers is expensive (many TinyLoRA forward passes).
- **All q_proj (30 layers)**: 30 parameters. Attention steering only.
- **One per decoder layer (30 layers)**: Pick one projection type (q_proj) per layer. 30 params.
- **Specific functional groups**: e.g., all gate_proj (controls MLP gating). 30 params.
- **Start small, grow**: Begin with 4, demonstrate learning, scale up.

Actually, the compute cost of TinyLoRA adapters is negligible -- it's just `scale * (x @ v^T) @ u^T`, a rank-1 matmul. Even 210 of them add almost nothing to forward time. The cost is in the BASE model's forward/backward, which we pay regardless.

For V1: adapt all 210 layers. 210 trainable scalars. This gives maximum flexibility for the RL to find which layers matter. We can analyze which scales moved the most post-training.

### KL Penalty

GRPO uses KL divergence between the current policy π_θ and a reference policy π_ref (usually the initial model). This prevents the model from degenerate solutions.

For TinyLoRA, the reference policy IS the base model (adapters initialized to 0). The KL penalty prevents adapters from growing too large. This is natural regularization.

Computing KL: need log-probs from both π_θ and π_ref for the generated sequences. This means we need TWO forward passes per sequence -- one with adapters, one without. That doubles our forward cost.

Alternative: skip KL for V1. The adapters are so tiny (210 scalars near 0) that they can't deviate far from the reference anyway. Add KL later if we see degenerate behavior.

Actually, no. KL is important for stability. But we can approximate it cheaply: since π_θ ≈ π_ref (tiny adapters), the KL is approximately:
```
KL ≈ Σ_i (scale_i)^2 * ||u_i||^2 * ||v_i||^2 / (something)
```

Actually this is getting complicated. Let's just do the two forward passes. One with adapters (π_θ), one without (set all scales to 0 temporarily). The forward pass is ~1.3s, so adding another 1.3s for reference log-probs is a 50% overhead but still manageable.

Wait, simpler: we can store the reference log-probs from the initial model once at the start of training and reuse them for each prompt. No -- that doesn't work because the sequences are generated fresh each time.

OK, the standard approach: for each generated sequence, compute log-probs under both π_θ and π_ref. Since π_ref is just the base model (adapters=0), we can do this by temporarily zeroing out the adapter scales, doing a forward pass, then restoring them. That avoids loading the model twice.

### Token-Level vs Sequence-Level

GRPO can operate at token level or sequence level:
- **Sequence-level**: one reward per completion, advantage applied uniformly to all tokens
- **Token-level**: reward at each token (harder to define for GSM8K)

For GSM8K with binary reward, sequence-level is natural. The advantage of the whole completion is +1 or -1.

But the actual GRPO loss operates per-token:
```
L = -Σ_t [min(ρ_t * A, clip(ρ_t, 1-ε, 1+ε) * A)]
```
where ρ_t = π_θ(y_t|y_<t, q) / π_old(y_t|y_<t, q)

So we need per-token log-probs even though the advantage is sequence-level.

### Memory Concerns

At 64GB RAM, we need to be careful. Current model uses ~4.5GB. Packed weights for soft-chip: ~497MB. FP32 LM head copy: ~1.3GB. Total ~6.3GB.

For GRPO, we need to store:
- G=4 generated sequences per prompt (just token IDs, negligible)
- Log-probs for each token in each sequence (float32, ~100 tokens × 4 sequences × 128256 vocab × 4 bytes... wait, we only need the log-prob of the chosen token, not the full distribution)
- Log-probs under π_ref (same size)
- Gradient computation buffers

Actually, log-probs are just one float per token: log π(y_t | y_<t, q). So 100 tokens × 4 sequences = 400 floats. Negligible.

The memory bottleneck is the forward/backward pass itself, which we've already validated works at ~6-8GB total. We have 64GB, so no issue.

### GSM8K Data Loading

Need to load GSM8K dataset. HuggingFace datasets: `load_dataset("gsm8k", "main")`. Training set: 7,473 examples. Test set: 1,319 examples.

Each example has:
- `question`: the math problem
- `answer`: the solution with reasoning, ending in `#### <number>`

We use the question as the prompt, and evaluate whether the model's completion produces the correct number.

### Prompt Format

BitNet b1.58 2B4T is a base model, not chat-tuned. We need to format prompts carefully:

```
Solve this math problem step by step.

Question: <question>
Answer: Let me solve this step by step.
```

Then we let the model generate freely and extract the answer.

Actually, for a base model, it might be better to use few-shot prompting:
```
Question: There are 15 trees in the grove. Grove workers will plant trees in the grove today. After they are done, there will be 21 trees. How many trees did the grove workers plant today?
Answer: There are 15 trees originally. Then there were 21 trees after some more were planted. So there must have been 21 - 15 = 6 trees planted. The answer is 6.

Question: <new question>
Answer:
```

Few-shot works better for base models and helps establish the output format.

### Putting It All Together: V1 Pipeline

```
1. Load model + soft-chip patches
2. Attach TinyLoRA adapters to all 210 layers
3. Load GSM8K training set
4. For each training step:
   a. Sample a batch of prompts (B=1 for V1)
   b. Generate G=4 completions per prompt (autoregressive, ~60s each)
   c. Extract answers, compute binary rewards
   d. Compute advantages within the group
   e. For each completion:
      - Forward pass with adapters → get log-probs π_θ
      - Forward pass without adapters → get log-probs π_ref
      - Compute GRPO loss per token
   f. Backward pass through the loss
   g. Optimizer step (Adam on 210 scalars)
   h. Log: step, reward, loss, adapter scale magnitudes
5. Evaluate on GSM8K test set periodically
```

### Timing Estimate for V1

Per training step (B=1 prompt, G=4 completions):
- 4 × rollout generation: 4 × 60s = 240s (4 min)
- 4 × forward (π_θ log-probs): 4 × 1.3s = 5.2s
- 4 × forward (π_ref log-probs): 4 × 1.3s = 5.2s
- 1 × backward: ~2.4s
- Total: ~253s ≈ 4.2 minutes per step

Overnight (8 hours): ~114 steps = 114 prompts processed.

This is enough. TinyLoRA paper trains for 200-500 steps. We can run two nights.

### Critical Simplification: No π_old

In standard GRPO/PPO, ρ = π_θ / π_old where π_old is the policy that generated the data. In our case, we generate data and immediately update -- there's no off-policy correction needed because we're doing on-policy updates. So ρ = 1 for the first (and only) epoch on each batch.

This simplifies the loss to:
```
L = -Σ_t [A * log π_θ(y_t | y_<t, q)] + β * KL(π_θ || π_ref)
```

This is just REINFORCE with a KL penalty! Even simpler.

Wait, but this loses the clipping that makes PPO/GRPO stable. Without clipping, a single high-advantage sequence could cause a huge update. With only 210 scalar parameters though, the damage is bounded -- each scalar can only change so much. And we have a learning rate to control step size.

For V1: use simple REINFORCE with group-normalized advantages and optional KL penalty. Add clipping later if needed.

### The Simplest Possible Implementation

Actually, stripping it down even further. What's the MINIMUM viable GRPO?

```python
for step in range(num_steps):
    # Sample prompt
    prompt = random.choice(gsm8k_train)
    
    # Generate G completions
    completions = [generate(model, prompt, max_tokens=256, temp=0.7) for _ in range(G)]
    
    # Score
    rewards = [1.0 if extract_answer(c) == prompt.answer else 0.0 for c in completions]
    
    # Advantages (group normalize)
    mean_r = mean(rewards)
    std_r = std(rewards) + 1e-8
    advantages = [(r - mean_r) / std_r for r in rewards]
    
    # Skip if no signal (all same reward)
    if std_r < 1e-6:
        continue
    
    # Policy gradient
    loss = 0
    for completion, advantage in zip(completions, advantages):
        log_probs = model.forward(prompt + completion)  # get per-token log probs
        loss += -advantage * log_probs.sum()
    loss /= G
    
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()
```

That's maybe 50 lines of actual training logic. The infrastructure (generation, answer extraction, soft-chip patching) is the real work.

### What Could Go Wrong

1. **Reward hacking**: model learns to output numbers that match training set answers without reasoning. Unlikely with 210 params.
2. **No signal**: if the base model never gets any GSM8K problems right (or always gets them right), we get no gradient signal. Need to check base model accuracy first.
3. **Gradient vanishing**: with 210 scalars near 0, gradients might be too small to learn. We saw gradients of 5e-6 to 1.3e-5 in the validation test. Need appropriate learning rate.
4. **Generation quality**: base model might not follow the prompt format and produce nonsense. Need good few-shot prompting.
5. **Answer extraction failures**: model output might not contain a parseable number. Need robust extraction.

### Pre-flight Check: Base Model GSM8K Accuracy

Before training, we should measure the base model's accuracy on GSM8K. This gives us a baseline and tells us if we'll get gradient signal:
- If accuracy is 0%: model is too weak, no positive examples to learn from
- If accuracy is 100%: nothing to improve
- Sweet spot: 10-50% accuracy. Enough correct examples for positive signal, enough wrong ones for contrast.

BitNet b1.58 2B4T has ~29% on GSM8K (reported in the paper). That's in the sweet spot.

## Questions Arising
- What's the actual GSM8K few-shot prompt format that works best with BitNet?
- Should we use temperature sampling or nucleus sampling for rollouts?
- How many few-shot examples to include in the prompt?
- What learning rate works for 210 tiny scalar parameters with REINFORCE?
- Should we do per-prompt or batched updates?
- How do we checkpoint/save the 210 scalar parameters?

## First Instincts
- Start with the simplest possible implementation (REINFORCE + group advantages)
- Use 2-shot prompting with GSM8K format
- G=4, temperature=0.7
- Adapt all 210 layers
- Skip KL penalty in V1 (add if needed)
- Skip clipping in V1 (add if needed)
- Log everything: rewards, advantages, scale magnitudes, loss
- Checkpoint every 10 steps (just save 210 floats)
- Pre-flight check: measure base model accuracy on 20 GSM8K problems
