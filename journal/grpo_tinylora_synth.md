# Synthesis: GRPO Training Loop for TinyLoRA on BitNet b1.58

## Objective

Implement a GRPO (Group Relative Policy Optimization) training loop that uses reinforcement learning to train 210 TinyLoRA scalar parameters on the frozen BitNet b1.58 2B4T model, evaluated on GSM8K math problems.

---

## Architecture

```
┌──────────────────────────────────────────────────────────────┐
│                     GRPO TRAINING LOOP                        │
├──────────────────────────────────────────────────────────────┤
│                                                               │
│  SETUP:                                                       │
│  ├─ Load BitNet b1.58 2B4T (BF16 master weights)            │
│  ├─ Patch with soft-chip (CPU AVX2 backend)                  │
│  ├─ Patch LM head with FP32 fix                             │
│  ├─ Attach TinyLoRA to all 210 AutoBitLinear layers          │
│  ├─ Freeze all base model params                             │
│  └─ Load GSM8K dataset                                       │
│                                                               │
│  TRAINING STEP (repeat):                                      │
│  ├─ 1. SAMPLE: Pick random GSM8K training prompt             │
│  ├─ 2. ROLLOUT: Generate G=4 completions (autoregressive)    │
│  │      └─ Soft-chip forward, temp=0.7, top-p=0.9           │
│  │      └─ ~60s per completion, ~240s total                  │
│  ├─ 3. REWARD: Extract answer, compare to ground truth       │
│  │      └─ R=1 if correct, R=0 if wrong                     │
│  ├─ 4. ADVANTAGE: Group-normalize rewards                    │
│  │      └─ A_j = (r_j - mean(r)) / (std(r) + ε)           │
│  │      └─ Skip step if all rewards identical                │
│  ├─ 5. LOG-PROBS: Forward each completion through model      │
│  │      └─ With adapters → log π_θ per token                │
│  │      └─ Without adapters → log π_ref per token            │
│  ├─ 6. LOSS: REINFORCE + KL                                  │
│  │      └─ L = -Σ A_j * Σ_t log π_θ(y_t|...)              │
│  │      └─ + β * Σ_t [log π_θ - log π_ref]                 │
│  ├─ 7. BACKWARD: Compute gradients for 210 adapter scales   │
│  ├─ 8. UPDATE: Adam optimizer step                           │
│  └─ 9. LOG: Rewards, loss, KL, scale magnitudes             │
│                                                               │
│  EVALUATE (every 25 steps):                                   │
│  └─ Greedy decode on 20 GSM8K test problems                 │
│  └─ Report accuracy                                          │
│                                                               │
│  CHECKPOINT (every 10 steps):                                 │
│  └─ Save 210 adapter scales + optimizer state (~2KB)         │
└──────────────────────────────────────────────────────────────┘
```

---

## Implementation Spec

### File: `grpo_train.py`

Single self-contained training script. No CLI arguments — edit constants directly (matching project convention from `train.py`).

### Key Components

#### 1. TinyLoRA Adapter (reuse from validation, extend to all layers)

```python
class TinyLoRA(nn.Module):
    def __init__(self, base_layer, seed=42):
        super().__init__()
        self.base_layer = base_layer
        out_features, in_features = base_layer.weight.shape
        gen = torch.Generator().manual_seed(seed)
        u = torch.randn(out_features, 1, generator=gen, dtype=torch.bfloat16)
        v = torch.randn(1, in_features, generator=gen, dtype=torch.bfloat16)
        self.register_buffer('u', u / u.norm())
        self.register_buffer('v', v / v.norm())
        self.scale = nn.Parameter(torch.zeros(1, dtype=torch.bfloat16))

    def forward(self, x):
        return self.base_layer(x) + (x @ self.v.T) @ self.u.T * self.scale
```

#### 2. Adapter Injection

```python
def inject_adapters(model):
    """Attach TinyLoRA to all 210 AutoBitLinear layers."""
    adapters = []
    for name, module in list(model.named_modules()):
        if module.__class__.__name__ != "AutoBitLinear":
            continue
        adapter = TinyLoRA(module, seed=hash(name) % (2**31))
        parent_name, attr_name = name.rsplit(".", 1)
        parent = dict(model.named_modules())[parent_name]
        setattr(parent, attr_name, adapter)
        adapters.append((name, adapter))
    # Freeze everything, then unfreeze adapter scales
    for p in model.parameters():
        p.requires_grad = False
    for _, adapter in adapters:
        adapter.scale.requires_grad = True
    return adapters
```

#### 3. Autoregressive Generation

```python
@torch.no_grad()
def generate(model, tokenizer, prompt_ids, max_new_tokens=256,
             temperature=0.7, top_p=0.9, stop_strings=None):
    """Generate one completion autoregressively using the patched model."""
    input_ids = prompt_ids.clone()
    for _ in range(max_new_tokens):
        logits = model(input_ids).logits[:, -1, :]  # [1, vocab_size]
        logits = logits / temperature
        # Top-p sampling
        sorted_logits, sorted_idx = torch.sort(logits, descending=True)
        cumprobs = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1)
        mask = cumprobs - torch.softmax(sorted_logits, dim=-1) >= top_p
        sorted_logits[mask] = float('-inf')
        probs = torch.softmax(sorted_logits, dim=-1)
        next_token_pos = torch.multinomial(probs, 1)
        next_token = sorted_idx.gather(-1, next_token_pos)
        input_ids = torch.cat([input_ids, next_token], dim=-1)
        # Check stop condition
        if stop_strings and should_stop(tokenizer, input_ids, stop_strings):
            break
    return input_ids
```

#### 4. Answer Extraction

```python
import re

def extract_answer(text):
    """Extract the final numerical answer from model output."""
    # Try "The answer is X" format first
    match = re.search(r'[Tt]he answer is\s*\$?\\?boxed\{?(\-?[\d,]+\.?\d*)', text)
    if match:
        return match.group(1).replace(',', '')
    # Try "#### X" format (GSM8K native)
    match = re.search(r'####\s*(\-?[\d,]+\.?\d*)', text)
    if match:
        return match.group(1).replace(',', '')
    # Fall back to last number in text
    numbers = re.findall(r'\-?[\d,]+\.?\d*', text)
    if numbers:
        return numbers[-1].replace(',', '')
    return None

def extract_gsm8k_answer(answer_text):
    """Extract ground truth from GSM8K answer field."""
    match = re.search(r'####\s*(\-?[\d,]+\.?\d*)', answer_text)
    if match:
        return match.group(1).replace(',', '')
    return None
```

#### 5. GRPO Training Step

```python
def grpo_step(model, tokenizer, adapters, optimizer, prompt, answer,
              G=4, temperature=0.7, beta_kl=0.01):
    """One GRPO training step for a single prompt."""
    
    # 1. Generate G completions
    prompt_ids = tokenizer(prompt, return_tensors="pt").input_ids
    completions = []
    for g in range(G):
        comp_ids = generate(model, tokenizer, prompt_ids, temperature=temperature)
        completions.append(comp_ids)
    
    # 2. Score completions
    rewards = []
    for comp_ids in completions:
        text = tokenizer.decode(comp_ids[0], skip_special_tokens=True)
        pred = extract_answer(text)
        reward = 1.0 if pred is not None and pred == answer else 0.0
        rewards.append(reward)
    
    # 3. Compute advantages
    mean_r = sum(rewards) / len(rewards)
    std_r = (sum((r - mean_r)**2 for r in rewards) / len(rewards)) ** 0.5
    if std_r < 1e-8:
        return None  # No signal (all same reward)
    advantages = [(r - mean_r) / (std_r + 1e-8) for r in rewards]
    
    # 4. Compute log-probs and loss
    total_loss = torch.tensor(0.0, dtype=torch.float32)
    total_kl = 0.0
    
    for comp_ids, advantage in zip(completions, advantages):
        prompt_len = prompt_ids.shape[1]
        
        # Forward with adapters (π_θ)
        outputs_theta = model(comp_ids)
        logits_theta = outputs_theta.logits[:, prompt_len-1:-1, :]
        log_probs_theta = torch.log_softmax(logits_theta.float(), dim=-1)
        target_tokens = comp_ids[:, prompt_len:]
        token_log_probs_theta = log_probs_theta.gather(
            -1, target_tokens.unsqueeze(-1)
        ).squeeze(-1)
        
        # Forward without adapters (π_ref) — zero out scales temporarily
        old_scales = []
        for _, adapter in adapters:
            old_scales.append(adapter.scale.data.clone())
            adapter.scale.data.zero_()
        
        with torch.no_grad():
            outputs_ref = model(comp_ids)
        logits_ref = outputs_ref.logits[:, prompt_len-1:-1, :]
        log_probs_ref = torch.log_softmax(logits_ref.float(), dim=-1)
        token_log_probs_ref = log_probs_ref.gather(
            -1, target_tokens.unsqueeze(-1)
        ).squeeze(-1)
        
        # Restore scales
        for (_, adapter), old_scale in zip(adapters, old_scales):
            adapter.scale.data.copy_(old_scale)
        
        # REINFORCE loss: -advantage * sum of log probs
        reinforce_loss = -advantage * token_log_probs_theta.sum()
        
        # KL penalty: sum of (log π_θ - log π_ref)
        kl = (token_log_probs_theta - token_log_probs_ref).sum()
        total_kl += kl.item()
        
        total_loss = total_loss + reinforce_loss + beta_kl * kl
    
    total_loss = total_loss / G
    
    # 5. Backward + update
    optimizer.zero_grad()
    total_loss.backward()
    optimizer.step()
    
    return {
        'rewards': rewards,
        'mean_reward': mean_r,
        'advantages': advantages,
        'loss': total_loss.item(),
        'kl': total_kl / G,
    }
```

---

## Hyperparameters

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| G (group size) | 4 | Balance signal rate vs rollout cost |
| temperature | 0.7 | Diverse but coherent generation |
| top_p | 0.9 | Cut long tail |
| max_new_tokens | 256 | Generous for GSM8K solutions |
| β_KL | 0.01 | Light regularization |
| learning rate | 0.01, cosine → 0.001 | Conservative for noisy REINFORCE |
| optimizer | Adam(β1=0.9, β2=0.999) | Standard, good for sparse updates |
| adapted layers | all 210 | Let RL discover which matter |
| num_few_shot | 2 | Format guidance without excessive length |
| eval_every | 25 steps | ~2 hours between evals |
| checkpoint_every | 10 steps | ~40 min between saves |
| max_steps | 500 | ~35 hours total, run over multiple nights |
| time_budget | 28800 (8h) | Single overnight session |

---

## Prompt Template

```
Solve the following math problems step by step.

Q: There are 15 trees in the grove. Grove workers will plant trees in the grove today. After they are done, there will be 21 trees. How many trees did the grove workers plant today?
A: There are 15 trees originally. Then there were 21 trees after some more were planted. So there must have been 21 - 15 = 6. The answer is 6.

Q: If there are 3 cars in the parking lot and 2 more cars arrive, how many cars are in the parking lot?
A: There are originally 3 cars. Then 2 more arrive. So there are 3 + 2 = 5 cars. The answer is 5.

Q: {question}
A:
```

---

## Pre-flight Checklist

Before starting training:
1. [ ] Load model + soft-chip + LM head patch + 210 adapters
2. [ ] Verify: 210 trainable params, all others frozen
3. [ ] Test generation on 1 prompt (verify parseable output)
4. [ ] Test answer extraction on 5 GSM8K examples
5. [ ] Measure base model accuracy on 10 GSM8K problems
6. [ ] Time one full training step (rollout + forward + backward + update)
7. [ ] Measure peak memory
8. [ ] Run 3 training steps, verify loss decreases and adapters update

---

## Expected Outputs

### Training Log (console)
```
[step 001] rewards=[0,0,1,0] mean_r=0.25 loss=2.34 kl=0.001 |scales|=0.002 time=253s
[step 002] rewards=[1,0,0,1] mean_r=0.50 loss=1.87 kl=0.003 |scales|=0.005 time=261s
[step 003] rewards=[0,0,0,0] SKIP (no signal)
...
[step 025] EVAL: 4/20 correct (20.0%) [base: 6/20 (30.0%)]
```

### Checkpoint (`checkpoints/grpo_step_010.pt`)
```python
{
    'step': 10,
    'adapter_scales': tensor([...]),  # 210 floats
    'adapter_names': [...],           # 210 layer names
    'optimizer_state': {...},
    'config': {...},
    'eval_history': [...],
}
```

---

## Success Criteria

1. **Minimum:** Training loop runs for 25+ steps without crash
2. **Signal:** At least some steps show non-zero gradient updates
3. **Learning:** Adapter scales move from zero (measurable change)
4. **Stretch:** Measurable change in GSM8K accuracy (even 1% up or down is informative)
5. **Science:** Which of the 210 layers' scales moved most? (attention vs MLP, early vs late layers)

---

## What This Unlocks

If GRPO training works (even partially):
1. **First GRPO on BitNet:** RL training on a natively ternary model
2. **First TinyLoRA + RL on BitNet:** Sub-rank-1 LoRA with reinforcement learning on 1.58-bit weights
3. **Autoresearch foundation:** The training loop becomes the inner loop of the autonomous research agent
4. **Publication-ready result:** 210 parameters improving (or measurably affecting) GSM8K accuracy on a 2.4B ternary model

---

## One-Sentence Summary

A single-file GRPO training loop that uses 210 learnable scalars to search for minimal parameter interventions that improve math reasoning in a 2.4-billion-parameter ternary language model.
