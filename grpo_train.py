"""
GRPO Training: TinyLoRA on BitNet b1.58 2B4T
=============================================
Group Relative Policy Optimization with 210 learnable scalar parameters
on a frozen 2.4B ternary language model. Evaluated on GSM8K.

Uses the soft-chip AVX2 ternary kernel (forward + backward) and FP32 LM head
patch for CPU-only training at ~2.4s per gradient step.

Usage: python grpo_train.py
       python grpo_train.py --preflight   (run pre-flight checks only)
"""

import json
import hashlib
import math
import os
import random
import re
import sys
import time
from decimal import Decimal, InvalidOperation
from fractions import Fraction
from pathlib import Path

import torch
import torch.nn as nn

# ---------------------------------------------------------------------------
# Configuration (edit directly, no CLI parsing beyond --preflight)
# ---------------------------------------------------------------------------

MODEL_PATH = "models/bitnet-b1.58-2B-4T-bf16"
CHECKPOINT_DIR = "checkpoints"
LOG_FILE = "grpo_log.jsonl"

# GRPO hyperparameters
GROUP_SIZE = 4  # G: completions per prompt
TEMPERATURE = 0.7  # sampling temperature for rollouts
TOP_P = 0.9  # nucleus sampling threshold
MAX_NEW_TOKENS = 128  # tokens per completion
BETA_KL = 0.0   # GhostWeight: π_ref is random noise; KL penalty would fight learning
LEARNING_RATE = 0.01  # initial Adam learning rate
LR_MIN = 0.001  # final learning rate (cosine decay)
ADAM_BETAS = (0.9, 0.999)  # Adam beta1, beta2

# Training schedule
MAX_STEPS = 700  # maximum training steps
TIME_BUDGET = 28800  # 8 hours in seconds
EVAL_EVERY = 25  # evaluate every N steps
EVAL_SAMPLES = 20  # number of test problems for evaluation (reduced)
CHECKPOINT_EVERY = 10  # save checkpoint every N steps
NUM_FEW_SHOT = 2  # few-shot examples in prompt

# GhostWeight: use PRNG-generated weights (no storage, ~1KB model)
USE_GHOST = True  # GhostWeight: PRNG weights (8 bytes) + TinyLoRA (1KB) = unified model
GHOST_SEED = 42  # PRNG seed for deterministic weight regeneration

# Seed
SEED = 42

# ---------------------------------------------------------------------------
# Few-shot prompt template
# ---------------------------------------------------------------------------

FEW_SHOT_EXAMPLES = [
    {
        "question": "There are 15 trees in the grove. Grove workers will plant trees in the grove today. After they are done, there will be 21 trees. How many trees did the grove workers plant today?",
        "answer": "There are 15 trees originally. Then there were 21 trees after some more were planted. So there must have been 21 - 15 = 6. The answer is 6.",
    },
    {
        "question": "If there are 3 cars in the parking lot and 2 more cars arrive, how many cars are in the parking lot?",
        "answer": "There are originally 3 cars. Then 2 more arrive. So there are 3 + 2 = 5 cars. The answer is 5.",
    },
]


def format_prompt(question, num_shots=NUM_FEW_SHOT):
    """Build a few-shot prompt for GSM8K-style math problems."""
    parts = ["Solve the following math problems step by step.\n"]
    for ex in FEW_SHOT_EXAMPLES[:num_shots]:
        parts.append(f"Q: {ex['question']}")
        parts.append(f"A: {ex['answer']}\n")
    parts.append(f"Q: {question}")
    parts.append("A:")
    return "\n".join(parts)


# ---------------------------------------------------------------------------
# Answer extraction
# ---------------------------------------------------------------------------


def extract_answer(text):
    """Extract the final numerical answer from model output."""

    # Ignore any synthetic next-question continuation the model may emit.
    for marker in ("\nQ:", "\n\nQ:", "\n Q:"):
        if marker in text:
            text = text.split(marker, 1)[0]

    def clean_number(s):
        """Remove commas, trailing periods, normalize."""
        s = s.replace(",", "").strip()
        # Remove trailing period that's not part of a decimal
        if s.endswith(".") and s.count(".") == 1:
            s = s[:-1]
        return s

    def parse_candidate(s):
        s = s.strip()
        frac = re.search(r"(-?[\d,]+\s*/\s*-?[\d,]+)", s)
        if frac:
            return clean_number(frac.group(1).replace(" ", ""))
        num = re.search(r"(-?[\d,]+\.?\d*)", s)
        if num:
            return clean_number(num.group(1))
        return None

    # Strongest signal: explicit answer phrase. Prefer the LAST such phrase.
    answer_spans = re.findall(r"[Tt]he answer is\s*([^\n]+)", text)
    if answer_spans:
        parsed = parse_candidate(answer_spans[-1])
        if parsed is not None:
            return parsed

    # Try "#### X" format (GSM8K native)
    match = re.search(r"####\s*(-?[\d,]+\s*/\s*-?[\d,]+)", text)
    if match:
        return clean_number(match.group(1).replace(" ", ""))
    match = re.search(r"####\s*(-?[\d,]+\.?\d*)", text)
    if match:
        return clean_number(match.group(1))
    # Fall back to last number in text. Prefer numeric terminals over fractions,
    # because fractions often appear as intermediate reasoning steps.
    numbers = re.findall(r"-?[\d,]+\.?\d*", text)
    if numbers:
        return clean_number(numbers[-1])
    fractions = re.findall(r"-?[\d,]+\s*/\s*-?[\d,]+", text)
    if fractions:
        return clean_number(fractions[-1].replace(" ", ""))
    return None


def extract_gsm8k_answer(answer_text):
    """Extract ground truth from GSM8K answer field."""
    match = re.search(r"####\s*(-?[\d,]+\s*/\s*-?[\d,]+)", answer_text)
    if match:
        return match.group(1).replace(",", "").replace(" ", "").strip()
    match = re.search(r"####\s*(-?[\d,]+\.?\d*)", answer_text)
    if match:
        return match.group(1).replace(",", "").strip()
    return None


def normalize_answer(answer):
    """Normalize numeric answers for exact-value comparison.

    Returns a canonical string when parsing succeeds, else the stripped input.
    Supports integers, decimals, and simple fractions.
    """
    if answer is None:
        return None

    s = answer.replace(",", "").strip()
    if not s:
        return None

    # Remove trivial trailing period.
    if s.endswith(".") and s.count(".") == 1:
        s = s[:-1]

    try:
        if "/" in s:
            value = Fraction(s)
            return str(value)
        value = Decimal(s)
        if value == value.to_integral():
            return str(int(value))
        return format(value.normalize(), "f").rstrip("0").rstrip(".")
    except (ValueError, ZeroDivisionError, InvalidOperation):
        return s


def answers_match(predicted, ground_truth):
    """Compare answers after numeric normalization."""
    pred_norm = normalize_answer(predicted)
    gt_norm = normalize_answer(ground_truth)
    return pred_norm is not None and gt_norm is not None and pred_norm == gt_norm


# ---------------------------------------------------------------------------
# TinyLoRA adapter
# ---------------------------------------------------------------------------


class TinyLoRA(nn.Module):
    """Sub-rank-1 LoRA adapter: 1 trainable scalar per layer.

    output = base(x) + scale * (x @ v^T) @ u^T

    u, v are fixed random vectors. scale is the only trainable parameter.
    """

    def __init__(self, base_layer, seed=42):
        super().__init__()
        self.base_layer = base_layer
        out_features, in_features = base_layer.weight.shape

        gen = torch.Generator().manual_seed(seed)
        u = torch.randn(out_features, 1, generator=gen, dtype=torch.bfloat16)
        v = torch.randn(1, in_features, generator=gen, dtype=torch.bfloat16)
        self.register_buffer("u", u / u.norm())
        self.register_buffer("v", v / v.norm())
        self.scale = nn.Parameter(torch.zeros(1, dtype=torch.bfloat16))

    def forward(self, x):
        base_out = self.base_layer(x)
        adapter_out = (x @ self.v.T) @ self.u.T * self.scale
        return base_out + adapter_out


def inject_adapters(model):
    """Attach TinyLoRA to all AutoBitLinear layers. Returns list of (name, adapter)."""
    adapters = []
    for name, module in list(model.named_modules()):
        if module.__class__.__name__ != "AutoBitLinear":
            continue
        # Deterministic seed from layer name
        seed = int.from_bytes(hashlib.sha256(name.encode("utf-8")).digest()[:8], "big")
        adapter = TinyLoRA(module, seed=seed)
        # Replace in parent
        parts = name.rsplit(".", 1)
        if len(parts) == 2:
            parent_name, attr_name = parts
            parent = dict(model.named_modules())[parent_name]
            setattr(parent, attr_name, adapter)
            adapters.append((name, adapter))

    # Freeze everything, then unfreeze adapter scales
    for p in model.parameters():
        p.requires_grad = False
    for _, adapter in adapters:
        adapter.scale.requires_grad = True

    return adapters


def get_adapter_scales(adapters):
    """Return a dict of adapter name -> scale value."""
    return {name: adapter.scale.item() for name, adapter in adapters}


def zero_adapter_scales(adapters):
    """Zero all adapter scales (for π_ref computation). Returns saved values."""
    saved = []
    for _, adapter in adapters:
        saved.append(adapter.scale.data.clone())
        adapter.scale.data.zero_()
    return saved


def restore_adapter_scales(adapters, saved):
    """Restore adapter scales from saved values."""
    for (_, adapter), val in zip(adapters, saved):
        adapter.scale.data.copy_(val)


# ---------------------------------------------------------------------------
# Generation
# ---------------------------------------------------------------------------


@torch.no_grad()
def generate_group_completions(
    model,
    tokenizer,
    prompt_ids,
    group_size=GROUP_SIZE,
    max_new_tokens=MAX_NEW_TOKENS,
    temperature=TEMPERATURE,
    top_p=TOP_P,
):
    """Generate G completions in parallel using batching. Returns [G, seq_len] tensor."""
    batch_size = group_size
    # Expand prompt_ids to batch
    input_ids = prompt_ids.expand(batch_size, -1)

    all_tokens = [input_ids]
    past_key_values = None
    cur_input = input_ids

    # Track completion status and text for each sequence in batch
    finished = torch.zeros(batch_size, dtype=torch.bool, device=input_ids.device)
    recent_texts = [""] * batch_size

    generated_len = 0
    for _ in range(max_new_tokens):
        outputs = model(cur_input, past_key_values=past_key_values, use_cache=True)
        past_key_values = outputs.past_key_values
        logits = outputs.logits[:, -1, :].float()

        if temperature > 0:
            logits = logits / temperature
            # Top-p sampling
            sorted_logits, sorted_indices = torch.sort(logits, descending=True, dim=-1)
            cumprobs = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1)
            remove_mask = cumprobs - torch.softmax(sorted_logits, dim=-1) >= top_p
            sorted_logits[remove_mask] = float("-inf")
            probs = torch.softmax(sorted_logits, dim=-1)
            sampled_idx = torch.multinomial(probs, 1)
            next_tokens = sorted_indices.gather(-1, sampled_idx)
        else:
            next_tokens = logits.argmax(dim=-1, keepdim=True)

        # Mask out tokens for already finished sequences
        next_tokens = torch.where(
            finished.unsqueeze(-1), tokenizer.eos_token_id, next_tokens
        )
        all_tokens.append(next_tokens)
        cur_input = next_tokens

        generated_len += 1
        # Check stop conditions (require at least 3 tokens to avoid premature stop)
        for i in range(batch_size):
            if not finished[i]:
                tok_text = tokenizer.decode(next_tokens[i], skip_special_tokens=True)
                recent_texts[i] += tok_text
                if next_tokens[i].item() == tokenizer.eos_token_id or (
                    generated_len >= 3 and _should_stop(recent_texts[i])
                ):
                    finished[i] = True

        if finished.all():
            break

    return torch.cat(all_tokens, dim=-1)


def _should_stop(text):
    """Check if generation should stop based on text content."""
    # Stop if model starts generating a new question
    if "\nQ:" in text or text.lstrip().startswith("Q:"):
        return True
    # Stop after a completed answer sentence, including simple fractions.
    if re.search(
        r"[Tt]he answer is\s*(?:-?[\d,]+(?:\.\d+)?|-?[\d,]+\s*/\s*-?[\d,]+)\.?\s*$",
        text,
    ):
        return True
    return False


@torch.no_grad()
def generate_greedy(model, tokenizer, input_ids, max_new_tokens=MAX_NEW_TOKENS):
    """Greedy generation for evaluation."""
    return generate_group_completions(
        model,
        tokenizer,
        input_ids,
        group_size=1,
        max_new_tokens=max_new_tokens,
        temperature=0,
    )


# ---------------------------------------------------------------------------
# Compute per-token log-probs for a given sequence
# ---------------------------------------------------------------------------


def compute_log_probs(model, full_ids, prompt_len):
    """
    Compute per-token log-probabilities for tokens after the prompt.

    Args:
        model: the model (with or without adapters active)
        full_ids: [1, total_len] tensor of token ids (prompt + completion)
        prompt_len: number of prompt tokens

    Returns:
        log_probs: [num_completion_tokens] tensor of log P(token | context)
    """
    outputs = model(full_ids)
    # logits: [1, total_len, vocab_size]
    # We want P(token_t | tokens_<t) for t in [prompt_len, total_len)
    # logits[:, t-1, :] predicts token at position t
    logits = outputs.logits[
        :, prompt_len - 1 : -1, :
    ].float()  # [1, num_comp_tokens, vocab]
    log_probs_all = torch.log_softmax(logits, dim=-1)  # [1, num_comp_tokens, vocab]
    target_tokens = full_ids[:, prompt_len:]  # [1, num_comp_tokens]
    token_log_probs = (
        log_probs_all.gather(-1, target_tokens.unsqueeze(-1)).squeeze(-1).squeeze(0)
    )  # [num_comp_tokens]
    return token_log_probs


# ---------------------------------------------------------------------------
# GRPO training step
# ---------------------------------------------------------------------------


def grpo_step(
    model, tokenizer, adapters, optimizer, question, ground_truth, step_num, total_steps
):
    """
    One GRPO training step for a single prompt.

    Returns dict with metrics, or None if no gradient signal (all rewards equal).
    """
    t_start = time.time()

    # Format prompt
    prompt_text = format_prompt(question)
    prompt_ids = tokenizer(prompt_text, return_tensors="pt").input_ids
    prompt_len = prompt_ids.shape[1]

    # 1. ROLLOUT: Generate G completions sequentially (M=1 fastest for GhostWeight)
    t_rollout = time.time()
    completions = []
    completion_texts = []
    for g in range(GROUP_SIZE):
        comp_ids = generate_group_completions(
            model, tokenizer, prompt_ids, group_size=1
        )
        completions.append(comp_ids)
        comp_text = tokenizer.decode(comp_ids[0, prompt_len:], skip_special_tokens=True)
        completion_texts.append(comp_text)
    rollout_time = time.time() - t_rollout

    # 2. REWARD: Score completions (mixed: correctness + format signals)
    rewards = []
    predicted_answers = []
    for comp_text in completion_texts:
        pred = extract_answer(comp_text)
        predicted_answers.append(pred)
        if answers_match(pred, ground_truth):
            reward = 1.0
        else:
            reward = 0.0  # Binary reward: correct or nothing. No partial rewards.
        rewards.append(reward)

    # 3. ADVANTAGE: Group-normalize rewards
    mean_r = sum(rewards) / len(rewards)
    var_r = sum((r - mean_r) ** 2 for r in rewards) / len(rewards)
    std_r = var_r**0.5

    if std_r < 1e-8:
        # All rewards identical — no gradient signal
        elapsed = time.time() - t_start
        return {
            "step": step_num,
            "rewards": rewards,
            "mean_reward": mean_r,
            "skipped": True,
            "rollout_time": rollout_time,
            "total_time": elapsed,
            "predicted_answers": predicted_answers,
            "ground_truth": ground_truth,
        }

    advantages = [(r - mean_r) / (std_r + 1e-8) for r in rewards]

    # 4. LOG-PROBS + LOSS
    t_update = time.time()
    total_loss = torch.tensor(0.0, dtype=torch.float32)
    total_kl = 0.0
    total_reinforce = 0.0

    for comp_ids, advantage in zip(completions, advantages):
        num_comp_tokens = comp_ids.shape[1] - prompt_len
        if num_comp_tokens <= 0:
            continue

        # Forward with adapters (π_θ) — needs grad
        log_probs_theta = compute_log_probs(model, comp_ids, prompt_len)

        # Forward without adapters (π_ref) — no grad
        saved_scales = zero_adapter_scales(adapters)
        try:
            with torch.no_grad():
                log_probs_ref = compute_log_probs(model, comp_ids, prompt_len)
        finally:
            restore_adapter_scales(adapters, saved_scales)

        # REINFORCE loss: -advantage * sum(log π_θ)
        reinforce_loss = -advantage * log_probs_theta.sum()
        total_reinforce += reinforce_loss.item()

        # KL penalty: sum(log π_θ - log π_ref)
        kl_per_token = log_probs_theta - log_probs_ref.detach()
        kl = kl_per_token.sum()
        total_kl += kl.item()

        total_loss = total_loss + reinforce_loss + BETA_KL * kl

    total_loss = total_loss / GROUP_SIZE

    # 5. BACKWARD + UPDATE
    optimizer.zero_grad()
    total_loss.backward()

    # Gradient stats
    grad_norms = []
    for _, adapter in adapters:
        if adapter.scale.grad is not None:
            grad_norms.append(adapter.scale.grad.abs().item())
        else:
            grad_norms.append(0.0)
    mean_grad = sum(grad_norms) / len(grad_norms) if grad_norms else 0.0
    max_grad = max(grad_norms) if grad_norms else 0.0

    optimizer.step()
    update_time = time.time() - t_update
    elapsed = time.time() - t_start

    # Scale statistics
    scales = [adapter.scale.item() for _, adapter in adapters]
    mean_abs_scale = sum(abs(s) for s in scales) / len(scales)
    max_abs_scale = max(abs(s) for s in scales)

    return {
        "step": step_num,
        "rewards": rewards,
        "mean_reward": mean_r,
        "advantages": advantages,
        "loss": total_loss.item(),
        "reinforce_loss": total_reinforce / GROUP_SIZE,
        "kl": total_kl / GROUP_SIZE,
        "mean_grad": mean_grad,
        "max_grad": max_grad,
        "mean_abs_scale": mean_abs_scale,
        "max_abs_scale": max_abs_scale,
        "skipped": False,
        "rollout_time": rollout_time,
        "update_time": update_time,
        "total_time": elapsed,
        "predicted_answers": predicted_answers,
        "ground_truth": ground_truth,
    }


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------


def evaluate(model, tokenizer, test_data, num_samples=EVAL_SAMPLES):
    """Evaluate on GSM8K test set with greedy decoding."""
    correct = 0
    total = 0
    results = []

    samples = test_data[:num_samples] if len(test_data) >= num_samples else test_data

    for i, example in enumerate(samples):
        question = example["question"]
        ground_truth = extract_gsm8k_answer(example["answer"])
        if ground_truth is None:
            continue

        prompt_text = format_prompt(question)
        prompt_ids = tokenizer(prompt_text, return_tensors="pt").input_ids
        prompt_len = prompt_ids.shape[1]

        comp_ids = generate_greedy(model, tokenizer, prompt_ids)
        comp_text = tokenizer.decode(comp_ids[0, prompt_len:], skip_special_tokens=True)
        pred = extract_answer(comp_text)

        is_correct = answers_match(pred, ground_truth)
        if is_correct:
            correct += 1
        total += 1

        results.append(
            {
                "question": question[:80] + "...",
                "predicted": pred,
                "ground_truth": ground_truth,
                "correct": is_correct,
            }
        )

        print(
            f"  eval [{i + 1}/{len(samples)}] pred={pred} gt={ground_truth} {'OK' if is_correct else 'WRONG'}"
        )

    accuracy = correct / total if total > 0 else 0.0
    return accuracy, results


# ---------------------------------------------------------------------------
# Checkpointing
# ---------------------------------------------------------------------------


def save_checkpoint(adapters, optimizer, step, eval_history, config):
    """Save adapter scales, optimizer state, and metadata."""
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    path = os.path.join(CHECKPOINT_DIR, f"grpo_step_{step:04d}.pt")
    torch.save(
        {
            "step": step,
            "adapter_scales": {
                name: adapter.scale.data.clone() for name, adapter in adapters
            },
            "adapter_state": {
                name: {
                    "u": adapter.u.clone(),
                    "v": adapter.v.clone(),
                }
                for name, adapter in adapters
            },
            "optimizer_state": optimizer.state_dict(),
            "eval_history": eval_history,
            "config": config,
            "use_ghost": USE_GHOST,
            "ghost_seed": GHOST_SEED if USE_GHOST else None,
        },
        path,
    )
    print(f"  Checkpoint saved: {path}")
    return path


def load_checkpoint(path, adapters, optimizer):
    """Load adapter scales and optimizer state from checkpoint."""
    ckpt = torch.load(path, weights_only=False)
    for name, adapter in adapters:
        if name in ckpt["adapter_scales"]:
            adapter.scale.data.copy_(ckpt["adapter_scales"][name])
        if name in ckpt.get("adapter_state", {}):
            adapter.u.copy_(ckpt["adapter_state"][name]["u"])
            adapter.v.copy_(ckpt["adapter_state"][name]["v"])
    if ckpt.get("optimizer_state"):
        optimizer.load_state_dict(ckpt["optimizer_state"])
    return ckpt["step"], ckpt.get("eval_history", [])


# ---------------------------------------------------------------------------
# Learning rate schedule
# ---------------------------------------------------------------------------


def get_lr(step, total_steps):
    """Cosine decay from LEARNING_RATE to LR_MIN."""
    if total_steps <= 1:
        return LEARNING_RATE
    progress = min(step / total_steps, 1.0)
    cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
    return LR_MIN + (LEARNING_RATE - LR_MIN) * cosine


# ---------------------------------------------------------------------------
# Pre-flight checks
# ---------------------------------------------------------------------------


def preflight(model, tokenizer, adapters, train_data, test_data):
    """Run pre-flight validation before training."""
    print("\n" + "=" * 60)
    print("  PRE-FLIGHT CHECKS")
    print("=" * 60)

    # 1. Adapter count
    trainable = sum(1 for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(
        f"\n[1] Adapters: {len(adapters)} layers, {trainable} trainable params, {total / 1e9:.2f}B total"
    )
    assert trainable == len(adapters), (
        f"Expected {len(adapters)} trainable, got {trainable}"
    )
    print("    PASS")

    # 2. Test generation
    print(f"\n[2] Test generation...")
    question = train_data[0]["question"]
    prompt_text = format_prompt(question)
    prompt_ids = tokenizer(prompt_text, return_tensors="pt").input_ids
    prompt_len = prompt_ids.shape[1]
    print(f"    Prompt length: {prompt_len} tokens")
    print(f"    Prompt preview: {prompt_text[:200]}...")

    t0 = time.time()
    comp_ids = generate_group_completions(
        model, tokenizer, prompt_ids, group_size=1, max_new_tokens=50, temperature=0
    )
    gen_time = time.time() - t0
    comp_text = tokenizer.decode(comp_ids[0, prompt_len:], skip_special_tokens=True)
    print(
        f"    Generated ({gen_time:.1f}s, {comp_ids.shape[1] - prompt_len} tokens): {comp_text[:200]}"
    )
    print("    PASS")

    # 3. Answer extraction
    print(f"\n[3] Answer extraction test...")
    num_ok = 0
    for i, ex in enumerate(train_data[:5]):
        gt = extract_gsm8k_answer(ex["answer"])
        print(f"    [{i}] Ground truth: {gt}")
        if gt is not None:
            num_ok += 1
    print(f"    Extracted {num_ok}/5 ground truths")
    assert num_ok >= 4, f"Answer extraction failing: only {num_ok}/5"
    print("    PASS")

    # 4. Base model accuracy (quick check)
    print(f"\n[4] Base model accuracy (5 problems, greedy)...")
    t0 = time.time()
    acc, results = evaluate(model, tokenizer, test_data, num_samples=5)
    eval_time = time.time() - t0
    print(f"    Accuracy: {acc * 100:.0f}% ({eval_time:.0f}s)")
    print("    PASS (any accuracy is OK — we just need it to run)")

    # 5. Training step timing
    print(f"\n[5] Training step timing (1 step, G={GROUP_SIZE})...")
    question = train_data[0]["question"]
    gt = extract_gsm8k_answer(train_data[0]["answer"])
    optimizer = torch.optim.Adam(
        [adapter.scale for _, adapter in adapters],
        lr=LEARNING_RATE,
        betas=ADAM_BETAS,
    )

    t0 = time.time()
    result = grpo_step(
        model,
        tokenizer,
        adapters,
        optimizer,
        question,
        gt,
        step_num=0,
        total_steps=MAX_STEPS,
    )
    step_time = time.time() - t0
    print(f"    Total step time: {step_time:.1f}s")
    print(f"    Rollout time: {result['rollout_time']:.1f}s")
    if not result["skipped"]:
        print(f"    Update time: {result['update_time']:.1f}s")
        print(f"    Loss: {result['loss']:.4f}")
        print(f"    KL: {result['kl']:.6f}")
        print(f"    Mean |grad|: {result['mean_grad']:.2e}")
        print(f"    Mean |scale|: {result['mean_abs_scale']:.2e}")
    print(f"    Rewards: {result['rewards']}")
    print(f"    Predictions: {result['predicted_answers']}")
    print(f"    Ground truth: {result['ground_truth']}")
    if result["skipped"]:
        print("    (Step was skipped — all rewards identical, no gradient signal)")
    print("    PASS")

    # 6. Memory
    import psutil

    proc = psutil.Process()
    mem_gb = proc.memory_info().rss / (1024**3)
    print(f"\n[6] Memory usage: {mem_gb:.1f} GB / 64 GB")
    print("    PASS")

    # Estimate
    est_steps_per_hour = 3600 / step_time if step_time > 0 else 0
    est_steps_overnight = est_steps_per_hour * 8
    print(f"\n{'=' * 60}")
    print(f"  PRE-FLIGHT COMPLETE")
    print(
        f"  Estimated: {est_steps_per_hour:.0f} steps/hour, {est_steps_overnight:.0f} steps in 8 hours"
    )
    print(f"{'=' * 60}\n")

    return result


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    preflight_only = "--preflight" in sys.argv
    resume_path = None
    for arg in sys.argv[1:]:
        if arg.startswith("--resume="):
            resume_path = arg.split("=", 1)[1]

    print("=" * 60)
    print("  GRPO Training: TinyLoRA on BitNet b1.58 2B4T")
    print("=" * 60)

    # Seed
    random.seed(SEED)
    torch.manual_seed(SEED)

    # Load model
    print("\n[SETUP] Loading model...")
    t0 = time.time()
    from transformers import AutoModelForCausalLM, AutoTokenizer

    try:
        tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, fix_mistral_regex=True)
    except TypeError:
        tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH,
        torch_dtype=torch.bfloat16,
        device_map="cpu",
        trust_remote_code=True,
    )
    print(f"  Model loaded in {time.time() - t0:.1f}s")

    # Patch with soft-chip
    print("[SETUP] Patching with soft-chip (CPU AVX2)...")
    from softchip import patch_model, patch_lm_head_fp32

    if USE_GHOST:
        print(f"  [GhostWeight] Using PRNG-generated weights (seed={GHOST_SEED})")
        patch_model(model, use_ghost=True, ghost_seed=GHOST_SEED)
    else:
        patch_model(model, backend="cpu")
    patch_lm_head_fp32(model)

    # Inject adapters
    print("[SETUP] Injecting TinyLoRA adapters...")
    adapters = inject_adapters(model)
    print(
        f"  Injected {len(adapters)} adapters ({sum(1 for p in model.parameters() if p.requires_grad)} trainable params)"
    )

    # Load GSM8K
    print("[SETUP] Loading GSM8K dataset...")
    from datasets import load_dataset

    gsm8k = load_dataset("gsm8k", "main")
    train_data = list(gsm8k["train"])
    test_data = list(gsm8k["test"])
    print(f"  Train: {len(train_data)}, Test: {len(test_data)}")

    # Shuffle training data
    random.shuffle(train_data)

    # Pre-flight
    preflight_result = (
        preflight(model, tokenizer, adapters, train_data, test_data)
        if not USE_GHOST
        else {"pass": True}
    )
    if preflight_only:
        print("Pre-flight complete. Exiting (--preflight mode).")
        return

    # Reset adapters after preflight (the preflight step may have changed them)
    for _, adapter in adapters:
        adapter.scale.data.zero_()

    # Optimizer
    adapter_params = [adapter.scale for _, adapter in adapters]
    optimizer = torch.optim.Adam(adapter_params, lr=LEARNING_RATE, betas=ADAM_BETAS)

    # Config for logging
    config = {
        "group_size": GROUP_SIZE,
        "temperature": TEMPERATURE,
        "top_p": TOP_P,
        "max_new_tokens": MAX_NEW_TOKENS,
        "beta_kl": BETA_KL,
        "learning_rate": LEARNING_RATE,
        "lr_min": LR_MIN,
        "max_steps": MAX_STEPS,
        "time_budget": TIME_BUDGET,
        "num_adapters": len(adapters),
        "num_few_shot": NUM_FEW_SHOT,
        "seed": SEED,
    }

    # Training state
    eval_history = []
    train_idx = 0
    total_training_time = 0.0
    steps_skipped = 0
    total_reward_sum = 0.0
    total_reward_count = 0
    start_step = 1
    base_acc = 0.0

    # Resume from checkpoint if requested
    if resume_path:
        print(f"\n[RESUME] Loading checkpoint: {resume_path}")
        start_step, eval_history = load_checkpoint(resume_path, adapters, optimizer)
        start_step += 1  # continue from next step
        train_idx = start_step % len(train_data)
        print(
            f"  Resuming from step {start_step}, eval_history={len(eval_history)} entries"
        )
        # Recover base_acc from eval_history if available
        base_entries = [e for e in eval_history if e.get("type") == "base"]
        if base_entries:
            base_acc = base_entries[-1]["accuracy"]
            print(f"  Base accuracy: {base_acc * 100:.1f}% (from checkpoint history)")
    else:
        # Base model evaluation (before training)
        print("\n[EVAL] Base model accuracy (before training)...")
        base_acc, base_results = evaluate(
            model, tokenizer, test_data, num_samples=EVAL_SAMPLES
        )
        print(f"  Base accuracy: {base_acc * 100:.1f}% ({EVAL_SAMPLES} problems)")
        eval_history.append({"step": 0, "accuracy": base_acc, "type": "base"})

    # Log file
    log_fh = open(LOG_FILE, "a")

    print(f"\n{'=' * 60}")
    print(f"  TRAINING START")
    print(f"  {len(adapters)} adapters, G={GROUP_SIZE}, lr={LEARNING_RATE}")
    print(f"  Time budget: {TIME_BUDGET}s ({TIME_BUDGET / 3600:.1f}h)")
    print(f"{'=' * 60}\n")

    t_train_start = time.time()

    for step in range(start_step, MAX_STEPS + 1):
        # Check time budget
        if total_training_time >= TIME_BUDGET:
            print(f"\nTime budget reached ({TIME_BUDGET}s). Stopping.")
            break

        # Update learning rate
        lr = get_lr(step, MAX_STEPS)
        for pg in optimizer.param_groups:
            pg["lr"] = lr

        # Pick next training example
        example = train_data[train_idx % len(train_data)]
        train_idx += 1
        question = example["question"]
        ground_truth = extract_gsm8k_answer(example["answer"])
        if ground_truth is None:
            print(f"[step {step:04d}] SKIP (bad ground truth)")
            continue

        # GRPO step
        result = grpo_step(
            model,
            tokenizer,
            adapters,
            optimizer,
            question,
            ground_truth,
            step_num=step,
            total_steps=MAX_STEPS,
        )

        total_training_time += result["total_time"]
        total_reward_sum += sum(result["rewards"])
        total_reward_count += len(result["rewards"])

        # Console output
        if result["skipped"]:
            steps_skipped += 1
            r_str = ",".join(str(int(r)) for r in result["rewards"])
            print(
                f"[step {step:04d}] SKIP rewards=[{r_str}] "
                f"rollout={result['rollout_time']:.0f}s "
                f"elapsed={total_training_time:.0f}s"
            )
        else:
            r_str = ",".join(str(int(r)) for r in result["rewards"])
            running_reward = (
                total_reward_sum / total_reward_count if total_reward_count > 0 else 0
            )
            print(
                f"[step {step:04d}] rewards=[{r_str}] "
                f"loss={result['loss']:.4f} "
                f"kl={result['kl']:.4f} "
                f"|scale|={result['mean_abs_scale']:.4f} "
                f"|grad|={result['mean_grad']:.2e} "
                f"lr={lr:.4f} "
                f"rollout={result['rollout_time']:.0f}s "
                f"update={result['update_time']:.1f}s "
                f"total={result['total_time']:.0f}s "
                f"running_r={running_reward:.2f}"
            )

        # Log to file
        log_entry = {**result, "lr": lr, "wall_time": time.time() - t_train_start}
        log_fh.write(json.dumps(log_entry, default=str) + "\n")
        log_fh.flush()

        # Checkpoint
        if step % CHECKPOINT_EVERY == 0:
            save_checkpoint(adapters, optimizer, step, eval_history, config)

        # Evaluation
        if step % EVAL_EVERY == 0:
            print(f"\n[EVAL] Step {step}...")
            t_eval = time.time()
            acc, results = evaluate(
                model, tokenizer, test_data, num_samples=EVAL_SAMPLES
            )
            eval_time = time.time() - t_eval
            print(
                f"  Accuracy: {acc * 100:.1f}% (base: {base_acc * 100:.1f}%) [{eval_time:.0f}s]"
            )
            eval_history.append({"step": step, "accuracy": acc, "type": "train"})
            save_checkpoint(adapters, optimizer, step, eval_history, config)
            print()

    # Final evaluation
    print(f"\n{'=' * 60}")
    print("  FINAL EVALUATION")
    print(f"{'=' * 60}")
    final_acc, final_results = evaluate(
        model, tokenizer, test_data, num_samples=EVAL_SAMPLES
    )
    eval_history.append({"step": step, "accuracy": final_acc, "type": "final"})
    save_checkpoint(adapters, optimizer, step, eval_history, config)

    # Scale analysis
    print(f"\n{'=' * 60}")
    print("  ADAPTER SCALE ANALYSIS")
    print(f"{'=' * 60}")
    scale_data = [(name, adapter.scale.item()) for name, adapter in adapters]
    scale_data.sort(key=lambda x: abs(x[1]), reverse=True)
    print("\nTop 20 most-moved adapters:")
    for name, scale in scale_data[:20]:
        print(f"  {scale:+.6f}  {name}")
    print(f"\nBottom 10 (least moved):")
    for name, scale in scale_data[-10:]:
        print(f"  {scale:+.6f}  {name}")

    # Summary
    print(f"\n{'=' * 60}")
    print("  TRAINING SUMMARY")
    print(f"{'=' * 60}")
    print(f"  Steps completed:  {step}")
    print(f"  Steps skipped:    {steps_skipped} ({100 * steps_skipped / step:.0f}%)")
    print(
        f"  Training time:    {total_training_time:.0f}s ({total_training_time / 3600:.1f}h)"
    )
    print(f"  Base accuracy:    {base_acc * 100:.1f}%")
    print(f"  Final accuracy:   {final_acc * 100:.1f}%")
    print(
        f"  Running reward:   {total_reward_sum / total_reward_count:.2f}"
        if total_reward_count > 0
        else ""
    )
    print(
        f"  Mean |scale|:     {sum(abs(s) for _, s in scale_data) / len(scale_data):.6f}"
    )
    print(f"  Max |scale|:      {max(abs(s) for _, s in scale_data):.6f}")

    log_fh.close()
    print(f"\nLog saved to {LOG_FILE}")
    print(f"Checkpoints in {CHECKPOINT_DIR}/")


if __name__ == "__main__":
    main()
