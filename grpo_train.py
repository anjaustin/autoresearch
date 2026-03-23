"""
GRPO Training: GhostChain on BitNet b1.58 2B4T
=============================================
Group Relative Policy Optimization with 840 learnable scalar parameters
(GhostChain: 3 Experts + 1 Observer per layer) on a frozen 2.4B ternary model.

Uses a custom C/AVX2 Ghost Inference Engine for ultra-fast rollouts,
bypassing Python overhead and staying entirely within the L-cache.
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
# Configuration
# ---------------------------------------------------------------------------

MODEL_PATH = "models/bitnet-b1.58-2B-4T-bf16"
CHECKPOINT_DIR = "checkpoints"
LOG_FILE = "grpo_log.jsonl"

# GRPO hyperparameters
GROUP_SIZE = 4
TEMPERATURE = 0.963
TOP_P = 0.9
MAX_NEW_TOKENS = 256
CURRICULUM_MAX_TOKENS = 256  # match training budget — mismatch caused 51% all-correct skips (Pass 17)
BETA_KL = 0.0  # KL penalty disabled — no reference policy needed
LEARNING_RATE = 0.0174  # validated in 15-step pinned LoRA test (Pass 17)
LR_MIN = 0.001
ADAM_BETAS = (0.9, 0.999)

# Training schedule
MAX_STEPS = 700
TIME_BUDGET = 28800  # 8 hours (override with --time-budget=N)
EVAL_EVERY = 25
EVAL_SAMPLES = 20
CHECKPOINT_EVERY = 10
NUM_FEW_SHOT = 2

# Curriculum filtering
# Only train on problems the current model fails on (greedy decode, reward=0).
# This eliminates the 81% "all-same-reward → skip" waste from the previous run
# where the base model already solved ~87% of training questions.
USE_CURRICULUM_FILTER = True
CURRICULUM_SKIP_MAX = 20  # scan up to N candidates per step before giving up

# GhostWeight settings
# Set False to use real BitNet ternary weights (fast, ~138s/step).
# GhostWeight (PRNG on-the-fly generation) is correct but slow on CPU:
# each batched_ghost_matmul allocates+fills a 4MB temp buffer 79K times
# per rollout, which is impractical at training time.  Validate curriculum
# filter + scalar adapters on real weights first; tackle PRNG perf separately.
USE_GHOST = False
GHOST_SEED = 42
SCALES_PATH = "models/bitnet-b1.58-2B-4T-bf16/weight_scales.pt"

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
    for marker in ("\nQ:", "\n\nQ:", "\n Q:"):
        if marker in text:
            text = text.split(marker, 1)[0]

    def clean_number(s):
        s = s.replace(",", "").strip()
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

    answer_spans = re.findall(r"[Tt]he answer is\s*([^\n]+)", text)
    if answer_spans:
        parsed = parse_candidate(answer_spans[-1])
        if parsed is not None:
            return parsed

    match = re.search(r"####\s*(-?[\d,]+\s*/\s*-?[\d,]+)", text)
    if match:
        return clean_number(match.group(1).replace(" ", ""))
    match = re.search(r"####\s*(-?[\d,]+\.?\d*)", text)
    if match:
        return clean_number(match.group(1))

    numbers = re.findall(r"-?[\d,]+\.?\d*", text)
    if numbers:
        return clean_number(numbers[-1])
    return None


def extract_gsm8k_answer(answer_text):
    match = re.search(r"####\s*(-?[\d,]+\s*/\s*-?[\d,]+)", answer_text)
    if match:
        return match.group(1).replace(",", "").replace(" ", "").strip()
    match = re.search(r"####\s*(-?[\d,]+\.?\d*)", answer_text)
    if match:
        return match.group(1).replace(",", "").strip()
    return None


def normalize_answer(answer):
    if answer is None:
        return None
    s = answer.replace(",", "").strip()
    if not s:
        return None
    if s.endswith(".") and s.count(".") == 1:
        s = s[:-1]
    try:
        if "/" in s:
            return str(Fraction(s))
        value = Decimal(s)
        if value == value.to_integral():
            return str(int(value))
        return format(value.normalize(), "f").rstrip("0").rstrip(".")
    except:
        return s


def answers_match(predicted, ground_truth):
    pred_norm = normalize_answer(predicted)
    gt_norm = normalize_answer(ground_truth)
    return pred_norm is not None and gt_norm is not None and pred_norm == gt_norm


def strict_extract_answer(text):
    """Extract answer requiring explicit 'the answer is' or '#### N' format.
    No last-number fallback.  Used for curriculum pre-checks where a false
    positive (accidental last-number match) would incorrectly mark a problem
    as 'already solved' and exclude it from training."""
    for marker in ("\nQ:", "\n\nQ:", "\n Q:"):
        if marker in text:
            text = text.split(marker, 1)[0]

    def clean_number(s):
        s = s.replace(",", "").strip()
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

    answer_spans = re.findall(r"[Tt]he answer is\s*([^\n]+)", text)
    if answer_spans:
        parsed = parse_candidate(answer_spans[-1])
        if parsed is not None:
            return parsed

    match = re.search(r"####\s*(-?[\d,]+\s*/\s*-?[\d,]+)", text)
    if match:
        return clean_number(match.group(1).replace(" ", ""))
    match = re.search(r"####\s*(-?[\d,]+\.?\d*)", text)
    if match:
        return clean_number(match.group(1))

    return None  # no fallback to last number


# ---------------------------------------------------------------------------
# LoRA Adapter (rank-4, trainable A and B — replaces GhostChain scalars)
#
# Validated in Pass 17: flipped 2 locked questions in 15 steps that 840
# scalar adapters could not move in 225 steps.
# Architecture: ΔW = B × A, both learned.  B init=0 → identity at step 0.
# ---------------------------------------------------------------------------

LORA_RANK = 4  # rank-4 validated; rank-1 would be ~1.3M params (untested)


class LoRALayer(nn.Module):
    def __init__(self, base_layer, rank, seed):
        super().__init__()
        self.base_layer = base_layer
        out_f, in_f = base_layer.weight.shape
        gen = torch.Generator().manual_seed(seed % (2**63))
        # A: projects input down to rank — scaled normal for stable initial gradients
        A = torch.randn(rank, in_f, generator=gen, dtype=torch.bfloat16) / (in_f ** 0.5)
        self.A = nn.Parameter(A)
        # B: projects rank back up — zero init so adapter starts as identity
        self.B = nn.Parameter(torch.zeros(out_f, rank, dtype=torch.bfloat16))

    def forward(self, x):
        base_out = self.base_layer(x)
        lora_out = (x.float() @ self.A.T.float() @ self.B.T.float()).to(x.dtype)
        return base_out + lora_out


def inject_adapters(model):
    """Attach LoRALayer (rank-4, trainable A+B) to all AutoBitLinear layers."""
    adapters = []
    modules_dict = dict(model.named_modules())
    for name, module in list(model.named_modules()):
        if module.__class__.__name__ != "AutoBitLinear":
            continue
        seed = int.from_bytes(hashlib.sha256(name.encode()).digest()[:8], "big")
        lora = LoRALayer(module, rank=LORA_RANK, seed=seed)
        parts = name.rsplit(".", 1)
        if len(parts) == 2:
            parent_name, attr_name = parts
            parent = modules_dict[parent_name]
            setattr(parent, attr_name, lora)
            adapters.append((name, lora))
    for p in model.parameters():
        p.requires_grad = False
    for _, lora in adapters:
        lora.A.requires_grad = True
        lora.B.requires_grad = True
    total = sum(p.numel() for _, l in adapters for p in [l.A, l.B])
    print(f"  LoRA rank-{LORA_RANK}: {len(adapters)} layers, {total:,} trainable params")
    return adapters


def get_adapter_norms(adapters):
    """Return dict of per-adapter Frobenius norm of B (proxy for adaptation magnitude)."""
    return {name: lora.B.data.norm().item() for name, lora in adapters}


def zero_adapter_scales(adapters):
    """Zero B matrices for π_ref computation (used when BETA_KL > 0)."""
    saved = []
    for _, lora in adapters:
        saved.append(lora.B.data.clone())
        lora.B.data.zero_()
    return saved


def restore_adapter_scales(adapters, saved):
    for (_, lora), val in zip(adapters, saved):
        lora.B.data.copy_(val)


# ---------------------------------------------------------------------------
# Compute log-probs for policy loss
# ---------------------------------------------------------------------------


def compute_log_probs(model, full_ids, prompt_len):
    outputs = model(full_ids)
    logits = outputs.logits[:, prompt_len - 1 : -1, :].float()
    log_probs_all = torch.log_softmax(logits, dim=-1)
    target_tokens = full_ids[:, prompt_len:]
    return log_probs_all.gather(-1, target_tokens.unsqueeze(-1)).squeeze(-1).squeeze(0)


def compute_log_probs_batch(model, full_ids, prompt_len):
    """Batched version: full_ids [G, total_len] → [G, comp_len]."""
    outputs = model(full_ids)
    logits = outputs.logits[:, prompt_len - 1 : -1, :].float()
    log_probs_all = torch.log_softmax(logits, dim=-1)
    target_tokens = full_ids[:, prompt_len:]
    return log_probs_all.gather(-1, target_tokens.unsqueeze(-1)).squeeze(-1)


def compute_log_probs_batch_kv(model, full_ids, prompt_len):
    """KV-detach: prompt under no_grad, backward only through completion tokens.

    The prompt is fixed — the scale parameters affect it, but only at O(scale²)
    through the KV cache path (second-order, negligible at scale ≈ 0.02).
    The dominant first-order gradient flows through completion token processing.

    Two separate forward passes:
      1. Prompt (no_grad)  → KV cache for all 30 layers
      2. Completion (grad) → attend over cached prompt KV as constants

    Backward cost ∝ comp_len instead of (prompt_len + comp_len), reducing it by
    roughly prompt_len / (prompt_len + comp_len) ≈ 57% at current token lengths.

    Returns [G, comp_len - 1] — one fewer token than compute_log_probs_batch
    (the first completion token's log-prob is skipped since its logit comes from
    the no_grad prompt forward; 1 of 96 tokens ≈ 1% bias, acceptable).
    """
    # Pass 1: prompt under no_grad — cache K/V for all layers
    with torch.no_grad():
        prompt_out = model(full_ids[:, :prompt_len], use_cache=True)
        past_kv = prompt_out.past_key_values

    # Pass 2: completion with grad — attend over cached prompt K/V as constants
    # Input: all completion tokens. With past_kv, logit[j] predicts token[prompt_len+j+1].
    # We use logits[:, :-1, :] to predict tokens prompt_len+1 .. prompt_len+comp_len-1.
    comp_ids = full_ids[:, prompt_len:]                  # [G, comp_len]
    out = model(comp_ids, past_key_values=past_kv)
    logits = out.logits[:, :-1, :].float()               # [G, comp_len-1, vocab]
    log_probs = torch.log_softmax(logits, dim=-1)
    targets = full_ids[:, prompt_len + 1:]               # [G, comp_len-1]
    return log_probs.gather(-1, targets.unsqueeze(-1)).squeeze(-1)


# ---------------------------------------------------------------------------
# Rollout dispatch
# ---------------------------------------------------------------------------


def python_rollout(model, tokenizer, prompt_ids, group_size, max_new_tokens, temp):
    """Generate completions using the Python model (torch.no_grad).

    Used when USE_GHOST=False so rollouts use the real stored ternary weights
    via the soft-chip AVX2 kernels, not the PRNG-per-token C ghost engine.
    The C ghost engine regenerates weight matrices from scratch for every token
    position, which is ~200x slower than reading cached weights from memory.

    Returns a (group_size, prompt_len + max_new_tokens) int tensor, zero-padded
    to a fixed length so the shape matches the C engine's output contract.
    """
    prompt_len = prompt_ids.shape[1]
    total_len = prompt_len + max_new_tokens
    # Generate all group_size completions in a single batched call instead of
    # G sequential calls.  Repeating the prompt gives G independent samples
    # with temp>0 sampling.  ~3x faster on CPU due to better kernel utilization.
    batched_prompt = prompt_ids.repeat(group_size, 1)
    with torch.no_grad():
        out = model.generate(
            batched_prompt,
            max_new_tokens=max_new_tokens,
            do_sample=(temp > 0.0),
            temperature=(temp if temp > 0.0 else None),
            top_p=(TOP_P if temp > 0.0 else None),
            pad_token_id=tokenizer.eos_token_id,
        )
    if out.shape[1] < total_len:
        pad = torch.zeros(group_size, total_len - out.shape[1], dtype=out.dtype)
        out = torch.cat([out, pad], dim=1)
    return out[:, :total_len]


def rollout(model, tokenizer, adapters, engine, prompt_ids, group_size, max_new_tokens, temp):
    """Dispatch to C ghost engine or Python model based on USE_GHOST."""
    if USE_GHOST:
        return engine.rollout_batch(adapters, prompt_ids, group_size, max_new_tokens, temp)
    return python_rollout(model, tokenizer, prompt_ids, group_size, max_new_tokens, temp)


# ---------------------------------------------------------------------------
# GRPO Logic
# ---------------------------------------------------------------------------


def grpo_step(
    model, tokenizer, adapters, optimizer, question, ground_truth, step_num, engine
):
    t_start = time.time()
    prompt_text = format_prompt(question)
    prompt_ids = tokenizer(prompt_text, return_tensors="pt").input_ids
    prompt_len = prompt_ids.shape[1]

    # 1. ROLLOUT
    t_rollout = time.time()
    batch_ids = rollout(model, tokenizer, adapters, engine, prompt_ids, GROUP_SIZE, MAX_NEW_TOKENS, TEMPERATURE)

    completions = []
    completion_texts = []
    for g in range(GROUP_SIZE):
        comp_ids = batch_ids[g : g + 1]
        completions.append(comp_ids)
        comp_text = tokenizer.decode(comp_ids[0, prompt_len:], skip_special_tokens=True)
        completion_texts.append(comp_text)
    rollout_time = time.time() - t_rollout

    # 2. REWARD
    rewards = []
    preds = []
    for txt in completion_texts:
        p = extract_answer(txt)
        preds.append(p)
        rewards.append(1.0 if answers_match(p, ground_truth) else 0.0)

    # 3. ADVANTAGE
    mean_r = sum(rewards) / GROUP_SIZE
    std_r = (sum((r - mean_r) ** 2 for r in rewards) / GROUP_SIZE) ** 0.5
    if std_r < 1e-8:
        return {
            "step": step_num,
            "rewards": rewards,
            "skipped": True,
            "rollout_time": rollout_time,
            "total_time": time.time() - t_start,
            "predicted_answers": preds,
            "ground_truth": ground_truth,
        }

    advantages = [(r - mean_r) / (std_r + 1e-8) for r in rewards]

    # 4. LOSS — single batched π_θ forward pass over all G completions.
    # π_ref pass is skipped entirely when BETA_KL=0 (saves ~50% of update time).
    t_update = time.time()
    valid_pairs = [(ids, adv) for ids, adv in zip(completions, advantages)
                   if ids.shape[1] > prompt_len]
    total_loss = torch.tensor(0.0)
    if valid_pairs:
        stacked_ids = torch.cat([ids for ids, _ in valid_pairs], dim=0)
        valid_advs = [adv for _, adv in valid_pairs]

        # Single π_θ forward + backward — KV-detach: prompt under no_grad so
        # the backward only traverses the completion tokens (~57% less backprop).
        log_probs_theta_all = compute_log_probs_batch_kv(model, stacked_ids, prompt_len)

        # π_ref only when KL penalty is active
        if BETA_KL > 0:
            saved = zero_adapter_scales(adapters)
            with torch.no_grad():
                log_probs_ref_all = compute_log_probs_batch_kv(model, stacked_ids, prompt_len)
            restore_adapter_scales(adapters, saved)

        for i, adv in enumerate(valid_advs):
            lp_theta = log_probs_theta_all[i]
            # Normalize by sequence length so gradient magnitude is independent of
            # how many tokens the model generated.  Without this, a 128-token
            # completion gets 128x the gradient of a 1-token completion, biasing
            # optimization toward short-answer format learning over reasoning.
            reinforce_loss = -adv * lp_theta.mean()
            if BETA_KL > 0:
                kl = (lp_theta - log_probs_ref_all[i]).mean()
                total_loss = total_loss + (reinforce_loss + BETA_KL * kl)
            else:
                total_loss = total_loss + reinforce_loss

    total_loss = total_loss / GROUP_SIZE
    optimizer.zero_grad()
    total_loss.backward()
    optimizer.step()

    grad_norms = []
    for _, lora in adapters:
        if lora.B.grad is not None:
            grad_norms.append(lora.B.grad.norm().item())
        if lora.A.grad is not None:
            grad_norms.append(lora.A.grad.norm().item())
    mean_b_norm = sum(lora.B.data.norm().item() for _, lora in adapters) / len(adapters)
    return {
        "step": step_num,
        "rewards": rewards,
        "loss": total_loss.item(),
        "mean_grad": sum(grad_norms) / len(grad_norms) if grad_norms else 0.0,
        "mean_abs_scale": mean_b_norm,  # repurposed: Frobenius norm of B (starts at 0)
        "skipped": False,
        "rollout_time": rollout_time,
        "update_time": time.time() - t_update,
        "total_time": time.time() - t_start,
        "predicted_answers": preds,
        "ground_truth": ground_truth,
    }


def evaluate(
    model, tokenizer, test_data, engine, num_samples=EVAL_SAMPLES, adapters=None
):
    correct = 0
    total = 0
    results = []
    for i, ex in enumerate(test_data[:num_samples]):
        prompt_ids = tokenizer(
            format_prompt(ex["question"]), return_tensors="pt"
        ).input_ids
        # Use rollout() dispatch so eval uses real trained weights (python_rollout)
        # when USE_GHOST=False, not the PRNG-based C ghost engine.
        comp_ids = rollout(model, tokenizer, adapters, engine, prompt_ids, 1, MAX_NEW_TOKENS, temp=0.0)
        pred = extract_answer(
            tokenizer.decode(
                comp_ids[0, prompt_ids.shape[1] :], skip_special_tokens=True
            )
        )
        gt = extract_gsm8k_answer(ex["answer"])
        match = answers_match(pred, gt)
        if match:
            correct += 1
        total += 1
        print(
            f"  eval [{i + 1}/{num_samples}] pred={pred} gt={gt} {'OK' if match else 'WRONG'}"
        )
    return correct / total if total > 0 else 0.0


# ---------------------------------------------------------------------------
# Curriculum Filtering
# ---------------------------------------------------------------------------


def find_hard_problem(model, engine, adapters, tokenizer, train_data, start_idx):
    """
    Two-step curriculum filter: find a problem in the borderline zone where
    the model fails greedily but can sometimes succeed with sampling.

    Step 1 — greedy check (temp=0): must fail (not too easy).
    Step 2 — stochastic check (temp=TEMPERATURE, n=1): prefer to succeed
              (confirms the problem is learnable, not just impossible).

    If a borderline problem is found (step 1 fail + step 2 succeed), return it
    immediately — this nearly guarantees mixed rewards in the main GROUP_SIZE
    rollout, eliminating the dominant source of wasted compute (83% skip rate).

    Fallback: if no stochastic success is found within CURRICULUM_SKIP_MAX
    candidates, return the first greedy-fail found.  This preserves current
    behavior when USE_GHOST=True (near-zero base accuracy makes stochastic
    success rare) or when the model is in a temporary hard patch.

    Returns (problem, new_idx, n_scanned, found_hard).
    """
    first_hard_ex = None
    first_hard_idx = None

    for n in range(CURRICULUM_SKIP_MAX):
        ex = train_data[start_idx % len(train_data)]
        start_idx += 1
        gt = extract_gsm8k_answer(ex["answer"])
        if gt is None:
            continue
        prompt_ids = tokenizer(format_prompt(ex["question"]), return_tensors="pt").input_ids

        # Step 1: greedy check — filter out questions the model already solves
        # Uses CURRICULUM_MAX_TOKENS (shorter than MAX_NEW_TOKENS) to keep scans fast.
        comp_ids = rollout(model, tokenizer, adapters, engine, prompt_ids, 1, CURRICULUM_MAX_TOKENS, temp=0.0)
        pred = strict_extract_answer(
            tokenizer.decode(comp_ids[0, prompt_ids.shape[1] :], skip_special_tokens=True)
        )
        if answers_match(pred, gt):
            continue  # too easy — skip

        # Save first greedy-fail as fallback
        if first_hard_ex is None:
            first_hard_ex = ex
            first_hard_idx = start_idx

        # Step 2: stochastic check — prefer borderline problems the model can
        # sometimes solve (guarantees mixed rewards in main rollout)
        comp_ids_s = rollout(model, tokenizer, adapters, engine, prompt_ids, 1, CURRICULUM_MAX_TOKENS, temp=TEMPERATURE)
        pred_s = extract_answer(
            tokenizer.decode(comp_ids_s[0, prompt_ids.shape[1] :], skip_special_tokens=True)
        )
        if answers_match(pred_s, gt):
            return ex, start_idx, n + 1, True  # borderline found

    # Fallback: return first greedy-fail if stochastic check never confirmed
    if first_hard_ex is not None:
        return first_hard_ex, first_hard_idx, CURRICULUM_SKIP_MAX, True
    return None, start_idx, CURRICULUM_SKIP_MAX, False  # exhausted


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    preflight_only = "--preflight" in sys.argv
    resume_path = None
    time_budget = TIME_BUDGET
    for arg in sys.argv[1:]:
        if arg.startswith("--resume="):
            resume_path = arg.split("=", 1)[1]
        elif arg.startswith("--time-budget="):
            time_budget = int(arg.split("=", 1)[1])

    print("=" * 60)
    print("  GRPO Training: GhostChain (3 Experts + 1 Observer)")
    print("=" * 60)

    random.seed(SEED)
    torch.manual_seed(SEED)

    from transformers import AutoModelForCausalLM, AutoTokenizer

    print("  Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    print("  Loading model...")
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH, torch_dtype=torch.bfloat16, device_map="cpu", trust_remote_code=True
    )
    print("  Model loaded.")

    print("  Injecting adapters...")

    adapters = inject_adapters(model)
    print(f"  Injected {len(adapters)} GhostChain experts.")

    # Ghost engine is only needed when USE_GHOST=True (PRNG weight path).
    # With USE_GHOST=False (real ternary weights), rollout() dispatches to
    # python_rollout() and engine is never called.
    engine = None
    if USE_GHOST:
        from softchip.ghost_engine_wrapper import GhostEngine
        print("  Initializing Ghost Engine...")
        engine = GhostEngine(model, tokenizer, GHOST_SEED, SCALES_PATH)
        print("  Ghost Engine initialized.")

    from softchip import patch_model, patch_lm_head_fp32

    print("  Patching model...")
    patch_model(model, backend="cpu", use_ghost=USE_GHOST, ghost_seed=GHOST_SEED, scales_path=SCALES_PATH)
    print("  Model patched. Patching LM head...")
    patch_lm_head_fp32(model)
    print("  LM head patched.")

    from datasets import load_dataset

    print("  Loading dataset...")
    gsm8k = load_dataset("gsm8k", "main")
    print("  Dataset loaded.")

    train_data, test_data = list(gsm8k["train"]), list(gsm8k["test"])
    # Use an isolated RNG seeded by SEED so the shuffle order is identical on
    # every cold start AND every resume.  The global random state is intentionally
    # not used here — touching it would make the order depend on whatever the
    # model-loading code consumed from the random stream beforehand.
    random.Random(SEED).shuffle(train_data)

    optimizer = torch.optim.Adam(
        [p for _, lora in adapters for p in [lora.A, lora.B]],
        lr=LEARNING_RATE, betas=ADAM_BETAS
    )

    if preflight_only:
        print("\n[PREFLIGHT] Testing step...")
        res = grpo_step(
            model,
            tokenizer,
            adapters,
            optimizer,
            train_data[0]["question"],
            extract_gsm8k_answer(train_data[0]["answer"]),
            0,
            engine,
        )
        print(
            f"  Step time: {res['total_time']:.1f}s (Rollout: {res['rollout_time']:.1f}s)"
        )
        return

    eval_history = []
    problem_ptr = 0
    start_step = 1
    if resume_path:
        ckpt = torch.load(resume_path, weights_only=False)
        for name, lora in adapters:
            if name in ckpt.get("adapter_scales", {}):
                d = ckpt["adapter_scales"][name]
                if isinstance(d, dict):
                    lora.A.data.copy_(d["A"])
                    lora.B.data.copy_(d["B"])
        optimizer.load_state_dict(ckpt["optimizer_state"])
        start_step = ckpt["step"] + 1
        eval_history = ckpt.get("eval_history", [])
        problem_ptr = ckpt.get("problem_ptr", start_step - 1)

    curriculum_tag = " +curriculum" if USE_CURRICULUM_FILTER else ""
    print(f"\nTraining Start: G={GROUP_SIZE}, Budget={time_budget}s{curriculum_tag}")
    t0 = time.time()
    total_time = 0

    for step in range(start_step, MAX_STEPS + 1):
        if total_time >= time_budget:
            break

        # Linear decay (simplified for loop)
        lr = LR_MIN + (LEARNING_RATE - LR_MIN) * (1 - step / MAX_STEPS)
        for pg in optimizer.param_groups:
            pg["lr"] = lr

        if USE_CURRICULUM_FILTER:
            ex, problem_ptr, n_scanned, found_hard = find_hard_problem(
                model, engine, adapters, tokenizer, train_data, problem_ptr
            )
            if not found_hard:
                # All CURRICULUM_SKIP_MAX candidates were already solved by the
                # current model.  Training on any of them would reproduce the
                # original 81%-skip problem.  Skip this step number entirely and
                # advance — the model has progressed into a harder regime and will
                # find hard problems again shortly.
                print(
                    f"[step {step:04d}] CURRICULUM_EXHAUSTED: all {n_scanned} candidates solved — skipping step"
                )
                continue
        else:
            ex = train_data[step % len(train_data)]
            problem_ptr = step
            n_scanned = 0

        res = grpo_step(
            model,
            tokenizer,
            adapters,
            optimizer,
            ex["question"],
            extract_gsm8k_answer(ex["answer"]),
            step,
            engine,
        )

        total_time += res["total_time"]
        scan_tag = f" scan={n_scanned}" if USE_CURRICULUM_FILTER else ""
        print(
            f"[step {step:04d}] rewards={res['rewards']} loss={res.get('loss', 0.0):.4f} |scale|={res.get('mean_abs_scale', 0.0):.4f} time={res['total_time']:.1f}s{scan_tag}"
        )

        if step % CHECKPOINT_EVERY == 0:
            path = f"{CHECKPOINT_DIR}/grpo_step_{step:04d}.pt"
            torch.save(
                {
                    "step": step,
                    "adapter_scales": {n: {"A": l.A.data, "B": l.B.data} for n, l in adapters},
                    "optimizer_state": optimizer.state_dict(),
                    "eval_history": eval_history,
                    "problem_ptr": problem_ptr,
                },
                path,
            )

        if step % EVAL_EVERY == 0:
            acc = evaluate(model, tokenizer, test_data, engine, adapters=adapters)
            print(f"  Accuracy at step {step}: {acc * 100:.1f}%")
            eval_history.append({"step": step, "accuracy": acc})


if __name__ == "__main__":
    main()
