"""
LoRA Validation Test — two phases, ~45 minutes total.

Phase 1 (~15 min): Stochastic ceiling check.
  Run 10 samples on each of the 10 locked eval questions.
  If the model NEVER produces the correct answer stochastically,
  no GRPO training (with any adapter) can help — reward is always 0.

Phase 2 (~30 min): Pinned LoRA overfit test.
  Inject LoRA rank-4 (both A and B learned, ~1.1M params vs 840 scalars).
  Pin the most promising locked questions as the ONLY training set.
  Run 15 GRPO steps. Check if any locked question flips.

Success criterion: ANY locked question flips WRONG→OK in Phase 2.
This validates LoRA has enough capacity for a full run.
Failure: if nothing flips after 15 pinned steps, we have a deeper problem.

Usage:
  python lora_validation.py                    # both phases
  python lora_validation.py --phase1-only      # stochastic ceiling only
  python lora_validation.py --phase2-only      # skip to LoRA test
  python lora_validation.py --rank=1           # use LoRA rank-1 (faster)
"""

import re
import sys
import time
import random
import hashlib
from decimal import Decimal
from fractions import Fraction
from pathlib import Path

import torch
import torch.nn as nn

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

MODEL_PATH = "models/bitnet-b1.58-2B-4T-bf16"
LORA_RANK = 4          # default; override with --rank=N
GROUP_SIZE = 4
TEMPERATURE = 0.963
TOP_P = 0.9
MAX_NEW_TOKENS = 128   # shorter than full run — faster steps
PHASE2_STEPS = 15
PHASE1_SAMPLES = 10    # stochastic samples per locked question
NUM_FEW_SHOT = 2
SEED = 42

# The 10 questions that have been wrong in every eval across 225 training steps.
# Index is 1-based position in first 20 GSM8K test questions.
LOCKED_INDICES = {3, 5, 6, 8, 9, 13, 15, 16, 18, 20}

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


def format_prompt(question):
    parts = ["Solve the following math problems step by step.\n"]
    for ex in FEW_SHOT_EXAMPLES[:NUM_FEW_SHOT]:
        parts.append(f"Q: {ex['question']}")
        parts.append(f"A: {ex['answer']}\n")
    parts.append(f"Q: {question}")
    parts.append("A:")
    return "\n".join(parts)


def extract_answer(text):
    for marker in ("\nQ:", "\n\nQ:", "\n Q:"):
        if marker in text:
            text = text.split(marker, 1)[0]

    def clean(s):
        s = s.replace(",", "").strip()
        if s.endswith(".") and s.count(".") == 1:
            s = s[:-1]
        return s

    def parse(s):
        s = s.strip()
        frac = re.search(r"(-?[\d,]+\s*/\s*-?[\d,]+)", s)
        if frac:
            return clean(frac.group(1).replace(" ", ""))
        num = re.search(r"(-?[\d,]+\.?\d*)", s)
        if num:
            return clean(num.group(1))
        return None

    spans = re.findall(r"[Tt]he answer is\s*([^\n]+)", text)
    if spans:
        p = parse(spans[-1])
        if p is not None:
            return p
    m = re.search(r"####\s*(-?[\d,]+\.?\d*)", text)
    if m:
        return clean(m.group(1))
    nums = re.findall(r"-?[\d,]+\.?\d*", text)
    if nums:
        return clean(nums[-1])
    return None


def extract_gsm8k_answer(text):
    m = re.search(r"####\s*(-?[\d,]+\.?\d*)", text)
    return m.group(1).replace(",", "").strip() if m else None


def normalize_answer(s):
    if s is None:
        return None
    s = s.replace(",", "").strip()
    if not s:
        return None
    if s.endswith(".") and s.count(".") == 1:
        s = s[:-1]
    try:
        if "/" in s:
            return str(Fraction(s))
        v = Decimal(s)
        if v == v.to_integral():
            return str(int(v))
        return format(v.normalize(), "f").rstrip("0").rstrip(".")
    except Exception:
        return s


def answers_match(pred, gt):
    return (normalize_answer(pred) is not None
            and normalize_answer(gt) is not None
            and normalize_answer(pred) == normalize_answer(gt))


# ---------------------------------------------------------------------------
# LoRA adapter (rank-N, both A and B learned)
# ---------------------------------------------------------------------------

class LoRALayer(nn.Module):
    """Wraps a frozen base linear layer with a rank-r LoRA adapter.

    ΔW = B × A  (out×rank) × (rank×in)
    B initialized to zero → adapter starts at identity output (same as base).
    A initialized to normal — provides diverse starting directions.
    """
    def __init__(self, base_layer, rank, seed):
        super().__init__()
        self.base_layer = base_layer
        out_f, in_f = base_layer.weight.shape
        gen = torch.Generator().manual_seed(seed % (2**63))
        # A: projects input down to rank — initialized with scaled normal
        A = torch.randn(rank, in_f, generator=gen, dtype=torch.bfloat16) / (in_f ** 0.5)
        self.A = nn.Parameter(A)
        # B: projects rank back up — initialized to zero
        self.B = nn.Parameter(torch.zeros(out_f, rank, dtype=torch.bfloat16))

    def forward(self, x):
        base_out = self.base_layer(x)
        # LoRA delta: x @ A.T @ B.T  (works for any batch shape)
        lora_out = (x.float() @ self.A.T.float() @ self.B.T.float()).to(x.dtype)
        return base_out + lora_out


def inject_lora(model, rank):
    """Replace all AutoBitLinear layers with LoRALayer wrappers."""
    adapters = []
    modules_dict = dict(model.named_modules())
    for name, module in list(model.named_modules()):
        if module.__class__.__name__ != "AutoBitLinear":
            continue
        seed = int.from_bytes(hashlib.sha256(name.encode()).digest()[:8], "big")
        lora = LoRALayer(module, rank=rank, seed=seed)
        parts = name.rsplit(".", 1)
        if len(parts) == 2:
            parent = modules_dict[parts[0]]
            setattr(parent, parts[1], lora)
            adapters.append((name, lora))
    # Freeze everything, then unfreeze only LoRA A and B
    for p in model.parameters():
        p.requires_grad = False
    for _, lora in adapters:
        lora.A.requires_grad = True
        lora.B.requires_grad = True
    total_params = sum(p.numel() for _, l in adapters for p in [l.A, l.B])
    return adapters, total_params


# ---------------------------------------------------------------------------
# Generation
# ---------------------------------------------------------------------------

def generate(model, tokenizer, prompt_ids, n, max_new_tokens, temp):
    batched = prompt_ids.repeat(n, 1)
    with torch.no_grad():
        out = model.generate(
            batched,
            max_new_tokens=max_new_tokens,
            do_sample=(temp > 0),
            temperature=(temp if temp > 0 else None),
            top_p=(TOP_P if temp > 0 else None),
            pad_token_id=tokenizer.eos_token_id,
        )
    return out


def decode_completions(tokenizer, batch_ids, prompt_len):
    return [
        tokenizer.decode(batch_ids[g, prompt_len:], skip_special_tokens=True)
        for g in range(batch_ids.shape[0])
    ]


# ---------------------------------------------------------------------------
# GRPO step
# ---------------------------------------------------------------------------

def compute_log_probs(model, full_ids, prompt_len):
    """KV-detach: prompt under no_grad, backprop only through completion."""
    with torch.no_grad():
        prompt_out = model(full_ids[:, :prompt_len], use_cache=True)
        past_kv = prompt_out.past_key_values
    comp_ids = full_ids[:, prompt_len:]
    out = model(comp_ids, past_key_values=past_kv)
    logits = out.logits[:, :-1, :].float()
    log_probs = torch.log_softmax(logits, dim=-1)
    targets = full_ids[:, prompt_len + 1:]
    return log_probs.gather(-1, targets.unsqueeze(-1)).squeeze(-1)


def grpo_step(model, tokenizer, optimizer, question, gt, step_num):
    t0 = time.time()
    prompt_text = format_prompt(question)
    prompt_ids = tokenizer(prompt_text, return_tensors="pt").input_ids
    prompt_len = prompt_ids.shape[1]

    # Rollout
    batch_ids = generate(model, tokenizer, prompt_ids, GROUP_SIZE, MAX_NEW_TOKENS, TEMPERATURE)
    texts = decode_completions(tokenizer, batch_ids, prompt_len)

    # Reward
    rewards = [1.0 if answers_match(extract_answer(t), gt) else 0.0 for t in texts]
    preds = [extract_answer(t) for t in texts]

    mean_r = sum(rewards) / GROUP_SIZE
    std_r = (sum((r - mean_r) ** 2 for r in rewards) / GROUP_SIZE) ** 0.5

    if std_r < 1e-8:
        print(f"  [step {step_num:02d}] rewards={rewards} SKIP (uniform) — {time.time()-t0:.0f}s")
        return rewards, preds, False

    advantages = [(r - mean_r) / (std_r + 1e-8) for r in rewards]

    # Loss
    total_loss = torch.tensor(0.0)
    valid = [(batch_ids[g:g+1], advantages[g]) for g in range(GROUP_SIZE)
             if batch_ids.shape[1] > prompt_len]
    if valid:
        stacked = torch.cat([ids for ids, _ in valid], dim=0)
        lp = compute_log_probs(model, stacked, prompt_len)
        for i, (_, adv) in enumerate(valid):
            total_loss = total_loss + (-adv * lp[i].mean())
    total_loss = total_loss / GROUP_SIZE

    optimizer.zero_grad()
    total_loss.backward()
    optimizer.step()

    elapsed = time.time() - t0
    print(f"  [step {step_num:02d}] rewards={rewards} loss={total_loss.item():.4f} "
          f"preds={preds} gt={gt} — {elapsed:.0f}s")
    return rewards, preds, True


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    phase1_only = "--phase1-only" in sys.argv
    phase2_only = "--phase2-only" in sys.argv
    rank = LORA_RANK
    lr = 0.001
    for arg in sys.argv[1:]:
        if arg.startswith("--rank="):
            rank = int(arg.split("=", 1)[1])
        elif arg.startswith("--lr="):
            lr = float(arg.split("=", 1)[1])

    random.seed(SEED)
    torch.manual_seed(SEED)

    from transformers import AutoModelForCausalLM, AutoTokenizer
    from datasets import load_dataset
    from softchip import patch_model, patch_lm_head_fp32

    print("Loading model...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH, torch_dtype=torch.bfloat16, device_map="cpu", trust_remote_code=True
    )
    patch_model(model, backend="cpu", use_ghost=False)
    patch_lm_head_fp32(model)
    print("  Model ready.")

    print("Loading dataset...")
    gsm8k = load_dataset("gsm8k", "main")
    test_data = list(gsm8k["test"])
    locked_questions = [
        {"idx": i, "question": test_data[i-1]["question"],
         "gt": extract_gsm8k_answer(test_data[i-1]["answer"])}
        for i in sorted(LOCKED_INDICES)
    ]
    print(f"  {len(locked_questions)} locked questions identified.\n")

    # -----------------------------------------------------------------------
    # Phase 1: Stochastic ceiling check (no adapter changes)
    # -----------------------------------------------------------------------
    phase1_results = {}
    if not phase2_only:
        print("=" * 60)
        print("PHASE 1: Stochastic ceiling check (base model, no adapters)")
        print(f"  {PHASE1_SAMPLES} samples per question @ {MAX_NEW_TOKENS} tokens, temp={TEMPERATURE}")
        print("=" * 60)
        t_phase1 = time.time()

        for q in locked_questions:
            prompt_ids = tokenizer(format_prompt(q["question"]), return_tensors="pt").input_ids
            prompt_len = prompt_ids.shape[1]
            hits = 0
            preds_seen = []
            for s in range(PHASE1_SAMPLES):
                out = generate(model, tokenizer, prompt_ids, 1, MAX_NEW_TOKENS, TEMPERATURE)
                text = tokenizer.decode(out[0, prompt_len:], skip_special_tokens=True)
                pred = extract_answer(text)
                preds_seen.append(pred)
                if answers_match(pred, q["gt"]):
                    hits += 1

            phase1_results[q["idx"]] = hits
            status = "SOLVABLE" if hits > 0 else "UNREACHABLE"
            unique_preds = list(dict.fromkeys(p for p in preds_seen if p))[:4]
            print(f"  Q{q['idx']:2d} gt={q['gt']:>7}  hits={hits}/{PHASE1_SAMPLES}  "
                  f"{status}  preds={unique_preds}")

        elapsed = time.time() - t_phase1
        solvable = [q for q in locked_questions if phase1_results[q["idx"]] > 0]
        print(f"\n  Phase 1 complete in {elapsed/60:.1f} min.")
        print(f"  Solvable (≥1 hit): {len(solvable)}/10 locked questions")
        if not solvable:
            print("\n  RESULT: Model NEVER produces correct answers for locked questions.")
            print("  No GRPO training can help — reward is always 0.")
            print("  This is a model capacity ceiling, not an adapter problem.")
            return
        print(f"  Candidates for Phase 2: {[q['idx'] for q in solvable]}\n")
    else:
        # phase2-only: assume all are solvable, pick best candidates manually
        solvable = [q for q in locked_questions if q["idx"] in {6, 9, 20}]

    if phase1_only:
        return

    # -----------------------------------------------------------------------
    # Phase 2: Pinned LoRA overfit test
    # -----------------------------------------------------------------------
    print("=" * 60)
    print(f"PHASE 2: Pinned LoRA rank-{rank} overfit test")
    print(f"  {PHASE2_STEPS} steps pinned to solvable locked questions")
    print(f"  MAX_NEW_TOKENS={MAX_NEW_TOKENS}, GROUP_SIZE={GROUP_SIZE}")
    print("=" * 60)

    # Inject LoRA
    adapters, n_params = inject_lora(model, rank=rank)
    print(f"  Injected LoRA rank-{rank} into {len(adapters)} layers: {n_params:,} trainable params")
    print(f"  Learning rate: {lr}\n")

    optimizer = torch.optim.Adam(
        [p for _, lora in adapters for p in [lora.A, lora.B]],
        lr=lr, betas=(0.9, 0.999)
    )

    # Cycle through solvable locked questions
    pin_questions = solvable[:3]  # at most 3

    print("Training questions (pinned):")
    for q in pin_questions:
        print(f"  Q{q['idx']:2d}  gt={q['gt']}  {q['question'][:70]}...")
    print()

    flip_log = []
    for step in range(1, PHASE2_STEPS + 1):
        q = pin_questions[(step - 1) % len(pin_questions)]
        rewards, preds, did_grad = grpo_step(
            model, tokenizer, optimizer, q["question"], q["gt"], step
        )
        for p, r in zip(preds, rewards):
            if r == 1.0:
                flip_log.append({"step": step, "idx": q["idx"], "pred": p, "gt": q["gt"]})

    # Final eval on all 10 locked questions
    print(f"\n{'='*60}")
    print("FINAL EVAL: All 10 locked questions (greedy, temp=0)")
    print("="*60)
    flipped = []
    for q in locked_questions:
        prompt_ids = tokenizer(format_prompt(q["question"]), return_tensors="pt").input_ids
        prompt_len = prompt_ids.shape[1]
        out = generate(model, tokenizer, prompt_ids, 1, MAX_NEW_TOKENS, temp=0.0)
        text = tokenizer.decode(out[0, prompt_len:], skip_special_tokens=True)
        pred = extract_answer(text)
        ok = answers_match(pred, q["gt"])
        tag = "  *** FLIPPED" if ok else ""
        print(f"  Q{q['idx']:2d} gt={q['gt']:>7}  pred={str(pred):>7}  {'OK' if ok else 'WRONG'}{tag}")
        if ok:
            flipped.append(q["idx"])

    print(f"\n{'='*60}")
    if flipped:
        print(f"RESULT: PASS — {len(flipped)} question(s) flipped: {flipped}")
        print("LoRA has sufficient capacity. Full run is warranted.")
    else:
        print("RESULT: FAIL — no locked questions flipped after pinned training.")
        print("Possible causes: learning rate too low, rank too small, or")
        print("questions require reasoning the base model cannot produce.")
    print(f"Correct answers during training rollouts: {len(flip_log)} hits")
    for h in flip_log:
        print(f"  Step {h['step']}: Q{h['idx']} pred={h['pred']} gt={h['gt']}")


if __name__ == "__main__":
    main()
