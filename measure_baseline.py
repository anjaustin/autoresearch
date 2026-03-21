"""
Baseline accuracy measurement for BitNet b1.58 2B4T on GSM8K.

Measures GSM8K test-set accuracy under three conditions:
  1. Zero adapters (pure base model, no GhostChain)
  2. Trained adapters from a checkpoint (what training has achieved)
  3. Random adapters at a specified scale (noise floor sanity check)

Usage:
    # Just the base model (most important number — do this first):
    python measure_baseline.py

    # Compare base model vs a trained checkpoint:
    python measure_baseline.py --checkpoint checkpoints/grpo_step_0120.pt

    # All three conditions:
    python measure_baseline.py --checkpoint checkpoints/grpo_step_0120.pt --random-scale 0.03

    # More eval samples for tighter confidence intervals:
    python measure_baseline.py --checkpoint checkpoints/grpo_step_0120.pt --n-samples 50

Why this matters:
    The base model already solves ~87% of *training* questions (per curriculum
    filter data), but that figure is: (a) training set not test set, (b) greedy
    decode not the few-shot eval format used here, (c) strict_extract_answer
    not extract_answer.  This script measures the number that actually matters
    for interpreting the training curve.
"""

import argparse
import sys
import time
from pathlib import Path

import torch

# ---------------------------------------------------------------------------
# Config (must match grpo_train.py exactly for apples-to-apples comparison)
# ---------------------------------------------------------------------------
MODEL_PATH = "models/bitnet-b1.58-2B-4T-bf16"
SCALES_PATH = "models/bitnet-b1.58-2B-4T-bf16/weight_scales.pt"
MAX_NEW_TOKENS = 96
NUM_FEW_SHOT = 2
SEED = 42


def main():
    parser = argparse.ArgumentParser(description="Measure GSM8K baseline accuracy")
    parser.add_argument(
        "--checkpoint", type=str, default=None,
        help="Path to a grpo_train.py checkpoint .pt file. "
             "If given, also evaluates the trained adapter scales.",
    )
    parser.add_argument(
        "--n-samples", type=int, default=20,
        help="Number of GSM8K test examples to evaluate (default: 20)",
    )
    parser.add_argument(
        "--random-scale", type=float, default=None,
        help="If given, also evaluate adapters initialised to this scale magnitude "
             "(sanity check: random directions at trained scale should ≈ base model).",
    )
    parser.add_argument(
        "--test-split", action="store_true",
        help="Use GSM8K test split (default). Pass to be explicit.",
    )
    args = parser.parse_args()
    n_samples = args.n_samples

    torch.manual_seed(SEED)

    # ------------------------------------------------------------------
    # Imports (inline to keep startup fast if model isn't available)
    # ------------------------------------------------------------------
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from datasets import load_dataset

    # Reuse the exact same functions from grpo_train.py so behaviour is identical
    sys.path.insert(0, str(Path(__file__).parent))
    from grpo_train import (
        inject_adapters,
        format_prompt,
        extract_answer,
        extract_gsm8k_answer,
        answers_match,
    )

    # ------------------------------------------------------------------
    # Load model
    # ------------------------------------------------------------------
    print(f"Loading tokenizer from {MODEL_PATH}...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    print(f"Loading model...")
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH, torch_dtype=torch.bfloat16, device_map="cpu",
        trust_remote_code=True,
    )
    print("Model loaded.")

    # Patch model with production soft-chip kernel + FP32 LM head
    from softchip import patch_model, patch_lm_head_fp32
    patch_model(model, backend="cpu", verbose=True)
    patch_lm_head_fp32(model, verbose=True)

    # Inject adapters (needed to load checkpoint scales; zero by default)
    adapters = inject_adapters(model)
    print(f"Adapters injected ({len(adapters)} experts, all scales = 0.0)")

    # ------------------------------------------------------------------
    # Load dataset
    # ------------------------------------------------------------------
    print("Loading GSM8K test split...")
    gsm8k = load_dataset("gsm8k", "main")
    test_data = list(gsm8k["test"])
    print(f"  {len(test_data)} test examples. Evaluating first {n_samples}.")

    # ------------------------------------------------------------------
    # Shared eval function
    # ------------------------------------------------------------------
    def run_eval(label, scale_override=None):
        """
        Evaluate model on the first n_samples GSM8K test examples.
        scale_override: if a float, set all adapter scales to this value.
                        if None, use whatever the adapters currently hold.
        """
        if scale_override is not None:
            for _, exp in adapters:
                exp.scale.data.fill_(scale_override)

        correct = 0
        results = []
        t0 = time.time()
        for i, ex in enumerate(test_data[:n_samples]):
            prompt_text = format_prompt(ex["question"])
            prompt_ids = tokenizer(prompt_text, return_tensors="pt").input_ids
            with torch.no_grad():
                out = model.generate(
                    prompt_ids,
                    max_new_tokens=MAX_NEW_TOKENS,
                    do_sample=False,
                    pad_token_id=tokenizer.eos_token_id,
                )
            completion = tokenizer.decode(
                out[0, prompt_ids.shape[1]:], skip_special_tokens=True
            )
            pred = extract_answer(completion)
            gt = extract_gsm8k_answer(ex["answer"])
            match = answers_match(pred, gt)
            if match:
                correct += 1
            results.append({"pred": pred, "gt": gt, "match": match})
            status = "OK" if match else "WRONG"
            print(f"  [{i+1:3d}/{n_samples}] pred={pred!s:>12}  gt={gt!s:>10}  {status}")

        elapsed = time.time() - t0
        accuracy = correct / n_samples
        print(f"\n  {label}: {correct}/{n_samples} = {accuracy*100:.1f}%  "
              f"({elapsed:.0f}s, {elapsed/n_samples:.1f}s/sample)")
        return accuracy, results

    # ------------------------------------------------------------------
    # Condition 1: Base model (zero adapters)
    # ------------------------------------------------------------------
    print("\n" + "="*60)
    print("CONDITION 1: Base model (all adapter scales = 0.0)")
    print("="*60)
    base_acc, _ = run_eval("Base model", scale_override=0.0)

    # ------------------------------------------------------------------
    # Condition 2: Trained checkpoint (if provided)
    # ------------------------------------------------------------------
    trained_acc = None
    if args.checkpoint:
        print("\n" + "="*60)
        print(f"CONDITION 2: Trained adapters from {args.checkpoint}")
        print("="*60)
        ckpt = torch.load(args.checkpoint, weights_only=False)
        loaded = 0
        for name, exp in adapters:
            if name in ckpt["adapter_scales"]:
                exp.scale.data.copy_(ckpt["adapter_scales"][name])
                loaded += 1
        step = ckpt.get("step", "?")
        mean_abs = sum(abs(exp.scale.item()) for _, exp in adapters) / len(adapters)
        print(f"  Loaded {loaded}/{len(adapters)} scales from step {step}. "
              f"Mean |scale| = {mean_abs:.4f}")
        trained_acc, _ = run_eval(f"Trained (step {step})")

    # ------------------------------------------------------------------
    # Condition 3: Random adapters at specified scale (if requested)
    # ------------------------------------------------------------------
    random_acc = None
    if args.random_scale is not None:
        print("\n" + "="*60)
        print(f"CONDITION 3: Random scale (all scales = ±{args.random_scale:.4f})")
        print("="*60)
        # Set all scales to +random_scale (fixed direction noise sanity check)
        random_acc, _ = run_eval(
            f"Random scale={args.random_scale:.4f}",
            scale_override=args.random_scale,
        )

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    print(f"  Base model (zero adapters):   {base_acc*100:.1f}%  ({n_samples} samples, GSM8K test)")
    if trained_acc is not None:
        delta = trained_acc - base_acc
        sign = "+" if delta >= 0 else ""
        print(f"  Trained adapters (step {ckpt['step']}):  {trained_acc*100:.1f}%  "
              f"({sign}{delta*100:.1f}pp vs base)")
    if random_acc is not None:
        delta = random_acc - base_acc
        sign = "+" if delta >= 0 else ""
        print(f"  Random scale={args.random_scale:.4f}:          {random_acc*100:.1f}%  "
              f"({sign}{delta*100:.1f}pp vs base)")
    print()
    print("Note: 95% confidence interval on N samples ≈ ±" +
          f"{1.96 * (base_acc*(1-base_acc)/n_samples)**0.5 * 100:.1f}pp")


if __name__ == "__main__":
    main()
