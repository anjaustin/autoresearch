#!/usr/bin/env python3
"""
Full model benchmark: soft-chip vs PyTorch for BitNet b1.58 forward pass.

Tests both M=19 (batched, simulating a full prompt) and M=1 (autoregressive,
simulating per-token generation during RL rollouts).
"""

import sys
import time

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_PATH = "models/bitnet-b1.58-2B-4T-bf16"


def bench_forward(model, input_ids, label, warmup=1, repeats=3):
    """Benchmark forward pass, return average time in seconds."""
    # Warmup
    for _ in range(warmup):
        with torch.no_grad():
            model(input_ids)

    times = []
    for _ in range(repeats):
        t0 = time.time()
        with torch.no_grad():
            model(input_ids)
        times.append(time.time() - t0)

    avg = sum(times) / len(times)
    print(f"  {label}: {avg:.3f}s (runs: {', '.join(f'{t:.3f}s' for t in times)})")
    return avg


def main():
    print("=" * 60)
    print("Full Model Benchmark: Soft-Chip vs PyTorch")
    print("=" * 60)

    # Load model
    print("\n[1] Loading model...")
    t0 = time.time()
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH,
        dtype=torch.bfloat16,
        device_map="cpu",
        trust_remote_code=False,
    )
    model.eval()
    print(f"    Loaded in {time.time() - t0:.1f}s")

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)

    # Prepare inputs
    prompt = (
        "The key insight of the TinyLoRA paper is that reinforcement learning enables"
    )
    input_ids = tokenizer(prompt, return_tensors="pt")["input_ids"]
    seq_len = input_ids.shape[1]
    print(f"\n    Prompt: '{prompt}'")
    print(f"    Tokens: {seq_len}")

    # Also prepare a single-token input for M=1 benchmark
    input_ids_1 = input_ids[:, :1]

    # ---------------------------------------------------------------
    # Benchmark: PyTorch (stock AutoBitLinear)
    # ---------------------------------------------------------------
    print(f"\n[2] PyTorch (stock AutoBitLinear)")
    pytorch_time = bench_forward(model, input_ids, f"M={seq_len}")
    pytorch_time_1 = bench_forward(model, input_ids_1, "M=1", warmup=1, repeats=3)

    # ---------------------------------------------------------------
    # Patch model with soft-chip
    # ---------------------------------------------------------------
    print(f"\n[3] Patching model with soft-chip...")
    from softchip.torch_ternary import patch_model, unpatch_model

    n_patched = patch_model(model)

    # ---------------------------------------------------------------
    # Benchmark: Soft-chip
    # ---------------------------------------------------------------
    print(f"\n[4] Soft-chip (AVX2 ternary kernel)")
    softchip_time = bench_forward(model, input_ids, f"M={seq_len}")
    softchip_time_1 = bench_forward(model, input_ids_1, "M=1", warmup=1, repeats=3)

    # ---------------------------------------------------------------
    # Verify outputs match
    # ---------------------------------------------------------------
    print(f"\n[5] Output verification...")
    unpatch_model(model, verbose=False)
    with torch.no_grad():
        out_pytorch = model(input_ids).logits

    patch_model(model, verbose=False)
    with torch.no_grad():
        out_softchip = model(input_ids).logits

    diff = (out_pytorch.float() - out_softchip.float()).abs()
    max_diff = diff.max().item()
    mean_diff = diff.mean().item()
    cos_sim = torch.nn.functional.cosine_similarity(
        out_pytorch.float().flatten().unsqueeze(0),
        out_softchip.float().flatten().unsqueeze(0),
    ).item()
    print(f"    Max abs diff:  {max_diff:.4f}")
    print(f"    Mean abs diff: {mean_diff:.4f}")
    print(f"    Cosine sim:    {cos_sim:.8f}")

    # ---------------------------------------------------------------
    # Summary
    # ---------------------------------------------------------------
    print(f"\n{'=' * 60}")
    print("SUMMARY")
    print("=" * 60)
    print(f"{'Metric':<30} {'PyTorch':>12} {'Soft-chip':>12} {'Speedup':>10}")
    print("-" * 64)
    print(
        f"{'Forward M=' + str(seq_len):<30} {pytorch_time:>11.3f}s {softchip_time:>11.3f}s {pytorch_time / softchip_time:>9.1f}x"
    )
    print(
        f"{'Forward M=1':<30} {pytorch_time_1:>11.3f}s {softchip_time_1:>11.3f}s {pytorch_time_1 / softchip_time_1:>9.1f}x"
    )
    print(
        f"{'Output match':<30} {'max_diff=' + f'{max_diff:.4f}':>12} {'cos_sim=' + f'{cos_sim:.6f}':>12}"
    )

    # Projected autoresearch impact
    print(f"\n--- Projected Impact on Autoresearch Loop ---")
    tok_per_rollout = 200
    rollout_time_pytorch = pytorch_time_1 * tok_per_rollout
    rollout_time_softchip = softchip_time_1 * tok_per_rollout
    print(
        f"200-token rollout (M=1 × 200): PyTorch={rollout_time_pytorch:.0f}s, "
        f"Soft-chip={rollout_time_softchip:.0f}s"
    )

    unpatch_model(model, verbose=False)
    del model

    return 0


if __name__ == "__main__":
    sys.exit(main())
