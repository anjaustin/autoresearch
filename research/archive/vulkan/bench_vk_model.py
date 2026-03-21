#!/usr/bin/env python3
"""
Instrumented model forward pass benchmark.

Measures:
1. Stock PyTorch forward (baseline)
2. CPU soft-chip forward
3. Vulkan soft-chip forward
4. Time breakdown: matmul time vs non-matmul overhead
5. Autoregressive (M=1) generation benchmark

Also runs a 200-token generation benchmark for the rollout projection.
"""

import ctypes
import sys
import time

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_PATH = "models/bitnet-b1.58-2B-4T-bf16"


def bench_forward(model, input_ids, label, warmup=1, repeats=3):
    """Benchmark forward pass, return average time in ms."""
    for _ in range(warmup):
        with torch.no_grad():
            model(input_ids)

    times = []
    for _ in range(repeats):
        t0 = time.time()
        with torch.no_grad():
            model(input_ids)
        times.append((time.time() - t0) * 1000)

    avg = sum(times) / len(times)
    print(f"  {label}: {avg:.0f} ms (runs: {', '.join(f'{t:.0f}' for t in times)})")
    return avg


def bench_generation(model, tokenizer, n_tokens, label, repeats=1):
    """Benchmark autoregressive generation of n_tokens."""
    prompt = "The meaning of life is"
    input_ids = tokenizer(prompt, return_tensors="pt").input_ids

    times = []
    for _ in range(repeats):
        t0 = time.time()
        with torch.no_grad():
            output = model.generate(
                input_ids,
                max_new_tokens=n_tokens,
                do_sample=False,
                use_cache=True,
            )
        elapsed = time.time() - t0
        times.append(elapsed)

    avg = sum(times) / len(times)
    tok_per_sec = n_tokens / avg
    text = tokenizer.decode(output[0], skip_special_tokens=True)
    print(f"  {label}: {avg:.1f}s for {n_tokens} tokens ({tok_per_sec:.1f} tok/s)")
    print(f"    Output: {text[:120]}...")
    return avg


def count_matmul_time(model):
    """Instrument patched layers to measure time spent in matmuls."""
    total_time = [0.0]
    call_count = [0]
    original_forwards = {}

    for name, module in model.named_modules():
        if hasattr(module, "_softchip_original_forward"):
            current_forward = module.forward

            def make_timed(fwd, nm):
                def timed_forward(x):
                    t0 = time.time()
                    result = fwd(x)
                    total_time[0] += time.time() - t0
                    call_count[0] += 1
                    return result

                return timed_forward

            original_forwards[name] = current_forward
            module.forward = make_timed(current_forward, name)

    return total_time, call_count, original_forwards


def restore_forwards(model, original_forwards):
    """Restore forwards after timing."""
    for name, module in model.named_modules():
        if name in original_forwards:
            module.forward = original_forwards[name]


def main():
    print("=" * 60)
    print("  Vulkan Soft-Chip: Full Model Benchmark")
    print("=" * 60)

    # Load model
    print("\n[1] Loading model and tokenizer...")
    t0 = time.time()
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH,
        dtype=torch.bfloat16,
        device_map="cpu",
        trust_remote_code=False,
    )
    model.eval()
    print(f"    Loaded in {time.time() - t0:.1f}s")

    # Prepare inputs
    prompt = "The capital of France is"
    input_ids = tokenizer(prompt, return_tensors="pt").input_ids
    print(f"    Prompt: {prompt!r} -> {input_ids.shape[1]} tokens")

    # === Stock PyTorch ===
    print("\n[2] Stock PyTorch forward pass...")
    stock_ms = bench_forward(model, input_ids, "Stock PyTorch", warmup=1, repeats=3)

    # === CPU soft-chip ===
    print("\n[3] CPU soft-chip forward pass...")
    from softchip.torch_ternary import patch_model, unpatch_model

    t0 = time.time()
    patch_model(model, backend="cpu", verbose=True)
    cpu_pack_time = time.time() - t0

    cpu_ms = bench_forward(model, input_ids, "CPU soft-chip", warmup=1, repeats=3)

    # Time breakdown
    total_time, call_count, orig_fwds = count_matmul_time(model)
    with torch.no_grad():
        total_time[0] = 0.0
        call_count[0] = 0
        t0 = time.time()
        model(input_ids)
        wall_ms = (time.time() - t0) * 1000
        matmul_ms = total_time[0] * 1000
    print(
        f"    Breakdown: matmul={matmul_ms:.0f}ms, non-matmul={wall_ms - matmul_ms:.0f}ms, "
        f"calls={call_count[0]}"
    )
    restore_forwards(model, orig_fwds)

    unpatch_model(model, verbose=False)

    # === Vulkan soft-chip ===
    print("\n[4] Vulkan soft-chip forward pass...")
    t0 = time.time()
    patch_model(model, backend="vulkan", verbose=True)
    vk_pack_time = time.time() - t0

    vk_ms = bench_forward(model, input_ids, "Vulkan soft-chip", warmup=1, repeats=3)

    # Time breakdown
    total_time, call_count, orig_fwds = count_matmul_time(model)
    with torch.no_grad():
        total_time[0] = 0.0
        call_count[0] = 0
        t0 = time.time()
        model(input_ids)
        wall_ms = (time.time() - t0) * 1000
        vk_matmul_ms = total_time[0] * 1000
    print(
        f"    Breakdown: matmul={vk_matmul_ms:.0f}ms, non-matmul={wall_ms - vk_matmul_ms:.0f}ms, "
        f"calls={call_count[0]}"
    )
    restore_forwards(model, orig_fwds)

    # === Summary ===
    print("\n" + "=" * 60)
    print("  FORWARD PASS SUMMARY (6 tokens)")
    print("=" * 60)
    print(f"  Stock PyTorch:     {stock_ms:.0f} ms")
    print(f"  CPU soft-chip:     {cpu_ms:.0f} ms  ({stock_ms / cpu_ms:.1f}x vs stock)")
    print(f"  Vulkan soft-chip:  {vk_ms:.0f} ms  ({stock_ms / vk_ms:.1f}x vs stock)")
    print(
        f"  Vulkan matmul:     {vk_matmul_ms:.0f} ms ({vk_matmul_ms / vk_ms * 100:.0f}% of forward)"
    )
    print(
        f"  Non-matmul:        {vk_ms - vk_matmul_ms:.0f} ms ({(vk_ms - vk_matmul_ms) / vk_ms * 100:.0f}% of forward)"
    )

    # === Autoregressive generation ===
    print("\n" + "=" * 60)
    print("  AUTOREGRESSIVE GENERATION BENCHMARK")
    print("=" * 60)

    # 20-token generation (quick test)
    print("\n[5] 20-token generation (Vulkan)...")
    gen20_time = bench_generation(model, tokenizer, 20, "Vulkan 20-tok", repeats=1)
    projected_200 = gen20_time * 10
    print(f"    Projected 200-token: {projected_200:.0f}s")

    # 50-token generation
    print("\n[6] 50-token generation (Vulkan)...")
    gen50_time = bench_generation(model, tokenizer, 50, "Vulkan 50-tok", repeats=1)
    projected_200_from50 = gen50_time * 4
    per_token_ms = gen50_time / 50 * 1000
    print(f"    Per-token: {per_token_ms:.0f} ms/tok")
    print(f"    Projected 200-token: {projected_200_from50:.0f}s")

    unpatch_model(model, verbose=False)

    # Baseline: CPU soft-chip 20-token for comparison
    print("\n[7] 20-token generation (CPU soft-chip)...")
    patch_model(model, backend="cpu", verbose=True)
    cpu_gen20_time = bench_generation(model, tokenizer, 20, "CPU 20-tok", repeats=1)
    cpu_projected_200 = cpu_gen20_time * 10
    cpu_per_token = cpu_gen20_time / 20 * 1000
    print(f"    Per-token: {cpu_per_token:.0f} ms/tok")
    print(f"    Projected 200-token: {cpu_projected_200:.0f}s")
    unpatch_model(model, verbose=False)

    # Final summary
    print("\n" + "=" * 60)
    print("  GENERATION SUMMARY")
    print("=" * 60)
    print(f"  Vulkan per-token:    {per_token_ms:.0f} ms/tok")
    print(f"  CPU per-token:       {cpu_per_token:.0f} ms/tok")
    print(f"  Vulkan 200-tok est:  {projected_200_from50:.0f}s")
    print(f"  CPU 200-tok est:     {cpu_projected_200:.0f}s")
    if cpu_per_token > 0:
        print(f"  Vulkan speedup:      {cpu_per_token / per_token_ms:.2f}x")
    print(f"\n  Target was ~31s for 200 tokens (Vulkan)")
    print(f"  Actual projection:   {projected_200_from50:.0f}s")


if __name__ == "__main__":
    main()
