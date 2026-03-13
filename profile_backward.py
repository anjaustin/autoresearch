#!/usr/bin/env python3
"""
LMM Pass 5 RAW: Profile the backward pass to identify bottlenecks.

Instruments the model to measure time in each operation category during backward.
Key finding: LM head (nn.Linear, 128256x2560, dense BF16) consumed 92.6% of
backward time due to MKL's catastrophically slow BF16 GEMM on Zen 3.
"""

import sys
import time
import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_PATH = "models/bitnet-b1.58-2B-4T-bf16"


def profile_stock_backward():
    """Profile stock PyTorch backward with autograd profiler."""
    print("=" * 60)
    print("  LMM Pass 5 RAW: Backward Pass Profiling")
    print("=" * 60)

    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH,
        dtype=torch.bfloat16,
        device_map="cpu",
        trust_remote_code=False,
    )

    prompt = "The capital of France is"
    input_ids = tokenizer(prompt, return_tensors="pt").input_ids
    print(f"\nPrompt: {prompt!r} -> {input_ids.shape[1]} tokens")

    # === 1. Stock PyTorch backward with torch.profiler ===
    print("\n[1] Stock PyTorch backward (profiled)...")
    model.train()
    model.zero_grad()

    # Warmup forward
    output = model(input_ids, labels=input_ids)
    loss = output.loss

    # Profile backward
    with torch.autograd.profiler.profile(use_cuda=False) as prof:
        loss.backward()

    # Print top operations by self CPU time
    print("\n  Top 20 operations by self CPU time (backward):")
    print(prof.key_averages().table(sort_by="self_cpu_time_total", row_limit=20))

    # Also group by operation type
    print("\n  Top 20 by total CPU time:")
    print(prof.key_averages().table(sort_by="cpu_time_total", row_limit=20))

    model.zero_grad()

    # === 2. Soft-chip backward profiled ===
    print("\n[2] CPU soft-chip backward (profiled)...")
    from softchip.torch_ternary import patch_model, unpatch_model

    patch_model(model, backend="cpu", verbose=True)
    model.train()
    model.zero_grad()

    # Forward
    output = model(input_ids, labels=input_ids)
    loss = output.loss

    # Profile backward
    with torch.autograd.profiler.profile(use_cuda=False) as prof2:
        loss.backward()

    print("\n  Top 20 operations by self CPU time (soft-chip backward):")
    print(prof2.key_averages().table(sort_by="self_cpu_time_total", row_limit=20))

    print("\n  Top 20 by total CPU time:")
    print(prof2.key_averages().table(sort_by="cpu_time_total", row_limit=20))

    model.zero_grad()

    # === 3. Manual timing: patch individual layer types to measure their backward ===
    print("\n[3] Per-operation backward timing...")
    unpatch_model(model, verbose=False)
    patch_model(model, backend="cpu", verbose=False)

    # Time just the backward
    model.zero_grad()
    output = model(input_ids, labels=input_ids)
    loss = output.loss

    t0 = time.time()
    loss.backward()
    total_bwd = time.time() - t0
    print(f"  Total backward: {total_bwd:.1f}s")

    # === 4. Measure backward WITHOUT autograd (manual gradient computation) ===
    print("\n[4] Counting autograd graph nodes...")
    model.zero_grad()
    output = model(input_ids, labels=input_ids)
    loss = output.loss

    # Count autograd graph nodes
    node_count = 0
    node_types = {}
    visited = set()

    def count_nodes(grad_fn):
        nonlocal node_count
        if grad_fn is None or id(grad_fn) in visited:
            return
        visited.add(id(grad_fn))
        node_count += 1
        name = type(grad_fn).__name__
        node_types[name] = node_types.get(name, 0) + 1
        for child, _ in grad_fn.next_functions:
            count_nodes(child)

    count_nodes(loss.grad_fn)
    print(f"  Total autograd nodes: {node_count}")
    print(f"  Unique node types: {len(node_types)}")
    print(f"\n  Node types (top 20 by count):")
    for name, cnt in sorted(node_types.items(), key=lambda x: -x[1])[:20]:
        print(f"    {name}: {cnt}")

    unpatch_model(model, verbose=False)


if __name__ == "__main__":
    profile_stock_backward()
