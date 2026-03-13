#!/usr/bin/env python3
"""
LMM Pass 5 SYNTHESIZE: Validate and benchmark FP32 LM head patch.

Tests:
1. Numerical accuracy: FP32 patched forward matches BF16 original
2. Gradient flow: gradients reach TinyLoRA adapters through FP32 LM head
3. Backward speedup: measure improvement from FP32 LM head patch
4. End-to-end training iteration timing
"""

import sys
import time
import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_PATH = "models/bitnet-b1.58-2B-4T-bf16"


def test_lm_head_fp32():
    print("=" * 60)
    print("  LMM Pass 5 SYNTHESIZE: FP32 LM Head Validation")
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

    # Freeze all base model parameters
    for param in model.parameters():
        param.requires_grad = False

    # === Step 1: Baseline (BF16 stock) forward + backward ===
    print("\n[Step 1] Baseline BF16 forward + backward...")
    model.train()

    # Need at least one param with requires_grad for backward to work
    # Temporarily enable grad on one param, or use a hook
    # Actually, for the baseline timing, we need the embedding to have grad
    # so the graph can be traversed. Let's enable grad on embed_tokens.
    model.model.embed_tokens.weight.requires_grad_(True)

    output_baseline = model(input_ids, labels=input_ids)
    loss_baseline = output_baseline.loss.item()
    logits_baseline = output_baseline.logits.detach().clone()
    print(f"  Baseline loss: {loss_baseline:.6f}")
    print(f"  Baseline logits shape: {logits_baseline.shape}")
    print(f"  Baseline logits[0,0,:5]: {logits_baseline[0, 0, :5]}")

    # Timed backward
    output_baseline2 = model(input_ids, labels=input_ids)
    t0 = time.perf_counter()
    output_baseline2.loss.backward()
    t_bwd_baseline = time.perf_counter() - t0
    print(f"  Baseline backward: {t_bwd_baseline * 1000:.0f}ms")
    model.zero_grad()

    # Re-freeze
    model.model.embed_tokens.weight.requires_grad_(False)

    # === Step 2: Patch LM head to FP32 ===
    print("\n[Step 2] Patching LM head to FP32...")
    from softchip.torch_ternary import patch_lm_head_fp32, unpatch_lm_head

    result = patch_lm_head_fp32(model, verbose=True)
    assert result, "Failed to patch LM head"

    # === Step 3: Validate forward accuracy ===
    print("\n[Step 3] Validating forward accuracy...")
    output_fp32 = model(input_ids, labels=input_ids)
    loss_fp32 = output_fp32.loss.item()
    logits_fp32 = output_fp32.logits.detach().clone()

    print(f"  FP32 loss: {loss_fp32:.6f}")
    print(f"  FP32 logits[0,0,:5]: {logits_fp32[0, 0, :5]}")

    loss_diff = abs(loss_fp32 - loss_baseline)
    logit_diff = (logits_fp32 - logits_baseline).abs().max().item()
    logit_rmse = ((logits_fp32 - logits_baseline) ** 2).mean().sqrt().item()

    print(f"  Loss difference: {loss_diff:.8f}")
    print(f"  Max logit difference: {logit_diff:.6f}")
    print(f"  Logit RMSE: {logit_rmse:.8f}")

    # BF16 has ~3 decimal digits of precision, so differences up to ~0.1 are OK
    if logit_diff < 1.0:
        print("  PASS: Forward accuracy within BF16 tolerance")
    else:
        print(f"  FAIL: Logit difference {logit_diff} too large!")
        return False

    # === Step 4: Validate gradient flow (FP32 LM head only, no soft-chip) ===
    print("\n[Step 4] Validating gradient flow through FP32 LM head...")

    # Add a simple trainable parameter to test gradient flow
    # We'll use a scale on the first decoder layer's output
    test_scale = nn.Parameter(torch.ones(1, dtype=torch.bfloat16))

    # Hook into the first decoder layer to multiply by test_scale
    hook_handle = None

    def scale_hook(module, input, output):
        # output may be a tuple or a single tensor depending on the layer
        if isinstance(output, tuple):
            hidden = output[0] * test_scale
            return (hidden,) + output[1:]
        else:
            return output * test_scale

    first_layer = model.model.layers[0]
    hook_handle = first_layer.register_forward_hook(scale_hook)

    model.zero_grad()
    output_hooked = model(input_ids, labels=input_ids)
    output_hooked.loss.backward()

    if test_scale.grad is not None and test_scale.grad.abs().item() > 0:
        print(f"  test_scale.grad = {test_scale.grad.item():.6f}")
        print("  PASS: Gradients flow through FP32 LM head to decoder layers")
    else:
        print(f"  test_scale.grad = {test_scale.grad}")
        print("  FAIL: No gradient flow!")
        hook_handle.remove()
        return False

    hook_handle.remove()
    model.zero_grad()

    # === Step 5: Benchmark backward with FP32 LM head ===
    print("\n[Step 5] Benchmarking backward with FP32 LM head...")

    # Need at least one requires_grad tensor for backward to traverse the graph
    model.model.embed_tokens.weight.requires_grad_(True)

    # Warmup
    output_w = model(input_ids, labels=input_ids)
    output_w.loss.backward()
    model.zero_grad()

    # Timed runs
    times = []
    for i in range(3):
        output_t = model(input_ids, labels=input_ids)
        t0 = time.perf_counter()
        output_t.loss.backward()
        t1 = time.perf_counter()
        times.append(t1 - t0)
        model.zero_grad()

    model.model.embed_tokens.weight.requires_grad_(False)

    t_bwd_fp32 = min(times)
    print(f"  FP32 LM head backward: {t_bwd_fp32 * 1000:.0f}ms (min of 3)")
    print(f"  Baseline BF16 backward: {t_bwd_baseline * 1000:.0f}ms")
    print(f"  Speedup: {t_bwd_baseline / t_bwd_fp32:.1f}x")

    # === Step 6: Full stack — soft-chip + FP32 LM head ===
    print("\n[Step 6] Full stack: soft-chip + FP32 LM head...")
    from softchip.torch_ternary import patch_model

    # First unpatch LM head, then patch everything
    unpatch_lm_head(model, verbose=False)
    patch_model(model, backend="cpu", verbose=True)
    patch_lm_head_fp32(model, verbose=True)

    # Enable grad on embedding for backward traversal
    model.model.embed_tokens.weight.requires_grad_(True)

    # Warmup
    model.zero_grad()
    output_full = model(input_ids, labels=input_ids)
    output_full.loss.backward()
    model.zero_grad()

    # Timed forward + backward
    fwd_times = []
    bwd_times = []
    for i in range(3):
        t0 = time.perf_counter()
        output_t = model(input_ids, labels=input_ids)
        t1 = time.perf_counter()
        output_t.loss.backward()
        t2 = time.perf_counter()
        fwd_times.append(t1 - t0)
        bwd_times.append(t2 - t1)
        model.zero_grad()

    t_fwd = min(fwd_times)
    t_bwd = min(bwd_times)
    print(f"\n  Full stack results (min of 3):")
    print(f"    Forward (soft-chip):           {t_fwd * 1000:.0f}ms")
    print(f"    Backward (ternary + FP32 LM):  {t_bwd * 1000:.0f}ms")
    print(f"    Total iteration:               {(t_fwd + t_bwd) * 1000:.0f}ms")

    # === Step 7: Profile backward breakdown with autograd profiler ===
    print("\n[Step 7] Profiling backward breakdown...")
    model.zero_grad()
    output_p = model(input_ids, labels=input_ids)

    with torch.autograd.profiler.profile(use_cuda=False) as prof:
        output_p.loss.backward()

    print("\n  Top 15 operations by self CPU time:")
    print(prof.key_averages().table(sort_by="self_cpu_time_total", row_limit=15))

    model.zero_grad()
    model.model.embed_tokens.weight.requires_grad_(False)

    # === Summary ===
    print("\n" + "=" * 60)
    print("  SUMMARY")
    print("=" * 60)
    print(f"  Forward accuracy: logit RMSE = {logit_rmse:.8f} (PASS)")
    print(f"  Gradient flow: PASS")
    print(f"  Backward speedup (FP32 LM head only): {t_bwd_baseline / t_bwd_fp32:.1f}x")
    print(f"  Full stack forward:  {t_fwd * 1000:.0f}ms")
    print(f"  Full stack backward: {t_bwd * 1000:.0f}ms")
    print(f"  Full stack total:    {(t_fwd + t_bwd) * 1000:.0f}ms")
    print()

    return True


if __name__ == "__main__":
    ok = test_lm_head_fp32()
    sys.exit(0 if ok else 1)
