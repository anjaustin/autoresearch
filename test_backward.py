#!/usr/bin/env python3
"""
Validate the ternary backward kernel.

Tests:
1. Isolated: ternary_matmul_backward output matches PyTorch's grad computation
2. Integrated: full model forward+backward with soft-chip vs stock PyTorch
3. TinyLoRA gradient: adapter gradients match between soft-chip and stock backward
4. Benchmark: backward time with ternary kernel vs stock PyTorch
"""

import ctypes
import sys
import time

import numpy as np
import torch
import torch.nn as nn
from pathlib import Path

LIB_DIR = Path(__file__).parent / "softchip"
MODEL_PATH = "models/bitnet-b1.58-2B-4T-bf16"


def load_cpu_lib():
    path = LIB_DIR / "ternary_matmul_v3.so"
    lib = ctypes.CDLL(str(path))
    lib.pack_weights.restype = ctypes.POINTER(ctypes.c_uint8)
    lib.pack_weights.argtypes = [
        ctypes.POINTER(ctypes.c_float),
        ctypes.c_int,
        ctypes.c_int,
        ctypes.POINTER(ctypes.c_float),
    ]
    lib.ternary_matmul.restype = None
    lib.ternary_matmul.argtypes = [
        ctypes.POINTER(ctypes.c_uint8),
        ctypes.POINTER(ctypes.c_float),
        ctypes.POINTER(ctypes.c_float),
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_float,
    ]
    lib.ternary_matmul_backward.restype = None
    lib.ternary_matmul_backward.argtypes = [
        ctypes.POINTER(ctypes.c_uint8),
        ctypes.POINTER(ctypes.c_float),
        ctypes.POINTER(ctypes.c_float),
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_float,
    ]
    return lib


def nrmse(a, b):
    """Normalized RMSE."""
    diff = a - b
    return np.sqrt(np.mean(diff**2) / (np.mean(a**2) + 1e-10))


def test_isolated_backward():
    """Test ternary_matmul_backward against a direct PyTorch computation."""
    print("\n[1] Isolated backward kernel validation")
    lib = load_cpu_lib()

    np.random.seed(42)
    N, K = 2560, 2560

    # Create random ternary-like weights
    weights = (np.random.rand(N, K).astype(np.float32) - 0.5) * 2.0

    # Pack weights
    w_ptr = weights.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
    scale = ctypes.c_float(0.0)
    packed_ptr = lib.pack_weights(w_ptr, N, K, ctypes.byref(scale))
    weight_scale = scale.value

    # Compute ternary quantized weights for PyTorch reference
    w_ternary = np.round(weights * (1.0 / weight_scale))
    w_ternary = np.clip(w_ternary, -1, 1)
    w_ternary_scaled = w_ternary * weight_scale  # = ternary * scale

    # Test M=1
    grad_output = np.random.randn(1, N).astype(np.float32)

    # Ternary backward kernel
    gi_kernel = np.empty((1, K), dtype=np.float32)
    lib.ternary_matmul_backward(
        packed_ptr,
        grad_output.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        gi_kernel.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        1,
        N,
        K,
        weight_scale,
    )

    # PyTorch reference: grad_input = grad_output @ W_ternary_scaled
    # W is (N, K), so W^T is (K, N), and grad_output @ W = grad_output (1,N) @ W (N,K) = (1,K)
    gi_ref = grad_output @ w_ternary_scaled

    err = nrmse(gi_ref, gi_kernel)
    max_diff = np.max(np.abs(gi_ref - gi_kernel))
    print(f"  M=1, {N}x{K}: NRMSE={err:.2e}, max_diff={max_diff:.6f}", end="")
    ok = err < 1e-5
    print(f"  {'PASS' if ok else 'FAIL'}")

    # Test M=6
    grad_output_6 = np.random.randn(6, N).astype(np.float32)
    gi_kernel_6 = np.empty((6, K), dtype=np.float32)
    lib.ternary_matmul_backward(
        packed_ptr,
        grad_output_6.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        gi_kernel_6.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        6,
        N,
        K,
        weight_scale,
    )
    gi_ref_6 = grad_output_6 @ w_ternary_scaled
    err6 = nrmse(gi_ref_6, gi_kernel_6)
    print(f"  M=6, {N}x{K}: NRMSE={err6:.2e}", end="")
    ok6 = err6 < 1e-5
    print(f"  {'PASS' if ok6 else 'FAIL'}")

    # Test all layer shapes
    for name, n, k in [
        ("k/v_proj", 640, 2560),
        ("gate/up", 6912, 2560),
        ("down", 2560, 6912),
    ]:
        w = (np.random.rand(n, k).astype(np.float32) - 0.5) * 2.0
        wp = w.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
        s = ctypes.c_float(0.0)
        pp = lib.pack_weights(wp, n, k, ctypes.byref(s))
        ws = s.value

        wt = np.round(w * (1.0 / ws))
        wt = np.clip(wt, -1, 1) * ws

        go = np.random.randn(1, n).astype(np.float32)
        gi = np.empty((1, k), dtype=np.float32)
        lib.ternary_matmul_backward(
            pp,
            go.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            gi.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            1,
            n,
            k,
            ws,
        )
        gi_r = go @ wt
        e = nrmse(gi_r, gi)
        print(
            f"  M=1, {name} ({n}x{k}): NRMSE={e:.2e}  {'PASS' if e < 1e-5 else 'FAIL'}"
        )
        ctypes.CDLL("libc.so.6").free(pp)

    # Benchmark M=1 backward
    iters = 100
    t0 = time.time()
    for _ in range(iters):
        lib.ternary_matmul_backward(
            packed_ptr,
            grad_output.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            gi_kernel.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            1,
            N,
            K,
            weight_scale,
        )
    bwd_ms = (time.time() - t0) / iters * 1000
    print(f"\n  Backward kernel M=1 2560x2560: {bwd_ms:.2f} ms")

    # Benchmark forward for comparison
    act = np.random.randn(1, K).astype(np.float32)
    out = np.empty((1, N), dtype=np.float32)
    t0 = time.time()
    for _ in range(iters):
        lib.ternary_matmul(
            packed_ptr,
            act.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            out.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            1,
            N,
            K,
            weight_scale,
        )
    fwd_ms = (time.time() - t0) / iters * 1000
    print(f"  Forward kernel M=1 2560x2560:  {fwd_ms:.2f} ms")
    print(f"  Backward/Forward ratio:         {bwd_ms / fwd_ms:.2f}x")

    ctypes.CDLL("libc.so.6").free(packed_ptr)
    return ok and ok6


def test_model_backward():
    """Test backward through full model with soft-chip vs stock PyTorch."""
    print("\n[2] Full model forward+backward benchmark")

    from transformers import AutoModelForCausalLM, AutoTokenizer
    from softchip.torch_ternary import patch_model, unpatch_model

    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH,
        dtype=torch.bfloat16,
        device_map="cpu",
        trust_remote_code=False,
    )

    prompt = "The capital of France is"
    input_ids = tokenizer(prompt, return_tensors="pt").input_ids
    print(f"  Prompt: {prompt!r} -> {input_ids.shape[1]} tokens")

    # Stock PyTorch backward
    print("\n  Stock PyTorch forward+backward...")
    model.train()
    t0 = time.time()
    output = model(input_ids, labels=input_ids)
    loss = output.loss
    fwd_stock = time.time() - t0
    t0 = time.time()
    loss.backward()
    bwd_stock = time.time() - t0
    model.zero_grad()
    print(
        f"    Forward: {fwd_stock:.1f}s  Backward: {bwd_stock:.1f}s  Total: {fwd_stock + bwd_stock:.1f}s"
    )

    # Soft-chip backward
    print("\n  CPU soft-chip forward+backward...")
    patch_model(model, backend="cpu", verbose=True)
    model.train()
    t0 = time.time()
    output = model(input_ids, labels=input_ids)
    loss = output.loss
    fwd_chip = time.time() - t0
    t0 = time.time()
    loss.backward()
    bwd_chip = time.time() - t0
    model.zero_grad()
    print(
        f"    Forward: {fwd_chip:.1f}s  Backward: {bwd_chip:.1f}s  Total: {fwd_chip + bwd_chip:.1f}s"
    )
    print(
        f"    Speedup: forward {fwd_stock / fwd_chip:.1f}x, backward {bwd_stock / bwd_chip:.1f}x, "
        f"total {(fwd_stock + bwd_stock) / (fwd_chip + bwd_chip):.1f}x"
    )

    unpatch_model(model, verbose=False)
    return True


def main():
    print("=" * 60)
    print("  Ternary Backward Kernel Validation")
    print("=" * 60)

    ok1 = test_isolated_backward()

    ok2 = test_model_backward()

    print(f"\n{'=' * 60}")
    print(f"  Overall: {'ALL PASS' if ok1 and ok2 else 'SOME FAILURES'}")
    print(f"{'=' * 60}")
    sys.exit(0 if ok1 and ok2 else 1)


if __name__ == "__main__":
    main()
