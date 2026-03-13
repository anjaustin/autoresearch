#!/usr/bin/env python3
"""
Numerical validation: soft-chip kernel vs PyTorch AutoBitLinear.

Loads a real AutoBitLinear layer from BitNet b1.58, runs the same input
through both PyTorch and our C kernel, and compares outputs element-by-element.

This is the hard gate for PyTorch integration -- if outputs don't match,
the soft-chip can't be used as a drop-in replacement.
"""

import ctypes
import sys
import time
import numpy as np
import torch

MODEL_PATH = "models/bitnet-b1.58-2B-4T-bf16"
SOFTCHIP_LIB = "softchip/ternary_matmul_v3.so"


# -----------------------------------------------------------------------
# Load C library
# -----------------------------------------------------------------------
def load_softchip():
    lib = ctypes.CDLL(SOFTCHIP_LIB)

    # uint8_t *pack_weights(const float *weights, int out_features, int in_features, float *out_weight_scale)
    lib.pack_weights.restype = ctypes.POINTER(ctypes.c_uint8)
    lib.pack_weights.argtypes = [
        ctypes.POINTER(ctypes.c_float),
        ctypes.c_int,
        ctypes.c_int,
        ctypes.POINTER(ctypes.c_float),
    ]

    # void ternary_matmul(const uint8_t *packed_w, const float *activation,
    #                     float *output, int batch, int out_features,
    #                     int in_features, float weight_scale)
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

    return lib


def run_softchip(lib, weights_f32, activation_f32, out_features, in_features, batch):
    """Run the C kernel on numpy arrays, return numpy output."""
    # Pack weights
    w_flat = np.ascontiguousarray(weights_f32.flatten(), dtype=np.float32)
    w_ptr = w_flat.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
    weight_scale = ctypes.c_float(0.0)
    packed = lib.pack_weights(
        w_ptr, out_features, in_features, ctypes.byref(weight_scale)
    )

    # Prepare activation
    act_flat = np.ascontiguousarray(activation_f32.flatten(), dtype=np.float32)
    act_ptr = act_flat.ctypes.data_as(ctypes.POINTER(ctypes.c_float))

    # Prepare output
    output = np.zeros(batch * out_features, dtype=np.float32)
    out_ptr = output.ctypes.data_as(ctypes.POINTER(ctypes.c_float))

    # Run
    lib.ternary_matmul(
        packed, act_ptr, out_ptr, batch, out_features, in_features, weight_scale.value
    )

    # Free packed weights (C allocated)
    libc = ctypes.CDLL("libc.so.6")
    libc.free(packed)

    return output.reshape(batch, out_features), weight_scale.value


def replicate_bitnet_forward(weight_bf16, activation_f32):
    """
    Replicate what AutoBitLinear does during forward pass:
    1. WeightQuant: quantize BF16 weights to ternary via absmean
    2. ActQuant: quantize FP32 activations to INT8 via per-token absmax
    3. Matrix multiply quantized_activation × quantized_weight^T
    4. Scale by weight_scale × act_scale
    """
    # Step 1: Weight quantization (absmean -> ternary)
    # BitNet WeightQuant: scale = mean(|W|), W_q = round(W / scale), clamp to [-1, 1]
    w = weight_bf16.float()
    mean_abs = w.abs().mean()
    w_scaled = w / mean_abs.clamp(min=1e-5)
    w_ternary = w_scaled.round().clamp(-1, 1)  # {-1, 0, +1}

    # Step 2: Activation quantization (per-token symmetric INT8)
    # BitNet ActQuant: scale = 127 / max(|x|), x_q = round(x * scale), clamp to [-128, 127]
    x = activation_f32
    x_absmax = x.abs().max(dim=-1, keepdim=True).values.clamp(min=1e-5)
    act_scale = 127.0 / x_absmax
    x_q = (x * act_scale).round().clamp(-128, 127)

    # Step 3: Integer matmul (ternary × int8)
    # In practice BitNet does: output = x_q @ w_ternary.T
    output = x_q @ w_ternary.T

    # Step 4: Rescale
    # output = output * mean_abs / act_scale
    # (mean_abs is weight scale, 1/act_scale is activation dequant)
    output = output * mean_abs / act_scale

    return output, mean_abs.item()


# -----------------------------------------------------------------------
# Main test
# -----------------------------------------------------------------------
def main():
    print("=" * 60)
    print("Numerical Validation: Soft-Chip vs AutoBitLinear")
    print("=" * 60)

    # Load library
    print("\n[1] Loading soft-chip library...")
    try:
        lib = load_softchip()
        print("    OK")
    except OSError as e:
        print(f"    FAIL: {e}")
        print(
            "    Build with: gcc -O3 -mavx2 -mfma -march=native -fopenmp -shared -fPIC "
            f"-o {SOFTCHIP_LIB} softchip/ternary_matmul_v3.c -lm"
        )
        sys.exit(1)

    # Load model (just one layer to save time)
    print("\n[2] Loading BitNet model...")
    t0 = time.time()
    from transformers import AutoModelForCausalLM

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH,
        torch_dtype=torch.bfloat16,
        device_map="cpu",
        trust_remote_code=False,
    )
    print(f"    Loaded in {time.time() - t0:.1f}s")

    # Test multiple layer types
    test_layers = [
        ("layer 0, q_proj", model.model.layers[0].self_attn.q_proj),
        ("layer 0, k_proj", model.model.layers[0].self_attn.k_proj),
        ("layer 0, gate_proj", model.model.layers[0].mlp.gate_proj),
        ("layer 0, down_proj", model.model.layers[0].mlp.down_proj),
        ("layer 15, q_proj", model.model.layers[15].self_attn.q_proj),
        ("layer 29, o_proj", model.model.layers[29].self_attn.o_proj),
    ]

    all_pass = True

    for name, layer in test_layers:
        print(f"\n[Test] {name}")
        weight = layer.weight.data  # BF16
        out_features, in_features = weight.shape
        print(f"    Shape: ({out_features}, {in_features})")

        # Create random activation (same for both paths)
        batch = 19
        torch.manual_seed(42)
        activation = torch.randn(batch, in_features, dtype=torch.float32)

        # --- Path A: Manual BitNet replication (our understanding) ---
        out_manual, wscale_manual = replicate_bitnet_forward(weight, activation)

        # --- Path B: Soft-chip C kernel ---
        w_f32 = weight.float().numpy()
        act_f32 = activation.numpy()
        out_softchip, wscale_softchip = run_softchip(
            lib, w_f32, act_f32, out_features, in_features, batch
        )
        out_softchip = torch.from_numpy(out_softchip)

        # --- Compare ---
        diff = (out_manual - out_softchip).abs()
        max_diff = diff.max().item()
        mean_diff = diff.mean().item()

        # Use NRMSE (normalized root mean squared error) relative to output range
        # This avoids the near-zero division problem of element-wise relative error
        output_range = (out_manual.max() - out_manual.min()).item()
        rmse = diff.pow(2).mean().sqrt().item()
        nrmse = rmse / max(output_range, 1e-8)

        # Cosine similarity (1.0 = perfect match)
        cos_sim = torch.nn.functional.cosine_similarity(
            out_manual.flatten().unsqueeze(0), out_softchip.flatten().unsqueeze(0)
        ).item()

        # Also compare weight scales
        wscale_diff = abs(wscale_manual - wscale_softchip) / max(
            abs(wscale_manual), 1e-8
        )

        pass_abs = max_diff < 1.0  # Max element diff < 1.0
        pass_nrmse = nrmse < 0.001  # NRMSE < 0.1%
        pass_cos = cos_sim > 0.99999  # Cosine sim > 0.99999
        pass_wscale = wscale_diff < 0.01

        status = (
            "PASS" if (pass_abs and pass_nrmse and pass_cos and pass_wscale) else "FAIL"
        )
        if status == "FAIL":
            all_pass = False

        print(
            f"    Weight scale: manual={wscale_manual:.6f}, softchip={wscale_softchip:.6f}, "
            f"diff={wscale_diff:.2e} {'OK' if pass_wscale else 'MISMATCH'}"
        )
        print(f"    Max abs diff:  {max_diff:.6f} {'OK' if pass_abs else 'MISMATCH'}")
        print(f"    Mean abs diff: {mean_diff:.6f}")
        print(f"    NRMSE:         {nrmse:.2e} {'OK' if pass_nrmse else 'MISMATCH'}")
        print(f"    Cosine sim:    {cos_sim:.8f} {'OK' if pass_cos else 'MISMATCH'}")
        print(f"    Manual  out[0][:5] = {out_manual[0, :5].tolist()}")
        print(f"    Softchip out[0][:5] = {out_softchip[0, :5].tolist()}")
        print(f"    --> {status}")

    # --- Also test against actual AutoBitLinear forward ---
    print(f"\n{'=' * 60}")
    print("Cross-check: soft-chip vs actual AutoBitLinear.forward()")
    print("=" * 60)

    layer = model.model.layers[0].self_attn.q_proj
    weight = layer.weight.data
    out_features, in_features = weight.shape

    torch.manual_seed(42)
    activation = torch.randn(1, 19, in_features, dtype=torch.bfloat16)

    # Actual AutoBitLinear forward
    with torch.no_grad():
        out_actual = layer(activation).float()
    out_actual = out_actual.squeeze(0)  # (19, out_features)

    # Soft-chip forward
    w_f32 = weight.float().numpy()
    act_f32 = activation.squeeze(0).float().numpy()
    out_sc, wsc = run_softchip(lib, w_f32, act_f32, out_features, in_features, 19)
    out_sc = torch.from_numpy(out_sc)

    diff = (out_actual - out_sc).abs()
    max_diff = diff.max().item()
    mean_diff = diff.mean().item()
    rel_diff = (diff / (out_actual.abs() + 1e-8)).mean().item()

    print(f"    Max abs diff:  {max_diff:.4f}")
    print(f"    Mean abs diff: {mean_diff:.4f}")
    print(f"    Mean rel diff: {rel_diff:.6f}")
    print(f"    Actual   out[0][:5] = {out_actual[0, :5].tolist()}")
    print(f"    Softchip out[0][:5] = {out_sc[0, :5].tolist()}")

    if max_diff < 2.0 and rel_diff < 0.2:
        print("    --> PASS (within expected tolerance)")
    else:
        print("    --> FAIL (divergence too large)")
        all_pass = False

    # --- Summary ---
    print(f"\n{'=' * 60}")
    if all_pass:
        print("OVERALL: PASS -- soft-chip kernel matches AutoBitLinear")
    else:
        print("OVERALL: FAIL -- numerical divergence detected")
    print("=" * 60)

    # Clean up
    del model
    return 0 if all_pass else 1


if __name__ == "__main__":
    sys.exit(main())
