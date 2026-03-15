import torch
import ctypes
import numpy as np
import time
import os

# Load the ghost kernel
lib = ctypes.CDLL("./softchip/ghost_matmul.so")

# Forward: void ghost_matmul_forward(const float* input, float* output, int M, int K, int N, uint64_t base_seed, int layer_id)
lib.ghost_matmul_forward.argtypes = [
    ctypes.POINTER(ctypes.c_float),
    ctypes.POINTER(ctypes.c_float),
    ctypes.c_int,
    ctypes.c_int,
    ctypes.c_int,
    ctypes.c_uint64,
    ctypes.c_int,
]

# Backward: void ghost_matmul_backward(const float* grad_output, float* grad_input, int M, int K, int N, uint64_t base_seed, int layer_id)
lib.ghost_matmul_backward.argtypes = [
    ctypes.POINTER(ctypes.c_float),
    ctypes.POINTER(ctypes.c_float),
    ctypes.c_int,
    ctypes.c_int,
    ctypes.c_int,
    ctypes.c_uint64,
    ctypes.c_int,
]


def test_ghost_kernel():
    M, K, N = 1, 2560, 2560
    seed = 42
    layer_id = 0

    # Setup inputs
    input_tensor = torch.randn(M, K, dtype=torch.float32)
    output_tensor = torch.zeros(M, N, dtype=torch.float32)

    input_ptr = input_tensor.data_ptr()
    output_ptr = output_tensor.data_ptr()

    print(f"Testing Ghost Matmul Forward (M={M}, K={K}, N={N})...")

    # Warmup
    lib.ghost_matmul_forward(
        ctypes.cast(input_ptr, ctypes.POINTER(ctypes.c_float)),
        ctypes.cast(output_ptr, ctypes.POINTER(ctypes.c_float)),
        M,
        K,
        N,
        seed,
        layer_id,
    )

    # Time it
    t0 = time.time()
    iters = 10
    for _ in range(iters):
        lib.ghost_matmul_forward(
            ctypes.cast(input_ptr, ctypes.POINTER(ctypes.c_float)),
            ctypes.cast(output_ptr, ctypes.POINTER(ctypes.c_float)),
            M,
            K,
            N,
            seed,
            layer_id,
        )
    dt = (time.time() - t0) / iters
    print(f"  Forward time: {dt * 1000:.2f} ms per layer")

    # Check deterministic behavior
    out1 = output_tensor.clone()
    lib.ghost_matmul_forward(
        ctypes.cast(input_ptr, ctypes.POINTER(ctypes.c_float)),
        ctypes.cast(output_ptr, ctypes.POINTER(ctypes.c_float)),
        M,
        K,
        N,
        seed,
        layer_id,
    )
    out2 = output_tensor.clone()
    print(f"  Deterministic: {torch.allclose(out1, out2)}")

    # Distribution Check
    # To check distribution, we pass a unit vector as input and see the outputs.
    # If input is [1, 0, 0...], output[n] = W[n, 0] * 1.0.
    # However, our kernel handles K in blocks. Let's try input of all 1.0.
    # output[n] = sum_k W[n, k].
    # Expected value of W[n, k] = 0.5*0 + 0.25*1 + 0.25*(-1) = 0.
    # Variance of W[n, k] = 0.5*(0^2) + 0.25*(1^2) + 0.25*((-1)^2) = 0.5.
    # For K=2560, sum_k W[n, k] should be ~ Normal(0, sqrt(2560 * 0.5)) = Normal(0, 35.7).

    input_ones = torch.ones(M, K, dtype=torch.float32)
    lib.ghost_matmul_forward(
        ctypes.cast(input_ones.data_ptr(), ctypes.POINTER(ctypes.c_float)),
        ctypes.cast(output_ptr, ctypes.POINTER(ctypes.c_float)),
        M,
        K,
        N,
        seed,
        layer_id,
    )

    mean = output_tensor.mean().item()
    std = output_tensor.std().item()
    print(f"  Distribution (sum of row): mean={mean:.2f}, std={std:.2f}")
    print(f"  Expected std: ~35.7")

    # Backward Test
    print(f"Testing Ghost Matmul Backward...")
    grad_output = torch.randn(M, N, dtype=torch.float32)
    grad_input = torch.zeros(M, K, dtype=torch.float32)

    t0 = time.time()
    lib.ghost_matmul_backward(
        ctypes.cast(grad_output.data_ptr(), ctypes.POINTER(ctypes.c_float)),
        ctypes.cast(grad_input.data_ptr(), ctypes.POINTER(ctypes.c_float)),
        M,
        K,
        N,
        seed,
        layer_id,
    )
    dt_back = time.time() - t0
    print(f"  Backward time: {dt_back * 1000:.2f} ms")
    print(
        f"  Grad input non-zero: {not torch.allclose(grad_input, torch.zeros_like(grad_input))}"
    )


if __name__ == "__main__":
    test_ghost_kernel()
