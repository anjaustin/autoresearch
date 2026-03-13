#!/usr/bin/env python3
"""
Profile the Vulkan dispatch overhead to find where time is spent.

Breaks down: ctypes call, activation upload, dispatch+fence, output download.
"""

import ctypes
import time
import numpy as np
from pathlib import Path

LIB_DIR = Path(__file__).parent / "softchip"


def main():
    # Load libraries
    cpu_lib = ctypes.CDLL(str(LIB_DIR / "ternary_matmul_v3.so"))
    cpu_lib.pack_weights.restype = ctypes.POINTER(ctypes.c_uint8)
    cpu_lib.pack_weights.argtypes = [
        ctypes.POINTER(ctypes.c_float),
        ctypes.c_int,
        ctypes.c_int,
        ctypes.POINTER(ctypes.c_float),
    ]

    vk_lib = ctypes.CDLL(str(LIB_DIR / "libvk_ternary.so"))
    vk_lib.vk_init.restype = ctypes.c_int
    vk_lib.vk_init.argtypes = [ctypes.c_char_p]
    vk_lib.vk_alloc_layer.restype = ctypes.c_int
    vk_lib.vk_alloc_layer.argtypes = [
        ctypes.POINTER(ctypes.c_uint32),
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_float,
    ]
    vk_lib.vk_dispatch.restype = ctypes.c_int
    vk_lib.vk_dispatch.argtypes = [
        ctypes.c_int,
        ctypes.POINTER(ctypes.c_float),
        ctypes.POINTER(ctypes.c_float),
    ]
    vk_lib.vk_dispatch_batch.restype = ctypes.c_int
    vk_lib.vk_dispatch_batch.argtypes = [
        ctypes.POINTER(ctypes.c_int),
        ctypes.c_int,
        ctypes.POINTER(ctypes.c_float),
        ctypes.POINTER(ctypes.POINTER(ctypes.c_float)),
    ]
    vk_lib.vk_shutdown.restype = None

    spv_path = str(LIB_DIR / "ternary_matmul_v3.spv").encode()
    rc = vk_lib.vk_init(spv_path)
    assert rc == 0, f"vk_init failed: {rc}"

    # Upload a 2560x2560 layer
    N, K = 2560, 2560
    np.random.seed(42)
    weights = (np.random.rand(N, K).astype(np.float32) - 0.5) * 2.0
    w_ptr = weights.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
    scale = ctypes.c_float(0.0)
    packed_ptr = cpu_lib.pack_weights(w_ptr, N, K, ctypes.byref(scale))
    packed_u32 = ctypes.cast(packed_ptr, ctypes.POINTER(ctypes.c_uint32))
    layer_id = vk_lib.vk_alloc_layer(packed_u32, N, K, scale.value)
    assert layer_id >= 0

    # Pre-allocate numpy arrays
    activation = np.random.randn(K).astype(np.float32)
    output = np.empty(N, dtype=np.float32)

    act_ptr = activation.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
    out_ptr = output.ctypes.data_as(ctypes.POINTER(ctypes.c_float))

    # Warmup
    for _ in range(5):
        vk_lib.vk_dispatch(layer_id, act_ptr, out_ptr)

    # Benchmark 1: Raw C dispatch with pre-extracted pointers
    iters = 500
    t0 = time.time()
    for _ in range(iters):
        vk_lib.vk_dispatch(layer_id, act_ptr, out_ptr)
    raw_ms = (time.time() - t0) / iters * 1000
    print(f"Raw dispatch (pre-extracted ptrs): {raw_ms:.3f} ms")

    # Benchmark 2: Include numpy->ctypes pointer extraction overhead
    t0 = time.time()
    for _ in range(iters):
        ap = activation.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
        op = output.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
        vk_lib.vk_dispatch(layer_id, ap, op)
    extract_ms = (time.time() - t0) / iters * 1000
    print(f"With ptr extraction:              {extract_ms:.3f} ms")

    # Benchmark 3: Include torch tensor -> numpy -> ctypes (simulate actual path)
    import torch

    t_act = torch.randn(K, dtype=torch.float32)
    t_out = torch.empty(N, dtype=torch.float32)

    # Warmup
    for _ in range(5):
        a_np = t_act.numpy()
        o_np = t_out.numpy()
        ap = a_np.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
        op = o_np.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
        vk_lib.vk_dispatch(layer_id, ap, op)

    t0 = time.time()
    for _ in range(iters):
        a_np = t_act.numpy()
        o_np = t_out.numpy()
        ap = a_np.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
        op = o_np.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
        vk_lib.vk_dispatch(layer_id, ap, op)
    torch_ms = (time.time() - t0) / iters * 1000
    print(f"With torch->numpy->ctypes:        {torch_ms:.3f} ms")

    # Benchmark 4: Full VulkanMatmulFunction path simulation
    t_act_bf16 = torch.randn(1, K, dtype=torch.bfloat16)
    t0 = time.time()
    for _ in range(iters):
        x_f32 = t_act_bf16.reshape(-1, K).float().contiguous().numpy()
        out_f32 = np.empty((1, N), dtype=np.float32)
        ap = x_f32.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
        op = out_f32.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
        vk_lib.vk_dispatch(layer_id, ap, op)
        result = torch.from_numpy(out_f32).to(dtype=torch.bfloat16)
    full_ms = (time.time() - t0) / iters * 1000
    print(f"Full autograd path simulation:    {full_ms:.3f} ms")

    # Benchmark 5: Just the Python overhead (no dispatch)
    t0 = time.time()
    for _ in range(iters):
        x_f32 = t_act_bf16.reshape(-1, K).float().contiguous().numpy()
        out_f32 = np.empty((1, N), dtype=np.float32)
        ap = x_f32.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
        op = out_f32.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
        result = torch.from_numpy(out_f32).to(dtype=torch.bfloat16)
    py_ms = (time.time() - t0) / iters * 1000
    print(f"Python overhead only (no GPU):    {py_ms:.3f} ms")

    # Benchmark 6: Batch dispatch (7 layers with same weights for testing)
    ids_7 = []
    for i in range(6):
        lid = vk_lib.vk_alloc_layer(packed_u32, N, K, scale.value)
        ids_7.append(lid)
    ids_7.insert(0, layer_id)
    ids_arr = (ctypes.c_int * 7)(*ids_7)
    outs_7 = [np.empty(N, dtype=np.float32) for _ in range(7)]
    out_ptrs_7 = (ctypes.POINTER(ctypes.c_float) * 7)(
        *[o.ctypes.data_as(ctypes.POINTER(ctypes.c_float)) for o in outs_7]
    )
    # Warmup
    for _ in range(5):
        vk_lib.vk_dispatch_batch(ids_arr, 7, act_ptr, out_ptrs_7)

    t0 = time.time()
    for _ in range(iters):
        vk_lib.vk_dispatch_batch(ids_arr, 7, act_ptr, out_ptrs_7)
    batch7_ms = (time.time() - t0) / iters * 1000
    print(
        f"\nBatch dispatch (7 layers):        {batch7_ms:.3f} ms ({batch7_ms / 7:.3f} ms/layer)"
    )
    print(
        f"Individual × 7:                   {raw_ms * 7:.3f} ms ({raw_ms:.3f} ms/layer)"
    )
    print(f"Batch speedup:                    {raw_ms * 7 / batch7_ms:.2f}x")

    print(f"\n--- Breakdown ---")
    print(f"Shader compute time (2560x2560):  ~0.30 ms")
    print(f"C dispatch overhead:              {raw_ms - 0.30:.3f} ms")
    print(f"Python data conversion:           {full_ms - raw_ms:.3f} ms")
    print(f"Total per dispatch:               {full_ms:.3f} ms")
    print(f"210 dispatches (projected):       {full_ms * 210:.0f} ms")

    vk_lib.vk_shutdown()
    libc = ctypes.CDLL("libc.so.6")
    libc.free(packed_ptr)


if __name__ == "__main__":
    main()
