import torch
import ctypes
import numpy as np
import time

ghost_lib = ctypes.CDLL("./softchip/ghost_matmul_lut.so")
mtp_lib = ctypes.CDLL("./softchip/mtp18_matmul.so")

ghost_lib.ghost_matmul_forward_lut.argtypes = [
    ctypes.POINTER(ctypes.c_float),
    ctypes.POINTER(ctypes.c_float),
    ctypes.c_int,
    ctypes.c_int,
    ctypes.c_int,
    ctypes.c_uint64,
    ctypes.c_int,
]
mtp_lib.mtp18_matmul_forward.argtypes = [
    ctypes.POINTER(ctypes.c_float),
    ctypes.POINTER(ctypes.c_float),
    ctypes.c_int,
    ctypes.c_int,
    ctypes.c_int,
    ctypes.c_uint64,
    ctypes.c_int,
]


def run_kernel(lib, name, M, K, N, seed, layer_id):
    inp = torch.randn(M, K, dtype=torch.float32)
    out = torch.zeros(M, N, dtype=torch.float32)
    func_name = (
        "ghost_matmul_forward_lut" if name == "Ghost" else "mtp18_matmul_forward"
    )
    getattr(lib, func_name)(
        ctypes.cast(inp.data_ptr(), ctypes.POINTER(ctypes.c_float)),
        ctypes.cast(out.data_ptr(), ctypes.POINTER(ctypes.c_float)),
        M,
        K,
        N,
        seed,
        layer_id,
    )
    return out


print("=== MTP18 vs GhostWeight LUT Kernel Comparison ===\n")

M, K, N = 1, 2560, 2560
seed, lid = 42, 0

for lib, name in [(ghost_lib, "Ghost"), (mtp_lib, "MTP18")]:
    t0 = time.time()
    for _ in range(5):
        out = run_kernel(lib, name, M, K, N, seed, lid)
    dt = (time.time() - t0) / 5
    print(f"{name}: {dt * 1000:.2f} ms/layer")

out_ghost = run_kernel(ghost_lib, "Ghost", M, K, N, seed, lid)
out_mtp = run_kernel(mtp_lib, "MTP18", M, K, N, seed, lid)

diff = (out_ghost - out_mtp).abs()
print(f"\nDifference: max={diff.max().item():.4f}, mean={diff.mean().item():.4f}")
print(f"Ghost stats: mean={out_ghost.mean():.4f}, std={out_ghost.std():.4f}")
print(f"MTP18 stats: mean={out_mtp.mean():.4f}, std={out_mtp.std():.4f}")
