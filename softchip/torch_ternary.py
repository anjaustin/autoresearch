"""
PyTorch integration for the soft-chip ternary matmul kernel.

Provides a drop-in replacement for AutoBitLinear's forward pass using
either the AVX2 CPU kernel or the Vulkan iGPU kernel. The backward pass
is untouched (falls back to PyTorch's STE path through BF16 master weights).

Backend selection:
    - "vulkan": Use Vega 7 iGPU via Vulkan compute (fastest for M=1)
    - "cpu": Use AVX2 kernel (fallback)
    - "auto": Try Vulkan first, fall back to CPU

Usage:
    from softchip.torch_ternary import patch_model, unpatch_model

    model = AutoModelForCausalLM.from_pretrained(...)
    patch_model(model, backend="auto")  # Vulkan if available, else CPU
    output = model(input)               # Uses soft-chip for forward
    unpatch_model(model)                # Restore original forward
"""

import ctypes
import os
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

# -----------------------------------------------------------------------
# Load the CPU C shared library
# -----------------------------------------------------------------------
_LIB_DIR = Path(__file__).parent
_LIB_PATH = _LIB_DIR / "ternary_matmul_v3.so"

_lib = None


def _ensure_lib():
    global _lib
    if _lib is not None:
        return _lib

    if not _LIB_PATH.exists():
        raise RuntimeError(
            f"Soft-chip library not found at {_LIB_PATH}. Build with:\n"
            f"  gcc -O3 -mavx2 -mfma -march=native -fopenmp -shared -fPIC "
            f"-o {_LIB_PATH} {_LIB_DIR}/ternary_matmul_v3.c -lm"
        )

    _lib = ctypes.CDLL(str(_LIB_PATH))

    _lib.pack_weights.restype = ctypes.POINTER(ctypes.c_uint8)
    _lib.pack_weights.argtypes = [
        ctypes.POINTER(ctypes.c_float),
        ctypes.c_int,
        ctypes.c_int,
        ctypes.POINTER(ctypes.c_float),
    ]

    _lib.ternary_matmul.restype = None
    _lib.ternary_matmul.argtypes = [
        ctypes.POINTER(ctypes.c_uint8),
        ctypes.POINTER(ctypes.c_float),
        ctypes.POINTER(ctypes.c_float),
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_float,
    ]

    _lib.ternary_matmul_backward.restype = None
    _lib.ternary_matmul_backward.argtypes = [
        ctypes.POINTER(ctypes.c_uint8),
        ctypes.POINTER(ctypes.c_float),
        ctypes.POINTER(ctypes.c_float),
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_float,
    ]

    return _lib


# -----------------------------------------------------------------------
# Load the Vulkan shared library
# -----------------------------------------------------------------------
_VK_LIB_PATH = _LIB_DIR / "libvk_ternary.so"
_SPV_PATH = _LIB_DIR / "ternary_matmul_v3.spv"

_vk_lib = None
_vk_available = None  # None = not checked yet, True/False after check


def _try_load_vulkan():
    """Try to load and initialize the Vulkan backend. Returns True on success."""
    global _vk_lib, _vk_available

    if _vk_available is not None:
        return _vk_available

    if not _VK_LIB_PATH.exists():
        _vk_available = False
        return False

    if not _SPV_PATH.exists():
        _vk_available = False
        return False

    try:
        _vk_lib = ctypes.CDLL(str(_VK_LIB_PATH))

        # Set up function signatures
        _vk_lib.vk_init.restype = ctypes.c_int
        _vk_lib.vk_init.argtypes = [ctypes.c_char_p]

        _vk_lib.vk_device_name.restype = ctypes.c_char_p
        _vk_lib.vk_device_name.argtypes = []

        _vk_lib.vk_alloc_layer.restype = ctypes.c_int
        _vk_lib.vk_alloc_layer.argtypes = [
            ctypes.POINTER(ctypes.c_uint32),
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_float,
        ]

        _vk_lib.vk_dispatch.restype = ctypes.c_int
        _vk_lib.vk_dispatch.argtypes = [
            ctypes.c_int,
            ctypes.POINTER(ctypes.c_float),
            ctypes.POINTER(ctypes.c_float),
        ]

        _vk_lib.vk_dispatch_batch.restype = ctypes.c_int
        _vk_lib.vk_dispatch_batch.argtypes = [
            ctypes.POINTER(ctypes.c_int),
            ctypes.c_int,
            ctypes.POINTER(ctypes.c_float),
            ctypes.POINTER(ctypes.POINTER(ctypes.c_float)),
        ]

        _vk_lib.vk_finalize_layers.restype = ctypes.c_int
        _vk_lib.vk_finalize_layers.argtypes = []

        _vk_lib.vk_shutdown.restype = None
        _vk_lib.vk_shutdown.argtypes = []

        # Initialize Vulkan
        rc = _vk_lib.vk_init(str(_SPV_PATH).encode())
        if rc != 0:
            _vk_available = False
            return False

        _vk_available = True
        return True

    except (OSError, AttributeError) as e:
        _vk_available = False
        return False


def _vk_device_name():
    """Return the Vulkan GPU device name, or None."""
    if _vk_lib and _vk_available:
        name = _vk_lib.vk_device_name()
        return name.decode() if name else None
    return None


# -----------------------------------------------------------------------
# Packed weight cache
# -----------------------------------------------------------------------
class PackedWeight:
    """Holds a packed weight matrix and its scale for one AutoBitLinear layer."""

    __slots__ = (
        "packed_ptr",
        "weight_scale",
        "out_features",
        "in_features",
        "packed_row_bytes",
        "vk_layer_id",
    )

    def __init__(self, weight_tensor, upload_vulkan=False):
        """Pack a BF16/FP32 weight tensor using the C kernel's pack_weights."""
        lib = _ensure_lib()

        w_f32 = weight_tensor.float().contiguous()
        self.out_features, self.in_features = w_f32.shape
        self.packed_row_bytes = (self.in_features + 3) // 4
        self.vk_layer_id = -1

        w_np = w_f32.numpy()
        w_ptr = w_np.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
        scale = ctypes.c_float(0.0)
        self.packed_ptr = lib.pack_weights(
            w_ptr, self.out_features, self.in_features, ctypes.byref(scale)
        )
        self.weight_scale = scale.value

        # Upload to Vulkan GPU if requested
        if upload_vulkan and _vk_available and _vk_lib:
            packed_row_uints = (self.in_features + 15) // 16
            total_uints = self.out_features * packed_row_uints
            # The CPU pack_weights returns uint8 pointer (4 weights per byte)
            # Vulkan needs uint32 pointer (16 weights per uint32) — same data
            packed_u32_ptr = ctypes.cast(
                self.packed_ptr, ctypes.POINTER(ctypes.c_uint32)
            )
            self.vk_layer_id = _vk_lib.vk_alloc_layer(
                packed_u32_ptr,
                self.out_features,
                self.in_features,
                self.weight_scale,
            )

    def __del__(self):
        """Free the C-allocated packed weights."""
        if hasattr(self, "packed_ptr") and self.packed_ptr:
            try:
                libc = ctypes.CDLL("libc.so.6")
                libc.free(self.packed_ptr)
            except Exception:
                pass


# -----------------------------------------------------------------------
# Custom autograd functions
# -----------------------------------------------------------------------
class TernaryMatmulFunction(torch.autograd.Function):
    """
    Forward: use soft-chip C kernel (CPU, fast, ternary-optimized)
    Backward: use ternary backward kernel (STE: grad_input = W^T @ grad_output)

    No activation quantization in backward -- gradients pass through unchanged
    per the Straight-Through Estimator. The ternary weight structure means
    backward is also just add/subtract/skip, same as forward.
    """

    @staticmethod
    def forward(ctx, input_tensor, packed_weight, original_layer):
        lib = _ensure_lib()
        pw = packed_weight

        orig_shape = input_tensor.shape
        x_2d = input_tensor.reshape(-1, pw.in_features)
        batch = x_2d.shape[0]

        x_f32 = x_2d.float().contiguous().numpy()
        x_ptr = x_f32.ctypes.data_as(ctypes.POINTER(ctypes.c_float))

        out_f32 = np.empty((batch, pw.out_features), dtype=np.float32)
        out_ptr = out_f32.ctypes.data_as(ctypes.POINTER(ctypes.c_float))

        lib.ternary_matmul(
            pw.packed_ptr,
            x_ptr,
            out_ptr,
            batch,
            pw.out_features,
            pw.in_features,
            pw.weight_scale,
        )

        output = torch.from_numpy(out_f32).to(
            device=input_tensor.device, dtype=input_tensor.dtype
        )
        output = output.reshape(*orig_shape[:-1], pw.out_features)

        # Save packed weight info for backward (not input_tensor — not needed)
        ctx.packed_weight = packed_weight
        ctx.orig_shape = orig_shape
        return output

    @staticmethod
    def backward(ctx, grad_output):
        lib = _ensure_lib()
        pw = ctx.packed_weight

        # grad_output: (*orig_shape[:-1], out_features)
        go_2d = grad_output.reshape(-1, pw.out_features)
        batch = go_2d.shape[0]

        go_f32 = go_2d.float().contiguous().numpy()
        go_ptr = go_f32.ctypes.data_as(ctypes.POINTER(ctypes.c_float))

        gi_f32 = np.empty((batch, pw.in_features), dtype=np.float32)
        gi_ptr = gi_f32.ctypes.data_as(ctypes.POINTER(ctypes.c_float))

        lib.ternary_matmul_backward(
            pw.packed_ptr,
            go_ptr,
            gi_ptr,
            batch,
            pw.out_features,
            pw.in_features,
            pw.weight_scale,
        )

        grad_input = torch.from_numpy(gi_f32).to(
            device=grad_output.device, dtype=grad_output.dtype
        )
        grad_input = grad_input.reshape(*ctx.orig_shape)

        return grad_input, None, None


class VulkanMatmulFunction(torch.autograd.Function):
    """
    Forward: use Vulkan iGPU compute shader (fastest for M=1)
    Backward: fall back to PyTorch (STE through BF16 master weights)
    """

    @staticmethod
    def forward(ctx, input_tensor, packed_weight, original_layer):
        pw = packed_weight

        orig_shape = input_tensor.shape
        x_2d = input_tensor.reshape(-1, pw.in_features)
        batch = x_2d.shape[0]

        # Vulkan dispatch is M=1 only; loop for batch > 1
        x_f32 = x_2d.float().contiguous().numpy()
        out_f32 = np.empty((batch, pw.out_features), dtype=np.float32)

        for m in range(batch):
            act_ptr = x_f32[m : m + 1].ctypes.data_as(ctypes.POINTER(ctypes.c_float))
            out_ptr = out_f32[m : m + 1].ctypes.data_as(ctypes.POINTER(ctypes.c_float))
            rc = _vk_lib.vk_dispatch(pw.vk_layer_id, act_ptr, out_ptr)
            if rc != 0:
                raise RuntimeError(f"Vulkan dispatch failed for layer {pw.vk_layer_id}")

        output = torch.from_numpy(out_f32).to(
            device=input_tensor.device, dtype=input_tensor.dtype
        )
        output = output.reshape(*orig_shape[:-1], pw.out_features)

        ctx.save_for_backward(input_tensor)
        ctx.original_layer = original_layer
        return output

    @staticmethod
    def backward(ctx, grad_output):
        (input_tensor,) = ctx.saved_tensors
        original_layer = ctx.original_layer

        with torch.enable_grad():
            input_detached = input_tensor.detach().requires_grad_(True)
            output = original_layer(input_detached)
            output.backward(grad_output)

        return input_detached.grad, None, None


# -----------------------------------------------------------------------
# Model patching
# -----------------------------------------------------------------------
_PATCH_ATTR = "_softchip_original_forward"
_PACKED_ATTR = "_softchip_packed_weight"
_BACKEND_ATTR = "_softchip_backend"


def patch_model(model, backend="auto", verbose=True):
    """
    Replace all AutoBitLinear forward passes with soft-chip kernel.

    Args:
        model: a HuggingFace model with AutoBitLinear layers
        backend: "vulkan", "cpu", or "auto" (try vulkan, fall back to cpu)
        verbose: print progress

    Returns:
        Number of layers patched
    """
    _ensure_lib()

    # Determine backend
    use_vulkan = False
    if backend == "vulkan":
        if not _try_load_vulkan():
            raise RuntimeError(
                "Vulkan backend requested but not available. "
                "Build with: gcc -O2 -shared -fPIC -o softchip/libvk_ternary.so "
                "softchip/vk_backend.c -lvulkan -lm -ldl"
            )
        use_vulkan = True
    elif backend == "auto":
        use_vulkan = _try_load_vulkan()
    elif backend != "cpu":
        raise ValueError(
            f"Unknown backend: {backend!r}. Use 'vulkan', 'cpu', or 'auto'"
        )

    if verbose and use_vulkan:
        dev = _vk_device_name() or "unknown"
        print(f"Soft-chip: using Vulkan backend ({dev})")
    elif verbose:
        print("Soft-chip: using CPU (AVX2) backend")

    count = 0
    vk_count = 0
    total_pack_time = 0.0

    for name, module in model.named_modules():
        class_name = module.__class__.__name__
        if class_name != "AutoBitLinear":
            continue

        # Pack weights (and upload to GPU if using Vulkan)
        t0 = time.time()
        packed = PackedWeight(module.weight.data, upload_vulkan=use_vulkan)
        pack_time = time.time() - t0
        total_pack_time += pack_time

        # Determine per-layer backend
        layer_use_vulkan = use_vulkan and packed.vk_layer_id >= 0

        # Store packed weight, original forward, and backend on the module
        setattr(module, _PACKED_ATTR, packed)
        setattr(module, _PATCH_ATTR, module.forward)
        setattr(module, _BACKEND_ATTR, "vulkan" if layer_use_vulkan else "cpu")

        # Create patched forward
        def make_forward(mod, pw, is_vulkan):
            def patched_forward(x):
                if is_vulkan:
                    return VulkanMatmulFunction.apply(x, pw, mod)
                else:
                    return TernaryMatmulFunction.apply(x, pw, mod)

            return patched_forward

        module.forward = make_forward(module, packed, layer_use_vulkan)
        count += 1
        if layer_use_vulkan:
            vk_count += 1

    # Finalize Vulkan: pre-allocate descriptor sets for all uploaded layers
    if use_vulkan and vk_count > 0 and _vk_lib:
        rc = _vk_lib.vk_finalize_layers()
        if rc != 0 and verbose:
            print("  Warning: vk_finalize_layers() failed, using legacy path")

    if verbose:
        backend_str = (
            f"{vk_count} Vulkan + {count - vk_count} CPU" if vk_count else "CPU"
        )
        print(
            f"Soft-chip: patched {count} AutoBitLinear layers "
            f"[{backend_str}] (weight packing: {total_pack_time:.1f}s)"
        )

    return count


def unpatch_model(model, verbose=True):
    """Restore original AutoBitLinear forward passes."""
    global _vk_available

    count = 0
    for name, module in model.named_modules():
        if hasattr(module, _PATCH_ATTR):
            module.forward = getattr(module, _PATCH_ATTR)
            delattr(module, _PATCH_ATTR)
            if hasattr(module, _PACKED_ATTR):
                delattr(module, _PACKED_ATTR)
            if hasattr(module, _BACKEND_ATTR):
                delattr(module, _BACKEND_ATTR)
            count += 1

    # Shut down Vulkan if it was initialized
    if _vk_available and _vk_lib:
        _vk_lib.vk_shutdown()
        _vk_available = None

    if verbose:
        print(f"Soft-chip: unpatched {count} layers")

    return count
