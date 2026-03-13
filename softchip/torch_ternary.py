"""
PyTorch integration for the soft-chip ternary matmul kernel.

Provides a drop-in replacement for AutoBitLinear's forward pass using
our AVX2 kernel. The backward pass is untouched (falls back to PyTorch's
STE path through the BF16 master weights).

Usage:
    from softchip.torch_ternary import patch_model, unpatch_model

    model = AutoModelForCausalLM.from_pretrained(...)
    patch_model(model)       # Replace AutoBitLinear forward with soft-chip
    output = model(input)    # Uses soft-chip for forward, PyTorch for backward
    unpatch_model(model)     # Restore original forward
"""

import ctypes
import os
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

# -----------------------------------------------------------------------
# Load the C shared library
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

    return _lib


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
    )

    def __init__(self, weight_tensor):
        """Pack a BF16/FP32 weight tensor using the C kernel's pack_weights."""
        lib = _ensure_lib()

        w_f32 = weight_tensor.float().contiguous()
        self.out_features, self.in_features = w_f32.shape
        self.packed_row_bytes = (self.in_features + 3) // 4

        w_np = w_f32.numpy()
        w_ptr = w_np.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
        scale = ctypes.c_float(0.0)
        self.packed_ptr = lib.pack_weights(
            w_ptr, self.out_features, self.in_features, ctypes.byref(scale)
        )
        self.weight_scale = scale.value

    def __del__(self):
        """Free the C-allocated packed weights."""
        if hasattr(self, "packed_ptr") and self.packed_ptr:
            try:
                libc = ctypes.CDLL("libc.so.6")
                libc.free(self.packed_ptr)
            except Exception:
                pass


# -----------------------------------------------------------------------
# Custom autograd function
# -----------------------------------------------------------------------
class TernaryMatmulFunction(torch.autograd.Function):
    """
    Forward: use soft-chip C kernel (fast, ternary-optimized)
    Backward: fall back to PyTorch (STE through BF16 master weights)

    The forward replaces the quantized matmul that AutoBitLinear does.
    The backward must go through the original layer to get correct STE gradients.
    """

    @staticmethod
    def forward(ctx, input_tensor, packed_weight, original_layer):
        """
        Args:
            input_tensor: (*, in_features) activation tensor
            packed_weight: PackedWeight instance
            original_layer: the original AutoBitLinear module (for backward)
        """
        lib = _ensure_lib()
        pw = packed_weight

        # Flatten input to 2D: (batch, in_features)
        orig_shape = input_tensor.shape
        x_2d = input_tensor.reshape(-1, pw.in_features)
        batch = x_2d.shape[0]

        # Convert to FP32 numpy for C kernel
        x_f32 = x_2d.float().contiguous().numpy()
        x_ptr = x_f32.ctypes.data_as(ctypes.POINTER(ctypes.c_float))

        # Allocate output
        out_f32 = np.empty((batch, pw.out_features), dtype=np.float32)
        out_ptr = out_f32.ctypes.data_as(ctypes.POINTER(ctypes.c_float))

        # Run kernel
        lib.ternary_matmul(
            pw.packed_ptr,
            x_ptr,
            out_ptr,
            batch,
            pw.out_features,
            pw.in_features,
            pw.weight_scale,
        )

        # Convert back to tensor, matching input dtype
        output = torch.from_numpy(out_f32).to(
            device=input_tensor.device, dtype=input_tensor.dtype
        )
        output = output.reshape(*orig_shape[:-1], pw.out_features)

        # Save for backward
        ctx.save_for_backward(input_tensor)
        ctx.original_layer = original_layer

        return output

    @staticmethod
    def backward(ctx, grad_output):
        """
        Fall back to PyTorch for backward (STE requires BF16 master weights).

        We recompute the forward through the original layer to build the
        autograd graph, then backprop through that.
        """
        (input_tensor,) = ctx.saved_tensors
        original_layer = ctx.original_layer

        # Recompute forward through original layer (builds autograd graph)
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


def patch_model(model, verbose=True):
    """
    Replace all AutoBitLinear forward passes with soft-chip kernel.

    Args:
        model: a HuggingFace model with AutoBitLinear layers
        verbose: print progress

    Returns:
        Number of layers patched
    """
    _ensure_lib()

    count = 0
    total_pack_time = 0.0

    for name, module in model.named_modules():
        class_name = module.__class__.__name__
        if class_name != "AutoBitLinear":
            continue

        # Pack weights
        t0 = time.time()
        packed = PackedWeight(module.weight.data)
        pack_time = time.time() - t0
        total_pack_time += pack_time

        # Store packed weight and original forward on the module
        setattr(module, _PACKED_ATTR, packed)
        setattr(module, _PATCH_ATTR, module.forward)

        # Create patched forward
        def make_forward(mod, pw):
            original_forward = getattr(mod, _PATCH_ATTR)

            def patched_forward(x):
                if x.requires_grad:
                    # Training mode: use custom autograd function
                    return TernaryMatmulFunction.apply(x, pw, mod)
                else:
                    # Inference mode: just use the C kernel directly
                    return TernaryMatmulFunction.apply(x, pw, mod)

            return patched_forward

        module.forward = make_forward(module, packed)
        count += 1

    if verbose:
        print(
            f"Soft-chip: patched {count} AutoBitLinear layers "
            f"(weight packing: {total_pack_time:.1f}s)"
        )

    return count


def unpatch_model(model, verbose=True):
    """Restore original AutoBitLinear forward passes."""
    count = 0
    for name, module in model.named_modules():
        if hasattr(module, _PATCH_ATTR):
            module.forward = getattr(module, _PATCH_ATTR)
            delattr(module, _PATCH_ATTR)
            if hasattr(module, _PACKED_ATTR):
                delattr(module, _PACKED_ATTR)
            count += 1

    if verbose:
        print(f"Soft-chip: unpatched {count} layers")

    return count
