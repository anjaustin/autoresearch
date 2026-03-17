"""
PyTorch integration for the soft-chip ternary matmul kernel.

Provides a drop-in replacement for AutoBitLinear's forward pass using
either the AVX2 CPU kernel or the Vulkan iGPU kernel. The backward pass
uses a ternary backward kernel (STE: grad_input = W^T @ grad_output).

Also provides patch_lm_head_fp32() to fix the catastrophic BF16 GEMM
performance on CPUs without AMX/VNNI (e.g. Ryzen) by casting the LM head
weight to FP32 and using a custom autograd function.

Backend selection:
    - "cpu": Use AVX2 kernel (production path — fastest end-to-end on Ryzen)
    - "vulkan": Use Vega 7 iGPU via Vulkan compute (faster raw shader but
      submit overhead makes it slower end-to-end than CPU)
    - "auto": Try Vulkan first, fall back to CPU

Usage:
    from softchip.torch_ternary import (
        patch_model, patch_lm_head_fp32, unpatch_model, unpatch_lm_head
    )

    model = AutoModelForCausalLM.from_pretrained(...)
    patch_model(model, backend="cpu")     # Ternary soft-chip for AutoBitLinear
    patch_lm_head_fp32(model)             # FP32 for LM head (18x backward speedup on Ryzen)
    output = model(input)
    unpatch_model(model)                  # Restore everything (incl. LM head)
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
# Load the GhostWeight shared library
# -----------------------------------------------------------------------
_GHOST_LIB_PATH = _LIB_DIR / "ghost_matmul.so"
_ghost_lib = None


def _ensure_ghost_lib():
    global _ghost_lib
    if _ghost_lib is not None:
        return _ghost_lib

    if not _GHOST_LIB_PATH.exists():
        raise RuntimeError(f"GhostWeight library not found at {_GHOST_LIB_PATH}.")

    _ghost_lib = ctypes.CDLL(str(_GHOST_LIB_PATH))

    _ghost_lib.ghost_matmul_forward.argtypes = [
        ctypes.POINTER(ctypes.c_float),
        ctypes.POINTER(ctypes.c_float),
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_uint64,
        ctypes.c_int,
        ctypes.c_float,
    ]
    _ghost_lib.ghost_matmul_backward.argtypes = [
        ctypes.POINTER(ctypes.c_float),
        ctypes.POINTER(ctypes.c_float),
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_uint64,
        ctypes.c_int,
        ctypes.c_float,
    ]
    return _ghost_lib


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


class GhostMatmulFunction(torch.autograd.Function):
    """
    Forward: regenerate ternary weight on-the-fly from PRNG seed (GhostWeight).
    Backward: rerun same PRNG to regenerate W^T and compute grad_input (STE).

    No stored weight matrix — the entire layer weight is derived from:
        base_seed (uint64) + layer_id (int)
    This collapses a 500 MB weight tensor to 8 bytes.
    """

    @staticmethod
    def forward(
        ctx, input_tensor, base_seed, layer_id, in_features, out_features, weight_scale
    ):
        lib = _ensure_ghost_lib()

        orig_shape = input_tensor.shape
        x_2d = input_tensor.reshape(-1, in_features)
        batch = x_2d.shape[0]

        x_f32 = x_2d.float().contiguous().numpy()
        x_ptr = x_f32.ctypes.data_as(ctypes.POINTER(ctypes.c_float))

        out_f32 = np.empty((batch, out_features), dtype=np.float32)
        out_ptr = out_f32.ctypes.data_as(ctypes.POINTER(ctypes.c_float))

        lib.ghost_matmul_forward(
            x_ptr,
            out_ptr,
            batch,
            in_features,
            out_features,
            ctypes.c_uint64(base_seed),
            ctypes.c_int(layer_id),
            ctypes.c_float(weight_scale),
        )

        output = torch.from_numpy(out_f32).to(
            device=input_tensor.device, dtype=input_tensor.dtype
        )
        output = output.reshape(*orig_shape[:-1], out_features)

        ctx.save_for_backward(input_tensor)
        ctx.base_seed = base_seed
        ctx.layer_id = layer_id
        ctx.in_features = in_features
        ctx.out_features = out_features
        ctx.weight_scale = weight_scale
        ctx.orig_shape = orig_shape
        return output

    @staticmethod
    def backward(ctx, grad_output):
        lib = _ensure_ghost_lib()

        go_2d = grad_output.reshape(-1, ctx.out_features)
        batch = go_2d.shape[0]

        go_f32 = go_2d.float().contiguous().numpy()
        go_ptr = go_f32.ctypes.data_as(ctypes.POINTER(ctypes.c_float))

        gi_f32 = np.empty((batch, ctx.in_features), dtype=np.float32)
        gi_ptr = gi_f32.ctypes.data_as(ctypes.POINTER(ctypes.c_float))

        lib.ghost_matmul_backward(
            go_ptr,
            gi_ptr,
            batch,
            ctx.in_features,
            ctx.out_features,
            ctypes.c_uint64(ctx.base_seed),
            ctypes.c_int(ctx.layer_id),
            ctypes.c_float(ctx.weight_scale),
        )

        grad_input = torch.from_numpy(gi_f32).to(
            device=grad_output.device, dtype=grad_output.dtype
        )
        grad_input = grad_input.reshape(*ctx.orig_shape)

        # 5 inputs to forward: input_tensor, base_seed, layer_id, in_features, out_features
        return grad_input, None, None, None, None


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


def patch_model(model, backend="auto", verbose=True, use_ghost=False, ghost_seed=42, scales_path=None):
    """
    Replace all AutoBitLinear forward passes with soft-chip kernel.

    Args:
        model: a HuggingFace model with AutoBitLinear layers
        backend: "vulkan", "cpu", or "auto" (try vulkan, fall back to cpu)
        verbose: print progress
        use_ghost: if True, use GhostWeight kernel (no stored weights, PRNG-only)
        ghost_seed: uint64 base seed for GhostWeight PRNG (default 42)

    Returns:
        Number of layers patched
    """
    # GhostWeight path: skip weight packing entirely, use PRNG kernel
    if use_ghost:
        _ensure_ghost_lib()
        if verbose:
            print(f"Soft-chip: using GhostWeight backend (seed={ghost_seed})")

        # Load pre-extracted scales if available (avoids loading 4.83GB BF16)
        _ghost_scales = {}
        _scales_search = [
            scales_path,
            'models/bitnet-b1.58-2B-4T-bf16/weight_scales.pt',
        ]
        for _sp in _scales_search:
            if _sp and __import__('os').path.exists(_sp):
                import torch as _t
                _ghost_scales = _t.load(_sp, weights_only=True)
                if verbose:
                    print(f'Soft-chip: loaded {len(_ghost_scales)} pre-extracted scales from {_sp}')
                break
        if not _ghost_scales and verbose:
            print('Soft-chip: no scales file found, computing from BF16 weights (slow)')

        count = 0
        for name, module in model.named_modules():
            if module.__class__.__name__ != "AutoBitLinear":
                continue

            out_features, in_features = module.weight.shape

            # Load weight scale from pre-extracted file if available,
            # otherwise compute from BF16 weights (slow path)
            if _ghost_scales and name in _ghost_scales:
                weight_scale = _ghost_scales[name]
            else:
                weight_scale = module.weight.abs().mean().item()

            layer_id = count  # stable index across calls

            setattr(module, _PATCH_ATTR, module.forward)
            setattr(module, _BACKEND_ATTR, "ghost")

            def make_ghost_forward(lid, K, N, seed, scale):
                def patched_forward(x):
                    return GhostMatmulFunction.apply(x, seed, lid, K, N, scale)

                return patched_forward

            module.forward = make_ghost_forward(
                layer_id, in_features, out_features, ghost_seed, weight_scale
            )
            count += 1

        if verbose:
            print(f"Soft-chip: patched {count} AutoBitLinear layers [GhostWeight]")
        return count

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
    """Restore original AutoBitLinear forward passes and LM head."""
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

    # Unpatch LM head if it was patched
    unpatch_lm_head(model, verbose=False)

    # Shut down Vulkan if it was initialized
    if _vk_available and _vk_lib:
        _vk_lib.vk_shutdown()
        _vk_available = None

    if verbose:
        print(f"Soft-chip: unpatched {count} layers")

    return count


# -----------------------------------------------------------------------
# FP32 LM Head patch — fixes catastrophic BF16 GEMM on CPUs w/o AMX/VNNI
# -----------------------------------------------------------------------
_LM_HEAD_PATCH_ATTR = "_softchip_lm_head_original_forward"
_LM_HEAD_FP32_WEIGHT_ATTR = "_softchip_lm_head_fp32_weight"


class FP32LMHeadFunction(torch.autograd.Function):
    """
    Custom autograd function for the LM head that does all matmuls in FP32.

    On CPUs without AMX/VNNI (e.g. Ryzen 5 PRO 5675U), MKL's BF16 GEMM is
    ~32x slower than FP32 GEMM. For the LM head (128256 x 2560), this means
    BF16 backward takes ~9s per call vs ~90ms in FP32.

    This function:
    - Forward: casts input BF16->FP32, matmuls in FP32, casts output->BF16
    - Backward: casts grad_output BF16->FP32, matmuls in FP32, casts grad_input->BF16
    - Never computes grad_weight (weight is frozen)

    Memory cost: +0.66 GB for FP32 copy of LM head weight (128256 x 2560).
    """

    @staticmethod
    def forward(ctx, input_bf16, weight_f32, bias):
        # input: [batch, seq, hidden_dim] in BF16
        # weight: [vocab_size, hidden_dim] in FP32 (pre-converted)
        input_f32 = input_bf16.float()
        output_f32 = input_f32 @ weight_f32.T
        if bias is not None:
            output_f32 = output_f32 + bias.float()
        ctx.save_for_backward(weight_f32)
        ctx.has_bias = bias is not None
        return output_f32.to(input_bf16.dtype)

    @staticmethod
    def backward(ctx, grad_output):
        (weight_f32,) = ctx.saved_tensors
        # grad_output: [batch, seq, vocab_size] in BF16
        # grad_input = grad_output @ weight (no transpose — weight is [vocab, hidden])
        grad_f32 = grad_output.float()
        grad_input_f32 = grad_f32 @ weight_f32
        grad_input = grad_input_f32.to(grad_output.dtype)
        # No grad for weight (frozen) or bias (frozen or None)
        return grad_input, None, None


def patch_lm_head_fp32(model, verbose=True):
    """
    Patch the model's LM head to use FP32 matmul instead of BF16.

    This fixes the catastrophic BF16 GEMM performance on CPUs without
    AMX/VNNI instructions. The LM head weight is converted to FP32 once
    and stored alongside the original.

    Must be called AFTER the model is loaded but BEFORE training.
    Works regardless of whether patch_model() has been called.

    Args:
        model: HuggingFace causal LM model with a .lm_head attribute
        verbose: print progress

    Returns:
        True if patched, False if lm_head not found or already patched
    """
    # Find the lm_head
    lm_head = getattr(model, "lm_head", None)
    if lm_head is None:
        if verbose:
            print("FP32 LM head: no lm_head found on model, skipping")
        return False

    if not isinstance(lm_head, nn.Linear):
        if verbose:
            print(
                f"FP32 LM head: lm_head is {type(lm_head).__name__}, not nn.Linear, skipping"
            )
        return False

    if hasattr(lm_head, _LM_HEAD_PATCH_ATTR):
        if verbose:
            print("FP32 LM head: already patched, skipping")
        return False

    # Convert weight to FP32 (keep original BF16 for weight tying)
    weight_f32 = lm_head.weight.data.float().contiguous()
    bias = lm_head.bias  # Usually None for LLMs

    if verbose:
        mem_mb = weight_f32.numel() * 4 / 1e6
        print(
            f"FP32 LM head: converting {tuple(lm_head.weight.shape)} "
            f"from {lm_head.weight.dtype} to float32 ({mem_mb:.0f} MB)"
        )

    # Store FP32 weight as a non-parameter buffer (won't show in parameters())
    # Use register_buffer so it moves with .to() calls but doesn't get gradients
    lm_head.register_buffer(_LM_HEAD_FP32_WEIGHT_ATTR, weight_f32)

    # Save original forward
    setattr(lm_head, _LM_HEAD_PATCH_ATTR, lm_head.forward)

    # Create patched forward
    def make_fp32_forward(head, w_f32_attr, head_bias):
        def patched_forward(x):
            w_f32 = getattr(head, w_f32_attr)
            return FP32LMHeadFunction.apply(x, w_f32, head_bias)

        return patched_forward

    lm_head.forward = make_fp32_forward(lm_head, _LM_HEAD_FP32_WEIGHT_ATTR, bias)

    if verbose:
        print("FP32 LM head: patched successfully")

    return True


def unpatch_lm_head(model, verbose=True):
    """Restore the original LM head forward pass."""
    lm_head = getattr(model, "lm_head", None)
    if lm_head is None or not hasattr(lm_head, _LM_HEAD_PATCH_ATTR):
        return False

    lm_head.forward = getattr(lm_head, _LM_HEAD_PATCH_ATTR)
    delattr(lm_head, _LM_HEAD_PATCH_ATTR)

    # Remove FP32 weight buffer
    if hasattr(lm_head, _LM_HEAD_FP32_WEIGHT_ATTR):
        delattr(lm_head, _LM_HEAD_FP32_WEIGHT_ATTR)

    if verbose:
        print("FP32 LM head: unpatched")

    return True
