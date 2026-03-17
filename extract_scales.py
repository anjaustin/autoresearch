#!/usr/bin/env python3
"""
Extract per-layer weight scales from BF16 BitNet model.
Run once — saves models/bitnet-b1.58-2B-4T-bf16/weight_scales.pt
After this, inference never needs to load the 4.83GB BF16 weights again.
"""
import sys, time, warnings
warnings.filterwarnings("ignore")
import torch

MODEL_PATH = "models/bitnet-b1.58-2B-4T-bf16"
OUT_PATH   = f"{MODEL_PATH}/weight_scales.pt"

print(f"Loading BF16 model from {MODEL_PATH}...")
t0 = time.time()
from transformers import AutoModelForCausalLM
model = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH, torch_dtype=torch.bfloat16,
    device_map="cpu", trust_remote_code=True,
)
print(f"  Loaded in {time.time()-t0:.1f}s")

print("Extracting weight scales from AutoBitLinear layers...")
scales = {}
for name, module in model.named_modules():
    if module.__class__.__name__ == "AutoBitLinear":
        scale = module.weight.abs().mean().item()
        scales[name] = scale

print(f"  Extracted {len(scales)} scales")
print(f"  Min scale: {min(scales.values()):.6f}")
print(f"  Max scale: {max(scales.values()):.6f}")
print(f"  Mean scale: {sum(scales.values())/len(scales.values()):.6f}")

torch.save(scales, OUT_PATH)
import os
size = os.path.getsize(OUT_PATH)
print(f"\nSaved: {OUT_PATH}")
print(f"  Size: {size} bytes ({size/1024:.2f} KB)")
print(f"\nDone. You can now delete the BF16 weights after packing ternary,")
print(f"or keep them for gradient-based training.")
