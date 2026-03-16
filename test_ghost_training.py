#!/usr/bin/env python3
import torch, sys
sys.path.insert(0, '.')
from softchip import patch_model, unpatch_model
from transformers import AutoModelForCausalLM, AutoTokenizer

print("Loading...")
model = AutoModelForCausalLM.from_pretrained("models/bitnet-b1.58-2B-4T-bf16", torch_dtype=torch.bfloat16, device_map="cpu", trust_remote_code=True)
tokenizer = AutoTokenizer.from_pretrained("models/bitnet-b1.58-2B-4T-bf16", trust_remote_code=True)

print("Patching GhostWeight...")
patch_model(model, use_ghost=True, ghost_seed=42, verbose=True)

for p in model.parameters():
    p.requires_grad = True

inputs = tokenizer("2+2=", return_tensors="pt")
out = model(**inputs)
loss = out.logits.mean()
loss.backward()

nz = sum(1 for p in model.parameters() if p.grad is not None and p.grad.abs().sum() > 0)
print(f"Loss: {loss.item():.4f}, non-zero grads: {nz}")
unpatch_model(model)
print("DONE")
