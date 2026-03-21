import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import sys
import ctypes

MODEL_PATH = "models/bitnet-b1.58-2B-4T-bf16"

print("Loading tokenizer...", flush=True)
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
print("Loading model...", flush=True)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH, torch_dtype=torch.bfloat16, device_map="cpu", trust_remote_code=True
)
print("Model loaded successfully!", flush=True)

from softchip.ghost_engine_wrapper import GhostEngine

GHOST_SEED = 42
SCALES_PATH = "models/bitnet-b1.58-2B-4T-bf16/weight_scales.pt"

print("Initializing Ghost Engine...", flush=True)
engine = GhostEngine(model, tokenizer, GHOST_SEED, SCALES_PATH)
print("Ghost Engine initialized successfully!", flush=True)
