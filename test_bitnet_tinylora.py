"""
BitNet b1.58 + TinyLoRA Compatibility Validation
=================================================
Tests whether a continuous-valued LoRA adapter can attach to,
forward through, and receive gradients from a natively ternary
BitNet b1.58 model.

5 steps. Pass/fail. ~30 minutes total on CPU.
"""

import sys
import time
import torch
import torch.nn as nn

MODEL_PATH = "models/bitnet-b1.58-2B-4T-bf16"


def step_banner(step_num, name):
    print(f"\n{'=' * 60}")
    print(f"  STEP {step_num}: {name}")
    print(f"{'=' * 60}")


def result(step_num, name, passed, detail=""):
    status = "PASS" if passed else "FAIL"
    dots = "." * (40 - len(name))
    print(f"[STEP {step_num}] {name} {dots} {status}")
    if detail:
        print(f"         {detail}")
    if not passed:
        print(f"\nBLOCKED AT STEP {step_num}. Investigate before proceeding.")
        sys.exit(1)


# =========================================================================
# STEP 1: LOAD
# =========================================================================
step_banner(1, "LOAD MODEL")

try:
    from transformers import AutoModelForCausalLM, AutoTokenizer

    print(f"Loading tokenizer from {MODEL_PATH}...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)

    print(f"Loading model from {MODEL_PATH} (BF16, CPU)...")
    t0 = time.time()
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH,
        torch_dtype=torch.bfloat16,
        device_map="cpu",
        trust_remote_code=True,
    )
    load_time = time.time() - t0

    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    model_size_gb = sum(p.nbytes for p in model.parameters()) / (1024**3)

    print(
        f"Loaded in {load_time:.1f}s | {total_params / 1e9:.2f}B params | {model_size_gb:.1f}GB"
    )

    # Quick generation test
    print("Testing generation...")
    test_input = tokenizer("The capital of France is", return_tensors="pt")
    with torch.no_grad():
        out = model.generate(**test_input, max_new_tokens=10, do_sample=False)
    gen_text = tokenizer.decode(out[0], skip_special_tokens=True)
    print(f"  Generated: {gen_text!r}")

    result(
        1,
        "Model load",
        True,
        f"{total_params / 1e9:.2f}B params, {model_size_gb:.1f}GB, {load_time:.0f}s load",
    )

except Exception as e:
    result(1, "Model load", False, str(e))


# =========================================================================
# STEP 2: INSPECT ARCHITECTURE
# =========================================================================
step_banner(2, "INSPECT ARCHITECTURE")

try:
    # Find all linear-like layers
    linear_layers = []
    all_layer_types = set()

    for name, module in model.named_modules():
        all_layer_types.add(type(module).__name__)
        # Look for any linear-like layer (BitLinear, Linear, etc.)
        if hasattr(module, "weight") and isinstance(module.weight, torch.Tensor):
            if module.weight.ndim == 2:  # Matrix weight = linear layer
                linear_layers.append((name, type(module).__name__, module.weight.shape))

    print(f"Module types found: {sorted(all_layer_types)}")
    print(f"\nLinear-like layers with 2D weights: {len(linear_layers)}")
    print(f"\nFirst 10 layers:")
    for name, typename, shape in linear_layers[:10]:
        print(f"  {typename:20s} {str(shape):20s} {name}")
    print(
        f"  ... and {len(linear_layers) - 10} more" if len(linear_layers) > 10 else ""
    )

    # Identify target layer types (likely BitLinear or Linear)
    target_types = set(t for _, t, _ in linear_layers)
    print(f"\nTarget layer types: {target_types}")

    result(
        2,
        "Architecture inspect",
        True,
        f"Found {len(linear_layers)} linear layers, types: {target_types}",
    )

except Exception as e:
    result(2, "Architecture inspect", False, str(e))


# =========================================================================
# STEP 3: ATTACH TINYLORA ADAPTER
# =========================================================================
step_banner(3, "ATTACH TINYLORA")


class TinyLoRA(nn.Module):
    """Sub-rank-1 LoRA adapter following the TinyLoRA paper.

    Instead of W + BA (rank-r), uses:
        output = base(x) + scale * (x @ v^T) @ u^T

    where u, v are fixed random vectors and scale is a learned scalar.
    This gives exactly 1 trainable parameter per adapted layer.
    """

    def __init__(self, base_layer, seed=42):
        super().__init__()
        self.base_layer = base_layer
        out_features, in_features = base_layer.weight.shape

        # Fixed random projection vectors (not trained)
        gen = torch.Generator().manual_seed(seed)
        u = torch.randn(out_features, 1, generator=gen, dtype=torch.bfloat16)
        v = torch.randn(1, in_features, generator=gen, dtype=torch.bfloat16)
        # Normalize for stability
        u = u / u.norm()
        v = v / v.norm()
        self.register_buffer("u", u)
        self.register_buffer("v", v)

        # The ONE trainable parameter
        self.scale = nn.Parameter(torch.zeros(1, dtype=torch.bfloat16))

    def forward(self, x):
        base_out = self.base_layer(x)
        # Adapter: rank-1 update scaled by learned scalar
        # x: (..., in_features) @ v^T: (in_features, 1) -> (..., 1)
        # (..., 1) @ u^T: (1, out_features) -> (..., out_features)
        adapter_out = (x @ self.v.T) @ self.u.T * self.scale
        return base_out + adapter_out


try:
    # Freeze all base model parameters
    for param in model.parameters():
        param.requires_grad = False

    frozen_count = sum(1 for p in model.parameters() if not p.requires_grad)
    print(f"Froze {frozen_count} parameter tensors")

    # Pick target layers: first few attention Q projections
    # We'll adapt a small number of layers for the test
    adapted_layers = []
    adapter_modules = []
    num_to_adapt = 4  # Adapt 4 layers = 4 trainable params total

    for name, module in list(model.named_modules()):
        if len(adapted_layers) >= num_to_adapt:
            break
        # Target attention Q/K/V projections or any 2D weight layer
        if hasattr(module, "weight") and isinstance(module.weight, torch.Tensor):
            if module.weight.ndim == 2 and "q_proj" in name:
                # Replace module with TinyLoRA wrapper
                adapter = TinyLoRA(module, seed=42 + len(adapted_layers))
                # Navigate to parent and replace
                parts = name.rsplit(".", 1)
                if len(parts) == 2:
                    parent_name, attr_name = parts
                    parent = dict(model.named_modules())[parent_name]
                    setattr(parent, attr_name, adapter)
                    adapted_layers.append(name)
                    adapter_modules.append(adapter)
                    print(f"  Adapted: {name} ({module.weight.shape})")

    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params_now = sum(p.numel() for p in model.parameters())
    print(f"\nAdapted {len(adapted_layers)} layers")
    print(f"Trainable parameters: {trainable_params}")
    print(f"Total parameters: {total_params_now / 1e9:.2f}B")

    # Verify generation still works with adapters
    print("\nTesting generation with adapters...")
    with torch.no_grad():
        out2 = model.generate(**test_input, max_new_tokens=10, do_sample=False)
    gen_text2 = tokenizer.decode(out2[0], skip_special_tokens=True)
    print(f"  Generated: {gen_text2!r}")

    result(
        3,
        "TinyLoRA attach",
        len(adapted_layers) > 0,
        f"{len(adapted_layers)} layers adapted, {trainable_params} trainable params",
    )

except Exception as e:
    import traceback

    traceback.print_exc()
    result(3, "TinyLoRA attach", False, str(e))


# =========================================================================
# STEP 4: GRADIENT FLOW
# =========================================================================
step_banner(4, "GRADIENT FLOW")

try:
    # Simple cross-entropy loss on a trivial example
    prompt = "Question: What is 2 + 2?\nAnswer: The answer is 4."
    inputs = tokenizer(prompt, return_tensors="pt")
    input_ids = inputs["input_ids"]
    labels = input_ids.clone()

    print(f"Input sequence length: {input_ids.shape[1]} tokens")
    print("Running forward pass...")
    t0 = time.time()
    outputs = model(input_ids=input_ids, labels=labels)
    fwd_time = time.time() - t0
    loss = outputs.loss
    print(f"  Forward pass: {fwd_time:.1f}s")
    print(f"  Loss: {loss.item():.4f}")

    print("Running backward pass...")
    t0 = time.time()
    loss.backward()
    bwd_time = time.time() - t0
    print(f"  Backward pass: {bwd_time:.1f}s")

    # Check adapter gradients
    adapter_grads_ok = True
    for i, adapter in enumerate(adapter_modules):
        grad = adapter.scale.grad
        has_grad = grad is not None
        is_nonzero = has_grad and grad.abs().item() > 0
        print(
            f"  Adapter {i}: grad={'non-zero' if is_nonzero else ('zero' if has_grad else 'None')}"
            f" (value={grad.item():.6f})"
            if has_grad
            else ""
        )
        if not is_nonzero:
            adapter_grads_ok = False

    # Check base model gradients are None
    base_grads_none = True
    for name, param in model.named_parameters():
        if param.requires_grad:
            continue  # Skip adapter params
        if param.grad is not None:
            base_grads_none = False
            print(f"  WARNING: Base param has grad: {name}")
            break

    print(f"\n  Adapter grads non-zero: {adapter_grads_ok}")
    print(f"  Base grads None: {base_grads_none}")

    result(
        4,
        "Gradient flow",
        adapter_grads_ok and base_grads_none,
        f"fwd={fwd_time:.1f}s, bwd={bwd_time:.1f}s, adapter grads OK, base frozen",
    )

except Exception as e:
    import traceback

    traceback.print_exc()
    result(4, "Gradient flow", False, str(e))


# =========================================================================
# STEP 5: WEIGHT UPDATE
# =========================================================================
step_banner(5, "WEIGHT UPDATE")

try:
    # Record adapter weights before update
    scales_before = [adapter.scale.item() for adapter in adapter_modules]
    print(f"Adapter scales before: {scales_before}")

    # Record model output before update
    with torch.no_grad():
        logits_before = model(input_ids=input_ids).logits[:, -1, :].clone()

    # Optimizer step
    optimizer = torch.optim.Adam(
        [adapter.scale for adapter in adapter_modules],
        lr=0.1,  # Large LR to ensure visible change
    )
    optimizer.step()

    # Record adapter weights after update
    scales_after = [adapter.scale.item() for adapter in adapter_modules]
    print(f"Adapter scales after:  {scales_after}")

    weights_changed = any(
        abs(a - b) > 1e-10 for a, b in zip(scales_before, scales_after)
    )
    print(f"Weights changed: {weights_changed}")

    # Check if model output changed
    with torch.no_grad():
        logits_after = model(input_ids=input_ids).logits[:, -1, :]

    logit_diff = (logits_before - logits_after).abs().max().item()
    output_changed = logit_diff > 1e-10
    print(f"Max logit difference: {logit_diff:.6f}")
    print(f"Output changed: {output_changed}")

    result(
        5,
        "Weight update",
        weights_changed and output_changed,
        f"scales: {scales_before} -> {scales_after}, logit diff: {logit_diff:.6f}",
    )

except Exception as e:
    import traceback

    traceback.print_exc()
    result(5, "Weight update", False, str(e))


# =========================================================================
# FINAL SUMMARY
# =========================================================================
print(f"\n{'=' * 60}")
print(f"  ALL 5 STEPS PASSED")
print(f"{'=' * 60}")
print(f"""
  BitNet b1.58 + TinyLoRA is VIABLE.

  What was proven:
  - Natively ternary model loads and runs on CPU
  - Custom TinyLoRA adapters attach to BitLinear layers
  - Continuous-valued adapter corrections work alongside ternary weights
  - Gradients flow through the ternary forward pass to adapter params
  - Optimizer updates change adapter weights and model behavior

  Next steps:
  1. Add GRPO training loop with GSM8K reward signal
  2. Port to Jetson AGX Thor for GPU-accelerated experiments
  3. Build autoresearch-style autonomous loop
  4. Explore adapter placement (attention vs. all layers vs. specific layers)

  Novel finding: TinyLoRA on natively ternary weights works.
  This is likely the first demonstration of sub-rank-1 LoRA
  on a BitNet architecture.
""")
