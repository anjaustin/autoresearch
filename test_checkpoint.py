#!/usr/bin/env python3
"""Quick test script to evaluate any checkpoint."""

import sys, torch, warnings, re

warnings.filterwarnings("ignore")


def test_checkpoint(checkpoint_path, problems):
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from softchip import patch_model, unpatch_model

    # Load model
    model = AutoModelForCausalLM.from_pretrained(
        "models/bitnet-b1.58-2B-4T-bf16",
        torch_dtype=torch.bfloat16,
        device_map="cpu",
        trust_remote_code=True,
    )
    patch_model(model, backend="cpu", verbose=False)
    tokenizer = AutoTokenizer.from_pretrained(
        "models/bitnet-b1.58-2B-4T-bf16", trust_remote_code=True
    )

    # Load adapters from checkpoint
    if checkpoint_path:
        ck = torch.load(checkpoint_path, map_location="cpu")
        state = ck.get("adapter_state", {})
        for name, adapter_state in state.items():
            for module_name, module in model.named_modules():
                if (
                    module_name == name
                    and "u" in adapter_state
                    and "v" in adapter_state
                ):
                    # Apply u @ v as delta (handle different shapes per layer)
                    u = adapter_state["u"].float()  # [K, 1]
                    v = adapter_state["v"].float()  # [1, N]
                    delta = (u @ v).squeeze()  # [K, N]
                    module.weight.data += delta

    # Test problems
    print(f"\n{'=' * 50}")
    print(f"Testing: {checkpoint_path}")
    print(f"{'=' * 50}")

    correct = 0
    for q, expected in problems:
        prompt = f"Solve: {q}\nAnswer:"
        ids = tokenizer(prompt, return_tensors="pt").input_ids
        with torch.no_grad():
            out = model.generate(ids, max_new_tokens=30, do_sample=False)
        pred = tokenizer.decode(out[0, ids.shape[1] :])
        nums = re.findall(r"-?\d+", pred)
        ans = nums[0] if nums else "?"
        status = "✓" if ans == expected else "✗"
        print(f"  {q[:40]:40} → {ans:>6} ({expected:>6}) {status}")
        if ans == expected:
            correct += 1

    print(f"\nScore: {correct}/{len(problems)} = {correct / len(problems) * 100:.0f}%")
    return correct


if __name__ == "__main__":
    problems = [
        ("15 + 27", "42"),
        ("100 - 37", "63"),
        ("12 × 8", "96"),
        (
            "A store has 48 apples. They sell 15 in the morning and 12 in the afternoon. How many are left?",
            "21",
        ),
    ]

    ck_path = sys.argv[1] if len(sys.argv) > 1 else "checkpoints/grpo_step_0200.pt"
    test_checkpoint(ck_path, problems)
