# Reflections: LFM2-24B-A2B + TinyLoRA on Jetson AGX Thor

## Core Insight

**The real project is not "autoresearch for LFM2" -- it is an autonomous TinyLoRA adapter search system that uses RL to discover minimal parameter interventions on a frozen hybrid MoE model, running on edge hardware.**

The autoresearch pattern (loop, evaluate, keep/discard) is the scaffolding. TinyLoRA's extreme parameter efficiency is the mechanism. The Jetson Thor's unified 128GB memory is the enabler. But the value proposition is the combination: finding that a 24B model can be meaningfully steered by 13-1000 parameters discovered through automated RL search on consumer-class hardware.

## Resolved Tensions

### Tension 1: RL reward signal (Node 1) vs. task selection (Node 7)
**Resolution:** Use GSM8K as the primary benchmark. It is:
- Verifiable (correct/incorrect answers)
- Well-studied (direct comparison to TinyLoRA paper results)
- Meaningful for LFM2 (the model card reports GSM8K scores, so we have baselines)
- Clean RL signal for GRPO

This is not limiting -- it's strategic. Prove the system works on the same benchmark TinyLoRA used, then generalize. LFM2-24B-A2B scores on GSM8K provide a direct baseline to beat.

A secondary track can explore IFEval (instruction following) once the math reasoning loop is validated, since IFEval also has relatively clean pass/fail evaluation.

### Tension 2: Compute budget (Node 2) vs. starting model size (Node 5)
**Resolution:** Start with LFM2-8B-A1B, validate, then scale to 24B. Rationale:
- 8B in BF16 is ~16GB, leaving 112GB headroom on Thor
- Quantized 8B (INT4) is ~4GB, leaving massive room for activations and rollout batches
- Same architecture pattern (conv+attn hybrid, MoE) so findings transfer
- 3-4x faster per-step throughput means 3-4x more experiments per overnight run
- The 24B is the prestige target but proving the loop on 8B first avoids an expensive dead end

Once validated on 8B, moving to 24B is a config change, not a redesign.

### Tension 3: LoRA placement -- attention (Node 3) vs. MoE router (Node 4)
**Resolution:** This is not a tension to resolve -- it's the search space itself. The autoresearch agent should explore both:
- Attention Q/K/V/O projections (safe, proven)
- MoE router weights (novel, potentially high-leverage)
- Conv layer projections (unknown, worth testing)

The agent's job is to discover which placement works best. Define all three as valid targets in the adapter config space. Let the autonomous loop find the answer through experimentation.

### Tension 4: Rewrite vs. retrofit (Node 6 vs. existing codebase)
**Resolution:** Clean rewrite of the loop. Reuse the *pattern*, not the code:
- `program.md` equivalent: instructions for the AI agent that runs the loop
- `experiment.py`: loads frozen LFM2, applies TinyLoRA config, runs GRPO training, evaluates on GSM8K
- `loop.py` or shell script: the autoresearch outer loop (propose, run, evaluate, keep/discard, log)
- Git-based experiment tracking carries over directly
- `results.tsv` format carries over

The existing `prepare.py` and `train.py` are specific to GPT pretraining. Nothing transfers except the philosophy.

### Tension 5: Quantization benefit vs. compatibility risk (Node 9 vs. Node 3)
**Resolution:** Use bitsandbytes QLoRA (4-bit NF4 quantization for the frozen base, BF16 for adapters). This is battle-tested with HuggingFace transformers and should work with LFM2's custom architecture since transformers>=5.0.0 supports it natively. If conv layers cause quantization issues, quantize only the attention and MoE layers while keeping conv layers in BF16.

## Challenged Assumptions

### Assumption: "5 minutes per experiment is the right cadence"
**Challenge:** This was calibrated for 50M-param from-scratch training on H100. With TinyLoRA + RL on Thor, the right cadence is whatever produces enough rollouts for GRPO to converge. This might be 10-15 minutes per experiment. That's fine -- even at 15 min/experiment, an overnight (8 hour) run yields 32 experiments. The original autoresearch ran 83 experiments total.

### Assumption: "We need the AI coding agent (Claude/Codex) to run the loop"
**Challenge:** For the first version, a simpler approach works: define a fixed search space (adapter config as a JSON), write a Python script that samples configurations, runs training, evaluates, and logs. The AI agent layer can come later. Get the loop working mechanically first.

### Assumption: "TinyLoRA's 13-parameter result will transfer to LFM2"
**Challenge:** The 13-param result was on Qwen2.5-8B, a dense transformer. LFM2 is a hybrid conv+MoE architecture. The extreme low end (13 params) may not transfer. The more robust finding -- that RL enables 1000x parameter reduction vs SFT -- is more likely to hold. Budget for adapter sizes of 100-10,000 parameters, not just 13.

### Assumption: "The Thor's GPU is slow for training"
**Challenge:** Actually, unified memory means no data transfer overhead. And TinyLoRA's forward pass is identical to inference (frozen model). The Thor's inference throughput for LFM2 might be quite good -- Liquid AI reports 293 tok/s decode on H100, and the Thor's Blackwell GPU should be competitive for inference. The backward pass through tiny adapters is minimal. The bottleneck is rollout generation (inference), which the Thor may handle well.

## What I Now Understand

The project decomposes into three clean layers:

1. **Infrastructure layer:** Frozen quantized LFM2 on Thor, software stack validation
2. **Adapter search layer:** TinyLoRA configs as the search space, GRPO as the optimizer, GSM8K as the metric
3. **Automation layer:** The autoresearch-pattern loop that proposes, evaluates, and tracks experiments

These layers are independent and can be built and validated sequentially. The infrastructure layer is the highest risk (software compatibility). The adapter search layer is the core intellectual contribution. The automation layer is engineering.

## The Key Leverage Point

TinyLoRA + MoE routing is the unexplored frontier. No one has applied sub-rank-1 LoRA to MoE router weights. If 13 parameters in the router can meaningfully change which experts activate, the implications extend far beyond this project -- it would mean MoE models can be task-specialized with essentially zero overhead.
