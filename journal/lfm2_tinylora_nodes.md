# Nodes of Interest: LFM2-24B-A2B + TinyLoRA on Jetson AGX Thor

## Node 1: The Reward Signal Mismatch
TinyLoRA's extreme parameter efficiency (13 params @ 91% GSM8K) only works with RL training, not SFT. RL needs verifiable rewards. Autoresearch uses val_bpb (a language modeling metric), which is an SFT-style signal. These are fundamentally different optimization targets.
**Why it matters:** If we keep val_bpb, TinyLoRA's efficiency advantage vanishes (need 100-1000x more params for SFT). If we switch to verifiable tasks, we're no longer doing language modeling research.

## Node 2: The Compute Budget on Thor
The Jetson AGX Thor is an edge device with Blackwell-class GPU and 128GB unified memory. Memory is sufficient for a 24B model in BF16 (~48GB). But training throughput is significantly below H100 (perhaps 3-10x slower). A 5-minute experiment window may yield too few RL rollouts for signal.
**Why it matters:** The experiment cadence is the heartbeat of the autoresearch loop. If one cycle takes 30+ minutes instead of 5, the overnight autonomous run covers far fewer experiments.

## Node 3: LFM2's Hybrid Architecture Complicates LoRA Placement
LFM2-24B-A2B has 30 gated short convolution layers and 10 grouped query attention layers. Standard LoRA targets attention projections (Q, K, V, O). Conv layers have different parameter structures. With TinyLoRA's sub-rank-1 parameterization, choosing the wrong layers means the 13 parameters have zero leverage.
**Why it matters:** Layer placement is the most critical hyperparameter for TinyLoRA on a non-standard architecture. This is also the thing the autoresearch agent could search over.

## Node 4: MoE Routing as an Adapter Target
LFM2-24B-A2B has 64 experts with top-4 routing per token. The router is a small linear layer per MoE block. Tiny perturbations to the router weights could redirect token flow across experts -- potentially a high-leverage target for TinyLoRA since it controls which 2.3B of the 24B get activated.
**Why it matters:** This is a novel and potentially high-leverage idea that TinyLoRA's extreme parameter efficiency is uniquely suited for.

## Node 5: Start Small with LFM2-8B-A1B
The LFM2 family includes an 8.3B MoE (1.5B active) with the same architecture pattern (18 conv + 6 attn, 32 experts). This is roughly 3x smaller and would validate the approach faster on the Thor before committing to the 24B model.
**Tension with Node 2:** Faster iteration cycles vs. ultimate target model.

## Node 6: The Autoresearch Pattern vs. Autoresearch Implementation
What we want is the *pattern* (autonomous loop: propose -> experiment -> evaluate -> keep/discard -> repeat) not the specific codebase (prepare.py, train.py with GPT). The implementation needs to be rebuilt from scratch for LoRA+RL, but the loop structure and git-based experiment tracking can carry over.
**Why it matters:** Trying to retrofit the existing autoresearch code would be more work than writing a clean new loop.

## Node 7: Task Selection Determines Viability
TinyLoRA was proven on math reasoning (GSM8K, AIME, MATH500). LFM2-24B-A2B is an instruct model described as "not recommended for coding." Natural targets: math reasoning improvement, tool-use accuracy, instruction following (IFEval), or domain specialization. Each has different reward signal complexity.
**Tension with Node 1:** Verifiable math tasks give clean RL signal but narrow the use case. Broader tasks (instruction following) have messier rewards but wider impact.

## Node 8: Software Stack Uncertainty on Jetson
LFM2 requires transformers>=5.0.0. The model uses custom architecture code (`Lfm2MoeForCausalLM`). Flash Attention 2 is optional but speeds things up. JetPack/L4T's CUDA and PyTorch versions need to be compatible. This is a non-trivial integration risk.
**Why it matters:** A week of environment debugging would stall the entire project.

## Node 9: Quantization as a Throughput Multiplier
Quantizing the frozen base model (INT8 or INT4) would dramatically reduce memory usage and increase inference throughput on Thor. The LoRA adapters stay in BF16/FP16. This is standard practice (QLoRA) and would directly address the compute budget concern (Node 2).
**Tension with Node 3:** Quantization can interact unpredictably with non-standard architectures and conv layers.

## Node 10: The Agent's Search Space is the Key Design Decision
In autoresearch, the agent modifies train.py (model architecture, hyperparams). In our system, the agent's search space would be: which layers get TinyLoRA adapters, the reparameterization scheme, learning rate, RL reward shaping, number of rollouts per cycle. This search space must be well-defined and small enough for an overnight run to explore meaningfully.
**Why it matters:** A poorly defined search space means the agent wastes experiments. A too-narrow space means it can't find improvements.

## Tensions Summary
- **Node 1 vs Node 7:** RL reward signal design -- clean (math) vs. broad (instruction following)
- **Node 2 vs Node 5:** Compute budget -- validate on 8B first or go straight to 24B
- **Node 3 vs Node 4:** LoRA placement -- safe (attention layers) vs. novel (MoE router)
- **Node 6 vs existing codebase:** Rewrite the loop vs. retrofit autoresearch
- **Node 9 vs Node 3:** Quantization benefit vs. architecture compatibility risk

## Dependencies
- Node 8 (software stack) blocks everything else
- Node 7 (task selection) determines Node 1 (reward signal)
- Node 2 (compute budget) informs Node 5 (start small or go big)
- Node 10 (search space) depends on Node 3 and Node 4 (layer targeting)
