# Project GhostWeight: Sub-1KB Adapters on BitNet b1.58
## GhostChain + GRPO on a 2.41B Ternary LLM — CPU-Only, AVX2

> **Fork:** Diverged from [karpathy/autoresearch](https://github.com/karpathy/autoresearch). This project explores extreme parameter efficiency for on-policy RL fine-tuning of large ternary language models on commodity CPU hardware, and pursues **Project GhostWeight**: the hypothesis that a 500MB frozen weight matrix can be replaced by a deterministic 8-byte PRNG seed.

---

## Current Status

**Architecture pivot validated.** GhostChain scalar adapters (840 params) have hit their ceiling at ~45% GSM8K accuracy. LoRA rank-4 is the confirmed next step.

| Track | Status | Key finding |
|---|---|---|
| **GhostChain scalar adapters** | **Ceiling reached** | 840 scalar params, ~45% accuracy max, 10/20 eval questions locked wrong across 225 training steps. Fixed random adapter directions cannot reroute wrong reasoning paths. |
| **LoRA rank-4 (trainable A+B)** | **Validated — ready to run** | 5.4M params. Flipped 2 questions in 15 pinned steps that scalars couldn't move in 225 steps. Stochastic ceiling: 7/10 locked questions are reachable. Theoretical ceiling: 13/20 = 65%. |
| **GhostWeight (PRNG replaces stored weights)** | **Archived** | 200× slower than real-weights path on CPU. Correct concept, wrong hardware. |

**Next action:** Full 8-hour GRPO run with LoRA rank-4, LR=0.0174, curriculum token mismatch fix.

---

## The Hypothesis

> A 2.41-billion parameter language model can reason using only **~1.6KB of learned parameters**, if the 500MB weight matrix is replaced by a deterministic PRNG seeded with 8 bytes.

The **GhostChain** adapter (840 scalar values ≈ 1.6KB in BF16) attaches 4 fixed rank-1 experts per layer — all reading from the same input activation, summed in parallel. Only the 4 scalars are learned. With 210 `AutoBitLinear` layers in BitNet b1.58, this gives 840 trainable parameters total.

This is an open hypothesis, not a demonstrated result. The validated finding underneath it — that **840 scalar adapters can measurably steer a 2.41B frozen ternary model via RL** — stands on its own.

---

## Validated Results

### Sub-rank-1 LoRA on a Natively Ternary LLM

4 scalar parameters (one per adapted layer) produce measurable, non-zero gradients and logit changes through BitNet b1.58's `AutoBitLinear` layers. Gradient flow works because the LoRA bypass path operates on full-precision activations and never touches the discrete ternary weights directly. This is the first known demonstration of sub-rank-1 LoRA on a natively 1.58-bit architecture.

With all 210 layers adapted (840 parameters), adapter scales grow monotonically from 0 to ~0.034 over 30 GRPO steps. Running reward on GSM8K reaches ~91% at step 220.

### 75x Backward Pass Speedup on CPU

The backward pass through 2.41B frozen ternary parameters initially took 82.9 seconds per iteration. Two kernel optimizations reduced this to 1.1 seconds — a **75x speedup** — making interactive CPU-side RL training practical.

**Optimization 1 — Ternary STE backward kernel (4.3x):**
The backward pass through frozen ternary weights computes `grad_input = W^T @ grad_output`. Since W is ternary, this is the same add/subtract/skip operation as the forward pass. A custom `ternary_matmul_backward` kernel exploits this: iterate over weight rows, scatter-add each row's contribution scaled by the upstream gradient. No multiplications in the hot path.

**Optimization 2 — FP32 LM head (18x on the remaining bottleneck):**
After the ternary backward kernel, profiling revealed that 92.6% of the remaining 19.5s came from two matmul calls in the LM head — a dense `nn.Linear(2560, 128256)` not patched by the ternary kernel. Root cause: **MKL's BF16 GEMM on Zen 3 (no AMX/VNNI) is 32–90x slower than FP32 for large dense matmuls.** The fix — `FP32LMHeadFunction`, a custom autograd function that casts BF16↔FP32 at the boundary and does all computation in FP32 — is a one-line model patch applicable to any BF16 LLM trained on AMD CPU without AVX512-BF16.

| Pass | Backward time (6 tokens) | Cumulative speedup |
|---|---|---|
| Stock PyTorch | 82.9s | 1x |
| + Ternary backward kernel | 19.5s | 4.3x |
| + FP32 LM head | **1.1s** | **75x** |

Full training iteration (forward + backward): 91.7s → **2.4s (38x total)**.

### 4.6x Forward Speedup — AVX2 LUT Ternary Matmul

BitNet's ternary weights mean matmuls reduce to add/subtract/skip with no multiplications. PyTorch doesn't exploit this. The soft-chip kernel packs 4 weights per byte (2-bit encoding), precomputes 256-entry LUTs for nonzero and sign masks, and processes 8 elements per loop iteration with pure XOR/AND/ADD AVX2 instructions.

| Metric | Value |
|---|---|
| Isolated layer speedup (2560×2560, M=1) | **33x** (1.6ms vs 53.5ms) |
| Full model autoregressive speedup (M=1) | **4.6x** (0.91s vs 4.2s/token) |
| 200-token rollout | ~182s vs ~840s |
| Output cosine similarity vs PyTorch | 0.9997 |

For M=1 (autoregressive generation), serial execution beats OpenMP parallel: fork/join overhead (~600μs for 12 threads) exceeds the 1.6ms single-core compute time. The kernel uses smart threading — serial below batch size 6, parallel above.

### Ternary Matrices Have Flat Singular Spectra

SVD-based compression was tested on BitNet's ternary weight matrices. Result: 82.5% reconstruction error at rank-64 (vs ~2% for dense FP32 models), with cross-layer singular vector cosine similarity of ~0.000. Ternary {-1, 0, +1} matrices distribute information uniformly across all singular dimensions — there is no low-rank structure to exploit. This rules out SVD compression for ternary models and provides a theoretical grounding for why PRNG (GhostWeight) is the only viable path to sub-MB weight storage.

---

## Full Results Summary

| Metric | Value | Status |
|---|---|---|
| Base model | Microsoft BitNet b1.58 2B4T (natively ternary {-1,0,+1}) | — |
| Training task | GSM8K mathematical reasoning (GRPO, on-policy RL) | — |
| Training hardware | AMD Ryzen 5 PRO 5675U — **CPU only**, no GPU | — |
| GhostChain parameters | **840** (4 scalars × 210 layers, ~1.6KB BF16) | Ceiling reached at ~45% |
| LoRA rank-4 parameters | **5,406,720** (~10MB, trainable A+B) | Validated — 2 questions flipped in 15 steps |
| Scalar adapter ceiling (GSM8K) | **~45%** (10/20 questions permanently locked) | Confirmed over 10 evals / 225 steps |
| LoRA theoretical ceiling (this eval set) | **~65%** (13/20 solvable, 3 unreachable) | Phase 1 stochastic ceiling check |
| Forward speedup vs stock PyTorch | **4.6x** (AVX2 LUT kernel, M=1 autoregressive) | Validated |
| Backward speedup vs stock PyTorch | **75x** (ternary backward + FP32 LM head) | Validated |
| Training iteration time | **2.4s** (down from 91.7s stock) | Validated |
| GhostWeight storage footprint | **8 bytes seed + ~1.6KB adapters** | Theoretical |
| GhostWeight GSM8K accuracy | — | Open — not yet trained |

---

## Architecture

The active training architecture (`USE_GHOST=False`): real stored ternary weights, GhostChain adapters, Python rollouts via the patched PyTorch model.

```
BitNet b1.58 (stored ternary weights, 500MB)
        │  packed 2-bit format, pre-loaded at startup
        ▼
  AVX2 LUT Matmul  (1.6ms/layer at M=1, 33x over PyTorch)
        │
        ▼
  GhostChain (per layer, 4 learned scalars):
    d1 = s1 · LoRA1(x)   ─┐
    d2 = s2 · LoRA2(x)    ├─ all 4 experts read same x (parallel)
    d3 = s3 · LoRA3(x)    │
    do = so · LoRAo(x)   ─┘
    y  = Base(x) + d1 + d2 + d3 + do
        │
        ▼
  GRPO reward → 75x-faster backward → scale update (840 params)
```

Each `LoRAi` is a fixed rank-1 projection with random basis (u, v vectors generated from layer name hash, never stored). Only the 4 scalar scales are learned per layer. The 4 experts run in parallel — mathematically equivalent to a single rank-4 LoRA with fixed random directions.

**GhostWeight path** (`USE_GHOST=True`, archived — currently ~200x slower on CPU): weights are not stored at all. Each matmul regenerates the weight matrix on-the-fly from a PRNG (SplitMix64 + XorShift128+ in AVX2 registers), seeded by `base_seed + layer_id`. The entire 2.41B parameter base is described by a single 8-byte uint64. This path requires different hardware (GPU-level PRNG parallelism) to be practical at training time.

See **[docs/ARCHITECTURE.md](docs/ARCHITECTURE.md)** for the full mathematical treatment.

---

## Quick Start

```bash
# 1. Compile the ternary matmul kernel
bash softchip/build_kernels.sh

# 2. Download model (4.5 GB BF16 master weights)
huggingface-cli download microsoft/bitnet-b1.58-2B-4T-bf16 \
    --local-dir models/bitnet-b1.58-2B-4T-bf16

# 3. Run training (USE_GHOST=False — real ternary weights, GhostChain adapters)
nohup python -u grpo_train.py > grpo_train.log 2>&1 &

# 4. Resume after interruption
nohup python -u grpo_train.py --resume=checkpoints/grpo_step_XXXX.pt \
    >> grpo_train.log 2>&1 &

# Monitor
tail -f grpo_train.log
```

## Analysis Tools

```bash
# Measure base model accuracy on GSM8K test set (zero adapters — the floor):
python measure_baseline.py

# Compare base model vs a trained checkpoint:
python measure_baseline.py --checkpoint checkpoints/grpo_step_XXXX.pt

# Analyse which layers are absorbing the training signal:
python analyze_layers.py                          # most recent checkpoint
python analyze_layers.py checkpoints/grpo_step_XXXX.pt  # specific checkpoint
python analyze_layers.py step_050.pt step_120.pt  # compare over time
```

> **Note on the GhostWeight path:** `USE_GHOST=True` in `grpo_train.py` enables PRNG weight generation instead of stored weights. On this CPU it is ~200x slower than the real-weights path. The GhostWeight kernels are preserved in `research/archive/softchip/` for reference.

---

## Hardware

- **CPU:** AMD Ryzen 5 PRO 5675U (6C/12T, Zen 3, AVX2+FMA)
- **RAM:** 64GB DDR4-3200
- **iGPU:** Vega 7 (Vulkan backend implemented; CPU AVX2 is faster for this workload)
- **OS:** Linux, Python 3.13, PyTorch 2.10

---

## Project Structure

```
grpo_train.py                    — GRPO training loop (GhostChain, USE_GHOST=False)
measure_baseline.py              — GSM8K accuracy with zero adapters (the floor)
analyze_layers.py                — Per-layer adapter scale analysis, Lorenz curve
softchip/
  build_kernels.sh               — Compile ternary_matmul_v3.so (the only required kernel)
  torch_ternary.py               — PyTorch integration: patch_model(), FP32 LM head patch
  ternary_matmul_v3.c            — AVX2 ternary matmul kernel (packed 2-bit, fwd+bwd)
checkpoints/                     — GRPO adapter scale checkpoints (.pt files)
docs/
  ARCHITECTURE.md                — GhostChain design and mathematical treatment
  KNOWN_BUGS.md                  — 9 documented bugs with root cause analysis and fixes
  NEXT_STEPS.md                  — Future directions: GMoE, Rank-1 expansion, Meta-Ghost
research/
  archive/                       — Archived code (ghost engine, Vulkan, old kernels, logs)
    README.md                    — What's in the archive and why
FINDINGS.md                      — Full technical writeup (Passes 1–14)
AUDIT_AUTORESEARCH.md            — External cold-read audit: novel/validated/open/fluff
RESEARCH_PLAN.md                 — Forward research direction and phase plan
```

---

*Original upstream README follows.*

---

# autoresearch

![teaser](progress.png)

*One day, frontier AI research used to be done by meat computers in between eating, sleeping, having other fun, and synchronizing once in a while using sound wave interconnect in the ritual of "group meeting". That era is long gone. Research is now entirely the domain of autonomous swarms of AI agents running across compute cluster megastructures in the skies. The agents claim that we are now in the 10,205th generation of the code base, in any case no one could tell if that's right or wrong as the "code" is now a self-modifying binary that has grown beyond human comprehension. This repo is the story of how it all began. -@karpathy, March 2026*.

The idea: give an AI agent a small but real LLM training setup and let it experiment autonomously overnight. It modifies the code, trains for 5 minutes, checks if the result improved, keeps or discards, and repeats. You wake up in the morning to a log of experiments and (hopefully) a better model. The training code here is a simplified single-GPU implementation of [nanochat](https://github.com/karpathy/nanochat). The core idea is that you're not touching any of the Python files like you normally would as a researcher. Instead, you are programming the `program.md` Markdown files that provide context to the AI agents and set up your autonomous research org. The default `program.md` in this repo is intentionally kept as a bare bones baseline, though it's obvious how one would iterate on it over time to find the "research org code" that achieves the fastest research progress, how you'd add more agents to the mix, etc. A bit more context on this project is here in this [tweet](https://x.com/karpathy/status/2029701092347630069).

## How it works

The repo is deliberately kept small and only really has three files that matter:

- **`prepare.py`** — fixed constants, one-time data prep (downloads training data, trains a BPE tokenizer), and runtime utilities (dataloader, evaluation). Not modified.
- **`train.py`** — the single file the agent edits. Contains the full GPT model, optimizer (Muon + AdamW), and training loop. Everything is fair game: architecture, hyperparameters, optimizer, batch size, etc. **This file is edited and iterated on by the agent**.
- **`program.md`** — baseline instructions for one agent. Point your agent here and let it go. **This file is edited and iterated on by the human**.

By design, training runs for a **fixed 5-minute time budget** (wall clock, excluding startup/compilation), regardless of the details of your compute. The metric is **val_bpb** (validation bits per byte) — lower is better, and vocab-size-independent so architectural changes are fairly compared.

If you are new to neural networks, this ["Dummy's Guide"](https://x.com/hooeem/status/2030720614752039185) looks pretty good for a lot more context.

## Quick start

**Requirements:** A single NVIDIA GPU (tested on H100), Python 3.10+, [uv](https://docs.astral.sh/uv/).

```bash

# 1. Install uv project manager (if you don't already have it)
curl -LsSf https://astral.sh/uv/install.sh | sh

# 2. Install dependencies
uv sync

# 3. Download data and train tokenizer (one-time, ~2 min)
uv run prepare.py

# 4. Manually run a single training experiment (~5 min)
uv run train.py
```

If the above commands all work ok, your setup is working and you can go into autonomous research mode.

## Running the agent

Simply spin up your Claude/Codex or whatever you want in this repo (and disable all permissions), then you can prompt something like:

```
Hi have a look at program.md and let's kick off a new experiment! let's do the setup first.
```

The `program.md` file is essentially a super lightweight "skill".

## Project structure

```
prepare.py      — constants, data prep + runtime utilities (do not modify)
train.py        — model, optimizer, training loop (agent modifies this)
program.md      — agent instructions
pyproject.toml  — dependencies
```

## Design choices

- **Single file to modify.** The agent only touches `train.py`. This keeps the scope manageable and diffs reviewable.
- **Fixed time budget.** Training always runs for exactly 5 minutes, regardless of your specific platform. This means you can expect approx 12 experiments/hour and approx 100 experiments while you sleep. There are two upsides of this design decision. First, this makes experiments directly comparable regardless of what the agent changes (model size, batch size, architecture, etc). Second, this means that autoresearch will find the most optimal model for your platform in that time budget. The downside is that your runs (and results) become not comparable to other people running on other compute platforms.
- **Self-contained.** No external dependencies beyond PyTorch and a few small packages. No distributed training, no complex configs. One GPU, one file, one metric.

## Platform support

This code currently requires that you have a single NVIDIA GPU. In principle it is quite possible to support CPU, MPS and other platforms but this would also bloat the code. I'm not 100% sure that I want to take this on personally right now. People can reference (or have their agents reference) the full/parent nanochat repository that has wider platform support and shows the various solutions (e.g. a Flash Attention 3 kernels fallback implementation, generic device support, autodetection, etc.), feel free to create forks or discussions for other platforms and I'm happy to link to them here in the README in some new notable forks section or etc.

Seeing as there seems to be a lot of interest in tinkering with autoresearch on much smaller compute platforms than an H100, a few extra words. If you're going to try running autoresearch on smaller computers (Macbooks etc.), I'd recommend one of the forks below. On top of this, here are some recommendations for how to tune the defaults for much smaller models for aspiring forks:

1. To get half-decent results I'd use a dataset with a lot less entropy, e.g. this [TinyStories dataset](https://huggingface.co/datasets/karpathy/tinystories-gpt4-clean). These are GPT-4 generated short stories. Because the data is a lot narrower in scope, you will see reasonable results with a lot smaller models (if you try to sample from them after training).
2. You might experiment with decreasing `vocab_size`, e.g. from 8192 down to 4096, 2048, 1024, or even - simply byte-level tokenizer with 256 possibly bytes after utf-8 encoding.
3. In `prepare.py`, you'll want to lower `MAX_SEQ_LEN` a lot, depending on the computer even down to 256 etc. As you lower `MAX_SEQ_LEN`, you may want to experiment with increasing `DEVICE_BATCH_SIZE` in `train.py` slightly to compensate. The number of tokens per fwd/bwd pass is the product of these two.
4. Also in `prepare.py`, you'll want to decrease `EVAL_TOKENS` so that your validation loss is evaluated on a lot less data.
5. In `train.py`, the primary single knob that controls model complexity is the `DEPTH` (default 8, here). A lot of variables are just functions of this, so e.g. lower it down to e.g. 4.
6. You'll want to most likely use `WINDOW_PATTERN` of just "L", because "SSSL" uses alternating banded attention pattern that may be very inefficient for you. Try it.
7. You'll want to lower `TOTAL_BATCH_SIZE` a lot, but keep it powers of 2, e.g. down to `2**14` (~16K) or so even, hard to tell.

I think these would be the reasonable hyperparameters to play with. Ask your favorite coding agent for help and copy paste them this guide, as well as the full source code.

## Notable forks

- [miolini/autoresearch-macos](https://github.com/miolini/autoresearch-macos) (MacOS)
- [trevin-creator/autoresearch-mlx](https://github.com/trevin-creator/autoresearch-mlx) (MacOS)
- [jsegov/autoresearch-win-rtx](https://github.com/jsegov/autoresearch-win-rtx) (Windows)

## License

MIT
