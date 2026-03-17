# Project GhostWeight: The 1KB LLM
## TinyLoRA + GRPO on BitNet b1.58 — CPU-Only, AVX2

> **Fork:** Diverged from [karpathy/autoresearch](https://github.com/karpathy/autoresearch). This project pursues **Project GhostWeight**: replacing a 500MB frozen LLM weight matrix with a deterministic 8-byte PRNG seed, then training a 1KB set of TinyLoRA adapters via GRPO to steer the resulting noise into coherent reasoning.

---

## The Claim

> A 2.41-billion parameter language model can reason using only **1KB of learned parameters**, if the 500MB weight matrix is replaced by a deterministic PRNG seeded with 8 bytes.

The TinyLoRA adapter (210 scalar values = ~1KB) learns to compensate for the structured noise introduced by the PRNG, steering it toward correct answers on mathematical reasoning tasks (GSM8K).

---

## Key Results

| Metric | Value |
|---|---|
| Model weights storage | **8 bytes** (PRNG seed) + **~1KB** (TinyLoRA adapters) |
| Base model | Microsoft BitNet b1.58 2B4T (natively ternary {-1,0,+1}) |
| Training task | GSM8K mathematical reasoning (GRPO, on-policy RL) |
| Training hardware | AMD Ryzen 5 PRO 5675U — **CPU only**, no GPU |
| Forward speedup vs stock PyTorch | **4.6x** (AVX2 LUT kernel, M=1) |
| Backward speedup vs stock PyTorch | **75x** (ternary backward + FP32 LM head) |
| GhostWeight kernel speedup | **3.1x** (LUT vs bit-manipulation, 3.6ms/layer) |
| TinyLoRA parameters | **210** (1 scalar per AutoBitLinear layer) |
| Running reward (TinyLoRA on real weights) | **~91%** at step 220 |

---

## Architecture

```
PRNG Seed (8 bytes)
        │
        ▼
  SplitMix64 + XorShift128+
        │ regenerate on-the-fly
        ▼
  Ternary weights {-1,0,+1} × weight_scale
        │
        ▼
  AVX2 LUT Matmul (3.6ms/layer, 2MB lookup table)
        │
        ▼
  TinyLoRA bypass: output += scale × (x @ v.T) @ u.T
        │ (scale = 1 learned scalar per layer)
        ▼
  Final output → GRPO reward → gradient → scale update
```

**The key insight:** The TinyLoRA bypass path operates entirely in full precision on the input `x`, independent of whether the base weight matrix contains trained BitNet values or PRNG noise. The gradients flow through the bypass cleanly. The 210 scalar `scale` parameters are the only learned state.

---

## Unified Solution: One Run to Rule Them All

Two intermediate solutions exist:
1. **TinyLoRA on real BitNet weights** — works, 91% reward, but 500MB model.
2. **GhostWeight kernel** — works as a kernel, but adapters must be trained against PRNG from step 0.

**The unified solution:** Set `USE_GHOST=True` in `grpo_train.py` and run from step 0. Adapters and PRNG base train together. This is the finish line.

```bash
# Compile kernels
gcc -O3 -mavx2 -mfma -march=native -fopenmp -shared -fPIC \
    -o softchip/ternary_matmul_v3.so softchip/ternary_matmul_v3.c -lm
gcc -O3 -mavx2 -mfma -shared -fPIC \
    -o softchip/ghost_matmul.so softchip/ghost_matmul_lut.c

# Run (USE_GHOST=True is set in grpo_train.py)
nohup python -u grpo_train.py > grpo_ghost.log 2>&1 &

# Resume after interruption
nohup python -u grpo_train.py --resume=checkpoints/grpo_step_XXXX.pt \
    >> grpo_ghost.log 2>&1 &
```

See **[FINDINGS.md](FINDINGS.md)** for the complete technical writeup, all LMM analysis passes, and reproduction instructions.

---

## Hardware

- **CPU:** AMD Ryzen 5 PRO 5675U (6C/12T, Zen 3, AVX2+FMA)
- **RAM:** 64GB DDR4-3200
- **iGPU:** Vega 7 (Vulkan backend implemented; CPU AVX2 is faster for this workload)
- **OS:** Linux, Python 3.13, PyTorch 2.10

---

## Project Structure

```
grpo_train.py               — GRPO training loop (TinyLoRA + GhostWeight)
softchip/
  torch_ternary.py          — PyTorch integration: patch_model(), GhostWeight
  ternary_matmul_v3.c       — AVX2 ternary matmul (packed BitNet, forward+backward)
  ghost_matmul_lut.c        — AVX2 GhostWeight LUT kernel (PRNG + M=4 unrolled)
  ghost_matmul.c            — GhostWeight original kernel (reference)
  mtp18_matmul.c            — MTP18 native base-3 kernel (experimental)
  vk_backend.c              — Vulkan iGPU backend (implemented, not faster on Ryzen)
checkpoints/                — GRPO adapter scale checkpoints
journal/                    — LMM analysis passes (raw/nodes/reflect/synth)
FINDINGS.md                 — Full technical writeup
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
