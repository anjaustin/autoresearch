# Audit: Project GhostWeight / autoresearch
**Date:** March 2026
**Auditor:** External (Claude Code, first-pass review)
**Scope:** Full codebase, documentation, findings log, and architecture claims
**Method:** Cold read of all primary source files — no prior context

---

## Table of Contents

1. [What This Project Is](#1-what-this-project-is)
2. [Novel Contributions](#2-novel-contributions)
3. [Useful and Understated Work](#3-useful-and-understated-work)
4. [Possibly Paper-Worthy](#4-possibly-paper-worthy)
5. [The Central Claim Is Unvalidated](#5-the-central-claim-is-unvalidated)
6. [Architecture Story Has Drifted From Implementation](#6-architecture-story-has-drifted-from-implementation)
7. [Fluff and Oversold Claims](#7-fluff-and-oversold-claims)
8. [Code Quality Assessment](#8-code-quality-assessment)
9. [What Would Make This Paper-Worthy](#9-what-would-make-this-paper-worthy)
10. [Summary Table](#10-summary-table)
11. [File-by-File Notes](#11-file-by-file-notes)

---

## 1. What This Project Is

This is a solo, CPU-only research project running on a Ryzen 5 PRO 5675U laptop with no GPU. It was forked from Karpathy's `autoresearch` repo (a Karpathy-style "give an AI agent a training setup and let it experiment overnight") and pivoted into something substantially different: a deep exploration of extreme parameter efficiency for on-policy RL fine-tuning of large ternary language models on commodity hardware.

The project pursues three intersecting ideas simultaneously:

**Idea 1 — Sub-rank-1 LoRA on BitNet b1.58 via GRPO:**
Microsoft's BitNet b1.58 is a natively ternary LLM (weights in {-1, 0, +1}) trained from scratch at 2.41B parameters. The question: can a handful of continuous-valued scalar corrections ("TinyLoRA"), trained with on-policy RL (GRPO), steer the frozen model toward improved task performance? Inspired by "Learning to Reason in 13 Parameters" (Morris et al. 2025) but applied for the first time to a natively quantized architecture.

**Idea 2 — GhostWeight (PRNG weight replacement):**
If TinyLoRA can steer a frozen model, do you even need the 500 MB weight matrix? The GhostWeight hypothesis: replace all stored weights with a deterministic PRNG seed (8 bytes), regenerate them on-the-fly at inference time, and train adapters to correct the resulting noise into coherent reasoning. The theoretical storage footprint of the model becomes ~1.6 KB (840 BF16 adapter scalars + 8-byte seed).

**Idea 3 — Custom C/AVX2 inference kernels:**
BitNet's ternary weights mean matmuls reduce to add/subtract/skip operations with no multiplications. PyTorch doesn't exploit this. A hand-tuned AVX2 "soft-chip" kernel was built to achieve 4.6x forward and 75x backward speedup on CPU, making GRPO training on this hardware practical at all.

The project is well-documented across 13 iterative "passes" in `FINDINGS.md`, each representing a research iteration: hypothesis, implementation, measurement, decision. The total pass progression covers model selection, kernel development, Vulkan iGPU exploration, backward pass optimization, GRPO loop implementation, GhostWeight exploration, SVD compression attempts, and architecture evolution.

---

## 2. Novel Contributions

### 2.1 Sub-rank-1 LoRA on a Natively 1.58-bit Architecture

**What was done:** TinyLoRA adapters (a single trainable scalar per layer, applied through fixed random rank-1 projection vectors) were attached to all 210 `AutoBitLinear` layers of BitNet b1.58 2B4T. Gradient flow was validated, GRPO updates were applied, and task performance on GSM8K was measured.

**Why it's novel:** Prior TinyLoRA work (Morris et al.) operated on standard float models (Qwen2.5-8B). BitNet b1.58's `AutoBitLinear` layers quantize BF16 master weights to {-1, 0, +1} during every forward pass via a `WeightQuant` (absmean) operation. The question of whether continuous-valued LoRA corrections can receive gradients through this discrete quantization is non-obvious. The answer is yes — because the LoRA bypass path `output = Base(x) + scale * (x @ v.T) @ u.T` operates entirely on full-precision activations `x` and never touches the ternary weight matrix directly. The gradient flows cleanly through the bypass, and the ternary base layer receives zero gradient (frozen). BitNet's STE (Straight-Through Estimator) means the discrete quantization is invisible to the backward pass.

**The result:** With 4 scalar parameters (one per adapted layer, 4 layers), gradient magnitudes of 5e-6 to 1.3e-5 were observed on a single update. With 210 parameters (all AutoBitLinear layers) in the full GRPO setup, adapter scales grew monotonically from 0 to ~0.034 over 30 training steps, confirming consistent gradient direction. A running reward of ~91% was achieved at step 220 with real BitNet weights.

**The claim to check:** The FINDINGS.md says "to our knowledge, this is the first demonstration of sub-rank-1 LoRA on a natively 1.58-bit architecture." This appears credible and is a clean, reproducible result that stands independently of the GhostWeight narrative.

**What it is not:** This is NOT the 1.6KB GhostWeight model. See Section 5.

---

### 2.2 AVX2 LUT Ternary Matmul + Backward Kernel

**What was done:** A hand-tuned C/AVX2 kernel was built for the forward and backward pass through ternary weight matrices. Key design choices:

- **Weight packing:** 2 bits per weight (code: 00=zero, 01=+1, 11=-1), 4 weights per byte. A 2560×2560 weight matrix compresses to ~1.6 MB — fits in L2 cache.
- **LUT decode (v2/v3):** Precompute two 256-entry lookup tables mapping each byte (4 ternary codes) to nonzero masks and sign masks. Inner loop: two LUT lookups → AVX2 load → XOR activations with sign mask → AND with nonzero mask → accumulate. No multiply instruction in the hot path. ~51% of weights are zero (free).
- **Smart threading (v3):** For M<6 (autoregressive generation), OpenMP fork/join overhead (~50μs × 12 threads) dominates 1.6ms single-core compute. v3 uses the serial path below the threshold, falling back to parallel for larger batches. This gives 4.3x improvement over the threaded path at M=1.
- **Backward kernel:** The STE backward for ternary weights is `grad_input = W^T @ grad_output`. Since W is ternary, this is the same add/subtract/skip pattern as the forward — exploited with the same packed-weight + accumulate-scatter approach.

**Benchmark results (2560×2560, M=1):**

| Implementation | Time | Speedup |
|---|---|---|
| PyTorch BF16 (BitNet default) | 53.5 ms | 1x |
| Soft-chip v1 (scalar decode) | 221.1 ms | 0.24x |
| Soft-chip v2 (LUT + AVX2) | 13.2 ms | 4.1x |
| Soft-chip v3 (smart threading) | **1.6 ms** | **33x** |

Full model forward (M=1, 30 layers, 210 AutoBitLinear):
- PyTorch: 4.2s → Soft-chip v3: 0.91s (**4.6x speedup**)
- 200-token rollout: ~840s → ~182s

Full backward speedup breakdown:
- Without optimization: 82.9s
- After ternary backward kernel: 19.5s (4.3x)
- After FP32 LM head (Section 2.3): 1.1s (**75x total**)

**Why it's novel:** Dedicated ternary matmul kernels exist in the literature (e.g., Microsoft's own BitBLAS), but the combination of a PyTorch-integrated custom autograd backward kernel exploiting ternary STE structure, with quantified end-to-end speedup including the backward pass, for use in RL training on CPU, is a new composition. The PyTorch integration (monkey-patching via `patch_model()`, custom `torch.autograd.Function` for both forward and backward) is clean and reusable.

**Numerical validation:** Cosine similarity against PyTorch reference ≥ 0.9997 across all 6 tested layer shapes. Differences from stock PyTorch are attributable to BF16 vs FP32 arithmetic, not ternary logic errors.

---

### 2.3 FP32 LM Head on Non-AMX CPU (Critical Performance Fix)

**What was done:** Profiling revealed that 92.6% of the backward pass time (after the ternary kernel was applied to all 210 AutoBitLinear layers) came from the LM head — a single `nn.Linear(2560, 128256)` dense BF16 layer. Two matmul calls (forward and backward through this layer) consumed ~18 seconds each.

**Root cause:** MKL's BF16 GEMM on Zen 3 architecture (Ryzen 5 PRO 5675U) is **32–90x slower than FP32** for the same matmul. Zen 3 lacks AMX (Intel) and VNNI (also largely Intel) instructions for accelerated BF16 computation. MKL falls back to an emulated BF16 path that is catastrophically slow compared to native FP32 BLAS.

**The fix:** `FP32LMHeadFunction` — a custom `torch.autograd.Function` that:
1. Casts input BF16 → FP32 at the boundary
2. Stores a pre-converted FP32 copy of the weight matrix (one-time cost at init)
3. Performs all matmuls in FP32
4. Casts output FP32 → BF16 at the boundary
5. In backward: casts grad BF16 → FP32, matmuls in FP32, returns grad in BF16. No grad for weight (frozen).

**Memory cost:** +657 MB for the FP32 copy (trivial with 64 GB RAM).

**Result:** LM head backward: 18.5s → 200ms (**92x speedup on that layer**). Total backward: 19.5s → 1.1s (**18x**). Total training iteration: 20.7s → 2.4s.

**Why it's useful:** Any project training a BF16 language model on an AMD CPU without AMX/VNNI will hit this exact problem. The fix is generalizable and the root cause (MKL BF16 fallback path) is non-obvious and poorly documented. The diagnosis (profiling revealing one non-ternary layer consuming 92.6% of backward time) is a good case study in where to look for bottlenecks in quantized model training.

---

### 2.4 Flat Singular Spectrum of Ternary Matrices

**What was done (Pass 9):** Attempted to compress BitNet ternary weights via SVD — truncate to top-k singular vectors and share basis vectors across layers. Results:

- SVD reconstruction error at rank-64: **82.5%** (vs ~2% for dense FP32 matrices)
- Cross-layer singular vector cosine similarity: **mean ≈ 0.000** — near-perfect orthogonality between layers
- Conclusion: ternary matrices' information content is uniformly distributed across all singular dimensions — they have flat singular spectra

**Why it matters:** This is a fundamental mathematical property of ternary matrices. It explains why SVD-based compression cannot beat packed 2-bit storage for ternary weights, and it establishes a theoretical lower bound on compression ratios for this class of models. It also has implications for low-rank approximation, pruning, and knowledge distillation on ternary architectures. The result supports the GhostWeight hypothesis as the only viable path to sub-MB storage: if SVD can't compress, and the weights contain no low-rank structure, then the only way to get below the packed 2-bit floor is to not store them at all.

---

### 2.5 Threading Interaction Between Custom Kernels and MKL (Negative Result)

**What was done (Passes 3 and 6):** Two separate attempts at OpenMP parallelization of custom kernels failed to improve end-to-end performance despite significant kernel-level speedups.

- **Pass 3 (forward kernel):** For M<6, OpenMP thread fork/join overhead (50μs × 12 threads = 600μs) exceeded the 1.6ms single-core compute time. Serial path is 4.3x faster than threaded for autoregressive generation.
- **Pass 6 (backward kernel):** OpenMP over the N dimension achieved **3.4x speedup in isolation** (548ms → 162ms, validated across all 4 layer shapes). End-to-end result: **9% regression** (1,065ms → 1,160ms). Five configurations tested (thread counts 3/6/12, OMP_WAIT_POLICY, higher activation thresholds) — none beat serial.

**Root cause of Pass 6 regression:** Rapid alternation between the custom OpenMP kernel and PyTorch's MKL thread pool (both claiming 12 threads on 6 physical cores) caused cache thrashing and OS scheduling overhead that exceeded the kernel speedup. The interaction is consistent across sessions and hardware states — this is a real ceiling, not a tuning failure.

**Why it's useful to document:** This is the kind of result that takes weeks to discover experimentally and is rarely published because it's a negative result. The pattern — micro-benchmark parallelism that doesn't transfer when multiple thread pools compete — is common in ML systems work and worth recording.

---

## 3. Useful and Understated Work

### 3.1 Bug Documentation (KNOWN_BUGS.md)

The `docs/KNOWN_BUGS.md` file contains nine documented bugs with root cause analysis, exact reproduction steps, and fixes. This is exceptional documentation quality for a solo research project and substantially more useful than the headline contributions to anyone attempting to reproduce this work. Selected highlights:

**BUG-001: torch.Generator silent hang on seed ≥ 2^63**
- SHA-256 truncated to 8 bytes produces values in `[0, 2^64)`. `torch.Generator.manual_seed()` silently hangs (no exception, no crash, process stays alive at ~100% CPU) for values ≥ 2^63.
- Symptom: training hangs silently after model load, log shows only `Loading weights: 100%`. Impossible to diagnose without knowing this specific PyTorch behavior.
- Fix: `seed % (2**63)` before any call to `manual_seed`.
- This bug will affect any code that hashes strings to produce torch seeds without bounds-checking.

**BUG-004: BF16 tensor data pointers passed as float32 to C struct**
- BF16 tensors are 2 bytes/element; C `float*` fields are 4 bytes/element. Passing raw `.data_ptr()` without conversion causes the C code to read garbage for every expert parameter. Silent data corruption — no crash, no assert, just wrong results.
- The lifetime issue (converted tensors must be kept alive until after the C call returns, preventing garbage collection) is correctly handled via `self._f32_buf`.

**BUG-005: Python GhostChain serial coupling mismatches C engine parallel addition**
- GRPO correctness requires that the same model computes both the rollout tokens and the log-probs for the policy gradient. With the Python model using serial coupling (expert2 receives `x + d1`) and the C engine using parallel coupling (all experts read from original `x`), the gradient points in the wrong direction.
- Practical magnitude at current scale (~0.14): O(scale²/dim) ≈ 8×10⁻⁶ per layer — small, but growing with scale and fundamentally wrong.

**BUG-008: GRPO loss not length-normalized**
- `sum()` over log-probs gives 128× the gradient magnitude for a 128-token completion vs a 1-token completion. Implicitly incentivizes short answers over chain-of-thought.
- Fix: `.mean()` instead of `.sum()`.
- This is a common subtle bug in GRPO/REINFORCE implementations and the analysis here is clear.

---

### 3.2 Curriculum Filtering Analysis

The problem: the base BitNet model already solves ~87% of GSM8K training questions without any adaptation. A naive GRPO loop produces `rewards = [1,1,1,1]` on 81% of steps, giving `std_r < 1e-8`, skipping the update, and wasting the full rollout budget (~320s per step) with no learning signal.

The solution implemented (`USE_CURRICULUM_FILTER = True`):
1. Before each training step, greedily decode the current model on the next candidate problem.
2. If the model answers correctly (under `strict_extract_answer` — no last-number fallback), skip this problem and try the next.
3. Only train on problems the current model fails.
4. If no failure found in `CURRICULUM_SKIP_MAX=20` tries, skip the step entirely rather than training on easy problems.

The `strict_extract_answer` vs `extract_answer` distinction (BUG-007) is subtle: the permissive fallback to last-number in the answer would incorrectly mark a problem as "solved" if the last number in the generated text coincidentally matches the answer. The strict version requires an explicit answer format marker.

The timing decomposition (curriculum probe: 5–20s, G=4 rollout: 250–320s, gradient update: 150–300s) correctly identifies that curriculum filtering is not the bottleneck and can be left on without significant cost.

---

### 3.3 KV-Detach Backward Optimization

The `compute_log_probs_batch_kv()` function splits the policy gradient forward pass into:
1. Prompt tokens run under `torch.no_grad()` with `use_cache=True` → KV cache built for all layers
2. Completion tokens only run with grad enabled, attending over cached prompt KV as constants

This reduces the backward graph to completion tokens only. The reasoning is sound: adapter scale parameters affect completion processing at first order. Their contribution through the prompt path is O(scale²) ≈ 4×10⁻⁴ at current scale magnitudes — negligible.

An important caveat is noted in the code: attention backward still traverses the full sequence (completion tokens attend over all prompt+completion positions in the KV cache). Only feedforward backward is reduced. The actual measured speedup (~16% on non-degenerate steps) was lower than the theoretical 40–57% prediction, and this discrepancy is explained honestly.

The one-token skip (first completion token's log-prob excluded because its logit comes from the no_grad prompt forward) is acknowledged: "1 of 96 tokens ≈ 1% bias, acceptable."

---

### 3.4 Rollout Generation Quality Controls

Two non-obvious implementation issues in few-shot prompted rollout generation were found and fixed:

**Stop condition false fires:** The initial stop condition `_should_stop` fired on any `Q:` token appearing mid-sentence, collapsing generations to near-zero length. Fix: require `\nQ:` (newline prefix) — a continuation marker always appears after a completed answer + newline.

**Answer extraction contamination:** With few-shot prompting, the model often answered correctly and then continued generating a synthetic next example. Naive "last number wins" extraction scored the wrong number (the number in the next synthetic question). Fix: strip everything after the first `\nQ:` before parsing.

These are subtle issues that would corrupt reward signals in ways that are hard to diagnose from training logs alone.

---

### 3.5 Vulkan iGPU Backend (Instructive Negative Result)

A full Vulkan compute shader backend was built for the Vega 7 iGPU (7 CUs, 448 shaders, 1.61 TFLOPS FP32). The work includes:
- GLSL compute shader with branchless 2-bit ternary decode
- XOR+AND bit trick to eliminate float multiply from inner loop
- Fully unrolled 16-weight decode per loop iteration
- LDS sizing via Vulkan specialization constants (per-variant for 2560 vs 6912 K-dim layers)
- Batched command buffer dispatch to amortize submit overhead
- Full PyTorch integration as `backend="vulkan"` in `patch_model()`

**The A/B test result that contradicted theory:** LMM analysis predicted transposed weight layout would be the top optimization (better cross-thread coalescing). A/B testing showed the opposite: row-major beat transposed. Explanation: on this 7-CU Vega with ~1 MB L2 cache, the L2 prefetcher's sequential access pattern (row-major) was more valuable than cross-thread coalescing. The winning shader (v3) kept row-major but added the XOR/AND bit trick and full unrolling.

**Why the iGPU ultimately loses:** 210 Vulkan dispatches × ~1ms submit overhead (command buffer record → queue submit → fence wait → readback) = ~210ms pure overhead, exceeding the total CPU kernel time. CPU wins because kernel dispatch is a direct function call with zero overhead. Even batching (3 dispatches per command buffer) only saves ~0.5ms per decoder block.

**The value of doing this:** The Vulkan shader work directly validates the ternary matmul optimization approach and will be transferable to CUDA on hardware where submit overhead is ~10μs instead of ~1ms (e.g., Jetson Thor).

---

## 4. Possibly Paper-Worthy

There are four independent contributions here with varying levels of polish and evidence:

### 4.1 "Sub-rank-1 LoRA on Natively Ternary LLMs" (near-ready)

**The paper:** A short technical paper demonstrating that continuous LoRA adapters can be trained on BitNet b1.58 via RL (GRPO), showing gradient flow analysis, adaptation dynamics, and task performance on GSM8K.

**Strength:** Clean result with solid quantitative validation. The gradient flow analysis (why the bypass path doesn't touch ternary weights) is correct and worth formalizing. The GRPO curriculum filtering details (BUG-007, BUG-008, BUG-009) are practical contributions to the RL fine-tuning literature.

**Gap:** The paper needs a complete training run with a proper evaluation curve (accuracy vs. step count) and comparison against the base model. The 91% reward at step 220 is promising but reward ≠ accuracy and eval was broken at that point (BUG: eval was calling the ghost engine instead of the real model, per FINDINGS.md Pass 13 §5).

**Closest related work:** Morris et al. "Learning to Reason in 13 Parameters" (2025) on Qwen2.5-8B. The BitNet-specific contribution is the ternary quantization analysis and the forward/backward kernel stack.

---

### 4.2 "Efficient CPU Training of Ternary LLMs: Forward, Backward, and RL" (strong infrastructure paper)

**The paper:** AVX2 LUT ternary matmul kernel + ternary backward kernel + FP32 LM head fix + batched GRPO loop on CPU. The combined system achieves 75x backward speedup over stock PyTorch and enables practical RL training on commodity hardware.

**Strength:** The numbers are well-measured and validated. The technique (2-bit packing + 256-entry LUT + XOR/AND + STE backward) is clearly described, reproducible, and generalizable to any ternary architecture. The FP32 LM head discovery is independently useful and clearly explained.

**Gap:** The paper should include a full end-to-end benchmark against alternatives (existing BitBLAS, llama.cpp with BitNet, etc.) for proper positioning. There is also no discussion of precision: the kernel uses INT8 activation quantization matching BitNet's `ActQuant`, but this should be formally analyzed.

---

### 4.3 "GhostWeight: Training RL Adapters on PRNG Weight Bases" (open hypothesis)

**The paper this would be:** A demonstration that GRPO adapter training converges meaningfully when the base model weights are PRNG-generated rather than trained. If it works: a 1.6KB model achieving >50% GSM8K accuracy would be a striking result. If it doesn't work: a careful analysis of why (activation statistics divergence, adapter capacity, weight scale calibration) would still be publishable.

**Current status:** Hypothesis only. `USE_GHOST=False` in all active training. The GhostWeight path is 200x slower than the real-weights path on this hardware. No extended GhostWeight training run has been completed successfully.

**What needs to happen:** Either (a) the GhostWeight path needs to be made fast enough to train (requires different hardware or a batching strategy that amortizes PRNG computation cost), or (b) a shorter proof-of-concept needs to demonstrate any non-trivial convergence under PRNG weights, even just 10-20% GSM8K accuracy with a clear improvement curve.

---

### 4.4 "Singular Spectrum of Ternary Matrices: Implications for Compression" (short note)

**The result:** Ternary {-1, 0, +1} matrices have flat singular spectra (82.5% reconstruction error at rank-64) and near-zero cross-layer singular vector similarity (cosine ≈ 0.000). This rules out SVD-based compression and low-rank factorization as viable approaches for ternary model compression.

**The value:** Short but clean. Provides a theoretical grounding for why GhostWeight is necessary rather than just novel, and establishes a practical lower bound on compression ratios achievable by rank-reduction methods on ternary architectures.

---

## 5. The Central Claim Is Unvalidated

This is the most important finding in this audit.

### 5.1 The Claim

The README states:

> A 2.41-billion parameter language model can reason using only **~1.6KB of learned parameters**, if the 500MB weight matrix is replaced by a deterministic PRNG seeded with 8 bytes.

The README's Key Results table includes "Running reward (TinyLoRA on real weights) **~91% at step 220 (prior baseline)**" alongside the 1.6KB claim, implying the 91% reward was achieved by the 1.6KB model. This reading is incorrect.

### 5.2 What Actually Happened

`grpo_train.py:65`:
```python
USE_GHOST = False  # GhostWeight: π_ref is random noise
```

Comment at this line:
> GhostWeight (PRNG on-the-fly generation) is correct but slow on CPU: each batched_ghost_matmul allocates+fills a 4MB temp buffer 79K times per rollout, which is impractical at training time. Validate curriculum filter + scalar adapters on real weights first; tackle PRNG perf separately.

FINDINGS.md Pass 13, §1:
> Each rollout step requires ~79,000 matmul calls (210 layers × 4 experts × ~94 tokens). For each call, the C engine must: (1) generate a 2560×2560 matrix from PRNG — ~4.2MB of computation; (2) apply AVX2 matmul; (3) discard the matrix (never stored). Weight regeneration dominates. The ghost engine is ~200× slower than the Python soft-chip path.
>
> **Decision:** `USE_GHOST=True` is archived.

### 5.3 What Was Demonstrated

What has been demonstrated is:
- **TinyLoRA on real BitNet weights via GRPO works.** 210 adapters (1 scalar/layer), trained on real stored ternary weights, achieve ~91% running reward at step 220 on GSM8K.
- **GhostWeight produces coherent output with correct weight scales** (Pass 10 fix), but has not been trained beyond a handful of steps due to the 200x speed penalty.
- **GhostWeight adapter transfer is impossible** (Pass 11): adapter scales trained under real BitNet weight activations produce `pred=None` (incoherent output) under PRNG weights. They must be trained from scratch with the PRNG base.

### 5.4 The Open Question

Is PRNG ternary noise + 840-scalar adapters actually learnable at all? The architecture docs argue it theoretically (Section 2.15 of ARCHITECTURE.md: "random weights contain every possible direction in activation space with equal probability"), but this is an argument for why it *could* work, not evidence that it does.

The theoretical argument is:
1. In expectation, random ternary weights project activations into random subspaces.
2. The adapters can learn to steer these random projections toward task-relevant directions.
3. Serial coupling means each adapter corrects the residual of the previous one.

The practical obstacle is compute: generating a 2560×2560 ternary matrix from PRNG takes ~4ms on this CPU; doing it 79,000 times per rollout step takes ~316 seconds just for weight generation, before any attention or MLP computation. The only viable paths to make this practical are:
- Hardware with faster PRNG (GPU parallelism, or dedicated hardware)
- Batch sizes large enough to amortize weight generation over many activations
- A different PRNG scheme that generates weights in smaller, cacheable chunks

Until one of these is solved, the headline claim cannot be validated on this hardware.

---

## 6. Architecture Story Has Drifted From Implementation

### 6.1 GhostChain Serial Coupling Is Documented but Not Implemented

The README, ARCHITECTURE.md, FINDINGS.md Pass 12, and GHOST_ENGINE.md all describe the GhostChain architecture with **serial coupling** between experts:

```
x1  = x  + s1 · LoRA1(x)
x2  = x1 + s2 · LoRA2(x1)   ← expert 2 receives x + d1 as input
x3  = x2 + s3 · LoRA3(x2)   ← expert 3 receives x + d2 as input
y   = Base(x) + (x3 - x) + sobs · LoRAobs(x3)
```

The stated benefits of serial coupling:
- Gradient of s1 depends on s2, s3, sobs (gradient coupling → cooperative optimization)
- Each downstream expert conditions on the accumulated correction (progressive refinement)
- Can approximate a rank-4 correction through multiplicative interaction terms

The actual `GhostChain.forward()` in `grpo_train.py:257–261`:

```python
d1 = self.expert1(x)    # all four read from same x
d2 = self.expert2(x)
d3 = self.expert3(x)
correction = self.observer(x)
return self.base_layer(x) + d1 + d2 + d3 + correction
```

This is **parallel coupling** — all experts read from the same input `x`. The docstring explains: this was changed in BUG-005 to match the C engine, which was always parallel.

**The practical consequence:** The current GhostChain is mathematically equivalent to a single rank-4 LoRA adapter with four fixed random directions (determined by the SHA-256 layer name hash) and four independent trainable scalars. It is *not* a serial chain. The gradient coupling properties described in ARCHITECTURE.md do not apply. The `math_validation_v2.py` result (serial coupling converges lower than parallel) validates a design that is not in the active training path.

**The ARCHITECTURE.md document is currently inconsistent with the code.** Anyone reading the docs and then the code will find a discrepancy.

### 6.2 The C Ghost Engine Is Archived But Prominently Featured

`softchip/ghost_engine.c`, `softchip/ghost_engine_wrapper.py`, and `softchip/build_engine.sh` receive a prominent call-out in the README's project structure, the Quick Start instructions, and the architecture diagram. The README says:

```
# 1. Compile the Ghost Inference Engine (C/AVX2)
bash softchip/build_engine.sh
```

In the active training configuration, `USE_GHOST=False`, which means the C ghost engine is initialized at startup (taking memory and time) but its `rollout_batch()` function is never called. The rollout dispatch function falls through to `python_rollout()` which uses PyTorch model generation with real ternary weights. The C ghost engine is dead code in the current training loop.

This is called out in FINDINGS.md Pass 13 but not yet reflected in the README.

### 6.3 Eval Was Broken for Steps 0–30

FINDINGS.md Pass 13 §5 documents a critical bug: `evaluate()` was hardcoded to call `engine.rollout_batch()` (the C ghost engine) regardless of the `USE_GHOST` flag. With `USE_GHOST=False`, this means every eval used the C engine's random PRNG weights instead of the trained model.

Evidence: step 25 eval showed `pred=None` for all 20 eval examples, 0% accuracy. This was not low accuracy — the ghost engine was generating incoherent token sequences with no parseable answers.

**All eval results from steps 0–30 are invalid.** The first valid eval is step 50 (after the fix). Any training curve or accuracy measurement from the early steps of the current run should be treated as garbage.

---

## 7. Fluff and Oversold Claims

### 7.1 The 1.6KB Model in the README Key Results Table

The README presents a "Key Results" table that mixes results from different experimental contexts:

| Metric in README | Reality |
|---|---|
| "Model weights storage: 8 bytes + ~1.6KB" | Theoretical, never trained at scale |
| "Running reward ~91% at step 220 (prior baseline)" | Real BitNet weights, USE_GHOST=False, not the 1.6KB model |
| "Forward speedup: 4.6x" | Real ternary weights path, not ghost path |

A reader of the README would reasonably conclude that the 1.6KB model achieves 91% reward. It does not. This is the most significant overstating in the project.

### 7.2 The "1112x Startup Speedup"

The commit message "Perf: 1112x scale loading speedup — pre-extract 210 weight scales to 840-byte file, eliminate BF16 tensor iteration at startup" is technically correct but contextually misleading.

The baseline being compared against is: loading the full 4.5GB BF16 model and calling `.abs().mean()` on all 210 weight matrices to extract their scales at every training startup. This was never a sensible approach. The 1112x speedup is real, but the baseline being beaten was self-inflicted overhead. A better framing: "eliminated unnecessary per-startup scale computation."

### 7.3 GhostChain "4× Degrees of Freedom Over TinyLoRA"

The README says: "This provides 4× the degrees of freedom of TinyLoRA while adding negligible compute overhead over the dominant ghost matmuls."

The 4× claim is true in a narrow sense (840 parameters vs 210). But the description conflates capacity with effectiveness. Four parallel rank-1 adapters with independent fixed random bases are not 4× more powerful than one rank-1 adapter:
- They span up to 4 independent directions (vs 1 for TinyLoRA), which is a real benefit
- But all four directions are fixed at init time from random seeds — they may not span the most relevant subspace for the task
- Standard rank-1 LoRA (1 per layer, but with trainable u and v vectors) would be more parameter-efficient for the same task

The comparison should be: GhostChain (4 fixed random directions, 4 scalars) vs rank-4 LoRA (4 learned directions, 8192 parameters for d=2560). GhostChain wins on parameter count but loses on expressivity per direction.

### 7.4 The Serial Coupling Advantage

Multiple documents claim the serial coupling architecture provides compounding benefits through gradient coupling. As established in Section 6.1, the serial coupling is not in the current code. The `math_validation_v2.py` result supporting this claim validates a design that was reverted.

### 7.5 Log File Accumulation

The repo root contains 14+ log files from experimental runs, many abandoned:

```
grpo_v3_run.log, grpo_batched.log, grpo_batched_recovery.log,
grpo_extended.log, grpo_extended_v2.log, grpo_fast.log,
grpo_final.log, grpo_ghost_chain.log, grpo_ghost_recovery.log,
grpo_ghost_recovery_v2.log, grpo_ghost_unified.log,
grpo_kv_detach.log, grpo_remediated.log, grpo_scaled_run.log
```

This is normal accumulation during active research but makes it difficult for a newcomer to identify the canonical training run. There is no README-level pointer to the current active log.

---

## 8. Code Quality Assessment

### 8.1 Strengths

**C kernels (ternary_matmul_v3.c, ghost_matmul_lut.c, ghost_engine.c):**
- Clear structure with explicit section headers
- AVX2 intrinsics are correctly used and well-commented
- LUT approach is idiomatic and well-explained in comments
- PRNG implementation (SplitMix64 + XorShift128+ in AVX2) is correct and clearly structured
- The `build_luts()` / `g_luts_ready` guard is proper lazy initialization

**torch_ternary.py:**
- `PackedWeight.__del__()` correctly frees C-allocated memory via libc
- `GhostMatmulFunction.backward()` correctly regenerates the PRNG matrix for the backward pass (STE)
- The `make_ghost_forward()` closure correctly captures `lid, K, N, seed, scale` by value (avoids Python closure-over-loop-variable bug)
- `patch_model()` correctly saves original `module.forward` before replacing it, enabling clean `unpatch_model()`

**grpo_train.py:**
- The `strict_extract_answer()` vs `extract_answer()` distinction is well-motivated and correctly applied
- `random.Random(SEED).shuffle(train_data)` (isolated RNG instance) is the right fix for deterministic shuffles across restarts
- The KV-detach backward comment correctly quantifies the gradient approximation error
- Checkpoint structure includes `problem_ptr` and `eval_history` — proper resume state

**KNOWN_BUGS.md:**
- Exceptional quality for a solo research project
- Each bug has: symptom, root cause, fix, verification, and "why it's hard to diagnose" where applicable
- BUG-001 includes a minimal reproduction script — a high bar for bug documentation

### 8.2 Issues

**grpo_train.py — docstring inconsistency at GhostChain.forward() (line ~247):**
The docstring describes the parallel coupling but references the old README serial design as context. The claim "Serial coupling (feeding x+d1 into expert2, etc.) is mathematically equivalent to parallel at small scales in high dimension" is offered as justification, but this justification is only approximately true (it relies on near-orthogonality of random unit vectors in R^2560, which improves with dimension but is not exact). The docstring is correct in its conclusion (use parallel) but misleading in its framing.

**grpo_train.py:65 — USE_GHOST = False while README promotes GhostWeight:**
The configuration file contradicts the project's headline claim. A comment should clarify the current state for anyone who clones the repo.

**grpo_train.py — optimizer is constructed before resume checkpoint is loaded:**
```python
optimizer = torch.optim.Adam([...], lr=LEARNING_RATE, betas=ADAM_BETAS)
if resume_path:
    optimizer.load_state_dict(ckpt["optimizer_state"])
```
This is correct (Adam state is overwritten by `load_state_dict`), but the LR schedule starts from `start_step` and ignores the fact that optimizer momentum state was saved at the checkpoint LR, not the resumed LR. Likely benign for Adam with cosine/linear decay but worth noting.

**ghost_engine.c — no bounds check on packed_row_bytes:**
`decode_row_packed()` computes `K/4` as `packed_row_bytes` assuming K is divisible by 4. For K=2560 and K=6912 this holds, but a future caller with non-multiple-of-4 K would write out of bounds silently.

**grpo_log.jsonl — no schema documentation:**
The JSONL training log format (keys: step, rewards, loss, mean_grad, mean_abs_scale, skipped, rollout_time, update_time, total_time, predicted_answers, ground_truth) is not documented anywhere. Anyone trying to parse the training history would need to reverse-engineer the schema from the code.

**Multiple test files with unclear current status:**
`test_crash.py`, `test_gmoe_logic.py`, `math_validation.py`, `math_validation_v2.py` are untracked files with no documentation of their purpose or whether their tests pass with the current code.

---

## 9. What Would Make This Paper-Worthy

### Path A: Infrastructure Paper (Near-Ready)

**"Efficient CPU Fine-Tuning of Natively Ternary LLMs via GRPO"**

This paper is mostly written. The required additions:
1. A full training run (steps 0–300+) with eval every 25 steps showing accuracy vs step count
2. A baseline comparison: base BitNet b1.58 without adapters, TinyLoRA (1 scalar/layer), GhostChain (4 parallel scalars/layer)
3. Brief related work section: TinyLoRA (Morris et al.), BitBLAS, llama.cpp BitNet
4. Fix the eval bug before running the final eval curve (fixed in current code as of step 50)

Content that should be in this paper from existing work:
- TinyLoRA gradient flow validation (Steps 1-5 table in FINDINGS.md §5-step validation)
- AVX2 ternary kernel design and benchmark (FINDINGS.md §AVX2 Soft-Chip section)
- FP32 LM head discovery and fix (Section 2.3 of this audit)
- GRPO implementation details: curriculum filtering, answer extraction, length normalization, KV-detach (all documented in KNOWN_BUGS.md and FINDINGS.md)
- Flat singular spectrum negative result (Pass 9)

Estimated time to complete: 1–2 weeks of training + 1 week of writing.

---

### Path B: GhostWeight Paper (Requires Validation)

**"GhostWeight: Reasoning with PRNG-Generated Ternary Weights"**

This is the most ambitious and most interesting paper, but requires answering the open question first. The required steps:

1. **Make the GhostWeight training path fast enough to run.** Options:
   - Port to a GPU where PRNG can generate weight matrices in parallel (the LUT kernel is already AVX2; a CUDA port would parallelize across rows)
   - Use a batching strategy where M activations are multiplied against one generated weight matrix (amortize generation cost over M)
   - Cache generated weight matrices for the full forward pass (800MB memory cost — feasible on this machine)

2. **Train from step 0 with USE_GHOST=True for 300+ steps.** The adapter scales must be trained from scratch on PRNG weights; transferring from real-weights training is documented as failing (Pass 11).

3. **Measure convergence:** Does accuracy improve at all? At what rate? What is the asymptotic performance?

4. **If it works:** Compare against (a) full real-weights fine-tuning, (b) TinyLoRA on real weights, (c) random 840-scalar baseline.

5. **If it doesn't work:** Document why. Is the PRNG weight distribution too far from the trained weight distribution for adapters to bridge? Is 840 scalars insufficient? What adapter capacity (rank-1 expansion) would be needed?

Either outcome is publishable. The negative result (PRNG weights + 840 scalars → no convergence, with analysis of why) would be a valuable contribution to the emerging literature on extreme model compression.

---

### Path C: GhostChain Serial Coupling (Architectural Experiment)

**"Serial Expert Coupling in Sub-rank-1 Adapters"**

This is the smaller experiment that was started (Pass 12) but never completed due to the Python-C mismatch and subsequent decision to use parallel coupling.

The hypothesis is: serial expert coupling (expert 2 conditions on expert 1's output) provides provably stronger gradient coupling and empirically better convergence than 4 parallel independent adapters.

The `math_validation_v2.py` result supports this on a toy target. What's needed:
1. Update the C engine to support serial coupling (currently hardwired to parallel)
2. Run matched comparison: 840 parallel scalars vs 840 serial scalars, same training setup
3. Measure convergence rate, final accuracy, and gradient coupling coefficients

This is a self-contained experiment that could be completed in ~1 week and would validate or invalidate the core architectural claim of GhostChain.

---

## 10. Summary Table

| Contribution | Status | Novelty | Evidence Quality | Paper-Worthy? |
|---|---|---|---|---|
| Sub-rank-1 LoRA on natively ternary BitNet | Validated | High | Strong (gradients, scale dynamics, reward curve) | Yes — short paper |
| AVX2 LUT ternary matmul kernel | Validated | Medium | Strong (4.6x forward, numerical validation) | Yes — infrastructure |
| Ternary STE backward kernel | Validated | Medium | Strong (4.3x, all shapes, end-to-end) | Yes — part of above |
| FP32 LM head on AMD CPU (BF16 GEMM bug) | Validated | Medium | Strong (18x speedup, root cause confirmed) | Yes — part of above |
| Flat singular spectrum of ternary matrices | Validated | Medium | Clean (82.5% reconstruction error) | Yes — short note |
| GRPO curriculum filtering on pretrained LLMs | Validated | Low-Medium | Good (timing decomposition, skip-rate analysis) | Useful supplement |
| GhostWeight (PRNG replaces stored weights) | Unvalidated | Very High | None (USE_GHOST=False, 200x too slow) | High potential, not yet |
| GhostChain serial coupling | Not implemented | High | Toy validation only (math_validation_v2.py) | Needs C engine update |
| Vulkan iGPU ternary backend | Validated (negative) | Low | Good (A/B testing, overhead measurement) | Useful negative result |
| Threading interaction / serial vs parallel | Validated (negative) | Low | Strong (3.4x kernel → 9% end-to-end regression) | Useful supplement |

---

## 11. File-by-File Notes

### grpo_train.py
- **Current state:** `USE_GHOST=False`, real BitNet weights, GhostChain (4 parallel scalars/layer), curriculum filtering, KV-detach backward, checkpointing every 10 steps
- **Known issues:** Docstring inconsistency in GhostChain.forward(); optimizer construction before checkpoint load is harmless but note for LR decay continuity
- **Key correctness fixes already applied:** BUG-001 through BUG-009

### softchip/torch_ternary.py
- **Current state:** Production. Supports CPU (AVX2), Vulkan (iGPU), Ghost (PRNG), and auto backends. FP32 LM head patch included.
- **Notable:** `PackedWeight.__del__` correctly calls `libc.free()` for C-allocated memory. The `make_ghost_forward` closure correctly binds by value to avoid loop variable capture bug.

### softchip/ghost_engine.c
- **Current state:** Archived (built but not called in active training path)
- **Code quality:** High. XorShift128+ in AVX2 registers is correctly implemented. LUT construction is lazy and thread-safe for single-threaded use. `ternary_dot()` loop correctly handles the K not-multiple-of-128 tail.
- **Issue:** `decode_row_packed()` assumes K is multiple of 4 (no bounds check for tail)

### softchip/ghost_engine_wrapper.py
- **Current state:** Initialized at startup (`GhostEngine` constructor runs), but `rollout_batch()` is never called with `USE_GHOST=False`
- **Notable:** BUG-004 fix (`_f32_buf` lifetime management) and BUG-002 fix (struct field order) are both present and correct

### docs/ARCHITECTURE.md
- **Current state:** Describes serial GhostChain coupling that is not in the code
- **Action needed:** Update to describe current parallel architecture; note that serial coupling is a future direction

### docs/KNOWN_BUGS.md
- **Current state:** All 9 bugs marked Fixed. Accurate.
- **Quality:** Exceptional. The root cause analysis on BUG-001 (torch.Generator silent hang) and BUG-005 (Python/C model mismatch in GRPO) is particularly good.

### docs/NEXT_STEPS.md
- **Current state:** Describes GMoE (multi-seed router), Rank-1 expansion, Meta-Ghost (evolutionary seed search), Recursive Ghosting, mobile deployment
- **Note:** These are all predicated on GhostWeight working in the first place. The immediate priority (validate that PRNG + adapters can converge at all) is understated relative to these downstream ideas.

### README.md
- **Current state:** Oversells GhostWeight; describes serial coupling that's not in code; shows C ghost engine as step 1 of quick start; blends USE_GHOST=False results with the 1.6KB narrative
- **Action needed:** Add a "Current Status" section that clearly states USE_GHOST=False, explains the speed problem, and separates validated results from open hypotheses

### FINDINGS.md
- **Current state:** 13 passes, ~700 lines, extremely detailed
- **Quality:** High. The most accurate and complete record of the project's actual state. Anyone auditing the project should read FINDINGS.md before the README — the README is aspirational, FINDINGS.md is empirical.

### math_validation_v2.py
- **Current state:** Untracked (not in git). Validates serial vs parallel GhostChain on a synthetic nonlinear target.
- **Note:** The result supports serial coupling, which is not in the current code. If serial coupling is ever re-implemented, this test should be the benchmark.

### extract_scales.py
- **Current state:** One-time utility to pre-extract weight scales from BF16 model tensors to a 10KB `.pt` file. Required for `USE_GHOST=True` startup.
- **Status:** Functions correctly. Output file (`models/bitnet-b1.58-2B-4T-bf16/weight_scales.pt`) is used by `patch_model(..., scales_path=...)`.

### bench_end_to_end.py, bench_softchip_model.py, bench_vk_model.py, bench_vk_dispatch_overhead.py
- **Status:** Benchmark scripts. Not part of training pipeline. Useful for profiling.

---

*End of audit. Total files reviewed: 27 Python files, 4 C source files, 6 Markdown documents, 1 JSONL log.*
