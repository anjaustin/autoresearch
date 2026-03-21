# Research Plan: Project GhostWeight
**Written:** March 2026
**Author:** External (Claude Code) — first-principles research direction assessment

---

## The Reframe

The project is currently organized around a product claim: *"a 2.41B parameter model can reason using only 1.6KB."* That framing has driven a lot of the work, but it has also created a structural problem: the most important experiment (does PRNG + adapters converge?) is blocked by hardware, while the work that *is* running doesn't directly test the headline.

I'd reframe the central research question entirely:

> **What is the minimum parameter budget for meaningful RL adaptation of a frozen ternary LLM, and where in the model does that budget need to go?**

This question is better because:
1. It has a spectrum of answers rather than a binary yes/no
2. It produces useful results regardless of whether GhostWeight works
3. It gives you a systematic path to *approach* the GhostWeight hypothesis from solid ground rather than jumping straight to it
4. The answer — which layers matter, how much, and what happens when you replace those layers with PRNG — is the GhostWeight story told rigorously

The paper at the end of this plan is more defensible, more interesting, and more achievable on this hardware than the current framing.

---

## What to Archive First

Before running another training step, clean house. The research debt is real and it makes every experiment harder to interpret.

### Archive to `research/archive/`

**`softchip/ghost_engine.c` and `softchip/ghost_engine_wrapper.py`**
These are ~700 lines of carefully debugged C and Python that are never called in the active training path. They represent a real investment and should be preserved, but they shouldn't be in the main `softchip/` directory implying they're part of the production stack. Move them. The `build_engine.sh` script currently builds both `ternary_matmul_v3.so` and `ghost_engine.so` — split it so the default build only produces what training actually uses.

**`softchip/vk_backend.c` and all Vulkan files**
The Vulkan result is worth keeping in FINDINGS.md as a documented exploration. The code can live in archive. It won't be needed until there's a port to hardware with a faster Vulkan submit path.

**All log files except the current active log**
`grpo_batched.log`, `grpo_extended.log`, `grpo_extended_v2.log`, `grpo_fast.log`, `grpo_ghost_chain.log`, `grpo_ghost_recovery.log`, `grpo_ghost_recovery_v2.log`, `grpo_ghost_unified.log`, `grpo_kv_detach.log`, `grpo_remediated.log`, `grpo_scaled_run.log`, `grpo_v3_run.log` — move all of these to `research/archive/logs/`. Keep `grpo_log.jsonl` (canonical JSONL) and `grpo_final.log` (current run). The clutter makes it impossible to tell at a glance which run represents the current state of the model.

**`math_validation.py`** (keep `math_validation_v2.py`)
`math_validation.py` is the earlier version. `v2` is the definitive serial-vs-parallel comparison. Keep one.

**`test_crash.py`, `test_gmoe_logic.py`**
Untracked experimental scripts. Either integrate them into the test suite properly or archive.

### Keep Everything Else

All `test_*.py` and `bench_*.py` files are useful and should stay. They're the validation and benchmarking infrastructure that proves the kernel claims.

---

## The Missing Baseline (Do This Today)

There is no published number in this repository for the following measurement:

**What accuracy does BitNet b1.58 2B4T achieve on the GSM8K test set under the exact evaluation setup used in training — few-shot format, greedy decode, 20-sample eval — with adapter scales set to zero?**

This is the most important missing number. You cannot claim improvement without it. The 87% figure in FINDINGS.md comes from curriculum filtering on the *training set* with greedy decode, which is different from the test set under the evaluation format. The reported "running reward" is not accuracy — it's the fraction of problems in the *curriculum-filtered* (already hard) training set where the model gets at least one correct answer out of 4 rollouts.

The measurement is trivial: call `evaluate()` once with all adapter scales manually zeroed. This is 20 greedy rollouts, about 30 minutes. Do it once for the record, write the number down, and never have to wonder again.

This baseline also establishes the target. If base BitNet scores 65% on the test set eval format, reaching 75% with adapters is a solid result. If it scores 82%, reaching 85% is modest. You don't know yet which situation you're in.

---

## Phase 0: Establish Clean Ground Truth (1 Week)

**Goal:** Know exactly where you are before deciding where to go.

### 0a. Base model accuracy

Run `evaluate()` with adapter scales zeroed on the GSM8K test set (20 samples, greedy, few-shot format). Record the number. This is the floor everything is measured against.

### 0b. Wait for step 50 eval

The first valid eval (after the BUG fix at step ~30) runs at step 50. This is the first real signal of whether the current training run is learning anything. Interpret it carefully:

- **Accuracy < base model:** Something is wrong. Check whether curriculum filter is selecting problems the model can plausibly learn, or whether the 840 PRNG-direction adapters happen to not span useful subspaces for these problems.
- **Accuracy ≈ base model:** The adapters are training (scales are growing) but not yet translating to new correct answers. This is the expected result at step 50 — base model is already very capable, and the margin for improvement is narrow.
- **Accuracy > base model:** The signal is clean and working. Proceed with confidence.

### 0c. Per-layer scale diagnostic

Starting from the current checkpoints (step 120+), extract the adapter scale magnitudes for all 840 parameters, grouped by:
- Layer index (0–29)
- Layer type (q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj)
- Expert index within GhostChain (expert1, expert2, expert3, observer)

Plot these. This is cheap to compute from existing checkpoints and will tell you more about what the adapters are actually doing than any other diagnostic. Almost certainly, a small fraction of layers are absorbing most of the signal.

```python
# Pseudocode — extractable from any checkpoint
ckpt = torch.load("checkpoints/grpo_step_0120.pt")
scales = ckpt["adapter_scales"]
# scales is a dict: {layer_name: scale_value}
# Group by layer type, plot distribution
```

---

## Phase 1: Complete and Characterize the Current Run (3–4 Weeks)

**Goal:** A clean accuracy curve from step 50 to step 300, with per-layer analysis at checkpoints.

This is not glamorous work, but it is the foundation. The current run is the best thing in the project and it deserves to be run to completion with proper measurement.

### 1a. Run to step 300

Continue the current training loop with no changes. The ~8.4 steps/hr throughput means step 300 is about 25 hours from step 120 — achievable over a few sessions.

Key things to watch:
- **Accuracy curve shape.** A flat curve after step 50 suggests the PRNG adapter directions don't span useful subspaces and you need trainable bases (Phase 3). A rising curve that plateaus suggests the 840-parameter capacity is the bottleneck.
- **Scale distribution evolution.** As training progresses, do the scales concentrate in specific layer types, or spread evenly? Concentration means sparsity is free for the taking.
- **Curriculum exhaustion rate.** If `CURRICULUM_EXHAUSTED` log lines increase over time, the model is solving a growing fraction of the training set. This is a good sign.

### 1b. Reduce eval cost

20 samples × ~100s/rollout = 33 minutes per eval pass. At step 300, you'll have run 12 evals for a total of 6+ hours of eval time — 28% of total wall time. Consider reducing to 10 samples until there's a stable signal, then returning to 20 for the final run. The improvement in eval statistics from 10→20 samples is modest (±5% confidence interval → ±3%), and the saved time is more valuable for training steps.

### 1c. Layer importance analysis (after step 300)

With the scale distribution in hand from 0a and the full scale evolution from the training run:

1. Rank all 210 adapter positions by their final mean absolute scale value
2. Compute the fraction of total adaptation (sum of |scales|) explained by top-K positions
3. This gives the "adaptation Lorenz curve" — almost certainly showing that 20-30 layers explain 80%+ of the total

This is the most important analysis in the project. It tells you whether sparse adaptation is viable, which informs everything downstream.

---

## Phase 2: Sparse Adaptation Experiment (2–3 Weeks)

**Goal:** Determine whether 840 parameters spread across 210 layers is better or worse than a smaller number of parameters concentrated in high-leverage layers.

This experiment is cheap to run (same training setup, just fewer adapters), produces a clean publishable comparison, and is the direct precursor to the partial GhostWeight experiment.

### 2a. Retrain from scratch with top-K layers only

Using the layer importance ranking from Phase 1, train new GhostChain runs with:
- Top-30 layers only (~120 parameters, 14% of 840)
- Top-60 layers only (~240 parameters, 29% of 840)
- Top-105 layers only (~420 parameters, 50% of 840)
- All 210 layers (840 parameters — baseline from Phase 1)

Each run goes to step 200. Compare accuracy curves and final accuracy.

**Why this matters:** If top-30 layers at 120 parameters matches or beats all-210 layers at 840 parameters, two things follow:
1. The 840-parameter claim can be tightened to 120 parameters — a stronger result
2. The ~180 low-leverage layers are candidates for PRNG replacement

**What to expect:** Based on the general literature on LoRA layer importance and the flat distribution of ternary weight information, I would expect a significant concentration. Attention layers in the middle-to-late decoder (layers 15–29) tend to be highest leverage for reasoning tasks; early embedding-adjacent layers tend to be low leverage. The MLP up/gate/down projection triad tends to matter more in later layers.

### 2b. Cross-adapter comparison

Run a matched comparison at the same parameter count (840 total):

| Configuration | Params | Description |
|---|---|---|
| GhostChain 4-parallel (current) | 840 | 4 fixed PRNG rank-1 experts per layer |
| TinyLoRA 1-scalar | 210 | 1 scalar per layer (original baseline) |
| TinyLoRA 4-scalar | 840 | 4 independent scalars, same PRNG directions as GhostChain |
| Rank-1 LoRA trainable | 840 | 2 × 420-element trainable u/v vectors for 1 layer |

This directly answers: is the benefit from GhostChain's 4x parameter count, or from the architecture? And how much does fixing the adapter directions (PRNG) cost relative to learning them (standard LoRA)?

The rank-1 trainable comparison is the important one. It has the same parameter count as GhostChain but lets the gradient choose the optimal direction rather than accepting a random PRNG direction. If trainable rank-1 LoRA significantly outperforms GhostChain at the same budget, the fixed-direction constraint is a meaningful bottleneck and the GhostWeight narrative needs to account for it.

---

## Phase 3: Partial GhostWeight (3–4 Weeks)

**Goal:** Test the GhostWeight compression idea in a form that can actually run on this hardware.

The original GhostWeight hypothesis (replace *all* 500MB with 8 bytes) requires generating 79,000 weight matrices per rollout — 200x too slow on this CPU. But the Phase 2 analysis gives you a ranked list of which layers matter. The partial GhostWeight experiment asks:

**What accuracy do you retain if you replace the bottom-K layers (by importance) with PRNG weights, while keeping the top layers real?**

### 3a. Graduated replacement

Using the layer importance ranking from Phase 1:

1. Replace the bottom 10% of layers (21 layers) with PRNG. Train from scratch. Measure accuracy.
2. Replace the bottom 25% (52 layers) with PRNG. Train. Measure.
3. Replace the bottom 50% (105 layers) with PRNG. Train. Measure.
4. Replace all 210 layers (pure GhostWeight, if compute allows). Train. Measure.

This produces an **accuracy vs. compression curve** — a clean scientific result regardless of where the curve breaks. Even if accuracy drops sharply at 25% PRNG, you've learned something important: which layers cannot tolerate PRNG replacement, and why.

**On the compute problem:** Replacing 50% of layers with PRNG means ~50% of matmuls regenerate weights on-the-fly. At ~4ms per PRNG matmul and ~105 PRNG layers × 7 projections × 4 experts, a single rollout costs roughly 105 × 7 × 4 × ~4ms ≈ 11.7s of pure PRNG computation, compared to the full 79K × 4ms ≈ 316s for all-PRNG. The 50%-PRNG case takes ~50% longer than real weights, not 200× longer. It's feasible.

### 3b. Interpret the curve

Three possible outcomes, all publishable:

**Outcome A — Gradual degradation:** Accuracy drops slowly as more layers become PRNG. The curve shows a knee at some compression level. This means partial GhostWeight is viable and the optimal compression ratio can be quantified. *This is the best outcome for the paper.*

**Outcome B — Sharp cliff:** Accuracy holds with 10% PRNG but collapses at 25%. This means there is a hard lower bound on real-weight information needed for reasoning. The cliff itself is a scientific finding — it reveals which structural properties of trained weights are irreplaceable. *This is still publishable and more interesting than a non-result.*

**Outcome C — PRNG layers are freely replaceable:** No accuracy drop even at 50% PRNG. This would be a surprising result suggesting that MLP weight structure encodes less task-relevant information than attention weights. It would strongly motivate pursuing the full GhostWeight hypothesis. *This is the most exciting outcome and the least likely.*

---

## Phase 4: Serial Coupling (1–2 Weeks, Parallel Track)

This is a small, contained experiment that addresses the most interesting architectural claim in the project — one that was reverted due to a Python/C mismatch, not because it was tested and found wanting.

Serial coupling (each expert conditions on the previous expert's output) should theoretically produce stronger gradient coupling and progressive error correction. The `math_validation_v2.py` toy result supports this. The reason it isn't in the current code is that the C ghost engine hardwires parallel addition, and GRPO requires the rollout model and the log-prob model to be identical.

With `USE_GHOST=False`, the C engine is not used for training rollouts — Python rollouts via `python_rollout()` are. This means you can implement serial coupling *purely in Python* without touching the C engine, and it will be consistent (Python model = Python rollouts).

The experiment:
1. Re-implement `GhostChain.forward()` with true serial coupling: expert2 receives `x + d1`, expert3 receives `x + d1 + d2`
2. Verify that the log-prob computation uses the same forward pass as the rollout
3. Run a matched comparison: 840 parallel vs 840 serial, same training setup, steps 0–200
4. Measure: convergence rate, final accuracy, gradient coupling coefficients (the cross-derivative ∂loss/∂s1 through s2, s3, sobs)

If serial coupling wins, it validates the original architecture claim and opens a path to deeper chains (8 experts, 12 experts) without increasing the parameter count significantly.

The key question the toy validation doesn't answer: does the serial coupling benefit persist in a high-dimensional, highly nonlinear context (a real transformer layer), or does it vanish because the random unit vectors are nearly orthogonal in R^2560 (making the coupling term O(scale²/dim) ≈ 8×10⁻⁶)?

---

## The Paper

If Phases 0–3 execute as described, the paper writes itself:

**Title:** *Sparse Sub-rank-1 Adaptation of Frozen Ternary LLMs: Layer Importance, Gradient Efficiency, and Partial Weight Replacement*

**Contributions:**
1. First characterization of per-layer adaptation importance in a natively ternary LLM under RL fine-tuning
2. Demonstration that sparse sub-rank-1 adapters (top-K layers only) match full-coverage performance with significantly fewer parameters
3. Partial GhostWeight: an accuracy/compression curve showing the tradeoff between PRNG weight replacement and reasoning performance
4. Infrastructure: 75x backward speedup (ternary STE backward kernel + FP32 LM head), 4.6x forward speedup (AVX2 LUT), enabling CPU-side RL training of 2.41B ternary models

**What makes it novel:**
- The layer importance analysis in the context of ternary quantization is new. Prior LoRA importance work (Zhang et al., AdaLoRA) focuses on rank allocation in float models; ternary models have different properties (flat singular spectra, 51% zero weights) that change the importance picture.
- The partial GhostWeight result — even a negative one showing sharp accuracy cliffs — is a contribution to the emerging literature on extreme model compression for inference at the edge.
- The infrastructure work (Section 4 of contributions) is independently useful and publishable.

**Realistic timeline from here:** 8–12 weeks on current hardware, assuming ~8 hours of active compute per day and roughly this sequence:
- Weeks 1–2: Phase 0 (baselines) + complete current run to step 300
- Weeks 3–4: Phase 1 (layer importance analysis)
- Weeks 5–7: Phase 2 (sparse adaptation experiments, 4 training runs)
- Weeks 8–10: Phase 3 (partial GhostWeight, 3–4 training runs)
- Weeks 11–12: Phase 4 (serial coupling) + writing

---

## What I Would Not Do

**I would not pursue the full GhostWeight (all-PRNG) training path on this hardware.** Not yet. It's 200x too slow to be the primary experimental axis. If the partial GhostWeight experiments (Phase 3) show a clean compression curve, then arguing for different hardware or a different batching strategy to test the all-PRNG extreme becomes a well-motivated next step rather than an act of faith.

**I would not pursue GMoE (multi-seed routing), Recursive Ghosting, or mobile deployment** at this stage. These are all downstream of the fundamental question — does PRNG weight replacement preserve learning capacity at all? — and that question isn't answered yet. They belong in the future work section of a paper, not on the active research agenda.

**I would not redesign the GRPO loop.** It works. The curriculum filtering is good. The KV-detach optimization is correct. The length normalization is correct. The answer extraction is thorough. There's nothing to improve here that would change the scientific conclusions.

**I would not keep chasing throughput.** ~8.4 steps/hr is the practical ceiling on this hardware within the current architecture. Any further speedup requires either architectural changes (fewer adapter parameters → fewer backward passes) or hardware changes. The science should drive those decisions, not a desire to go faster.

---

## The Honest Assessment of the Central Hypothesis

My belief, stated plainly: the strong form of GhostWeight — 8 bytes + 1.6KB → meaningful reasoning — is unlikely to work with the current adapter architecture, and here is the specific reason.

The base BitNet model achieves ~65–85% on GSM8K (the baseline from Phase 0 will tell us exactly). The GhostChain adapters in the current `USE_GHOST=False` setup are not *teaching* the model to reason — they are *nudging a model that already reasons well* toward the remaining hard problems. The base model does the heavy lifting; the adapters correct the marginal cases.

Under PRNG weights, there is no model that already reasons well. The PRNG weights produce activation distributions that are statistically similar to random projections — not zeroed out, but unstructured relative to the task. The 840 adapters would need to steer those random projections into coherent multi-step arithmetic reasoning from scratch. That is a categorically harder task than nudging a pretrained model at the margins.

The honest version of the hypothesis that I think has a chance: *partial* GhostWeight, where attention layers retain real weights (they carry learned positional and syntactic structure) and MLP layers are replaced with PRNG (they carry more distributed feature representations that may be more uniform). The attention mechanism encodes the relational reasoning structure; the MLP layers encode more local feature transformations. This split might be compressible in ways the full model is not.

If I'm wrong and the full GhostWeight converges — if 840 scalars applied to PRNG noise produce a model that solves 50%+ of GSM8K — it would be a genuinely surprising and important result. That's worth testing properly. The Phase 3 curve will tell you whether you're approaching that asymptote or whether there's a cliff in the way.

---

## Summary: Prioritized Action List

1. **This week:** Archive dead code (ghost engine, Vulkan, old logs). Establish base model baseline (20-sample eval with adapters zeroed). Wait for step 50 eval.

2. **Next 2 weeks:** Run current training to step 300. Reduce eval samples to 10 temporarily. Extract per-layer scale distribution at steps 50, 100, 200, 300.

3. **Weeks 3–4:** Layer importance analysis from step 300 checkpoint. Identify top-K layers by scale magnitude. Plot the adaptation Lorenz curve.

4. **Weeks 5–7:** Sparse adaptation experiment (top-30, top-60, top-105, all-210 layers). Run each to step 200. Compare accuracy curves.

5. **Weeks 8–10:** Partial GhostWeight experiment (10%, 25%, 50% PRNG replacement). Produce the accuracy/compression curve.

6. **Weeks 11–12:** Serial coupling comparison (parallel vs serial GhostChain at fixed parameter budget). Write the paper.

The work is coherent, the hardware is sufficient, and there is a real result at the end. What it needs most is not a new idea — it needs the current idea to be followed all the way to the answer.
