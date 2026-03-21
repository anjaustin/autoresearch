# Known Bugs

## BUG-001: torch.Generator Seed Overflow (Silent Hang)

**Status:** Fixed — applied in grpo_train.py at lines 173 and 210
**Severity:** Critical — causes silent hang, no error, no crash, no log output  
**Introduced:** GhostChain refactor (replacing TinyLoRA with 840-expert chain)

### Symptom

`grpo_train.py --preflight` hangs silently after model load. Output stops at:

```
Loading weights: 100%|██████████| 332/332
```

No further output. Process is alive and consuming ~1 core of CPU. No Python traceback. No segfault.

### Root Cause

`torch.Generator().manual_seed(seed)` silently hangs (does not raise) when `seed >= 2**63`.

In `GhostExpert.__init__`, the seed is derived from a SHA-256 hash:

```python
seed = int.from_bytes(hashlib.sha256(name.encode()).digest()[:8], "big")
```

This produces an unsigned 64-bit integer in the range `[0, 2**64)`. PyTorch's `manual_seed` only accepts values in `[0, 2**63)`. Values at or above `2**63` cause an internal hang rather than raising `ValueError`.

The hang occurs during `inject_adapters()` on the very first call to `GhostExpert.__init__`, which is called 840 times (one per expert across all layers). The model appears to load successfully — and it does — but the process never progresses past that point.

### Fix

Clamp the seed before passing it to `manual_seed`:

```python
# In GhostExpert.__init__:
gen = torch.Generator().manual_seed(seed % (2**63))

# In inject_adapters():
seed = int.from_bytes(hashlib.sha256(name.encode()).digest()[:8], "big") % (2**63)
```

**Files to fix:**
- `grpo_train.py:173` — `GhostExpert.__init__`
- `grpo_train.py:210` — `inject_adapters`

### Why It's Hard to Diagnose

1. The hang happens inside a `for` loop over 210 layers × 4 experts = 840 calls. The first hanging call is on the very first layer, so no adapters are injected before the hang.
2. `torch.Generator().manual_seed()` does not raise — it stalls internally in C++ with no Python-visible exception.
3. The model load completes and prints normally, so the log looks like a post-load crash rather than a hang during adapter injection.
4. `ps aux` shows the process alive with ~100% CPU, which looks like normal computation rather than a deadlock.

### Verification

This fix was confirmed working in an isolated test:

```python
import torch, hashlib
seed = int.from_bytes(hashlib.sha256(b"model.layers.0.self_attn.q_proj").digest()[:8], "big")
# seed = 14973271227228454452 — OVER 2**63, will hang:
gen = torch.Generator().manual_seed(seed)           # hangs
# Fix:
gen = torch.Generator().manual_seed(seed % (2**63)) # works fine
```

---

## BUG-002: ctypes Struct Field Order Mismatch (Segfault)

**Status:** Fixed in ghost_engine_wrapper.py  
**Severity:** Critical — immediate segfault on first C engine call  
**Introduced:** Initial ghost_engine_wrapper.py implementation

### Symptom

`python grpo_train.py --preflight` segfaults with:

```
Segmentation fault (core dumped)
```

Immediately after `GhostEngine` is initialized and the first `rollout_batch()` call is made.

### Root Cause

The Python `ctypes.Structure._fields_` order for `GhostModel` did not match the C struct field layout in `ghost_engine.c`. Specifically, `base_seed` and `weight_scales` were swapped.

**C struct (correct):**
```c
typedef struct {
    float *k_cache, *v_cache, *embeddings, *lm_head, *weight_scales;
    uint64_t base_seed;
    ExpertData *experts;
} GhostModel;
```

**Python wrapper (was wrong):**
```python
_fields_ = [
    ...
    ("base_seed", ctypes.c_uint64),    # was here
    ("weight_scales", ...),            # was here
    ...
]
```

When `ctypes` passed the struct by reference, C read `base_seed` from the memory offset of `weight_scales` (a pointer), causing an immediate invalid memory access.

### Fix

Reorder `_fields_` in `GhostModel` to exactly match the C declaration:

```python
_fields_ = [
    ("k_cache",       ctypes.POINTER(ctypes.c_float)),
    ("v_cache",       ctypes.POINTER(ctypes.c_float)),
    ("embeddings",    ctypes.POINTER(ctypes.c_float)),
    ("lm_head",       ctypes.POINTER(ctypes.c_float)),
    ("weight_scales", ctypes.POINTER(ctypes.c_float)),  # must come before base_seed
    ("base_seed",     ctypes.c_uint64),
    ("experts",       ctypes.POINTER(ExpertData)),
]
```

### Rule

**The `ctypes.Structure._fields_` list must be in identical order to the C struct declaration.** Any mismatch causes silent data corruption or segfault. There is no runtime check. When in doubt, print `ctypes.sizeof(GhostModel)` and compare to `sizeof(GhostModel)` in C.

---

## BUG-003: KV-Cache Size Undersized for Batched Rollout

**Status:** Fixed in ghost_engine_wrapper.py  
**Severity:** Medium — silent data corruption (no crash), incorrect outputs  
**Introduced:** Initial KV-cache allocation

### Symptom

Generated tokens are garbage or repetitive even after successful compilation and struct fix.

### Root Cause

Original KV-cache was sized for a single sequence:

```python
# Wrong — only fits G=1:
self.kv_size = 30 * 4096 * 5 * 128
```

The batched engine expects `G` independent KV-caches, one per completion in the group. With `GROUP_SIZE=4`, the cache needs to be 4× larger.

### Fix

```python
# Correct — fits G=4:
group_size = 4
self.kv_size = 30 * group_size * 4096 * 5 * 128
```

This allocates ~1.25 GB per rollout call. The KV-cache is reused across steps (no reallocation).

---

## BUG-004: BF16 Expert Tensors Cast as Float32 in C Engine

**Status:** Fixed in `softchip/ghost_engine_wrapper.py`
**Severity:** Critical — C engine reads garbage bit patterns for all expert u, v, scale values
**Introduced:** Initial ghost_engine_wrapper.py implementation

### Root Cause

`GhostExpert` stores `u`, `v`, and `scale` as BF16 tensors (2 bytes per element). The C
engine's `ExpertData` struct has `float *u, *v, *scale` (4 bytes per element). The wrapper
passed raw `.data_ptr()` pointers without conversion:

```python
# Wrong — passes BF16 data pointer to a float32 field:
self.experts_array[i].u = ctypes.cast(expert.u.data_ptr(), ctypes.POINTER(ctypes.c_float))
```

The C code reads each 4-byte float by combining two adjacent BF16 values (each 2 bytes).
Every u, v, and scale element was silently corrupted. All expert contributions to inference
were noise unrelated to the trained parameters.

### Fix

Convert to float32 contiguous tensors before obtaining pointers. Store the converted tensors
in `self._f32_buf` on the engine instance to keep them alive for the duration of the C call:

```python
u_f32 = expert.u.float().contiguous()
v_f32 = expert.v.float().contiguous()
s_f32 = expert.scale.float().contiguous()
self._f32_buf.extend([u_f32, v_f32, s_f32])
self.experts_array[i].u = ctypes.cast(u_f32.data_ptr(), ctypes.POINTER(ctypes.c_float))
```

**Why the lifetime matters:** If the tensors are created as temporaries and not stored, Python
may garbage-collect them before `ghost_engine_generate_batched` returns, leaving dangling
pointers in the C struct. `self._f32_buf` is overwritten at the start of each `rollout_batch`
call, releasing the previous batch's buffers after the C call completes.

---

## BUG-005: Python GhostChain Serial Coupling Mismatches C Engine Parallel Addition

**Status:** Fixed in `grpo_train.py`
**Severity:** Critical — GRPO policy gradient computed for wrong model; rollout tokens generated
by a different forward pass than the one used for log-prob computation
**Introduced:** GhostChain architecture design

### Root Cause

`GhostChain.forward()` in Python used serial coupling: expert2 received `x + d1` as input
(where `d1` is expert1's output), expert3 received `x + d2`, etc. The C engine's
`apply_experts_batched()` always reads from the original normalized input (`ins`) for all 4
experts — a flat parallel sum.

GRPO requires: `tokens = sample(π_θ)` and `loss = -A · log π_θ(tokens)` where both uses of
`π_θ` are the *same model*. With the mismatch, rollouts came from the parallel C model but
gradients were computed on the serial Python model. The gradient pointed in the wrong direction.

### Practical magnitude

At current scale magnitudes (~0.14) in R^2560, random unit vectors are nearly orthogonal
(`E[v2·u1] ≈ 0`), so the coupling term `s2 · (v2·d1) · u2` is O(scale²/dim) ≈ 8×10⁻⁶ per
layer — tiny but growing with scale. More importantly, the serial coupling advantage over
parallel requires nonlinearity between experts; without it, both are functionally equivalent
rank-4 adapters.

### Fix

Change `GhostChain.forward()` to parallel (all experts read from same `x`), matching the C
engine exactly. Serial coupling remains a future direction contingent on adding activation
functions between expert stages and updating the C engine accordingly.

---

## BUG-006: Non-Deterministic Shuffle Corrupts Curriculum Continuity on Resume

**Status:** Fixed in `grpo_train.py`
**Severity:** Medium — `problem_ptr` from checkpoint points into a different random permutation
after resume, losing all curriculum history
**Introduced:** Addition of `USE_CURRICULUM_FILTER` + checkpoint resume

### Root Cause

`random.shuffle(train_data)` used the global `random` state, which depends on what the
model-loading code consumed before line 482. On a fresh start the shuffle order was one
permutation; on resume (with different global state) it was a different permutation.
`problem_ptr = 208` from a checkpoint then indexed into the new permutation, not the original,
effectively starting curriculum from a random position in an unseen ordering.

### Fix

Use an isolated `random.Random(SEED)` instance so the shuffle is deterministic and identical
on every cold start and resume:

```python
random.Random(SEED).shuffle(train_data)
```

---

## BUG-007: Last-Number Fallback in Curriculum Pre-Check Misclassifies Hard Problems

**Status:** Fixed in `grpo_train.py`
**Severity:** Medium — hard problems excluded from curriculum; easy problems included
**Introduced:** Addition of `find_hard_problem()` using `extract_answer()`

### Root Cause

`extract_answer()` falls back to the last number in any generated text when no explicit
`"the answer is"` or `"####"` format is found. A model that generates
*"There are 3 bags with 4 apples each. Each bag weighs 12 grams."* gets pred=12 even if the
correct answer is 12 by coincidence, not reasoning. Such problems would be (incorrectly)
marked "already solved" and excluded from training.

The reverse also applies: a model that solves a problem but writes the answer without an
explicit marker (e.g. just "48") gets pred=None from strict extraction, incorrectly classifying
the problem as unsolved and including it in curriculum when it shouldn't be.

### Fix

Added `strict_extract_answer()` — identical to `extract_answer()` minus the last-number
fallback. Used exclusively in `find_hard_problem()`. Training rewards and eval continue using
the permissive `extract_answer()`.

---

## BUG-008: GRPO Loss Not Length-Normalized — Gradient Biased Toward Short Completions

**Status:** Fixed in `grpo_train.py`
**Severity:** Medium — optimizer implicitly rewards short completions, pushing toward format
learning over chain-of-thought reasoning
**Introduced:** Initial GRPO implementation

### Root Cause

```python
reinforce_loss = -adv * log_probs_theta.sum()  # sums over all tokens
```

A 128-token completion produces 128× the gradient magnitude of a 1-token completion with the
same advantage. The model is implicitly incentivized to emit short answers (e.g., just "48")
rather than the chain-of-thought format encouraged by the few-shot examples.

### Fix

```python
reinforce_loss = -adv * log_probs_theta.mean()  # mean over tokens
```

---

## BUG-009: Curriculum Exhaustion Fallback Trains on Already-Solved Problems

**Status:** Fixed in `grpo_train.py`
**Severity:** Medium — once the model learns enough problems, scan regularly hits
CURRICULUM_SKIP_MAX=20 and falls back to training on an easy problem, reintroducing the
original 81%-skip pathology
**Introduced:** Initial `find_hard_problem()` implementation

### Root Cause

When `find_hard_problem()` exhausted `CURRICULUM_SKIP_MAX` candidates without finding a
failure, it returned `(last_ex, idx, 20)` — the last candidate the model had just solved.
The training loop proceeded with this problem, almost certainly got all-reward=1, triggered
`std_r < 1e-8`, and skipped the GRPO update after burning ~140s on G=4 rollouts.

### Fix

`find_hard_problem()` now returns `found_hard=False` when exhausted, and the training loop
skips the step number entirely with a `CURRICULUM_EXHAUSTED` log line. The step counter
advances so training continues rather than hanging.
