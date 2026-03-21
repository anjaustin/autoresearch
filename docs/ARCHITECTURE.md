# GhostChain Architecture

## Overview

GhostChain is the adapter architecture used in Project GhostWeight. It replaces the single-scalar `TinyLoRA` with a 4-stage serial refinement chain per layer. The total learned state is **840 scalar parameters (~1.6KB in BF16)** for a 2.41B parameter base model.

---

## Background: TinyLoRA (Baseline)

The original adapter was a sub-rank-1 LoRA with a single trainable scalar per layer:

```
output = Base(x) + scale × (x @ v.T) @ u.T
```

Where `u` and `v` are fixed random unit vectors (no gradients), and `scale` is the only learned parameter. With 210 layers, this gives 210 parameters total.

**Limitation:** A single scalar can only shift the output along one fixed direction in the activation space. It cannot adapt to the content of `x` — the correction is always the same vector, just rescaled.

---

## GhostChain Design

Each of the 210 `AutoBitLinear` layers is wrapped with a `GhostChain` module containing 4 experts:

```
Expert 1: d1 = s1 · LoRA1(x)
Expert 2: d2 = s2 · LoRA2(x + d1)       ← sees Expert 1's output
Expert 3: d3 = s3 · LoRA3(x + d2)       ← sees Expert 2's output
Observer: do = sobs · LoRAobs(x + d1 + d2 + d3)  ← sees total nudge

Final:  y = Base(x) + d1 + d2 + d3 + do
```

Each `LoRAi` is a fixed rank-1 projection with random basis (no gradients). Only the 4 scalar scales `s1, s2, s3, sobs` are learned.

---

## Mathematical Properties

### Gradient Coupling

In standard LoRA, each adapter's gradient is independent. In GhostChain, because Expert 2 receives `x + d1` as input:

```
∂y/∂s1 = u1(v1·x) + s2·u2(v2·u1)(v1·x) + s3·u3(v3·u2)(v2·u1)(v1·x) + ...
```

Each downstream expert's gradient contains a term that depends on all upstream scale values. This forces cooperative optimization — Expert 1 must "cooperate" with Experts 2 and 3, not just independently minimize its own loss.

### Non-Linear Target Approximation

A single rank-1 adapter can only fit a rank-1 component of the residual. The GhostChain composed through 4 stages can approximate a rank-4 correction, even though each stage is rank-1, because the serial composition introduces multiplicative interaction terms.

**Empirical result (math_validation_v2.py):** On a non-linear target `y = sin(x)*0.1 + x`, a serial GhostChain converges to significantly lower loss than 4 parallel independent rank-1 adapters with the same parameter budget.

### Observer Role

The Observer is structurally different from the 3 Experts:
- Experts 1-3 each see only the *previous* expert's output.
- The Observer sees `x + total_nudge` — the full accumulated correction.

This makes the Observer a global error-correction gate that can compensate for cases where Experts 1-3 collectively over- or under-shoot.

---

## Parameter Accounting

| Component | Parameters per layer | Total (210 layers) |
|---|---|---|
| TinyLoRA (baseline) | 1 | 210 |
| GhostChain | 4 (s1, s2, s3, sobs) | **840** |

Storage at BF16 (2 bytes each): `840 × 2 = 1,680 bytes ≈ 1.6KB`

The fixed random vectors (`u`, `v`) per expert are not stored — they are deterministically regenerated from the layer name hash at load time.

---

## Deterministic Basis Generation

Each expert's random vectors are generated from a seed derived from the layer name:

```python
base_seed = SHA256(layer_name)[:8] as uint64  % (2**63)
expert1.seed = base_seed + 1
expert2.seed = base_seed + 2
expert3.seed = base_seed + 3
observer.seed = base_seed + 4
```

The `% (2**63)` clamp is mandatory. See [KNOWN_BUGS.md](KNOWN_BUGS.md#bug-001-torchgenerator-seed-overflow-silent-hang) for details on the silent hang that occurs without it.

This means the adapter state is fully described by 840 float16 values. The vectors `u` and `v` for all 840 experts are derived from those seeds and require zero storage.

---

## Comparison to Standard LoRA

| Property | Standard LoRA (rank r) | GhostChain |
|---|---|---|
| Params per layer | 2 × d × r | 4 scalars |
| Learned directions | r (full rank-r matrix) | 4 (fixed random) |
| Gradient coupling | None (independent) | Serial (cooperative) |
| Non-linear capacity | Linear | Approximately rank-4 |
| Storage (d=2560, r=1) | 10,240 values | 4 values |

GhostChain trades flexibility in the choice of learned directions (which LoRA has) for extreme parameter efficiency, relying on the serial coupling to extract more capacity per scalar.

---

## Why This Works (Hypothesis)

The base PRNG weights are random but not adversarial. In expectation, they contain every possible direction in activation space with equal probability. A well-chosen scalar correction along a fixed random direction will, in expectation, improve the output approximately as much as a learned direction correction — especially when multiple such corrections are composed.

The serial coupling amplifies this: Expert 2's correction is conditioned on the residual *after* Expert 1 has already acted, so it targets a lower-entropy error signal. The chain progressively refines toward the target output rather than making one large unconditioned jump.

---

## Implementation

- **Python (training):** `grpo_train.py` — `GhostExpert`, `GhostChain`, `inject_adapters`
- **C/AVX2 (inference):** `softchip/ghost_engine.c` — `apply_experts_batched`, `batched_transformer_block`
- **Bridge:** `softchip/ghost_engine_wrapper.py` — ctypes interface

See [GHOST_ENGINE.md](GHOST_ENGINE.md) for the C engine internals.
