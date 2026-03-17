# LMM Pass 11 RAW: Unification — GhostWeight Must Start Fresh

## The Disconnect

After training TinyLoRA adapters for 220 steps against real BitNet weights (running reward ~91%), we switched `USE_GHOST=True` and immediately saw:
- All predictions: `null`
- All rewards: `0.1` (minimum — non-empty output only)
- Model generated incoherent fragments, hitting Q: stop in <3 tokens

## Root Cause Analysis

The adapter scales at step 220 are `~0.06` in magnitude. The TinyLoRA formula is:

```
output = base_weight(x) + scale * (x @ v.T) @ u.T
```

The `scale` values were learned to make small nudges to activations produced by **trained BitNet weights**. When we swap the base from real weights to PRNG, the activations from `base_weight(x)` change completely. The small `scale` corrections are now nudging in completely wrong directions relative to the new activation landscape.

This is analogous to training a hearing aid for a specific person's hearing loss, then putting it in someone with completely different loss. The correction is calibrated to the wrong baseline.

## Batching Experiments

Attempted M=4 batched rollouts to speed up training:
- Wrote `generate_group_completions()` — batches G sequences through the model simultaneously
- M=4 kernel unrolling in `ghost_matmul_lut.c` — generates weights once, dots against 4 rows
- **Result:** 697s/step (worse than sequential 480s)

**Why batching didn't help:** The bottleneck is attention, not weight matmul. Attention cost scales as M × seq_len², so running 4 sequences costs 4× more attention work. Weight matmul savings from the unrolled kernel are smaller than the attention overhead increase.

## Stop Condition Bug

`_should_stop` fired on mid-sentence `Q:` (e.g., "...the quantity..."). Fixed to require `\nQ:` — only stop when the model starts a new question on its own line.

## Token Budget

Reducing `MAX_NEW_TOKENS` from 256 → 128:
- Rollout time: 480s → 350s (25% improvement)
- No accuracy loss — GSM8K answers fit in 128 tokens
- Fewer wasted tokens after the answer

## Resume Fix

Removed the expensive 20-problem re-evaluation on every `--resume`. Now recovers `base_acc` from checkpoint eval history (instantaneous).

## Decision

Kill the intermediate `USE_GHOST=False` run at step 224. Launch a fresh `USE_GHOST=True` run from step 0. The adapters must learn against PRNG from the first gradient update. This is the only path to the unified 1KB model.
