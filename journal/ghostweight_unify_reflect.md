# LMM Pass 11 REFLECT: Unification

## What We Got Wrong

We assumed adapter scales trained against real BitNet weights could be "transplanted" to a GhostWeight base. This was wrong. The scales are not universal corrections — they are corrections calibrated to a specific activation landscape.

## What We Got Right

- The GhostWeight kernel is correct and fast (3.6ms/layer, byte-exact LUT decode).
- The GRPO training loop is correct (91% reward on real weights proves it).
- The weight_scale capture is correct (model produces coherent text under GhostWeight+scale).

## The Deeper Question

If adapters must be trained from scratch against PRNG, can they actually learn to steer noise into reasoning? Or is there a minimum information floor that requires at least some structured weights?

**Hypothesis:** The PRNG weights have zero mutual information with GSM8K answers. The only source of signal is the TinyLoRA bypass path. If the bypass path has enough capacity (210 scalars × rank-1 outer products), it can learn to route reasoning through the PRNG noise, exploiting the statistical properties of the PRNG as a fixed random projection.

This is the research bet. The finish line will tell us if we were right.

## Why This Matters

If successful, this demonstrates that LLM "intelligence" is not stored in the weights — it is stored in the low-dimensional structure that the weights project inputs through. A PRNG provides that projection for free. The learned adapters are the intelligence.
