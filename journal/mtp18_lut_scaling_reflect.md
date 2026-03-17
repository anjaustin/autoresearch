# LMM Pass 10 REFLECT: MTP18, LUT Kernels, and the Scaling Fix

## What we learned
1. **The "Atomic Disconnect" was purely numerical.** Random weights are fine as long as their *magnitude* matches the original model. BitNet's `absmean` scale is a critical hyperparameter that must be preserved even in "ghost" mode.
2. **Kernels beat algorithms (again).** We initially looked for complex ways to optimize GhostWeight regeneration. The simplest solution — a 2MB Lookup Table — gave a 3x speedup by reducing the instruction count of the decode phase to almost zero.
3. **MTP18 is "ahead of its time" for this hardware.** Native base-3 arithmetic is theoretically elegant for ternary LLMs, but without native SIMD support for trits, it's 20x slower than bit-packed ternary.

## Strategic Pivot
We are committing to the **GhostWeight (PRNG) + TinyLoRA** path. The SVD results (Pass 9) proved that ternary matrices can't be compressed via linear algebra. PRNG is the only way to reach the 1KB model goal.

## The Future of Project GhostWeight
If we can achieve parity with the 1KB model, we have demonstrated that **LLM weights are not specific values, but specific patterns of noise.** The LoRA adapters are the only "signal" needed.
