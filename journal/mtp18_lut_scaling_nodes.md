# LMM Pass 10 NODES: MTP18, LUT Kernels, and the Scaling Fix

- **The Scale Gap:** GhostWeight needs `weight_scale` (~2.33) to match BitNet's activation distribution.
- **LUT Optimization:** 2MB Lookup Table + AVX2 = 3.1x faster GhostWeight forward pass.
- **MTP18:** Native base-3 floating point format. Hardware mismatch (x86) makes it 20x slower than ternary.
- **GRPO Robustness:** Resume logic must handle unbound `base_acc` and missing `eval_history`.
- **Memory Coherency:** LUT fits in L3 cache, ensuring fast random weight regeneration.
- **Verification:** 1KB model (seed+adapters) achieves 91% reward on GSM8K training set.
