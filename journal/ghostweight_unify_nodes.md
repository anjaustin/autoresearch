# LMM Pass 11 NODES: Unification

- **Adapter Transfer Incompatibility:** Scales trained on real weights → PRNG = meaningless. Must co-train.
- **Batching Regression:** M=4 batch costs 4× attention; kernel savings don't compensate. Sequential M=1 wins.
- **M=4 Kernel Unrolling:** Implemented and validated. Useful when attention is not the bottleneck (future, shorter sequences).
- **Stop Condition:** `Q:` must require preceding newline to avoid firing mid-sentence.
- **Token Budget:** 128 tokens sufficient for GSM8K. 256 was wasted compute.
- **Resume Speed:** Eliminating re-eval on resume saves ~8 minutes per restart.
- **The Unified Path:** `USE_GHOST=True` + fresh run from step 0 = the only valid finish line.
