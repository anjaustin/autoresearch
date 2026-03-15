# Raw Thoughts: Project GhostWeight Implementation

Intelligence as a Search Problem. We've optimized the training of 210 parameters to the point where they are "live" on a $500 CPU. Now we challenge the fundamental assumption of LLMs: that the weights themselves must be stored.

## The GhostWeight Thesis
A 2.4B parameter model contains ~4.8 billion bits of information (at 2 bits/param). This information is currently "static." In GhostWeight, we replace this static 500MB blob with a dynamic generator. The "Knowledge" then resides entirely in the **TinyLoRA Steering scalars** and the **PRNG Seed**.

## Technical Challenges

### 1. The PRNG Selection
We need a PRNG that is:
- **SIMD-Friendly:** Can generate 256 bits (or more) in parallel using AVX2.
- **Fast:** Must compete with L3/DDR4 memory bandwidth. If PRNG generation takes longer than a memory fetch, we lose the speed advantage.
- **Deterministic:** Given a (seed, layer_id, row, col_block), it must produce the exact same weights every time.

**Candidate: Xorshift128+ or Xoroshiro128+**
Xorshift is extremely simple (XOR, Shift, XOR). In AVX2, we can maintain 4 independent states in a YMM register and update them simultaneously.

### 2. The Distribution Mapping
BitNet b1.58 is ternary. Our PRNG output must be mapped to {-1, 0, 1}.
Ideal distribution: ~50% zero, ~25% +1, ~25% -1.

**Bit-Trick Mapping:**
1. Generate 2 bits per weight.
2. Bit A (Entropy): Determines if the weight is zero or non-zero.
3. Bit B (Sign): If non-zero, determines if +1 or -1.

Wait, to get exactly 50/25/25:
- Take 2 bits: `b1 b0`
- If `b0 == 0` -> Weight = 0 (50%)
- If `b0 == 1`:
    - If `b1 == 0` -> Weight = +1 (25%)
    - If `b1 == 1` -> Weight = -1 (25%)

This is perfect. One 256-bit PRNG draw gives us 128 ternary weights.

### 3. Seekability and the Backward Pass
The GRPO loop requires a backward pass.
`grad_input = W^T @ grad_output`.
Standard kernels access `W` by columns for backward. But our generator produces `W` by rows.
**Problem:** To generate a column of `W` procedurally, we would have to jump through the PRNG state, which is computationally expensive.

**Solution: The Regeneration Tradeoff.**
In the backward pass, we regenerate the row of `W`, then perform an **accumulate-scatter** update to `grad_input`.
This is exactly what our `ternary_matmul_backward` already does with stored weights! It iterates over rows of weights. GhostWeight just changes the "LOAD" to a "GENERATE".

### 4. Speed Projections
Memory fetch (DDR4): ~25 GB/s.
Xorshift AVX2: ~0.5 cycles per byte of entropy.
At 3.5GHz, that's ~7 GB/s of raw entropy per core.
One byte of entropy = 4 weights.
So 7 GB/s -> 28 billion weights/sec.
Compare to memory: 25 GB/s / 0.25 bytes (2 bits/weight) = 100 billion weights/sec.

**Insight:** Memory is actually faster than a single-core PRNG for raw throughput.
**Counter-Insight:** Memory bandwidth is shared across all 12 threads. 25 GB/s / 12 = ~2 GB/s per thread.
A vectorized PRNG in L1/Registers is private to the core and avoids the "Memory Wall."
GhostWeight might be **Faster** than memory-bound BitNet for multi-threaded workloads.

### 5. The Search Space
We have 2^64 possible "Seeds" (Universes).
In each Universe, we have 210 "Dials" (TinyLoRA scalars).
The GRPO loop now becomes a "Universe Hunter." We are looking for the Random Universe that is most compatible with human reasoning.

## Implementation Steps
1. Create `softchip/ghost_matmul.c`.
2. Implement `avx2_xorshift128plus_next()`.
3. Implement `ghost_forward` and `ghost_backward`.
4. Add `GhostLinear` module to `softchip/torch_ternary.py`.
5. Run a 1-Universe GRPO session to see if 210 params can steer pure noise.
