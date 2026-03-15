# Reflect: Project GhostWeight Implementation Strategy

## Evaluation of PRNG Robustness

A weak row-seeding mixing function could lead to visible patterns in the weights (e.g., every row looking slightly like the previous one). This would destroy the high-entropy assumption of BitNet.

**Refinement:**
Instead of a simple XOR, use a single round of a lightweight block cipher or a permutation.
`row_seed = splitmix64(base_seed ^ layer_id ^ row_index)`
`splitmix64` is very fast (few shifts and XORs) and produces high-quality initial states for Xorshift.

## The Memory Bandwidth vs. Compute Paradox

If we use JIT generation (generating weights as needed in the registers), we completely eliminate memory stalls for the weight matrix.
- `AutoBitLinear` is 512MB.
- DDR4-3200 is 25GB/s.
- Theoretical max speed for full model forward (CPU limited by RAM): ~20ms.
- Our current v3 speed: ~900ms (M=1).

**Reflection:**
Our current speed is not limited by RAM bandwidth; it's limited by CPU instruction throughput and PyTorch overhead.
**GhostWeight will actually be SLOWER than v3 initially**, because it adds ALU instructions (PRNG) to an already CPU-bound loop.

**Strategic Pivot:**
GhostWeight's primary value is **Storage Compression**, not raw speed. However, we should aim for at most a 2x slowdown compared to v3. If we can fit the 2B model in < 1MB, a 2-second forward pass is an acceptable price.

## Correctness of the Backward Pass

In `ternary_matmul_backward`, we currently do:
```c
for (int n = 0; n < N; n++) {
    const uint8_t* row = packed_weights + n * K_packed;
    for (int k = 0; k < K; k += 8) {
        // ... accumulate grad_input ...
    }
}
```
GhostWeight becomes:
```c
for (int n = 0; n < N; n++) {
    uint64_t state = row_seed(base_seed, layer_id, n);
    for (int k = 0; k < K; k += 128) { // Xorshift generates 128 weights per 256 bits
        __m256i bits = avx2_xorshift_next(&state);
        // ... process 128 weights ...
    }
}
```
This is mathematically identical to our current backward, as long as `row_seed` and `avx2_xorshift_next` are consistent between forward and backward.

## Final Plan

1. **Kernel:** Implement `softchip/ghost_matmul.c`.
2. **PRNG:** SIMD Xorshift128+ with `splitmix64` row seeding.
3. **Distribution:** 50% Zero, 25% +1, 25% -1 mapping.
4. **Integration:** Add `use_ghost=True` flag to `patch_model()`.
5. **Validation:** Check if a "Ghost" model can generate coherent text (initially it will be gibberish) and if GRPO gradients are non-zero.
