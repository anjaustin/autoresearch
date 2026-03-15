# Synthesis: GhostWeight Kernel Specification

## Implementation Strategy

### 1. The Core Data Structures
```c
typedef struct {
    uint64_t s[2];
} xorshift128p_state;

typedef struct {
    __m256i s0; // 4 lanes of state[0]
    __m256i s1; // 4 lanes of state[1]
} xorshift128p_avx2_state;
```

### 2. Seeding (SplitMix64)
Used to initialize the Xorshift state from a 64-bit row seed.
```c
static inline uint64_t splitmix64(uint64_t* state) {
    uint64_t z = (*state += 0x9e3779b97f4a7c15);
    z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9;
    z = (z ^ (z >> 27)) * 0x94d049bb133111eb;
    return z ^ (z >> 31);
}
```

### 3. Vectorized PRNG (AVX2 Xorshift128+)
Generates 256 bits (4 x 64-bit) per call.
```c
static inline __m256i xorshift128p_avx2_next(xorshift128p_avx2_state* state) {
    __m256i s1 = state->s0;
    __m256i s0 = state->s1;
    state->s0 = s0;
    s1 = _mm256_xor_si256(s1, _mm256_slli_epi64(s1, 23));
    state->s1 = _mm256_xor_si256(s1, _mm256_xor_si256(s0, 
                _mm256_xor_si256(_mm256_srli_epi64(s1, 17), 
                                 _mm256_srli_epi64(s0, 26))));
    return _mm256_add_epi64(state->s1, s0);
}
```

### 4. Bit-to-Ternary Mapping
One 256-bit PRNG result provides 128 weights (2 bits each).
- Bit 2k: Non-zero flag
- Bit 2k+1: Sign flag

**AVX2 Logic:**
1. Extract 128 bits for Non-zero flags.
2. Extract 128 bits for Sign flags.
3. Use `_mm256_permutevar8x32_epi32` and `_mm256_shuffle_epi8` to expand these into 8-weight masks for each AVX2 iteration.

Actually, simpler:
Generate 128 weights. Process in blocks of 8.
For 8 weights, we need 16 bits of entropy.
One `__m256i` from PRNG gives 256 bits = 16 blocks of 8 weights.

### 5. Kernel Interface
```c
void ghost_matmul_forward(
    const float* input,    // [M, K]
    float* output,         // [M, N]
    int M, int K, int N,
    uint64_t base_seed,
    int layer_id
);

void ghost_matmul_backward(
    const float* grad_output, // [M, N]
    float* grad_input,        // [M, K]
    int M, int K, int N,
    uint64_t base_seed,
    int layer_id
);
```

### 6. Integration Flag
In `softchip/torch_ternary.py`, `patch_model()` will accept `use_ghost=True`.
If True:
1. `AutoBitLinear` weights are NOT packed.
2. `TernaryMatmulFunction` calls `ghost_matmul_forward/backward`.
3. The original weight tensor can be deleted to save RAM.

---

## One-Sentence Summary
GhostWeight replaces the BitNet weight matrix with an on-the-fly SIMD generator, reducing the storage cost of 2.4 billion parameters to a single 64-bit seed while maintaining BitNet's ternary distribution.
