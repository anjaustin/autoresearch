#include <immintrin.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>
#include <math.h>

// ---------------------------------------------------------------------------
// PRNG: SplitMix64 (for seeding) and Xorshift128+ (for AVX2 generation)
// ---------------------------------------------------------------------------

static inline uint64_t splitmix64(uint64_t* state) {
    uint64_t z = (*state += 0x9e3779b97f4a7c15);
    z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9;
    z = (z ^ (z >> 27)) * 0x94d049bb133111eb;
    return z ^ (z >> 31);
}

typedef struct {
    __m256i s0; // 4 lanes of state[0]
    __m256i s1; // 4 lanes of state[1]
} xorshift128p_avx2_state;

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

// ---------------------------------------------------------------------------
// Ghost Matmul Forward - handles batch dimension M
// ---------------------------------------------------------------------------

void ghost_matmul_forward(
    const float* input,    // [M, K]
    float* output,         // [M, N]
    int M, int K, int N,
    uint64_t base_seed,
    int layer_id
) {
    uint64_t layer_seed = base_seed ^ ((uint64_t)layer_id * 0x85ebca6b);
    __m256i bit_select = _mm256_set_epi32(128, 64, 32, 16, 8, 4, 2, 1);

    // Process each batch row
    for (int m = 0; m < M; m++) {
        const float* input_row = &input[m * K];
        float* output_row = &output[m * N];

        for (int n = 0; n < N; n++) {
            uint64_t row_seed = layer_seed ^ ((uint64_t)n * 0xc2b2ae35);
            uint64_t sm_state = row_seed;
            
            xorshift128p_avx2_state prng;
            uint64_t v0 = splitmix64(&sm_state);
            uint64_t v1 = splitmix64(&sm_state);
            uint64_t v2 = splitmix64(&sm_state);
            uint64_t v3 = splitmix64(&sm_state);
            prng.s0 = _mm256_set_epi64x(v3, v2, v1, v0);
            
            v0 = splitmix64(&sm_state);
            v1 = splitmix64(&sm_state);
            v2 = splitmix64(&sm_state);
            v3 = splitmix64(&sm_state);
            prng.s1 = _mm256_set_epi64x(v3, v2, v1, v0);

            __m256 sum = _mm256_setzero_ps();

            for (int k = 0; k < K; k += 128) {
                __m256i bits = xorshift128p_avx2_next(&prng);
                
                uint32_t ent[8] __attribute__((aligned(32)));
                _mm256_store_si256((__m256i*)ent, bits);
                
                for (int i = 0; i < 8; i++) {
                    uint32_t entropy = ent[i];
                    
                    for (int sub = 0; sub < 2; sub++) {
                        uint32_t word = (entropy >> (sub * 16)) & 0xFFFF;
                        
                        uint32_t nz_bits = 0;
                        uint32_t sign_bits = 0;
                        for (int j = 0; j < 8; j++) {
                            nz_bits |= ((word >> (j*2)) & 1) << j;
                            sign_bits |= ((word >> (j*2+1)) & 1) << j;
                        }
                        
                        __m256i nz_mask = _mm256_cmpgt_epi32(_mm256_and_si256(_mm256_set1_epi32(nz_bits), bit_select), _mm256_setzero_si256());
                        __m256i sign_mask = _mm256_slli_epi32(_mm256_cmpgt_epi32(_mm256_and_si256(_mm256_set1_epi32(sign_bits), bit_select), _mm256_setzero_si256()), 31);
                        
                        int k_idx = k + i*16 + sub*8;
                        if (k_idx < K) {
                            __m256 act = _mm256_loadu_ps(&input_row[k_idx]);
                            __m256 contribution = _mm256_and_ps(_mm256_xor_ps(act, _mm256_castsi256_ps(sign_mask)), _mm256_castsi256_ps(nz_mask));
                            sum = _mm256_add_ps(sum, contribution);
                        }
                    }
                }
            }
            
            float fsum[8];
            _mm256_storeu_ps(fsum, sum);
            output_row[n] = fsum[0] + fsum[1] + fsum[2] + fsum[3] + fsum[4] + fsum[5] + fsum[6] + fsum[7];
        }
    }
}

// ---------------------------------------------------------------------------
// Ghost Matmul Backward - handles batch dimension M
// ---------------------------------------------------------------------------

void ghost_matmul_backward(
    const float* grad_output,  // [M, N]
    float* grad_input,         // [M, K]
    int M, int K, int N,
    uint64_t base_seed,
    int layer_id
) {
    uint64_t layer_seed = base_seed ^ ((uint64_t)layer_id * 0x85ebca6b);
    __m256i bit_select = _mm256_set_epi32(128, 64, 32, 16, 8, 4, 2, 1);

    // Zero initialize grad_input
    memset(grad_input, 0, M * K * sizeof(float));

    // Process each batch row
    for (int m = 0; m < M; m++) {
        const float* grad_out_row = &grad_output[m * N];
        float* grad_in_row = &grad_input[m * K];

        for (int n = 0; n < N; n++) {
            float go = grad_out_row[n];
            if (go == 0.0f) continue;
            __m256 go_vec = _mm256_set1_ps(go);

            uint64_t row_seed = layer_seed ^ ((uint64_t)n * 0xc2b2ae35);
            uint64_t sm_state = row_seed;
            
            xorshift128p_avx2_state prng;
            uint64_t v0 = splitmix64(&sm_state);
            uint64_t v1 = splitmix64(&sm_state);
            uint64_t v2 = splitmix64(&sm_state);
            uint64_t v3 = splitmix64(&sm_state);
            prng.s0 = _mm256_set_epi64x(v3, v2, v1, v0);
            v0 = splitmix64(&sm_state);
            v1 = splitmix64(&sm_state);
            v2 = splitmix64(&sm_state);
            v3 = splitmix64(&sm_state);
            prng.s1 = _mm256_set_epi64x(v3, v2, v1, v0);

            for (int k = 0; k < K; k += 128) {
                __m256i bits = xorshift128p_avx2_next(&prng);
                uint32_t ent[8] __attribute__((aligned(32)));
                _mm256_store_si256((__m256i*)ent, bits);
                
                for (int i = 0; i < 8; i++) {
                    uint32_t entropy = ent[i];
                    for (int sub = 0; sub < 2; sub++) {
                        uint32_t word = (entropy >> (sub * 16)) & 0xFFFF;
                        uint32_t nz_bits = 0;
                        uint32_t sign_bits = 0;
                        for (int j = 0; j < 8; j++) {
                            nz_bits |= ((word >> (j*2)) & 1) << j;
                            sign_bits |= ((word >> (j*2+1)) & 1) << j;
                        }
                        
                        __m256i nz_mask = _mm256_cmpgt_epi32(_mm256_and_si256(_mm256_set1_epi32(nz_bits), bit_select), _mm256_setzero_si256());
                        __m256i sign_mask = _mm256_slli_epi32(_mm256_cmpgt_epi32(_mm256_and_si256(_mm256_set1_epi32(sign_bits), bit_select), _mm256_setzero_si256()), 31);

                        int k_idx = k + i*16 + sub*8;
                        if (k_idx < K) {
                            __m256 gi = _mm256_loadu_ps(&grad_in_row[k_idx]);
                            __m256 weight_contribution = _mm256_and_ps(_mm256_xor_ps(go_vec, _mm256_castsi256_ps(sign_mask)), _mm256_castsi256_ps(nz_mask));
                            gi = _mm256_add_ps(gi, weight_contribution);
                            _mm256_storeu_ps(&grad_in_row[k_idx], gi);
                        }
                    }
                }
            }
        }
    }
}
