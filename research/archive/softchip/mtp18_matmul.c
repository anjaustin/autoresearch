/*
 * MTP18: Multi-Trit Floating Point 18
 * Native base-3 encoding for hardware coherency
 */

#include <immintrin.h>
#include <stdint.h>
#include <math.h>

static inline uint64_t splitmix64(uint64_t *s) {
    uint64_t z = (*s += 0x9e3779b97f4a7c15ULL);
    z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9ULL;
    z = (z ^ (z >> 27)) * 0x94d049bb133111ebULL;
    return z ^ (z >> 31);
}

typedef struct { __m256i s0, s1; } xs128p;

static inline __m256i xs_next(xs128p *st) {
    __m256i s1 = st->s0, s0 = st->s1;
    st->s0 = s0;
    s1 = _mm256_xor_si256(s1, _mm256_slli_epi64(s1, 23));
    st->s1 = _mm256_xor_si256(s1, _mm256_xor_si256(s0, 
        _mm256_xor_si256(_mm256_srli_epi64(s1, 17), _mm256_srli_epi64(s0, 26))));
    return _mm256_add_epi64(st->s1, s0);
}

static inline void seed_xs(xs128p *st, uint64_t seed) {
    uint64_t sm = seed;
    uint64_t a=splitmix64(&sm), b=splitmix64(&sm), c=splitmix64(&sm), d=splitmix64(&sm);
    st->s0 = _mm256_set_epi64x(d,c,b,a);
    a=splitmix64(&sm); b=splitmix64(&sm); c=splitmix64(&sm); d=splitmix64(&sm);
    st->s1 = _mm256_set_epi64x(d,c,b,a);
}

static inline float mtp18_decode(uint64_t e) {
    int s = (e >> 0) & 3;
    int sign = (s == 1) ? 1 : ((s == 2) ? -1 : 0);
    if (sign == 0) return 0.0f;
    int exp = (e >> 2) & 3;
    int mant = ((e >> 4) & 1023) % 243;
    float base = 1.0f;
    for (int i = 0; i < exp; i++) base *= 3.0f;
    return sign * base * (1.0f + (float)mant / 81.0f);
}

static inline float dot4(const float *a, const float *b) {
    return a[0]*b[0] + a[1]*b[1] + a[2]*b[2] + a[3]*b[3];
}

void mtp18_matmul_forward(
    const float *input,
    float *output,
    int M, int K, int N,
    uint64_t base_seed,
    int layer_id
) {
    uint64_t layer_seed = base_seed ^ ((uint64_t)layer_id * 0x85ebca6bULL);
    
    for (int n = 0; n < N; n++) {
        uint64_t row_seed = layer_seed ^ ((uint64_t)n * 0xc2b2ae35ULL);
        xs128p prng;
        seed_xs(&prng, row_seed);
        
        for (int m = 0; m < M; m++) {
            const float *x = &input[m * K];
            float sum = 0.0f;
            
            int k = 0;
            for (; k + 4 <= K; k += 4) {
                __m256i bits = xs_next(&prng);
                uint64_t buf[4];
                _mm256_storeu_si256((__m256i*)buf, bits);
                float w0 = mtp18_decode(buf[0]);
                float w1 = mtp18_decode(buf[1]);
                float w2 = mtp18_decode(buf[2]);
                float w3 = mtp18_decode(buf[3]);
                sum += x[k+0]*w0 + x[k+1]*w1 + x[k+2]*w2 + x[k+3]*w3;
            }
            
            for (; k < K; k++) {
                __m256i b = xs_next(&prng);
                uint64_t e = _mm256_extract_epi64(b, 0);
                sum += x[k] * mtp18_decode(e);
            }
            
            output[m * N + n] = sum;
        }
    }
}

void mtp18_matmul_backward(
    const float *grad_output,
    float *grad_input,
    int M, int K, int N,
    uint64_t base_seed,
    int layer_id
) {
    for (int i = 0; i < M * K; i++) grad_input[i] = 0.0f;
    
    uint64_t layer_seed = base_seed ^ ((uint64_t)layer_id * 0x85ebca6bULL);
    
    for (int n = 0; n < N; n++) {
        uint64_t row_seed = layer_seed ^ ((uint64_t)n * 0xc2b2ae35ULL);
        xs128p prng;
        seed_xs(&prng, row_seed);
        
        for (int m = 0; m < M; m++) {
            float go = grad_output[m * N + n];
            if (go == 0.0f) continue;
            
            float *gi = &grad_input[m * K];
            int k = 0;
            for (; k + 4 <= K; k += 4) {
                __m256i bits = xs_next(&prng);
                uint64_t buf[4];
                _mm256_storeu_si256((__m256i*)buf, bits);
                gi[k+0] += go * mtp18_decode(buf[0]);
                gi[k+1] += go * mtp18_decode(buf[1]);
                gi[k+2] += go * mtp18_decode(buf[2]);
                gi[k+3] += go * mtp18_decode(buf[3]);
            }
            
            for (; k < K; k++) {
                __m256i b = xs_next(&prng);
                uint64_t e = _mm256_extract_epi64(b, 0);
                gi[k] += go * mtp18_decode(e);
            }
        }
    }
}
