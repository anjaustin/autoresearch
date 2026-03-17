#include <immintrin.h>
#include <stdint.h>
#include <string.h>
#include <math.h>

// ---------------------------------------------------------------------------
// PRNG (identical to ghost_matmul.c)
// ---------------------------------------------------------------------------
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
             _mm256_xor_si256(_mm256_srli_epi64(s1, 17),
                              _mm256_srli_epi64(s0, 26))));
    return _mm256_add_epi64(st->s1, s0);
}
static inline void seed_xs(xs128p *st, uint64_t seed) {
    uint64_t sm = seed;
    uint64_t a=splitmix64(&sm), b=splitmix64(&sm),
             c=splitmix64(&sm), d=splitmix64(&sm);
    st->s0 = _mm256_set_epi64x(d,c,b,a);
    a=splitmix64(&sm); b=splitmix64(&sm);
    c=splitmix64(&sm); d=splitmix64(&sm);
    st->s1 = _mm256_set_epi64x(d,c,b,a);
}

// ---------------------------------------------------------------------------
// LUT: maps every 16-bit word -> 8 floats in {-1, 0, +1}
// Built once at first call. 65536 * 8 * 4 = 2 MB.
// ---------------------------------------------------------------------------
static float g_lut[65536][8] __attribute__((aligned(32)));
static int   g_lut_ready = 0;

static void build_lut(void) {
    for (int word = 0; word < 65536; word++) {
        for (int j = 0; j < 8; j++) {
            int nz = (word >> (j*2))   & 1;
            int sg = (word >> (j*2+1)) & 1;
            g_lut[word][j] = nz ? (sg ? -1.0f : 1.0f) : 0.0f;
        }
    }
    g_lut_ready = 1;
}

// ---------------------------------------------------------------------------
// Decode one weight row using LUT instead of bit-manipulation per word.
// Each uint32 from the PRNG gives two 16-bit words -> 16 weights via 2 LUT hits.
// ---------------------------------------------------------------------------
static void decode_row_lut(float *dst, int K, uint64_t row_seed) {
    if (!g_lut_ready) build_lut();

    xs128p prng; seed_xs(&prng, row_seed);
    int k = 0;
    for (; k + 128 <= K; k += 128) {
        __m256i bits = xs_next(&prng);
        uint32_t ent[8] __attribute__((aligned(32)));
        _mm256_store_si256((__m256i*)ent, bits);
        for (int i = 0; i < 8; i++) {
            uint32_t e = ent[i];
            uint16_t lo = e & 0xFFFF;
            uint16_t hi = (e >> 16) & 0xFFFF;
            // Each LUT row is 8 floats = 32 bytes = one AVX2 register
            __m256 wlo = _mm256_load_ps(g_lut[lo]);
            __m256 whi = _mm256_load_ps(g_lut[hi]);
            _mm256_storeu_ps(&dst[k + i*16],     wlo);
            _mm256_storeu_ps(&dst[k + i*16 + 8], whi);
        }
    }
    // scalar tail
    if (k < K) {
        xs128p pt; seed_xs(&pt, row_seed);
        for (int kk = 0; kk < k; kk += 128) xs_next(&pt);
        __m256i bits = xs_next(&pt);
        uint32_t ent[8] __attribute__((aligned(32)));
        _mm256_store_si256((__m256i*)ent, bits);
        for (int i = 0; i < 8 && k + i*16 < K; i++) {
            uint32_t e = ent[i];
            float tmp[16];
            memcpy(tmp,     g_lut[e & 0xFFFF],        8*sizeof(float));
            memcpy(tmp + 8, g_lut[(e >> 16) & 0xFFFF], 8*sizeof(float));
            for (int j = 0; j < 16 && k + i*16 + j < K; j++)
                dst[k + i*16 + j] = tmp[j];
        }
    }
}

// ---------------------------------------------------------------------------
// ghost_matmul_forward_lut
// ---------------------------------------------------------------------------
void ghost_matmul_forward(
    const float *__restrict__ input,
    float       *__restrict__ output,
    int M, int K, int N,
    uint64_t base_seed, int layer_id,
    float weight_scale
) {
    const uint64_t lseed = base_seed ^ ((uint64_t)layer_id * 0x85ebca6bULL);
    const int K8 = (K + 127) & ~127;
    float *wrow = (float*)__builtin_alloca((size_t)K8 * sizeof(float));

    for (int n = 0; n < N; n++) {
        decode_row_lut(wrow, K, lseed ^ ((uint64_t)n * 0xc2b2ae35ULL));
        
        // Apply scale
        if (weight_scale != 1.0f) {
            for (int k = 0; k < K; k++) wrow[k] *= weight_scale;
        }

        for (int m = 0; m < M; m++) {
            const float *x = &input[m*K];
            __m256 acc = _mm256_setzero_ps();
            int k;
            for (k = 0; k+8 <= K; k += 8)
                acc = _mm256_fmadd_ps(_mm256_loadu_ps(&x[k]),
                                      _mm256_loadu_ps(&wrow[k]), acc);
            float s = 0.f;
            for (; k < K; k++) s += x[k] * wrow[k];
            float buf[8]; _mm256_storeu_ps(buf, acc);
            output[m*N+n] = buf[0]+buf[1]+buf[2]+buf[3]+
                            buf[4]+buf[5]+buf[6]+buf[7]+s;
        }
    }
}

void ghost_matmul_backward(
    const float *__restrict__ grad_output,
    float       *__restrict__ grad_input,
    int M, int K, int N,
    uint64_t base_seed, int layer_id,
    float weight_scale
) {
    const uint64_t lseed = base_seed ^ ((uint64_t)layer_id * 0x85ebca6bULL);
    const int K8 = (K + 127) & ~127;
    float *wrow = (float*)__builtin_alloca((size_t)K8 * sizeof(float));

    memset(grad_input, 0, (size_t)M * K * sizeof(float));

    for (int n = 0; n < N; n++) {
        decode_row_lut(wrow, K, lseed ^ ((uint64_t)n * 0xc2b2ae35ULL));
        
        // Apply scale
        if (weight_scale != 1.0f) {
            for (int k = 0; k < K; k++) wrow[k] *= weight_scale;
        }

        for (int m = 0; m < M; m++) {
            float go = grad_output[m*N+n];
            if (go == 0.f) continue;
            __m256 gov = _mm256_set1_ps(go);
            float *gi = &grad_input[m*K];
            int k;
            for (k = 0; k+8 <= K; k += 8) {
                __m256 g = _mm256_loadu_ps(&gi[k]);
                g = _mm256_fmadd_ps(gov, _mm256_loadu_ps(&wrow[k]), g);
                _mm256_storeu_ps(&gi[k], g);
            }
            for (; k < K; k++) gi[k] += go * wrow[k];
        }
    }
}
