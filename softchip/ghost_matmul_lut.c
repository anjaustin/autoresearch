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

        if (M == 4) {
            const float *x0 = &input[0*K];
            const float *x1 = &input[1*K];
            const float *x2 = &input[2*K];
            const float *x3 = &input[3*K];
            __m256 acc0 = _mm256_setzero_ps();
            __m256 acc1 = _mm256_setzero_ps();
            __m256 acc2 = _mm256_setzero_ps();
            __m256 acc3 = _mm256_setzero_ps();
            int k;
            for (k = 0; k+8 <= K; k += 8) {
                __m256 w = _mm256_loadu_ps(&wrow[k]);
                acc0 = _mm256_fmadd_ps(_mm256_loadu_ps(&x0[k]), w, acc0);
                acc1 = _mm256_fmadd_ps(_mm256_loadu_ps(&x1[k]), w, acc1);
                acc2 = _mm256_fmadd_ps(_mm256_loadu_ps(&x2[k]), w, acc2);
                acc3 = _mm256_fmadd_ps(_mm256_loadu_ps(&x3[k]), w, acc3);
            }
            float s0=0.f, s1=0.f, s2=0.f, s3=0.f;
            for (; k < K; k++) {
                float w = wrow[k];
                s0 += x0[k]*w; s1 += x1[k]*w; s2 += x2[k]*w; s3 += x3[k]*w;
            }
            float buf0[8], buf1[8], buf2[8], buf3[8];
            _mm256_storeu_ps(buf0, acc0); _mm256_storeu_ps(buf1, acc1);
            _mm256_storeu_ps(buf2, acc2); _mm256_storeu_ps(buf3, acc3);
            output[0*N+n] = buf0[0]+buf0[1]+buf0[2]+buf0[3]+buf0[4]+buf0[5]+buf0[6]+buf0[7]+s0;
            output[1*N+n] = buf1[0]+buf1[1]+buf1[2]+buf1[3]+buf1[4]+buf1[5]+buf1[6]+buf1[7]+s1;
            output[2*N+n] = buf2[0]+buf2[1]+buf2[2]+buf2[3]+buf2[4]+buf2[5]+buf2[6]+buf2[7]+s2;
            output[3*N+n] = buf3[0]+buf3[1]+buf3[2]+buf3[3]+buf3[4]+buf3[5]+buf3[6]+buf3[7]+s3;
        } else {
            for (int m = 0; m < M; m++) {
                const float *x = &input[m*K];
                float *out_ptr = &output[m*N+n];
                __m256 acc = _mm256_setzero_ps();
                int k;
                for (k = 0; k+8 <= K; k += 8) {
                    __m256 w = _mm256_loadu_ps(&wrow[k]);
                    acc = _mm256_fmadd_ps(_mm256_loadu_ps(&x[k]), w, acc);
                }
                float s = 0.f;
                for (; k < K; k++) s += x[k] * wrow[k];
                float buf[8]; _mm256_storeu_ps(buf, acc);
                *out_ptr = buf[0]+buf[1]+buf[2]+buf[3]+
                           buf[4]+buf[5]+buf[6]+buf[7]+s;
            }
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
