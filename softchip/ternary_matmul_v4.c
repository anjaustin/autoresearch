/*
 * ternary_matmul_v4.c — Base-3 MTFP ternary matmul
 *
 * Packs 5 ternary {-1,0,+1} weights per byte (base-3 / MTFP1 encoding).
 * Decodes via 256×8 float LUT (~8KB) that lives permanently in L1 cache.
 * Achieves 99.1% of the Shannon trit-packing limit (5.047 trits/byte).
 * 25% denser than v3's 2-bit packing: 25% less weight data read from RAM.
 *
 * Encoding:
 *   byte b = t0 + t1×3 + t2×9 + t3×27 + t4×81
 *   trit code: 0 → weight -1,  1 → weight 0,  2 → weight +1
 *
 * Inner loop: SSE FMA for 4 trits per byte + scalar for 5th.
 * Handles any in_features (partial last byte uses code 1 = weight 0).
 * OMP strategy: serial for M<6 (autoregressive decode), batch-parallel M≥6.
 *
 * Compile:
 *   gcc -O3 -mavx2 -mfma -msse4.1 -march=native -fopenmp -shared -fPIC \
 *       -o ternary_matmul_v4.so ternary_matmul_v4.c -lm
 */

#include <immintrin.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <omp.h>

/* -----------------------------------------------------------------------
 * 256-entry decode LUT: lut5f[b][j] = j-th ternary weight from byte b
 * 8 floats per entry (5 real + 3 zero padding for aligned SSE loads).
 * Size: 256 × 8 × 4 = 8192 bytes — fits entirely in L1 data cache.
 * ----------------------------------------------------------------------- */
static float lut5f[256][8] __attribute__((aligned(32)));
static int lut5f_initialized = 0;

static void init_lut5f(void) {
    if (lut5f_initialized) return;
    for (int b = 0; b < 256; b++) {
        int val = b;
        for (int j = 0; j < 5; j++) {
            int code = val % 3;               /* 0, 1, or 2               */
            lut5f[b][j] = (float)(code - 1);  /* 0→-1.0, 1→0.0, 2→+1.0  */
            val /= 3;
        }
        lut5f[b][5] = lut5f[b][6] = lut5f[b][7] = 0.0f;
    }
    lut5f_initialized = 1;
}

/* -----------------------------------------------------------------------
 * Weight packing: base-3, 5 trits per byte
 *
 *   byte = t0 + t1×3 + t2×9 + t3×27 + t4×81,  trit = round(w×scale) + 1
 *
 * Partial last byte: missing positions use code 1 (weight 0), so they
 * contribute zero to any dot product regardless of activation values.
 *
 * Returns heap-allocated packed array (caller must free()).
 * Sets *out_weight_scale to mean(|W|) for de-quantization.
 * ----------------------------------------------------------------------- */
uint8_t *pack_weights_b3(const float *weights, int out_features, int in_features,
                          float *out_weight_scale) {
    double sum_abs = 0.0;
    long total = (long)out_features * in_features;
    for (long i = 0; i < total; i++)
        sum_abs += fabsf(weights[i]);
    float mean_abs = (float)(sum_abs / total);
    float scale    = 1.0f / fmaxf(mean_abs, 1e-5f);
    *out_weight_scale = mean_abs;

    int k_bytes = (in_features + 4) / 5;
    uint8_t *packed = (uint8_t *)calloc((size_t)out_features * k_bytes, 1);

    for (int row = 0; row < out_features; row++) {
        const float *w = weights + (size_t)row * in_features;
        uint8_t     *p = packed  + (size_t)row * k_bytes;
        int k = 0;
        for (int kb = 0; kb < k_bytes; kb++) {
            uint8_t byte  = 0;
            int     power = 1;
            for (int j = 0; j < 5; j++) {
                int code;
                if (k < in_features) {
                    float wq = roundf(w[k++] * scale);
                    if (wq >  1.0f) wq =  1.0f;
                    if (wq < -1.0f) wq = -1.0f;
                    code = (int)wq + 1;  /* -1→0, 0→1, +1→2 */
                } else {
                    code = 1;            /* padding: weight 0 */
                }
                byte += (uint8_t)(code * power);
                power *= 3;
            }
            p[kb] = byte;
        }
    }
    return packed;
}

/* -----------------------------------------------------------------------
 * Activation quantization (symmetric INT8, matching BitNet ActQuant)
 * Same spec as v3 — quantizes activations to {-127..127} float grid.
 * ----------------------------------------------------------------------- */
static void quantize_activations(const float *act_row, float *act_q, int in_features) {
    __m256 vmax = _mm256_setzero_ps();
    int k = 0;
    for (; k + 7 < in_features; k += 8) {
        __m256 v  = _mm256_loadu_ps(act_row + k);
        __m256 va = _mm256_andnot_ps(_mm256_set1_ps(-0.0f), v);
        vmax = _mm256_max_ps(vmax, va);
    }
    __m128 hi = _mm256_extractf128_ps(vmax, 1);
    __m128 lo = _mm256_castps256_ps128(vmax);
    __m128 m4 = _mm_max_ps(lo, hi);
    m4 = _mm_max_ps(m4, _mm_movehl_ps(m4, m4));
    m4 = _mm_max_ps(m4, _mm_movehdup_ps(m4));
    float amax = _mm_cvtss_f32(m4);
    for (; k < in_features; k++) { float av = fabsf(act_row[k]); if (av > amax) amax = av; }

    float act_scale = 127.0f / fmaxf(amax, 1e-5f);
    float act_inv   = 1.0f / act_scale;
    __m256 vs  = _mm256_set1_ps(act_scale);
    __m256 vis = _mm256_set1_ps(act_inv);
    __m256 vlo = _mm256_set1_ps(-128.0f), vhi = _mm256_set1_ps(127.0f);
    k = 0;
    for (; k + 7 < in_features; k += 8) {
        __m256 v = _mm256_loadu_ps(act_row + k);
        v = _mm256_round_ps(_mm256_mul_ps(v, vs),
                            _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
        v = _mm256_mul_ps(_mm256_min_ps(_mm256_max_ps(v, vlo), vhi), vis);
        _mm256_storeu_ps(act_q + k, v);
    }
    for (; k < in_features; k++) {
        float v = roundf(act_row[k] * act_scale);
        if (v < -128.0f) v = -128.0f;
        if (v >  127.0f) v =  127.0f;
        act_q[k] = v * act_inv;
    }
}

/* -----------------------------------------------------------------------
 * Inner dot product: one output neuron vs one quantized activation row
 *
 * AVX2 8-wide FMA per byte: loads 8 floats from LUT (aligned, slots 5-7 = 0)
 * and 8 floats from act_q (unaligned). The activation buffer is allocated with
 * k_bytes*5+8 zeroed floats, so reads beyond in_features safely return 0.
 * Partial-byte trit positions are packed as code 1 (weight 0), so lut5f
 * returns 0.0f there — both sources zero, no contribution. No scalar tail.
 * ----------------------------------------------------------------------- */
static inline float b3_dot_row(const uint8_t *packed_w, const float *act_q,
                                int k_bytes) {
    __m256 acc = _mm256_setzero_ps();

    for (int kb = 0; kb < k_bytes; kb++) {
        __m256 wf = _mm256_load_ps(lut5f[packed_w[kb]]);  /* aligned 8-float LUT */
        __m256 af = _mm256_loadu_ps(act_q + kb * 5);      /* unaligned act, zeroed tail */
        acc = _mm256_fmadd_ps(wf, af, acc);
    }

    __m128 hi   = _mm256_extractf128_ps(acc, 1);
    __m128 lo   = _mm256_castps256_ps128(acc);
    __m128 sum4 = _mm_add_ps(lo, hi);
    __m128 shuf = _mm_movehdup_ps(sum4);
    __m128 sum2 = _mm_add_ps(sum4, shuf);
    __m128 shuf2= _mm_movehl_ps(sum2, sum2);
    __m128 sum1 = _mm_add_ss(sum2, shuf2);
    return _mm_cvtss_f32(sum1);
}

/* -----------------------------------------------------------------------
 * Scatter-add for backward: grad_input += W_row × scalar
 * SSE for first 4 trits per byte, scalar for 5th.
 * ----------------------------------------------------------------------- */
static inline void b3_scatter_row(const uint8_t *packed_w, float *grad_in,
                                   float scalar, int in_features, int k_bytes) {
    __m128 vs   = _mm_set1_ps(scalar);
    int    full = in_features / 5;

    for (int kb = 0; kb < full; kb++) {
        const float *wf     = lut5f[packed_w[kb]];
        __m128 contrib      = _mm_mul_ps(_mm_load_ps(wf), vs);
        __m128 cur          = _mm_loadu_ps(grad_in + kb * 5);
        _mm_storeu_ps(grad_in + kb * 5, _mm_add_ps(cur, contrib));
        grad_in[kb * 5 + 4] += wf[4] * scalar;
    }

    if (full < k_bytes) {
        const float *wf = lut5f[packed_w[full]];
        int rem = in_features - full * 5;
        for (int j = 0; j < rem; j++)
            grad_in[full * 5 + j] += wf[j] * scalar;
    }
}

/* -----------------------------------------------------------------------
 * Public API: ternary_matmul_b3 — forward pass
 *
 * Threading strategy (same rationale as v3):
 *   M < 6: serial — fork/join overhead exceeds parallelism benefit
 *   M >= 6: OMP batch-parallel — enough rows for load balancing
 * ----------------------------------------------------------------------- */
void ternary_matmul_b3(const uint8_t *packed_w, const float *activation,
                        float *output, int batch, int out_features,
                        int in_features, float weight_scale) {
    init_lut5f();

    int k_bytes     = (in_features + 4) / 5;
    int num_threads = omp_get_max_threads();
    /* Buffer must cover k_bytes*5+8 floats: AVX2 reads 8-wide starting at kb*5,
     * so the last byte (kb=k_bytes-1) reads up to (k_bytes-1)*5+7 = k_bytes*5+2.
     * Round up to 32-byte (8-float) alignment. Zeroed so tail reads return 0. */
    int buf_floats  = ((k_bytes * 5 + 8 + 7) & ~7);

    static float **act_bufs    = NULL;
    static int    act_buf_sz   = 0;
    static int    act_buf_thds = 0;

    if (!act_bufs || act_buf_thds < num_threads || act_buf_sz < buf_floats) {
        if (act_bufs) {
            for (int t = 0; t < act_buf_thds; t++) free(act_bufs[t]);
            free(act_bufs);
        }
        act_bufs = (float **)malloc(num_threads * sizeof(float *));
        for (int t = 0; t < num_threads; t++) {
            act_bufs[t] = (float *)aligned_alloc(32, buf_floats * sizeof(float));
            memset(act_bufs[t], 0, buf_floats * sizeof(float));  /* zero tail */
        }
        act_buf_sz   = buf_floats;
        act_buf_thds = num_threads;
    }

    if (batch >= 6) {
        #pragma omp parallel for schedule(dynamic)
        for (int m = 0; m < batch; m++) {
            int    tid   = omp_get_thread_num();
            float *act_q = act_bufs[tid];
            quantize_activations(activation + (size_t)m * in_features, act_q, in_features);
            for (int n = 0; n < out_features; n++) {
                output[(size_t)m * out_features + n] =
                    b3_dot_row(packed_w + (size_t)n * k_bytes, act_q,
                               k_bytes) * weight_scale;
            }
        }
    } else {
        for (int m = 0; m < batch; m++) {
            float *act_q = act_bufs[0];
            quantize_activations(activation + (size_t)m * in_features, act_q, in_features);
            for (int n = 0; n < out_features; n++) {
                output[(size_t)m * out_features + n] =
                    b3_dot_row(packed_w + (size_t)n * k_bytes, act_q,
                               k_bytes) * weight_scale;
            }
        }
    }
}

/* -----------------------------------------------------------------------
 * Public API: ternary_matmul_b3_backward — STE backward pass
 *
 * grad_input[m,k] = weight_scale × Σ_n(W_ternary[n,k] × grad_output[m,n])
 * ----------------------------------------------------------------------- */
void ternary_matmul_b3_backward(const uint8_t *packed_w, const float *grad_output,
                                 float *grad_input, int batch, int out_features,
                                 int in_features, float weight_scale) {
    init_lut5f();

    int k_bytes = (in_features + 4) / 5;

    if (batch >= 6) {
        #pragma omp parallel for schedule(dynamic)
        for (int m = 0; m < batch; m++) {
            const float *go = grad_output + (size_t)m * out_features;
            float       *gi = grad_input  + (size_t)m * in_features;
            memset(gi, 0, in_features * sizeof(float));
            for (int n = 0; n < out_features; n++) {
                float g = go[n] * weight_scale;
                if (fabsf(g) < 1e-30f) continue;
                b3_scatter_row(packed_w + (size_t)n * k_bytes, gi, g,
                               in_features, k_bytes);
            }
        }
    } else {
        for (int m = 0; m < batch; m++) {
            const float *go = grad_output + (size_t)m * out_features;
            float       *gi = grad_input  + (size_t)m * in_features;
            memset(gi, 0, in_features * sizeof(float));
            for (int n = 0; n < out_features; n++) {
                float g = go[n] * weight_scale;
                if (fabsf(g) < 1e-30f) continue;
                b3_scatter_row(packed_w + (size_t)n * k_bytes, gi, g,
                               in_features, k_bytes);
            }
        }
    }
}

/* -----------------------------------------------------------------------
 * Standalone benchmark
 * Compile: gcc -O3 -mavx2 -mfma -msse4.1 -march=native -fopenmp \
 *              -DSTANDALONE_BENCH -o ternary_bench_v4 ternary_matmul_v4.c -lm
 * ----------------------------------------------------------------------- */
#ifdef STANDALONE_BENCH
#include <stdio.h>
#include <time.h>

static double wall_ms(void) {
    struct timespec t;
    clock_gettime(CLOCK_MONOTONIC, &t);
    return t.tv_sec * 1e3 + t.tv_nsec * 1e-6;
}

int main(void) {
    const int K = 2560, N = 2560;
    printf("ternary_matmul_v4 (base-3 MTFP + SSE LUT): K=%d N=%d\n", K, N);
    printf("Threads: %d\n\n", omp_get_max_threads());

    float *weights = (float *)aligned_alloc(32, (size_t)N * K * sizeof(float));
    float *act1    = (float *)aligned_alloc(32, K * sizeof(float));
    float *out1    = (float *)aligned_alloc(32, N * sizeof(float));

    srand(42);
    for (int i = 0; i < N * K; i++)
        weights[i] = ((float)rand() / RAND_MAX < 0.33f) ? 0.0f
                   : ((float)rand() / RAND_MAX < 0.5f)  ? -1.0f : 1.0f;
    for (int i = 0; i < K; i++)
        act1[i] = ((float)rand() / RAND_MAX - 0.5f) * 2.0f;

    float wscale;
    uint8_t *packed_b3 = pack_weights_b3(weights, N, K, &wscale);
    int k_bytes = (K + 4) / 5;
    printf("v4 packed: %d bytes/row × %d rows = %.1f KB total\n",
           k_bytes, N, (double)(N * k_bytes) / 1024.0);
    printf("v3 packed: %d bytes/row × %d rows = %.1f KB total\n",
           (K + 3) / 4, N, (double)(N * ((K + 3) / 4)) / 1024.0);
    printf("Ratio:     %.3f× (%.1f%% of v3 size)\n\n",
           (double)(N * k_bytes) / (N * ((K + 3) / 4)),
           100.0 * (N * k_bytes) / (N * ((K + 3) / 4)));

    /* Warmup */
    ternary_matmul_b3(packed_b3, act1, out1, 1, N, K, wscale);

    /* Benchmark M=1 */
    int iters = 300;
    double t0 = wall_ms();
    for (int i = 0; i < iters; i++)
        ternary_matmul_b3(packed_b3, act1, out1, 1, N, K, wscale);
    double ms1 = (wall_ms() - t0) / iters;
    printf("M=1 (autoregressive): %.3f ms/call\n", ms1);
    printf("  → %.2f GFLOP/s equivalent\n\n",
           2.0 * N * K / (ms1 / 1000.0) / 1e9);

    free(packed_b3); free(weights); free(act1); free(out1);
    return 0;
}
#endif
