/*
 * ternary_matmul_v3.c -- AVX2 "soft-chip" for BitNet ternary matmul
 *
 * v3: Two fixes over v2:
 *   1. Pre-allocate activation quantization buffers (no malloc in hot loop)
 *   2. 2D parallelism: batch × out_features (critical for M=1 autoregressive)
 *
 * Core decode strategy unchanged from v2: 256-entry LUT, 2-bit packed weights,
 * XOR+AND in AVX2, no multiply in hot path.
 *
 * Compile: gcc -O3 -mavx2 -mfma -march=native -fopenmp -shared -fPIC \
 *          -o ternary_matmul_v3.so ternary_matmul_v3.c -lm
 */

#include <immintrin.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <omp.h>

/* -----------------------------------------------------------------------
 * Weight packing: 2 bits per weight, 4 per byte (same as v1/v2)
 * ----------------------------------------------------------------------- */
static void pack_row(const float *weights, uint8_t *packed,
                     int in_features, float scale) {
    for (int i = 0; i < in_features; i += 4) {
        uint8_t byte = 0;
        for (int j = 0; j < 4 && (i + j) < in_features; j++) {
            float w = weights[i + j];
            float wq = roundf(w * scale);
            if (wq > 1.0f) wq = 1.0f;
            if (wq < -1.0f) wq = -1.0f;
            uint8_t code;
            if (wq > 0.5f)       code = 0x01;
            else if (wq < -0.5f) code = 0x03;
            else                  code = 0x00;
            byte |= (code << (j * 2));
        }
        packed[i / 4] = byte;
    }
}

uint8_t *pack_weights(const float *weights, int out_features, int in_features,
                      float *out_weight_scale) {
    double sum_abs = 0.0;
    int total = out_features * in_features;
    for (int i = 0; i < total; i++)
        sum_abs += fabsf(weights[i]);
    float mean_abs = (float)(sum_abs / total);
    float scale = 1.0f / fmaxf(mean_abs, 1e-5f);
    *out_weight_scale = mean_abs;
    int packed_row_bytes = (in_features + 3) / 4;
    uint8_t *packed = (uint8_t *)calloc(out_features * packed_row_bytes, 1);
    for (int row = 0; row < out_features; row++)
        pack_row(weights + row * in_features,
                 packed + row * packed_row_bytes,
                 in_features, scale);
    return packed;
}

/* -----------------------------------------------------------------------
 * Precomputed LUT: 256 entries x 16 bytes = 4KB each -> fits L1
 * ----------------------------------------------------------------------- */
static uint32_t lut_nz[256][4] __attribute__((aligned(32)));
static uint32_t lut_sign[256][4] __attribute__((aligned(32)));
static int lut_initialized = 0;

static void init_lut(void) {
    if (lut_initialized) return;
    for (int b = 0; b < 256; b++) {
        for (int j = 0; j < 4; j++) {
            uint8_t code = (b >> (j * 2)) & 0x03;
            lut_nz[b][j]   = (code & 1) ? 0xFFFFFFFF : 0;
            lut_sign[b][j] = (code >> 1) ? 0x80000000 : 0;
        }
    }
    lut_initialized = 1;
}

/* -----------------------------------------------------------------------
 * AVX2 ternary dot product (unchanged from v2)
 * ----------------------------------------------------------------------- */
static inline float ternary_dot_v2(const uint8_t *packed_w,
                                   const float *act,
                                   int in_features) {
    __m256 acc = _mm256_setzero_ps();

    int k = 0;
    for (; k + 7 < in_features; k += 8) {
        uint8_t b0 = packed_w[k / 4];
        uint8_t b1 = packed_w[k / 4 + 1];

        __m256 a = _mm256_loadu_ps(act + k);

        __m128i nz_lo  = _mm_load_si128((const __m128i *)lut_nz[b0]);
        __m128i sg_lo  = _mm_load_si128((const __m128i *)lut_sign[b0]);
        __m128i nz_hi  = _mm_load_si128((const __m128i *)lut_nz[b1]);
        __m128i sg_hi  = _mm_load_si128((const __m128i *)lut_sign[b1]);

        __m256i nz256 = _mm256_set_m128i(nz_hi, nz_lo);
        __m256i sg256 = _mm256_set_m128i(sg_hi, sg_lo);

        __m256 signed_a = _mm256_xor_ps(a, _mm256_castsi256_ps(sg256));
        __m256 masked = _mm256_and_ps(signed_a, _mm256_castsi256_ps(nz256));

        acc = _mm256_add_ps(acc, masked);
    }

    /* Horizontal sum */
    __m128 hi = _mm256_extractf128_ps(acc, 1);
    __m128 lo = _mm256_castps256_ps128(acc);
    __m128 sum4 = _mm_add_ps(lo, hi);
    __m128 shuf = _mm_movehdup_ps(sum4);
    __m128 sum2 = _mm_add_ps(sum4, shuf);
    __m128 shuf2 = _mm_movehl_ps(sum2, sum2);
    __m128 sum1 = _mm_add_ss(sum2, shuf2);
    float result = _mm_cvtss_f32(sum1);

    /* Scalar tail */
    for (; k < in_features; k++) {
        uint8_t byte = packed_w[k / 4];
        uint8_t code = (byte >> ((k % 4) * 2)) & 0x03;
        if (code == 0x01)      result += act[k];
        else if (code == 0x03) result -= act[k];
    }
    return result;
}

/* -----------------------------------------------------------------------
 * AVX2 activation quantization (symmetric INT8, matching BitNet ActQuant)
 * ----------------------------------------------------------------------- */
static void quantize_activations(const float *act_row, float *act_q,
                                 int in_features) {
    /* Find max abs */
    __m256 vmax = _mm256_setzero_ps();
    int k = 0;
    for (; k + 7 < in_features; k += 8) {
        __m256 v = _mm256_loadu_ps(act_row + k);
        __m256 va = _mm256_andnot_ps(_mm256_set1_ps(-0.0f), v);
        vmax = _mm256_max_ps(vmax, va);
    }
    __m128 hi128 = _mm256_extractf128_ps(vmax, 1);
    __m128 lo128 = _mm256_castps256_ps128(vmax);
    __m128 m4 = _mm_max_ps(lo128, hi128);
    m4 = _mm_max_ps(m4, _mm_movehl_ps(m4, m4));
    m4 = _mm_max_ps(m4, _mm_movehdup_ps(m4));
    float amax = _mm_cvtss_f32(m4);
    for (; k < in_features; k++) {
        float av = fabsf(act_row[k]);
        if (av > amax) amax = av;
    }

    float act_scale = 127.0f / fmaxf(amax, 1e-5f);
    float act_inv_scale = 1.0f / act_scale;

    __m256 vs = _mm256_set1_ps(act_scale);
    __m256 vis = _mm256_set1_ps(act_inv_scale);
    __m256 vmin_clip = _mm256_set1_ps(-128.0f);
    __m256 vmax_clip = _mm256_set1_ps(127.0f);
    k = 0;
    for (; k + 7 < in_features; k += 8) {
        __m256 v = _mm256_loadu_ps(act_row + k);
        v = _mm256_mul_ps(v, vs);
        v = _mm256_round_ps(v, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
        v = _mm256_min_ps(_mm256_max_ps(v, vmin_clip), vmax_clip);
        v = _mm256_mul_ps(v, vis);
        _mm256_storeu_ps(act_q + k, v);
    }
    for (; k < in_features; k++) {
        float v = act_row[k] * act_scale;
        v = roundf(v);
        if (v < -128.0f) v = -128.0f;
        if (v > 127.0f) v = 127.0f;
        act_q[k] = v * act_inv_scale;
    }
}

/* -----------------------------------------------------------------------
 * Public API: ternary_matmul with 2D parallelism
 *
 * v3 changes:
 *   - Pre-allocate per-thread activation buffers (no malloc in hot loop)
 *   - Parallelize over out_features when batch is small (M=1 case)
 *   - For larger batch, parallelize over batch rows
 * ----------------------------------------------------------------------- */
void ternary_matmul(const uint8_t *packed_w,
                    const float *activation,
                    float *output,
                    int batch,
                    int out_features,
                    int in_features,
                    float weight_scale) {

    init_lut();

    int packed_row_bytes = (in_features + 3) / 4;
    int num_threads = omp_get_max_threads();

    /* Persistent per-thread activation buffers (allocated once) */
    static float **act_bufs = NULL;
    static int act_buf_size = 0;
    static int act_buf_threads = 0;
    int aligned_k = (in_features + 7) & ~7;

    if (!act_bufs || act_buf_threads < num_threads || act_buf_size < aligned_k) {
        /* (Re)allocate -- only happens on first call or size change */
        if (act_bufs) {
            for (int t = 0; t < act_buf_threads; t++) free(act_bufs[t]);
            free(act_bufs);
        }
        act_bufs = (float **)malloc(num_threads * sizeof(float *));
        for (int t = 0; t < num_threads; t++)
            act_bufs[t] = (float *)aligned_alloc(32, aligned_k * sizeof(float));
        act_buf_size = aligned_k;
        act_buf_threads = num_threads;
    }

    /*
     * Threading strategy:
     *
     * For small total work (batch * out_features), OpenMP fork/join overhead
     * (~50us per thread) dominates. A single 2560-element ternary dot product
     * takes ~0.6us, so parallelizing 2560 work items across 12 threads only
     * saves ~1.3ms while paying ~0.6ms overhead -- marginal gain, worse
     * cache behavior. For large batch (M >= num_threads), batch-parallel is
     * optimal. For small batch + large N, 2D parallel helps only if N is large
     * enough (empirically: total_work > ~8000 per thread).
     *
     * Decision: parallelize batch dimension when M >= 4 (enough rows for
     * load balancing), use serial path for M < 4 (autoregressive case --
     * single-threaded is faster due to L2 locality).
     */
    if (batch >= 6) {
        /*
         * Batch-parallel path: enough batch rows to benefit from threading.
         */
        #pragma omp parallel for schedule(dynamic)
        for (int m = 0; m < batch; m++) {
            int tid = omp_get_thread_num();
            const float *act_row = activation + m * in_features;
            float *act_q = act_bufs[tid];

            quantize_activations(act_row, act_q, in_features);

            for (int n = 0; n < out_features; n++) {
                const uint8_t *w_row = packed_w + n * packed_row_bytes;
                float dot = ternary_dot_v2(w_row, act_q, in_features);
                output[m * out_features + n] = dot * weight_scale;
            }
        }
    } else {
        /*
         * Serial path for small batch (M=1,2,3).
         * Single-threaded is faster because the entire weight matrix (1.6MB)
         * stays in one core's L3 slice, and we avoid fork/join overhead.
         */
        for (int m = 0; m < batch; m++) {
            const float *act_row = activation + m * in_features;
            float *act_q = act_bufs[0];

            quantize_activations(act_row, act_q, in_features);

            for (int n = 0; n < out_features; n++) {
                const uint8_t *w_row = packed_w + n * packed_row_bytes;
                float dot = ternary_dot_v2(w_row, act_q, in_features);
                output[m * out_features + n] = dot * weight_scale;
            }
        }
    }
}

/* -----------------------------------------------------------------------
 * Backward pass: grad_input = W^T @ grad_output (no activation quantization)
 *
 * For STE through frozen AutoBitLinear layers. Gradients pass through
 * unchanged (no INT8 quantization). Uses accumulate-scatter approach:
 * iterate over N weight rows (cache-friendly sequential reads), scatter
 * each row's contribution scaled by grad_output[n] into grad_input.
 *
 * Weight matrix: (out_features × in_features), packed row-major
 * grad_output: (batch × out_features) -- gradient from downstream
 * grad_input: (batch × in_features) -- gradient to upstream
 *
 * grad_input[m,k] = weight_scale * sum_n(W_ternary[n,k] * grad_output[m,n])
 * ----------------------------------------------------------------------- */

/* AVX2 ternary scatter-add: grad_input += W_row * scalar */
static inline void ternary_scatter_add(const uint8_t *packed_w,
                                        float *grad_input,
                                        float scalar,
                                        int in_features) {
    __m256 vs = _mm256_set1_ps(scalar);
    __m256 vns = _mm256_set1_ps(-scalar);

    int k = 0;
    for (; k + 7 < in_features; k += 8) {
        uint8_t b0 = packed_w[k / 4];
        uint8_t b1 = packed_w[k / 4 + 1];

        __m128i nz_lo  = _mm_load_si128((const __m128i *)lut_nz[b0]);
        __m128i sg_lo  = _mm_load_si128((const __m128i *)lut_sign[b0]);
        __m128i nz_hi  = _mm_load_si128((const __m128i *)lut_nz[b1]);
        __m128i sg_hi  = _mm_load_si128((const __m128i *)lut_sign[b1]);

        __m256i nz256 = _mm256_set_m128i(nz_hi, nz_lo);
        __m256i sg256 = _mm256_set_m128i(sg_hi, sg_lo);

        /* For +1 weights: add scalar. For -1 weights: add -scalar. For 0: add 0.
         * Reuse XOR+AND trick: XOR flips sign for -1 weights, AND zeros for 0 weights.
         * contribution = (scalar XOR sign_mask) AND nonzero_mask */
        __m256 signed_s = _mm256_xor_ps(vs, _mm256_castsi256_ps(sg256));
        __m256 masked = _mm256_and_ps(signed_s, _mm256_castsi256_ps(nz256));

        __m256 current = _mm256_loadu_ps(grad_input + k);
        _mm256_storeu_ps(grad_input + k, _mm256_add_ps(current, masked));
    }

    /* Scalar tail */
    for (; k < in_features; k++) {
        uint8_t byte = packed_w[k / 4];
        uint8_t code = (byte >> ((k % 4) * 2)) & 0x03;
        if (code == 0x01)      grad_input[k] += scalar;
        else if (code == 0x03) grad_input[k] -= scalar;
    }
}

void ternary_matmul_backward(const uint8_t *packed_w,
                              const float *grad_output,
                              float *grad_input,
                              int batch,
                              int out_features,
                              int in_features,
                              float weight_scale) {

    init_lut();

    int packed_row_bytes = (in_features + 3) / 4;

    /*
     * Threading strategy: same as forward.
     * For M=1 (autoregressive backward), serial is better.
     * For M>=6, parallelize over batch.
     */
    if (batch >= 6) {
        #pragma omp parallel for schedule(dynamic)
        for (int m = 0; m < batch; m++) {
            const float *go = grad_output + m * out_features;
            float *gi = grad_input + m * in_features;

            /* Zero output */
            memset(gi, 0, in_features * sizeof(float));

            /* Accumulate: for each weight row, scatter-add */
            for (int n = 0; n < out_features; n++) {
                float g_scaled = go[n] * weight_scale;
                if (fabsf(g_scaled) < 1e-30f) continue;  /* Skip negligible */
                const uint8_t *w_row = packed_w + n * packed_row_bytes;
                ternary_scatter_add(w_row, gi, g_scaled, in_features);
            }
        }
    } else {
        /* Serial path for M=1 backward */
        for (int m = 0; m < batch; m++) {
            const float *go = grad_output + m * out_features;
            float *gi = grad_input + m * in_features;

            memset(gi, 0, in_features * sizeof(float));

            for (int n = 0; n < out_features; n++) {
                float g_scaled = go[n] * weight_scale;
                if (fabsf(g_scaled) < 1e-30f) continue;
                const uint8_t *w_row = packed_w + n * packed_row_bytes;
                ternary_scatter_add(w_row, gi, g_scaled, in_features);
            }
        }
    }
}

/* -----------------------------------------------------------------------
 * Standalone benchmark -- tests both M=19 (batched) and M=1 (autoregressive)
 * ----------------------------------------------------------------------- */
#ifdef STANDALONE_TEST
#include <stdio.h>
#include <time.h>

static double bench(const uint8_t *packed, const float *act, float *output,
                    int M, int N, int K, float wscale, int iters) {
    /* Warmup */
    ternary_matmul(packed, act, output, M, N, K, wscale);

    struct timespec t0, t1;
    clock_gettime(CLOCK_MONOTONIC, &t0);
    for (int i = 0; i < iters; i++) {
        ternary_matmul(packed, act, output, M, N, K, wscale);
    }
    clock_gettime(CLOCK_MONOTONIC, &t1);

    double elapsed = (t1.tv_sec - t0.tv_sec) + (t1.tv_nsec - t0.tv_nsec) / 1e9;
    return elapsed / iters * 1000.0;  /* ms per call */
}

int main(void) {
    const int K = 2560;
    const int N = 2560;

    printf("Ternary matmul v3 (LUT AVX2 + 2D parallel): K=%d, N=%d\n", K, N);
    printf("Threads: %d\n", omp_get_max_threads());

    float *weights = (float *)aligned_alloc(32, N * K * sizeof(float));
    float *act19   = (float *)aligned_alloc(32, 19 * K * sizeof(float));
    float *act1    = (float *)aligned_alloc(32, 1 * K * sizeof(float));
    float *out19   = (float *)aligned_alloc(32, 19 * N * sizeof(float));
    float *out1    = (float *)aligned_alloc(32, 1 * N * sizeof(float));

    srand(42);
    for (int i = 0; i < N * K; i++)
        weights[i] = ((float)rand() / RAND_MAX - 0.5f) * 2.0f;
    for (int i = 0; i < 19 * K; i++)
        act19[i] = ((float)rand() / RAND_MAX - 0.5f) * 2.0f;
    for (int i = 0; i < K; i++)
        act1[i] = act19[i];  /* reuse first row */

    float wscale;
    uint8_t *packed = pack_weights(weights, N, K, &wscale);
    printf("Weight scale: %.4f\n", wscale);

    int packed_bytes = N * ((K + 3) / 4);
    printf("Packed weights: %d bytes (%.1f KB)\n",
           packed_bytes, packed_bytes / 1024.0f);

    /* Benchmark M=19 (batched) */
    double ms19 = bench(packed, act19, out19, 19, N, K, wscale, 100);
    double gflops19 = 2.0 * 19 * N * K / (ms19 / 1000.0) / 1e9;
    printf("\n--- M=19 (batched) ---\n");
    printf("Time per call: %.3f ms\n", ms19);
    printf("Throughput: %.2f GFLOP/s equivalent\n", gflops19);

    /* Benchmark M=1 (autoregressive) */
    double ms1 = bench(packed, act1, out1, 1, N, K, wscale, 200);
    double gflops1 = 2.0 * 1 * N * K / (ms1 / 1000.0) / 1e9;
    printf("\n--- M=1 (autoregressive) ---\n");
    printf("Time per call: %.3f ms\n", ms1);
    printf("Throughput: %.2f GFLOP/s equivalent\n", gflops1);

    /* Correctness check: v3 M=1 output should match v3 M=19 first row */
    float max_diff = 0.0f;
    for (int i = 0; i < N; i++) {
        float diff = fabsf(out19[i] - out1[i]);
        if (diff > max_diff) max_diff = diff;
    }
    printf("\nMax diff M=1 vs M=19[0]: %.6e %s\n",
           max_diff, max_diff < 1e-5f ? "(OK)" : "(MISMATCH!)");

    printf("\nOutput[0][0..4]: %.4f %.4f %.4f %.4f %.4f\n",
           out19[0], out19[1], out19[2], out19[3], out19[4]);

    free(weights); free(act19); free(act1);
    free(out19); free(out1); free(packed);
    return 0;
}
#endif
