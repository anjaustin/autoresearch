/*
 * ternary_matmul_v2.c -- AVX2 "soft-chip" for BitNet ternary matmul
 *
 * v2: Fully vectorized inner loop. No scalar unpacking in the hot path.
 *
 * Strategy: pack weights 16 per uint32. Process 8 FP32 at a time.
 * Use a precomputed lookup table (LUT) to convert 8 ternary codes
 * (16 bits) into nonzero and sign masks directly.
 *
 * Actually, 16 bits -> 2^16 = 64K LUT entries is too large.
 * Instead: process 4 ternary values (8 bits) via 256-entry LUT,
 * then combine two halves for 8 values.
 *
 * Better approach: pack 4 weights per byte. For each byte, use
 * a 256-entry LUT that gives us a 4-element mask pattern.
 * Two LUT lookups per 8 elements -> fully vectorized.
 *
 * Even better: use vpshufb (PSHUFB) as a 4-bit LUT.
 * Pack weights densely, then use shuffle-based decode.
 *
 * Simplest fast approach for AVX2:
 * - Pack 128 ternary values per 32-byte row chunk (2 bits each)
 * - For 8 values: extract 16 bits, use shifts+masks to build
 *   8x int32 masks WITHOUT scalar loop.
 *
 * Compile: gcc -O3 -mavx2 -mfma -march=native -shared -fPIC \
 *          -o ternary_matmul_v2.so ternary_matmul_v2.c
 */

#include <immintrin.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <omp.h>

/* -----------------------------------------------------------------------
 * Weight packing: same as v1 -- 2 bits per weight, 4 per byte
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
 * Precomputed LUT: for each byte (4 ternary values), store
 * 4x uint32 nonzero masks and 4x uint32 sign masks.
 *
 * lut_nz[byte][j] = 0xFFFFFFFF if value j is nonzero, else 0
 * lut_sign[byte][j] = 0x80000000 if value j is -1, else 0
 * ----------------------------------------------------------------------- */

/* Aligned for AVX loads: 256 entries x 16 bytes = 4KB each -> fits L1 */
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
 * AVX2 ternary dot product -- v2: LUT-based decode
 *
 * Process 8 activations per iteration via two 4-element LUT lookups.
 * ----------------------------------------------------------------------- */
static inline float ternary_dot_v2(const uint8_t *packed_w,
                                   const float *act,
                                   int in_features) {
    __m256 acc = _mm256_setzero_ps();

    int k = 0;
    for (; k + 7 < in_features; k += 8) {
        /* Two bytes encode 8 ternary values (4 per byte) */
        uint8_t b0 = packed_w[k / 4];
        uint8_t b1 = packed_w[k / 4 + 1];

        /* Load 8 activations */
        __m256 a = _mm256_loadu_ps(act + k);

        /* Build 8-element masks from two 4-element LUT lookups */
        /* Lower 4: from b0 */
        __m128i nz_lo  = _mm_load_si128((const __m128i *)lut_nz[b0]);
        __m128i sg_lo  = _mm_load_si128((const __m128i *)lut_sign[b0]);
        /* Upper 4: from b1 */
        __m128i nz_hi  = _mm_load_si128((const __m128i *)lut_nz[b1]);
        __m128i sg_hi  = _mm_load_si128((const __m128i *)lut_sign[b1]);

        /* Combine into 256-bit */
        __m256i nz256 = _mm256_set_m128i(nz_hi, nz_lo);
        __m256i sg256 = _mm256_set_m128i(sg_hi, sg_lo);

        /* Flip sign where weight == -1 */
        __m256 signed_a = _mm256_xor_ps(a, _mm256_castsi256_ps(sg256));
        /* Zero out where weight == 0 */
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
 * Public API
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

    #pragma omp parallel for schedule(dynamic)
    for (int m = 0; m < batch; m++) {
        const float *act_row = activation + m * in_features;

        /* --- Activation quantization (per-token symmetric INT8) --- */
        float *act_q = (float *)aligned_alloc(32, in_features * sizeof(float));

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

        /* --- Ternary matmul --- */
        for (int n = 0; n < out_features; n++) {
            const uint8_t *w_row = packed_w + n * packed_row_bytes;
            float dot = ternary_dot_v2(w_row, act_q, in_features);
            output[m * out_features + n] = dot * weight_scale;
        }

        free(act_q);
    }
}

/* -----------------------------------------------------------------------
 * Standalone benchmark
 * ----------------------------------------------------------------------- */
#ifdef STANDALONE_TEST
#include <stdio.h>
#include <time.h>

int main(void) {
    const int M = 19;
    const int K = 2560;
    const int N = 2560;

    printf("Ternary matmul v2 (LUT-based AVX2): M=%d, K=%d, N=%d\n", M, K, N);

    float *weights = (float *)aligned_alloc(32, N * K * sizeof(float));
    float *act     = (float *)aligned_alloc(32, M * K * sizeof(float));
    float *output  = (float *)aligned_alloc(32, M * N * sizeof(float));

    srand(42);
    for (int i = 0; i < N * K; i++)
        weights[i] = ((float)rand() / RAND_MAX - 0.5f) * 2.0f;
    for (int i = 0; i < M * K; i++)
        act[i] = ((float)rand() / RAND_MAX - 0.5f) * 2.0f;

    float wscale;
    uint8_t *packed = pack_weights(weights, N, K, &wscale);
    printf("Weight scale: %.4f\n", wscale);

    int packed_bytes = N * ((K + 3) / 4);
    printf("Packed weights: %d bytes (%.1f KB) -- fits L2? %s\n",
           packed_bytes, packed_bytes / 1024.0f,
           packed_bytes < 3 * 1024 * 1024 ? "YES" : "NO");

    /* Warmup */
    ternary_matmul(packed, act, output, M, N, K, wscale);

    /* Benchmark */
    int iters = 100;
    struct timespec t0, t1;
    clock_gettime(CLOCK_MONOTONIC, &t0);
    for (int i = 0; i < iters; i++) {
        ternary_matmul(packed, act, output, M, N, K, wscale);
    }
    clock_gettime(CLOCK_MONOTONIC, &t1);

    double elapsed = (t1.tv_sec - t0.tv_sec) + (t1.tv_nsec - t0.tv_nsec) / 1e9;
    double per_call_ms = elapsed / iters * 1000.0;
    double gflops_eq = 2.0 * M * N * K / (per_call_ms / 1000.0) / 1e9;

    printf("Time per call: %.3f ms\n", per_call_ms);
    printf("Throughput: %.2f GFLOP/s equivalent\n", gflops_eq);
    printf("Output[0][0..4]: %.4f %.4f %.4f %.4f %.4f\n",
           output[0], output[1], output[2], output[3], output[4]);

    free(weights); free(act); free(output); free(packed);
    return 0;
}
#endif
