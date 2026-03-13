/*
 * ternary_matmul.c -- AVX2 "soft-chip" for BitNet ternary matmul
 *
 * Performs: output = ternary_weights @ activation^T
 *
 * Ternary weights are packed 2 bits each: 00=zero, 01=+1, 11=-1
 * Activations are FP32 (converted from BF16 by caller).
 * Output is FP32.
 *
 * The key insight: ternary matmul needs NO multiply.
 *   +1 -> add the activation
 *   -1 -> subtract the activation
 *    0 -> skip (51% of weights -- free)
 *
 * AVX2 processes 8 FP32 values per instruction.
 * We pack 128 ternary weights into 32 bytes (one AVX2 register).
 *
 * Compile: gcc -O3 -mavx2 -mfma -march=native -shared -fPIC \
 *          -o ternary_matmul.so ternary_matmul.c
 */

#include <immintrin.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

/* -----------------------------------------------------------------------
 * Weight packing: BF16 master weights -> 2-bit ternary
 *
 * Encoding per weight: 2 bits
 *   00 = 0
 *   01 = +1
 *   11 = -1
 *
 * Pack 4 weights per byte, row-major.
 * For a (out_features x in_features) matrix:
 *   packed size = out_features * ceil(in_features / 4) bytes
 * ----------------------------------------------------------------------- */

/*
 * Pack a single row of FP32 weights into ternary 2-bit representation.
 * weights: FP32 master weights (length = in_features)
 * packed:  output buffer (length = ceil(in_features / 4) bytes)
 * scale:   1.0 / mean(|weights|), used for absmean quantization
 */
static void pack_row(const float *weights, uint8_t *packed,
                     int in_features, float scale) {
    for (int i = 0; i < in_features; i += 4) {
        uint8_t byte = 0;
        for (int j = 0; j < 4 && (i + j) < in_features; j++) {
            float w = weights[i + j];
            float wq = roundf(w * scale);
            if (wq > 1.0f)  wq = 1.0f;
            if (wq < -1.0f) wq = -1.0f;

            uint8_t code;
            if (wq > 0.5f)       code = 0x01;  /* +1 */
            else if (wq < -0.5f) code = 0x03;  /* -1 */
            else                  code = 0x00;  /*  0 */

            byte |= (code << (j * 2));
        }
        packed[i / 4] = byte;
    }
}

/*
 * Pack entire weight matrix from FP32 to 2-bit ternary.
 * Returns malloc'd buffer. Caller must free.
 * Also returns the weight scale (1/mean_abs) for output rescaling.
 */
uint8_t *pack_weights(const float *weights, int out_features, int in_features,
                      float *out_weight_scale) {
    /* Compute absmean scale */
    double sum_abs = 0.0;
    int total = out_features * in_features;
    for (int i = 0; i < total; i++) {
        sum_abs += fabsf(weights[i]);
    }
    float mean_abs = (float)(sum_abs / total);
    float scale = 1.0f / fmaxf(mean_abs, 1e-5f);

    *out_weight_scale = mean_abs;  /* Store for output rescaling */

    int packed_row_bytes = (in_features + 3) / 4;
    uint8_t *packed = (uint8_t *)calloc(out_features * packed_row_bytes, 1);

    for (int row = 0; row < out_features; row++) {
        pack_row(weights + row * in_features,
                 packed + row * packed_row_bytes,
                 in_features, scale);
    }
    return packed;
}

/* -----------------------------------------------------------------------
 * AVX2 ternary matmul kernel
 *
 * Computes: output[m, n] = sum_k( ternary_w[n, k] * activation[m, k] )
 *
 * For each output row n and input row m:
 *   Walk along k in chunks of 8 (AVX2 FP32 width).
 *   Unpack 8 ternary codes, generate add/sub masks, accumulate.
 * ----------------------------------------------------------------------- */

/*
 * Ternary dot product of one packed weight row with one activation row.
 * Returns the FP32 result (before weight_scale rescaling).
 *
 * packed_w:  2-bit packed ternary weights (in_features / 4 bytes)
 * act:       FP32 activation vector (in_features)
 * in_features: vector length (must be multiple of 8 for AVX2 fast path)
 */
static inline float ternary_dot_avx2(const uint8_t *packed_w,
                                     const float *act,
                                     int in_features) {
    __m256 acc = _mm256_setzero_ps();

    int k = 0;

    /* Main loop: process 8 elements per iteration.
     * 8 ternary values = 16 bits = 2 bytes of packed data. */
    for (; k + 7 < in_features; k += 8) {
        /* Load 2 bytes of packed weights (8 ternary values) */
        uint16_t pack16 = *(const uint16_t *)(packed_w + k / 4);

        /* Load 8 FP32 activations */
        __m256 a = _mm256_loadu_ps(act + k);

        /* Decode ternary values and build masks.
         * For each of 8 positions: extract 2-bit code.
         * code=01 -> +1 (add), code=11 -> -1 (subtract), code=00 -> 0 (skip)
         */

        /* Strategy: separate into "nonzero" mask and "sign" mask.
         * nonzero: bit0 of each 2-bit pair (code & 1) -> 1 if +1 or -1
         * sign:    bit1 of each 2-bit pair (code >> 1 & 1) -> 1 if -1
         *
         * result = nonzero ? (sign ? -act : +act) : 0
         *        = nonzero_mask & (act XOR sign_mask)
         *
         * where sign_mask flips the sign bit of FP32.
         */
        /* Build per-element int32 masks from packed bits */
        __m256i nonzero_mask_i = _mm256_setzero_si256();
        __m256i sign_mask_i = _mm256_setzero_si256();

        /* Unroll: extract each 2-bit code into its lane */
        int32_t nz[8], sg[8];
        for (int j = 0; j < 8; j++) {
            uint8_t code = (pack16 >> (j * 2)) & 0x03;
            nz[j] = (code & 1) ? (int32_t)0xFFFFFFFF : 0;
            sg[j] = (code >> 1)  ? (int32_t)0x80000000 : 0;  /* FP32 sign bit */
        }

        nonzero_mask_i = _mm256_loadu_si256((const __m256i *)nz);
        sign_mask_i = _mm256_loadu_si256((const __m256i *)sg);

        /* Flip sign of activation where weight == -1 */
        __m256 signed_a = _mm256_xor_ps(a, _mm256_castsi256_ps(sign_mask_i));

        /* Zero out where weight == 0 */
        __m256 masked = _mm256_and_ps(signed_a, _mm256_castsi256_ps(nonzero_mask_i));

        acc = _mm256_add_ps(acc, masked);
    }

    /* Horizontal sum of 8 accumulators */
    __m128 hi = _mm256_extractf128_ps(acc, 1);
    __m128 lo = _mm256_castps256_ps128(acc);
    __m128 sum4 = _mm_add_ps(lo, hi);
    __m128 shuf = _mm_movehdup_ps(sum4);
    __m128 sum2 = _mm_add_ps(sum4, shuf);
    __m128 shuf2 = _mm_movehl_ps(sum2, sum2);
    __m128 sum1 = _mm_add_ss(sum2, shuf2);
    float result = _mm_cvtss_f32(sum1);

    /* Scalar tail for remainder */
    for (; k < in_features; k++) {
        uint8_t byte = packed_w[k / 4];
        uint8_t code = (byte >> ((k % 4) * 2)) & 0x03;
        if (code == 0x01)      result += act[k];
        else if (code == 0x03) result -= act[k];
    }

    return result;
}

/* -----------------------------------------------------------------------
 * Public API: ternary matmul
 *
 * output[m][n] = sum_k ternary(weight[n][k]) * act_quantized[m][k]
 *
 * Handles activation quantization (symmetric INT8-style per-token scaling)
 * and output rescaling by weight_scale, matching BitNet's forward pass.
 *
 * packed_w:     2-bit packed weights (out_features x ceil(in_features/4))
 * activation:   FP32 input (batch x in_features), already BF16->FP32
 * output:       FP32 output (batch x out_features)
 * weight_scale: mean(|master_weights|) for output rescaling
 * ----------------------------------------------------------------------- */
void ternary_matmul(const uint8_t *packed_w,
                    const float *activation,
                    float *output,
                    int batch,
                    int out_features,
                    int in_features,
                    float weight_scale) {

    int packed_row_bytes = (in_features + 3) / 4;

    /* Activation quantization buffer (symmetric INT8 simulation in FP32) */
    float *act_q = (float *)aligned_alloc(32, in_features * sizeof(float));

    for (int m = 0; m < batch; m++) {
        const float *act_row = activation + m * in_features;

        /* --- Activation quantization (per-token symmetric INT8) --- */
        /* Find max absolute value */
        __m256 vmax = _mm256_setzero_ps();
        int k = 0;
        for (; k + 7 < in_features; k += 8) {
            __m256 v = _mm256_loadu_ps(act_row + k);
            /* Absolute value: clear sign bit */
            __m256 va = _mm256_andnot_ps(_mm256_set1_ps(-0.0f), v);
            vmax = _mm256_max_ps(vmax, va);
        }
        /* Horizontal max */
        __m128 hi = _mm256_extractf128_ps(vmax, 1);
        __m128 lo = _mm256_castps256_ps128(vmax);
        __m128 m4 = _mm_max_ps(lo, hi);
        m4 = _mm_max_ps(m4, _mm_movehl_ps(m4, m4));
        m4 = _mm_max_ps(m4, _mm_movehdup_ps(m4));
        float amax = _mm_cvtss_f32(m4);
        /* Scalar tail */
        for (; k < in_features; k++) {
            float av = fabsf(act_row[k]);
            if (av > amax) amax = av;
        }

        float act_scale = 127.0f / fmaxf(amax, 1e-5f);
        float act_inv_scale = 1.0f / act_scale;

        /* Quantize activations: round to [-128,127], then dequantize */
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

        /* --- Ternary matmul: dot product each weight row with quantized act --- */
        for (int n = 0; n < out_features; n++) {
            const uint8_t *w_row = packed_w + n * packed_row_bytes;
            float dot = ternary_dot_avx2(w_row, act_q, in_features);
            /* Rescale by weight_scale (undo the 1/mean_abs normalization) */
            output[m * out_features + n] = dot * weight_scale;
        }
    }

    free(act_q);
}

/* -----------------------------------------------------------------------
 * Benchmark entry point: standalone test
 * ----------------------------------------------------------------------- */
#ifdef STANDALONE_TEST
#include <stdio.h>
#include <time.h>

int main(void) {
    const int M = 19;      /* batch/seq_len */
    const int K = 2560;    /* in_features */
    const int N = 2560;    /* out_features */

    printf("Ternary matmul benchmark: M=%d, K=%d, N=%d\n", M, K, N);

    /* Allocate */
    float *weights = (float *)aligned_alloc(32, N * K * sizeof(float));
    float *act     = (float *)aligned_alloc(32, M * K * sizeof(float));
    float *output  = (float *)aligned_alloc(32, M * N * sizeof(float));

    /* Random init (simulate BitNet weight distribution) */
    srand(42);
    for (int i = 0; i < N * K; i++)
        weights[i] = ((float)rand() / RAND_MAX - 0.5f) * 2.0f;
    for (int i = 0; i < M * K; i++)
        act[i] = ((float)rand() / RAND_MAX - 0.5f) * 2.0f;

    /* Pack weights */
    float wscale;
    uint8_t *packed = pack_weights(weights, N, K, &wscale);
    printf("Weight scale (mean_abs): %.4f\n", wscale);

    int packed_row_bytes = (K + 3) / 4;
    printf("Packed weight size: %d bytes (%.2f KB)\n",
           N * packed_row_bytes, (float)(N * packed_row_bytes) / 1024.0f);

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
    double per_call = elapsed / iters * 1000.0; /* ms */
    printf("Time per call: %.3f ms (%d iters, %.3f s total)\n",
           per_call, iters, elapsed);
    printf("Throughput: %.2f GFLOP/s equivalent\n",
           2.0 * M * N * K / (per_call / 1000.0) / 1e9);

    /* Print a few output values for sanity */
    printf("Output[0][0..4]: %.4f %.4f %.4f %.4f %.4f\n",
           output[0], output[1], output[2], output[3], output[4]);

    free(weights);
    free(act);
    free(output);
    free(packed);
    return 0;
}
#endif
