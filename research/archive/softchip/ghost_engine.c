#include <immintrin.h>
#include <stdint.h>
#include <math.h>
#include <string.h>
#include <stdlib.h>
#include <stdio.h>
#include <omp.h>

// ---------------------------------------------------------------------------
// Constants & Settings
// ---------------------------------------------------------------------------
#define HIDDEN_DIM 2560
#define INTERMEDIATE_DIM 6912
#define N_LAYERS 30
#define N_HEADS 20
#define N_KV_HEADS 5
#define HEAD_DIM 128
#define VOCAB_SIZE 128256
#define MAX_SEQ_LEN 4096

// ---------------------------------------------------------------------------
// PRNG (SplitMix64 + XorShift128+)
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
// LUT for Weight Regeneration (Packed 2-bit format)
// ---------------------------------------------------------------------------
static uint32_t g_lut_nz[256][4] __attribute__((aligned(32)));
static uint32_t g_lut_sign[256][4] __attribute__((aligned(32)));
static int g_luts_ready = 0;

static void build_luts(void) {
    if (g_luts_ready) return;
    for (int b = 0; b < 256; b++) {
        for (int j = 0; j < 4; j++) {
            uint8_t code = (b >> (j * 2)) & 0x03;
            g_lut_nz[b][j]   = (code & 1) ? 0xFFFFFFFF : 0;
            g_lut_sign[b][j] = (code >> 1) ? 0x80000000 : 0;
        }
    }
    g_luts_ready = 1;
}

static void decode_row_packed(uint8_t *dst, int K, uint64_t row_seed) {
    xs128p prng; seed_xs(&prng, row_seed);
    int k = 0;
    for (; k + 128 <= K; k += 128) {
        __m256i bits = xs_next(&prng);
        _mm256_storeu_si256((__m256i*)&dst[k/4], bits);
    }
}

// ---------------------------------------------------------------------------
// AVX2 Ternary Kernels
// ---------------------------------------------------------------------------
static inline float ternary_dot(const uint8_t *packed_w, const float *act, int K) {
    __m256 acc = _mm256_setzero_ps();
    for (int k = 0; k < K; k += 8) {
        uint8_t b0 = packed_w[k / 4];
        uint8_t b1 = packed_w[k / 4 + 1];
        __m256 a = _mm256_loadu_ps(act + k);
        __m256i nz = _mm256_set_m128i(_mm_load_si128((const __m128i *)g_lut_nz[b1]), _mm_load_si128((const __m128i *)g_lut_nz[b0]));
        __m256i sg = _mm256_set_m128i(_mm_load_si128((const __m128i *)g_lut_sign[b1]), _mm_load_si128((const __m128i *)g_lut_sign[b0]));
        __m256 signed_a = _mm256_xor_ps(a, _mm256_castsi256_ps(sg));
        acc = _mm256_add_ps(acc, _mm256_and_ps(signed_a, _mm256_castsi256_ps(nz)));
    }
    float tmp[8]; _mm256_storeu_ps(tmp, acc);
    return tmp[0]+tmp[1]+tmp[2]+tmp[3]+tmp[4]+tmp[5]+tmp[6]+tmp[7];
}

static void batched_ghost_matmul(float *outs, const float *ins, int G, int K, int N, uint64_t lseed, float scale) {
    int packed_row_bytes = K / 4;
    uint8_t *packed_matrix = (uint8_t*)malloc(N * packed_row_bytes);
    
    #pragma omp parallel for
    for (int n = 0; n < N; n++) {
        decode_row_packed(packed_matrix + n * packed_row_bytes, K, lseed ^ ((uint64_t)n * 0xc2b2ae35ULL));
    }
    
    #pragma omp parallel for collapse(2)
    for (int g = 0; g < G; g++) {
        for (int n = 0; n < N; n++) {
            float d = ternary_dot(packed_matrix + n * packed_row_bytes, ins + g * HIDDEN_DIM, K);
            outs[g * N + n] = d * scale;
        }
    }
    
    free(packed_matrix);
}

// ---------------------------------------------------------------------------
// Expert & Norm Helpers
// ---------------------------------------------------------------------------
static void rmsnorm(float *out, const float *x, int dim) {
    __m256 sum_sq = _mm256_setzero_ps();
    for (int i = 0; i < dim; i += 8) {
        __m256 v = _mm256_loadu_ps(&x[i]);
        sum_sq = _mm256_fmadd_ps(v, v, sum_sq);
    }
    float tmp[8]; _mm256_storeu_ps(tmp, sum_sq);
    float s = tmp[0]+tmp[1]+tmp[2]+tmp[3]+tmp[4]+tmp[5]+tmp[6]+tmp[7];
    float r = 1.0f / sqrtf(s / dim + 1e-5f);
    __m256 vr = _mm256_set1_ps(r);
    for (int i = 0; i < dim; i += 8) _mm256_storeu_ps(&out[i], _mm256_mul_ps(_mm256_loadu_ps(&x[i]), vr));
}

typedef struct { float *u, *v, *scale; } ExpertData;

static void apply_experts_batched(float *outs, const float *ins, int G, int in_f, int out_f, const ExpertData *experts) {
    for (int g = 0; g < G; g++) {
        for (int e = 0; e < 4; e++) {
            const ExpertData *exp = &experts[e];
            float dot = 0;
            __m256 acc = _mm256_setzero_ps();
            for (int i = 0; i < in_f; i += 8) acc = _mm256_fmadd_ps(_mm256_loadu_ps(&ins[g * HIDDEN_DIM + i]), _mm256_loadu_ps(&exp->v[i]), acc);
            float tmp[8]; _mm256_storeu_ps(tmp, acc);
            for(int i=0; i<8; i++) dot += tmp[i];
            float s = dot * (*exp->scale);
            __m256 vs = _mm256_set1_ps(s);
            for (int i = 0; i < out_f; i += 8) {
                __m256 vo = _mm256_loadu_ps(&outs[g * out_f + i]);
                _mm256_storeu_ps(&outs[g * out_f + i], _mm256_fmadd_ps(vs, _mm256_loadu_ps(&exp->u[i]), vo));
            }
        }
    }
}

// ---------------------------------------------------------------------------
// RoPE Precomputation
// ---------------------------------------------------------------------------
static float g_rope_cos[MAX_SEQ_LEN][HEAD_DIM/2];
static float g_rope_sin[MAX_SEQ_LEN][HEAD_DIM/2];
static int g_rope_ready = 0;

static void init_rope(void) {
    if (g_rope_ready) return;
    for (int t = 0; t < MAX_SEQ_LEN; t++) {
        for (int i = 0; i < HEAD_DIM; i += 2) {
            float freq = powf(500000.0f, -(float)i / HEAD_DIM);
            g_rope_cos[t][i/2] = cosf(t * freq);
            g_rope_sin[t][i/2] = sinf(t * freq);
        }
    }
    g_rope_ready = 1;
}

static void apply_rope(float *v, int pos, int head_dim) {
    for (int i = 0; i < head_dim; i += 2) {
        float c = g_rope_cos[pos][i/2], s = g_rope_sin[pos][i/2];
        float v0 = v[i], v1 = v[i+1];
        v[i] = v0 * c - v1 * s;
        v[i+1] = v0 * s + v1 * c;
    }
}

// ---------------------------------------------------------------------------
// Model & Generator
// ---------------------------------------------------------------------------
typedef struct {
    float *k_cache, *v_cache, *embeddings, *lm_head, *weight_scales;
    uint64_t base_seed;
    ExpertData *experts;
} GhostModel;

void batched_transformer_block(int layer_idx, float *xs, int G, int pos, GhostModel *m) {
    float *x_norms = (float*)aligned_alloc(32, G * HIDDEN_DIM * sizeof(float));
    for (int g = 0; g < G; g++) rmsnorm(x_norms + g * HIDDEN_DIM, xs + g * HIDDEN_DIM, HIDDEN_DIM);
    
    uint64_t lseed = m->base_seed ^ ((uint64_t)layer_idx * 0x85ebca6bULL);
    float *qs = (float*)malloc(G * HIDDEN_DIM * sizeof(float));
    float *ks = (float*)malloc(G * N_KV_HEADS * HEAD_DIM * sizeof(float));
    float *vs = (float*)malloc(G * N_KV_HEADS * HEAD_DIM * sizeof(float));
    
    batched_ghost_matmul(qs, x_norms, G, HIDDEN_DIM, HIDDEN_DIM, lseed ^ 0x1111ULL, m->weight_scales[layer_idx*7 + 0]);
    batched_ghost_matmul(ks, x_norms, G, HIDDEN_DIM, N_KV_HEADS * HEAD_DIM, lseed ^ 0x2222ULL, m->weight_scales[layer_idx*7 + 1]);
    batched_ghost_matmul(vs, x_norms, G, HIDDEN_DIM, N_KV_HEADS * HEAD_DIM, lseed ^ 0x3333ULL, m->weight_scales[layer_idx*7 + 2]);
    
    apply_experts_batched(qs, x_norms, G, HIDDEN_DIM, HIDDEN_DIM, &m->experts[(layer_idx*7 + 0)*4]);
    apply_experts_batched(ks, x_norms, G, HIDDEN_DIM, N_KV_HEADS * HEAD_DIM, &m->experts[(layer_idx*7 + 1)*4]);
    apply_experts_batched(vs, x_norms, G, HIDDEN_DIM, N_KV_HEADS * HEAD_DIM, &m->experts[(layer_idx*7 + 2)*4]);

    for (int g = 0; g < G; g++) {
        for (int h = 0; h < N_HEADS; h++) apply_rope(qs + g * HIDDEN_DIM + h * HEAD_DIM, pos, HEAD_DIM);
        for (int h = 0; h < N_KV_HEADS; h++) {
            apply_rope(ks + g * N_KV_HEADS * HEAD_DIM + h * HEAD_DIM, pos, HEAD_DIM);
            memcpy(m->k_cache + ((layer_idx * G + g) * MAX_SEQ_LEN + pos) * N_KV_HEADS * HEAD_DIM, ks + g * N_KV_HEADS * HEAD_DIM, HEAD_DIM * sizeof(float));
            memcpy(m->v_cache + ((layer_idx * G + g) * MAX_SEQ_LEN + pos) * N_KV_HEADS * HEAD_DIM, vs + g * N_KV_HEADS * HEAD_DIM, HEAD_DIM * sizeof(float));
        }
    }

    float *attn_outs = (float*)calloc(G * HIDDEN_DIM, sizeof(float));
    #pragma omp parallel for
    for (int g = 0; g < G; g++) {
        for (int h = 0; h < N_HEADS; h++) {
            int kv_h = h / (N_HEADS / N_KV_HEADS);
            float scores[MAX_SEQ_LEN];
            float *qh = qs + g * HIDDEN_DIM + h * HEAD_DIM;
            for (int t = 0; t <= pos; t++) {
                float *kt = m->k_cache + ((layer_idx * G + g) * MAX_SEQ_LEN + t) * N_KV_HEADS * HEAD_DIM + kv_h * HEAD_DIM;
                float dot = 0; for (int d = 0; d < HEAD_DIM; d++) dot += qh[d] * kt[d];
                scores[t] = dot / sqrtf(HEAD_DIM);
            }
            float max_s = -1e30f; for (int i = 0; i <= pos; i++) if (scores[i] > max_s) max_s = scores[i];
            float sum_s = 0; for (int i = 0; i <= pos; i++) { scores[i] = expf(scores[i] - max_s); sum_s += scores[i]; }
            for (int i = 0; i <= pos; i++) {
                float *vt_row = m->v_cache + ((layer_idx * G + g) * MAX_SEQ_LEN + i) * N_KV_HEADS * HEAD_DIM + kv_h * HEAD_DIM;
                float scale = scores[i] / sum_s;
                for (int d = 0; d < HEAD_DIM; d++) attn_outs[g * HIDDEN_DIM + h * HEAD_DIM + d] += scale * vt_row[d];
            }
        }
    }

    float *o_projs = (float*)malloc(G * HIDDEN_DIM * sizeof(float));
    batched_ghost_matmul(o_projs, attn_outs, G, HIDDEN_DIM, HIDDEN_DIM, lseed ^ 0x4444ULL, m->weight_scales[layer_idx*7 + 3]);
    apply_experts_batched(o_projs, attn_outs, G, HIDDEN_DIM, HIDDEN_DIM, &m->experts[(layer_idx*7 + 3)*4]);
    for (int i = 0; i < G * HIDDEN_DIM; i++) xs[i] += o_projs[i];

    for (int g = 0; g < G; g++) rmsnorm(x_norms + g * HIDDEN_DIM, xs + g * HIDDEN_DIM, HIDDEN_DIM);
    float *gates = (float*)malloc(G * INTERMEDIATE_DIM * sizeof(float));
    float *ups = (float*)malloc(G * INTERMEDIATE_DIM * sizeof(float));
    float *downs = (float*)malloc(G * HIDDEN_DIM * sizeof(float));
    
    batched_ghost_matmul(gates, x_norms, G, HIDDEN_DIM, INTERMEDIATE_DIM, lseed ^ 0x5555ULL, m->weight_scales[layer_idx*7 + 4]);
    batched_ghost_matmul(ups,   x_norms, G, HIDDEN_DIM, INTERMEDIATE_DIM, lseed ^ 0x6666ULL, m->weight_scales[layer_idx*7 + 5]);
    apply_experts_batched(gates, x_norms, G, HIDDEN_DIM, INTERMEDIATE_DIM, &m->experts[(layer_idx*7 + 4)*4]);
    apply_experts_batched(ups,   x_norms, G, HIDDEN_DIM, INTERMEDIATE_DIM, &m->experts[(layer_idx*7 + 5)*4]);
    
    for (int i = 0; i < G * INTERMEDIATE_DIM; i++) {
        float v = (gates[i] > 0) ? gates[i] * gates[i] * ups[i] : 0;
        gates[i] = v;
    }
    batched_ghost_matmul(downs, gates, G, INTERMEDIATE_DIM, HIDDEN_DIM, lseed ^ 0x7777ULL, m->weight_scales[layer_idx*7 + 6]);
    apply_experts_batched(downs, gates, G, INTERMEDIATE_DIM, HIDDEN_DIM, &m->experts[(layer_idx*7 + 6)*4]);
    for (int i = 0; i < G * HIDDEN_DIM; i++) xs[i] += downs[i];
    
    free(x_norms); free(qs); free(ks); free(vs); free(attn_outs); free(o_projs); free(gates); free(ups); free(downs);
}

void ghost_engine_generate_batched(GhostModel *m, int *tokens, int G, int prompt_len, int gen_len, float temp) {
    build_luts(); init_rope();
    float *xs = (float*)malloc(G * HIDDEN_DIM * sizeof(float));
    int *finished = (int*)calloc(G, sizeof(int));
    
    for (int p = 0; p < prompt_len + gen_len - 1; p++) {
        for (int g = 0; g < G; g++) {
            if (finished[g]) memset(xs + g * HIDDEN_DIM, 0, HIDDEN_DIM * sizeof(float));
            else memcpy(xs + g * HIDDEN_DIM, m->embeddings + tokens[g * (prompt_len + gen_len) + p] * HIDDEN_DIM, HIDDEN_DIM * sizeof(float));
        }
        
        for (int l = 0; l < N_LAYERS; l++) batched_transformer_block(l, xs, G, p, m);
        
        if (p >= prompt_len - 1) {
            #pragma omp parallel for
            for (int g = 0; g < G; g++) {
                if (finished[g]) continue;
                float max_l = -1e30f; int best = 0;
                for (int v = 0; v < VOCAB_SIZE; v++) {
                    float dot = 0; for (int d = 0; d < HIDDEN_DIM; d++) dot += xs[g * HIDDEN_DIM + d] * m->lm_head[v * HIDDEN_DIM + d];
                    if (dot > max_l) { max_l = dot; best = v; }
                }
                tokens[g * (prompt_len + gen_len) + p + 1] = best;
                if (best == 128001) finished[g] = 1;
            }
            int all_done = 1; for(int g=0; g<G; g++) if(!finished[g]) all_done = 0;
            if (all_done) break;
        }
    }
    free(xs); free(finished);
}
