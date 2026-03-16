#include <immintrin.h>
#include <stdint.h>
#include <string.h>
#include <math.h>

// ---------------------------------------------------------------------------
// PRNG: SplitMix64 + Xorshift128+ (AVX2)
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

// Decode one weight row into dst[0..K-1] using on-the-fly PRNG.
// Each pair of bits encodes: bit0=nonzero, bit1=sign.
static void decode_row(float *dst, int K, uint64_t row_seed) {
    const __m256i bsel = _mm256_set_epi32(128,64,32,16,8,4,2,1);
    const __m256i zero = _mm256_setzero_si256();
    xs128p prng; seed_xs(&prng, row_seed);
    int k = 0;
    for (; k + 128 <= K; k += 128) {
        __m256i bits = xs_next(&prng);
        uint32_t ent[8] __attribute__((aligned(32)));
        _mm256_store_si256((__m256i*)ent, bits);
        for (int i = 0; i < 8; i++) {
            uint32_t e = ent[i];
            for (int s = 0; s < 2; s++) {
                uint32_t w = (e >> (s*16)) & 0xFFFF;
                uint32_t nz=0, sg=0;
                for (int j=0;j<8;j++) { nz|=((w>>(j*2))&1)<<j; sg|=((w>>(j*2+1))&1)<<j; }
                __m256i nzm = _mm256_cmpgt_epi32(_mm256_and_si256(_mm256_set1_epi32(nz),bsel),zero);
                __m256i sgm = _mm256_slli_epi32(_mm256_cmpgt_epi32(_mm256_and_si256(_mm256_set1_epi32(sg),bsel),zero),31);
                __m256 wv = _mm256_and_ps(_mm256_xor_ps(_mm256_set1_ps(1.f),_mm256_castsi256_ps(sgm)),_mm256_castsi256_ps(nzm));
                _mm256_storeu_ps(&dst[k+i*16+s*8], wv);
            }
        }
    }
    if (k < K) {
        xs128p pt; seed_xs(&pt, row_seed);
        for (int kk=0;kk<k;kk+=128) xs_next(&pt);
        __m256i bits = xs_next(&pt);
        uint32_t ent[8] __attribute__((aligned(32)));
        _mm256_store_si256((__m256i*)ent, bits);
        const __m256i bsel2=_mm256_set_epi32(128,64,32,16,8,4,2,1);
        const __m256i z2=_mm256_setzero_si256();
        for (int i=0;i<8&&k+i*16<K;i++) {
            uint32_t e=ent[i];
            for (int s=0;s<2;s++) {
                int ki=k+i*16+s*8; if(ki>=K) break;
                uint32_t w=(e>>(s*16))&0xFFFF;
                uint32_t nz=0,sg=0;
                for(int j=0;j<8;j++){nz|=((w>>(j*2))&1)<<j;sg|=((w>>(j*2+1))&1)<<j;}
                __m256i nzm=_mm256_cmpgt_epi32(_mm256_and_si256(_mm256_set1_epi32(nz),bsel2),z2);
                __m256i sgm=_mm256_slli_epi32(_mm256_cmpgt_epi32(_mm256_and_si256(_mm256_set1_epi32(sg),bsel2),z2),31);
                __m256 wv=_mm256_and_ps(_mm256_xor_ps(_mm256_set1_ps(1.f),_mm256_castsi256_ps(sgm)),_mm256_castsi256_ps(nzm));
                float tmp[8]; _mm256_storeu_ps(tmp,wv);
                for(int j=0;j<8&&ki+j<K;j++) dst[ki+j]=tmp[j];
            }
        }
    }
}

// ---------------------------------------------------------------------------
// ghost_matmul_forward: output[M,N] = input[M,K] x W[N,K]^T
// Strategy: generate each weight row once, dot against all M batch rows.
// PRNG cost amortised over batch — no heap alloc, stack only.
// ---------------------------------------------------------------------------
void ghost_matmul_forward(
    const float *__restrict__ input,
    float       *__restrict__ output,
    int M, int K, int N,
    uint64_t base_seed, int layer_id
) {
    const uint64_t lseed = base_seed ^ ((uint64_t)layer_id * 0x85ebca6bULL);
    const int K8 = (K + 7) & ~7;
    float *wrow = (float*)__builtin_alloca((size_t)K8 * sizeof(float));

    for (int n = 0; n < N; n++) {
        decode_row(wrow, K, lseed ^ ((uint64_t)n * 0xc2b2ae35ULL));
        for (int m = 0; m < M; m++) {
            const float *x = &input[m*K];
            __m256 acc = _mm256_setzero_ps();
            int k;
            for (k = 0; k+8 <= K; k += 8)
                acc = _mm256_fmadd_ps(_mm256_loadu_ps(&x[k]), _mm256_loadu_ps(&wrow[k]), acc);
            float s = 0.f;
            for (; k < K; k++) s += x[k] * wrow[k];
            float buf[8]; _mm256_storeu_ps(buf, acc);
            output[m*N+n] = buf[0]+buf[1]+buf[2]+buf[3]+buf[4]+buf[5]+buf[6]+buf[7]+s;
        }
    }
}

// ---------------------------------------------------------------------------
// ghost_matmul_backward: grad_input[M,K] += grad_output[M,N] x W[N,K]
// ---------------------------------------------------------------------------
void ghost_matmul_backward(
    const float *__restrict__ grad_output,
    float       *__restrict__ grad_input,
    int M, int K, int N,
    uint64_t base_seed, int layer_id
) {
    const uint64_t lseed = base_seed ^ ((uint64_t)layer_id * 0x85ebca6bULL);
    const int K8 = (K + 7) & ~7;
    float *wrow = (float*)__builtin_alloca((size_t)K8 * sizeof(float));

    memset(grad_input, 0, (size_t)M * K * sizeof(float));

    for (int n = 0; n < N; n++) {
        decode_row(wrow, K, lseed ^ ((uint64_t)n * 0xc2b2ae35ULL));
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

// No-op stubs kept for ABI compatibility
void ghost_free_cache(void) {}
void ghost_warm_cache(int layer_id, int K, int N, uint64_t base_seed) {
    (void)layer_id; (void)K; (void)N; (void)base_seed;
}
