# Ghost Inference Engine

## Purpose

The Ghost Inference Engine (`softchip/ghost_engine.c`) is a pure C/AVX2 implementation of the full GhostWeight transformer forward pass. It was written to eliminate Python overhead during GRPO rollouts.

**The problem it solves:** In the original Python-based rollout loop, PyTorch made 210 `hand-off` calls per token (once per layer), and 107,520 hand-offs per training step (512 tokens × 210 layers). Each hand-off crosses the Python/C boundary, stalling the CPU pipeline. The Ghost Engine moves the entire token generation loop into C, eliminating all Python overhead during rollout.

---

## Build

```bash
bash softchip/build_engine.sh
```

Which runs:
```bash
gcc -O3 -mavx2 -mfma -march=native -fopenmp -shared -fPIC \
    -o softchip/ghost_engine.so softchip/ghost_engine.c -lm
```

The output `ghost_engine.so` is loaded at runtime via ctypes.

---

## Data Structures

### ExpertData

```c
typedef struct {
    float *u;      // Output projection vector [out_features]
    float *v;      // Input projection vector [in_features]
    float *scale;  // Pointer to the single learned scalar (lives in PyTorch)
} ExpertData;
```

`u` and `v` are owned by PyTorch (BF16 buffers, cast to float* for the C engine). `scale` is a pointer into the PyTorch parameter tensor — the C engine reads its current value on every call, so gradient updates from the Python training loop are immediately visible to the next rollout without any sync step.

### GhostModel

```c
typedef struct {
    float *k_cache;       // KV-cache keys   [N_LAYERS × G × MAX_SEQ_LEN × N_KV_HEADS × HEAD_DIM]
    float *v_cache;       // KV-cache values [same]
    float *embeddings;    // Token embeddings [VOCAB_SIZE × HIDDEN_DIM]
    float *lm_head;       // LM head weights  [VOCAB_SIZE × HIDDEN_DIM] (FP32)
    float *weight_scales; // Per-layer weight scales [N_LAYERS × 7]
    uint64_t base_seed;   // PRNG seed for ghost weight generation
    ExpertData *experts;  // Array of 840 experts [N_LAYERS × 7 × 4]
} GhostModel;
```

> **CRITICAL:** The field order in the Python ctypes wrapper **must exactly match** this C declaration. Any mismatch causes an immediate segfault. See [KNOWN_BUGS.md](KNOWN_BUGS.md#bug-002-ctypes-struct-field-order-mismatch-segfault).

---

## KV-Cache Layout

The KV-cache is sized for the full group:

```
[N_LAYERS=30][G=4][MAX_SEQ_LEN=4096][N_KV_HEADS=5][HEAD_DIM=128]
= 30 × 4 × 4096 × 5 × 128 × 4 bytes = 1.25 GB
```

The cache is allocated once at `GhostEngine.__init__` and reused across all rollout calls. It is reset implicitly at the start of each new sequence (position 0 overwrites previous data).

---

## Cache-Coherency as Loop Management

The engine exploits a key insight: **by keeping the entire active working set in the CPU L3 cache, the loop itself becomes cache-resident**.

Memory footprint analysis:
| Buffer | Size | Cache Level |
|---|---|---|
| GhostChain adapter vectors (840 × u,v) | ~8.4 MB | L3 (fits in 16MB) |
| 2MB LUT (weight decode) | 2 MB | L3 (pinned) |
| Active layer activations (G=4 × HIDDEN) | ~40 KB | L1/L2 |
| One weight row (HIDDEN × float) | 10 KB | L1 |

The token generation loop never waits for RAM. Each layer's computation runs entirely from cache. This transforms the engine from **memory-bound** (like all normal LLMs) to **compute-bound** — every CPU cycle is doing arithmetic, not waiting for data.

---

## Ghost Weight Generation

Weights are never stored. They are regenerated from the PRNG for each row of each matmul:

```c
// Per row n of a weight matrix:
uint64_t row_seed = layer_seed ^ (n * 0xc2b2ae35ULL);
decode_row_packed(packed_row, K, row_seed);
float dot = ternary_dot(packed_row, activation, K);
output[n] = dot * weight_scale;
```

The PRNG used is **XorShift128+** seeded via **SplitMix64**, operating on 256-bit AVX2 vectors to generate 128 weights per clock cycle.

Weight encoding is packed 2-bit (`{00=0, 01=+1, 11=-1}`), 4 weights per byte. A 2560-element row requires only 640 bytes of temporary buffer, which fits in L1.

---

## Batched Rollout

`ghost_engine_generate_batched` processes `G` completions simultaneously:

```c
void ghost_engine_generate_batched(GhostModel *m, int *tokens,
                                   int G, int prompt_len, int gen_len, float temp);
```

**Token buffer layout:** `tokens[g * total_len + pos]` — group `g`, position `pos`.

**Key optimization:** Ghost weights for each layer are generated **once** and reused across all `G` activation vectors. This amortizes the expensive PRNG cost over the group size, reducing effective weight-generation overhead by `G×`.

```c
// Generate matrix once
#pragma omp parallel for
for (int n = 0; n < N; n++) decode_row_packed(matrix + n * row_bytes, K, seed ^ n*SALT);

// Apply to all G activations
#pragma omp parallel for collapse(2)
for (int g = 0; g < G; g++)
    for (int n = 0; n < N; n++)
        out[g*N + n] = ternary_dot(matrix + n*row_bytes, in + g*HIDDEN, K) * scale;
```

---

## Python Bridge (ghost_engine_wrapper.py)

The wrapper is responsible for:
1. Loading `ghost_engine.so` via `ctypes.CDLL`
2. Keeping PyTorch tensors alive (preventing garbage collection while C holds pointers)
3. Updating expert scale pointers before each rollout (scales change each training step)
4. Translating between numpy token arrays and PyTorch tensors

**Critical invariant:** All tensors passed to the C engine must be `.contiguous()` and `float32`. BF16 tensors (like expert `u`, `v` buffers) must be cast before passing their `data_ptr()`.

---

## Model Constants

```c
#define HIDDEN_DIM      2560    // BitNet 2B4T hidden size
#define INTERMEDIATE_DIM 6912  // MLP intermediate size
#define N_LAYERS          30   // Transformer depth
#define N_HEADS           20   // Attention heads
#define N_KV_HEADS         5   // KV heads (GQA: 4:1 ratio)
#define HEAD_DIM         128   // Attention head dimension
#define VOCAB_SIZE     128256  // Llama-3 tokenizer
#define MAX_SEQ_LEN     4096   // Maximum context length
```

---

## Limitations

1. **No temperature sampling** — current implementation uses greedy decoding (argmax). Temperature parameter is accepted but unused. Multinomial sampling for GRPO diversity is not yet implemented.
2. **No RMSNorm weights** — the engine uses unweighted RMSNorm (scale=1.0). The actual model has per-layer learned norms. This introduces a small systematic bias that the GhostChain adapters are expected to absorb.
3. **No MLP activation bias** — uses ReLU² instead of the model's actual SwiGLU. Same reasoning applies.
4. **Single machine** — no distributed support. Designed exclusively for single-socket CPU inference.
