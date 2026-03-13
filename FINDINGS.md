# TinyLoRA on BitNet b1.58: Findings

## Summary

We validated that sub-rank-1 LoRA adapters (TinyLoRA) can successfully attach to, forward through, and receive gradients from Microsoft's natively ternary BitNet b1.58 2B4T model. **4 scalar parameters can measurably steer a 2.41-billion-parameter ternary model.** To our knowledge, this is the first demonstration of sub-rank-1 LoRA on a natively 1.58-bit architecture.

## Background

### The Convergence of Three Ideas

1. **Autoresearch** (Karpathy): An autonomous AI agent that iteratively experiments on a model -- propose a change, train, evaluate, keep or discard, repeat overnight.

2. **TinyLoRA** ("Learning to Reason in 13 Parameters", Morris et al. 2025): Low-rank adapters scaled down to as few as 1 trainable parameter per layer. Key insight: RL (not SFT) enables extreme parameter reduction -- 1000x fewer parameters to reach 90% of full fine-tuning performance.

3. **BitNet b1.58** (Microsoft, 2024-2025): A natively ternary LLM where every weight is {-1, 0, +1}, trained from scratch with quantization-aware training. 2B parameters, 0.4GB inference memory, 29ms/token on CPU.

### The Question

Can continuous-valued LoRA corrections produce gradients through a model whose weights are quantized to three discrete values during the forward pass?

## Validation Setup

- **Model:** `microsoft/bitnet-b1.58-2B-4T-bf16` (BF16 master weights, 4.5GB)
- **Hardware:** AMD Ryzen 5 PRO 5675U (6C/12T), 64GB RAM, CPU only
- **Software:** Python 3.13.9, PyTorch 2.10.0, Transformers 5.3.0
- **Adapter:** Custom TinyLoRA implementation (sub-rank-1 LoRA with 1 learned scalar per layer)

### TinyLoRA Implementation

```python
class TinyLoRA(nn.Module):
    """Sub-rank-1 LoRA: output = base(x) + scale * (x @ v^T) @ u^T
    
    u, v are fixed random vectors. scale is the only trainable parameter.
    This gives exactly 1 trainable parameter per adapted layer.
    """
    def __init__(self, base_layer, seed=42):
        super().__init__()
        self.base_layer = base_layer
        out_features, in_features = base_layer.weight.shape
        gen = torch.Generator().manual_seed(seed)
        self.register_buffer('u', F.normalize(torch.randn(out_features, 1, generator=gen), dim=0))
        self.register_buffer('v', F.normalize(torch.randn(1, in_features, generator=gen), dim=0))
        self.scale = nn.Parameter(torch.zeros(1))  # 1 trainable param

    def forward(self, x):
        return self.base_layer(x) + (x @ self.v.T) @ self.u.T * self.scale
```

## Results

### 5-Step Validation

| Step | Test | Result | Detail |
|------|------|--------|--------|
| 1 | Model loads on CPU | PASS | 2.41B params, 4.5GB, 3.5s load time |
| 2 | Architecture inspection | PASS | 212 `AutoBitLinear` layers identified |
| 3 | TinyLoRA adapter attachment | PASS | 4 layers adapted (4 total trainable params), generation coherent |
| 4 | Gradient flow | PASS | All 4 adapter grads non-zero, base model grads None (frozen) |
| 5 | Weight update | PASS | Scales updated, max logit difference = 1.69 |

### Key Measurements

| Metric | Value |
|--------|-------|
| Model parameters | 2.41B |
| Adapter parameters | 4 (one per adapted layer) |
| Parameter ratio | 1 : 602,000,000 |
| Forward pass (19 tokens, CPU) | 12.1s |
| Backward pass (19 tokens, CPU) | 208.8s |
| Adapter gradient magnitudes | 5e-6 to 1.3e-5 |
| Logit change after 1 update (lr=0.1) | 1.69 |

### Architecture Details

BitNet b1.58 2B4T uses `AutoBitLinear` layers across 30 decoder layers:

```
Per layer:
  - self_attn.q_proj: AutoBitLinear (2560, 2560)
  - self_attn.k_proj: AutoBitLinear (640, 2560)
  - self_attn.v_proj: AutoBitLinear (640, 2560)
  - self_attn.o_proj: AutoBitLinear (2560, 2560)
  - mlp.gate_proj:    AutoBitLinear (6912, 2560)
  - mlp.up_proj:      AutoBitLinear (6912, 2560)
  - mlp.down_proj:    AutoBitLinear (2560, 6912)
```

Total: 210 `AutoBitLinear` layers + embedding + LM head = 212 linear-like layers.

## Analysis

### Why It Works

Standard LoRA adds a bypass path: `output = base(x) + BA(x)`. The bypass operates on the full-precision input `x` independently of what the base layer does internally. BitNet's `AutoBitLinear` quantizes master weights to {-1, 0, +1} during the forward pass, but the LoRA bypass path never touches those ternary weights. The gradient flows through the bypass path cleanly.

### The Backward Pass: From 82.9s to 1.1s (75x Speedup)

The backward pass through 2.41B parameters on a Ryzen 5 CPU initially took 82.9 seconds (6 tokens, soft-chip forward already applied). Two key optimizations brought this to ~1 second:

**Optimization 1: Ternary backward kernel** — The backward pass through frozen AutoBitLinear layers computes `grad_input = W^T @ grad_output`. Since W is ternary post-STE, this is the **same add/subtract/skip pattern** the soft-chip exploits for forward. A `ternary_matmul_backward` kernel was added using an **accumulate-scatter** approach: iterate over N weight rows, scatter-add each row's contribution scaled by `grad_output[n]`. This brought backward from 82.9s to 19.5s (4.3x).

**Optimization 2: FP32 LM head (Pass 5)** — Profiling revealed 92.6% of the remaining 19.5s was spent in TWO matmul calls from the LM head (`nn.Linear(2560, 128256)`) — a dense BF16 layer that wasn't patched by the ternary kernel. Root cause: MKL's BF16 GEMM on Zen 3 (no AMX/VNNI) is 32-90x slower than FP32. A custom `FP32LMHeadFunction` casts BF16↔FP32 at boundaries and does all computation in FP32.

| Metric (6 tokens) | Stock PyTorch | Ternary Backward | + FP32 LM Head | Total Speedup |
|-------------------|--------------|-----------------|----------------|---------------|
| Forward | 8.8s | 1.2s | 1.3s | **6.8x** |
| **Backward** | **82.9s** | **19.5s** | **1.1s** | **75x** |
| **Total** | **91.7s** | **20.7s** | **2.4s** | **38x** |

Backward breakdown after both optimizations:
- TernaryMatmulBackward (210 layers): 755ms (68%)
- FP32 LM head backward: 200ms (18%)
- Other (attention, norms, etc.): 110ms (14%)

A training iteration now takes **2.4 seconds** — fast enough for interactive experimentation (25 iterations/minute).

**Parallelization attempt (LMM Pass 6):** OpenMP parallelization over N within the backward kernel achieved 3.4x kernel speedup in isolation but caused a 9% end-to-end regression due to OpenMP/MKL thread pool contention. Serial backward remains the production path. The backward is at its practical ceiling on this hardware within the PyTorch autograd framework.

### What Remains Unknown

1. **Does RL (GRPO) with TinyLoRA improve BitNet's task performance?** The validation proves mechanics, not effectiveness. The TinyLoRA paper showed RL enables extreme parameter reduction on dense transformers (Qwen2.5-8B). Whether this transfers to ternary architectures requires actual training experiments.

2. **Which layers are highest leverage for adaptation?** We adapted q_proj in the first 4 layers. The full search space (210 layers, various types) is what the autoresearch loop should explore.

3. **What is the minimum parameter count that produces meaningful improvement?** TinyLoRA achieved 91% GSM8K with 13 params on Qwen2.5-8B. The equivalent number for BitNet is unknown.

## AVX2 Soft-Chip: Ternary Matmul Kernel

### Motivation

The 208-second backward pass and 12-second forward pass on CPU are dominated by matrix multiplications through BitNet's `AutoBitLinear` layers. BitNet's ternary weights ({-1, 0, +1}) mean these matmuls don't need any multiplication -- just add, subtract, or skip. PyTorch doesn't exploit this; it runs standard BF16 matmul.

We built a hand-tuned AVX2 "soft-chip" kernel that exploits the ternary structure.

### Architecture Insights from BitNet Internals

Inspection of `AutoBitLinear` revealed:
- `online_quant=True`: BF16 master weights are quantized to ternary via `WeightQuant` (absmean) in the forward pass
- Activations are quantized to symmetric INT8 via `ActQuant` (per-token absmax scaling)
- Backward pass uses Straight-Through Estimator (STE) -- gradients pass through quantization unchanged
- Weight distribution post-quantization: ~51% zeros, ~24.5% +1, ~24.5% -1

### Kernel Design

**Weight packing:** 2 bits per weight (00=zero, 01=+1, 11=-1), 4 weights per byte. A 2560x2560 weight matrix packs to 1.6MB -- fits in L2 cache.

**AVX2 inner loop (v2, LUT-based):**
1. Precompute 256-entry LUTs mapping each byte (4 ternary values) to nonzero masks and sign masks (4KB each, fits L1)
2. Per 8-element chunk: two LUT lookups -> 8-element nonzero and sign masks
3. XOR activations with sign mask (flips sign for -1 weights)
4. AND with nonzero mask (zeros out where weight is 0)
5. Accumulate via `_mm256_add_ps`

No multiply instruction in the hot path. 51% of elements are zero -- free.

**Activation quantization:** Vectorized symmetric INT8 simulation matching BitNet's `ActQuant`.

### Benchmark Results

#### Isolated Layer (M=19, K=2560, N=2560)

| Implementation | Time | Throughput | Speedup vs PyTorch |
|---|---|---|---|
| PyTorch BF16 (BitNet forward) | 53.5 ms | 4.7 GFLOP/s eq | 1x (baseline) |
| Soft-chip v1 (scalar decode) | 221.1 ms | 1.1 GFLOP/s eq | 0.24x |
| Soft-chip v2 (LUT + AVX2) | 13.2 ms | 18.9 GFLOP/s eq | 4.1x |
| **Soft-chip v3 (+ smart threading)** | **13.3 ms** | **18.8 GFLOP/s eq** | **4.0x** |

v3 matches v2 for batched workloads, but adds intelligent threading:

| Batch Size | v2 Time | v3 Time | Notes |
|---|---|---|---|
| M=1 (autoregressive) | 6.9 ms | **1.6 ms** | v3 uses serial path (avoids thread overhead) |
| M=19 (batched) | 12.7 ms | 13.3 ms | Both use batch-parallel OpenMP |

**Key v3 insight:** For small batch sizes (M<6), OpenMP fork/join overhead (~50us/thread x 12 threads) dominates the 1.6ms single-core compute time. Running serial keeps the packed weights in one core's L3 slice, giving 4.3x improvement over the threaded path for M=1.

#### Full Model Forward Pass (30 layers, all 210 AutoBitLinear)

| Metric | PyTorch | Soft-chip v3 | Speedup |
|---|---|---|---|
| Forward M=15 (15-token prompt) | 8.9s | 4.6s | **2.0x** |
| Forward M=1 (autoregressive) | 4.2s | **0.91s** | **4.6x** |
| 200-token rollout (M=1 x 200) | ~840s (14 min) | ~182s (3 min) | **4.6x** |
| Output cosine similarity | - | 0.9997 | - |

The full-model speedup is lower than the isolated kernel speedup because the model includes non-linear layers (RMSNorm, RoPE, attention softmax, SiLU, etc.) and Python/PyTorch overhead that we don't accelerate. The 4.6x M=1 speedup is critical -- it means autoregressive rollout generation (the RL loop bottleneck) is 4.6x faster.

### Numerical Validation

The soft-chip kernel was validated against AutoBitLinear across 6 layers of varying sizes:

| Layer | Shape | NRMSE | Cosine Sim |
|---|---|---|---|
| layer 0, q_proj | 2560x2560 | 2.7e-8 | 1.0000005 |
| layer 0, k_proj | 640x2560 | 3.3e-8 | 1.0000000 |
| layer 0, gate_proj | 6912x2560 | 3.2e-8 | 1.0000005 |
| layer 0, down_proj | 2560x6912 | 7.2e-6 | 1.0000002 |
| layer 15, q_proj | 2560x2560 | 3.2e-8 | 1.0000000 |
| layer 29, o_proj | 2560x2560 | 3.1e-8 | 1.0000000 |

The kernel produces essentially identical output to PyTorch's AutoBitLinear when both use the same FP32 precision for quantization. The larger NRMSE for `down_proj` (6912 input features) is due to longer accumulation chains. Cross-validation against actual AutoBitLinear.forward() (which uses BF16 internally) shows max diff = 0.63 and cosine sim = 0.9997 -- the difference is from BF16 vs FP32 arithmetic, not from the ternary logic.

### PyTorch Integration

The soft-chip is integrated as a drop-in replacement via monkey-patching:

```python
from softchip.torch_ternary import patch_model, patch_lm_head_fp32, unpatch_model

model = AutoModelForCausalLM.from_pretrained(...)
patch_model(model)            # Replaces AutoBitLinear forward+backward with C kernel
patch_lm_head_fp32(model)     # FP32 LM head (17x backward speedup on Ryzen)
output = model(input)         # 6.8x faster forward, 75x faster backward
unpatch_model(model)          # Restore original
```

Weight packing (one-time cost at load): ~48s for 210 layers. The backward pass uses the ternary backward kernel for AutoBitLinear layers (same add/sub/skip as forward) and FP32 matmul for the LM head.

### Source

- `softchip/ternary_matmul.c` -- v1 prototype (scalar decode, 221ms)
- `softchip/ternary_matmul_v2.c` -- v2 LUT decode (LUT + AVX2, 13.2ms)
- `softchip/ternary_matmul_v3.c` -- v3 production (smart threading, 1.6ms M=1, + backward kernel)
- `softchip/torch_ternary.py` -- PyTorch integration (patch_model/unpatch_model, ternary forward+backward, FP32 LM head)
- `test_softchip_accuracy.py` -- numerical validation test
- `test_backward.py` -- backward kernel validation and benchmark
- `test_lm_head_fp32.py` -- FP32 LM head validation and full stack benchmark
- `profile_backward.py` -- backward pass profiling (LMM Pass 5 RAW data collection)
- `bench_softchip_model.py` -- full model benchmark

```bash
# Compile kernel
gcc -O3 -mavx2 -mfma -march=native -fopenmp -shared -fPIC \
    -o softchip/ternary_matmul_v3.so softchip/ternary_matmul_v3.c -lm

# Standalone benchmark
gcc -O3 -mavx2 -mfma -march=native -fopenmp -DSTANDALONE_TEST \
    -o softchip/ternary_bench_v3 softchip/ternary_matmul_v3.c -lm
./softchip/ternary_bench_v3

# Numerical validation (requires model)
python test_softchip_accuracy.py

# Full model benchmark
python bench_softchip_model.py
```

## Vulkan iGPU Soft-Chip: Vega 7 Compute

### Discovery

The Ryzen 5 PRO 5675U has a Vega 7 iGPU (7 CUs, 448 shaders, 1800 MHz, GCN 5 / GFX9) with 1.61 TFLOPS peak FP32 -- 7.6x more raw compute than the CPU's AVX2 path. We built a Vulkan compute shader proof-of-concept to exploit this.

### iGPU Specs

| Spec | Value |
|------|-------|
| Architecture | GCN 5 (Vega), GFX9 |
| Compute Units | 7 |
| Shaders | 448 |
| Max clock | 1800 MHz |
| FP32 peak | 1.61 TFLOPS |
| LDS (shared memory) | 64 KB per CU |
| VRAM (dedicated) | 512 MB (carved from system RAM) |
| Memory | Shared DDR4-3200 dual-channel (~51 GB/s, shared with CPU) |

### Key Feasibility Insight

All 210 layers' packed weights total **497 MB** -- fits in the 512 MB dedicated VRAM by 15 MB. The activation vector (10-27 KB) fits in LDS. This means weights can be loaded into VRAM once at init, and the hot path reads weights from VRAM through L2 cache with activations broadcast from LDS. No round-trips to shared system RAM in steady state.

### Shader Design

#### v2 (proof-of-concept)

Vulkan GLSL compute shader with branchless ternary decode:
- 2-bit weight codes: bit0 = nonzero flag, bit1 = sign flag
- `contribution = activation * float(code & 1) * (1.0 - 2.0 * float(code >> 1))`
- No divergent branches in the wavefront (critical for GPU efficiency)
- Activation vector loaded into shared memory (LDS) via collaborative workgroup load
- RADV (Mesa Vulkan) driver on Linux, no ROCm required

#### v3 (optimized)

LMM Pass 4 analysis identified the GPU as **memory-bound** (1.3% of peak FP32). A/B testing of 4 shader variants revealed:

1. **Transposed weight layout (REJECTED):** Cross-thread coalescing improved, but strided sequential reads within each thread destroyed L2 prefetch. Net: slower than row-major for scalar loop, marginal with vec4.
2. **XOR+AND bit trick (ADOPTED):** Eliminates float multiply from inner loop. `uintBitsToFloat((act ^ sign_mask) & nz_mask)` — all single-cycle VALU ops. Significant win.
3. **Fully unrolled vec4 processing (ADOPTED):** Process all 16 weights per uint32 in 4 groups of 4, eliminating the inner loop entirely. 4 independent accumulators enable instruction-level parallelism.
4. **LDS right-sizing via specialization constants (ADOPTED):** Two pipeline variants (2560: 10 KB LDS → 60% occupancy; 6912: 27 KB LDS → 20% occupancy).

The winning shader (v3) keeps v2's **row-major weight layout** but uses the bit trick, full unrolling, and LDS sizing. The key insight: on this GPU, reducing instruction count and improving instruction-level parallelism matters more than memory coalescing.

#### A/B Testing Methodology

All 4 variants were tested on the same hardware, same data, same Vulkan session. Each variant was benchmarked with both individual submits (includes Vulkan overhead) and batched submits (7 dispatches per command buffer, amortizes overhead). Outputs were cross-validated element-by-element against v2 (NRMSE < 1e-6 for all variants).

| Variant | Layout | Optimizations | Batched (ms) | vs v2 |
|---------|--------|--------------|-------------|-------|
| v2 (baseline) | row-major | branchless float mul | 0.80 | 1.0x |
| transposed+vec4 | transposed | bit trick + vec4 (4 per iter) | 0.49 | 1.6x |
| transposed+scalar | transposed | bit trick only | 0.96 | 0.8x |
| **v3 (winner)** | **row-major** | **bit trick + full unroll (16 per uint)** | **0.30** | **2.0x** |

The transposed layout's failure was instructive: it improved cross-thread coalescing but introduced a stride of `num_groups × 4 = 640 bytes` between sequential reads within each thread. On a 7-CU Vega with ~1 MB L2 cache, the L2 prefetcher's sequential access pattern (row-major) was more valuable than cross-thread coalescing. This contradicted the LMM REFLECT prediction but was confirmed by repeatable A/B results.

### Benchmark Results

#### v2 (proof-of-concept, 2560×2560 only)

| Method | Time | Throughput | vs CPU |
|--------|------|-----------|--------|
| CPU soft-chip v3 (AVX2) | 1.56 ms | 8.4 GFLOP/s | 1x |
| GPU v2 individual submit | 1.04 ms | 12.7 GFLOP/s | 1.5x |
| GPU v2 batched (7 dispatch) | 0.61 ms | 21.6 GFLOP/s | 2.6x |

#### v3 (optimized, all layer shapes)

| Layer | Shape | Batched (ms) | GFLOP/s eq | vs v2 |
|-------|-------|-------------|-----------|-------|
| **q_proj / o_proj** | 2560×2560 | **0.30** | **43.8** | **2.0x** |
| k_proj / v_proj | 640×2560 | 0.12 | 27.3 | — |
| gate_proj / up_proj | 6912×2560 | 0.88 | 40.0 | — |
| down_proj | 2560×6912 | 1.44 | 24.6 | — |

All 4 layer shapes pass numerical validation (NRMSE < 2e-6 vs CPU reference).

#### Projected Full Model Forward (M=1)

Per decoder layer kernel time: q(0.30) + k(0.12) + v(0.12) + o(0.30) + gate(0.88) + up(0.88) + down(1.44) = **4.05 ms**

| Metric | CPU soft-chip | GPU v2 (projected) | GPU v3 (projected) |
|--------|--------------|-------------------|-------------------|
| 30-layer kernel time | 910 ms | ~128 ms | **~121 ms** |
| + submit overhead (120 submits) | — | ~52 ms | ~36 ms |
| **Total forward (M=1)** | **910 ms** | **~180 ms** | **~157 ms** |
| **200-token rollout** | **182 s (3 min)** | **~36 s** | **~31 s** |
| vs stock PyTorch (4200 ms) | 4.6x | ~23x | **~27x** |

### Vulkan Backend Library and PyTorch Integration

The optimized Vulkan shader was wrapped in a shared library (`vk_backend.c`) exposing a clean C API, then integrated into the PyTorch `patch_model()` system as a new `backend="vulkan"` option.

#### C Backend API

```c
int  vk_init(const char *spv_path);           // Initialize Vulkan + load shader
int  vk_alloc_layer(uint32_t *packed, int N, int K, float scale);  // Upload packed weights
int  vk_dispatch(int layer_id, float *act, float *out);            // Single dispatch
int  vk_dispatch_batch(int *ids, int n, float *act, float **outs); // Batched dispatch
void vk_shutdown(void);                        // Cleanup
```

Key design: `vk_dispatch_batch()` records multiple dispatches into a single command buffer, eliminating per-dispatch submit overhead. Tested with q+k+v (3 dispatches): **1.49x speedup** over 3 individual submits.

#### Backend Validation (`test_vk_backend.py`)

| Layer Shape | CPU vs VK NRMSE | VK vs VK | Vulkan Dispatch (ms) |
|-------------|----------------|----------|---------------------|
| q_proj/o_proj (2560×2560) | 4.11e-03 | bit-exact | 0.677 |
| k_proj/v_proj (640×2560) | 4.03e-03 | bit-exact | 0.433 |
| gate_proj/up_proj (6912×2560) | 3.89e-03 | bit-exact | 0.888 |
| down_proj (2560×6912) | 3.91e-03 | bit-exact | 1.400 |

The ~4e-3 NRMSE between CPU and Vulkan is expected: the CPU kernel applies INT8 activation quantization (`ActQuant`) while the Vulkan shader operates on raw FP32 activations. Both paths are numerically valid — the GPU matches its own output exactly (bit-exact determinism).

Batch dispatch benchmark:
- **3 dispatches in 1 submit:** 1.084 ms (0.361 ms/dispatch)
- **3 individual submits:** 1.613 ms (0.538 ms/dispatch)
- **Batch speedup: 1.49x**

#### End-to-End Model Validation (`test_vk_model.py`)

Full BitNet b1.58 2B4T model loaded and run through all three backends on the same 6-token prompt ("The capital of France is"):

| Backend | Forward Time | Top-1 Prediction |
|---------|-------------|-----------------|
| Stock PyTorch | 8443 ms | " Paris" |
| CPU soft-chip | 1762 ms | " Paris" |
| Vulkan soft-chip | 1801 ms | " Paris" |

All backends produce identical top-5 predictions: [" Paris", " not", " a", " the", " called"].

| Comparison | NRMSE |
|-----------|-------|
| CPU vs stock PyTorch | 2.97e-02 |
| Vulkan vs stock PyTorch | 5.11e-02 |
| Vulkan vs CPU | 3.09e-02 |
| **Vulkan vs Vulkan** | **0.00e+00 (bit-exact)** |

Full-model NRMSE is higher than per-layer (~4e-3) because quantization differences accumulate through 30 decoder layers (210 matmuls). The key result: all backends agree on functional output (same top-k predictions) and Vulkan is perfectly deterministic.

#### Full Model Forward and Generation Benchmark

Detailed profiling revealed that **Vulkan submit/fence overhead dominates**, making the iGPU path slower than the CPU soft-chip for end-to-end inference. Each of the 210 dispatches pays ~1ms submit overhead regardless of shader compute time.

| Metric | Stock PyTorch | CPU Soft-Chip | Vulkan Soft-Chip |
|--------|--------------|--------------|-----------------|
| **Forward (6 tokens)** | **11568 ms** | **1328 ms (8.7x)** | **2654 ms (4.4x)** |
| Matmul time | — | 1181 ms | 2577 ms |
| Non-matmul time | — | 147 ms | 77 ms |
| **Per-token (M=1)** | — | **579 ms** | **628 ms** |
| **200-token estimate** | — | **116s** | **126s** |
| Tokens/sec | — | **1.7** | **1.6** |

The CPU soft-chip **wins** because:
1. **Zero submit overhead**: CPU kernel is a direct function call with memory access. Vulkan requires command buffer record → queue submit → fence wait → readback per dispatch, costing ~1ms each.
2. **210 submits × ~1ms = ~210ms** of pure overhead, exceeding the total CPU kernel time for many layer shapes.
3. **Batching helps but can't close the gap**: Even batching q+k+v (3 dispatches → 1 submit) only saves ~0.5ms per decoder block (15ms total for 30 blocks). The remaining 4 serial dispatches per block (o, gate, up, down) still pay individual submit costs.

**Key insight**: The ~157ms projection was based on raw shader times without submit overhead. The actual per-dispatch cost is shader_time + ~1ms_submit. For small layers (k/v at 640×2560, ~0.12ms shader), the overhead is 8x the compute. The Vulkan iGPU path would only win with:
- A fully GPU-resident execution graph (all ops on GPU, no host round-trips)
- Timeline semaphores with async readback (eliminate fence wait per dispatch)
- A GPU with much faster submit path (Jetson Thor's CUDA path is ~10μs submit)

**Decision**: Use CPU soft-chip as the production inference path for the Ryzen development machine. The Vulkan shader work validates the ternary matmul optimization approach and will be directly applicable on the Jetson Thor where CUDA's lower submit overhead and unified memory eliminate the host-device copy bottleneck.

### Source

- `softchip/ternary_matmul.comp` -- v2 Vulkan GLSL shader (proof-of-concept, kept for reference)
- `softchip/ternary_matmul_v3.comp` -- v3 optimized shader (production)
- `softchip/vk_ternary.c` -- v2 dispatch harness + benchmark
- `softchip/vk_ternary_v3.c` -- v3 dispatch harness + multi-shape benchmark
- `softchip/vk_backend.c` -- Vulkan backend shared library (init/alloc/dispatch/batch/shutdown)
- `softchip/torch_ternary.py` -- PyTorch integration (now supports `backend="vulkan"|"cpu"|"auto"`)
- `softchip/test_vk_backend.py` -- Vulkan backend validation test (per-layer + batch)
- `test_vk_model.py` -- End-to-end model validation (all 3 backends)
- `bench_vk_model.py` -- Full model benchmark with generation (forward + autoregressive)
- `bench_vk_dispatch_overhead.py` -- Dispatch overhead profiler (breakdown by component)

```bash
# Compile v3 shader
glslangValidator -V softchip/ternary_matmul_v3.comp -o softchip/ternary_matmul_v3.spv

# Compile Vulkan backend library
gcc -O2 -shared -fPIC -o softchip/libvk_ternary.so softchip/vk_backend.c -lvulkan -lm -ldl

# Run per-layer Vulkan validation
VK_ICD_FILENAMES=/usr/share/vulkan/icd.d/radeon_icd.x86_64.json python softchip/test_vk_backend.py

# Run end-to-end model validation (requires model in models/)
VK_ICD_FILENAMES=/usr/share/vulkan/icd.d/radeon_icd.x86_64.json python test_vk_model.py

# Compile and run standalone v3 benchmark
gcc -O2 -o softchip/vk_ternary_v3 softchip/vk_ternary_v3.c -lvulkan -lm -ldl
VK_ICD_FILENAMES=/usr/share/vulkan/icd.d/radeon_icd.x86_64.json ./softchip/vk_ternary_v3
```

## Next Steps

### Phase 1: GRPO Training Loop (NEXT)
- With 2.4s per training iteration, CPU-only GRPO training is now practical
- Implement GRPO reward computation on GSM8K (verifiable correct/incorrect answers)
- BitNet b1.58 baseline: 58.38% GSM8K accuracy
- Use soft-chip for fast rollout generation, full optimized stack for gradient computation
- Run first TinyLoRA + GRPO training cycle, measure improvement

### Phase 2: Thor Deployment
- Port validation code + soft-chip to Jetson AGX Thor (128GB unified memory, Blackwell GPU)
- Verify GPU-accelerated forward/backward pass (~1-5s expected)
- BF16 GEMM will be fast natively on Blackwell (FP32 LM head patch not needed)
- Set up bitnet.cpp for fast rollout generation (~35 tok/s on CPU, faster on GPU)

### Phase 3: Autoresearch Loop
- Build autonomous experiment loop (propose adapter config, train, evaluate, keep/discard)
- Search space: layer selection, number of adapted layers, learning rate, GRPO hyperparams
- Target: overnight runs producing 30+ experiments

### Phase 4: Novel Experiments
- Adapt all layer types (attention projections, MLP gates, all at once)
- Explore higher-rank TinyLoRA (4, 16, 64 params per layer)
- Compare RL vs SFT parameter efficiency on ternary models
- Benchmark against full LoRA (rank 8, 16) to measure efficiency frontier

## Reproduction

```bash
# Install dependencies
pip install torch transformers peft trl accelerate datasets

# Download model
huggingface-cli download microsoft/bitnet-b1.58-2B-4T-bf16 --local-dir models/bitnet-b1.58-2B-4T-bf16

# Note: remove auto_map from config.json if using transformers >= 5.0
# (BitNet is natively supported, custom code files not needed)

# Run TinyLoRA validation
python test_bitnet_tinylora.py

# Build and benchmark soft-chip CPU kernel (requires AVX2)
gcc -O3 -mavx2 -mfma -march=native -fopenmp -shared -fPIC \
    -o softchip/ternary_matmul_v3.so softchip/ternary_matmul_v3.c -lm
python test_softchip_accuracy.py    # Numerical validation
python test_backward.py             # Backward kernel validation
python test_lm_head_fp32.py         # FP32 LM head validation + full stack benchmark
python profile_backward.py          # Backward pass profiling
python bench_softchip_model.py      # Full model benchmark

# Build and benchmark soft-chip GPU kernel (requires Vulkan + AMD iGPU)
sudo apt-get install libvulkan-dev glslang-tools mesa-vulkan-drivers
glslangValidator -V softchip/ternary_matmul_v3.comp -o softchip/ternary_matmul_v3.spv
gcc -O2 -shared -fPIC -o softchip/libvk_ternary.so softchip/vk_backend.c -lvulkan -lm -ldl
VK_ICD_FILENAMES=/usr/share/vulkan/icd.d/radeon_icd.x86_64.json python softchip/test_vk_backend.py
VK_ICD_FILENAMES=/usr/share/vulkan/icd.d/radeon_icd.x86_64.json python test_vk_model.py
VK_ICD_FILENAMES=/usr/share/vulkan/icd.d/radeon_icd.x86_64.json python bench_vk_model.py
VK_ICD_FILENAMES=/usr/share/vulkan/icd.d/radeon_icd.x86_64.json python bench_vk_dispatch_overhead.py
```

## LMM Analysis

The Lincoln Manifold Method was used to analyze this problem through six complete passes:

### Pass 1: LFM2-24B-A2B + TinyLoRA on Jetson AGX Thor
- `journal/lfm2_tinylora_raw.md` through `journal/lfm2_tinylora_synth.md`
- Concluded: viable but complex (hybrid conv+MoE architecture, 48GB BF16, uncertain LoRA compatibility on conv layers)

### Pass 2: BitNet b1.58 + TinyLoRA on CPU (local validation)
- `journal/bitnet_tinylora_raw.md` through `journal/bitnet_tinylora_synth.md`
- Concluded: simpler, smaller, standard transformer architecture, MIT license, natively CPU-efficient
- Led directly to the successful validation documented here

The pivot from LFM2-24B to BitNet b1.58 was driven by the REFLECT phase identifying that BitNet's standard transformer architecture eliminates the LoRA compatibility uncertainty that plagued the LFM2 approach, while its 0.4GB footprint makes the Thor's 128GB memory absurdly generous rather than barely sufficient.

### Pass 3: Soft-Chip Optimization Opportunities
- `journal/softchip_opt_raw.md` through `journal/softchip_opt_synth.md`
- Identified: at ~9% of peak FP32, bottleneck is instruction throughput not memory bandwidth
- Key finding: M=1 (autoregressive) is the critical use case; OpenMP thread overhead makes parallel worse than serial for small batch
- Decision: smart threading (serial for M<6, parallel for M>=6) + numerical validation gate before PyTorch integration
- Led directly to v3 kernel (4.3x M=1 improvement) and validated PyTorch integration (4.6x full-model speedup)

### Pass 4: iGPU Shader Optimization
- `journal/igpu_opt_raw.md` through `journal/igpu_opt_synth.md`
- Analysis predicted: GPU is memory-bound (1.3% of peak), transposed layout would be the #1 optimization
- **A/B testing overturned this:** transposed layout hurt sequential L2 prefetch more than it helped cross-thread coalescing. Row-major + bit trick + full unrolling was 2x faster.
- Winning optimizations: XOR+AND bit trick (no float multiply), fully unrolled 16-weight decode, specialization constants for LDS sizing
- Result: 2560×2560 batched 0.30 ms (2.0x over v2), full model projected ~157 ms (5.8x over CPU)
- **Projection vs reality:** The ~157ms projection assumed raw shader times. Actual end-to-end Vulkan forward was 2654ms (210 dispatches × ~1ms submit overhead each). CPU soft-chip (1328ms) is faster than Vulkan for this hardware due to zero-overhead function calls vs GPU submit/fence cycles.
- Key lesson #1: on this GPU, instruction-level parallelism and L2 sequential prefetch matter more than cross-thread memory coalescing for our access pattern
- Key lesson #2: Vulkan submit/fence overhead (~1ms/dispatch on RADV/Vega) makes iGPU compute uncompetitive for many small dispatches. The optimization applies when ported to CUDA where submit overhead is ~10μs.

### Pass 5: Backward Pass Optimization — FP32 LM Head
- `journal/backward_opt_raw.md` through `journal/backward_opt_synth.md`
- Profiling revealed the remaining 19.5s backward was 92.6% from TWO matmul calls in the LM head — an `nn.Linear(2560, 128256)` that was NOT patched by the ternary backward kernel because it's NOT ternary (standard dense BF16)
- Root cause: MKL's BF16 GEMM on Zen 3 (no AMX/VNNI) is **32-90x slower** than FP32 for the same matmul
- Fix: `FP32LMHeadFunction` — custom autograd function that casts BF16↔FP32 at boundaries, does all computation in FP32
- The LM head is weight-tied with the embedding layer; gradients MUST flow through it to reach TinyLoRA adapters in the decoder
- Result: backward 19,500ms → **1,065ms** (**18.3x speedup**), total iteration 20,700ms → **2,410ms** (**8.6x speedup**)
- Memory cost: +657 MB for FP32 copy (trivial with 64 GB RAM)
- Cumulative speedup from stock PyTorch: **38x** (91,700ms → 2,410ms)

### Pass 6: Parallel Backward — Negative Result
- `journal/backward_parallel_raw.md` through `journal/backward_parallel_synth.md`
- Attempted OpenMP parallelization over N (out_features) within `ternary_matmul_backward()` using per-thread static buffers + AVX2 reduction
- **Kernel-level result:** 3.4x speedup in isolation (548ms → 162ms, all 4 shapes validated)
- **End-to-end result:** 9% regression (1,065ms → 1,160ms). Reverted to serial.
- Root cause: OpenMP/MKL thread pool contention. Rapid alternation between our OpenMP threads and PyTorch's MKL threads (both claiming 12 threads on 6 cores) caused cache thrashing and scheduling overhead that negated the kernel speedup.
- Tested 5 configurations (thread counts 3/6/12, OMP_WAIT_POLICY, higher thresholds) — none beat serial end-to-end
- Same class of finding as Pass 3 (serial beats threaded for M<6 forward): on this CPU, micro-benchmark parallelism doesn't transfer when multiple thread pools compete in rapid alternation
- **Decision:** Serial backward is the production path. The backward is at its practical ceiling on this hardware within PyTorch autograd.
