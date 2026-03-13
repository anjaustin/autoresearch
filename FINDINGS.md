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

### The 208s Backward Pass

The backward pass through 2.41B parameters on a Ryzen 5 CPU takes 208 seconds. This confirms that CPU-only GRPO training (which needs many forward+backward passes per step) is impractical for iterative experiments. However, on a Jetson AGX Thor with Blackwell GPU, this would be ~1-5 seconds.

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
from softchip.torch_ternary import patch_model, unpatch_model

model = AutoModelForCausalLM.from_pretrained(...)
patch_model(model)       # Replaces AutoBitLinear.forward with C kernel
output = model(input)    # 4.6x faster forward pass
unpatch_model(model)     # Restore original
```

Weight packing (one-time cost at load): ~48s for 210 layers. The backward pass is unchanged -- STE requires BF16 master weights, so backward falls through to PyTorch.

### Source

- `softchip/ternary_matmul.c` -- v1 prototype (scalar decode, 221ms)
- `softchip/ternary_matmul_v2.c` -- v2 LUT decode (LUT + AVX2, 13.2ms)
- `softchip/ternary_matmul_v3.c` -- v3 production (smart threading, 1.6ms M=1)
- `softchip/torch_ternary.py` -- PyTorch integration (patch_model/unpatch_model)
- `test_softchip_accuracy.py` -- numerical validation test
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

## Next Steps

### Phase 1: Thor Deployment (NEXT)
- Port validation code + soft-chip to Jetson AGX Thor (128GB unified memory, Blackwell GPU)
- Verify GPU-accelerated forward/backward pass (~1-5s expected)
- Set up bitnet.cpp for fast rollout generation (~35 tok/s on CPU, faster on GPU)
- The soft-chip CPU kernel may also be useful on Thor's 14-core ARM Neoverse V3AE cores

### Phase 2: GRPO Training Loop
- Implement GRPO reward computation on GSM8K (verifiable correct/incorrect answers)
- BitNet b1.58 baseline: 58.38% GSM8K accuracy
- Use soft-chip for fast rollout generation, PyTorch for gradient computation
- Run first TinyLoRA + GRPO training cycle, measure improvement

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

# Build and benchmark soft-chip kernel (requires AVX2)
gcc -O3 -mavx2 -mfma -march=native -fopenmp -shared -fPIC \
    -o softchip/ternary_matmul_v3.so softchip/ternary_matmul_v3.c -lm
python test_softchip_accuracy.py    # Numerical validation
python bench_softchip_model.py      # Full model benchmark
```

## LMM Analysis

The Lincoln Manifold Method was used to analyze this problem through two complete passes:

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
