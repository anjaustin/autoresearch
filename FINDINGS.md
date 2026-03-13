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

## Next Steps

### Phase 1: Thor Deployment
- Port validation code to Jetson AGX Thor (128GB unified memory, Blackwell GPU)
- Verify GPU-accelerated forward/backward pass (~1-5s expected)
- Set up bitnet.cpp for fast rollout generation (~35 tok/s on CPU, faster on GPU)

### Phase 2: GRPO Training Loop
- Implement GRPO reward computation on GSM8K (verifiable correct/incorrect answers)
- BitNet b1.58 baseline: 58.38% GSM8K accuracy
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

# Run validation
python test_bitnet_tinylora.py
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
