# Synthesis: Autonomous TinyLoRA Adapter Search for LFM2 on Jetson AGX Thor

## Project Name
**TinyMoE-Search** -- Autonomous discovery of minimal LoRA interventions on Liquid Foundation Models via reinforcement learning on edge hardware.

---

## Architecture

```
┌──────────────────────────────────────────────────────────────┐
│                    TINYMOE-SEARCH SYSTEM                      │
├──────────────────────────────────────────────────────────────┤
│                                                               │
│  ┌─────────────────────────────────────────────────────────┐ │
│  │ AGENT LOOP (program.md equivalent)                       │ │
│  │                                                           │ │
│  │  1. Sample adapter config from search space              │ │
│  │  2. git commit config                                     │ │
│  │  3. Run experiment.py (GRPO training + eval)             │ │
│  │  4. Extract GSM8K accuracy from log                       │ │
│  │  5. If improved: keep commit. Else: git reset            │ │
│  │  6. Log to results.tsv                                    │ │
│  │  7. Repeat                                                │ │
│  └─────────────────────────────────────────────────────────┘ │
│                          │                                    │
│                          ▼                                    │
│  ┌─────────────────────────────────────────────────────────┐ │
│  │ experiment.py                                             │ │
│  │                                                           │ │
│  │  ┌─────────────┐    ┌──────────────┐   ┌─────────────┐ │ │
│  │  │ Frozen LFM2  │───▶│ TinyLoRA     │──▶│ GRPO        │ │ │
│  │  │ (Q4/INT8)    │    │ Adapters     │   │ Training    │ │ │
│  │  │ 8B or 24B    │    │ (13-10K      │   │ (GSM8K      │ │ │
│  │  │              │    │  params)     │   │  rollouts)  │ │ │
│  │  └─────────────┘    └──────────────┘   └──────┬──────┘ │ │
│  │                                                │        │ │
│  │                                         ┌──────▼──────┐ │ │
│  │                                         │ Evaluate    │ │ │
│  │                                         │ GSM8K acc   │ │ │
│  │                                         └─────────────┘ │ │
│  └─────────────────────────────────────────────────────────┘ │
│                                                               │
│  Hardware: Jetson AGX Thor, 128GB unified memory              │
└──────────────────────────────────────────────────────────────┘
```

---

## Key Decisions

### 1. Start with LFM2-8B-A1B, graduate to 24B
Because validation speed matters more than prestige in the proof-of-concept phase. Same architecture, 3x faster experiments. Move to 24B once the loop is proven.

### 2. GSM8K as primary benchmark
Because it provides clean verifiable RL rewards, TinyLoRA has published baselines on it, and LFM2 has reported scores -- giving us three-way comparison data.

### 3. GRPO as the RL algorithm
Because it's the standard for TinyLoRA-style training (used in the paper), works well with small parameter budgets, and has mature implementations (trl, OpenRLHF).

### 4. Adapter search space includes attention layers, MoE router, AND conv layers
Because the most interesting result would be discovering that MoE router perturbation with ~100 parameters outperforms traditional attention LoRA. Let the autonomous loop discover this.

### 5. Quantize frozen base with bitsandbytes NF4
Because it cuts memory from ~48GB (24B BF16) to ~12GB (24B NF4), freeing 116GB for activations and rollout batches. Use BF16 for the tiny adapters.

### 6. 15-minute experiment cycles, not 5
Because RL needs enough rollouts for signal, and Thor throughput is lower than H100. 15 min yields ~32 experiments per 8-hour overnight run. The original autoresearch ran 83 total with many failures.

### 7. Clean implementation, not a fork of autoresearch
Because the data pipeline, model architecture, training loop, and evaluation metric are all different. Only the loop structure and git tracking transfer.

---

## Implementation Spec

### Phase 0: Environment Setup (Day 1)

```bash
# On Jetson AGX Thor
pip install torch  # JetPack-compatible build
pip install transformers>=5.0.0 bitsandbytes peft trl datasets accelerate
```

Validate:
```python
from transformers import AutoModelForCausalLM, AutoTokenizer
model = AutoModelForCausalLM.from_pretrained(
    "LiquidAI/LFM2-8B-A1B",
    device_map="auto",
    load_in_4bit=True,
)
tokenizer = AutoTokenizer.from_pretrained("LiquidAI/LFM2-8B-A1B")
# Generate a simple completion to verify the model works
```

Success criterion: model loads and generates text on Thor.

### Phase 1: Manual TinyLoRA + GRPO Experiment (Days 2-3)

Build `experiment.py`:
1. Load frozen quantized LFM2-8B-A1B
2. Apply TinyLoRA adapter to attention Q projections (safe default)
3. Load GSM8K dataset (train split for RL, test split for eval)
4. Run GRPO training for 15 minutes wall-clock
5. Evaluate accuracy on GSM8K test (or a held-out subset)
6. Report: adapter config, param count, GSM8K accuracy, wall time

TinyLoRA implementation:
```python
# Reparameterize LoRA below rank-1 following the paper:
# Instead of W + BA (rank-r), use W + s * u @ v^T
# where u, v are fixed random vectors and s is a learned scalar
# This gives 1 trainable parameter per adapted layer
```

Search space config (JSON):
```json
{
    "target_layers": ["attn_q", "attn_k", "attn_v", "attn_o",
                      "moe_router", "conv_proj"],
    "num_adapted_layers": [1, 2, 4, 8],
    "param_per_layer": [1, 4, 16, 64, 256],
    "learning_rate": [1e-3, 3e-3, 1e-2, 3e-2],
    "grpo_rollouts_per_step": [4, 8, 16],
    "grpo_steps": [50, 100, 200]
}
```

### Phase 2: Automated Loop (Days 4-5)

Build `loop.py`:
```
while True:
    config = sample_config(search_space, strategy="explore")
    git_commit(config)
    result = run_experiment(config, time_budget=900)  # 15 min
    if result.accuracy > best_accuracy:
        best_accuracy = result.accuracy
        log_result(config, result, status="KEPT")
    else:
        git_reset()
        log_result(config, result, status="DISCARDED")
```

Sampling strategy starts random, then shifts toward configurations similar to kept experiments (Thompson sampling or UCB over the config space).

### Phase 3: Scale to 24B (Days 6-7)

Once the loop produces improvements on LFM2-8B-A1B:
1. Switch model_id to "LiquidAI/LFM2-24B-A2B"
2. Verify memory fits (NF4: ~12GB + adapters + activations)
3. Adjust time budget if needed (24B forward pass is slower)
4. Run overnight

### Phase 4: Analysis and Novel Experiments (Day 8+)

- Which layer types (conv vs attn vs router) produced the biggest gains?
- What's the minimum parameter count that improves GSM8K?
- Does MoE router adaptation work? (The novel finding)
- Can results from 8B transfer to 24B? (Same adapter config on bigger model)
- Secondary benchmark: IFEval for instruction following

---

## File Structure

```
tinymoe-search/
├── program.md              # Agent instructions (autoresearch pattern)
├── experiment.py           # Single experiment: load, adapt, train, eval
├── loop.py                 # Autonomous loop: sample, run, keep/discard
├── tinylora.py             # TinyLoRA implementation (sub-rank-1 LoRA)
├── evaluate.py             # GSM8K evaluation harness
├── search_space.json       # Adapter config search space definition
├── results.tsv             # Experiment log (gitignored)
├── journal/                # LMM analysis artifacts
│   ├── lfm2_tinylora_raw.md
│   ├── lfm2_tinylora_nodes.md
│   ├── lfm2_tinylora_reflect.md
│   └── lfm2_tinylora_synth.md
└── .gitignore
```

---

## Success Criteria

- [ ] LFM2-8B-A1B loads and runs inference on Jetson AGX Thor
- [ ] TinyLoRA adapter (<=1000 params) successfully trains with GRPO on GSM8K
- [ ] Single experiment completes within 15-minute time budget
- [ ] Autonomous loop runs for 8+ hours without crashing
- [ ] At least one adapter config improves GSM8K accuracy over base model
- [ ] Results logged to results.tsv with full reproducibility info
- [ ] MoE router adaptation tested as a target (novel contribution)
- [ ] Findings transfer from 8B to 24B model (same adapter config works)

## Stretch Goals

- [ ] Discover that MoE router TinyLoRA (< 100 params) improves task performance
- [ ] Achieve comparable GSM8K improvement to the TinyLoRA paper's Qwen2.5 results
- [ ] Run secondary benchmark (IFEval) and show improvement there too
- [ ] Package as a reusable framework for TinyLoRA search on arbitrary HF models

---

## Risk Register

| Risk | Likelihood | Impact | Mitigation |
|------|-----------|--------|------------|
| Software stack incompatibility on Jetson | Medium | High | Validate in Phase 0 before anything else. Fall back to Docker container if needed. |
| Thor compute too slow for meaningful RL | Low-Medium | High | Quantization + 8B model first. Extend time budget. Reduce rollout count. |
| TinyLoRA doesn't transfer to hybrid conv+MoE arch | Medium | Medium | Start with proven attention targets. Conv/router are stretch experiments. |
| GRPO doesn't converge in 15 min with tiny params | Medium | Medium | Increase time budget or param count. The 1000x reduction vs SFT should still hold even if not at the 13-param extreme. |
| GSM8K eval is too slow on Thor | Low | Medium | Use a subset (200 problems) for fast loop eval, full set for final reporting. |

---

## The One-Sentence Pitch

An autonomous system that discovers, through reinforcement learning, which 13-1000 parameters in a frozen 24B hybrid MoE model need to change to unlock new capabilities -- running entirely on a single edge device.
