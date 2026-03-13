# Raw Thoughts: LFM2-24B-A2B + TinyLoRA on Jetson AGX Thor

## Stream of Consciousness

We want to take the autoresearch concept -- an autonomous AI agent that iteratively experiments on a model to improve it -- and apply it to LFM2-24B-A2B using TinyLoRA on a Jetson AGX Thor with 128GB unified memory.

My first instinct is that this is exciting but the pieces don't fit together trivially. Autoresearch trains from scratch in 5-minute cycles. We can't train a 24B model from scratch on any single device in 5 minutes. But TinyLoRA changes the equation radically: we're talking about 13 to ~1000 trainable parameters. The forward/backward pass through the frozen model is the bottleneck, not optimizer states.

The Thor has unified memory -- no PCIe bottleneck between CPU and GPU. The full BF16 model is ~48GB, which fits with room to spare in 128GB. But what's the Thor's actual compute throughput? It's a Blackwell-architecture GPU, but it's an edge device, not a data center card. Training throughput will be much lower than H100. How much lower? This matters for whether a 5-minute experiment window produces enough signal.

TinyLoRA uses RL, not SFT. That's a key constraint. RL needs a reward signal. What's our reward signal? Autoresearch uses val_bpb (validation bits per byte) -- a language modeling metric. But TinyLoRA was demonstrated on math reasoning tasks (GSM8K, AIME) where you have verifiable correct/incorrect answers. RL for language modeling is harder -- what's the reward? Perplexity improvement? That's basically SFT territory (minimize loss). The TinyLoRA paper explicitly says SFT needs 100-1000x more parameters. So if our task is "improve language modeling," TinyLoRA's extreme parameter efficiency may not apply.

This is a real tension. The autoresearch loop evaluates val_bpb. TinyLoRA works best with RL on tasks with verifiable rewards. These are different optimization targets.

Maybe the right move is to change the evaluation target. Instead of val_bpb, use a reasoning benchmark that has verifiable answers. The agent proposes TinyLoRA adapter configurations, trains with GRPO or similar RL, evaluates on a held-out reasoning set. But then we're not really doing "autoresearch" in Karpathy's sense -- we're doing automated RL adapter search.

Or maybe the right framing is: the autoresearch *pattern* (autonomous loop, propose change, evaluate, keep/discard) applied to TinyLoRA adapter search on LFM2-24B-A2B. The specific evaluation metric changes, but the loop structure stays.

What about the LFM2 architecture specifically? It's a hybrid conv+attention model with MoE. Which layers accept LoRA adapters? Standard LoRA targets attention QKV and MLP projections. LFM2 has 30 conv layers and 10 attention layers. The conv layers are gated short convolutions -- do they have weight matrices that LoRA can target? TinyLoRA reparameterizes below rank-1, so the layer choice becomes even more important -- you might be putting your 13 parameters in the wrong place entirely.

The MoE aspect adds another dimension. With 64 experts and top-4 routing, LoRA on the expert MLPs would be sparse -- only 4 experts are active per token. Does that help or hurt TinyLoRA? The routing itself could be a target -- tiny adjustments to the router could redirect token flow.

LFM2-24B-A2B is already post-trained as an instruct model. Liquid AI says it's not recommended for coding. So what capability gap are we trying to fill? Math reasoning? Domain-specific knowledge? Tool use improvement? The choice of task determines whether TinyLoRA's RL approach even applies.

Practical concerns: the Thor runs JetPack/L4T (Linux for Tegra). PyTorch works on Jetson, but the transformers library support for LFM2 requires transformers>=5.0.0. Is that available on JetPack? CUDA support should be native. Flash Attention on Jetson? The model card mentions flash_attention_2 as an option.

The autoresearch repo uses prepare.py with a specific tokenizer (vocab 8192) and dataset. LFM2 uses vocab 65536 and was trained on different data. We'd need to replace the entire data pipeline, or just use HuggingFace datasets directly for the RL reward tasks.

What scares me: the compute budget on Thor might make even a single RL training run take too long for the 5-minute cadence. If one forward+backward pass through 24B params takes seconds, and RL needs hundreds of rollouts... the math might not work for short experiment cycles.

What would be the naive approach? Load the model, slap LoRA on the attention layers, run SFT on some dataset, evaluate loss. That would work but would miss the TinyLoRA insight (RL, extreme parameter reduction) and wouldn't leverage the autoresearch loop pattern well.

## Questions Arising
- What is the Thor's actual BF16 TFLOPS for training workloads?
- Can TinyLoRA's extreme efficiency (13 params) work with RL on non-math tasks?
- Which LFM2 layers are amenable to LoRA injection (conv vs attention vs MoE router)?
- Is 5 minutes enough time for an RL training cycle on Thor with a 24B model?
- What's the right reward signal if not val_bpb?
- Does the MoE routing interact well or badly with tiny adapter perturbations?
- What's the software stack compatibility story for LFM2 on Jetson?

## First Instincts
- Change the evaluation target from val_bpb to a verifiable reasoning task
- Keep the autoresearch loop structure but replace train.py with an RL+TinyLoRA script
- Target the 10 attention layers for LoRA injection as the safe starting point
- Consider quantizing the base model (Q8 or Q4) to speed up forward passes
- Extend the time budget beyond 5 minutes if needed -- maybe 15-30 min per experiment
- Start with the LFM2-8B-A1B (smaller) to validate the approach before scaling to 24B
