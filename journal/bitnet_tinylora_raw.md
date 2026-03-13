# Raw Thoughts: BitNet b1.58 + TinyLoRA -- CPU-First Local Validation

## Stream of Consciousness

The problem has shifted. We started with LFM2-24B on a Jetson Thor. Now we're looking at BitNet b1.58 2B on a Ryzen 5 laptop with 64GB RAM and no GPU. This is a completely different animal and I need to rethink from scratch.

The appeal of BitNet here is visceral: the model is 0.4GB in its native ternary form. Even the BF16 master weights (needed for training) are ~4GB. On a 64GB machine, memory is not even a discussion. The question is entirely about compute throughput on a CPU.

My first instinct: CPU-only GRPO training on a 2B model is going to be painfully slow. One forward pass through 2B parameters in BF16 on a Ryzen 5 is probably 1-3 seconds. Generation is autoregressive -- each token requires a full forward pass. A GSM8K problem might need 200 tokens of chain-of-thought generation. That's 200-600 seconds per completion. GRPO needs multiple rollouts per problem. Even 2 rollouts on 5 problems is 2000-6000 seconds -- 30 to 100 minutes just for the rollouts of a single training step.

This is the fundamental tension: TinyLoRA's magic is that it needs RL, and RL needs rollouts, and rollouts are inference, and inference on CPU for a 2B model is slow.

But wait. BitNet's whole value proposition IS fast CPU inference. The official bitnet.cpp gets 29ms/token on CPU. That's ~35 tokens/second. At that rate, 200 tokens of generation takes ~6 seconds per rollout. 2 rollouts x 5 problems = 60 seconds for the rollout phase. THAT changes the math completely.

The catch: bitnet.cpp is a C++ inference engine using the ternary weights. It's not PyTorch. We can't just "do backward pass" through it. TinyLoRA training requires PyTorch for gradient computation. So we have two different runtimes: bitnet.cpp for fast rollout generation (forward only), and PyTorch for gradient updates (forward + backward through the tiny adapter).

Actually, let me reconsider. The TinyLoRA paper's trick is that you're only training 13-1000 parameters. The backward pass through the frozen model doesn't need to be fast -- you only need gradients with respect to the tiny adapter. With PyTorch's autograd and a frozen model, the backward pass is proportional to the adapter, not the model. Wait, no -- you still need the full forward pass to compute the loss, and the backward pass propagates through the entire computational graph to get gradients at the adapter insertion point. The full backward pass through a 2B model on CPU is going to be slow regardless of adapter size.

So the two-runtime approach (bitnet.cpp for rollouts, PyTorch for gradients) might actually be necessary, not just nice-to-have. Use bitnet.cpp for the RL rollout phase (generate completions, compute rewards). Then for the actual parameter update: replay the chosen sequences through PyTorch, compute the loss, backprop through the tiny adapter. The PyTorch backward pass is slow, but you only need to do it once per GRPO step (on the selected sequences), not on every rollout.

How slow is the PyTorch path? The HuggingFace model card explicitly warns: "Please do NOT expect performance efficiency gains when using this model with the standard transformers library." So PyTorch inference of BitNet through transformers will be ~normal 2B model speed on CPU: maybe 1-3 seconds per forward pass for a full sequence. If we batch the selected sequences and do one forward+backward pass per GRPO step, that's tolerable.

Let me think about what "testing the hypothesis locally" actually means. We don't need to run a full overnight autoresearch loop. We need to prove:

1. BitNet b1.58 BF16 weights load in PyTorch on this machine
2. We can attach a TinyLoRA adapter and it doesn't break the model
3. The model can generate GSM8K-style completions
4. We can compute a reward (correct/incorrect answer extraction)
5. We can compute a gradient through the adapter and update it
6. After an update, the model's behavior changes (even if not necessarily improves)

That's a pipeline validation, not a training run. Steps 1-4 are straightforward. Steps 5-6 are the actual hypothesis test: does TinyLoRA's gradient flow work on BitNet's architecture?

There's a deeper question here about BitNet + LoRA compatibility. BitNet uses `BitLinear` layers where weights are ternary {-1, 0, +1}. The BF16 master weights are the *latent* continuous weights that get quantized during the forward pass via absmean quantization. LoRA adds a low-rank perturbation: output = (W_quantized + BA)x. But in BitNet, W is quantized from the master weights in the forward pass. Does LoRA get applied before or after quantization? If after (on the ternary weights), you're adding continuous corrections to ternary values -- that's fine mathematically. If before (on the master weights), the LoRA perturbation gets quantized away. This matters a lot.

Standard PEFT/LoRA implementations apply the adapter AFTER the base weight computation: output = W(x) + BA(x). The base weight W is frozen. So the LoRA bypass goes around whatever W does internally (including quantization). This should work with BitNet out of the box. But I should verify this.

Another thought: BitNet's architecture uses RoPE, squared ReLU, and subln normalization. These are all compatible with standard LoRA. The model architecture is a modified transformer, not a hybrid conv+attention like LFM2. This is actually MUCH simpler. No question about which layers to target -- it's all transformer layers with standard attention and FFN.

What about the transformers version? The model card says `pip install git+https://github.com/huggingface/transformers.git@096f25ae...` -- a specific commit. Our local machine has transformers 4.52.0.dev0. BitNet uses `custom_code` (trust_remote_code=True). Need to check if our version is compatible.

The GSM8K evaluation is straightforward -- it's a well-studied benchmark with simple answer extraction (the final number after "####"). Many existing implementations to draw from.

What scares me about this approach: the two-runtime complexity (bitnet.cpp for rollouts, PyTorch for gradients) could be a significant engineering burden for V1. Maybe for the local proof-of-concept, we skip bitnet.cpp entirely and just accept slow rollouts through PyTorch. The point of local testing is pipeline validation, not speed. Speed optimization (bitnet.cpp) can come when we move to the Thor.

Actually, for a CPU-only proof-of-concept, we could even skip GRPO entirely and do a simpler test: SFT-style fine-tuning on a few GSM8K solutions with TinyLoRA, just to prove the adapter mechanics work. The TinyLoRA paper says SFT needs 100-1000x more params, but we just need to prove the gradient flows. We can switch to GRPO later.

## Questions Arising
- Does PEFT/LoRA work with BitNet's `BitLinear` layers out of the box?
- Is transformers 4.52.0.dev0 compatible with the BitNet model, or do we need the specific commit?
- How fast is a single forward pass of BitNet BF16 on this Ryzen 5 through PyTorch?
- Can we install bitnet.cpp on this machine for fast inference benchmarking?
- What's the simplest possible end-to-end test that proves TinyLoRA works on BitNet?
- Does the LoRA adapter survive BitNet's ternary quantization in the forward pass?

## First Instincts
- Start with pure PyTorch, skip bitnet.cpp for now
- Test PEFT compatibility with BitNet before writing any custom TinyLoRA code
- Use SFT on a tiny dataset as the first gradient-flow test (faster than setting up GRPO)
- Keep the local test minimal: load, adapt, forward, backward, verify adapter weights changed
- Defer speed optimization to the Thor deployment
- If PEFT doesn't work with BitLinear, we may need a custom adapter implementation
