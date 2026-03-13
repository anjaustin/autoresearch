# Reflect: Soft-Chip Optimization Opportunities

## What Surprised Me

The analysis shows we're at ~9% of theoretical peak FP32 throughput. That sounds bad, but it's actually reasonable for a memory-streaming workload with per-element branch-like operations (LUT lookups). The real surprise is that we appear to be **compute-bound, not memory-bound** (Node 7). The L3 bandwidth would allow sub-millisecond weight reads, but we're at 13.2ms. This means the bottleneck is instruction throughput in the inner loop, not data movement. That's good news -- it means parallelism and tiling have clear headroom.

The M=1 problem (Node 1) is the most consequential realization. Autoregressive generation is the primary use case for the autoresearch loop (RL rollouts), and the current kernel gets ZERO parallelism for it. A 4.1x speedup that only works for batched forward passes and delivers 0x improvement for autoregressive generation is... not very useful for our actual goal.

## Where I Was Wrong (or Incomplete)

In the RAW phase I initially thought L2 was 512KB per core based on "3 MiB across 6 instances." But Zen 3 Cezanne (Ryzen 5 5675U) actually has 512KB L2 per core as separate instances. So I was right -- the 1.6MB packed weight matrix spans ~3 L2 caches. But the key insight I almost missed: with tiling (Node 2), we don't need the ENTIRE weight matrix in cache -- we only need the tile's weight rows plus the activation vector.

I also initially framed PSHUFB vs LUT as alternatives (Node 4 vs Node 2). On reflection, these are orthogonal: PSHUFB improves the decode step, tiling improves data reuse. Both could be applied. But the ROI ordering is clear: parallelism > tiling > PSHUFB.

## What the Tensions Reveal

The core tension is **Node 1 vs Node 9**: how much to invest in CPU optimization when we're deploying to GPU. The resolution is pragmatic:

1. **Out-features parallelism (Node 1)** is essential even as a short-term investment. Without it, the soft-chip can't help with autoregressive generation at all. This is maybe 20 lines of code change.

2. **Malloc fix (Node 3)** is a correctness/hygiene issue, not an optimization. Do it regardless.

3. **Numerical validation (Node 5)** is a hard gate. We cannot integrate with PyTorch without it.

4. **PyTorch integration (Node 6)** is the deliverable that makes everything else matter.

5. **Tiling (Node 2)** and **PSHUFB (Node 4)** are nice-to-haves that can wait.

The **Node 5 → Node 6** dependency chain is the critical path. Everything else is secondary.

## The Uncomfortable Question

Do we even need the soft-chip for the next step? The backward pass (208s) is the actual wall-clock bottleneck for training, and it can't use the ternary trick. The forward pass is 12s. Even at 4.1x, we save ~9s per forward pass. For a training step that takes 220s (forward + backward), that's a 4% improvement. Not nothing, but not transformative.

The soft-chip becomes transformative when we separate rollout generation (forward-only, soft-chip-accelerated) from gradient computation (forward+backward, PyTorch-only). In the GRPO loop: generate completions via soft-chip (fast), then replay only the selected completions through PyTorch for gradients (slow but infrequent). This two-pass architecture is what makes the 4.1x forward speedup matter -- because the rollout phase is 90% of the wall clock in GRPO.

But on CPU, even with a 4.1x forward speedup, autoregressive generation of 200 tokens through 30 layers is: 200 × 30 × 7 × (weighted average of layer times) / 4.1. Still slow. The soft-chip's real win is on Thor where the forward pass through custom CUDA kernels will be dramatically faster, and the CPU kernel becomes irrelevant.

So the honest assessment: the soft-chip is a worthwhile proof of concept and a moderate win for CPU development iteration, but it's not the bottleneck removal we need. The real bottleneck removal is deploying to Thor.

## Revised Priority Ordering

1. **Numerical validation** (Node 5) -- hard gate, do first
2. **Out-features parallelism** (Node 1) -- essential for M=1, small effort
3. **Malloc fix** (Node 3) -- trivial hygiene
4. **PyTorch C extension** (Node 6) -- the actual deliverable
5. **Full 30-layer benchmark** (Node 8) -- validate the projected speedup
6. **Tiling** (Node 2) -- nice-to-have, defer
7. **PSHUFB** (Node 4) -- nice-to-have, defer

Items 1-5 are the actionable work. Items 6-7 are optimization opportunities to revisit if CPU performance remains critical (e.g., if Thor deployment is delayed).
