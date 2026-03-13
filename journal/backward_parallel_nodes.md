# Nodes: Parallelizing Ternary Backward — Options

## The Problem

TernaryMatmulBackward takes 755ms of the 1,065ms backward pass (71%). The kernel itself runs 548ms across 210 serial calls. The remaining 207ms is Python/autograd overhead. Can we reduce the 548ms?

## Options

### Option A: OpenMP over N within each kernel call (per-thread buffers)
- **What:** Change the M=1 backward serial path to use `#pragma omp parallel for` over the N (out_features) loop. Each thread accumulates into its own grad_input buffer, then reduce (sum) at the end.
- **Implementation:** ~40 lines of C. Allocate thread-local buffers (10-27 KB each), parallelize the row loop, add final reduction.
- **Projected speedup:** 548ms → ~95ms (5.7x on kernel), total backward ~302ms
- **Memory:** 6 × 27 KB = 162 KB temporary (negligible)
- **Risk:** Low. Same algorithm, just threaded. OpenMP is already in the codebase. Per-thread buffers avoid all race conditions.
- **Caveat:** Diminishing returns — the 207ms autograd overhead becomes dominant at ~69% of total.

### Option B: Python-level cross-layer parallelism (gate+up, q+k+v)
- **What:** Use ThreadPoolExecutor to run independent backward kernels concurrently. Requires custom autograd hooks to intercept and parallelize.
- **Measured:** gate+up: 1.65x speedup. q+k+v: 1.37x.
- **Projected:** 548ms → ~398ms (1.4x overall)
- **Risk:** Medium. Needs careful autograd hook management. Thread dispatch overhead eats into short kernel times.
- **Verdict:** Modest gain, high complexity. Not worth it alone.

### Option C: Combined OpenMP + cross-layer (Option A + B)
- **What:** Both kernel-level and layer-level parallelism.
- **Problem:** Oversubscription. 6 OpenMP threads × 2 parallel calls = 12 on 6 cores.
- **Projected with 3T/kernel × 2 parallel:** 548ms → ~135ms (4.1x). Worse than Option A alone.
- **Verdict:** Complexity kills it. Option A is simpler and faster.

### Option D: Fused multi-layer backward C function
- **What:** New C function `ternary_backward_group()` that takes multiple packed weights + grad_outputs and runs them on separate threads internally. Eliminates Python thread dispatch overhead.
- **Projected:** Similar to Option B but without Python overhead.
- **Problem:** Still limited by the dependency graph (only 2-3 layers per group). Complex Python-level integration to gather the right layers into groups.
- **Verdict:** Complex plumbing for modest gain over Option A.

### Option E: Reduce autograd overhead (custom backward graph)
- **What:** Bypass PyTorch autograd entirely for the ternary backward. Implement a single C function that takes ALL 210 layers' packed weights and runs the full backward pass, layer by layer, with OpenMP within each layer.
- **Projected:** Eliminates ~207ms overhead entirely. With Option A kernel: total backward ~95ms + ~200ms LM head = ~295ms.
- **Risk:** High. Need to replicate autograd's backward graph traversal (attention, norms, residuals) in C or a minimal Python loop with no autograd.
- **Verdict:** Maximum gain but maximum complexity. Out of scope for now.

## Winner: Option A

Option A gives the best speedup-to-complexity ratio by far:
- 5.7x kernel speedup (548ms → 95ms)
- ~40 lines of C changes
- No Python-side changes (same API)
- No risk of breaking existing functionality
- Well-understood technology (OpenMP reduction)

Total backward projection: ~302ms (vs 1,065ms current = 3.5x improvement).
Total iteration projection: ~1,700ms (vs 2,410ms current = 1.4x improvement).
Cumulative from stock PyTorch: 91.7s → 1.7s = **54x**.
