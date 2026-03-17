# LMM Pass 10 RAW: MTP18, LUT Kernels, and the Scaling Fix

## The Problem
GhostWeight (Pass 8) successfully replaced 500MB of weights with a PRNG, but the model output was gibberish. Initial rewards were 0.0, and the model failed simple addition (1+1).

## Observations
1. **The Scale Gap:** BitNet `AutoBitLinear` doesn't just use {-1, 0, +1}. It calculates a `weight_scale` (mean absolute value of master weights) per layer, typically around 2.33. GhostWeight was using 1.0.
2. **The Matmul Bottleneck:** Regenerating weights from PRNG on every forward/backward pass is expensive. The initial `ghost_matmul.c` was slower than the packed ternary kernel.
3. **MTP18 Idea:** Instead of ternary, what if we use a native base-3 floating point format? 8 trits = {sign:1, exp:2, mant:5}.

## Technical Breakthroughs

### 1. The Scaling Fix
We updated `softchip/torch_ternary.py` to capture `module.weight.abs().mean().item()` during patching. This scale is passed to the C kernel, which multiplies the PRNG output by this scale before the dot product. 
**Result:** GhostWeight now produces coherent text similar to the base model.

### 2. LUT-Optimized Kernel
Built `softchip/ghost_matmul_lut.c`. 
- Precomputes a 2MB Lookup Table mapping every 16-bit word (8 ternary values) to an AVX2 register (8 floats).
- Replaces complex bit-shifting and sign-extension with two LUT loads and two AVX2 stores.
- Speedup: 11.1ms -> 3.6ms (3.1x) for M=1.

### 3. MTP18 Experiment
Drafted `softchip/mtp18_matmul.c`. 
- Logic: `sign * 3^exp * (1 + mant/81)`.
- Performance: 66ms/layer. Currently much slower than ternary because x86 has no native base-3 instructions and the decode logic is complex.
- Verdict: Parked for now, but valid for future base-3 hardware search.

## GRPO Resume Fix
Identified a bug in `grpo_train.py` where `base_acc` was unbound when resuming from a checkpoint. Fixed by recalculating or defaulting to 0.0.

## Current State
Training resumed at Step 220. Running reward is 91%. The "1KB Model" is finally learning.
