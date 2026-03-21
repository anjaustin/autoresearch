# Project GhostWeight: The Next Step-Change

Following the successful validation of the 1KB model (TinyLoRA + GhostWeight), the project is moving from "steering noise" to "structured recovery." This document outlines the strategic technical leaps that will define the next phase of development.

## 1. The Ghost Mixture of Experts (GMoE)
Instead of one fixed 8-byte seed for the entire 2B parameter base, we can utilize a bank of 16-32 seeds.
- **The Implementation:** A tiny learned "router" (a few extra KB) selects or blends weights from different PRNG streams per layer or per token.
- **The Impact:** This effectively multiplies the "base knowledge" the model can draw from without increasing the storage footprint beyond a few hundred bytes. It allows the model to find "lucky" sub-distributions in the PRNG space that match specific linguistic patterns.

## 2. Rank-1 Expansion (The "2MB LLM")
Moving from rank-0 (1 scalar per layer) to Rank-1 LoRA (a trainable $U$ and $V$ vector pair per layer).
- **The Implementation:** Transition from `output = base(x) + scale * (x @ v^T) @ u^T` (where $u,v$ are fixed) to making $u$ and $v$ trainable.
- **The Impact:** For a 2560-dim layer, this moves from 1 parameter to ~5,120 parameters per layer. Total model size increases to ~2MB. While larger than 1KB, 2MB is still negligible for modern hardware and provides the degrees of freedom needed to actively correct PRNG errors rather than just nudging them.

## 3. Evolutionary Seed Searching (Meta-Ghost)
The current seed `42` is arbitrary. The seed itself should be a hyperparameter.
- **The Implementation:** Use the autonomous agent to search the `GHOST_SEED` space. Run short, high-learning-rate bursts across different seeds to identify "gold-mine seeds."
- **The Impact:** In ternary space, weight distributions are highly sensitive. A seed that naturally exhibits lower initial entropy or better-aligned feature detectors for the target task can significantly accelerate GRPO convergence.

## 4. Recursive Ghosting (Adapter Compression)
If the adapter parameters themselves become a storage concern in extreme environments:
- **The Implementation:** Replace the learned adapters with a secondary, even smaller PRNG + seed system.
- **The Impact:** This creates a recursive "Inception" architecture—weights generated from noise, steered by adapters that are also generated from noise, controlled by a handful of "Master Parameters."

## 5. Zero-Load Mobile Inference
The ultimate portability milestone.
- **The Implementation:** Port the `ghost_matmul_lut` kernel to a mobile-friendly framework (ExecuTorch, or a C++ Android/iOS wrapper).
- **The Impact:** Shipping a "2B Parameter Model" as a text string (e.g., in a message). This enables a 2B LLM to run on a mobile device with **zero weight-loading time**, as the base weights are generated in-memory.

## Immediate Technical Priority
**Rank-0 vs. Rank-1 Comparison:** Evaluate if the degrees of freedom in a Rank-1 model (2MB) provide a non-linear jump in reasoning capability compared to the current Rank-0 (1KB) constraint. If Rank-1 reaches near-parity with the full 500MB model, the "Ghost" approach becomes a viable alternative to traditional model compression.
