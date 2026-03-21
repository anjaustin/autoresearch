"""
Layer importance analysis for GhostChain adapters.

Loads one or more grpo_train.py checkpoints and analyses which of the 840
adapter positions are absorbing the training signal.

Outputs:
  - Per-layer-type summary (q_proj, k_proj, v_proj, o_proj, gate_proj,
    up_proj, down_proj) — mean and max |scale| per type
  - Per-expert summary (expert1, expert2, expert3, observer) — which
    position in the chain carries the most weight
  - Per-decoder-layer summary (layers 0–29) — early vs late depth profile
  - Top-K adapter ranking — which specific adapters are doing the most work
  - Adaptation Lorenz curve — what fraction of total |scale| lives in top-K%
    of adapters (measures concentration vs spread)

Usage:
    # Analyse the most recent checkpoint:
    python analyze_layers.py

    # Analyse a specific checkpoint:
    python analyze_layers.py checkpoints/grpo_step_0120.pt

    # Compare multiple checkpoints side-by-side:
    python analyze_layers.py checkpoints/grpo_step_0050.pt \
                             checkpoints/grpo_step_0100.pt \
                             checkpoints/grpo_step_0120.pt

    # Show the full ranked list of all 840 adapters:
    python analyze_layers.py --top 840
"""

import argparse
import re
import sys
from pathlib import Path
from collections import defaultdict


# ---------------------------------------------------------------------------
# Layer name parsing
# ---------------------------------------------------------------------------
# Expected name format from inject_adapters():
#   model.layers.{layer_idx}.{component}.{proj_name}.{expert_name}
# e.g.: model.layers.7.self_attn.q_proj.expert1
#       model.layers.12.mlp.gate_proj.observer

# GhostChain format (840 params): model.layers.N.component.proj.expertX
_GHOST_RE = re.compile(
    r"model\.layers\.(\d+)\."
    r"([\w_]+)\."
    r"([\w_]+)\."
    r"(expert\d+|observer)"
)
# TinyLoRA format (210 params): model.layers.N.component.proj
_TINYLORA_RE = re.compile(
    r"model\.layers\.(\d+)\."
    r"([\w_]+)\."
    r"([\w_]+)$"
)

PROJ_ORDER = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
EXPERT_ORDER = ["expert1", "expert2", "expert3", "observer"]
COMPONENT_MAP = {
    "q_proj": "attn", "k_proj": "attn", "v_proj": "attn", "o_proj": "attn",
    "gate_proj": "mlp",  "up_proj": "mlp",  "down_proj": "mlp",
}


def parse_name(name):
    """Parse an adapter name into (layer_idx, component, proj, expert).
    Handles both GhostChain (4 experts/layer) and TinyLoRA (1 scalar/layer).
    Returns None if the name doesn't match either pattern."""
    m = _GHOST_RE.search(name)
    if m:
        return int(m.group(1)), m.group(2), m.group(3), m.group(4)
    m = _TINYLORA_RE.search(name)
    if m:
        return int(m.group(1)), m.group(2), m.group(3), "scale"
    return None


# ---------------------------------------------------------------------------
# Analysis
# ---------------------------------------------------------------------------

def analyse_checkpoint(path):
    """Load a checkpoint and return a flat list of (name, abs_scale) pairs."""
    import torch
    ckpt = torch.load(path, weights_only=False)
    scales = ckpt["adapter_scales"]
    step = ckpt.get("step", "?")
    records = []
    for name, tensor in scales.items():
        val = abs(float(tensor))
        parsed = parse_name(name)
        if parsed is not None:
            layer_idx, component, proj, expert = parsed
            records.append({
                "name": name,
                "abs_scale": val,
                "layer_idx": layer_idx,
                "component": component,
                "proj": proj,
                "expert": expert,
                "block": COMPONENT_MAP.get(proj, "other"),
            })
    return step, records


def group_by(records, key, order=None):
    """Group records by a key, return dict of {key: [abs_scale, ...]}."""
    groups = defaultdict(list)
    for r in records:
        groups[r[key]].append(r["abs_scale"])
    if order:
        return {k: groups[k] for k in order if k in groups}
    return dict(sorted(groups.items()))


def lorenz_curve(records, n_points=10):
    """
    Compute Lorenz curve data: what fraction of total |scale| is held by
    the top-K% of adapters.

    Returns list of (cumulative_fraction_of_adapters, cumulative_fraction_of_signal)
    from the *most active* adapter downward.
    """
    vals = sorted([r["abs_scale"] for r in records], reverse=True)
    total = sum(vals) or 1e-12
    n = len(vals)
    points = []
    cumsum = 0
    step_size = max(1, n // n_points)
    for i, v in enumerate(vals):
        cumsum += v
        if (i + 1) % step_size == 0 or i == n - 1:
            points.append(((i + 1) / n, cumsum / total))
    return points


def print_group_table(title, groups, total_signal, bar_width=30):
    """Print a sorted table of group statistics."""
    print(f"\n  {title}")
    print(f"  {'Group':<18}  {'Count':>5}  {'Mean |s|':>10}  {'Max |s|':>10}  "
          f"{'% signal':>8}  {'Distribution'}")
    print(f"  {'-'*18}  {'-'*5}  {'-'*10}  {'-'*10}  {'-'*8}  {'-'*bar_width}")

    rows = []
    for k, vals in groups.items():
        mean_v = sum(vals) / len(vals)
        max_v = max(vals)
        pct = sum(vals) / total_signal * 100
        rows.append((k, vals, mean_v, max_v, pct))
    rows.sort(key=lambda x: x[2], reverse=True)  # sort by mean

    for k, vals, mean_v, max_v, pct in rows:
        bar_len = int(pct / 100 * bar_width)
        bar = "#" * bar_len + "." * (bar_width - bar_len)
        print(f"  {str(k):<18}  {len(vals):>5}  {mean_v:>10.5f}  {max_v:>10.5f}  "
              f"{pct:>7.1f}%  {bar}")


def print_lorenz(records):
    """Print a text-art Lorenz curve."""
    points = lorenz_curve(records, n_points=10)
    print("\n  Adaptation Lorenz Curve  (top adapters → cumulative signal)")
    print("  (perfect equality = diagonal; higher curve = more concentrated)")
    print(f"\n  {'Top-K adapters':>16}  {'Cumul. signal':>14}")
    print(f"  {'-'*16}  {'-'*14}")
    for frac_adapters, frac_signal in points:
        n = int(frac_adapters * 100)
        bar = "#" * int(frac_signal * 40)
        print(f"  {'top ' + str(n) + '%':>16}  {frac_signal*100:>6.1f}%  {bar}")

    # Gini coefficient (0 = perfectly equal, 1 = all signal in one adapter)
    vals = sorted([r["abs_scale"] for r in records])
    n = len(vals)
    total = sum(vals) or 1e-12
    gini = (2 * sum((i + 1) * v for i, v in enumerate(vals)) / (n * total)) - (n + 1) / n
    print(f"\n  Gini coefficient: {gini:.3f}  "
          f"(0 = all adapters equal, 1 = one adapter dominates)")


def print_depth_profile(records, n_layers=30, bar_width=40):
    """Show mean |scale| across decoder depth."""
    by_layer = defaultdict(list)
    for r in records:
        by_layer[r["layer_idx"]].append(r["abs_scale"])

    max_mean = max(sum(v)/len(v) for v in by_layer.values()) or 1e-12

    print("\n  Depth Profile  (mean |scale| per decoder layer, 0=earliest)")
    print(f"  {'Layer':>6}  {'Mean |s|':>10}  {'Max |s|':>10}  Bar")
    print(f"  {'-'*6}  {'-'*10}  {'-'*10}  {'-'*bar_width}")
    for layer_idx in range(n_layers):
        vals = by_layer.get(layer_idx, [])
        if not vals:
            continue
        mean_v = sum(vals) / len(vals)
        max_v = max(vals)
        bar_len = int(mean_v / max_mean * bar_width)
        bar = "#" * bar_len
        print(f"  {layer_idx:>6}  {mean_v:>10.5f}  {max_v:>10.5f}  {bar}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Analyse GhostChain adapter layer importance")
    parser.add_argument(
        "checkpoints", nargs="*",
        help="Checkpoint .pt file(s). If omitted, uses the most recent in checkpoints/.",
    )
    parser.add_argument(
        "--top", type=int, default=30,
        help="Number of individual adapters to show in the top-K ranking (default: 30)",
    )
    args = parser.parse_args()

    import torch  # defer so --help works without torch

    # Resolve checkpoint paths
    ckpt_paths = [Path(p) for p in args.checkpoints]
    if not ckpt_paths:
        ckpt_dir = Path("checkpoints")
        candidates = sorted(ckpt_dir.glob("grpo_step_*.pt"))
        if not candidates:
            print("No checkpoints found in checkpoints/. Pass a path explicitly.")
            sys.exit(1)
        ckpt_paths = [candidates[-1]]
        print(f"No checkpoint specified. Using most recent: {ckpt_paths[0]}")

    for ckpt_path in ckpt_paths:
        print("\n" + "=" * 70)
        print(f"Checkpoint: {ckpt_path}")
        print("=" * 70)

        step, records = analyse_checkpoint(str(ckpt_path))
        n = len(records)
        total_signal = sum(r["abs_scale"] for r in records) or 1e-12
        mean_overall = total_signal / n
        max_overall = max(r["abs_scale"] for r in records)

        print(f"\n  Step: {step}  |  Total adapters: {n}  |  "
              f"Mean |scale|: {mean_overall:.5f}  |  Max |scale|: {max_overall:.5f}")

        # ---- By projection type ----
        by_proj = group_by(records, "proj", order=PROJ_ORDER)
        print_group_table("By Projection Type", by_proj, total_signal)

        # ---- By attention vs MLP block ----
        by_block = group_by(records, "block")
        print_group_table("By Block Type (attn vs mlp)", by_block, total_signal)

        # ---- By expert position ----
        by_expert = group_by(records, "expert", order=EXPERT_ORDER)
        print_group_table("By Expert Position", by_expert, total_signal)

        # ---- Depth profile ----
        print_depth_profile(records)

        # ---- Lorenz curve ----
        print_lorenz(records)

        # ---- Top-K individual adapters ----
        top_k = args.top
        sorted_records = sorted(records, key=lambda r: r["abs_scale"], reverse=True)
        cumulative = 0
        print(f"\n  Top-{top_k} individual adapters by |scale|:")
        print(f"  {'Rank':>5}  {'|scale|':>10}  {'Cumul%':>7}  Name")
        print(f"  {'-'*5}  {'-'*10}  {'-'*7}  {'-'*60}")
        for rank, r in enumerate(sorted_records[:top_k], 1):
            cumulative += r["abs_scale"]
            pct = cumulative / total_signal * 100
            # Shorten name for display
            short = r["name"].replace("model.layers.", "L").replace(
                ".self_attn.", ".attn.").replace(".mlp.", ".mlp.")
            print(f"  {rank:>5}  {r['abs_scale']:>10.5f}  {pct:>6.1f}%  {short}")

        # ---- Summary insight ----
        top_10pct = sorted([r["abs_scale"] for r in records], reverse=True)[:n//10]
        top_10pct_signal = sum(top_10pct) / total_signal * 100
        top_20pct = sorted([r["abs_scale"] for r in records], reverse=True)[:n//5]
        top_20pct_signal = sum(top_20pct) / total_signal * 100
        print(f"\n  Signal concentration:")
        print(f"    Top 10% of adapters ({n//10} positions) → {top_10pct_signal:.1f}% of total signal")
        print(f"    Top 20% of adapters ({n//5} positions) → {top_20pct_signal:.1f}% of total signal")

        # Sparse adapter recommendation
        threshold_90 = None
        cumsum = 0
        for rank, r in enumerate(sorted_records, 1):
            cumsum += r["abs_scale"]
            if cumsum / total_signal >= 0.90 and threshold_90 is None:
                threshold_90 = rank
        if threshold_90:
            print(f"\n  Sparse adapter recommendation:")
            print(f"    {threshold_90} adapters ({threshold_90/n*100:.1f}% of 840) capture 90% of signal.")
            # How many unique layers does this span?
            top_layers = {records[i]["layer_idx"]
                          for i in range(len(sorted_records))
                          if sorted_records.index(sorted_records[i]) < threshold_90}
            # Recompute correctly:
            top_layer_set = set()
            top_proj_set = set()
            for r in sorted_records[:threshold_90]:
                top_layer_set.add(r["layer_idx"])
                top_proj_set.add(r["proj"])
            print(f"    These span {len(top_layer_set)} decoder layers and "
                  f"{len(top_proj_set)} projection types.")


if __name__ == "__main__":
    main()
