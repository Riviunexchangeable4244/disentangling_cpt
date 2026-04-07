#!/usr/bin/env python3
"""
Compute correlations between averaged weight differences and benchmark performance.

- Uses weight diff JSON with `interface_avg` and `semantic_hub_avg` per checkpoint
- Computes Pearson correlation with benchmark metric (e.g., accuracy)
"""

import json
import argparse
from pathlib import Path
import numpy as np
from scipy.stats import pearsonr


# ----------------------------
# Utilities
# ----------------------------
def load_json(path: Path) -> dict:
    with open(path, "r") as f:
        return json.load(f)


# ----------------------------
# Compute correlations
# ----------------------------
def compute_correlations(
    benchmark_json: dict,
    weight_json: dict,
    checkpoint_stride: int = 1,
) -> dict:
    """
    Compute Pearson correlations for interface and semantic hub layers.

    Args:
        benchmark_json: Benchmark results per checkpoint
        weight_json: Weight difference JSON (pre-averaged per category)
        checkpoint_stride: Sample every Nth checkpoint

    Returns:
        Dictionary of correlations
    """
    results = {}
    checkpoints = sorted(benchmark_json.keys(), key=int)[::checkpoint_stride]

    # Get the first benchmark key dynamically (for metric name)
    first_ckpt = benchmark_json[checkpoints[0]]
    metric_key = next(iter(first_ckpt.keys()))

    for layer_type, avg_key in [("interface", "interface_avg"), ("semantic_hub", "semantic_hub_avg")]:
        x, y = [], []

        for ckpt in checkpoints:
            ckpt_str = str(ckpt)
            if ckpt_str not in weight_json["checkpoints"] or ckpt_str not in benchmark_json:
                continue

            x_val = weight_json["checkpoints"][ckpt_str].get(avg_key, 0.0)
            y_val = benchmark_json[ckpt_str][metric_key]["acc"]

            x.append(float(x_val))
            y.append(float(y_val))

        if len(x) < 2:
            print(f"Warning: Not enough data points for {layer_type}, skipping")
            continue

        corr, _ = pearsonr(x, y)
        results[layer_type] = {
            "correlation": float(corr),
        }

    return results


# ----------------------------
# Main
# ----------------------------
def main():
    parser = argparse.ArgumentParser(
        description="Compute Pearson correlations between averaged weight differences and benchmark performance"
    )
    parser.add_argument("benchmark", type=Path, help="Benchmark JSON per checkpoint")
    parser.add_argument("weight_diff", type=Path, help="Weight diff JSON (pre-averaged)")
    parser.add_argument("-o", "--output", type=Path, required=True, help="Output JSON file")
    parser.add_argument("--checkpoint-stride", type=int, default=1, help="Sample every Nth checkpoint")

    args = parser.parse_args()

    if not args.benchmark.exists():
        parser.error(f"Benchmark file not found: {args.benchmark}")
    if not args.weight_diff.exists():
        parser.error(f"Weight diff file not found: {args.weight_diff}")

    print("[INFO] Loading JSON files...")
    benchmark_json = load_json(args.benchmark)
    weight_json = load_json(args.weight_diff)

    print("[INFO] Computing correlations...")
    correlations = compute_correlations(
        benchmark_json,
        weight_json,
        checkpoint_stride=args.checkpoint_stride
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(correlations, f, indent=4)

    print(f"[INFO] Correlations saved to {args.output}")
    print("\n=== Summary ===")
    for layer_type, values in correlations.items():
        print(f"{layer_type}: correlation={values['correlation']:+.3f}, p-value={values['p_value']:.3e}, points={values['n_points']}")


if __name__ == "__main__":
    main()
