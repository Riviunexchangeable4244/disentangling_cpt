#!/usr/bin/env python3
"""
Compute layerwise L1-norm differences between a base model and a checkpoint.

- Separates attention vs FFN (MLP) components
- Outputs:
    - Layerwise JSON with per-layer max L1-norm for attention and FFN
    - Summary JSON with mean L1-norm across all layers
"""

import os
import json
import argparse
import torch
import numpy as np
from transformers import AutoModelForCausalLM

ATTENTION_COMPONENTS = ["q_proj", "k_proj", "v_proj", "o_proj"]
MLP_COMPONENTS = ["gate_proj", "up_proj", "down_proj"]

OUTPUT_LAYERWISE_DIR = "json_layerwise"
OUTPUT_SUMMARY_DIR = "json_summary"


def l1_norm(a: torch.Tensor, b: torch.Tensor) -> float:
    return torch.abs(a - b).mean().item()


def compute_layerwise_delta(base_model_path: str, ckpt_model_path: str, device="cpu"):
    print(f"[INFO] Loading base model: {base_model_path}")
    base_model = AutoModelForCausalLM.from_pretrained(base_model_path, device_map="cpu", torch_dtype=torch.float32)
    print(f"[INFO] Loading checkpoint model: {ckpt_model_path}")
    ckpt_model = AutoModelForCausalLM.from_pretrained(ckpt_model_path, device_map="cpu", torch_dtype=torch.float32)

    base_layers = base_model.model.layers
    ckpt_layers = ckpt_model.model.layers

    layer_stats = {}
    attention_max_list = []
    ffn_max_list = []

    for idx, (base_layer, ckpt_layer) in enumerate(zip(base_layers, ckpt_layers)):
        # Attention L1-norms
        attn_vals = []
        for comp in ATTENTION_COMPONENTS:
            if hasattr(base_layer.self_attn, comp):
                a = getattr(base_layer.self_attn, comp).weight.detach().cpu()
                b = getattr(ckpt_layer.self_attn, comp).weight.detach().cpu()
                attn_vals.append(l1_norm(a, b))
        attn_max = np.max(attn_vals) if attn_vals else 0.0

        # MLP/FFN L1-norms
        ffn_vals = []
        for comp in MLP_COMPONENTS:
            if hasattr(base_layer.mlp, comp):
                a = getattr(base_layer.mlp, comp).weight.detach().cpu()
                b = getattr(ckpt_layer.mlp, comp).weight.detach().cpu()
                ffn_vals.append(l1_norm(a, b))
        ffn_max = np.max(ffn_vals) if ffn_vals else 0.0

        layer_stats[f"layer_{idx}"] = {
            "attention_max": attn_max,
            "ffn_max": ffn_max,
            "delta_attn_ffn": attn_max - ffn_max
        }

        attention_max_list.append(attn_max)
        ffn_max_list.append(ffn_max)

    # Summary across all layers
    summary_stats = {
        "attn": float(np.mean(attention_max_list)),
        "ffn": float(np.mean(ffn_max_list))
    }

    model_name = os.path.basename(ckpt_model_path.rstrip("/"))
    os.makedirs(OUTPUT_LAYERWISE_DIR, exist_ok=True)
    os.makedirs(OUTPUT_SUMMARY_DIR, exist_ok=True)

    layerwise_path = os.path.join(OUTPUT_LAYERWISE_DIR, f"{model_name}_layerwise.json")
    with open(layerwise_path, "w") as f:
        json.dump({"layers": layer_stats}, f, indent=4)
    print(f"[INFO] Saved layerwise results to {layerwise_path}")

    summary_path = os.path.join(OUTPUT_SUMMARY_DIR, f"{model_name}.json")
    with open(summary_path, "w") as f:
        json.dump(summary_stats, f, indent=4)
    print(f"[INFO] Saved summary to {summary_path}")


def main():
    parser = argparse.ArgumentParser(description="Compute layerwise L1-norm attention vs FFN differences between base and checkpoint models")
    parser.add_argument("--base-model", required=True, help="Base model path or HF hub name")
    parser.add_argument("--ckpt-model", required=True, help="Checkpoint model path")
    parser.add_argument("--device", default="cpu", help="Device: cpu or cuda")

    args = parser.parse_args()

    compute_layerwise_delta(args.base_model, args.ckpt_model, device=args.device)


if __name__ == "__main__":
    main()
