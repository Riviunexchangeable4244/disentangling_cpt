#!/usr/bin/env python3
"""
Weight diff Analysis (Interface & Semantic Hub Layers, Averaged L1 Norm)

- Compare base vs. checkpoints
- Restrict to interface + semantic hub layers
- Compute L1 norm differences
- Average across layers to produce a single drift score per category per checkpoint
"""

import os
import re
import json
import torch
import argparse
from transformers import AutoModelForCausalLM
from pathlib import Path

# Components of interest
COMPONENTS = ["q_proj", "k_proj", "v_proj", "o_proj",
              "gate_proj", "up_proj", "down_proj"]


def l1_norm(a: torch.Tensor, b: torch.Tensor) -> float:
    return torch.abs(a - b).mean().item()


def load_model(model_path: str, device_map="cpu") -> AutoModelForCausalLM:
    print(f"[INFO] Loading model from {model_path}")
    return AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        device_map=device_map,
    )


def detect_checkpoints(model_dir: str):
    pattern = re.compile(r"checkpoint-\d+")
    try:
        ckpts = [os.path.join(model_dir, f) for f in os.listdir(model_dir) if pattern.match(f)]
    except Exception:
        return []
    return sorted(ckpts, key=lambda x: int(x.split("-")[-1]))


def extract_checkpoint_step(ckpt_path: str) -> int:
    match = re.search(r"checkpoint-(\d+)", ckpt_path)
    if match:
        return int(match.group(1))
    return -1


def load_layer_list(path: str) -> list[int]:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Layer selection file not found: {path}")
    data = json.load(open(path))
    return data.get("layers", [])


def compare_parameters(base_model_path: str, target_model_path: str, output_json: str,
                       interface_path: str, hub_path: str, device: str = "cpu"):
    checkpoints = detect_checkpoints(target_model_path)
    if not checkpoints:
        checkpoints = [target_model_path]

    base_model = load_model(base_model_path, device_map=device)
    base_layers = base_model.model.layers

    interface_layers = load_layer_list(interface_path)
    hub_layers = load_layer_list(hub_path)

    results = {
        "base_model": base_model_path,
        "target_model": target_model_path,
        "checkpoints": {},
    }

    for ckpt in checkpoints:
        print(f"[INFO] Comparing with {ckpt}")
        target_model = load_model(ckpt, device_map=device)
        target_layers = target_model.model.layers

        interface_vals = []
        hub_vals = []

        for idx, (base_layer, tgt_layer) in enumerate(zip(base_layers, target_layers)):
            if idx not in interface_layers and idx not in hub_layers:
                continue

            comp_diffs = []
            for comp in COMPONENTS:
                if hasattr(base_layer.self_attn, comp):  # Attention comp
                    a = getattr(base_layer.self_attn, comp).weight.detach().cpu()
                    b = getattr(tgt_layer.self_attn, comp).weight.detach().cpu()
                elif hasattr(base_layer.mlp, comp):  # MLP comp
                    a = getattr(base_layer.mlp, comp).weight.detach().cpu()
                    b = getattr(tgt_layer.mlp, comp).weight.detach().cpu()
                else:
                    continue
                comp_diffs.append(l1_norm(a, b))

            if comp_diffs:
                avg_layer_diff = sum(comp_diffs) / len(comp_diffs)
                if idx in interface_layers:
                    interface_vals.append(avg_layer_diff)
                if idx in hub_layers:
                    hub_vals.append(avg_layer_diff)

        ckpt_results = {
            "interface_avg": float(sum(interface_vals) / len(interface_vals)) if interface_vals else 0.0,
            "semantic_hub_avg": float(sum(hub_vals) / len(hub_vals)) if hub_vals else 0.0,
        }

        results["checkpoints"][extract_checkpoint_step(ckpt)] = ckpt_results
        del target_model
        torch.cuda.empty_cache()

    with open(output_json, "w") as f:
        json.dump(results, f, indent=4)
    print(f"[INFO] Saved results to {output_json}")


def main():
    parser = argparse.ArgumentParser(
        description="Weight diff analysis (L1 norm average) for interface & semantic hub layers."
    )
    parser.add_argument("--base-model", required=True, help="Base model (HF hub or local path)")
    parser.add_argument("--target-dir", required=True, help="Target model dir with checkpoints")
    parser.add_argument("--interface-json", required=True, help="JSON file with interface layers")
    parser.add_argument("--hub-json", required=True, help="JSON file with semantic hub layers")
    parser.add_argument("--output", required=True, help="Output JSON file path")
    parser.add_argument("--device", default="cpu", help="Device: cpu or cuda")

    args = parser.parse_args()

    compare_parameters(
        args.base_model,
        args.target_dir,
        args.output,
        args.interface_json,
        args.hub_json,
        device=args.device,
    )


if __name__ == "__main__":
    main()
