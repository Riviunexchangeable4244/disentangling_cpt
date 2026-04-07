"""
L1 norm diff between weights of base model and checkpoints on CPT models
Only considers gate_proj and language-specific neurons (LSN)
"""

import os
import re
import json
import argparse
import torch
from transformers import AutoModelForCausalLM

def l1_norm(a: torch.Tensor, b: torch.Tensor) -> float:
    """Compute mean L1 norm over all elements"""
    return torch.abs(a - b).mean().item()

def load_model(model_path: str, device_map="cpu") -> AutoModelForCausalLM:
    print(f"Loading model from {model_path}")
    return AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        device_map=device_map,
    )

def detect_checkpoints(model_dir: str):
    """Detect checkpoint folders in a model directory"""
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

def compare_parameters(base_model_path: str, target_model_path: str, output_json: str, mask: dict, lang: str):
    checkpoints = detect_checkpoints(target_model_path)
    if not checkpoints:
        checkpoints = [target_model_path]

    base_model = load_model(base_model_path, device_map="cpu")
    base_layers = base_model.model.layers

    results = {
        "base_model": base_model_path,
        "target_model": target_model_path,
        "checkpoints": {},
    }

    for ckpt in checkpoints:
        print(f"Comparing with {ckpt}")
        target_model = load_model(ckpt, device_map="cpu")
        target_layers = target_model.model.layers

        ckpt_results = {"layers": {}}

        for idx, (base_layer, tgt_layer) in enumerate(zip(base_layers, target_layers)):
            neurons = mask.get(lang, {}).get(str(idx), [])
            if not neurons:
                continue

            if hasattr(base_layer.mlp, "gate_proj") and hasattr(tgt_layer.mlp, "gate_proj"):
                a = base_layer.mlp.gate_proj.weight.detach()[neurons].cpu()
                b = tgt_layer.mlp.gate_proj.weight.detach()[neurons].cpu()
                ckpt_results["layers"][f"layer_{idx}"] = {"gate_proj": l1_norm(a, b)}

        results["checkpoints"][extract_checkpoint_step(ckpt)] = ckpt_results
        del target_model
        torch.cuda.empty_cache()

    os.makedirs(os.path.dirname(output_json) or ".", exist_ok=True)
    with open(output_json, "w") as f:
        json.dump(results, f, indent=4)
    print(f"Saved results to {output_json}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Compare L1 norm differences for language-specific neurons weight between base and target models"
    )
    parser.add_argument(
        "--base-model",
        type=str,
        required=True,
        help="Path to base model"
    )
    parser.add_argument(
        "--target-model",
        type=str,
        required=True,
        help="Path to target model or checkpoint directory"
    )
    parser.add_argument(
        "--mask-file",
        type=str,
        required=True,
        help="Path to mask JSON file containing language-specific neurons"
    )
    parser.add_argument(
        "--lang",
        type=str,
        required=True,
        help="Language code to extract from mask (e.g., 'zh', 'ga', 'eu')"
    )
    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="Output JSON file path"
    )

    args = parser.parse_args()

    if not os.path.exists(args.mask_file):
        print(f"Error: Mask file not found: {args.mask_file}")
        exit(1)

    with open(args.mask_file, "r") as f:
        mask = json.load(f)

    print(f"Comparing base: {args.base_model} with target: {args.target_model}")
    compare_parameters(args.base_model, args.target_model, args.output, mask, args.lang)