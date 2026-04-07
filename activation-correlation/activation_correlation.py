#!/usr/bin/env python3
import json
from pathlib import Path
import re
import argparse
from typing import Dict, Tuple
from scipy.stats import pearsonr
import numpy as np


ATTENTION_COMPONENTS = ["q_proj", "k_proj", "v_proj", "o_proj"]


def parse_gate_means(json_path: str) -> Dict[int, float]:
    with open(json_path, "r") as f:
        data = json.load(f)
    result = {}
    for ckpt_str, ckpt_data in data["checkpoints"].items():
        gate_vals = [
            comps["gate_proj"] 
            for comps in ckpt_data["layers"].values() 
            if "gate_proj" in comps
        ]
        if gate_vals:
            result[int(ckpt_str)] = sum(gate_vals) / len(gate_vals)
    return result


def parse_attention_means(json_path: str, layers: list[int]) -> Dict[int, float]:
    with open(json_path, "r") as f:
        data = json.load(f)
    result = {}
    for ckpt_str, ckpt_data in data["checkpoints"].items():
        ckpt = int(ckpt_str)
        layer_vals = []
        for layer_name, comps in ckpt_data["layers"].items():
            layer_idx = int(layer_name.replace("layer_", ""))
            if layer_idx not in layers:
                continue
            vals = [comps[c] for c in ATTENTION_COMPONENTS if c in comps]
            if vals:
                layer_vals.append(max(vals))
        if layer_vals:
            result[ckpt] = sum(layer_vals) / len(layer_vals)
    return result


def load_activation_result_mean(folder_path: str) -> Tuple[Dict[int, float], Dict[int, float]]:
    folder = Path(folder_path)
    non_random_mean = {}
    random_mean = {}
    
    for file in folder.glob("*.json"):
        with open(file, "r") as f:
            raw_data = json.load(f)
        
        if "all_layers" in raw_data:
            mean_value = raw_data["all_layers"]
        else:
            data = {int(k): v for k, v in raw_data.items()}
            mean_value = sum(data.values()) / len(data) if data else 0.0
        
        match = re.search(r"checkpoint-(\d+)", file.stem)
        if not match:
            continue
        ckpt_num = int(match.group(1))
        
        if "-r" in file.stem:
            random_mean[ckpt_num] = mean_value
        else:
            non_random_mean[ckpt_num] = mean_value
    
    return non_random_mean, random_mean


def load_base_activation(folder: Path, lang: str) -> Tuple[float, float]:
    non_random_val = json.load(open(folder / f"{lang}.98304.json"))["all_layers"]
    random_val = json.load(open(folder / f"r.{lang}.98304.json"))["all_layers"]
    return non_random_val, random_val


def load_layer_selection(layer_selection_path: str) -> list[int]:
    path = Path(layer_selection_path)
    if not path.exists():
        raise FileNotFoundError(f"Layer selection file not found: {path}")
    
    data = json.load(open(path))
    interface_layers = data.get("interface_layers")
    
    if not interface_layers:
        raise ValueError(f"No interface_layers found in {path}")
    
    return interface_layers


def calculate_correlation(
    model_name: str,
    lang: str,
    base_tag: str,
    lsn_activation_path: str,
    all_weight_diff_path: str,
    lsn_weight_diff_path: str,
    layer_selection_path: str,
    output_path: str
):
    output_path = Path(output_path)
    output_path.mkdir(parents=True, exist_ok=True)
    
    cpt_non_random, cpt_random = load_activation_result_mean(
        Path(lsn_activation_path) / model_name
    )
    ckpts = sorted(cpt_non_random.keys())
    
    base_non_random, base_random = load_base_activation(
        Path(lsn_activation_path) / base_tag, 
        lang
    )
    
    cpt_sub = {
        ckpt: cpt_non_random[ckpt] - cpt_random.get(ckpt, 0.0) 
        for ckpt in ckpts
    }
    delta_sub = np.array([
        cpt_sub[ckpt] - (base_non_random - base_random) 
        for ckpt in ckpts
    ])
    
    interface_layers = load_layer_selection(layer_selection_path)
    
    lsn_gate_mean = parse_gate_means(Path(lsn_weight_diff_path))
    attention_mean = parse_attention_means(Path(all_weight_diff_path), interface_layers)
    
    x_attn = np.abs(np.array([attention_mean[ckpt] for ckpt in ckpts]))
    x_gate = np.abs(np.array([lsn_gate_mean[ckpt] for ckpt in ckpts]))
    delta_abs = np.abs(delta_sub)
    
    attn_corr, _ = pearsonr(delta_abs, x_attn)
    gate_corr, _ = pearsonr(delta_abs, x_gate)
    
    results = {
        "interface_attention_correlation_strength": attn_corr,
        "lsn_ffn_correlation_strength": gate_corr
    }
    
    json_file = output_path / f"{model_name}_correlation_strength.json"
    with open(json_file, "w") as f:
        json.dump(results, f, indent=4)
    
    return results


def main():
    parser = argparse.ArgumentParser(
        description="Calculate correlation strength between interface attention and LSN FFN weight differences."
    )
    parser.add_argument("--model-name", required=True)
    parser.add_argument("--lang", required=True)
    parser.add_argument("--base-tag", required=True)
    parser.add_argument("--lsn-activation-path", required=True)
    parser.add_argument("--all-weight-diff-path", required=True)
    parser.add_argument("--lsn-weight-diff-path", required=True)
    parser.add_argument("--layer-selection-path", required=True)
    parser.add_argument("--output-dir", required=True)
    
    args = parser.parse_args()
    
    calculate_correlation(
        model_name=args.model_name,
        lang=args.lang,
        base_tag=args.base_tag,
        lsn_activation_path=args.lsn_activation_path,
        all_weight_diff_path=args.all_weight_diff_path,
        lsn_weight_diff_path=args.lsn_weight_diff_path,
        layer_selection_path=args.layer_selection_path,
        output_path=args.output_dir
    )


if __name__ == "__main__":
    main()