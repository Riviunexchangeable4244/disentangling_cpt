#!/usr/bin/env python3
"""
Swap layers between base and target models using different strategies.
Supports semantic hub swapping (non-interface layers) and random layer swapping.
"""

import json
import random
import argparse
from pathlib import Path
from typing import List, Dict
from transformers import AutoModelForCausalLM, AutoTokenizer


def load_model(model_path: Path):
    """Load a causal language model from the given path."""
    return AutoModelForCausalLM.from_pretrained(str(model_path))


def swap_layers(base_model, target_model, layer_indices: List[int]):
    """
    Swap selected layers from base_model into target_model.
    
    Args:
        base_model: Source model (provides layers to copy)
        target_model: Target model (receives swapped layers)
        layer_indices: List of layer indices to swap
    
    Returns:
        Modified target_model
    """
    base_layers = base_model.model.layers
    target_layers = target_model.model.layers

    for idx in layer_indices:
        target_layers[idx].load_state_dict(base_layers[idx].state_dict())
        print(f"  Swapped layer {idx}")

    return target_model


def select_random_layers(num_layers: int, n: int, seed: int) -> List[int]:
    """Select n random layer indices from range [0, num_layers)."""
    random.seed(seed)
    return random.sample(list(range(num_layers)), n)


def get_semantic_hub_layers(num_layers: int, interface_layers: List[int]) -> List[int]:
    """Return all layer indices that are NOT in interface_layers."""
    return [l for l in range(num_layers) if l not in interface_layers]


def main():
    parser = argparse.ArgumentParser(
        description="Swap layers between base and target models"
    )
    parser.add_argument(
        "base_model",
        type=Path,
        help="Path to base model (source of semantic hub layers)"
    )
    parser.add_argument(
        "target_model",
        type=Path,
        help="Path to target model to modify"
    )
    parser.add_argument(
        "interface_layers",
        type=Path,
        help="JSON file with interface_layers list"
    )
    parser.add_argument(
        "-o", "--output",
        type=Path,
        required=True,
        help="Output directory for swapped models"
    )
    parser.add_argument(
        "--methods",
        nargs="+",
        choices=["semantic_hub", "random"],
        default=["semantic_hub", "random"],
        help="Swapping methods to use (default: both)"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility (default: 42)"
    )
    
    args = parser.parse_args()
    
    # Validate paths
    if not args.base_model.exists():
        parser.error(f"Base model not found: {args.base_model}")
    if not args.target_model.exists():
        parser.error(f"Target model not found: {args.target_model}")
    if not args.interface_layers.exists():
        parser.error(f"Interface layers file not found: {args.interface_layers}")
    
    # Load interface layers
    with open(args.interface_layers) as f:
        data = json.load(f)
        interface_layers = data.get("interface_layers", [])
    
    print(f"Interface layers: {interface_layers}")
    
    # Load models
    print("Loading base model...")
    base_model = load_model(args.base_model)
    
    print("Loading target model...")
    target_model = load_model(args.target_model)
    
    # Get number of layers
    num_layers = len(target_model.model.layers)
    print(f"Total layers: {num_layers}")
    
    # Calculate layer indices for each method
    semantic_hub_layers = get_semantic_hub_layers(num_layers, interface_layers)
    random_layers = select_random_layers(num_layers, len(semantic_hub_layers), args.seed)
    
    print(f"Semantic hub layers: {semantic_hub_layers}")
    print(f"Random layers: {random_layers}")
    
    # Save metadata
    metadata = {
        "base_model": str(args.base_model),
        "target_model": str(args.target_model),
        "interface_layers": interface_layers,
        "semantic_hub_layers": semantic_hub_layers,
        "random_layers": random_layers,
        "seed": args.seed
    }
    
    args.output.mkdir(parents=True, exist_ok=True)
    meta_file = args.output / "metadata.json"
    with open(meta_file, "w") as f:
        json.dump(metadata, f, indent=2)
    print(f"Saved metadata to {meta_file}")
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(str(args.base_model))
    
    # Perform swapping for each method
    method_map = {
        "semantic_hub": semantic_hub_layers,
        "random": random_layers
    }
    
    for method in args.methods:
        print(f"\n=== Processing {method} swap ===")
        layer_indices = method_map[method]
        
        # Reload target model for each swap
        swapped_model = load_model(args.target_model)
        swapped_model = swap_layers(base_model, swapped_model, layer_indices)
        
        # Save swapped model
        save_dir = args.output / method
        save_dir.mkdir(parents=True, exist_ok=True)
        swapped_model.save_pretrained(str(save_dir))
        tokenizer.save_pretrained(str(save_dir))
        print(f"Saved {method} model to {save_dir}")


if __name__ == "__main__":
    main()