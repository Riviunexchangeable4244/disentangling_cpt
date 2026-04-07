#!/usr/bin/env python3
import json
import argparse
from pathlib import Path
from transformers import AutoModelForCausalLM, AutoTokenizer


def load_model(model_path: Path):
    """Load a causal language model from the given path."""
    return AutoModelForCausalLM.from_pretrained(str(model_path))


def swap_ffn_layers(base_model, cpt_model, layers_to_swap: list[int]):
    """
    Swap the entire FFN (mlp) weights for layers in layers_to_swap list.
    Copies gate_proj, up_proj, and down_proj from base_model to cpt_model.
    """
    base_layers = base_model.model.layers
    cpt_layers = cpt_model.model.layers

    for layer_idx in layers_to_swap:
        cpt_layers[layer_idx].mlp.gate_proj.weight.data.copy_(
            base_layers[layer_idx].mlp.gate_proj.weight.data
        )
        cpt_layers[layer_idx].mlp.up_proj.weight.data.copy_(
            base_layers[layer_idx].mlp.up_proj.weight.data
        )
        cpt_layers[layer_idx].mlp.down_proj.weight.data.copy_(
            base_layers[layer_idx].mlp.down_proj.weight.data
        )

        print(f"Swapped full FFN for layer {layer_idx}")

    return cpt_model


def run(base_model_path: str, checkpoint_path: str, layer_list_path: str, output_path: str):
    """
    Main execution function.
    
    Args:
        base_model_path: Path to the base model
        checkpoint_path: Path to the checkpoint directory containing checkpoint-* folders
        layer_list_path: Path to the JSON file containing list of layers to swap
        output_path: Directory where swapped models will be saved
    """
    base_model_path = Path(base_model_path)
    checkpoint_path = Path(checkpoint_path)
    layer_list_path = Path(layer_list_path)
    output_path = Path(output_path)
    
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Load base model and tokenizer
    print(f"Loading base model from: {base_model_path}")
    tokenizer = AutoTokenizer.from_pretrained(str(base_model_path))
    base_model = load_model(base_model_path)
    
    # Load layer list
    if not layer_list_path.exists():
        raise FileNotFoundError(f"Layer list file not found: {layer_list_path}")
    
    with open(layer_list_path, "r") as f:
        layers_to_swap = json.load(f)
    
    if not layers_to_swap:
        raise ValueError(f"No layers to swap found in {layer_list_path}")
    
    print(f"Layers to swap FFN: {layers_to_swap}")
    
    # Find all checkpoint directories
    checkpoint_dirs = sorted(checkpoint_path.glob("checkpoint-*"))
    if not checkpoint_dirs:
        raise FileNotFoundError(f"No checkpoint-* folders found in {checkpoint_path}")
    
    # Process each checkpoint
    for ckpt_dir in checkpoint_dirs:
        print(f"\nProcessing checkpoint: {ckpt_dir.name}")
        cpt_model = load_model(ckpt_dir)
        
        # Swap FFN layers
        new_model = swap_ffn_layers(base_model, cpt_model, layers_to_swap)
        
        # Save
        save_dir = output_path / ckpt_dir.name
        save_dir.mkdir(parents=True, exist_ok=True)
        new_model.save_pretrained(str(save_dir))
        tokenizer.save_pretrained(str(save_dir))
        print(f"Saved swapped model to {save_dir}")
    
    print("\nAll checkpoints processed successfully!")


def main():
    parser = argparse.ArgumentParser(
        description="Swap FFN (MLP) layers from a base model into checkpoint models based on layer list."
    )
    parser.add_argument(
        "--base-model",
        required=True,
        help="Path to the base model (e.g., meta-llama/Llama-2-7b-hf or local path)"
    )
    parser.add_argument(
        "--checkpoint-dir",
        required=True,
        help="Path to directory containing checkpoint-* folders"
    )
    parser.add_argument(
        "--layer-list",
        required=True,
        help="Path to JSON file containing list of layer indices to swap"
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        help="Directory where swapped models will be saved"
    )
    
    args = parser.parse_args()
    
    run(
        base_model_path=args.base_model,
        checkpoint_path=args.checkpoint_dir,
        layer_list_path=args.layer_list,
        output_path=args.output_dir
    )


if __name__ == "__main__":
    main()