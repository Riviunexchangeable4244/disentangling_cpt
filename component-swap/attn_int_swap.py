#!/usr/bin/env python3
import json
import argparse
from pathlib import Path
from transformers import AutoModelForCausalLM, AutoTokenizer


def load_model(model_path: Path):
    """Load a causal language model from the given path."""
    return AutoModelForCausalLM.from_pretrained(str(model_path))


def swap_attention_layers(base_model, cpt_model, interface_layers: list[int]):
    """
    Swap attention components (q_proj, k_proj, v_proj, o_proj)
    from base_model into cpt_model, only for given interface layers.
    """
    base_layers = base_model.model.layers
    cpt_layers = cpt_model.model.layers

    for layer_idx in interface_layers:
        cpt_layer = cpt_layers[layer_idx]
        base_layer = base_layers[layer_idx]

        for comp in ["q_proj", "k_proj", "v_proj", "o_proj"]:
            print(f"Swapping {comp} for layer {layer_idx}")
            getattr(cpt_layer.self_attn, comp).weight.data.copy_(
                getattr(base_layer.self_attn, comp).weight.data
            )
            
            if getattr(cpt_layer.self_attn, comp).bias is not None:
                getattr(cpt_layer.self_attn, comp).bias.data.copy_(
                    getattr(base_layer.self_attn, comp).bias.data
                )

        print(f"Completed swapping attention components for layer {layer_idx}")

    return cpt_model


def run(base_model_path: str, checkpoint_path: str, layer_selection_path: str, output_path: str):
    """
    Main execution function.
    
    Args:
        base_model_path: Path to the base model
        checkpoint_path: Path to the checkpoint directory containing checkpoint-* folders
        layer_selection_path: Path to the JSON file with interface_layers
        output_path: Directory where swapped models will be saved
    """
    base_model_path = Path(base_model_path)
    checkpoint_path = Path(checkpoint_path)
    layer_selection_path = Path(layer_selection_path)
    output_path = Path(output_path)
    
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Load base model and tokenizer
    print(f"Loading base model from: {base_model_path}")
    tokenizer = AutoTokenizer.from_pretrained(str(base_model_path))
    base_model = load_model(base_model_path)
    
    # Load layer selection
    if not layer_selection_path.exists():
        raise FileNotFoundError(f"Layer selection file not found: {layer_selection_path}")
    
    with open(layer_selection_path, "r") as f:
        selection = json.load(f)
    
    interface_layers = selection.get("interface_layers", [])
    if not interface_layers:
        raise ValueError(f"No interface_layers found in {layer_selection_path}")
    
    print(f"Interface layers to swap: {interface_layers}")
    
    # Find all checkpoint directories
    checkpoint_dirs = sorted(checkpoint_path.glob("checkpoint-*"))
    if not checkpoint_dirs:
        raise FileNotFoundError(f"No checkpoint-* folders found in {checkpoint_path}")
    
    # Process each checkpoint
    for ckpt_dir in checkpoint_dirs:
        print(f"\nProcessing checkpoint: {ckpt_dir.name}")
        cpt_model = load_model(ckpt_dir)
        
        # Swap only interface layers
        new_model = swap_attention_layers(base_model, cpt_model, interface_layers)
        
        # Save
        save_dir = output_path / ckpt_dir.name
        save_dir.mkdir(parents=True, exist_ok=True)
        new_model.save_pretrained(str(save_dir))
        tokenizer.save_pretrained(str(save_dir))
        print(f"Saved swapped model to {save_dir}")
    
    print("\nAll checkpoints processed successfully!")


def main():
    parser = argparse.ArgumentParser(
        description="Swap attention layers from a base model into checkpoint models based on interface layer selection."
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
        "--layer-selection",
        required=True,
        help="Path to JSON file containing interface_layers list"
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
        layer_selection_path=args.layer_selection,
        output_path=args.output_dir
    )


if __name__ == "__main__":
    main()