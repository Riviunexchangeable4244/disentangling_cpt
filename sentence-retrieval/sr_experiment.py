"""
Sentence retrieval task accuracy on FLORES-PRO dataset with plotting

Ref 1: https://aclanthology.org/2024.loresmt-1.20.pdf
Ref 2: https://arxiv.org/pdf/1811.01136
"""

import argparse
from datetime import datetime
import json
from pathlib import Path

import torch
from torch import Tensor
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
import matplotlib.pyplot as plt


# -------------------------- Model & Dataset --------------------------
def load_model(model_path: str, device_map="auto") -> AutoModelForCausalLM:
    """Load a model"""
    print(f"Loading model from {model_path}")
    return AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        device_map=device_map,
    )

def load_sla_word(lang: str) -> tuple[list[str], list[str]]:
    """
    Load words from the SLA dataset for a specific language and their English equivalents.

    Args:
        lang (str): ISO language code (e.g., "gle" for Irish, "eus" for Basque, "cmn" for Mandarin Chinese).

    Returns:
        Tuple[List[str], List[str]]:
            - List of words in the target language (specified by `lang`).
            - List of corresponding English words.
    
    Example:
        foreign, english = load_sla_word("gle")
        print(foreign[:5])
        ['agus', 'ar', 'bain', 'bean', 'beidh']
        print(english[:5])
        ['and', 'on', 'use', 'woman', 'will be']
    """
    SLA_WORD = "tvkain/sla"  # Hugging Face dataset repo path
    # Load the dataset (train split)
    ds = load_dataset(SLA_WORD, split="train")
    
    # Filter the dataset for the given language
    filtered_ds = ds.filter(lambda x: x["lang"] == lang)
    
    # Extract the foreign language words and their English equivalents
    foreign_words = [entry[lang] for entry in filtered_ds]
    english_words = [entry["eng"] for entry in filtered_ds]
    
    return foreign_words, english_words



def load_flores(dataset: str) -> list[str]:
    FLORES = "openlanguagedata/flores_plus"
    ds = load_dataset(FLORES, dataset)["dev"]
    return [d["text"] for d in ds]


# -------------------------- Utilities --------------------------
def cosine_similarity(first: Tensor, second: Tensor) -> Tensor:
    first_norm = first / torch.linalg.norm(first, dim=1, keepdim=True)
    second_norm = second / torch.linalg.norm(second, dim=1, keepdim=True)
    return first_norm @ second_norm.T


def margin_based_scoring(first: Tensor, second: Tensor, k=4, variant="ratio", eps=1e-6) -> Tensor:
    a = cosine_similarity(first, second)
    if variant == "absolute":
        return a
    nn_k_row = torch.topk(a, k=min(k, a.size(1)), dim=1).values
    nn_k_col = torch.topk(a, k=min(k, a.size(0)), dim=0).values
    row_mean = nn_k_row.mean(dim=1, keepdim=True)
    col_mean = nn_k_col.mean(dim=0, keepdim=True)
    b = (row_mean + col_mean) * 0.5 + eps
    return a - b if variant == "distance" else a / b


def layer_accuracy(score: Tensor) -> float:
    device = score.device
    preds = score.argmax(dim=1)
    return (preds == torch.arange(score.size(0), device=device)).float().mean().item()


@torch.no_grad()
def layer_sentence_representation(model, tokenizer, sentences: list[str], batch_size=32, device=None) -> dict[int, Tensor]:
    temp: dict[int, list[Tensor]] = {}
    for start in range(0, len(sentences), batch_size):
        batch = sentences[start:start + batch_size]
        tokens = tokenizer(batch, return_tensors="pt", padding=True)
        tokens = {k: v.to(device) for k, v in tokens.items()}
        hidden_states = model(**tokens, output_hidden_states=True, use_cache=False).hidden_states
        for i, hs in enumerate(hidden_states):
            mean_hs = mean_hidden_state(hs, tokens["attention_mask"])
            temp.setdefault(i, []).append(mean_hs)
        del hidden_states, tokens
        torch.cuda.empty_cache()
    return {i: torch.cat(temp[i], dim=0) for i in temp}


def mean_hidden_state(hidden_states: Tensor, attention_mask: Tensor) -> Tensor:
    mask = attention_mask.unsqueeze(-1).to(hidden_states.dtype)
    summed = (hidden_states * mask).sum(dim=1)
    count = mask.sum(dim=1)
    return summed / count


def generate_json_artifact(model_path: str, layer_acc: dict[int, float], base_lang: str, target_lang: str, save_dir: str):
    artifact = {
        "model_path": model_path,
        "base_lang": base_lang,
        "target_lang": target_lang,
        "timestamp": datetime.now().isoformat(),
        "layer_accuracy": sorted(layer_acc.items()),
    }
    Path(save_dir).mkdir(exist_ok=True, parents=True)
    checkpoint = Path(model_path).name if Path(model_path).exists() else "hf_model"
    timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
    json_path = Path(save_dir) / f"{timestamp_str}_{checkpoint}.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(artifact, f, indent=4, ensure_ascii=False)
    print(f"Artifact saved to {json_path}")
    return json_path


# -------------------------- Plotting --------------------------
def plot_layer_accuracy(json_files: list[str], save_folder: str):
    plt.figure(figsize=(10, 5))  # wider figure

    def checkpoint_sort_key(jf_path):
        data = json.load(open(jf_path, "r", encoding="utf-8"))
        checkpoint_name = Path(data["model_path"]).name
        try:
            return int(checkpoint_name.split("-")[-1])
        except ValueError:
            return 0

    json_files = sorted(json_files, key=checkpoint_sort_key)

    for jf in json_files:
        data = json.load(open(jf, "r", encoding="utf-8"))
        layer_acc_pairs = data.get("layer_accuracy", [])
        layer_acc_pairs.sort(key=lambda x: x[0])
        layers = [p[0] for p in layer_acc_pairs]
        acc = [p[1] for p in layer_acc_pairs]
        model_name = Path(data["model_path"]).parent.name
        checkpoint = Path(data["model_path"]).name
        base_lang = data.get("base_lang", "unknown")
        target_lang = data.get("target_lang", "unknown")
        label = f"{model_name}/{checkpoint} ({base_lang}-{target_lang})"
        plt.plot(layers, acc, marker="o", label=label)

    plt.xlabel("Layer")
    plt.ylabel("Accuracy")
    plt.title("Sentence Retrieval Accuracy per Layer")
    plt.grid(True)
    plt.legend(loc="center left", bbox_to_anchor=(1, 0.5))
    plt.subplots_adjust(right=0.75)

    Path(save_folder).mkdir(exist_ok=True, parents=True)
    timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
    plot_path = Path(save_folder) / f"{timestamp_str}_plot.png"
    plt.savefig(plot_path, bbox_inches="tight")
    plt.close()
    print(f"Plot saved to {plot_path}")
    return plot_path


# -------------------------- Main --------------------------
def main():
    parser = argparse.ArgumentParser(description="Sentence retrieval task with plotting")
    parser.add_argument("--model_path", type=str, required=True, help="Folder with checkpoints or Hugging Face model ID")
    parser.add_argument("--base", type=str, required=True, help="Base language ISO code")
    parser.add_argument("--target", type=str, required=True, help="Target language ISO code")
    parser.add_argument("--device", type=str, default="auto", help="Device to use (cuda or cpu)")
    parser.add_argument("--artifact-out", type=str, default="artifacts", help="Folder to save JSON artifacts and plot")
    parser.add_argument("--margin-variant", type=str, default="ratio", help="Margin variant for scoring")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() and args.device != "cpu" else "cpu")
    print(f"Using device: {device}")

    print(f"Loading {args.base} and {args.target} FLORES datasets...")
    base_data = load_flores(args.base)
    target_data = load_flores(args.target)

    model_path = Path(args.model_path)
    if model_path.exists() and model_path.is_dir():
        checkpoint_dirs = sorted(
            model_path.glob("checkpoint-*"),
            key=lambda x: int(x.name.split("-")[-1]) if "-" in x.name else 0
        )
        if not checkpoint_dirs:
            checkpoint_dirs = [model_path]
    else:
        checkpoint_dirs = [args.model_path]

    json_files = []
    for ckpt in checkpoint_dirs:
        print(f"\n=== Processing checkpoint/model: {ckpt} ===")
        model = load_model(str(ckpt))
        model.to(device)
        model.eval()
        tokenizer = AutoTokenizer.from_pretrained(str(ckpt))
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token or tokenizer.unk_token

        print(f"Computing layer embeddings for {args.base}...")
        base_ret = layer_sentence_representation(model, tokenizer, base_data, device=device)
        print(f"Computing layer embeddings for {args.target}...")
        target_ret = layer_sentence_representation(model, tokenizer, target_data, device=device)

        print("Computing layer accuracy...")
        layer_acc = {i: layer_accuracy(margin_based_scoring(base_ret[i], target_ret[i], variant=args.margin_variant))
                     for i in base_ret.keys()}

        json_file = generate_json_artifact(str(ckpt), layer_acc, args.base, args.target, args.artifact_out)
        json_files.append(json_file)

    # Generate plot
    plot_layer_accuracy(json_files, args.artifact_out)


if __name__ == "__main__":
    main()