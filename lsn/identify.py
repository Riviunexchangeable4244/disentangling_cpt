#!/usr/bin/env python
import os
import sys
import torch
import json
import argparse
import numpy as np

torch.set_printoptions(profile="full")

FILTER_RATE = 0.95
TOP_RATE = 0.01
ACTIVATION_BAR_RATIO = 0.95

# ------------------------
# Args
# ------------------------
parser = argparse.ArgumentParser(description="Identify language-specific neurons")
parser.add_argument("-t", "--tag", required=True, type=str,
                    help="Model tag (e.g. l2-7b-eu)")
parser.add_argument("-l", "--langs", required=True, type=str,
                    help="Comma-separated languages (e.g. en,eu)")
parser.add_argument("-o", "--outdir", default="result", help="Output directory")
args = parser.parse_args()

base_path = "activation_count"
model_tag = args.tag
langs = args.langs.split(",")

# ------------------------
# Load data
# ------------------------
n, over_zero = [], []

for lang in langs:
    path = os.path.join(base_path, f"{model_tag}.{lang}.pt")
    if not os.path.exists(path):
        print(f"Error: File not found -> {path}", file=sys.stderr)
        sys.exit(1)

    data = torch.load(path)
    n.append(data["n"])
    over_zero.append(data["over_zero"])

# Convert to tensors
n = torch.Tensor(n)  # (lang_num)
over_zero = torch.stack(over_zero, dim=-1)  # (layer_num, neuron_num, lang_num)
num_layers, intermediate_size, lang_num = over_zero.size()

# ------------------------
# 1. Activation probability
# ------------------------
activation_probs = over_zero / n  # broadcast

# 2. Normalized activation probability
normed_activation_probs = activation_probs / activation_probs.sum(dim=-1, keepdim=True)

# 3. LAPE (entropy)
log_prop = torch.where(
    normed_activation_probs > 0,
    normed_activation_probs.log(),
    torch.zeros_like(normed_activation_probs),
)
entropy = -(normed_activation_probs * log_prop).sum(dim=-1)

# 4. Filter neurons using 95th percentile
flat_probs = activation_probs.flatten()
thresh = flat_probs.kthvalue(int(flat_probs.numel() * FILTER_RATE)).values
valid_mask = (activation_probs > thresh).any(dim=-1)  # [layers, neurons]
entropy[~valid_mask] = float("inf")

# 5. Select top-k neurons with lowest entropy
flat_entropy = entropy.flatten()
topk = int(flat_entropy.numel() * TOP_RATE)
_, idx = flat_entropy.topk(topk, largest=False)

layer_idx = idx // intermediate_size
neuron_idx = idx % intermediate_size

# 6. Group by languages
selection_props = activation_probs[layer_idx, neuron_idx]  # [topk, lang_num]
bar = flat_probs.kthvalue(int(flat_probs.numel() * ACTIVATION_BAR_RATIO)).values
lang_mask = (selection_props > bar).T  # [lang_num, topk]

final_mask = {}
for i, lang in enumerate(langs):
    neuron_ids = torch.where(lang_mask[i])[0]
    layer_dict = {l: [] for l in range(num_layers)}
    for nid in neuron_ids.tolist():
        l = layer_idx[nid].item()
        h = neuron_idx[nid].item()
        layer_dict[l].append(h)
    final_mask[lang] = layer_dict

# ------------------------
# Save as JSON
# ------------------------
langs_joined = ".".join(langs)
os.makedirs(args.outdir, exist_ok=True)
out_path = os.path.join(args.outdir, f"{model_tag}.json")

# Ensure ints, not tensors
final_mask_serializable = {
    lang: {int(k): [int(x) for x in v] for k, v in layer_dict.items()}
    for lang, layer_dict in final_mask.items()
}

with open(out_path, "w") as f:
    json.dump(final_mask_serializable, f, indent=2)

print(f"Saved language-specific neurons to {out_path}")