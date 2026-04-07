#!/usr/bin/env python
import argparse
import os
import json
from types import MethodType

import torch
from vllm import LLM, SamplingParams

parser = argparse.ArgumentParser(description="Track activations for masked neurons")
parser.add_argument("-m", "--model", type=str, required=True)
parser.add_argument("-i", "--id_path", type=str, required=True)
parser.add_argument("-t", "--type", type=str, default="llama")
parser.add_argument("-s", "--save_folder", type=str, required=True)
parser.add_argument("-n", "--name", type=str, required=True)
parser.add_argument("-f", "--mask_file", type=str, required=True)
parser.add_argument("-l", "--lang", type=str, required=True)
args = parser.parse_args()

lang = args.lang

with open(args.mask_file, "r") as f:
    mask_all = json.load(f)
if lang not in mask_all:
    raise ValueError(f"Language {lang} not found in mask file")
mask = mask_all[lang]

if args.type == "llama":
    max_length = 4096
elif args.type == "qwen":
    max_length = 2048
else:
    raise ValueError(f"Unknown model type {args.type}")

device_count = torch.cuda.device_count()
model = LLM(
    model=args.model,
    tensor_parallel_size=device_count,
    enforce_eager=True,
    max_model_len=max_length,
)
num_layers = model.llm_engine.model_config.hf_config.num_hidden_layers

sums = {int(l): {neuron: 0.0 for neuron in mask[l]} for l in mask}
counts = {int(l): {neuron: 0 for neuron in mask[l]} for l in mask}


def factory(idx):
    neuron_idx = torch.tensor(mask[str(idx)]).to("cuda")

    def llama_forward(self, x):
        gate_up, _ = self.gate_up_proj(x)
        i = gate_up.size(-1)
        gate_up[:, : i // 2] = torch.nn.SiLU()(gate_up[:, : i // 2])
        activation = gate_up[:, : i // 2].float()

        mean_activations = activation[:, neuron_idx].mean(dim=0)
        for n, val in zip(mask[str(idx)], mean_activations.tolist()):
            sums[idx][n] += val
            counts[idx][n] += 1

        x = gate_up[:, : i // 2] * gate_up[:, i // 2 :]
        x, _ = self.down_proj(x)
        return x

    def qwen_forward(self, x):
        gate_up, _ = self.gate_up_proj(x)
        h = gate_up.size(-1) // 2
        gate = gate_up[:, :h]
        up = gate_up[:, h:]
        gate_activation = torch.nn.functional.silu(gate)

        mean_activations = gate_activation[:, neuron_idx].mean(dim=0)
        for n, val in zip(mask[str(idx)], mean_activations.tolist()):
            sums[idx][n] += val
            counts[idx][n] += 1

        x, _ = self.down_proj(gate_activation * up)
        return x

    return llama_forward if args.type == "llama" else qwen_forward


for i in range(num_layers):
    if mask[str(i)]:
        mlp = model.llm_engine.model_executor.driver_worker.model_runner.model.model.layers[
            i
        ].mlp
        mlp.forward = MethodType(factory(i), mlp)

ids = torch.load(args.id_path)
l = min(ids.size(0), 100_000)
l = (l // max_length) * max_length
input_ids = ids[:l].reshape(-1, max_length)

_ = model.generate(
    prompt_token_ids=input_ids.tolist(), sampling_params=SamplingParams(max_tokens=1)
)

total_count = sum(counts[layer][neuron] for layer in counts for neuron in counts[layer])
final_activation = (
    sum(sums[layer][neuron] / total_count for layer in sums for neuron in sums[layer])
    if total_count > 0
    else 0.0
)

os.makedirs(args.save_folder, exist_ok=True)
save_path = os.path.join(args.save_folder, f"{args.name}.{lang}.{l}.json")
with open(save_path, "w") as f:
    json.dump({"all_layers": final_activation}, f, indent=2)

print(f"Saved masked activations for {lang} to {save_path}")