import argparse
import os
from types import MethodType

import torch
from vllm import LLM, SamplingParams

parser = argparse.ArgumentParser()
parser.add_argument("-m", "--model", type=str, default="meta-llama/Llama-2-7b-hf")
parser.add_argument("-i", "--id_path", type=str, required=True)
parser.add_argument("-t", "--type", type=str, default="llama")
parser.add_argument("-s", "--save_folder", type=str, required=True)
parser.add_argument("-n", "--name", type=str, required=True)
args = parser.parse_args()

if args.type == "llama":
    max_length = 4096
elif args.type == "qwen":
    max_length = 2048

model = LLM(
    model=args.model,
    tensor_parallel_size=torch.cuda.device_count(),
    enforce_eager=True,
    max_model_len=max_length,
)


num_layers = model.llm_engine.model_config.hf_config.num_hidden_layers
intermediate_size = model.llm_engine.model_config.hf_config.intermediate_size

over_zero = torch.zeros(num_layers, intermediate_size, dtype=torch.int32).to("cuda")


def extract_lang(id_path):
    parts = id_path.split("/")
    id = parts[-1]

    lang = id.split(".")[1]
    return lang


def factory(idx):
    def llama_forward(self, x):
        gate_up, _ = self.gate_up_proj(x)  # l, 2i
        i = gate_up.size(-1)
        gate_up[:, : i // 2] = torch.nn.SiLU()(gate_up[:, : i // 2])
        activation = gate_up[:, : i // 2].float()  # l, i
        over_zero[idx, :] += (activation > 0).sum(dim=0)
        x = gate_up[:, : i // 2] * gate_up[:, i // 2 :]
        x, _ = self.down_proj(x)
        return x

    def qwen_forward(self, x):
        gate_up, _ = self.gate_up_proj(x)  # (s, 2h)
        intermediate_size = gate_up.size(-1) // 2
        gate = gate_up[..., :intermediate_size]  # (s, h)
        up = gate_up[..., intermediate_size:]  # (s, h)
        gate_activation = torch.nn.functional.silu(gate)
        over_zero[idx, :] += (gate_activation > 0).sum(dim=0)
        x, _ = self.down_proj(gate_activation * up)
        return x

    if args.type == "llama":
        return llama_forward
    else:
        return qwen_forward


for i in range(num_layers):
    obj = model.llm_engine.model_executor.driver_worker.model_runner.model.model.layers[
        i
    ].mlp
    obj.forward = MethodType(factory(i), obj)


ids = torch.load(args.id_path)

l = ids.size(0)
l = min(l, 99999744) // max_length * max_length
input_ids = ids[:l].reshape(-1, max_length)

_ = model.generate(
    prompt_token_ids=input_ids.tolist(), sampling_params=SamplingParams(max_tokens=1)
)

output = dict(n=l, over_zero=over_zero.to("cpu"))


os.makedirs(args.save_folder, exist_ok=True)

save_path = os.path.join(
    args.save_folder, f"{args.name}.{extract_lang(args.id_path)}.pt"
)

torch.save(output, save_path)
