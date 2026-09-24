"""Turn a fitting checkpoint (running sum) into an fp16 safetensors lens: J.{layer}."""

import sys

import torch
from safetensors.torch import save_file

ckpt, out = sys.argv[1], sys.argv[2]
state = torch.load(ckpt, map_location="cpu", weights_only=True)
n = state["n_done"]
tensors = {f"J.{l}": (s / n).to(torch.float16).contiguous() for l, s in state["jacobian_sum"].items()}
save_file(tensors, out, metadata={"n_prompts": str(n), "target_layer": str(state["target_layer"]),
                                  "estimator": "jlens: cotangent summed over targets t'>=t, mean over sources"})
print(f"wrote {out}: {len(tensors)} layers, n_prompts={n}")
