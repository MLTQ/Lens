"""Reference residuals from the torch ternary model (CPU) for cross-checking MLX."""
import sys, numpy as np, torch
sys.path.insert(0, ".")
from transformers import PreTrainedTokenizerFast
from bonsai_lens.ternary import load_bonsai
torch.set_num_threads(10)
tok = PreTrainedTokenizerFast(tokenizer_file="models/bonsai-meta/tokenizer.json")
model, _, _ = load_bonsai("models/bonsai-gguf/Ternary-Bonsai-2-27B-PQ2_0.gguf", "models/bonsai-meta/config.json", device="cpu", verbose=False)
text = "Fact: The currency used in the country shaped like a boot is"
ids = tok(text, return_tensors="pt").input_ids
hs = {}
hooks = [model.model.layers[l].register_forward_hook(lambda m, i, o, l=l: hs.__setitem__(l, o[0] if isinstance(o, tuple) else o)) for l in (0, 10, 31, 62, 63)]
with torch.no_grad():
    logits = model(input_ids=ids, use_cache=False).logits
np.savez("runs/ref.npz", ids=ids.numpy(), logits_last=logits[0, -1].float().numpy(), **{f"h{l}": v[0].float().numpy() for l, v in hs.items()})
print("ids", ids.tolist(), "top5", [tok.decode([i]) for i in logits[0, -1].topk(5).indices])
