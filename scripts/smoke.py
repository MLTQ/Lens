"""Load the ternary model and sanity-check it: loss on plain text + greedy generation."""

import sys
import time

import torch
from transformers import PreTrainedTokenizerFast

sys.path.insert(0, ".")
from bonsai_lens.ternary import load_bonsai

GGUF = "models/bonsai-gguf/Ternary-Bonsai-2-27B-PQ2_0.gguf"
META = "models/bonsai-meta"

t0 = time.time()
model, config, _ = load_bonsai(GGUF, f"{META}/config.json", device="cuda:0")
print(f"loaded in {time.time() - t0:.0f}s; cuda mem {torch.cuda.memory_allocated() / 1e9:.2f} GB")
tok = PreTrainedTokenizerFast(tokenizer_file=f"{META}/tokenizer.json")

text = (
    "The Eiffel Tower is a wrought-iron lattice tower on the Champ de Mars in Paris, France. "
    "It is named after the engineer Gustave Eiffel, whose company designed and built the tower "
    "from 1887 to 1889. Locally nicknamed \"La dame de fer\" (French for \"Iron Lady\"), it was "
    "constructed as the centerpiece of the 1889 World's Fair."
)
ids = tok(text, return_tensors="pt").input_ids.cuda()
with torch.no_grad():
    t0 = time.time()
    out = model(input_ids=ids, labels=ids, use_cache=False)
    torch.cuda.synchronize()
print(f"loss {out.loss.item():.3f} nats/token over {ids.shape[1]} tokens ({time.time() - t0:.2f}s)")

prompt = "Fact: The capital of the country shaped like a boot is"
ids = tok(prompt, return_tensors="pt").input_ids.cuda()
with torch.no_grad():
    for _ in range(12):
        nxt = model(input_ids=ids, use_cache=False).logits[0, -1].argmax()
        ids = torch.cat([ids, nxt.view(1, 1)], 1)
print(repr(tok.decode(ids[0])))
print(f"peak mem {torch.cuda.max_memory_allocated() / 1e9:.2f} GB")
