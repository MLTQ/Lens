"""Check batched (vmapped) Jacobian rows == per-row backprop, then time a few batch sizes."""

import sys
import time

import torch
from transformers import PreTrainedTokenizerFast

sys.path.insert(0, ".")
import jlens
from bonsai_lens.fit import jacobian_rows_for_ids
from bonsai_lens.ternary import load_bonsai

GGUF = "models/bonsai-gguf/Ternary-Bonsai-2-27B-PQ2_0.gguf"
META = "models/bonsai-meta"
model, config, _ = load_bonsai(GGUF, f"{META}/config.json", device="cuda:0", verbose=False)
tok = PreTrainedTokenizerFast(tokenizer_file=f"{META}/tokenizer.json")
lm = jlens.from_hf(model, tok, force_bos=False)
print(lm)

text = open("vendor-jacobian-lens/README.md").read()[:4000]
ids = tok(text, return_tensors="pt").input_ids[:, :int(__import__("os").environ.get("T", 128))].cuda()
src, tgt = [0, 20, 40, 62], 63
d = lm.d_model

a = {l: torch.zeros(d, d) for l in src}
b = {l: torch.zeros(d, d) for l in src}
jacobian_rows_for_ids(lm, ids, src, tgt, a, dim_batch=4, batched=True, max_dims=4)
jacobian_rows_for_ids(lm, ids, src, tgt, b, dim_batch=4, batched=False, max_dims=4)
for l in src:
    x, y = a[l][:4], b[l][:4]
    print(f"L{l}: rel err {(x - y).norm() / y.norm():.2e}  |row|={y.norm(dim=1).mean():.3f}  diag={torch.diagonal(y[:, :4]).tolist()}")

src = list(range(63))
for B in [int(x) for x in sys.argv[1:]] or [16, 32]:
    acc = {l: torch.zeros(d, d) for l in src}
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize(); t0 = time.time()
    jacobian_rows_for_ids(lm, ids, src, tgt, acc, dim_batch=B, max_dims=4 * B)
    torch.cuda.synchronize(); dt = time.time() - t0
    per_row = dt / (4 * B)
    print(f"B={B}: {dt:.1f}s for {4*B} rows -> {per_row*1e3:.0f} ms/row -> "
          f"{per_row * d / 60:.1f} min/prompt; peak {torch.cuda.max_memory_allocated()/1e9:.2f} GB")
