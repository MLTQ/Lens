"""Collect non-verbal remainders on a large, fresh corpus for the dictionary-size sweep.

Same definition as scripts/train_atoms.py (canonical basis z = J_l h_l; remainder after NNLS
on the top-25 lens tokens; privileged coordinates from lenses/atoms/privileged.json zeroed),
but on ~1,600 new FineWeb documents that neither the lens fit nor the first dictionary saw,
split into train and held-out eval sets, written as fp16 memmaps plus token metadata so later
scripts can show contexts.

    python scripts/sweep_collect.py --train-docs 1400 --eval-docs 200
"""

import argparse
import json
import os
import sys
import time

import numpy as np
import torch
from safetensors.torch import load_file
from transformers import PreTrainedTokenizerFast

sys.path.insert(0, ".")
from bonsai_lens.ternary import dequantize, fwht, load_bonsai

p = argparse.ArgumentParser()
p.add_argument("--train-docs", type=int, default=1400)
p.add_argument("--eval-docs", type=int, default=200)
p.add_argument("--layers", default="16,24,32,40,48,56,62")
p.add_argument("--skip-rows", type=int, default=30000, help="stream past the rows the first corpus came from")
p.add_argument("--out", default="lenses/sweep")
args = p.parse_args()
os.makedirs(args.out, exist_ok=True)
dev = "cuda"
t0 = time.time()
log = lambda m: print(f"[{time.time() - t0:6.0f}s] {m}", flush=True)
LAYERS = [int(x) for x in args.layers.split(",")]
T_MAX, SKIP = 128, 16
tok = PreTrainedTokenizerFast(tokenizer_file="models/bonsai-meta/tokenizer.json")

# ---------------------------------------------------------------- corpus
corpus_path = f"{args.out}/corpus.jsonl"
n_docs = args.train_docs + args.eval_docs
if not os.path.exists(corpus_path):
    from datasets import load_dataset

    ds = load_dataset("HuggingFaceFW/fineweb", "sample-10BT", split="train", streaming=True).skip(args.skip_rows)
    docs = []
    for row in ds:
        ids = tok(row["text"]).input_ids
        if len(ids) >= T_MAX:
            docs.append(ids[:T_MAX])
        if len(docs) >= n_docs:
            break
    with open(corpus_path, "w") as f:
        for d in docs:
            f.write(json.dumps({"ids": d}) + "\n")
docs = [json.loads(l)["ids"] for l in open(corpus_path)]
log(f"corpus: {len(docs)} docs of {T_MAX} tokens")

# ---------------------------------------------------------------- model + lens
model, config, _ = load_bonsai("models/bonsai-gguf/Ternary-Bonsai-2-27B-PQ2_0.gguf",
                               "models/bonsai-meta/config.json", device=dev, verbose=False)
J = {int(k.split(".")[1]): v.to(dev) for k, v in load_file("lenses/bonsai27b-j.safetensors").items()
     if int(k.split(".")[1]) in LAYERS}
norm, head = model.model.norm, model.lm_head
gamma = 1.0 + norm.weight.float()
D = gamma.shape[0]
PRIV = [p["dim"] for p in json.load(open("lenses/atoms/privileged.json"))]


def token_dirs(ids):
    flat = ids.reshape(-1)
    w = dequantize(head.packed[flat], head.scales[flat], torch.float32)
    return (fwht(w, head.block, head.signs.float(), inverse=True) * gamma).reshape(*ids.shape, D)


def nnls(A, y, iters=5):
    G = A @ A.transpose(1, 2)
    b = (A @ y.unsqueeze(-1)).squeeze(-1)
    N, K = b.shape
    eye = torch.eye(K, device=A.device).expand(N, K, K)
    m = torch.ones(N, K, device=A.device)
    for _ in range(iters):
        mm = m.unsqueeze(-1) * m.unsqueeze(-2)
        reg = 1e-4 * eye * G.diagonal(dim1=1, dim2=2).mean(-1, keepdim=True).unsqueeze(-1)
        c = torch.linalg.solve(G * mm + eye * (1 - m).unsqueeze(-1) + reg, (b * m).unsqueeze(-1)).squeeze(-1)
        neg = (c < 0) & (m > 0)
        if not neg.any():
            break
        m = m * (~neg)
    return (c * m).clamp_min(0)


captured = {}
hooks = [model.model.layers[l].register_forward_hook(
    lambda m, i, o, l=l: captured.__setitem__(l, o[0] if isinstance(o, tuple) else o)) for l in LAYERS]
per_doc = len(LAYERS) * (T_MAX - 1 - SKIP)
splits = {"train": docs[: args.train_docs], "eval": docs[args.train_docs :]}
with torch.no_grad():
    for split, dl in splits.items():
        n = len(dl) * per_doc
        R = np.lib.format.open_memmap(f"{args.out}/{split}_R.npy", mode="w+", dtype=np.float16, shape=(n, D))
        meta = np.zeros((n, 3), np.int32)  # doc index (within split), position, layer
        i = 0
        for di, ids in enumerate(dl):
            x = torch.tensor([ids], device=dev)
            model.model(input_ids=x, use_cache=False)
            pos = torch.arange(SKIP, T_MAX - 1, device=dev)
            for l in LAYERS:
                z = captured[l][0, pos].float() @ J[l].float().T
                top = head(norm(z.to(torch.bfloat16))).float().topk(25, dim=-1).indices
                A = token_dirs(top)
                r = z - (nnls(A, z).unsqueeze(-1) * A).sum(1)
                r[:, PRIV] = 0
                k = len(pos)
                R[i : i + k] = r.half().cpu().numpy()
                meta[i : i + k, 0] = di; meta[i : i + k, 1] = pos.cpu().numpy(); meta[i : i + k, 2] = l
                i += k
            if di % 200 == 0:
                log(f"{split}: doc {di}/{len(dl)}")
        R.flush()
        np.save(f"{args.out}/{split}_meta.npy", meta)
        log(f"{split}: {n} remainder vectors")
json.dump({"layers": LAYERS, "train_docs": args.train_docs, "eval_docs": args.eval_docs, "skip": SKIP,
           "privileged": PRIV}, open(f"{args.out}/info.json", "w"))
log("done")
