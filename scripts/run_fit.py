"""Fit the Jacobian lens on Ternary-Bonsai-2-27B over a pretraining-like corpus.

Resumable. The checkpoint holds the running *sum* of per-prompt Jacobians, so
any checkpoint is itself a usable lens (divide by n_done). Prompts are a
deterministic shuffle of FineWeb documents, first 128 tokens each, so every
prefix of the run is an unbiased sample.

    python scripts/run_fit.py --n-prompts 120 --out lenses/bonsai27b.ckpt
"""

import argparse
import json
import logging
import os
import random
import sys

import torch
from transformers import PreTrainedTokenizerFast

sys.path.insert(0, ".")
import jlens
from bonsai_lens.fit import fit_stream
from bonsai_lens.ternary import load_bonsai

p = argparse.ArgumentParser()
p.add_argument("--n-prompts", type=int, default=120)
p.add_argument("--seq-len", type=int, default=128)
p.add_argument("--dim-batch", type=int, default=32)
p.add_argument("--out", default="lenses/bonsai27b.ckpt")
p.add_argument("--corpus", default="lenses/corpus.jsonl")
args = p.parse_args()

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
os.makedirs(os.path.dirname(args.out), exist_ok=True)
tok = PreTrainedTokenizerFast(tokenizer_file="models/bonsai-meta/tokenizer.json")

if not os.path.exists(args.corpus):
    from datasets import load_dataset

    ds = load_dataset("HuggingFaceFW/fineweb", "sample-10BT", split="train", streaming=True)
    docs = []
    for row in ds:
        ids = tok(row["text"]).input_ids
        if len(ids) >= args.seq_len:
            docs.append(tok.decode(ids[: args.seq_len]))
        if len(docs) >= 4 * args.n_prompts:
            break
    random.Random(0).shuffle(docs)
    with open(args.corpus, "w") as f:
        for d in docs:
            f.write(json.dumps({"text": d}) + "\n")
texts = [json.loads(l)["text"] for l in open(args.corpus)]
logging.info("corpus: %d docs", len(texts))

model, config, _ = load_bonsai(
    "models/bonsai-gguf/Ternary-Bonsai-2-27B-PQ2_0.gguf", "models/bonsai-meta/config.json",
    verbose=False,
)
lm = jlens.from_hf(model, tok, force_bos=False)


def batches():
    for t in texts:
        ids = tok(t, return_tensors="pt").input_ids[:, : args.seq_len]
        yield ids.cuda()


n = config.num_hidden_layers
fit_stream(
    lm, batches(), source_layers=list(range(n - 1)), target_layer=n - 1,
    checkpoint_path=args.out, dim_batch=args.dim_batch, checkpoint_every=3,
    max_prompts=args.n_prompts,
)
logging.info("done")
