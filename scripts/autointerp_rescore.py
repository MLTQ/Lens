"""Re-score atom descriptions with more held-out evidence (15 firing + 15 silent contexts).

The first pass used 5 + 5 contexts per atom; its control showed 17% of *mismatched*
descriptions still reaching AUROC >= 0.8, so single-atom scores are noisy. This re-scores the
lowest-scoring atoms (the xeno candidates) and a random comparison set on fresh contexts
(never shown to the describer, not used in the first score), and also scores each candidate
with a shuffled description as a per-atom baseline.

    python scripts/autointerp_rescore.py --low 50 --random 50
"""

import argparse
import json
import random
import sys
import time

import numpy as np
import torch
from transformers import PreTrainedTokenizerFast

sys.path.insert(0, ".")
from bonsai_lens.ternary import load_bonsai

p = argparse.ArgumentParser()
p.add_argument("--low", type=int, default=50)
p.add_argument("--random", type=int, default=50)
p.add_argument("--n", type=int, default=15)
args = p.parse_args()
dev = "cuda"
t0 = time.time()
log = lambda m: print(f"[{time.time() - t0:6.0f}s] {m}", flush=True)
random.seed(7)

tok = PreTrainedTokenizerFast(tokenizer_file="models/bonsai-meta/tokenizer.json")
tok.chat_template = open("models/bonsai-meta/chat_template.jinja").read()
YES, NO = tok.encode("Yes", add_special_tokens=False)[0], tok.encode("No", add_special_tokens=False)[0]
state = json.load(open("lenses/atoms/autointerp.json"))
scored = {int(k): v for k, v in state["atoms"].items() if v.get("auroc") is not None}

# ---------------------------------------------------------------- where atoms fire (as in autointerp.py)
sae = torch.load("lenses/atoms/sae.pt", map_location=dev)
W_enc, b_enc, b_dec = sae["W_enc"].float(), sae["b_enc"].float(), sae["b_dec"].float()
scale, K = sae["scale"], sae["k"]
NA = W_enc.shape[0]
corpus = [json.loads(l)["ids"] for l in open("lenses/sweep/corpus.jsonl")]
info = json.load(open("lenses/sweep/info.json"))
TOPN = 120
best_v = torch.full((NA, TOPN), -1.0, device=dev)
best_key = torch.zeros((NA, TOPN), dtype=torch.long, device=dev)
pool_k, pool_f = [], []
for split, doc_off in (("train", 0), ("eval", info["train_docs"])):
    R = np.load(f"lenses/sweep/{split}_R.npy", mmap_mode="r")
    M = np.load(f"lenses/sweep/{split}_meta.npy")
    for a in range(0, min(len(R), 700_000), 16384):
        x = torch.from_numpy(np.ascontiguousarray(R[a : a + 16384])).to(dev).float() * scale
        pre = torch.relu((x - b_dec) @ W_enc.T + b_enc)
        v, i = pre.topk(K, dim=-1)
        dense = torch.zeros_like(pre).scatter_(1, i, v)
        keys = torch.from_numpy((M[a : a + len(x), 0].astype(np.int64) + doc_off) * 1000 + M[a : a + len(x), 1]).to(dev)
        tv, ti = torch.cat([best_v, dense.T], 1).topk(TOPN, dim=1)
        best_key = torch.cat([best_key, keys.expand(NA, -1)], 1).gather(1, ti); best_v = tv
        pick = torch.randint(0, len(keys), (128,), device=dev)
        pool_k.append(keys[pick].cpu().numpy()); pool_f.append((dense[pick] > 0).cpu().numpy())
pool_k, pool_f = np.concatenate(pool_k), np.concatenate(pool_f)
log("firing contexts collected")


def context(key, width=28):
    d, p_ = divmod(int(key), 1000)
    ids = corpus[d]
    left = tok.decode(ids[max(0, p_ - width) : p_]).replace("\n", " ⏎ ")
    right = tok.decode(ids[p_ + 1 : p_ + 5]).replace("\n", " ⏎ ")
    return f"{left}[[{tok.decode([ids[p_]]).replace(chr(10), '⏎')}]]{right}"


model, _, _ = load_bonsai("models/bonsai-gguf/Ternary-Bonsai-2-27B-PQ2_0.gguf", "models/bonsai-meta/config.json",
                          device=dev, verbose=False)


@torch.no_grad()
def margin(desc, ctx):
    msg = (f'A hidden feature of a language model is described as: "{desc}"\n\nExcerpt: {ctx}\n\n'
           "Does this description fit the marked token [[…]] in this excerpt? Answer Yes or No.")
    text = tok.apply_chat_template([{"role": "user", "content": msg}], tokenize=False,
                                   add_generation_prompt=True, enable_thinking=False)
    ids = tok(text, return_tensors="pt", add_special_tokens=False).input_ids.to(dev)
    lg = model(input_ids=ids, use_cache=False).logits[0, -1].float()
    return float(lg[YES] - lg[NO])


def auroc(sp, sn):
    return float(np.mean([1.0 if a > b else 0.5 if a == b else 0.0 for a in sp for b in sn]))


low = sorted(scored, key=lambda a: scored[a]["auroc"])[: args.low]
rand = random.sample([a for a in scored if a not in set(low)], args.random)
out = {}
for group, ids in (("lowest", low), ("random", rand)):
    for j, a in enumerate(ids):
        rec = scored[a]
        used = set(rec["examples"] + rec["held_out"])  # contexts already shown or scored
        pos_keys = []
        for v, k in zip(best_v[a].tolist(), best_key[a].tolist()):
            c = context(k)
            if v > 0 and c not in used and k not in pos_keys:
                pos_keys.append(k)
        # skip the 12 strongest (what the describer saw), sample from the rest
        fresh = pos_keys[12:]
        if len(fresh) < args.n:
            out[a] = {"group": group, "note": "not enough fresh firing contexts"}
            continue
        pos = random.sample(fresh, args.n)
        silent = [int(k) for k, f in zip(pool_k, pool_f[:, a]) if not f and context(k) not in used]
        neg = random.sample(silent, args.n)
        sp = [margin(rec["desc"], context(k)) for k in pos]
        sn = [margin(rec["desc"], context(k)) for k in neg]
        other = scored[random.choice([o for o in scored if o != a])]["desc"]
        cp = [margin(other, context(k)) for k in pos]
        cn = [margin(other, context(k)) for k in neg]
        out[a] = {"group": group, "desc": rec["desc"], "first_auroc": rec["auroc"],
                  "auroc15": auroc(sp, sn), "shuffled15": auroc(cp, cn),
                  "pos": [context(k) for k in pos[:4]], "neg": [context(k) for k in neg[:2]]}
        if j % 10 == 0:
            log(f"{group} {j}/{len(ids)}: ξ{a} first {rec['auroc']:.2f} -> 15+15 {out[a]['auroc15']:.2f} (shuffled {out[a]['shuffled15']:.2f})")
json.dump(out, open("lenses/atoms/autointerp_rescore.json", "w"), ensure_ascii=False, indent=1)
for group in ("lowest", "random"):
    r = [v for v in out.values() if v.get("group") == group and "auroc15" in v]
    a15, s15 = np.array([v["auroc15"] for v in r]), np.array([v["shuffled15"] for v in r])
    log(f"{group}: n={len(r)} median 15+15 AUROC {np.median(a15):.2f} (shuffled {np.median(s15):.2f}); "
        f"< 0.65: {(a15 < 0.65).sum()}")
log("done")
