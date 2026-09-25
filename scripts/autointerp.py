"""Let Bonsai describe its own non-verbal atoms, then test whether the descriptions are true.

For each atom of the viewer's dictionary (lenses/atoms/sae.pt):
  1. find where it fires on the fresh sweep corpus (lenses/sweep; never used to train it),
  2. show Bonsai its 12 strongest contexts and ask for a <=12-word description,
  3. score that description on HELD-OUT contexts: 5 more places the atom fires and 5 where it
     is silent. Bonsai judges "does the description fit the marked token?" and we take
     logit(Yes) - logit(No); the score is the AUROC over the 25 (positive, negative) pairs.
     1.0 = the description picks out exactly where the atom fires; 0.5 = chance.
Low-scoring atoms (stable, frequent enough to score, but not describable by the model in
words) are the xeno-representation candidates in the paper's sense.

Also names the token-map regions from lenses/joint/quality/regions.json.

    python scripts/autointerp.py            # resumable; writes lenses/atoms/autointerp.json
"""

import json
import os
import random
import sys
import time

import numpy as np
import torch
from transformers import PreTrainedTokenizerFast

sys.path.insert(0, ".")
from bonsai_lens.ternary import load_bonsai

dev = "cuda"
t0 = time.time()
log = lambda m: print(f"[{time.time() - t0:6.0f}s] {m}", flush=True)
OUT = "lenses/atoms/autointerp.json"
random.seed(0)

tok = PreTrainedTokenizerFast(tokenizer_file="models/bonsai-meta/tokenizer.json")
tok.chat_template = open("models/bonsai-meta/chat_template.jinja").read()
YES, NO = tok.encode("Yes", add_special_tokens=False)[0], tok.encode("No", add_special_tokens=False)[0]
EOS = [248046, 248044]

# ---------------------------------------------------------------- where each atom fires
sae = torch.load("lenses/atoms/sae.pt", map_location=dev)
W_enc, b_enc, W_dec, b_dec = (sae[k].float() for k in ("W_enc", "b_enc", "W_dec", "b_dec"))
scale, K = sae["scale"], sae["k"]
NA = W_enc.shape[0]
corpus = [json.loads(l)["ids"] for l in open("lenses/sweep/corpus.jsonl")]
info = json.load(open("lenses/sweep/info.json"))
TOPN = 60
best_v = torch.full((NA, TOPN), -1.0, device=dev)
best_key = torch.zeros((NA, TOPN), dtype=torch.long, device=dev)  # doc * 1000 + pos
fired = torch.zeros(NA, device=dev)
n_seen = 0
silent_pool = []  # (key) of sampled contexts, for negatives
for split, doc_off in (("train", 0), ("eval", info["train_docs"])):
    R = np.load(f"lenses/sweep/{split}_R.npy", mmap_mode="r")
    M = np.load(f"lenses/sweep/{split}_meta.npy")
    lim = min(len(R), 700_000)  # plenty of evidence per atom; keeps the pass short
    for a in range(0, lim, 16384):
        x = torch.from_numpy(np.ascontiguousarray(R[a : a + 16384])).to(dev).float() * scale
        pre = torch.relu((x - b_dec) @ W_enc.T + b_enc)
        v, i = pre.topk(K, dim=-1)
        dense = torch.zeros_like(pre).scatter_(1, i, v)
        fired += (dense > 0).float().sum(0)
        keys = torch.from_numpy((M[a : a + len(x), 0].astype(np.int64) + doc_off) * 1000 + M[a : a + len(x), 1]).to(dev)
        cat_v = torch.cat([best_v, dense.T], 1)
        cat_k = torch.cat([best_key, keys.expand(NA, -1)], 1)
        tv, ti = cat_v.topk(TOPN, dim=1)
        best_v, best_key = tv, cat_k.gather(1, ti)
        silent_pool.append((keys[torch.randint(0, len(keys), (64,), device=dev)].cpu().numpy(),
                            (dense[torch.randint(0, len(keys), (64,), device=dev)] > 0).cpu().numpy()))
        n_seen += len(x)
freq = (fired / n_seen).cpu().numpy()
log(f"encoded {n_seen} remainder vectors")
pool_keys = np.concatenate([k for k, _ in silent_pool])
pool_act = np.concatenate([f for _, f in silent_pool])  # [pool, NA] fired or not (for that sample's layer)


def context(key, width=28):
    d, p = divmod(int(key), 1000)
    ids = corpus[d]
    left = tok.decode(ids[max(0, p - width) : p]).replace("\n", " ⏎ ")
    right = tok.decode(ids[p + 1 : p + 5]).replace("\n", " ⏎ ")
    return f"{left}[[{tok.decode([ids[p]]).replace(chr(10), '⏎')}]]{right}"


def positives(a):
    seen, out = set(), []
    for v, k in zip(best_v[a].tolist(), best_key[a].tolist()):
        if v <= 0 or k in seen:
            continue
        seen.add(k); out.append(k)
    return out  # distinct contexts, strongest first


# ---------------------------------------------------------------- model
model, config, _ = load_bonsai("models/bonsai-gguf/Ternary-Bonsai-2-27B-PQ2_0.gguf",
                               "models/bonsai-meta/config.json", device=dev, verbose=False)
model.generation_config.eos_token_id = EOS
model.generation_config.pad_token_id = EOS[0]


def chat(msg):
    return tok.apply_chat_template([{"role": "user", "content": msg}], tokenize=False,
                                   add_generation_prompt=True, enable_thinking=False)


@torch.no_grad()
def generate(msg, n=40):
    ids = tok(chat(msg), return_tensors="pt", add_special_tokens=False).input_ids.to(dev)
    out = model.generate(ids, max_new_tokens=n, do_sample=False)
    return tok.decode(out[0, ids.shape[1] :], skip_special_tokens=True).strip().split("\n")[0].strip()


@torch.no_grad()
def yes_margin(msg):
    ids = tok(chat(msg), return_tensors="pt", add_special_tokens=False).input_ids.to(dev)
    lg = model(input_ids=ids, use_cache=False).logits[0, -1].float()
    return float(lg[YES] - lg[NO])


# sanity check that generation works before the long loop
log("smoke test: " + generate("Name the capital of Italy in one word."))

state = json.load(open(OUT)) if os.path.exists(OUT) else {"atoms": {}, "regions": {}}


def save():
    json.dump(state, open(OUT + ".tmp", "w"), ensure_ascii=False)
    os.replace(OUT + ".tmp", OUT)


# ---------------------------------------------------------------- regions of the token map
rj = json.load(open("lenses/joint/quality/regions.json"))
vocab = json.load(open("lenses/space/meta.json"))["tokens"]
for r in rj["regions"]:
    if str(r["id"]) in state["regions"]:
        continue
    sample = ", ".join(json.dumps(vocab[i], ensure_ascii=False) for i in r["top"][:40])
    name = generate("Here are tokens from one group in a language model's vocabulary (they may be word pieces, "
                    f"in any language):\n{sample}\n\nGive this group a short name of 1 to 4 English words "
                    "describing what they have in common. Reply with the name only.", n=12)
    state["regions"][str(r["id"])] = name
log(f"named {len(state['regions'])} regions")
save()

# ---------------------------------------------------------------- atoms
DESCRIBE, HELD, NEG = 12, 5, 5
order = list(range(NA))
random.shuffle(order)  # any prefix of the run is a random sample
for c, a in enumerate(order):
    if str(a) in state["atoms"]:
        continue
    pos = positives(a)
    rec = {"freq": float(freq[a]), "n_contexts": len(pos)}
    if len(pos) < DESCRIBE + HELD:
        rec["desc"] = None
        rec["note"] = "too rare in the sweep corpus to describe and score"
        state["atoms"][str(a)] = rec
        continue
    ex = pos[:DESCRIBE]
    held = random.sample(pos[DESCRIBE:], HELD)
    silent = [int(k) for k, act in zip(pool_keys, pool_act[:, a]) if not act and int(k) not in set(pos)]
    neg = random.sample(silent, NEG)
    listing = "\n".join(f"{i + 1}. {context(k)}" for i, k in enumerate(ex))
    desc = generate("Below are excerpts from web text. In each one, a single token is marked like [[this]]. "
                    "A hidden feature inside a language model responds strongly to the marked token, in its "
                    f"context.\n\n{listing}\n\nIn at most 12 words, describe what the marked tokens have in common: "
                    "their meaning, their grammatical role, or the situation around them. Reply with the description only.")
    judge = lambda k: yes_margin(f'A hidden feature of a language model is described as: "{desc}"\n\n'
                                 f"Excerpt: {context(k)}\n\nDoes this description fit the marked token [[…]] "
                                 "in this excerpt? Answer Yes or No.")
    sp, sn = [judge(k) for k in held], [judge(k) for k in neg]
    auroc = float(np.mean([1.0 if p > n else 0.5 if p == n else 0.0 for p in sp for n in sn]))
    rec.update({"desc": desc, "auroc": auroc, "pos_margin": float(np.mean(sp)), "neg_margin": float(np.mean(sn)),
                "examples": [context(k) for k in ex[:4]], "held_out": [context(k) for k in held],
                "negatives": [context(k) for k in neg]})
    state["atoms"][str(a)] = rec
    if c % 25 == 0:
        done = [v for v in state["atoms"].values() if v.get("auroc") is not None]
        log(f"{len(state['atoms'])}/{NA} atoms; scored {len(done)}, median AUROC {np.median([v['auroc'] for v in done]):.2f}; "
            f"latest ξ{a}: {desc!r} ({auroc:.2f})")
        save()
save()
log("done")
