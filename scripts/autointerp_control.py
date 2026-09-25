"""Control for the describability score of scripts/autointerp.py.

For a sample of scored atoms, re-score each atom's held-out contexts (the same 5 firing and
5 silent contexts) with a DIFFERENT atom's description. If the scoring is sound, the AUROC
should collapse to ~0.5; if it stays high, the judge is flattering any description and the
real scores mean little. Also re-scores a few atoms with their own description to check
the scores are reproducible.

    python scripts/autointerp_control.py --n 120
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
p.add_argument("--n", type=int, default=120)
args = p.parse_args()
dev = "cuda"
t0 = time.time()
log = lambda m: print(f"[{time.time() - t0:6.0f}s] {m}", flush=True)
random.seed(1)

tok = PreTrainedTokenizerFast(tokenizer_file="models/bonsai-meta/tokenizer.json")
tok.chat_template = open("models/bonsai-meta/chat_template.jinja").read()
YES, NO = tok.encode("Yes", add_special_tokens=False)[0], tok.encode("No", add_special_tokens=False)[0]
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


def auroc(desc, rec):
    sp = [margin(desc, c) for c in rec["held_out"]]
    sn = [margin(desc, c) for c in rec["negatives"]]
    return float(np.mean([1.0 if a > b else 0.5 if a == b else 0.0 for a in sp for b in sn]))


state = json.load(open("lenses/atoms/autointerp.json"))
scored = {k: v for k, v in state["atoms"].items() if v.get("auroc") is not None}
keys = random.sample(sorted(scored), min(args.n, len(scored)))
shuffled, own = [], []
for j, k in enumerate(keys):
    other = random.choice([o for o in scored if o != k and scored[o]["desc"] != scored[k]["desc"]])
    shuffled.append(auroc(scored[other]["desc"], scored[k]))
    if j < 20:
        own.append((scored[k]["auroc"], auroc(scored[k]["desc"], scored[k])))
    if j % 20 == 0:
        log(f"{j}/{len(keys)}: shuffled-description AUROC median so far {np.median(shuffled):.2f}")
real = [scored[k]["auroc"] for k in keys]
out = {"n": len(keys), "real_median": float(np.median(real)), "shuffled_median": float(np.median(shuffled)),
       "real_mean": float(np.mean(real)), "shuffled_mean": float(np.mean(shuffled)),
       "real_ge_0.8": float(np.mean(np.array(real) >= 0.8)), "shuffled_ge_0.8": float(np.mean(np.array(shuffled) >= 0.8)),
       "rescore_pairs": own}
json.dump(out, open("lenses/atoms/autointerp_control.json", "w"), indent=1)
log(json.dumps({k: v for k, v in out.items() if k != "rescore_pairs"}))
log(f"re-score reproducibility (first 20): max |diff| {max(abs(a - b) for a, b in own):.3f}")
