"""Non-verbal atoms: a dictionary for what the J-lens cannot read.

Everything happens in the canonical (final-layer) basis the lens decodes in, so the atoms
can later be embedded together with the token readout directions u_v = gamma * W_eff[v]:

  1. z = J_l h_l                           transport each activation (held-out FineWeb docs)
  2. verbal = argmin_{c >= 0} ||z - sum_v c_v u_v||   over the top-k lens tokens (NNLS)
  3. r = z - verbal                        the non-verbal remainder
  4. privileged coordinates (massive dims such as d3994) are measured, named, and zeroed in r
  5. a TopK sparse autoencoder learns atoms (unit decoder directions) for r, pooled over layers

Outputs (lenses/atoms/): atoms.npy [n, d] unit directions, sae.pt, meta.json (per-atom
frequency, layer profile, best cosine to any token + nearest tokens, top exemplar contexts),
privileged.json, stats.json (verbal / privileged / remainder share of ||z||^2 by layer).

    python scripts/train_atoms.py --docs 240 --atoms 2048 --k 24
"""

import argparse
import json
import math
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
p.add_argument("--docs", type=int, default=240)
p.add_argument("--doc-offset", type=int, default=120, help="skip the docs the lens was fitted on")
p.add_argument("--layers", default="16,20,24,28,32,36,40,44,48,52,56,60,62")
p.add_argument("--topk-tokens", type=int, default=25)
p.add_argument("--atoms", type=int, default=2048)
p.add_argument("--k", type=int, default=24)
p.add_argument("--epochs", type=int, default=12)
p.add_argument("--out", default="lenses/atoms")
args = p.parse_args()
os.makedirs(args.out, exist_ok=True)
dev = "cuda"
t0 = time.time()
log = lambda m: print(f"[{time.time() - t0:6.0f}s] {m}", flush=True)
torch.manual_seed(0)

LAYERS = [int(x) for x in args.layers.split(",")]
tok = PreTrainedTokenizerFast(tokenizer_file="models/bonsai-meta/tokenizer.json")
model, config, _ = load_bonsai("models/bonsai-gguf/Ternary-Bonsai-2-27B-PQ2_0.gguf",
                               "models/bonsai-meta/config.json", device=dev, verbose=False)
J = {l: t.to(dev) for l, t in load_file("lenses/bonsai27b-j.safetensors").items() if int(l.split(".")[1]) in LAYERS}
J = {int(k.split(".")[1]): v for k, v in J.items()}
norm, head = model.model.norm, model.lm_head
gamma = (1.0 + norm.weight.float())  # HF zero-centred RMSNorm gain
D = gamma.shape[0]
log(f"model + {len(J)} J matrices loaded")


def token_dirs(ids: torch.Tensor) -> torch.Tensor:
    """u_v = gamma * W_eff[v] for a tensor of ids (any shape) -> [..., D] fp32."""
    flat = ids.reshape(-1)
    w = dequantize(head.packed[flat], head.scales[flat], torch.float32)
    w = fwht(w, head.block, head.signs.float(), inverse=True) * gamma
    return w.reshape(*ids.shape, D)


def nnls(A: torch.Tensor, y: torch.Tensor, iters: int = 5) -> torch.Tensor:
    """Batched non-negative least squares on a small support. A [N,K,D], y [N,D] -> c [N,K]."""
    G = A @ A.transpose(1, 2)
    b = (A @ y.unsqueeze(-1)).squeeze(-1)
    N, K = b.shape
    eye = torch.eye(K, device=A.device).expand(N, K, K)
    m = torch.ones(N, K, device=A.device)
    for _ in range(iters):
        mm = m.unsqueeze(-1) * m.unsqueeze(-2)
        c = torch.linalg.solve(G * mm + eye * (1 - m).unsqueeze(-1) + 1e-4 * eye * G.diagonal(dim1=1, dim2=2).mean(-1, keepdim=True).unsqueeze(-1), (b * m).unsqueeze(-1)).squeeze(-1)
        neg = (c < 0) & (m > 0)
        if not neg.any():
            break
        m = m * (~neg)
    return (c * m).clamp_min(0)


# ------------------------------------------------------------------ stage 1: remainders
texts = [json.loads(l)["text"] for l in open("lenses/corpus.jsonl")][args.doc_offset : args.doc_offset + args.docs]
captured = {}
hooks = [model.model.layers[l].register_forward_hook(lambda m, i, o, l=l: captured.__setitem__(l, o[0] if isinstance(o, tuple) else o)) for l in LAYERS]
SKIP = 16
per_doc = len(LAYERS) * 111
R = torch.empty(len(texts) * per_doc, D, dtype=torch.float16)
meta_doc = np.zeros(len(R), np.int32); meta_pos = np.zeros(len(R), np.int16); meta_layer = np.zeros(len(R), np.int16)
stats = {l: {"verbal": 0.0, "priv": 0.0, "rest": 0.0, "n": 0} for l in LAYERS}
energy = torch.zeros(D, device=dev)  # coordinate energy of the remainder, to find privileged dims
doc_ids, n = [], 0
PRIV = None
with torch.no_grad():
    for di, text in enumerate(texts):
        ids = tok(text, return_tensors="pt").input_ids[:, :128].to(dev)
        doc_ids.append(ids[0].tolist())
        model.model(input_ids=ids, use_cache=False)
        T = ids.shape[1]
        pos = torch.arange(SKIP, T - 1, device=dev)
        for l in LAYERS:
            h = captured[l][0, pos].float()                       # [P, D]
            z = h @ J[l].float().T                                # transport to canonical basis
            logits = head(norm(z.to(torch.bfloat16))).float()     # the lens readout
            top = logits.topk(args.topk_tokens, dim=-1).indices   # [P, K]
            A = token_dirs(top)                                   # [P, K, D]
            c = nnls(A, z)
            verbal = (c.unsqueeze(-1) * A).sum(1)
            r = z - verbal
            zz = (z * z).sum(-1)
            if PRIV is None:
                energy += (r * r).sum(0)
                pr = torch.zeros_like(zz)
            else:
                pr = (r[:, PRIV] ** 2).sum(-1)
                r[:, PRIV] = 0
            s = stats[l]
            s["verbal"] += float(((verbal * verbal).sum(-1) / zz).sum()); s["priv"] += float((pr / zz).sum())
            s["rest"] += float(((r * r).sum(-1) / zz).sum()); s["n"] += len(pos)
            k = len(pos)
            R[n : n + k] = r.half().cpu()
            meta_doc[n : n + k] = di; meta_pos[n : n + k] = pos.cpu().numpy(); meta_layer[n : n + k] = l
            n += k
        if di == 15 and PRIV is None:
            # Privileged coordinates: those carrying >1% of the remainder's energy on their own.
            share = (energy / energy.sum()).cpu()
            PRIV = torch.nonzero(share > 0.01).flatten().tolist()
            log("privileged dims: " + ", ".join(f"d{i} {share[i] * 100:.1f}%" for i in PRIV))
            R[:n, PRIV] = 0  # zero them retroactively in what was already stored
            for s in stats.values():
                s.update(verbal=0.0, priv=0.0, rest=0.0, n=0)  # restart stats with the split in place
        if di % 40 == 0:
            log(f"doc {di}/{len(texts)}  samples {n}")
for h in hooks:
    h.remove()
R = R[:n]; meta_doc, meta_pos, meta_layer = meta_doc[:n], meta_pos[:n], meta_layer[:n]
stats = {l: {k: (v / s["n"] if k != "n" else v) for k, v in s.items()} for l, s in stats.items()}
log("share of ||z||^2 by layer (verbal / privileged / remainder):")
for l, s in stats.items():
    log(f"  L{l}: {s['verbal'] * 100:5.1f}% / {s['priv'] * 100:5.1f}% / {s['rest'] * 100:5.1f}%")

# token directions (unit) for nearest-token lookups, computed before freeing the model
U = torch.empty(head.packed.shape[0], D, dtype=torch.float16, device=dev)
for a in range(0, len(U), 8192):
    u = token_dirs(torch.arange(a, min(a + 8192, len(U)), device=dev))
    U[a : a + len(u)] = (u / u.norm(dim=-1, keepdim=True).clamp_min(1e-8)).half()
del model, J
torch.cuda.empty_cache()
log(f"{n} remainder vectors collected")

# ------------------------------------------------------------------ stage 2: TopK SAE
X = R.to(dev)
scale = math.sqrt(D) / X.float().norm(dim=-1).mean().item()
Na, K = args.atoms, args.k
b_dec = torch.nn.Parameter(X[:65536].float().mean(0) * scale)
W_dec = torch.nn.Parameter(torch.randn(D, Na, device=dev))
with torch.no_grad():
    W_dec /= W_dec.norm(dim=0, keepdim=True)
W_enc = torch.nn.Parameter(W_dec.detach().T.clone())
b_enc = torch.nn.Parameter(torch.zeros(Na, device=dev))
opt = torch.optim.Adam([b_dec, W_dec, W_enc, b_enc], lr=2e-4)


def encode(x):
    pre = torch.relu((x - b_dec) @ W_enc.T + b_enc)
    v, i = pre.topk(K, dim=-1)
    return v, i


def decode(v, i):
    return (W_dec[:, i] * v.unsqueeze(1)).sum(-1) if False else torch.einsum("bk,dbk->bd", v, W_dec[:, i]) + b_dec


last_fired = torch.zeros(Na, device=dev)
step, B = 0, 4096
for ep in range(args.epochs):
    perm = torch.randperm(n, device=dev)
    tot, cnt = 0.0, 0
    for a in range(0, n, B):
        x = X[perm[a : a + B]].float() * scale
        v, i = encode(x)
        xh = decode(v, i)
        loss = ((xh - x) ** 2).sum(-1).mean() / ((x - x.mean(0)) ** 2).sum(-1).mean()
        opt.zero_grad(); loss.backward(); opt.step()
        with torch.no_grad():
            W_dec /= W_dec.norm(dim=0, keepdim=True)
            last_fired += 1
            last_fired[i.flatten()] = 0
        tot += loss.item(); cnt += 1; step += 1
    dead = int((last_fired > 20).sum())
    if dead and ep < args.epochs - 2:  # resample dead atoms toward badly reconstructed inputs
        with torch.no_grad():
            x = X[torch.randint(0, n, (16384,), device=dev)].float() * scale
            err = ((decode(*encode(x)) - x) ** 2).sum(-1)
            pick = x[err.topk(dead).indices] - b_dec
            d_ids = torch.nonzero(last_fired > 20).flatten()
            W_dec[:, d_ids] = (pick / pick.norm(dim=-1, keepdim=True)).T
            W_enc[d_ids] = W_dec[:, d_ids].T * 0.2
            b_enc[d_ids] = 0
            last_fired[d_ids] = 0
    log(f"epoch {ep + 1}/{args.epochs}: FVU {tot / cnt:.3f}  dead {dead}")

# ------------------------------------------------------------------ stage 3: describe atoms
with torch.no_grad():
    atoms = W_dec.T.contiguous()  # [Na, D] unit
    freq = torch.zeros(Na, device=dev)
    layer_mass = torch.zeros(Na, len(LAYERS), device=dev)
    L_index = {l: j for j, l in enumerate(LAYERS)}
    ml = torch.tensor([L_index[int(l)] for l in meta_layer], device=dev)
    TOPN = 12
    best_v = torch.full((Na, TOPN), -1.0, device=dev)
    best_i = torch.zeros((Na, TOPN), dtype=torch.long, device=dev)
    fvu_num, fvu_den = 0.0, 0.0
    for a in range(0, n, 8192):
        x = X[a : a + 8192].float() * scale
        v, i = encode(x)
        xh = decode(v, i)
        fvu_num += float(((xh - x) ** 2).sum()); fvu_den += float(((x - x.mean(0)) ** 2).sum())
        dense = torch.zeros(len(x), Na, device=dev).scatter_(1, i, v)
        freq += (dense > 0).float().sum(0)
        layer_mass.index_add_(1, ml[a : a + len(x)], dense.T)
        cat_v = torch.cat([best_v, dense.T], 1)
        idx = torch.arange(a, a + len(x), device=dev).expand(Na, -1)
        cat_i = torch.cat([best_i, idx], 1)
        tv, ti = cat_v.topk(TOPN, dim=1)
        best_v, best_i = tv, cat_i.gather(1, ti)
    sims = (atoms.half() @ U.T).float()  # [Na, vocab] cosine to every token direction
    max_cos, nn_ids = sims.abs().topk(10, dim=1)
    nn_sign = sims.gather(1, nn_ids).sign()
    priv_sims = U[:, PRIV].float().T if PRIV else torch.zeros(0, len(U), device=dev)  # e_i . u_v
    priv_nn = priv_sims.abs().topk(10, dim=1) if PRIV else None

log(f"final FVU {fvu_num / fvu_den:.3f}; median best |cos| to any token {max_cos[:, 0].median():.3f}")


def context(di, pos, width=12):
    ids = doc_ids[di]
    left = tok.decode(ids[max(0, pos - width) : pos])
    return {"left": left, "tok": tok.decode([ids[pos]]), "right": tok.decode(ids[pos + 1 : pos + 4])}


meta = {
    "layers": LAYERS, "n_samples": n, "k": K, "fvu": fvu_num / fvu_den, "scale": scale,
    "basis": "canonical (final-layer) basis: z = J_l h_l; remainder after NNLS on top-%d lens tokens" % args.topk_tokens,
    "atoms": [],
}
lm_np = (layer_mass / layer_mass.sum(1, keepdim=True).clamp_min(1e-9)).cpu().numpy()
for a in range(Na):
    ex = []
    for v, si in zip(best_v[a].tolist(), best_i[a].tolist()):
        if v <= 0:
            continue
        ex.append({"act": round(v, 3), "layer": int(meta_layer[si]), **context(int(meta_doc[si]), int(meta_pos[si]))})
    meta["atoms"].append({
        "id": f"ξ{a}", "freq": float(freq[a] / n), "layer_profile": [round(float(x), 3) for x in lm_np[a]],
        "best_cos": round(float(max_cos[a, 0]), 4),
        "nearest": [{"id": int(t), "cos": round(float(c * s), 3)} for t, c, s in zip(nn_ids[a], max_cos[a], nn_sign[a])],
        "exemplars": ex,
    })
priv = [{"id": f"d{d}", "dim": d,
         "best_cos": round(float(priv_nn.values[j, 0]), 4),
         "nearest": [{"id": int(t), "cos": round(float(c), 3)} for t, c in zip(priv_nn.indices[j], priv_nn.values[j])]}
        for j, d in enumerate(PRIV)] if PRIV else []
np.save(f"{args.out}/atoms.npy", atoms.cpu().numpy().astype(np.float32))
torch.save({"W_enc": W_enc.detach().cpu(), "b_enc": b_enc.detach().cpu(), "W_dec": W_dec.detach().cpu(),
            "b_dec": b_dec.detach().cpu(), "scale": scale, "k": K, "privileged": PRIV}, f"{args.out}/sae.pt")
json.dump(meta, open(f"{args.out}/meta.json", "w"), ensure_ascii=False)
json.dump(priv, open(f"{args.out}/privileged.json", "w"), ensure_ascii=False)
json.dump({str(l): s for l, s in stats.items()}, open(f"{args.out}/stats.json", "w"))
log(f"wrote {args.out}")
