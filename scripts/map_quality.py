"""Make the token space mean more, and mean it more accurately.

1. Layout fidelity, per point and per view: the share of a point's 15 nearest neighbours in the
   space the layout was built from (A: token PCA-128; B/densMAP/C: the denoised joint space)
   that are also among its 15 nearest neighbours in the drawn layout. Low fidelity = where
   the picture's neighbourhoods are projection artifacts.
2. Regions: k-means (cosine) on the token directions in the denoised token space; each region
   gets representative tokens (closest to its centre) and a coherence score, for map labels.
   Names are added later by scripts/autointerp.py.

Writes lenses/joint/quality/: fid_{A3,A2,B,D,C}.u8 (0..15), regions.json, region_of.u16
"""

import json
import os
import sys
import time

import numpy as np
import torch
from scipy.spatial import cKDTree

sys.path.insert(0, ".")
from bonsai_lens.ternary import TernaryLinear, transcode

OUT = "lenses/joint/quality"
os.makedirs(OUT, exist_ok=True)
dev = "cuda"
t0 = time.time()
log = lambda m: print(f"[{time.time() - t0:6.0f}s] {m}", flush=True)
torch.manual_seed(0)
K = 15

# ---------------------------------------------------------------- vectors (as in embed_joint.py)
from gguf import GGUFReader

reader = GGUFReader("models/bonsai-gguf/Ternary-Bonsai-2-27B-PQ2_0.gguf")
fields = {k: v.contents() for k, v in reader.fields.items()}
tensors = {t.name: t for t in reader.tensors}
head, norm_t = tensors["output.weight"], tensors["output_norm.weight"]
rows, width = (int(n) for n in head.shape[::-1])
packed, scales = transcode(head.data.tobytes(), rows, width, head.tensor_type.name)
block = int(fields["prism.hadamard.block_size"])
values = np.asarray(fields["prism.hadamard.sign_values"], dtype=np.float32)
off = 0
for w in fields["prism.hadamard.sign_widths"]:
    if w == width:
        signs = torch.from_numpy(values[off : off + w].copy())
    off += w
gamma = torch.from_numpy(np.asarray(norm_t.data, dtype=np.float32).copy()).to(dev)
P, S = torch.from_numpy(packed).to(dev), torch.from_numpy(scales).to(dev)
U = torch.empty(rows, width, dtype=torch.float16, device=dev)
for a in range(0, rows, 16384):
    u = TernaryLinear(P[a : a + 16384], S[a : a + 16384], block, signs.to(dev), torch.float32).dense() * gamma
    U[a : a + len(u)] = (u / u.norm(dim=-1, keepdim=True).clamp_min(1e-8)).half()
del P, S
atoms = torch.from_numpy(np.load("lenses/atoms/atoms.npy")).to(dev)
atoms = atoms / atoms.norm(dim=-1, keepdim=True)
priv = json.load(open("lenses/atoms/privileged.json"))
E = torch.zeros(len(priv), width, device=dev)
for j, p in enumerate(priv):
    E[j, p["dim"]] = 1
NT, NA, NP = len(U), len(atoms), len(E)
N = NT + NA + NP

mu = torch.zeros(width, device=dev)
for a in range(0, NT, 32768):
    mu += U[a : a + 32768].float().sum(0)
mu /= NT
g = torch.Generator(device=dev).manual_seed(0)
samp = torch.randperm(NT, device=dev, generator=g)[:60000]
_, _, Vt = torch.pca_lowrank(U[samp].float() - mu, q=128, center=False, niter=4)
_, _, Va = torch.pca_lowrank(atoms - atoms.mean(0), q=64, center=False, niter=4)
Q, _ = torch.linalg.qr(torch.cat([Vt, Va], 1))
Xd = torch.cat([(U[a : a + 32768].float() - mu) @ Q for a in range(0, NT, 32768)] + [(atoms - mu) @ Q, (E - mu) @ Q])
Xd = (Xd / Xd.norm(dim=-1, keepdim=True).clamp_min(1e-8)).half()
del U
log("denoised joint space rebuilt")


def knn_hd(Y):
    out = np.empty((len(Y), K), np.int32)
    for a in range(0, len(Y), 1024):
        s = (Y[a : a + 1024] @ Y.T).float()
        s[torch.arange(len(s)), torch.arange(a, a + len(s), device=dev)] = -2
        out[a : a + len(s)] = s.topk(K, dim=1).indices.cpu().numpy()
    return out


WIDE = 10 * K  # "loose" fidelity: true neighbours found anywhere among the 150 nearest on screen


def knn_ld(C):
    return cKDTree(C).query(C, k=WIDE + 1, workers=12)[1][:, 1:].astype(np.int32)


def overlap(hd, ld):
    # vectorised set overlap: for each row, how many of hd's ids appear in ld's ids
    return (hd[:, :, None] == ld[:, None, :]).any(-1).sum(1).astype(np.uint8)


def fidelity(hd, ld):
    strict = np.concatenate([overlap(hd[a : a + 20000], ld[a : a + 20000, :K]) for a in range(0, len(hd), 20000)])
    loose = np.concatenate([overlap(hd[a : a + 20000], ld[a : a + 20000]) for a in range(0, len(hd), 20000)])
    return strict, loose


hd_joint = knn_hd(Xd)
log("joint high-d kNN done")
Z = torch.from_numpy(np.load("lenses/space/pca128.npy").astype(np.float32)).to(dev)
Z = (Z / Z.norm(dim=-1, keepdim=True).clamp_min(1e-8)).half()
hd_tok = knn_hd(Z)
log("token (view A) high-d kNN done")

layouts = {
    "A3": (np.load("lenses/space/coords3.npy"), hd_tok),
    "A2": (np.load("lenses/space/coords2.npy"), hd_tok),
    "B": (np.fromfile("lenses/joint/coords_b3.f32", np.float32).reshape(N, 3), hd_joint),
    "D": (np.fromfile("lenses/joint/coords_d3.f32", np.float32).reshape(N, 3), hd_joint),
}
b2 = np.fromfile("lenses/joint/coords_b2.f32", np.float32).reshape(N, 2)
dist = np.fromfile("lenses/joint/dist_word.f32", np.float32)
layouts["C"] = (np.c_[b2[:, 0], dist * 1.1, -b2[:, 1]], hd_joint)
summary = {}
for name, (C, hd) in layouts.items():
    strict, f = fidelity(hd[: len(C)], knn_ld(C.astype(np.float32)))
    full = np.zeros(N, np.uint8); full[: len(f)] = f  # per-point colour uses the loose measure
    full.tofile(f"{OUT}/fid_{name}.u8")
    at = slice(NT, NT + NA)
    summary[name] = {"strict_tokens": float(strict[:NT].mean() / K), "loose_tokens": float(f[:NT].mean() / K),
                     "strict_atoms": float(strict[at].mean() / K) if len(f) > NT else None,
                     "loose_atoms": float(f[at].mean() / K) if len(f) > NT else None,
                     "loose_hist": np.bincount(f[:NT], minlength=K + 1).tolist()}
    s_ = summary[name]
    log(f"fidelity {name}: tokens strict {s_['strict_tokens']:.3f} loose {s_['loose_tokens']:.3f}"
        + (f" | atoms strict {s_['strict_atoms']:.3f} loose {s_['loose_atoms']:.3f}" if s_["strict_atoms"] is not None else ""))
# reference: the same measures for a random layout of the same points would be ~K/N and ~WIDE/N

# ---------------------------------------------------------------- regions (cosine k-means on tokens)
R = 256
Xt = Xd[:NT].float()
cent = Xt[torch.randperm(NT, device=dev, generator=g)[:R]].clone()
for it in range(25):
    lab = torch.cat([(Xt[a : a + 65536] @ cent.T).argmax(1) for a in range(0, NT, 65536)])
    new = torch.zeros_like(cent).index_add_(0, lab, Xt)
    cnt = torch.bincount(lab, minlength=R).float()
    empty = cnt == 0
    new[empty] = Xt[torch.randint(0, NT, (int(empty.sum()),), device=dev, generator=g)]
    cent = new / new.norm(dim=-1, keepdim=True).clamp_min(1e-8)
sims = torch.cat([(Xt[a : a + 65536] * cent[lab[a : a + 65536]]).sum(-1) for a in range(0, NT, 65536)])
lab_np, sims_np = lab.cpu().numpy(), sims.cpu().numpy()
lab_np.astype(np.uint16).tofile(f"{OUT}/region_of.u16")
tok_meta = json.load(open("lenses/space/meta.json"))
tokens, cats, catnames = tok_meta["tokens"], tok_meta["cat"], tok_meta["categories"]
regions = []
for r in range(R):
    idx = np.where(lab_np == r)[0]
    if not len(idx):
        continue
    order = idx[np.argsort(-sims_np[idx])]
    cat_counts = np.bincount([cats[i] for i in idx], minlength=len(catnames))
    regions.append({"id": r, "size": int(len(idx)), "coherence": float(sims_np[idx].mean()),
                    "top": [int(i) for i in order[:40]],
                    "script_mix": {catnames[c]: int(n) for c, n in enumerate(cat_counts) if n},
                    "name": None})
json.dump({"regions": regions, "k": R, "space": "cosine k-means on token directions in the denoised joint space"},
          open(f"{OUT}/regions.json", "w"), ensure_ascii=False)
json.dump({"k": K, "wide": WIDE, "fidelity": summary,
           "note": "strict = share of a point's 15 true nearest neighbours among its 15 nearest on screen; "
                   "loose = share found among its 150 nearest on screen. Random layout ~0.00006 / ~0.0006."},
          open(f"{OUT}/summary.json", "w"), indent=1)
log(f"{len(regions)} regions; done")
