"""Joint token + non-verbal-atom space (views B, B-densMAP, C) and separation statistics.

Points, all unit vectors in the canonical (final-layer) basis the J-lens decodes in:
  tokens      u_v = gamma * W_eff[v]            (248,320)
  atoms       SAE decoder directions            (from scripts/train_atoms.py)
  privileged  coordinate axes e_i (e.g. d3994)

The kNN graph is exact cosine in the full 5120-d space (GPU), not after PCA, so nothing is
pre-squashed before UMAP. Layouts: B = UMAP 3D, D = densMAP 3D (also preserves local
density, so isolated groups stay spread out), F = UMAP 2D floor for view C, whose height is
the measured cosine distance to the nearest word direction (0 for every token by definition).

Statistics (full space, no projection):
  best cosine to any token, for atoms vs tokens-to-nearest-other-token vs random directions
  vs random directions drawn like the remainder data; neighbourhood purity of atoms.
"""

import json
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, ".")
from bonsai_lens.ternary import TernaryLinear, transcode

import argparse

ap = argparse.ArgumentParser()
ap.add_argument("--space", choices=["denoised", "full"], default="denoised",
                help="denoised: kNN in (token PCA-128 + atom PCA-64) after centring on the token mean; full: raw 5120-d")
ap.add_argument("--out", default="lenses/joint")
ap.add_argument("--atom-dims", type=int, default=512,
                help="atom principal directions kept in the denoised space (64 kept only 27%% of atom variance and made atoms look 2.5x more alike)")
args = ap.parse_args()
OUT = args.out
os.makedirs(OUT, exist_ok=True)
dev = "cuda"
t0 = time.time()
log = lambda m: print(f"[{time.time() - t0:6.0f}s] {m}", flush=True)
torch.manual_seed(0)

# ---------------------------------------------------------------- vectors
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
    lin = TernaryLinear(P[a : a + 16384], S[a : a + 16384], block, signs.to(dev), torch.float32)
    u = lin.dense() * gamma
    U[a : a + len(u)] = (u / u.norm(dim=-1, keepdim=True).clamp_min(1e-8)).half()
del P, S
atoms = torch.from_numpy(np.load("lenses/atoms/atoms.npy")).to(dev)
atoms = (atoms / atoms.norm(dim=-1, keepdim=True)).half()
priv = json.load(open("lenses/atoms/privileged.json"))
E = torch.zeros(len(priv), width, device=dev, dtype=torch.float16)
for j, p in enumerate(priv):
    E[j, p["dim"]] = 1
X = torch.cat([U, atoms, E])  # [N, d]
NT, NA, NP = len(U), len(atoms), len(E)
N = len(X)
log(f"points: {NT} tokens + {NA} atoms + {NP} privileged = {N}")


def best_cos(Q, exclude_self_offset=None):
    """Best (signed) cosine of each row of Q with any token direction."""
    out = torch.empty(len(Q), device=dev)
    for a in range(0, len(Q), 2048):
        s = (Q[a : a + 2048] @ U.T).float()
        if exclude_self_offset is not None:
            idx = torch.arange(a, a + len(s), device=dev) + exclude_self_offset
            s[torch.arange(len(s)), idx] = -2
        out[a : a + len(s)] = s.max(1).values
    return out


# ---------------------------------------------------------------- statistics
g = torch.Generator(device=dev).manual_seed(0)
tok_sample = torch.randperm(NT, device=dev, generator=g)[:8000]
st_tok = best_cos(U[tok_sample])  # includes self -> recompute excluding self
st_tok = torch.empty(len(tok_sample), device=dev)
for a in range(0, len(tok_sample), 2048):
    ids = tok_sample[a : a + 2048]
    s = (U[ids] @ U.T).float()
    s[torch.arange(len(ids)), ids] = -2
    st_tok[a : a + len(ids)] = s.max(1).values
st_atom = best_cos(atoms)
rand = torch.randn(4000, width, device=dev, generator=g)
st_rand = best_cos((rand / rand.norm(dim=-1, keepdim=True)).half())
# random directions shaped like the remainder data: Gaussian with the atoms' covariance
A32 = atoms.float()
rand_r = torch.randn(4000, NA, device=dev, generator=g) @ A32
st_randr = best_cos((rand_r / rand_r.norm(dim=-1, keepdim=True)).half())
st_priv = best_cos(E)


def summ(x):
    x = x.float().cpu().numpy()
    return {"median": float(np.median(x)), "p10": float(np.percentile(x, 10)), "p90": float(np.percentile(x, 90)),
            "hist": np.histogram(x, bins=40, range=(-0.1, 1.0))[0].tolist()}


stats = {"best_cos_to_any_token": {
    "tokens (nearest other token)": summ(st_tok), "atoms": summ(st_atom),
    "random directions": summ(st_rand), "random mixes of atoms": summ(st_randr),
    "privileged": st_priv.float().cpu().tolist()}, "hist_range": [-0.1, 1.0]}
for k, v in stats["best_cos_to_any_token"].items():
    if isinstance(v, dict):
        log(f"best cos to any token — {k:30s} median {v['median']:.3f} (p10 {v['p10']:.3f}, p90 {v['p90']:.3f})")

# ---------------------------------------------------------------- exact kNN (cosine)
K = 30


def exact_knn(Y, unit=False):
    """Exact cosine kNN over the rows of Y, self first. ``unit``: rows are already unit fp16."""
    if not unit:
        Y = (Y / Y.norm(dim=-1, keepdim=True).clamp_min(1e-8)).half()
    ki = np.empty((N, K), np.int32); kd = np.empty((N, K), np.float32)
    for a in range(0, N, 4096):
        s = (Y[a : a + 4096] @ Y.T).float()
        s[torch.arange(len(s)), torch.arange(a, a + len(s), device=dev)] = 2
        v, i = s.topk(K, dim=1)
        ki[a : a + len(s)] = i.cpu().numpy(); kd[a : a + len(s)] = (1 - v.clamp(-1, 1)).cpu().numpy()
    kd[:, 0] = 0
    return ki, kd


def graph_health(ki, name):
    """Hubness (skew of how often each point is someone's neighbour) and fragmentation."""
    from scipy.sparse import csr_matrix
    from scipy.sparse.csgraph import connected_components
    occ = np.bincount(ki[:, 1:].ravel(), minlength=N).astype(np.float64)
    skew = float(((occ - occ.mean()) ** 3).mean() / occ.std() ** 3)
    g = csr_matrix((np.ones(N * (K - 1)), (np.repeat(np.arange(N), K - 1), ki[:, 1:].ravel())), shape=(N, N))
    nc, lab = connected_components(g, directed=True, connection="weak")
    sizes = np.bincount(lab)
    h = {"k_occurrence_skew": skew, "never_a_neighbour": float((occ == 0).mean()), "max_k_occurrence": int(occ.max()),
         "components": int(nc), "outside_largest_component": float(1 - sizes.max() / N)}
    log(f"graph [{name}]: " + json.dumps(h))
    return h


knn_full = exact_knn(X, unit=True)  # X rows are unit fp16 already; avoid a 5 GB fp32 copy
health = {"full 5120-d (raw)": graph_health(knn_full[0], "full")}
if args.space == "denoised":
    # Centre on the token mean (the readout directions share a large common component), then keep
    # the top-128 token principal directions plus the top-64 atom principal directions.
    mu = torch.zeros(width, device=dev)
    for a in range(0, NT, 32768):
        mu += U[a : a + 32768].float().sum(0)
    mu /= NT
    samp = torch.randperm(NT, device=dev, generator=g)[:60000]  # PCA directions from a 60k-token sample
    _, _, Vt = torch.pca_lowrank(U[samp].float() - mu, q=128, center=False, niter=4)
    _, _, Va = torch.pca_lowrank(atoms.float() - atoms.float().mean(0), q=args.atom_dims, center=False, niter=4)
    Q, _ = torch.linalg.qr(torch.cat([Vt, Va], 1))
    Xd = torch.cat([(X[a : a + 32768].float() - mu) @ Q for a in range(0, N, 32768)])  # [N, 192]
    knn_i, knn_d = exact_knn(Xd)
    health[f"denoised (token PCA-128 + atom PCA-{args.atom_dims}, centred)"] = graph_health(knn_i, "denoised")
else:
    knn_i, knn_d = knn_full
log("exact kNN done")
is_atom = np.zeros(N, bool); is_atom[NT:] = True
purity_full = is_atom[knn_full[0][NT : NT + NA, 1:]].mean(1)
purity = is_atom[knn_i[NT : NT + NA, 1:]].mean(1)
rand_ix = np.random.default_rng(0).choice(NT, NA, replace=False)
purity_tok = is_atom[knn_i[rand_ix, 1:]].mean(1)
stats["purity"] = {"atoms: share of 29 nearest neighbours that are atoms": float(purity.mean()),
                   "tokens: share of 29 nearest neighbours that are atoms": float(purity_tok.mean()),
                   "chance (atoms / all points)": NA / N,
                   "atoms with no token among nearest 29": float((purity == 1).mean()),
                   "atoms: share of neighbours that are atoms, raw full space": float(purity_full.mean())}
stats["graph_health"] = health
log("purity: " + json.dumps(stats["purity"]))

# ---------------------------------------------------------------- layouts
import umap

# X for UMAP's own bookkeeping (spectral init uses the graph)
Z = (Xd if args.space == "denoised" else X.float() - X.float().mean(0))
_, _, V = torch.pca_lowrank(Z, q=64, center=True, niter=3)
Z = (Z @ V).cpu().numpy()
knn = (knn_i, knn_d, None)
layouts = {}
for name, dim, dens in (("b3", 3, False), ("d3", 3, True), ("b2", 2, False)):
    r = umap.UMAP(n_components=dim, n_neighbors=K, min_dist=0.08, metric="cosine", precomputed_knn=knn,
                  densmap=dens, n_jobs=12, low_memory=True, verbose=False)
    Y = r.fit_transform(Z).astype(np.float32)
    Y -= Y.mean(0)
    Y /= np.abs(Y).max()
    layouts[name] = Y
    Y.tofile(f"{OUT}/coords_{name}.f32")
    log(f"layout {name} done")

# ---------------------------------------------------------------- metadata
am = json.load(open("lenses/atoms/meta.json"))
dist_word = np.concatenate([np.zeros(NT), 1 - st_atom.cpu().numpy(), 1 - st_priv.cpu().numpy()]).astype(np.float32)
dist_word.tofile(f"{OUT}/dist_word.f32")
meta = {
    "n_tokens": NT, "n_atoms": NA, "n_priv": NP,
    "atoms": [{k: a[k] for k in ("id", "freq", "layer_profile", "best_cos", "nearest", "exemplars")} for a in am["atoms"]],
    "privileged": priv, "layers": am["layers"], "fvu": am["fvu"], "stats": stats,
    "space": (f"exact cosine kNN (k=30) in a denoised joint space (centred on the token mean; token PCA-128 + atom PCA-{args.atom_dims})"
              if args.space == "denoised" else "exact cosine kNN (k=30) in the raw 5120-d canonical basis")
             + "; B = UMAP, D = densMAP, C = UMAP-2D floor + distance-to-nearest-word height",
}
json.dump(meta, open(f"{OUT}/meta_joint.json", "w"), ensure_ascii=False)
log(f"wrote {OUT}")
