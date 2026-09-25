"""Which projection keeps the token map most faithful? A measured bake-off.

Input: the token directions in the same space view A is built from (lenses/space/pca128.npy,
unit-normalised). Truth: each token's 15 nearest neighbours there (exact cosine, GPU).
Candidates: UMAP with several settings, PaCMAP. Score: strict (15 true neighbours among 15
nearest on screen) and loose (among 150 nearest), plus a global score: Spearman correlation of
pairwise distances for 20k random pairs (does "far on screen" mean "far in the model"?).

    python scripts/layout_bakeoff.py
"""

import json
import time

import numpy as np
import torch
from scipy.spatial import cKDTree
from scipy.stats import spearmanr

t0 = time.time()
log = lambda m: print(f"[{time.time() - t0:6.0f}s] {m}", flush=True)
K, WIDE = 15, 150
dev = "cuda"

Z = np.load("lenses/space/pca128.npy").astype(np.float32)
Z /= np.linalg.norm(Z, axis=1, keepdims=True) + 1e-8
N = len(Z)
Zt = torch.from_numpy(Z).to(dev).half()
hd = np.empty((N, K), np.int32)
for a in range(0, N, 1024):
    s = (Zt[a : a + 1024] @ Zt.T).float()
    s[torch.arange(len(s)), torch.arange(a, a + len(s), device=dev)] = -2
    hd[a : a + len(s)] = s.topk(K, dim=1).indices.cpu().numpy()
del Zt
torch.cuda.empty_cache()
log("true neighbours done")

rng = np.random.default_rng(0)
pairs = rng.integers(0, N, (20000, 2))
hd_dist = 1 - (Z[pairs[:, 0]] * Z[pairs[:, 1]]).sum(1)


def score(Y):
    ld = cKDTree(Y).query(Y, k=WIDE + 1, workers=12)[1][:, 1:]
    strict = loose = 0
    for a in range(0, N, 20000):
        h, l = hd[a : a + 20000], ld[a : a + 20000]
        strict += (h[:, :, None] == l[:, None, :K]).any(-1).sum()
        loose += (h[:, :, None] == l[:, None, :]).any(-1).sum()
    ld_dist = np.linalg.norm(Y[pairs[:, 0]] - Y[pairs[:, 1]], axis=1)
    return {"strict": strict / (N * K), "loose": loose / (N * K), "global_spearman": float(spearmanr(hd_dist, ld_dist)[0])}


results = {"current A3 (UMAP k=30, min_dist=0.08)": score(np.load("lenses/space/coords3.npy"))}
log(f"current: {results['current A3 (UMAP k=30, min_dist=0.08)']}")

import umap

cands = {
    "UMAP k=15, min_dist=0.0": dict(n_neighbors=15, min_dist=0.0),
    "UMAP k=50, min_dist=0.1": dict(n_neighbors=50, min_dist=0.1),
    "UMAP k=30, min_dist=0.0, 1000 epochs": dict(n_neighbors=30, min_dist=0.0, n_epochs=1000),
}
for name, kw in cands.items():
    Y = umap.UMAP(n_components=3, metric="cosine", n_jobs=12, low_memory=True, **kw).fit_transform(Z)
    results[name] = score(Y)
    np.save(f"lenses/space/bakeoff_{name.split(',')[0].replace(' ', '_').replace('=', '')}_{len(results)}.npy", Y.astype(np.float32))
    log(f"{name}: {results[name]}")
    json.dump(results, open("lenses/space/bakeoff.json", "w"), indent=1)

try:
    import pacmap

    Y = pacmap.PaCMAP(n_components=3, n_neighbors=15, MN_ratio=0.5, FP_ratio=2.0).fit_transform(Z, init="pca")
    results["PaCMAP"] = score(Y)
    np.save("lenses/space/bakeoff_pacmap.npy", Y.astype(np.float32))
    log(f"PaCMAP: {results['PaCMAP']}")
except Exception as e:  # noqa: BLE001
    results["PaCMAP"] = {"error": repr(e)}
    log(f"PaCMAP failed: {e!r}")
json.dump(results, open("lenses/space/bakeoff.json", "w"), indent=1)
log("done")
