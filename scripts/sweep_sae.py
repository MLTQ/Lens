"""How many non-verbal atoms are there? Dictionary-size sweep on held-out data.

Trains TopK sparse autoencoders (k active atoms per activation) of increasing size on the
remainders from scripts/sweep_collect.py and reports, on HELD-OUT documents:
  FVU (fraction of variance unexplained), dead atoms, firing frequencies,
plus a linear reference: FVU of PCA with m components (how many dimensions the remainder
occupies if you only allow straight directions). If FVU keeps falling as the dictionary
doubles, the repertoire is large; a plateau means it is small.

AuxK (Gao et al. 2024, "Scaling and evaluating sparse autoencoders"): atoms that have not
fired recently are trained to explain the main reconstruction's error, which keeps large
dictionaries from dying.

    python scripts/sweep_sae.py --sizes 1024,2048,4096,8192,16384,32768 --k 24
"""

import argparse
import json
import math
import time

import numpy as np
import torch

p = argparse.ArgumentParser()
p.add_argument("--data", default="lenses/sweep")
p.add_argument("--sizes", default="1024,2048,4096,8192,16384,32768")
p.add_argument("--k", type=int, default=24)
p.add_argument("--epochs", type=int, default=8)
p.add_argument("--save", default="8192,32768", help="sizes whose weights are kept")
p.add_argument("--extra", default="8192:64,8192:128", help="extra size:k runs, to test how dense the code is")
p.add_argument("--resume", action="store_true", help="keep finished runs from sweep_results.json")
args = p.parse_args()
dev = "cuda"
t0 = time.time()
log = lambda m: print(f"[{time.time() - t0:6.0f}s] {m}", flush=True)
torch.manual_seed(0)

tr = np.load(f"{args.data}/train_R.npy", mmap_mode="r")
ev = np.load(f"{args.data}/eval_R.npy", mmap_mode="r")
N, D = tr.shape
X = torch.empty(N, D, dtype=torch.float16, device=dev)
for a in range(0, N, 65536):
    X[a : a + 65536] = torch.from_numpy(np.ascontiguousarray(tr[a : a + 65536])).to(dev)
XE = torch.from_numpy(np.ascontiguousarray(ev[:])).pin_memory()  # eval stays on the CPU, streamed per batch
scale = math.sqrt(D) / X[:200000].float().norm(dim=-1).mean().item()
log(f"train {N} x {D}, eval {len(XE)}; scale {scale:.3f}")


def fvu_of(recon_fn, data, bs=8192):
    num = den = 0.0
    mu = data[:100000].to(dev).float().mean(0) * scale
    for a in range(0, len(data), bs):
        x = data[a : a + bs].to(dev, non_blocking=True).float() * scale
        xh = recon_fn(x)
        num += float(((xh - x) ** 2).sum()); den += float(((x - mu) ** 2).sum())
    return num / den


# ---------------------------------------------------------------- linear reference (PCA)
mu = XE[:100000].to(dev).float().mean(0)
_, S, V = torch.pca_lowrank(X[torch.randperm(N, device=dev)[:100000]].float() * scale - mu * scale, q=1024, center=False, niter=4)
pca = {}
for m in (24, 64, 128, 256, 512, 1024):
    Vm = V[:, :m]
    pca[m] = fvu_of(lambda x: ((x - mu * scale) @ Vm) @ Vm.T + mu * scale, XE)
log("PCA eval FVU: " + ", ".join(f"{m}:{v:.3f}" for m, v in pca.items()))


# ---------------------------------------------------------------- TopK SAE
class SAE(torch.nn.Module):
    def __init__(self, n, k):
        super().__init__()
        self.k = k
        self.b_dec = torch.nn.Parameter(X[:65536].float().mean(0) * scale)
        W = torch.randn(D, n, device=dev)
        W /= W.norm(dim=0, keepdim=True)
        self.W_dec = torch.nn.Parameter(W)
        self.W_enc = torch.nn.Parameter(W.T.clone())
        self.b_enc = torch.nn.Parameter(torch.zeros(n, device=dev))

    def pre(self, x):
        with torch.autocast("cuda", dtype=torch.bfloat16):
            h = (x - self.b_dec) @ self.W_enc.T
        return torch.relu(h.float() + self.b_enc)

    def decode(self, v, i):
        # sparse codes as a dense [B, n] matrix times the dictionary: far less memory than gathering
        # a [d, B, k] slice of the decoder
        z = torch.zeros(v.shape[0], self.W_dec.shape[1], device=v.device, dtype=v.dtype).scatter(1, i, v)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            out = z @ self.W_dec.T
        return out.float() + self.b_dec

    def forward(self, x):
        pre = self.pre(x)
        v, i = pre.topk(self.k, dim=-1)
        return self.decode(v, i), pre, v, i


prev = json.load(open(f"{args.data}/sweep_results.json")) if args.resume else {"sae": []}
results = prev["sae"]
done = {(r["n"], r["k"]) for r in results}
runs = [(int(s), args.k) for s in args.sizes.split(",")] + [tuple(int(x) for x in e.split(":")) for e in args.extra.split(",") if e]
for n, k in runs:
    if (n, k) in done:
        log(f"skip n={n} k={k} (done)")
        continue
    sae = SAE(n, k)
    opt = torch.optim.Adam(sae.parameters(), lr=2e-4 / math.sqrt(n / 2048), betas=(0.9, 0.999))
    since = torch.zeros(n, device=dev)
    B, aux_k, dead_after = 4096, min(512, n // 2), 300
    steps_per_epoch = N // B
    for ep in range(args.epochs):
        perm = torch.randperm(N, device=dev)
        for s in range(steps_per_epoch):
            x = X[perm[s * B : (s + 1) * B]].float() * scale
            xh, pre, v, i = sae(x)
            err = x - xh
            loss = (err**2).sum(-1).mean() / ((x - x.mean(0)) ** 2).sum(-1).mean()
            dead = since > dead_after
            if dead.any():  # AuxK: dead atoms learn to explain the residual error
                pd = pre.masked_fill(~dead, 0)
                va, ia = pd.topk(min(aux_k, int(dead.sum())), dim=-1)
                eh = sae.decode(va, ia) - sae.b_dec
                loss = loss + (1 / 32) * ((err.detach() - eh) ** 2).sum(-1).mean() / (err.detach() ** 2).sum(-1).mean()
            opt.zero_grad(set_to_none=True); loss.backward(); opt.step()
            with torch.no_grad():
                sae.W_dec /= sae.W_dec.norm(dim=0, keepdim=True)
                since += 1; since[i.flatten()] = 0
        with torch.no_grad():
            f = fvu_of(lambda x: sae(x)[0], XE)
        log(f"n={n:6d} k={k:3d} epoch {ep + 1}/{args.epochs}: eval FVU {f:.4f}  dead(>{dead_after} steps) {int((since > dead_after).sum())}")
    with torch.no_grad():
        fired = torch.zeros(n, device=dev)
        for a in range(0, len(XE), 8192):
            _, _, v, i = sae(XE[a : a + 8192].to(dev).float() * scale)
            fired.index_add_(0, i.flatten(), (v.flatten() > 0).float())
        freq = (fired / len(XE)).cpu().numpy()
        f_eval = fvu_of(lambda x: sae(x)[0], XE)
        f_train = fvu_of(lambda x: sae(x)[0], X[:len(XE)])
    r = {"n": n, "k": k, "eval_fvu": f_eval, "train_fvu": f_train,
         "dead_on_eval": float((freq == 0).mean()), "median_freq": float(np.median(freq)),
         "freq_hist": np.histogram(np.log10(freq + 1e-7), bins=30, range=(-7, 0))[0].tolist()}
    results.append(r)
    log(f"== n={n} k={k}: eval FVU {f_eval:.4f} (train {f_train:.4f}), dead on eval {r['dead_on_eval'] * 100:.1f}%")
    if str(n) in args.save.split(",") and k == args.k:
        torch.save({k: t.detach().half().cpu() for k, t in sae.state_dict().items()} | {"scale": scale, "k": args.k},
                   f"{args.data}/sae_{n}.pt")
    json.dump({"pca_eval_fvu": pca, "sae": results, "k": args.k, "n_train": N, "n_eval": len(XE)},
              open(f"{args.data}/sweep_results.json", "w"), indent=1)
    del sae, opt
    torch.cuda.empty_cache()
log("done")
