"""Exact isolation of every point: 1 - cosine to its nearest other point, full 5120-d space."""
import json, sys, numpy as np, torch
sys.path.insert(0, ".")
from bonsai_lens.ternary import TernaryLinear, transcode
from gguf import GGUFReader
r = GGUFReader("models/bonsai-gguf/Ternary-Bonsai-2-27B-PQ2_0.gguf")
f = {k: v.contents() for k, v in r.fields.items()}; t = {x.name: x for x in r.tensors}
h, nt = t["output.weight"], t["output_norm.weight"]
rows, width = (int(n) for n in h.shape[::-1])
pk, sc = transcode(h.data.tobytes(), rows, width, h.tensor_type.name)
vals = np.asarray(f["prism.hadamard.sign_values"], dtype=np.float32); off = 0
for w in f["prism.hadamard.sign_widths"]:
    if w == width: signs = torch.from_numpy(vals[off:off+w].copy())
    off += w
g = torch.from_numpy(np.asarray(nt.data, dtype=np.float32).copy()).cuda()
P, S = torch.from_numpy(pk).cuda(), torch.from_numpy(sc).cuda()
U = torch.empty(rows, width, dtype=torch.float16, device="cuda")
for a in range(0, rows, 16384):
    u = TernaryLinear(P[a:a+16384], S[a:a+16384], int(f["prism.hadamard.block_size"]), signs.cuda(), torch.float32).dense() * g
    U[a:a+len(u)] = (u / u.norm(dim=-1, keepdim=True)).half()
del P, S
A = torch.from_numpy(np.load("lenses/atoms/atoms.npy")).cuda(); A = (A / A.norm(dim=-1, keepdim=True)).half()
E = torch.zeros(1, width, device="cuda", dtype=torch.float16); E[0, json.load(open("lenses/atoms/privileged.json"))[0]["dim"]] = 1
X = torch.cat([U, A, E]); N = len(X); NT = rows; NA = len(A)
iso = np.empty(N, np.float32)
for a in range(0, N, 4096):
    s = (X[a:a+4096] @ X.T).float(); s[torch.arange(len(s)), torch.arange(a, a+len(s), device="cuda")] = -2
    iso[a:a+len(s)] = (1 - s.max(1).values).cpu().numpy()
iso.tofile("lenses/joint/isolation.f32")
w, at = iso[:NT], iso[NT:NT + NA]
print(f"isolation (1 - cosine to nearest other point): words median {np.median(w):.3f} (p10 {np.percentile(w,10):.3f}, p90 {np.percentile(w,90):.3f}) | atoms median {np.median(at):.3f} (p10 {np.percentile(at,10):.3f}, p90 {np.percentile(at,90):.3f})")
print(f"words more isolated than the median atom: {(w > np.median(at)).mean()*100:.1f}%")
json.dump({"words_median": float(np.median(w)), "atoms_median": float(np.median(at)),
           "words_p90": float(np.percentile(w, 90)), "atoms_p10": float(np.percentile(at, 10)),
           "words_above_atom_median": float((w > np.median(at)).mean())}, open("lenses/joint/isolation.json", "w"))
