"""Canonical token space for the J-lens (step 2 of the J-Volume plan).

The J-lens transports every layer into the final-layer basis and decodes with
the model's own readout, logit_v = u_v . x_normed with u_v = gamma * W_eff[v].
So all 64 layers share one space, in which token v sits at its readout
direction u_v. We embed every vocabulary row:

    u_v -> unit-normalize -> PCA(128) -> kNN graph (cosine) -> UMAP 3D and 2D

and store per-token metadata (string, script/category, readout norm) so
padding, under-trained and fragment regions are visible rather than filtered.

    python scripts/embed_vocab.py --out lenses/space
"""

import argparse
import json
import os
import sys
import time
import unicodedata

import numpy as np
import torch

sys.path.insert(0, ".")
from bonsai_lens.ternary import TernaryLinear, transcode

p = argparse.ArgumentParser()
p.add_argument("--gguf", default="models/bonsai-gguf/Ternary-Bonsai-2-27B-PQ2_0.gguf")
p.add_argument("--tokenizer", default="models/bonsai-meta/tokenizer.json")
p.add_argument("--out", default="lenses/space")
p.add_argument("--pca", type=int, default=128)
p.add_argument("--neighbors", type=int, default=30)
args = p.parse_args()
os.makedirs(args.out, exist_ok=True)
torch.set_num_threads(10)
t0 = time.time()
log = lambda m: print(f"[{time.time() - t0:6.0f}s] {m}", flush=True)

# ---------------------------------------------------------------- readout matrix
from gguf import GGUFReader

reader = GGUFReader(args.gguf)
fields = {k: v.contents() for k, v in reader.fields.items()}
tensors = {t.name: t for t in reader.tensors}
head, norm_t = tensors["output.weight"], tensors["output_norm.weight"]
rows, width = (int(n) for n in head.shape[::-1])
packed, scales = transcode(head.data.tobytes(), rows, width, head.tensor_type.name)
block = int(fields["prism.hadamard.block_size"])
assert "output.weight" in set(fields["prism.hadamard.weight_names"])
widths = fields["prism.hadamard.sign_widths"]
values = np.asarray(fields["prism.hadamard.sign_values"], dtype=np.float32)
off = 0
for w in widths:
    if w == width:
        signs = torch.from_numpy(values[off : off + w].copy())
    off += w
# Stored value is (1 + w_hf) for zero-centered norms, i.e. exactly the gain applied.
gamma = torch.from_numpy(np.asarray(norm_t.data, dtype=np.float32).copy())
lin = TernaryLinear(torch.from_numpy(packed), torch.from_numpy(scales), block, signs, torch.float32)

U = np.empty((rows, width), dtype=np.float32)
for a in range(0, rows, 16384):
    sl = TernaryLinear(lin.packed[a : a + 16384], lin.scales[a : a + 16384], block, signs, torch.float32)
    U[a : a + 16384] = (sl.dense() * gamma).numpy()
log(f"readout matrix {U.shape}, gamma mean {gamma.mean():.3f}")

# Self-check: the effective rows must reproduce the real (rotated, packed) head.
x = torch.randn(3, width)
sub = TernaryLinear(lin.packed[:512], lin.scales[:512], block, signs, torch.float32)
ref = sub(x * gamma)
err = (ref - x @ torch.from_numpy(U[:512]).T).abs().max() / ref.abs().max()
log(f"self-check vs packed forward: rel max err {err:.2e}")
assert err < 1e-4, "effective readout rows do not match the model's head"

norms = np.linalg.norm(U, axis=1)
U /= np.maximum(norms[:, None], 1e-8)

# ---------------------------------------------------------------- token metadata
from tokenizers import Tokenizer

tok = Tokenizer.from_file(args.tokenizer)
n_tok = tok.get_vocab_size(with_added_tokens=True)
strings, cats = [], []
SCRIPTS = [("CJK", "cjk"), ("HIRAGANA", "japanese"), ("KATAKANA", "japanese"), ("HANGUL", "korean"),
           ("CYRILLIC", "cyrillic"), ("ARABIC", "arabic"), ("HEBREW", "hebrew"), ("GREEK", "greek"),
           ("THAI", "thai"), ("DEVANAGARI", "indic"), ("BENGALI", "indic"), ("TAMIL", "indic"),
           ("TELUGU", "indic"), ("GUJARATI", "indic"), ("KANNADA", "indic"), ("MALAYALAM", "indic"),
           ("GURMUKHI", "indic"), ("ARMENIAN", "other script"), ("GEORGIAN", "other script"),
           ("ETHIOPIC", "other script"), ("MYANMAR", "other script"), ("KHMER", "other script"),
           ("LAO", "other script"), ("TIBETAN", "other script"), ("SINHALA", "other script")]
CODE = set("_{}()[];=<>/\\`$#@|&*^~%\"'")
special = set(tok.get_added_tokens_decoder().keys()) if hasattr(tok, "get_added_tokens_decoder") else set()


def category(i: int, s: str) -> str:
    if i >= n_tok:
        return "padding"
    if i in special or (s.startswith("<|") and s.endswith("|>")):
        return "special"
    if "�" in s:
        return "byte fragment"
    core = s.strip()
    if not core:
        return "whitespace"
    for c in core:
        if c.isalpha() and ord(c) > 127:
            name = unicodedata.name(c, "")
            for key, cat in SCRIPTS:
                if key in name:
                    return cat
            if "LATIN" in name:
                return "latin (accented)"
            return "other script"
    if core.isdigit():
        return "digits"
    if not any(c.isalnum() for c in core):
        return "punctuation/symbol"
    if any(c in CODE for c in core) or ("." in core and len(core) > 1 and not core.endswith(".")):
        return "code/markup"
    return "english/latin"


for i in range(rows):
    s = tok.decode([i], skip_special_tokens=False) if i < n_tok else ""
    strings.append(s)
    cats.append(category(i, s))
counts = {c: cats.count(c) for c in sorted(set(cats))}
log("categories: " + json.dumps(counts))

# ---------------------------------------------------------------- PCA -> kNN -> UMAP
Ut = torch.from_numpy(U)
mean = Ut.mean(0, keepdim=True)
_, S, V = torch.pca_lowrank(Ut, q=args.pca, center=True, niter=4)
Z = ((Ut - mean) @ V[:, : args.pca]).numpy()
ev = (S**2) / (S**2).sum()
log(f"PCA {Z.shape}; top-128 carry (of the sampled spectrum) first 5: {np.round(ev[:5].numpy(), 3).tolist()}")
del Ut, U

import umap
from umap.umap_ import nearest_neighbors

knn = nearest_neighbors(Z, n_neighbors=args.neighbors, metric="cosine", metric_kwds=None,
                        angular=False, random_state=None, n_jobs=12, low_memory=True)
log("kNN graph done")
out = {}
for dim in (3, 2):
    reducer = umap.UMAP(n_components=dim, n_neighbors=args.neighbors, min_dist=0.08, metric="cosine",
                        precomputed_knn=knn, n_jobs=12, low_memory=True, verbose=False)
    Y = reducer.fit_transform(Z).astype(np.float32)
    Y -= Y.mean(0)
    Y /= np.abs(Y).max()
    out[dim] = Y
    log(f"UMAP {dim}D done")

np.save(f"{args.out}/coords3.npy", out[3])
np.save(f"{args.out}/coords2.npy", out[2])
np.save(f"{args.out}/pca128.npy", Z.astype(np.float16))
out[3].tofile(f"{args.out}/coords3.f32")
out[2].tofile(f"{args.out}/coords2.f32")
cat_names = sorted(set(cats))
meta = {
    "n": rows, "n_tokenizer": n_tok, "space": "unit(gamma * W_eff[v]) -> PCA128 -> UMAP (cosine, k=%d)" % args.neighbors,
    "categories": cat_names, "category_counts": counts,
    "cat": [cat_names.index(c) for c in cats],
    "norm": np.round(norms, 3).tolist(),
    "tokens": strings,
}
with open(f"{args.out}/meta.json", "w") as f:
    json.dump(meta, f, ensure_ascii=False)
log(f"wrote {args.out}")
