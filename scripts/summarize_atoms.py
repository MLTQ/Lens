"""Summarise the atom descriptions (autointerp) into a markdown report section.

    python scripts/summarize_atoms.py > runs/atoms_summary.md
"""

import json
import sys

import numpy as np
from scipy.stats import spearmanr

ai = json.load(open(sys.argv[1] if len(sys.argv) > 1 else "lenses/joint/autointerp.json"))
meta = json.load(open("lenses/joint/meta_joint.json"))
ctrl_path = "lenses/joint/autointerp_control.json"
try:
    ctrl = json.load(open(ctrl_path))
except FileNotFoundError:
    ctrl = None
layers = meta["layers"]
atoms = meta["atoms"]
rows = []
for k, r in ai["atoms"].items():
    a = atoms[int(k)]
    prof = a["layer_profile"]
    rows.append({"id": int(k), "desc": r.get("desc"), "auroc": r.get("auroc"), "freq": a["freq"],
                 "peak_layer": layers[int(np.argmax(prof))], "best_cos": a["best_cos"], "rec": r})
scored = [r for r in rows if r["auroc"] is not None]
au = np.array([r["auroc"] for r in scored])
print(f"## Can Bonsai describe its own non-verbal atoms?\n")
print(f"- Atoms processed: **{len(rows)}** of {meta['n_atoms']}; scored: **{len(scored)}** "
      f"(the rest fire too rarely in the fresh corpus to describe *and* test).")
print(f"- Describability (AUROC on 5 held-out firing vs 5 silent contexts; 0.5 = chance): "
      f"median **{np.median(au):.2f}**, quartiles {np.percentile(au, 25):.2f}–{np.percentile(au, 75):.2f}.")
print(f"- ≥ 0.9: **{(au >= 0.9).mean() * 100:.0f}%** · 0.7–0.9: {((au >= 0.7) & (au < 0.9)).mean() * 100:.0f}% · "
      f"< 0.7: **{(au < 0.7).mean() * 100:.0f}%** · ≤ 0.5 (no better than chance): {(au <= 0.5).mean() * 100:.0f}%")
if ctrl:
    print(f"- **Control** ({ctrl['n']} atoms re-scored with a *different* atom's description): median "
          f"**{ctrl['shuffled_median']:.2f}** (vs {ctrl['real_median']:.2f} with their own); ≥ 0.8 in "
          f"{ctrl['shuffled_ge_0.8'] * 100:.0f}% vs {ctrl['real_ge_0.8'] * 100:.0f}%. "
          f"Re-scoring reproducibility: max difference {max(abs(a - b) for a, b in ctrl['rescore_pairs']):.3f}.")
f = np.array([r["freq"] for r in scored]); pl = np.array([r["peak_layer"] for r in scored]); bc = np.array([r["best_cos"] for r in scored])
print(f"- Describability vs how often the atom fires: Spearman {spearmanr(au, f)[0]:+.2f}; vs peak layer: "
      f"{spearmanr(au, pl)[0]:+.2f}; vs best cosine to any word: {spearmanr(au, bc)[0]:+.2f}.")
for lo, hi in ((16, 32), (36, 48), (52, 62)):
    m = (pl >= lo) & (pl <= hi)
    if m.any():
        print(f"  - atoms peaking in L{lo}–{hi}: n={m.sum()}, median describability {np.median(au[m]):.2f}")


def show(r, n_ex=3):
    rec = r["rec"]
    ex = "\n".join(f"    - `{e}`" for e in rec.get("examples", [])[:n_ex])
    held = "\n".join(f"    - ✓ `{e}`" for e in rec.get("held_out", [])[:2])
    neg = "\n".join(f"    - ✗ `{e}`" for e in rec.get("negatives", [])[:1])
    return (f"- **ξ{r['id']}** — “{r['desc']}” · describability **{r['auroc']:.2f}** · fires on {r['freq'] * 100:.2f}% · "
            f"peaks at L{r['peak_layer']}\n  - strongest contexts:\n{ex}\n  - held-out test:\n{held}\n{neg}")


print("\n### Most describable\n")
for r in sorted(scored, key=lambda r: (-r["auroc"], -r["freq"]))[:6]:
    print(show(r, 2))
print("\n### Least describable (xeno-candidates, pending checks)\n")
print("Stable enough to fire in held-out text, but Bonsai's own description predicts them no better than chance. "
      "Caveat: this measures *Bonsai's* ability to describe from 12 examples, not human conceptual limits; "
      "many will be describable with more context, better prompts, or a stronger describer.\n")
for r in sorted(scored, key=lambda r: (r["auroc"], -r["freq"]))[:10]:
    print(show(r))
