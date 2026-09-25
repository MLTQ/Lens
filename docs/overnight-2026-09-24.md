# Overnight report — 24 Sep 2026

You asked: how many non-verbal atoms are there really, and can the map be made to *mean more, and mean it more accurately*? Short answers first, details after.

## TL;DR

1. **The non-verbal part is not a small set of concepts.** On held-out text, dictionaries plateau at ~57% of the remainder explained around 4k atoms (bigger ones only memorize), while allowing *more atoms at once* keeps helping (65% with 128 active). The remainder is **high-dimensional and densely coded**: many things at once, not a few discrete ones.
2. **The maps are coarse atlases, and now they say so.** Only ~5–8% of a word's true nearest neighbours are its nearest on screen (~20–24% within its 150 nearest). A measured bake-off of projection settings improves local accuracy by ~60% at best; no 3D layout gets close to faithful. New: a per-point *layout fidelity* colour, a *local-faithful* view, and region labels shown only where a region is actually compact.
3. **Bonsai described its own 2,048 atoms, and the descriptions were tested.** Median describability 0.78 (0.5 = chance); a shuffled-description control drops to 0.48, so the test measures real fit. Re-tested with 3× the evidence, **~14% of atoms stay at chance**: 39 robust *xeno-candidates* are flagged in the viewer (search `candidates`). They are what *Bonsai* can't describe — not yet what no human can; some are nameable on inspection.
4. **Non-verbal atoms now appear in the trace and playback** (lime), next to the word branches.

## 1. How many non-verbal atoms are there?

Data: 1,600 fresh FineWeb documents (never used by the lens or the first dictionary), 1,400 to train / 200 held out; 7 layers (16–62); ~1.1M training vectors. Same remainder definition as before (activation → lens basis → minus its best non-negative combination of top-25 lens words → d3994 zeroed).

| Dictionary | Active per activation | **Held-out explained** | Train explained | Dead on held-out |
|---|---|---|---|---|
| 1,024 | 24 | 54.7% | 57.7% | 0.1% |
| 2,048 | 24 | 56.2% | 60.5% | 1.9% |
| 4,096 | 24 | **57.1%** | 62.8% | 6.0% |
| 8,192 | 24 | 57.0% | 63.7% | 12.4% |
| 16,384 | 24 | 56.3% | 63.4% | 24.3% |
| 2,048 | 64 | 61.5% | 65.3% | 0% |
| 8,192 | 64 | 62.7% | 69.3% | 0% |
| 8,192 | 128 | **65.4%** | 71.4% | 0% |

Linear reference (plain principal directions): 24 → 38%, 128 → 53%, 512 → 66%, **1,024 → 74%**.

**Reading it.** More atoms stop helping after ~4k (training fit keeps rising while held-out falls — memorization; a quarter of 16k atoms never fire on new text). More *simultaneous* atoms keeps helping, and 2k atoms × 64 active beats 16k × 24. Even 1,024 plain directions leave a quarter unexplained. So: the reliably findable repertoire *at this data scale* is a few thousand directions, but each activation uses many of them at once and a sizable part stays unexplained by any sparse code — dense, distributed structure (or noise; ternary quantization adds some). With 10–100× more data the plateau would likely move; that's the obvious next experiment. (A 32k dictionary ran out of GPU memory; skipped since 16k was already worse than 8k.)

## 2. How accurate is the map?

Per point: of its 15 true nearest neighbours (in the space the view is built from), how many are among its 15 nearest on screen (*exact*) and its 150 nearest (*neighbourhood*). Random layout: ~0.006% / ~0.06%.

| View | Exact | Neighbourhood |
|---|---|---|
| A 3D (current) | 5.0% | 20.2% |
| **A 3D local-faithful (new)** | **8.2%** | **24.4%** |
| A 2D | 1.9% | 12.2% |
| B UMAP (words / atoms) | 5.7% / 5.1% | 19.6% / 25.6% |
| B densMAP | 3.8% / 2.2% | 14.1% / 14.4% |
| C floor | 2.0% / 2.7% | 11.9% / 15.6% |

Bake-off on the word map (exact / neighbourhood / global rank-correlation of distances): current UMAP 5.0 / 20.2 / 0.40 · **UMAP k=15, min_dist 0: 8.2 / 24.4 / 0.38** · UMAP k=30, min_dist 0, 1000 epochs: 7.3 / 23.6 / 0.38 · UMAP k=50: 3.9 / 18.1 / **0.41** · PaCMAP: 2.7 / 12.9 / 0.37. Local and global accuracy trade off; nothing is close to faithful. **The vocabulary is too high-dimensional for any 3D picture — trust regions, not exact neighbours; click a point for its true neighbours.** I added the best local layout as a separate view rather than replacing the one you like.

## 3. What the map now says

- **Colour → layout fidelity**: red = this point's neighbourhood is misplaced in this picture; green = trustworthy.
- **Region labels** (256 k-means regions of the vocabulary, named by Bonsai): shown only if the region is compact in the current view, fading as it scatters; max 40; they recede behind an active trace. Examples: *Past participles* (producido, cambiato, produzido, recebido — cross-lingual), *Programming identifiers*, *Closing punctuation*, *Japanese sentence endings*, *Korean particles*, *Modal particles* (Chinese).
- **Atom descriptions**: every atom card shows Bonsai's description and its describability score (with the held-out test contexts, ✓ firing / ✗ silent). Colour → atom describability. In views B / densMAP / C, search also matches atom descriptions (try `vulgar`, `legal`, `paragraph`).
- **About this map** (ⓘ in the Space controls): this view's accuracy, the bake-off, the dictionary sweep, the describability results with control, and the separation statistics.

### Describability, carefully

- Method: Bonsai sees an atom's 12 strongest contexts (marked token + ~28 tokens of context) and writes a ≤12-word description; then, on contexts it never saw, judges "does the description fit?" (logit Yes − No). Score = AUROC firing-vs-silent. 1.0 perfect, 0.5 chance.
- All atoms (5 + 5 test contexts): median **0.78**; ≥ 0.9: 32%; ≤ 0.5: 19%.
- **Control** (120 atoms scored with a *different* atom's description): median **0.48** → the score measures real fit. But 17% of mismatched descriptions still reach 0.8, so single first-pass scores are noisy.
- **Re-test** (15 + 15 fresh contexts): 50 lowest scorers **0.49** (shuffled 0.47); 50 random atoms **0.78** (shuffled 0.54). Describable atoms stay describable; the low scorers really are at chance.
- **39 robust candidates** (at chance on re-test, no better than a random description): 32 from the lowest 50 and 7 from the random 50 — i.e. roughly **14% of atoms (~280 of 2,048)** are beyond Bonsai's own description.
- Crucial caveat: *Bonsai can't describe it* ≠ *no human concept exists*. Reading contexts, some are nameable: **ξ1211** fires on infinitival "to" after a noun ("opportunity **to** enjoy", "the idea **to** help", "families **to** visit") while Bonsai guessed "before a new list item". Bonsai's failure mode is falling back to "function words or punctuation". The candidate list is the right *starting set* for a xeno hunt; the next filter is a stronger describer (or you).
- Weak hint worth testing: atoms peaking in earlier layers (L16–32, n = 28) are less describable (median 0.57) than late-peaking ones (0.78). Small sample.

Best described (first-pass scores 0.98–1.00): ξ971 *nouns in legal or criminal contexts*, ξ8 *line breaks before a new paragraph*, ξ1963 *linking verbs*, ξ748 *sentence-final periods*, ξ216 *first/third-person singular pronouns*, ξ224 *explicit or vulgar contexts*, ξ949 *parts of academic degree titles*.

## 4. Non-verbal atoms in the trace and playback

Space tab → view B / densMAP / C → trace controls → **atoms ≥ 1% of activation**. The server splits every cell of the run into verbal part / d3994 / non-verbal remainder (in the lens basis) and encodes the remainder; the strongest atoms become lime branches into each position's prediction, and pop (lime) during playback. The 44 always-on atoms (firing on >10% of activations, e.g. the word-continuation ones) are hidden as background machinery. On the boot prompt: 206 word branches + 93 atom branches. Hover a lime node for its share, layers, and Bonsai's description.

## Housekeeping

- The 4090 box is idle; **QwenPlayground is still stopped** (restart from `~/Code/QwenPlayground`: `.venv/bin/python -u main.py --listen 0.0.0.0 --port 8188 --cuda-device 0`).
- New data on the GPU box: `lenses/sweep/` (~13 GB of remainder vectors — delete if you need the space; `sweep_results.json` is copied locally), `lenses/atoms/autointerp*.json`.
- New scripts: `sweep_collect.py`, `sweep_sae.py`, `map_quality.py`, `layout_bakeoff.py`, `autointerp.py`, `autointerp_control.py`, `autointerp_rescore.py`, `summarize_atoms.py`; endpoint `/api/nonverbal`. README updated.
- Viewer changes load on a normal page reload (scripts are version-stamped now).

## Suggested next steps

1. **Human pass on the 39 candidates** (search `candidates`): which ones can you name? Every one you can moves from X to H — that boundary-moving is the paper's programme in miniature.
2. **Stronger describer** on the candidates (e.g. Claude with longer contexts and contrastive examples), then re-test; what survives is a much better xeno list.
3. **Atom knockouts** (the step we deferred): does removing a candidate atom change behaviour specifically? Describable + causal, or undescribable + causal, is where it gets interesting.
4. **10× data** for the dictionary sweep, to see whether the plateau at ~4k atoms moves.
