"""Local J-lens viewer server.

    .venv/bin/python -m bonsai_lens.server --pack models/bonsai-mlx --lens lenses/bonsai27b-j.safetensors

Endpoints
  GET  /              viewer page
  GET  /api/info      model + lens metadata
  POST /api/run       {prompt, lens, k, generate} -> tokens + top-k readout per (layer, position);
                      generate=N first extends the prompt by N greedy tokens
  POST /api/ranks     {token_ids, lens} -> rank of each token in every cell of the last run
  POST /api/tokenize  {text} -> candidate token ids for pinning a word
  POST /api/translate {ids} -> English gloss per token id (gloss table, else the model itself)
"""

from __future__ import annotations

import argparse
import asyncio
import functools
import gzip
import json
import threading
import time
from pathlib import Path

import mlx.core as mx
import uvicorn
from fastapi import FastAPI
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from bonsai_lens.mlx_backend import BonsaiMLX, intervene

STATIC = Path(__file__).parent / "static"
app = FastAPI()
app.mount("/static", StaticFiles(directory=STATIC), name="static")


@app.middleware("http")
async def no_cache_for_code(request, call_next):
    """The page and its scripts change while you work: never let the browser reuse stale copies."""
    response = await call_next(request)
    if request.url.path == "/" or request.url.path.startswith("/static/"):
        response.headers["Cache-Control"] = "no-store"
    return response
_lock = threading.Lock()

# MLX binds GPU streams (and any lazily computed, cached arrays) to the thread that created them;
# FastAPI would otherwise run each request on an arbitrary pool thread, which fails with
# "There is no Stream(gpu, N) in current thread". All model work runs on this one thread.
from concurrent.futures import ThreadPoolExecutor

_MLX = ThreadPoolExecutor(max_workers=1, thread_name_prefix="mlx")


def on_mlx(fn):
    """Run a (sync) endpoint on the dedicated MLX thread; FastAPI still sees fn's signature."""
    @functools.wraps(fn)
    async def wrapper(*args, **kwargs):
        return await asyncio.get_running_loop().run_in_executor(_MLX, functools.partial(fn, *args, **kwargs))
    return wrapper
S: dict = {"model": None, "last": None}

# Token-id -> English gloss for the Qwen3.5 vocabulary, from anthropics/jacobian-lens
# (assets/qwen_gloss.json.gz, Apache-2.0). Covers ~92k of 248k tokens.
GLOSS: dict[int, str] = {
    int(k): v for k, v in json.load(gzip.open(Path(__file__).parent / "data" / "qwen_gloss.json.gz")).items()
}
_MODEL_GLOSS: dict[int, str] = {}


def _needs_gloss(text: str) -> bool:
    return any(ord(c) > 127 and c.isalpha() for c in text)


MAX_TOKENS = 256  # lens cost grows with sequence length (64 readouts per token)


class RunReq(BaseModel):
    prompt: str
    lens: str = "jacobian"
    k: int = 10
    generate: int = 0
    # Re-read an exact earlier sequence (e.g. prompt + continuation) with another
    # lens, without re-tokenizing; n_prompt keeps the generated boundary.
    ids: list[int] | None = None
    n_prompt: int | None = None


class RankReq(BaseModel):
    token_ids: list[int]
    lens: str = "jacobian"


class TokReq(BaseModel):
    text: str


class TransReq(BaseModel):
    ids: list[int]


def _layers(m: BonsaiMLX, lens: str) -> list[int]:
    if lens == "jacobian":
        return sorted(set(m.J) | {m.n_layers - 1})
    return list(range(m.n_layers))


@app.get("/")
def index():
    """The page, with module script URLs stamped by file mtime so edits are never served stale."""
    from fastapi.responses import HTMLResponse

    html = (STATIC / "index.html").read_text()
    for name in ("volume.js", "space.js"):
        html = html.replace(f'src="/static/{name}"', f'src="/static/{name}?v={int((STATIC / name).stat().st_mtime)}"')
    return HTMLResponse(html, headers={"Cache-Control": "no-store"})


@app.get("/api/info")
@on_mlx
def info():
    m: BonsaiMLX = S["model"]
    return {"n_layers": m.n_layers, "d_model": 5120, "lens": m.lens_info,
            "fitted_layers": sorted(m.J)}


@app.post("/api/run")
@on_mlx
def run(req: RunReq):
    m: BonsaiMLX = S["model"]
    with _lock:
        t0 = time.time()
        if req.ids:
            ids = req.ids[:MAX_TOKENS]
            n_prompt = min(req.n_prompt or len(ids), len(ids))
        else:
            ids = m.encode(req.prompt)[:MAX_TOKENS]
            n_prompt = len(ids)
        gen: list[int] = []
        gen_ms = 0
        if req.generate > 0 and ids:
            t1 = time.time()
            gen = m.generate(ids, min(req.generate, MAX_TOKENS - n_prompt))
            gen_ms = round((time.time() - t1) * 1000)
            ids = ids + gen
        hs = m.residuals(ids)
        S["last"] = {"ids": ids, "hs": hs, "n_prompt": n_prompt, "lens": req.lens}
        out = _payload(m, ids, hs, n_prompt, req.lens, req.k)
        out.update({
            "stopped_early": bool(req.generate) and len(gen) < min(req.generate, MAX_TOKENS - n_prompt),
            "gen_ms": gen_ms, "ms": round((time.time() - t0) * 1000),
        })
        return out


def _payload(m: BonsaiMLX, ids, hs, n_prompt, lens="jacobian", k=10):
    """Top-k lens readout for every (layer, position) of residuals ``hs``, as the page expects."""
    layers = _layers(m, lens)
    cells = []
    for l in layers:
        logits = m.readout(hs[l], l, lens)
        probs = mx.softmax(logits, axis=-1)
        top = mx.argpartition(-logits, k, axis=-1)[:, :k]
        p = mx.take_along_axis(probs, top, axis=-1)
        order = mx.argsort(-p, axis=-1)
        top, p = mx.take_along_axis(top, order, -1), mx.take_along_axis(p, order, -1)
        ent = -(probs * mx.log(probs + 1e-12)).sum(-1)
        mx.eval(top, p, ent)
        top, p, ent = top.tolist(), p.tolist(), ent.tolist()
        cells.append([
            {"ids": top[t], "p": [round(x, 4) for x in p[t]], "H": round(ent[t], 3)}
            for t in range(len(ids))
        ])
    vocab = {i for row in cells for c in row for i in c["ids"]} | set(ids)
    vocab_text = {i: m.decode_token(i) for i in vocab}
    return {
        "ids": ids,
        "n_prompt": n_prompt,
        "continuation": m.tok.decode(ids[n_prompt:]),
        "tokens": [m.decode_token(i) for i in ids],
        "layers": layers,
        "cells": cells,
        "vocab": vocab_text,
        "gloss": {i: _cached_gloss(i) for i, t in vocab_text.items() if _needs_gloss(t) and _cached_gloss(i)},
    }


@app.post("/api/ranks")
@on_mlx
def ranks(req: RankReq):
    m: BonsaiMLX = S["model"]
    last = S["last"]
    if last is None:
        return {"error": "run a prompt first"}
    with _lock:
        tid = mx.array(req.token_ids)
        out = {}
        layers = _layers(m, req.lens)
        per = []
        for l in layers:
            logits = m.readout(last["hs"][l], l, req.lens)  # [T, V]
            target = logits[:, tid]  # [T, n]
            r = (logits[:, :, None] > target[:, None, :]).sum(1)  # [T, n]
            mx.eval(r)
            per.append(r.tolist())
        for j, t in enumerate(req.token_ids):
            out[t] = [[per[li][p][j] for p in range(len(last["ids"]))] for li in range(len(layers))]
        return {"layers": layers, "ranks": out,
                "labels": {t: m.decode_token(t) for t in req.token_ids}}


@app.post("/api/reload_lens")
@on_mlx
def reload_lens():
    with _lock:
        S["model"].load_lens(S["lens_path"])
        S["last"] = None
    return info.__wrapped__()  # already on the MLX thread: call the plain function, not the async wrapper


def _cached_gloss(i: int) -> dict | None:
    if i in GLOSS:
        return {"en": GLOSS[i], "src": "gloss table"}
    if i in _MODEL_GLOSS:
        return {"en": _MODEL_GLOSS[i], "src": "Bonsai"}
    return None


@app.post("/api/translate")
@on_mlx
def translate(req: TransReq):
    m: BonsaiMLX = S["model"]
    out = {}
    for i in req.ids[:8]:
        g = _cached_gloss(i)
        if g is None:
            with _lock:
                _MODEL_GLOSS[i] = m.translate(m.decode_token(i)) or "?"
            g = _cached_gloss(i)
        out[i] = g
    return out


@app.get("/api/jspace")
@on_mlx
def jspace(layers: str = "8,16,24,32,40,48,56,62", k: int = 25, seed: int = 0):
    """How much of each activation (last run) lies in J-space: share of ||h||^2 captured by
    a non-negative combination of its top-k J-lens vectors, vs k random tokens' vectors,
    vs the 5 largest coordinates. Approximates the paper's gradient-pursuit decomposition."""
    import numpy as np

    m: BonsaiMLX = S["model"]
    last = S["last"]
    if last is None:
        return {"error": "run a prompt first"}
    rng = np.random.default_rng(seed)
    with _lock:
        hs, T = last["hs"], len(last["ids"])
        # self-check: reconstructed directions reproduce the model's own logits at the output
        h = hs[m.n_layers - 1][-1:].astype(mx.float32)
        ids = [int(i) for i in mx.argsort(-m.readout(h, m.n_layers - 1, "logit")[0])[:5].tolist()]
        lin = (h / mx.sqrt((h * h).mean(-1, keepdims=True) + 1e-6)) @ m.lens_dirs(m.n_layers - 1, ids).T
        ref = m.readout(h, m.n_layers - 1, "logit")[0, mx.array(ids)]
        check = float(mx.abs(lin[0] - ref).max() / mx.abs(ref).max())
        rows = []
        for l in [int(x) for x in layers.split(",")]:
            hl = hs[l]
            logits = m.readout(hl, l, "jacobian" if l < m.n_layers - 1 else "logit")
            top = mx.argsort(-logits, axis=-1)[:, :k].tolist()
            rand = [rng.integers(0, 248077, k).tolist() for _ in range(T)]
            fj = m.jspace_fraction(hl, l, top)
            fr = m.jspace_fraction(hl, l, rand)
            H = np.array(hl.astype(mx.float32))
            sq = H**2
            top5 = np.sort(sq, axis=-1)[:, -5:].sum(-1) / sq.sum(-1)
            skip = slice(min(1, T - 1), None)  # position 0 is an attention sink; report it apart
            rows.append({"layer": l, "jspace": float(np.median(fj[skip])), "jspace_max": float(np.max(fj[skip])),
                         "random": float(np.median(fr[skip])), "top5_dims": float(np.median(top5[skip])),
                         "pos0_jspace": fj[0], "pos0_top5_dims": float(top5[0]), "cells": fj})
        return {"k": k, "T": T, "self_check_rel_err": check, "layers": rows}


class KnockReq(BaseModel):
    token_id: int
    positions: list[int] | None = None   # None = every position (incl. generated ones)
    layers: list[int] = [20, 62]         # inclusive range of residual layers to ablate
    generate: int | None = None          # continuation length; default = last run's
    control: bool = True                 # also ablate a random direction as a control


@app.post("/api/knockout")
@on_mlx
def knockout(req: KnockReq):
    """Knock a concept out of the residual stream and measure what changes.

    At every layer in ``layers`` and every position in ``positions`` the positive
    component along the token's J-lens vector a_u = J_l^T (gamma * w_u) is projected out
    of the residual stream, and the forward pass continues from the edited state (later
    layers see the edit). We compare, against the unedited run: the model's next-token
    distribution at every position (same token sequence), and a fresh greedy
    continuation of the prompt. A random unit direction, ablated identically, is the
    control for generic disruption.
    """
    import numpy as np

    m: BonsaiMLX = S["model"]
    last = S["last"]
    if last is None:
        return {"error": "run a prompt first"}
    ids, n_prompt, out_l = last["ids"], last["n_prompt"], m.n_layers - 1
    lo, hi = max(0, req.layers[0]), min(m.n_layers - 2, req.layers[-1])
    n_gen = req.generate if req.generate is not None else max(len(ids) - n_prompt, 12)
    t0 = time.time()
    with _lock:
        dirs = {l: m.lens_dirs(l, [req.token_id])[0] for l in range(lo, hi + 1)}
        rng = np.random.default_rng(0)
        ctrl = {l: mx.array(rng.standard_normal(m.lm.model.norm.weight.shape[0]).astype(np.float32)) for l in dirs}

        def next_token_probs(hs):
            return mx.softmax(m.readout(hs[out_l], out_l, "logit"), axis=-1)

        def removed_share(hs_before, d):  # how much of ||h||^2 the edit took out, at ablated cells
            vals = []
            for l, v in d.items():
                u = v / mx.linalg.norm(v)
                h = hs_before[l].astype(mx.float32)
                c = mx.maximum((h * u).sum(-1), 0)
                share = (c * c) / (h * h).sum(-1)
                sel = share if req.positions is None else share[mx.array([p for p in req.positions if p < len(ids)])]
                vals.append(float(sel.mean()))
            return float(np.mean(vals))

        base_p = next_token_probs(last["hs"])
        with intervene(dirs, req.positions):
            hs_ko = m.residuals(ids)
            gen_ko = m.generate(ids[:n_prompt], n_gen)
        ko_p = next_token_probs(hs_ko)
        if req.control:
            with intervene(ctrl, req.positions):
                hs_c = m.residuals(ids)
                gen_c = m.generate(ids[:n_prompt], n_gen)
            c_p = next_token_probs(hs_c)
        base_gen = ids[n_prompt:] if len(ids) - n_prompt >= n_gen else m.generate(ids[:n_prompt], n_gen)

        def top(pr, t, k=5):
            row = pr[t]
            ix = mx.argsort(-row)[:k].tolist()
            return [[int(i), round(float(row[i]), 4)] for i in ix]

        rows = []
        for t in range(len(ids)):
            b = top(base_p, t)
            orig = b[0][0]
            r = {"t": t, "base": b, "ko": top(ko_p, t), "p_orig_ko": round(float(ko_p[t, orig]), 4),
                 "ablated": req.positions is None or t in req.positions}
            if req.control:
                r["ctrl"] = top(c_p, t)
                r["p_orig_ctrl"] = round(float(c_p[t, orig]), 4)
            rows.append(r)
        vocab = {i for r in rows for key in ("base", "ko", "ctrl") for i, _ in r.get(key, [])}
        vocab |= set(base_gen) | set(gen_ko) | (set(gen_c) if req.control else set())
        run = _payload(m, ids, hs_ko, n_prompt, last.get("lens", "jacobian"))
        return {
            "token_id": req.token_id, "token": m.decode_token(req.token_id),
            "layers": [lo, hi], "positions": req.positions,
            "removed_share": removed_share(last["hs"], dirs),
            "removed_share_ctrl": removed_share(last["hs"], ctrl) if req.control else None,
            "rows": rows,
            "continuation": {"base": m.tok.decode(base_gen), "ko": m.tok.decode(gen_ko),
                             "ctrl": m.tok.decode(gen_c) if req.control else None},
            "vocab": {i: m.decode_token(i) for i in vocab},
            "run": run,
            "ms": round((time.time() - t0) * 1000),
        }


@app.get("/api/massive")
@on_mlx
def massive(top: int = 6):
    """Residual-stream coordinates that carry the most squared norm in the last run, per layer
    (the 'massive activation' dims), with their share of ||h||^2 and sign, position 0 apart."""
    import numpy as np

    last = S["last"]
    if last is None:
        return {"error": "run a prompt first"}
    H = np.array(last["hs"].astype(mx.float32))  # [L, T, d]
    out = []
    for l in range(H.shape[0]):
        rest = H[l, 1:] if H.shape[1] > 1 else H[l]
        e = (rest**2).sum(0)
        share = e / e.sum()
        ix = np.argsort(-share)[:top]
        e0 = H[l, 0] ** 2
        out.append({"layer": l, "dims": [{"dim": int(i), "share": round(float(share[i]), 4),
                                          "mean": round(float(rest[:, i].mean()), 1)} for i in ix],
                    "pos0_top": [int(i) for i in np.argsort(-e0)[:top]],
                    "pos0_share": round(float(np.sort(e0)[-top:].sum() / e0.sum()), 3)})
    return {"layers": out}


@app.get("/api/nonverbal")
@on_mlx
def nonverbal(lo: int = 16, hi: int = 62, top: int = 4):
    """Per cell of the last run (layers lo..hi): verbal / privileged / non-verbal shares and the
    strongest non-verbal atoms. The atom dictionary was trained on layers 16-62."""
    m: BonsaiMLX = S["model"]
    last = S["last"]
    if last is None:
        return {"error": "run a prompt first"}
    if getattr(m, "sae", None) is None:
        return {"error": "no atom dictionary loaded (lenses/atoms/sae.pt)"}
    with _lock:
        t0 = time.time()
        layers = list(range(max(lo, 0), min(hi, m.n_layers - 2) + 1))
        cells = {l: m.nonverbal(last["hs"][l], l, top_atoms=top) for l in layers}
        return {"layers": layers, "cells": cells, "ms": round((time.time() - t0) * 1000)}


_SPACE: dict = {}


@app.get("/api/space_nn")
def space_nn(id: int, k: int = 30):
    """Nearest tokens to ``id`` by cosine in the PCA-128 readout space (the layout
    distorts distances; these are the neighbours before UMAP)."""
    import numpy as np

    if "Z" not in _SPACE:
        Z = np.load(Path(S["space_dir"]) / "pca128.npy").astype(np.float32)
        Z /= np.linalg.norm(Z, axis=1, keepdims=True) + 1e-8
        _SPACE["Z"] = Z
    Z = _SPACE["Z"]
    if not 0 <= id < len(Z):
        from fastapi import HTTPException
        raise HTTPException(400, f"id {id} is not a vocabulary token (atoms have their own cards)")
    sims = Z @ Z[id]
    top = np.argpartition(-sims, k + 1)[: k + 1]
    top = top[np.argsort(-sims[top])]
    return {"id": id, "neighbors": [{"id": int(j), "sim": round(float(sims[j]), 3)} for j in top if j != id][:k]}


@app.post("/api/tokenize")
@on_mlx
def tokenize(req: TokReq):
    m: BonsaiMLX = S["model"]
    cands = []
    for v in (req.text, " " + req.text.strip(), req.text.strip().capitalize(),
              " " + req.text.strip().capitalize()):
        ids = m.encode(v)
        if ids and ids[0] not in [c["id"] for c in cands]:
            cands.append({"id": ids[0], "text": m.decode_token(ids[0]), "n_pieces": len(ids)})
    return {"candidates": cands}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pack", default="models/bonsai-mlx")
    ap.add_argument("--lens", default="lenses/bonsai27b-j.safetensors")
    ap.add_argument("--port", type=int, default=8765)
    ap.add_argument("--space", default="lenses/space", help="output dir of scripts/embed_vocab.py")
    args = ap.parse_args()
    t0 = time.time()
    S["model"] = _MLX.submit(BonsaiMLX, args.pack).result()  # created on the MLX thread
    S["lens_path"] = args.lens
    S["space_dir"] = args.space
    # Token-space files (meta.json, coords*.f32); the page reports if they are missing.
    app.mount("/space", StaticFiles(directory=args.space, check_dir=False), name="space")
    # Joint token + non-verbal-atom layouts (scripts/embed_joint.py); optional.
    app.mount("/joint", StaticFiles(directory=str(Path(args.space).parent / "joint"), check_dir=False), name="joint")
    if Path(args.lens).exists():
        _MLX.submit(S["model"].load_lens, args.lens).result()
    atoms = Path(args.space).parent / "atoms" / "sae.pt"
    if atoms.exists():
        _MLX.submit(S["model"].load_atoms, atoms).result()
    print(f"ready in {time.time() - t0:.0f}s; lens={S['model'].lens_info}", flush=True)
    uvicorn.run(app, host="127.0.0.1", port=args.port, log_level="warning")


if __name__ == "__main__":
    main()
