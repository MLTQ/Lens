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

from bonsai_lens.mlx_backend import BonsaiMLX

STATIC = Path(__file__).parent / "static"
app = FastAPI()
app.mount("/static", StaticFiles(directory=STATIC), name="static")
_lock = threading.Lock()
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
    return FileResponse(STATIC / "index.html")


@app.get("/api/info")
def info():
    m: BonsaiMLX = S["model"]
    return {"n_layers": m.n_layers, "d_model": 5120, "lens": m.lens_info,
            "fitted_layers": sorted(m.J)}


@app.post("/api/run")
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
        layers = _layers(m, req.lens)
        cells = []
        for l in layers:
            logits = m.readout(hs[l], l, req.lens)
            probs = mx.softmax(logits, axis=-1)
            top = mx.argpartition(-logits, req.k, axis=-1)[:, : req.k]
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
        S["last"] = {"ids": ids, "hs": hs}
        return {
            "ids": ids,
            "n_prompt": n_prompt,
            "continuation": m.tok.decode(gen) if gen else m.tok.decode(ids[n_prompt:]),
            "stopped_early": bool(req.generate) and len(gen) < min(req.generate, MAX_TOKENS - n_prompt),
            "gen_ms": gen_ms,
            "tokens": [m.decode_token(i) for i in ids],
            "layers": layers,
            "cells": cells,
            "vocab": vocab_text,
            "gloss": {i: _cached_gloss(i) for i, t in vocab_text.items()
                      if _needs_gloss(t) and _cached_gloss(i)},
            "ms": round((time.time() - t0) * 1000),
        }


@app.post("/api/ranks")
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
def reload_lens():
    with _lock:
        S["model"].load_lens(S["lens_path"])
        S["last"] = None
    return info()


def _cached_gloss(i: int) -> dict | None:
    if i in GLOSS:
        return {"en": GLOSS[i], "src": "gloss table"}
    if i in _MODEL_GLOSS:
        return {"en": _MODEL_GLOSS[i], "src": "Bonsai"}
    return None


@app.post("/api/translate")
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
    sims = Z @ Z[id]
    top = np.argpartition(-sims, k + 1)[: k + 1]
    top = top[np.argsort(-sims[top])]
    return {"id": id, "neighbors": [{"id": int(j), "sim": round(float(sims[j]), 3)} for j in top if j != id][:k]}


@app.post("/api/tokenize")
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
    S["model"] = BonsaiMLX(args.pack)
    S["lens_path"] = args.lens
    S["space_dir"] = args.space
    # Token-space files (meta.json, coords*.f32); the page reports if they are missing.
    app.mount("/space", StaticFiles(directory=args.space, check_dir=False), name="space")
    if Path(args.lens).exists():
        S["model"].load_lens(args.lens)
    print(f"ready in {time.time() - t0:.0f}s; lens={S['model'].lens_info}", flush=True)
    uvicorn.run(app, host="127.0.0.1", port=args.port, log_level="warning")


if __name__ == "__main__":
    main()
