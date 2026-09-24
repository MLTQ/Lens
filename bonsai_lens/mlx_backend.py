"""Jacobian-lens readout on Apple Silicon via the Prism MLX pack.

Loads the pack through its own ``vision_artifact.load_vl_model`` (the official
path, which applies the Hadamard transforms and GDN head layout), records the
residual stream after every decoder block, and decodes

    lens_l(h) = lm_head( norm( J_l h ) )

with J_l fitted in PyTorch (see ``bonsai_lens.fit``) and exported as fp16
safetensors (``scripts/export_lens.py``). ``J = None`` gives the logit lens.
"""

from __future__ import annotations

import sys
from pathlib import Path

import mlx.core as mx
import numpy as np

_RECORD: list | None = None
# Active intervention: {"dirs": {layer: unit vector [d] fp32}, "positions": set[int] | None (= all)}.
# ``_OFFSET`` is the absolute position of the first token of the current forward call
# (0 for a full pass; the cache length during incremental generation).
_INTERVENE: dict | None = None
_OFFSET = 0


def _install_recorder():
    from mlx_vlm.models.qwen3_5 import language

    cls = language.Qwen3_5DecoderLayer
    if getattr(cls, "_lens_patched", False):
        return
    orig = cls.__call__

    def __call__(self, x, *args, **kwargs):
        out = orig(self, x, *args, **kwargs)
        if _INTERVENE is not None and self._lens_idx in _INTERVENE["dirs"]:
            out = _ablate(out, _INTERVENE["dirs"][self._lens_idx], _INTERVENE["positions"])
        if _RECORD is not None:
            _RECORD.append(out)
        return out

    cls.__call__ = __call__
    cls._lens_patched = True


def _ablate(h: mx.array, d: mx.array, positions) -> mx.array:
    """Remove the positive component of ``h`` along unit direction ``d`` (a knockout:
    the concept is taken out where present; an anti-aligned residual is left alone)."""
    L = h.shape[1]
    coef = mx.maximum((h.astype(mx.float32) * d).sum(-1, keepdims=True), 0.0)  # [B, L, 1]
    if positions is not None:
        pos = mx.arange(_OFFSET, _OFFSET + L)
        mask = mx.array([int(p) in positions for p in pos.tolist()])[None, :, None]
        coef = coef * mask
    return (h.astype(mx.float32) - coef * d).astype(h.dtype)


class BonsaiMLX:
    def __init__(self, pack: str | Path):
        pack = Path(pack)
        sys.path.insert(0, str(pack / "runtime"))
        from vision_artifact import load_vl_model
        from tokenizers import Tokenizer

        _install_recorder()
        model, _, _ = load_vl_model(pack, load_processor=False)
        self.lm = model.language_model
        for i, layer in enumerate(self.lm.model.layers):
            layer._lens_idx = i
        from runtime import fwht as prism_fwht
        self._fwht = prism_fwht
        self.tok = Tokenizer.from_file(str(pack / "tokenizer.json"))
        self.n_layers = len(self.lm.model.layers)
        self.d_model = self.lm.model.embed_tokens.weight.shape[0] and 5120
        self.J: dict[int, mx.array] = {}
        self.lens_info: dict = {}

    # ------------------------------------------------------------------ lens
    def load_lens(self, path: str | Path):
        from safetensors import safe_open

        J = {}
        with safe_open(str(path), framework="numpy") as f:
            meta = f.metadata() or {}
            for k in f.keys():
                J[int(k.split(".")[1])] = mx.array(f.get_tensor(k)).astype(mx.float16)
        mx.eval(list(J.values()))
        self.J = J
        self.lens_info = {"path": str(path), **meta, "layers": sorted(J)}

    # --------------------------------------------------------------- forward
    def encode(self, text: str) -> list[int]:
        return self.tok.encode(text, add_special_tokens=False).ids

    def decode_token(self, i: int) -> str:
        return self.tok.decode([int(i)], skip_special_tokens=False)

    # ------------------------------------------------------------ directions
    def readout_rows(self, ids: list[int]) -> mx.array:
        """Effective unembedding rows [n, d] (fp32) in the native residual basis:
        logit_u = w_u . x for the (post-norm) input x of the head."""
        head, idx = self.lm.lm_head, mx.array(ids)
        w = mx.dequantize(head.weight[idx], head.scales[idx], head.biases[idx],
                          group_size=128, bits=2).astype(mx.float32)
        return self._fwht(w, head.block, head.signs, inverse=True) if head.block else w

    def lens_dirs(self, layer: int, ids: list[int]) -> mx.array:
        """J-lens vectors for ``ids`` at ``layer`` ([n, d] fp32): a_u = J_l^T (gamma * w_u),
        the direction in layer-l residual space whose lens logit for token u it is."""
        g = self.readout_rows(ids) * self.lm.model.norm.weight.astype(mx.float32)
        if layer < self.n_layers - 1:
            g = g @ self.J[layer].astype(mx.float32)
        return g

    def jspace_fraction(self, h: mx.array, layer: int, support: list[list[int]], iters: int = 4):
        """Share of ||h_t||^2 captured by a non-negative combination of the J-lens vectors
        in ``support[t]`` (least squares, negatives clamped and refit). h: [T, d]."""
        import numpy as np

        H = np.array(h.astype(mx.float32), dtype=np.float64)
        out = []
        for t, ids in enumerate(support):
            A = np.array(self.lens_dirs(layer, ids), dtype=np.float64)  # [k, d]
            keep = np.arange(len(ids))
            fit = lambda k: np.linalg.solve(A[k] @ A[k].T + 1e-6 * np.eye(len(k)), A[k] @ H[t])
            c = fit(keep)
            for _ in range(iters):  # drop negative coefficients and refit
                if (c >= 0).all() or not (c > 0).any():
                    break
                keep = keep[c > 0]
                c = fit(keep)
            c = np.maximum(c, 0)  # any residual negatives after the last refit are clamped
            rec = A[keep].T @ c
            out.append(float(rec @ rec / (H[t] @ H[t])))
        return out

    # -------------------------------------------------------------- forward
    def residuals(self, ids: list[int]) -> mx.array:
        """[n_layers, T, d] residual stream after each block (fp16)."""
        global _RECORD, _OFFSET
        _RECORD = []
        _OFFSET = 0
        try:
            x = mx.array([ids], dtype=mx.int32)
            self.lm.model(x, cache=self.lm.make_cache())  # mlx-vlm attention needs a cache
            hs = mx.stack([h[0] for h in _RECORD])
            mx.eval(hs)
        finally:
            _RECORD = None
        return hs

    _TRANSLATE_PROMPT = (
        "Translate each token into English. A token may be a word fragment; "
        "give its closest English meaning in a few words.\n\n"
        "猫 → cat\nмаленький → small\nです → is (polite copula)\n"
        "بيت → house\n的事 → the matter of\n{text} →"
    )

    STOP_IDS = (248044, 248046)  # <|endoftext|>, <|im_end|>

    def generate(self, ids: list[int], n: int, stop=None) -> list[int]:
        """Greedy (argmax) continuation of ``ids`` for up to ``n`` tokens, raw
        completion: no chat template, no sampling. Stops early on end-of-text
        or when ``stop(token_id)`` is true (the stopping token is not kept)."""
        global _OFFSET
        cache = self.lm.make_cache()
        x = mx.array([ids], dtype=mx.int32)
        out: list[int] = []
        for step in range(n):
            _OFFSET = 0 if step == 0 else len(ids) + step - 1
            nxt = int(mx.argmax(self.lm(x, cache=cache).logits[0, -1]).item())
            if nxt in self.STOP_IDS or (stop and stop(nxt)):
                break
            out.append(nxt)
            x = mx.array([[nxt]], dtype=mx.int32)
        return out

    def translate(self, text: str, max_tokens: int = 10) -> str:
        """Greedy few-shot gloss of ``text`` by the model itself."""
        ids = self.encode(self._TRANSLATE_PROMPT.format(text=text.strip()))
        out = self.generate(ids, max_tokens, stop=lambda i: "\n" in self.decode_token(i))
        return self.tok.decode(out).strip()

    def readout(self, h: mx.array, layer: int, lens: str = "jacobian") -> mx.array:
        """h: [T, d] at ``layer`` -> logits [T, vocab] (fp32)."""
        if lens == "jacobian" and layer < self.n_layers - 1:
            J = self.J.get(layer)
            if J is None:
                raise KeyError(f"no fitted J for layer {layer}")
            h = h.astype(mx.float16) @ J.T
        logits = self.lm.lm_head(self.lm.model.norm(h.astype(mx.float16)))
        return logits.astype(mx.float32)


class intervene:
    """Context manager: knock out ``dirs`` ({layer: vector}) at ``positions`` (None = all)."""

    def __init__(self, dirs: dict, positions=None):
        self.cfg = {"dirs": {l: v / mx.linalg.norm(v) for l, v in dirs.items()},
                    "positions": None if positions is None else set(positions)}

    def __enter__(self):
        global _INTERVENE
        _INTERVENE = self.cfg
        return self

    def __exit__(self, *exc):
        global _INTERVENE
        _INTERVENE = None
