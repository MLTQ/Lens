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


def _install_recorder():
    from mlx_vlm.models.qwen3_5 import language

    cls = language.Qwen3_5DecoderLayer
    if getattr(cls, "_lens_patched", False):
        return
    orig = cls.__call__

    def __call__(self, x, *args, **kwargs):
        out = orig(self, x, *args, **kwargs)
        if _RECORD is not None:
            _RECORD.append(out)
        return out

    cls.__call__ = __call__
    cls._lens_patched = True


class BonsaiMLX:
    def __init__(self, pack: str | Path):
        pack = Path(pack)
        sys.path.insert(0, str(pack / "runtime"))
        from vision_artifact import load_vl_model
        from tokenizers import Tokenizer

        _install_recorder()
        model, _, _ = load_vl_model(pack, load_processor=False)
        self.lm = model.language_model
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

    def residuals(self, ids: list[int]) -> mx.array:
        """[n_layers, T, d] residual stream after each block (fp16)."""
        global _RECORD
        _RECORD = []
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
        cache = self.lm.make_cache()
        x = mx.array([ids], dtype=mx.int32)
        out: list[int] = []
        for _ in range(n):
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
