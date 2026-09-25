"""Exact per-source decomposition of Bonsai's token mixers, for the attention views.

Bonsai (Qwen3.5 hybrid) mixes information across positions in two ways:

* 16 softmax-attention layers (every 4th: 3, 7, ... 63): 24 query heads sharing 4 KV heads,
  per-dimension sigmoid output gate. Output at t, head h:  sigma(g_th) * sum_s A_h[t,s] v_s.
* 48 Gated DeltaNet layers: per head a 128x128 memory S, updated each token by
      S_t = S_{t-1} A_t + beta_t v_t k_t^T,   A_t = g_t (I - beta_t k_t k_t^T)
  and read after the update, y_t = S_t q_t. Unrolled, y_t = sum_s w[t,s] v_s with the exact
  "effective attention"  w[t,s] = beta_s k_s^T (A_{s+1} ... A_t) q_t  (signed; includes all
  decay and delta-rule overwrites in between). Then a gated RMSNorm: silu(z_t) * rms_norm(y_t).

In both cases, for a fixed destination t the mixer output is linear in the sources:

    contribution(t <- s, head h) = W_O,h [ w_h[t,s] * (D_h[t] (.) v_h[s]) ]

with D = sigmoid gate (attention) or weight * silu(z_t) / rms(y_t) (DeltaNet). Summing over s and
h reproduces the layer's mixer output exactly (checked per layer). Contribution sizes use the
per-head Gram matrix W_O,h^T W_O,h, so no residual-size vectors are built until they are decoded.
"""

from __future__ import annotations

import math

import mlx.core as mx
import mlx.nn as nn
import numpy as np


def _dense(module) -> mx.array:
    """Effective [out, in] weight of a (possibly Prism-packed, Hadamard-folded) linear layer."""
    if hasattr(module, "block") and hasattr(module, "scales"):
        from runtime import fwht  # the pack's own runtime (on sys.path via BonsaiMLX)

        w = mx.dequantize(module.weight, module.scales, module.biases, group_size=128, bits=2).astype(mx.float32)
        return fwht(w, module.block, module.signs, inverse=True) if module.block else w
    return module.weight.astype(mx.float32)


class MixerAnalysis:
    def __init__(self, m):
        self.m = m
        self.layers = m.lm.model.layers
        self._gram: dict[int, np.ndarray] = {}

    # ---------------------------------------------------------------- structure
    def kind(self, l: int) -> str:
        return "gdn" if self.layers[l].is_linear else "attn"

    def describe(self, l: int) -> dict:
        if self.kind(l) == "gdn":
            a = self.layers[l].linear_attn
            return {"kind": "gdn", "heads": a.num_v_heads, "key_heads": a.num_k_heads, "dim": a.head_v_dim}
        a = self.layers[l].self_attn
        return {"kind": "attn", "heads": a.num_attention_heads, "kv_heads": a.num_key_value_heads, "dim": a.head_dim}

    def out_module(self, l: int):
        return self.layers[l].linear_attn.out_proj if self.kind(l) == "gdn" else self.layers[l].self_attn.o_proj

    def gram(self, l: int) -> np.ndarray:
        """Per-head W_O,h^T W_O,h, [H, dv, dv] (cached)."""
        if l not in self._gram:
            W = _dense(self.out_module(l))  # [5120, H*dv]
            d = self.describe(l)
            H, dv = d["heads"], d["dim"]
            Wh = W.reshape(W.shape[0], H, dv).transpose(1, 2, 0)  # [H, dv, 5120]
            G = Wh @ Wh.transpose(0, 2, 1)
            mx.eval(G)
            self._gram[l] = np.array(G, dtype=np.float32)
        return self._gram[l]

    # ---------------------------------------------------------------- one layer
    def mixer(self, x_in: mx.array, l: int) -> dict:
        """Decompose layer l's mixer for residual input x_in [T, d].

        Returns numpy arrays: w [H, T, T] (weights; lower-triangular), V [H, T, dv] (values per
        head, already mapped to query/value heads), D [H, T, dv] (destination scaling), and
        ref [T, d] (the layer's own mixer output, for the exactness check)."""
        layer = self.layers[l]
        xn = layer.input_layernorm(x_in[None].astype(mx.float16))  # [1, T, d]
        T = x_in.shape[0]
        if layer.is_linear:
            g = layer.linear_attn
            mixed, z, b, a = g.in_proj_qkv(xn), g.in_proj_z(xn), g.in_proj_b(xn), g.in_proj_a(xn)
            conv_in = mx.concatenate([mx.zeros((1, g.conv_kernel_size - 1, g.conv_dim), dtype=mixed.dtype), mixed], axis=1)
            conv = nn.silu(g.conv1d(conv_in))
            q, k, v = [t_.reshape(1, T, h, d) for t_, h, d in zip(
                mx.split(conv, [g.key_dim, 2 * g.key_dim], -1),
                [g.num_k_heads, g.num_k_heads, g.num_v_heads], [g.head_k_dim, g.head_k_dim, g.head_v_dim])]
            inv = g.head_k_dim ** -0.5
            q = (inv**2) * mx.fast.rms_norm(q, None, 1e-6)
            k = inv * mx.fast.rms_norm(k, None, 1e-6)
            rep = g.num_v_heads // g.num_k_heads
            q, k = mx.repeat(q, rep, -2), mx.repeat(k, rep, -2)  # value head h reads key head h // rep
            decay = mx.exp(-mx.exp(g.A_log.astype(mx.float32)) * nn.softplus(a.astype(mx.float32) + g.dt_bias))  # [1,T,Hv]
            beta = mx.sigmoid(b.astype(mx.float32))
            Q = np.array(q[0].astype(mx.float32)).transpose(1, 0, 2)  # [H, T, dk]
            K = np.array(k[0].astype(mx.float32)).transpose(1, 0, 2)
            V = np.array(v[0].astype(mx.float32)).transpose(1, 0, 2)  # [H, T, dv]
            Gd = np.array(decay[0]).T  # [H, T]
            Bt = np.array(beta[0]).T
            H = Q.shape[0]
            w = np.zeros((H, T, T), np.float32)
            R = np.zeros_like(Q)  # R[:, t] carries (A_{s+1} ... A_t) q_t as s descends
            for s in range(T - 1, -1, -1):
                R[:, s] = Q[:, s]
                kr = np.einsum("hd,htd->ht", K[:, s], R[:, s:])  # k_s . R_t for t >= s
                w[:, s:, s] = Bt[:, s, None] * kr
                R[:, s:] = Gd[:, s, None, None] * (R[:, s:] - Bt[:, s, None, None] * kr[..., None] * K[:, s, None, :])
            Y = np.einsum("hts,hsd->htd", w, V)  # the memory read y_t, per head
            rms = np.sqrt((Y**2).mean(-1, keepdims=True) + g.norm.eps)
            Z = np.array(z.reshape(1, T, g.num_v_heads, g.head_v_dim)[0].astype(mx.float32)).transpose(1, 0, 2)  # [H, T, dv]
            silu = Z / (1 + np.exp(-Z))
            D = np.array(g.norm.weight.astype(mx.float32))[None, None] * silu / rms
            ref = g(xn)[0]
            QK = (Q, K)  # per value head, after conv, norm and key-head sharing
        else:
            at = layer.self_attn
            qo, keys, values = at.q_proj(xn), at.k_proj(xn), at.v_proj(xn)
            queries, gate = mx.split(qo.reshape(1, T, at.num_attention_heads, -1), 2, axis=-1)
            queries = at.q_norm(queries).transpose(0, 2, 1, 3)
            keys = at.k_norm(keys.reshape(1, T, at.num_key_value_heads, -1)).transpose(0, 2, 1, 3)
            values = values.reshape(1, T, at.num_key_value_heads, -1).transpose(0, 2, 1, 3)
            pos = mx.tile(mx.arange(T)[None, None, :], (3, 1, 1))
            queries, keys = at.rotary_emb.apply_rotary(queries, keys, pos, unsqueeze_dim=1)
            rep = at.num_attention_heads // at.num_key_value_heads  # query head h uses KV head h // rep
            kf = mx.repeat(keys, rep, 1).astype(mx.float32)
            s_ = (queries.astype(mx.float32) @ kf.transpose(0, 1, 3, 2)) * at.scale
            causal = mx.triu(mx.full((T, T), -mx.inf), k=1)
            A = mx.softmax(s_ + causal, axis=-1)[0]  # [H, T, T]
            w = np.array(A, dtype=np.float32)
            V = np.array(mx.repeat(values, rep, 1)[0].astype(mx.float32))  # [H, T, dv]
            D = np.array(mx.sigmoid(gate.astype(mx.float32))[0]).transpose(1, 0, 2)  # [H, T, dv]
            from mlx_vlm.models.cache import KVCache

            ref = at(xn, mask="causal", cache=KVCache())[0]
            QK = (np.array(queries[0].astype(mx.float32)), np.array(kf[0]))  # after norm + RoPE, [H, T, d]
        mx.eval(ref)
        return {"w": w, "V": V, "D": D, "ref": np.array(ref.astype(mx.float32)), "Q": QK[0], "K": QK[1]}

    def reconstruct(self, l: int, dec: dict) -> np.ndarray:
        """Sum of all per-source contributions through the layer's real output projection [T, d]."""
        U = np.einsum("hts,hsd->thd", dec["w"], dec["V"]) * dec["D"].transpose(1, 0, 2)  # [T, H, dv]
        out = self.out_module(l)(mx.array(U.reshape(U.shape[0], -1)).astype(mx.float16))
        return np.array(out.astype(mx.float32))

    def norms(self, l: int, dec: dict) -> np.ndarray:
        """||contribution(t <- s, head h)|| for all h, t, s: [H, T, T] (0 above the diagonal)."""
        G = mx.array(self.gram(l))
        V, D, w = mx.array(dec["V"]), mx.array(dec["D"]), dec["w"]
        H, T, dv = dec["V"].shape
        out = np.zeros((H, T, T), np.float32)
        for h in range(H):
            Yh = D[h][:, None, :] * V[h][None, :, :]  # [T(t), T(s), dv]
            q = ((Yh @ G[h]) * Yh).sum(-1)
            out[h] = np.sqrt(np.maximum(np.array(q), 0))
        return np.abs(w) * out

    def contributions(self, l: int, dec: dict, t: int, sources, heads=None) -> np.ndarray:
        """Actual residual-space vectors for destination t: one row per (head or None=all, s)."""
        H, T, dv = dec["V"].shape
        rows = []
        for h, s in sources:
            u = np.zeros((H, dv), np.float32)
            hs_ = range(H) if h is None else [h]
            for hh in hs_:
                u[hh] = dec["w"][hh, t, s] * dec["D"][hh, t] * dec["V"][hh, s]
            rows.append(u.reshape(-1))
        out = self.out_module(l)(mx.array(np.stack(rows)).astype(mx.float16))
        return out.astype(mx.float32)


def key_basis(dec: dict, h: int) -> np.ndarray:
    """Head h's two main directions of key variation over the whole sequence [2, d] (cached in dec).
    Fixed per head so a key's angle stays put as the query moves from token to token."""
    cache = dec.setdefault("kbasis", {})
    if h not in cache:
        Kk = dec["K"][h]
        if len(Kk) >= 3:
            _, _, Vt = np.linalg.svd(Kk - Kk.mean(0), full_matrices=False)
            cache[h] = Vt[:2]
        else:
            cache[h] = np.eye(2, Kk.shape[1], dtype=np.float32)
    return cache[h]


def kv_geometry(dec: dict, h: int, t: int, scale: float) -> dict:
    """Keys of head h (positions 0..t) in a frame whose x axis is the query of t.

    x_s = q_t . k_s / |q_t| (so x * |q_t| * scale is exactly the attention logit for softmax heads);
    y, z = the head's fixed main directions of key variation (key_basis), made orthogonal to the query."""
    q = dec["Q"][h, t]
    Kk = dec["K"][h, : t + 1]
    qn = np.linalg.norm(q) + 1e-9
    e1 = q / qn
    x = Kk @ e1
    rest = Kk - np.outer(x, e1)
    B = key_basis(dec, h)
    B = B - np.outer(B @ e1, e1)  # Gram-Schmidt against the query, keeping the basis' orientation
    b2 = B[0] / (np.linalg.norm(B[0]) + 1e-9)
    b3 = B[1] - (B[1] @ b2) * b2
    b3 = b3 / (np.linalg.norm(b3) + 1e-9)
    y, z = rest @ b2, rest @ b3
    return {"x": x.tolist(), "y": y.tolist(), "z": z.tolist(), "q_norm": float(qn),
            "logit": (x * qn * scale).tolist(), "weight": dec["w"][h, t, : t + 1].tolist()}
