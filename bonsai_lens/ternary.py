"""Load Prism ternary (PQ2_0 / PTQ1_0) Qwen3.5 GGUF checkpoints into a
HuggingFace ``Qwen3_5ForCausalLM`` that supports autograd w.r.t. activations.

Weights stay packed on the GPU (2 bits/weight) and are dequantized per call, in
both the forward and the backward pass, so the retained autograd graph never
holds full-precision weight copies. The Prism checkpoints fold a blockwise
normalized Sylvester-Hadamard rotation (with explicit signs) into the *input*
dimension of most matrices; we apply the matching rotation to activations, so
the residual stream stays in the model's native, unrotated basis. That matters:
the Jacobian lens is defined on the residual stream.

Mirrors ``runtime/runtime.py`` from prism-ml/Ternary-Bonsai-2-27B-mlx-2bit.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import torch
from torch import nn

BLOCK_BYTES = {"PQ2_0": 34, "PTQ1_0": 28}


# --------------------------------------------------------------------------- #
# Packing
# --------------------------------------------------------------------------- #


def _ptq1_codes(data: np.ndarray) -> np.ndarray:
    """Dense trit packing -> [blocks, 128] uint8 codes in {0,1,2}."""
    pieces = []
    for lo, hi, count in [(0, 16, 5), (16, 24, 5), (24, 26, 4)]:
        packed = data[:, lo:hi].astype(np.uint16)
        for trit in range(count):
            remainder = (packed * (3**trit)) & 255
            pieces.append(((remainder * 3) >> 8).astype(np.uint8))
    return np.concatenate(pieces, axis=1)


def transcode(raw: np.ndarray, rows: int, width: int, source: str):
    """Return (packed uint8 [rows, width/4], fp16 scales [rows, width/128]).

    Each packed byte holds four 2-bit codes, code k at bits 2k..2k+1, in natural
    column order. Weight = (code - 1) * scale.
    """
    blocks = rows * width // 128
    data = np.frombuffer(raw, dtype=np.uint8).reshape(blocks, BLOCK_BYTES[source])
    if source == "PQ2_0":
        scales = data[:, :2].copy().view("<f2").reshape(rows, width // 128)
        packed = data[:, 2:].copy().reshape(rows, width // 4)
    else:
        scales = data[:, 26:28].copy().view("<f2").reshape(rows, width // 128)
        codes = _ptq1_codes(data).reshape(rows, width // 4, 4).astype(np.uint8)
        packed = (codes << np.array([0, 2, 4, 6], dtype=np.uint8)).sum(
            -1, dtype=np.uint8
        )
    if not np.isfinite(scales).all():
        raise ValueError("non-finite quantization scale")
    return np.ascontiguousarray(packed), np.ascontiguousarray(scales)


_SHIFTS = {}

try:
    import triton
    import triton.language as tl

    @triton.jit
    def _dequant_kernel(packed_ptr, scales_ptr, out_ptr, n_bytes, BLOCK: tl.constexpr):
        # One program handles BLOCK packed bytes = 4*BLOCK weights. A 128-weight
        # quant group is 32 bytes, so byte i belongs to group i // 32.
        pid = tl.program_id(0)
        offs = pid * BLOCK + tl.arange(0, BLOCK)
        m = offs < n_bytes
        b = tl.load(packed_ptr + offs, mask=m, other=0).to(tl.int32)
        sc = tl.load(scales_ptr + offs // 32, mask=m, other=0).to(tl.float32)
        for k in tl.static_range(4):
            c = ((b >> (2 * k)) & 3).to(tl.float32) - 1.0
            tl.store(out_ptr + offs * 4 + k, (c * sc).to(out_ptr.dtype.element_ty), mask=m)

    HAVE_TRITON = True
except ImportError:  # pragma: no cover
    HAVE_TRITON = False


def dequantize(packed: torch.Tensor, scales: torch.Tensor, dtype=torch.bfloat16):
    """[rows, width/4] uint8 + [rows, width/128] fp16 -> [rows, width] ``dtype``."""
    rows = packed.shape[0]
    if HAVE_TRITON and packed.is_cuda:
        packed, scales = packed.contiguous(), scales.contiguous()
        out = torch.empty(rows, packed.shape[1] * 4, dtype=dtype, device=packed.device)
        n = packed.numel()
        _dequant_kernel[(triton.cdiv(n, 1024),)](packed, scales, out, n, BLOCK=1024)
        return out
    dev = packed.device
    if dev not in _SHIFTS:
        _SHIFTS[dev] = torch.tensor([0, 2, 4, 6], dtype=torch.uint8, device=dev)
    codes = (packed.unsqueeze(-1) >> _SHIFTS[dev]) & 3  # [rows, w/4, 4]
    # (code - 1) in {-1, 0, 1} is exact in any float dtype, so scaling in
    # ``dtype`` rounds exactly like scaling in fp32 then casting.
    w = codes.reshape(rows, -1, 128).to(dtype) - 1
    w = w * scales.to(dtype).unsqueeze(-1)
    return w.reshape(rows, -1)


#: Max weights dequantized at once; bounds transient memory (~2 bytes/weight).
CHUNK_WEIGHTS = 1 << 27


def _row_chunks(packed: torch.Tensor):
    rows, width = packed.shape[0], packed.shape[1] * 4
    step = max(1, CHUNK_WEIGHTS // width)
    return [(r, min(r + step, rows)) for r in range(0, rows, step)]


def ternary_matmul(x, packed, scales):
    """x @ W^T, dequantizing W in row chunks."""
    chunks = _row_chunks(packed)
    if len(chunks) == 1:
        return x @ dequantize(packed, scales, x.dtype).T
    return torch.cat(
        [x @ dequantize(packed[a:b], scales[a:b], x.dtype).T for a, b in chunks], -1
    )


def ternary_matmul_t(g, packed, scales):
    """g @ W (the input-gradient of x @ W^T), dequantizing W in row chunks."""
    chunks = _row_chunks(packed)
    if len(chunks) == 1:  # a full-range slice is an alias op, which vmap rejects
        return g @ dequantize(packed, scales, g.dtype)
    out = None
    for a, b in chunks:
        part = g[..., a:b] @ dequantize(packed[a:b], scales[a:b], g.dtype)
        out = part if out is None else out + part
    return out


# --------------------------------------------------------------------------- #
# Hadamard
# --------------------------------------------------------------------------- #

_HADAMARD = {}


def hadamard_matrix(n: int, device, dtype=torch.float32) -> torch.Tensor:
    """Normalized Sylvester Hadamard matrix (symmetric, orthonormal, self-inverse)."""
    key = (n, device, dtype)
    if key not in _HADAMARD:
        h = torch.ones(1, 1, dtype=torch.float64)
        while h.shape[0] < n:
            h = torch.cat([torch.cat([h, h], 1), torch.cat([h, -h], 1)], 0)
        _HADAMARD[key] = (h / math.sqrt(n)).to(device=device, dtype=dtype)
    return _HADAMARD[key]


def fwht(x: torch.Tensor, block: int, signs: torch.Tensor, inverse: bool = False):
    """Blockwise rotation matching Prism's runtime ``fwht``.

    forward:  H (signs * x);   inverse:  signs * (H x)

    Runs in the activation dtype on tensor cores; H entries (+-1/sqrt(block))
    and signs are exact in bf16 and accumulation is fp32 inside the GEMM.
    """
    shape, dtype = x.shape, x.dtype
    h = hadamard_matrix(block, x.device, dtype)
    s = signs.to(dtype)
    if not inverse:
        x = x * s
    x = (x.reshape(*shape[:-1], shape[-1] // block, block) @ h).reshape(shape)
    if inverse:
        x = x * s
    return x


# --------------------------------------------------------------------------- #
# Modules
# --------------------------------------------------------------------------- #


class _TernaryMatmul(torch.autograd.Function):
    """y = x @ W^T with W re-dequantized in backward (never saved)."""

    @staticmethod
    def forward(x, packed, scales):
        return ternary_matmul(x, packed, scales)

    @staticmethod
    def setup_context(ctx, inputs, output):
        _, packed, scales = inputs
        ctx.save_for_backward(packed, scales)

    @staticmethod
    def backward(ctx, grad_out):
        packed, scales = ctx.saved_tensors
        return ternary_matmul_t(grad_out, packed, scales), None, None

    # Lets torch.autograd.grad(..., is_grads_batched=True) vmap over cotangents.
    generate_vmap_rule = True


class TernaryLinear(nn.Module):
    def __init__(self, packed, scales, block=0, signs=None, dtype=torch.bfloat16):
        super().__init__()
        self.register_buffer("packed", packed)
        self.register_buffer("scales", scales)
        self.register_buffer("signs", signs if signs is not None else torch.empty(0))
        # jlens reads ``.weight.device/.dtype`` off the LM head / embedding.
        self.register_buffer("weight", torch.empty(0, dtype=dtype, device=packed.device))
        self.block = block
        self.out_features, self.in_features = packed.shape[0], packed.shape[1] * 4

    def forward(self, x):
        if self.block:
            x = fwht(x, self.block, self.signs)
        return _TernaryMatmul.apply(x, self.packed, self.scales)

    def dense(self, dtype=torch.float32) -> torch.Tensor:
        """Effective weight in the unrotated input basis: W_eff = W' H D.

        forward computes W' (H (s * x)) = (s * (H w'_row)) . x per output row, so each
        row is the *inverse* rotation of the stored row."""
        w = dequantize(self.packed, self.scales, torch.float32)
        if self.block:
            w = fwht(w, self.block, self.signs.float(), inverse=True)
        return w.to(dtype)

    def extra_repr(self):
        return f"{self.in_features}->{self.out_features}, ternary, hadamard={self.block}"


class TernaryEmbedding(nn.Module):
    def __init__(self, packed, scales, block=0, signs=None, dtype=torch.bfloat16):
        super().__init__()
        self.register_buffer("packed", packed)
        self.register_buffer("scales", scales)
        self.register_buffer("signs", signs if signs is not None else torch.empty(0))
        self.register_buffer("weight", torch.empty(0, dtype=dtype, device=packed.device))
        self.block, self.dtype = block, dtype
        self.num_embeddings, self.embedding_dim = packed.shape[0], packed.shape[1] * 4

    def forward(self, ids):
        flat = ids.reshape(-1)
        out = dequantize(self.packed[flat], self.scales[flat], self.dtype)
        out = out.reshape(*ids.shape, -1)
        return fwht(out, self.block, self.signs, inverse=True) if self.block else out


# --------------------------------------------------------------------------- #
# Loader
# --------------------------------------------------------------------------- #

_LAYER_MAP = {
    "attn_norm.weight": "input_layernorm.weight",
    "post_attention_norm.weight": "post_attention_layernorm.weight",
    "ffn_gate.weight": "mlp.gate_proj",
    "ffn_up.weight": "mlp.up_proj",
    "ffn_down.weight": "mlp.down_proj",
    "attn_q.weight": "self_attn.q_proj",
    "attn_k.weight": "self_attn.k_proj",
    "attn_v.weight": "self_attn.v_proj",
    "attn_output.weight": "self_attn.o_proj",
    "attn_q_norm.weight": "self_attn.q_norm.weight",
    "attn_k_norm.weight": "self_attn.k_norm.weight",
    "attn_qkv.weight": "linear_attn.in_proj_qkv",
    "attn_gate.weight": "linear_attn.in_proj_z",
    "ssm_alpha.weight": "linear_attn.in_proj_a",
    "ssm_beta.weight": "linear_attn.in_proj_b",
    "ssm_out.weight": "linear_attn.out_proj",
    "ssm_norm.weight": "linear_attn.norm.weight",
    "ssm_a": "linear_attn.A_log",
    "ssm_dt.bias": "linear_attn.dt_bias",
    "ssm_conv1d.weight": "linear_attn.conv1d.weight",
}
_GLOBAL_MAP = {
    "output.weight": "lm_head",
    "output_norm.weight": "model.norm.weight",
    "token_embd.weight": "model.embed_tokens",
}
# HF Qwen3_5RMSNorm computes x * (1 + w); llama.cpp converters store 1 + w.
_ZERO_CENTERED = (
    "input_layernorm.weight",
    "post_attention_layernorm.weight",
    "model.norm.weight",
    "q_norm.weight",
    "k_norm.weight",
)


def _set(root: nn.Module, dotted: str, value):
    *parents, leaf = dotted.split(".")
    mod = root
    for p in parents:
        mod = mod[int(p)] if p.isdigit() else getattr(mod, p)
    if isinstance(value, nn.Module):
        setattr(mod, leaf, value)
    else:
        old = getattr(mod, leaf)
        if isinstance(old, nn.Linear):  # unquantized small projection
            mod, leaf, old = old, "weight", old.weight
        if tuple(old.shape) != tuple(value.shape):
            raise ValueError(f"{dotted}: shape {tuple(value.shape)} != {tuple(old.shape)}")
        setattr(mod, leaf, nn.Parameter(value, requires_grad=False))


def load_bonsai(
    gguf_path: str | Path,
    config_path: str | Path,
    *,
    device: str | torch.device = "cuda",
    dtype: torch.dtype = torch.bfloat16,
    layer_devices: dict[int, str] | None = None,
    verbose: bool = True,
):
    """Build a ``Qwen3_5ForCausalLM`` from a Prism ternary GGUF.

    ``layer_devices`` optionally maps block index -> device for splitting the
    stack across GPUs (embedding follows block 0, norm/head follow the last).
    """
    from gguf import GGUFReader
    from transformers import Qwen3_5ForCausalLM, Qwen3_5TextConfig

    reader = GGUFReader(str(gguf_path))
    fields = {k: v.contents() for k, v in reader.fields.items()}
    if fields.get("general.architecture") != "qwen35":
        raise ValueError(f"expected qwen35 GGUF, got {fields.get('general.architecture')}")
    tensors = {t.name: t for t in reader.tensors}

    cfg = json.loads(Path(config_path).read_text())
    cfg = cfg.get("text_config", cfg)
    cfg.pop("model_type", None)
    config = Qwen3_5TextConfig(**cfg)
    config._attn_implementation = "sdpa"

    with torch.device("meta"):
        model = Qwen3_5ForCausalLM(config)

    n_layers = config.num_hidden_layers
    layer_devices = layer_devices or {}
    def dev_for(name: str):
        if name.startswith("blk."):
            return torch.device(layer_devices.get(int(name.split(".")[1]), device))
        if name == "token_embd.weight":
            return torch.device(layer_devices.get(0, device))
        return torch.device(layer_devices.get(n_layers - 1, device))

    block = int(fields["prism.hadamard.block_size"])
    widths = fields["prism.hadamard.sign_widths"]
    values = np.asarray(fields["prism.hadamard.sign_values"], dtype=np.float32)
    sign_np, off = {}, 0
    for w in widths:
        sign_np[int(w)] = values[off : off + w]
        off += w
    assert off == len(values)
    signs_cache: dict = {}
    def signs(width, dev):
        if (width, dev) not in signs_cache:
            signs_cache[(width, dev)] = torch.from_numpy(sign_np[width]).to(dev)
        return signs_cache[(width, dev)]

    folded = set(fields["prism.hadamard.weight_names"])
    inverse = set(fields.get("prism.hadamard.inverse_weight_names", []))
    assert inverse == {"token_embd.weight"}, inverse
    if not fields.get("prism.hadamard.gdn_v_grouped", False):
        raise ValueError("ungrouped GDN output layout not supported")

    nv, nk = config.linear_num_value_heads, config.linear_num_key_heads
    hd, hk = config.linear_value_head_dim, config.linear_key_head_dim

    def vperm(unit):
        return np.arange(nv * unit).reshape(nv // nk, nk, unit).transpose(1, 0, 2).reshape(-1)

    def reorder(a, stem):
        if nv == nk:
            return a
        if stem in ("attn_qkv.weight", "ssm_conv1d.weight"):
            qk = 2 * nk * hk
            return np.concatenate([a[:qk], a[qk:][vperm(hd)]], axis=0)
        if stem == "attn_gate.weight":
            return a[vperm(hd)]
        if stem in ("ssm_alpha.weight", "ssm_beta.weight", "ssm_a", "ssm_dt.bias"):
            return a[vperm(1)]
        return a

    norm_means = []
    assigned = set()
    for i, (name, t) in enumerate(tensors.items()):
        if name in _GLOBAL_MAP:
            stem, target = name, _GLOBAL_MAP[name]
        elif name.startswith("blk."):
            _, layer, stem = name.split(".", 2)
            if stem not in _LAYER_MAP:
                raise ValueError(f"unmapped tensor {name}")
            target = f"model.layers.{layer}.{_LAYER_MAP[stem]}"
        else:
            raise ValueError(f"unmapped tensor {name}")
        dev = dev_for(name)
        kind = t.tensor_type.name
        if kind in BLOCK_BYTES:
            rows, width = (int(n) for n in t.shape[::-1])
            packed, scales = transcode(t.data.tobytes() if hasattr(t.data, "tobytes") else bytes(t.data), rows, width, kind)
            packed, scales = reorder(packed, stem), reorder(scales, stem)
            active = name in folded or name in inverse
            args = (
                torch.from_numpy(np.ascontiguousarray(packed)).to(dev),
                torch.from_numpy(np.ascontiguousarray(scales)).to(dev),
                block if active else 0,
                signs(width, dev) if active else None,
                dtype,
            )
            mod = TernaryEmbedding(*args) if name in inverse else TernaryLinear(*args)
            _set(model, target, mod)
        else:
            if kind not in ("F32", "F16", "BF16"):
                raise ValueError(f"unsupported aux type {kind} for {name}")
            if name in folded or name in inverse:
                raise ValueError(f"transformed float matrix {name} not supported")
            a = np.asarray(t.data)
            if kind == "BF16":  # the reader hands back raw bytes
                a = (a.view(np.uint16).astype(np.uint32) << 16).view(np.float32)
            a = a.astype(np.float32).reshape([int(n) for n in t.shape[::-1]])
            a = reorder(a, stem)
            if stem == "ssm_a":
                assert (a < 0).all()
                a = np.log(-a)
            if stem == "ssm_conv1d.weight":
                a = a[:, None, :]
            if target.endswith(_ZERO_CENTERED):
                norm_means.append(float(a.mean()))
                a = a - 1.0
            pdtype = torch.float32 if stem in ("ssm_a", "ssm_dt.bias") else dtype
            _set(model, target, torch.from_numpy(np.ascontiguousarray(a)).to(dev, pdtype))
        assigned.add(target)
        if verbose and i % 100 == 0:
            print(f"  loaded {i}/{len(tensors)} tensors", flush=True)

    # Rotary buffers were created on meta; rebuild them on the right devices.
    from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5TextRotaryEmbedding
    model.model.rotary_emb = Qwen3_5TextRotaryEmbedding(config, device=dev_for("token_embd.weight"))

    leftover = [n for n, p in list(model.named_parameters()) + list(model.named_buffers()) if p.is_meta]
    if leftover:
        raise ValueError(f"unassigned parameters: {leftover[:10]} (+{len(leftover) - 10})")
    if verbose:
        print(f"  zero-centered norm weights: mean of stored = {np.mean(norm_means):.3f} "
              "(expect ~1 if +1 folded; we subtract 1)")
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return model, config, fields
