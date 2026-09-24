"""Memory-lean Jacobian-lens fitting for large models.

Same estimator as ``jlens.fitting.jacobian_for_prompt`` (one-hot cotangent at
output dim i on every valid target position, backprop, mean the resulting
gradient over valid source positions -> row i of J_l), but instead of
replicating the prompt ``dim_batch`` times along the batch axis we run a single
forward and vmap the backward over a batch of cotangents
(``torch.autograd.grad(..., is_grads_batched=True)``). Activation memory is
then that of one sequence regardless of how many rows we compute per pass.

Accumulation is a running *sum* on the CPU in fp32; divide by ``n_done``.
"""

from __future__ import annotations

import logging
import os
import time

import torch

from jlens.fitting import SKIP_FIRST_N_POSITIONS, valid_position_mask
from jlens.hooks import ActivationRecorder

log = logging.getLogger(__name__)


def jacobian_rows_for_ids(
    model,
    input_ids: torch.Tensor,
    source_layers: list[int],
    target_layer: int,
    accum: dict[int, torch.Tensor],
    *,
    dim_batch: int = 32,
    batched: bool = True,
    skip_first: int = SKIP_FIRST_N_POSITIONS,
    max_dims: int | None = None,
) -> int:
    """Add this prompt's J_l into ``accum[l]`` (CPU fp32 [d, d]) for each source layer."""
    d_model = model.d_model
    seq_len = input_ids.shape[1]
    mask = valid_position_mask(seq_len, skip_first=skip_first)
    n_valid = int(mask.sum())

    with ActivationRecorder(
        model.layers, at=[*source_layers, target_layer], start_graph_at=min(source_layers)
    ) as rec, torch.enable_grad():
        model.forward(input_ids)
        target = rec.activations[target_layer]  # [1, T, d]
        sources = [rec.activations[l] for l in source_layers]
        dev = target.device
        valid = mask.nonzero(as_tuple=True)[0].to(dev)
        n_dims = d_model if max_dims is None else max_dims
        n_passes = (n_dims + dim_batch - 1) // dim_batch
        for p, start in enumerate(range(0, n_dims, dim_batch)):
            n = min(dim_batch, n_dims - start)
            b = torch.arange(n, device=dev)
            if batched:
                cot = torch.zeros((n, *target.shape), dtype=target.dtype, device=dev)
                cot[b[:, None], 0, valid[None, :], start + b[:, None]] = 1.0
                grads = torch.autograd.grad(
                    target, sources, grad_outputs=cot,
                    retain_graph=p < n_passes - 1, is_grads_batched=True,
                )  # each [n, 1, T, d]
                rows = [g[:, 0, valid.to(g.device), :].float().mean(1) for g in grads]
            else:
                rows_l = [[] for _ in source_layers]
                for i in range(n):
                    cot = torch.zeros_like(target)
                    cot[0, valid, start + i] = 1.0
                    gs = torch.autograd.grad(target, sources, grad_outputs=cot,
                                             retain_graph=True)
                    for k, g in enumerate(gs):
                        rows_l[k].append(g[0, valid.to(g.device)].float().mean(0))
                rows = [torch.stack(r) for r in rows_l]
            for l, r in zip(source_layers, rows):
                accum[l][start : start + n] += r.cpu()
            del rows
    return n_valid


def fit_stream(
    model,
    id_batches,
    *,
    source_layers: list[int],
    target_layer: int,
    checkpoint_path: str,
    dim_batch: int = 32,
    checkpoint_every: int = 5,
    max_prompts: int | None = None,
):
    """Fit over an iterable of ``[1, T]`` id tensors with resumable checkpoints."""
    d = model.d_model
    if os.path.exists(checkpoint_path):
        state = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
        assert state["source_layers"] == source_layers and state["target_layer"] == target_layer
        accum, n_done = state["jacobian_sum"], state["n_done"]
        log.info("resumed: %d prompts done", n_done)
    else:
        accum = {l: torch.zeros(d, d) for l in source_layers}
        n_done = 0

    def save():
        tmp = f"{checkpoint_path}.tmp"
        torch.save({"jacobian_sum": accum, "n_done": n_done,
                    "source_layers": source_layers, "target_layer": target_layer}, tmp)
        os.replace(tmp, checkpoint_path)

    for idx, ids in enumerate(id_batches):
        if idx < n_done:
            continue
        if max_prompts is not None and n_done >= max_prompts:
            break
        t0 = time.time()
        jacobian_rows_for_ids(model, ids, source_layers, target_layer, accum, dim_batch=dim_batch)
        n_done += 1
        mid = source_layers[len(source_layers) // 2]
        log.info("prompt %d done in %.0fs  ||J_mid||/sqrt(d)=%.3f", n_done, time.time() - t0,
                 (accum[mid] / n_done).norm().item() / d**0.5)
        if n_done % checkpoint_every == 0:
            save()
    save()
    return accum, n_done
