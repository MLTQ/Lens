# Lens — Jacobian-lens viewer for Ternary-Bonsai-2-27B

The Jacobian lens (Gurnee et al. 2026, "Verbalizable representations form a global
workspace") applied to `prism-ml/Ternary-Bonsai-2-27B` (ternary Qwen3.8-27B, hybrid
Gated-DeltaNet / full attention, 64 layers, d=5120).

    lens_l(h) = lm_head( norm( J_l h ) ),   J_l = E[ ∂h_final / ∂h_l ]

## Layout

| Path | What |
|---|---|
| `bonsai_lens/ternary.py` | PyTorch loader for the Prism PQ2_0/PTQ1_0 GGUF → HF `Qwen3_5ForCausalLM`; weights stay 2-bit packed on GPU (Triton dequant), Hadamard folding undone on activations so the residual stream is in the native basis; autograd w.r.t. activations works. |
| `bonsai_lens/fit.py` | Reference jlens estimator, but one forward + vmapped backward over batched cotangents (fits a 27B in 24 GB). Verified against per-row backprop. |
| `bonsai_lens/mlx_backend.py` | Apple-Silicon readout via the Prism MLX pack (official `load_vl_model` path). Residuals match the torch model (cos ≥ 0.999, top-5 identical). |
| `bonsai_lens/server.py`, `static/index.html` | Local viewer: layer × position grid, top-k per cell, J-lens / logit-lens toggle, pinned-token rank heatmaps, greedy "Continue N tokens" (raw completion, no chat template; generated positions marked), per-column 64×10 top-k map, J-Volume 3D view (`static/volume.js`, three.js: position × layer × rank voxels, highlight of the actual next token), hover-to-translate for non-English tokens (gloss table from `bonsai_lens/data/qwen_gloss.json.gz`, else Bonsai translates the token itself; cached). |
| `scripts/embed_vocab.py`, `static/space.js` | Canonical token space: every vocabulary token at its readout direction γ⊙W_eff[v] (the basis every J-lens layer decodes in) → PCA-128 → UMAP 3D/2D; Space tab with type/norm coloring, search, click-for-nearest-neighbours (`/api/space_nn`). "Trace run" draws the current run into the space: main line = the model's prediction per position (colour = position), branches = other lens readouts at that position (colour = mean layer, alpha = peak probability, one edge per position×token). Playback (▶) replays it token by token: input lights up → each layer's readouts pop in → rays converge on the output prediction, which stays lit and extends the main line → fade. A follow-along strip (input token over its prediction, in position colours) builds up as it plays; click a token to jump there. Edges animate in their direction of flow (readout → prediction; main line in sequence order). |
| Knockout (⊘ in a cell's readout list; `/api/knockout`) | Projects the positive component along a token's J-lens vector a_u = J_lᵀ(γ⊙w_u) out of the residual stream at chosen layers/positions, continues the forward pass, and compares next-token predictions and a fresh greedy continuation against the original and a random-direction control. The edited run can be loaded into every view. `/api/jspace` measures how much of each activation J-space captures. |
| `scripts/run_fit.py` | Resumable fit on FineWeb (128-token docs), checkpoint = running sum. |
| `scripts/export_lens.py`, `scripts/pull_lens.sh` | Checkpoint → fp16 safetensors → copied here → viewer hot-reloads. |
| `vendor-jacobian-lens/` | Anthropic's reference implementation (Apache-2.0). |

## Run

GPU box (`m@192.168.0.202`, RTX 4090), in `~/Code/Lens`:

    CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=1 \
      nohup .venv/bin/python scripts/run_fit.py --n-prompts 120 > runs/fit.log 2>&1 &

~11 min/prompt at 128 tokens (5120 backward rows each). Needs the Prism fork of
`gguf-py` (`vendor/prism-llama.cpp/gguf-py`) for quant types 142/143.

Mac (this directory):

    .venv/bin/python -m bonsai_lens.server      # http://127.0.0.1:8765
    ./scripts/pull_lens.sh                       # refresh J from the latest checkpoint
