#!/usr/bin/env bash
# Export the latest fitting checkpoint on the GPU box, copy it here, hot-reload the viewer.
set -euo pipefail
cd "$(dirname "$0")/.."
REMOTE=${REMOTE:-m@192.168.0.202}
ssh "$REMOTE" 'cd ~/Code/Lens && .venv/bin/python scripts/export_lens.py lenses/bonsai27b.ckpt lenses/bonsai27b-j.safetensors'
rsync -a --progress "$REMOTE":Code/Lens/lenses/bonsai27b-j.safetensors lenses/ | tail -1
curl -s -X POST localhost:8765/api/reload_lens && echo
