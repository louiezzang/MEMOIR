#!/bin/bash
# Train baseline models using uv
# Usage: bash scripts/train.sh [model_name] [--config CONFIG] [--dataset DATASET]

set -e

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"

# Sync dependencies if not already synced
if [ ! -f ".venv/pyvenv.cfg" ]; then
    echo "Creating virtual environment with uv..."
    uv sync
fi

MODEL="${1:-}"
shift || true

CONFIG="${CONFIG:-configs/default.yaml}"
DATASET="${DATASET:-}"

if [ -z "$MODEL" ]; then
    echo "Usage: bash scripts/train.sh [model_name] [--config CONFIG] [--dataset DATASET]"
    echo ""
    echo "Models: gru4rec, sasrec, bert4rec, cl4srec, duorec, cllmrec, sracl"
    exit 1
fi

# Check device and warn about MPS performance
DEVICE=$(PYTHONPATH=. uv run python -c "import torch; print(torch.device('mps' if torch.backends.mps.is_available() else 'cuda' if torch.cuda.is_available() else 'cpu'))")
if [[ "$DEVICE" == "mps" ]]; then
    echo "[INFO] Running on MPS (Apple Silicon) - this is slower than CUDA"
    echo "[INFO] Consider using a GPU machine for faster training"
    export PYTORCH_ENABLE_MPS_FALLBACK=1
fi

PYTHONPATH=. uv run python baselines/train_baseline.py \
    --config "$CONFIG" \
    --model "$MODEL" \
    ${DATASET:+--dataset "$DATASET"} \
    "$@"
