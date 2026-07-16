#!/bin/bash
# Train MEMOIR model using uv
# Usage: bash scripts/train_memoir.sh [--config CONFIG] [--dataset DATASET]

set -e

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"

# Sync dependencies if not already synced (with llm extras)
if [ ! -f ".venv/pyvenv.cfg" ]; then
    echo "Creating virtual environment with uv..."
    uv sync --extra llm
fi

# Parse arguments manually to handle --config and --dataset properly
CONFIG="configs/memoir.yaml"
DATASET=""

while [[ $# -gt 0 ]]; do
    case $1 in
        --config)
            CONFIG="$2"
            shift 2
            ;;
        --dataset)
            DATASET="$2"
            shift 2
            ;;
        *)
            shift
            ;;
    esac
done

# Check device and warn about MPS performance
DEVICE=$(PYTHONPATH=. uv run python -c "import torch; print(torch.device('mps' if torch.backends.mps.is_available() else 'cuda' if torch.cuda.is_available() else 'cpu'))")
if [[ "$DEVICE" == "mps" ]]; then
    echo "[INFO] Running on MPS (Apple Silicon) - this is slower than CUDA"
    echo "[INFO] Consider using a GPU machine for faster training"
fi

echo "Training MEMOIR with config: $CONFIG"
echo "Dataset: ${DATASET:-default}"
PYTHONPATH=. uv run python train.py \
    --config "$CONFIG" \
    ${DATASET:+--dataset "$DATASET"}
