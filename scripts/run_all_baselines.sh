#!/bin/bash
# Run sequential baselines on a given dataset
# Usage: bash scripts/run_all_baselines.sh [dataset_name] [model_name]
#   If model_name is omitted, runs all baselines

set -e

# Get the project root directory (where this script lives's parent)
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

# Activate uv virtual environment (create if needed)
if [ ! -d "$PROJECT_ROOT/.venv" ]; then
    echo "Creating virtual environment with uv..."
    cd "$PROJECT_ROOT" && uv sync
fi
source "$PROJECT_ROOT/.venv/bin/activate"

DATASET="${1:-amazon}"
MODEL="${2:-all}"
CONFIG="$PROJECT_ROOT/configs/default.yaml"

mkdir -p "$PROJECT_ROOT/checkpoints" "$PROJECT_ROOT/logs"

echo "=== Running baseline(s) on ${DATASET} ==="

# Add project root to Python path so imports work
export PYTHONPATH="$PROJECT_ROOT:$PYTHONPATH"

if [ "$MODEL" != "all" ]; then
    echo ""
    echo ">>> Training ${MODEL}..."
    python baselines/train_baseline.py --config "$CONFIG" --model "$MODEL" --dataset "$DATASET" \
        2>&1 | tee "$PROJECT_ROOT/logs/${MODEL}_${DATASET}.log"
else
    for MODEL in gru4rec sasrec bert4rec cl4srec duorec cllmrec; do
        echo ""
        echo ">>> Training ${MODEL}..."
        python baselines/train_baseline.py --config "$CONFIG" --model "$MODEL" --dataset "$DATASET" \
            2>&1 | tee "$PROJECT_ROOT/logs/${MODEL}_${DATASET}.log"
    done
fi

echo ""
echo "=== Baseline(s) complete ==="
echo "Check logs/ for detailed output."
