#!/bin/bash
# Preprocess Amazon dataset for MEMOIR (creates windows.parquet)
# Usage: bash scripts/preprocess_amazon.sh [max_users] [--force-regenerate]

set -e

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"

MAX_USERS=""
FORCE_REGENERATE=""

# Check if first arg is a number (positional max-users) or flag
if [[ "$1" =~ ^[0-9]+$ ]]; then
    MAX_USERS="$1"
    shift
fi

while [[ $# -gt 0 ]]; do
    case $1 in
        --max-users)
            MAX_USERS="$2"
            shift 2
            ;;
        --force-regenerate)
            FORCE_REGENERATE="--force-regenerate"
            shift
            ;;
        *)
            shift
            ;;
    esac
done

if [ -n "$MAX_USERS" ]; then
    PYTHONPATH=. uv run python scripts/preprocess_amazon.py --max-users "$MAX_USERS" $FORCE_REGENERATE
else
    PYTHONPATH=. uv run python scripts/preprocess_amazon.py $FORCE_REGENERATE
fi
