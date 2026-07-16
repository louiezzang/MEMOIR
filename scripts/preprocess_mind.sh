#!/bin/bash
# Preprocess MIND dataset for MEMOIR (creates windows.parquet)
# Usage: bash scripts/preprocess_mind.sh [--size small|large]

set -e

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"

PYTHONPATH=. uv run python scripts/preprocess_mind.py "$@"
