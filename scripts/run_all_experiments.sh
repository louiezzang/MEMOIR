#!/bin/bash
# Run all MEMOIR experiments: baselines + MEMOIR + ablations + sensitivity
# Usage: bash scripts/run_all_experiments.sh [--skip-done] [--ablation] [--sensitivity]
#
# Results are saved to logs/<model>_amazon/metrics.json. Ablation/sensitivity
# MEMOIR runs have their checkpoint dir deleted once each run completes (only
# metrics.json is needed for comparison) — each checkpoint is multi-GB since
# freeze_llm=True still saves the frozen LLM in every state_dict, and this can
# fill disk over a long sequential run otherwise. The main MEMOIR run's
# checkpoint is kept.

set -e

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"

SKIP_DONE=false
RUN_ABLATION=false
RUN_SENSITIVITY=false

for arg in "$@"; do
    case $arg in
        --skip-done)    SKIP_DONE=true ;;
        --ablation)     RUN_ABLATION=true ;;
        --sensitivity)  RUN_SENSITIVITY=true ;;
    esac
done

export PYTORCH_ENABLE_MPS_FALLBACK=1

log() { echo "[$(date '+%H:%M:%S')] $*"; }

has_results() {
    local log_dir="$1"
    [ -f "$log_dir/metrics.json" ] && python3 -c "
import json, sys
data = json.load(open('$log_dir/metrics.json'))
evals = [x for x in data if x.get('type') == 'eval']
sys.exit(0 if evals else 1)
" 2>/dev/null
}

run_baseline() {
    local model="$1"
    local log_dir="logs/${model}_amazon"
    if $SKIP_DONE && has_results "$log_dir"; then
        log "SKIP $model (results found in $log_dir)"
        return
    fi
    log "START baseline: $model"
    PYTHONPATH=. uv run python baselines/train_baseline.py \
        --config configs/default.yaml \
        --model "$model" \
        --dataset amazon
    log "DONE baseline: $model"
}

run_memoir() {
    local variant="$1"        # full | no_evo_cl | no_dir_loss | no_temporal | random_items | ...
    local ablation_flags="${2:-}"   # dedicated boolean flags, e.g. "--no-evo-cl"
    local overrides="${3:-}"        # KEY=VALUE pairs (no --override prefix), e.g. "model.temperature=0.03"
    local cleanup_ckpt="${4:-false}" # true: delete the checkpoint dir once done — only metrics.json
                                      # is needed for ablation/sweep comparison, and each checkpoint is
                                      # multi-GB (freeze_llm=True still saves the frozen LLM in every
                                      # state_dict), which can fill disk across many sequential runs.
    local log_dir="logs/memoir_amazon${variant:+_$variant}"
    local ckpt_dir="checkpoints${variant:+_$variant}"
    if $SKIP_DONE && has_results "$log_dir"; then
        log "SKIP memoir_$variant (results found in $log_dir)"
        return
    fi
    log "START MEMOIR variant: ${variant:-full}"
    PYTHONPATH=. uv run python train.py \
        --config configs/memoir.yaml \
        --dataset amazon \
        ${variant:+--log-suffix "_$variant"} \
        $ablation_flags \
        --override logging.save_every=1000 $overrides
    log "DONE MEMOIR variant: ${variant:-full}"
    if $cleanup_ckpt && [ -d "$ckpt_dir" ]; then
        log "Cleaning up $ckpt_dir ($(du -sh "$ckpt_dir" 2>/dev/null | cut -f1)) — metrics already in $log_dir/metrics.json"
        rm -rf "$ckpt_dir"
    fi
}

# ─── Phase 1: Baselines ───────────────────────────────────────────────────────
log "=== Phase 1: Baselines ==="
run_baseline gru4rec
run_baseline sasrec
run_baseline cl4srec
run_baseline duorec
run_baseline sracl

# ─── Phase 2: MEMOIR main ─────────────────────────────────────────────────────
log "=== Phase 2: MEMOIR (main) ==="
run_memoir ""   # keep this checkpoint (cleanup_ckpt=false) — it's the deployable/reference model

# ─── Phase 3: Ablations ───────────────────────────────────────────────────────
if $RUN_ABLATION; then
    log "=== Phase 3: Ablation Study ==="
    run_memoir "no_evo_cl"     "--no-evo-cl"   ""  true
    run_memoir "no_dir_loss"   "--no-dir-loss" ""  true
    run_memoir "no_temporal"   "--no-temporal" ""  true
    run_memoir "random_items"  ""  "model.item_encoder.type=random"  true
fi

# ─── Phase 4: Hyperparameter sensitivity (τ and W only) ──────────────────────
if $RUN_SENSITIVITY; then
    log "=== Phase 4: Hyperparameter Sensitivity ==="
    for tau in 0.03 0.2; do
        run_memoir "tau${tau}" "" "model.temperature=${tau}" true
    done
    for W in 3 12; do
        run_memoir "W${W}" "" "model.num_memory_windows=${W}" true
    done
fi

log "=== All experiments complete ==="
log "Collect results with: python scripts/collect_results.py"
