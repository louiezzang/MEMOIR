# MEMOIR: Temporal Behavioral Memory for Recommendation Across the Preference-Drift Spectrum

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

> 📄 The paper is expected to be released here in mid-August 2026.

MEMOIR models user preferences as **evolving trajectories** across temporal windows, combining LLM-based behavioral memory encoding with contrastive learning for preference evolution and trajectory extrapolation.

---

## Table of Contents

- [Overview](#overview)
- [Quick Start](#quick-start)
- [Installation](#installation)
- [Dataset Preparation](#dataset-preparation)
- [Training](#training)
- [Configuration Reference](#configuration-reference)
- [Performance Guide](#performance-guide)
- [Cloud Training](#cloud-training)
- [Baseline Models](#baseline-models)
- [Preference Drift Analysis](#preference-drift-analysis)
- [Collecting Results](#collecting-results)
- [Project Structure](#project-structure)
- [Troubleshooting](#troubleshooting)
- [Citation](#citation)

---

## Overview

MEMOIR combines four components into a unified sequential recommendation framework:

| Component | Description |
|-----------|-------------|
| **LLM Memory Encoder** | TinyLlama encodes user behavioral windows into dense memory vectors |
| **Evolution Contrastive Loss** | Contrasts past vs. future preference trajectories |
| **Behavioral Consistency Loss** | Enforces smooth preference evolution across windows |
| **Trajectory Extrapolation** | Predicts next-step preference embeddings from historical trajectory |

### Architecture

```
User History
     │
     ▼
Temporal Segmentation (monthly / weekly / quarterly windows)
     │
     ▼
LLM Memory Encoder (TinyLlama + LoRA  or  frozen + cached)
     │
     ▼
Trajectory Predictor (GRU) ──► Evolution-Aware Aggregation
     │
     ├── Evolution Contrastive Loss
     ├── Consistency Loss
     └── Extrapolation Loss
          │
          ▼
     Recommendation Score
```

---

## Quick Start

```bash
# 1. Clone and install
git clone https://github.com/louiezzang/MEMOIR.git
cd MEMOIR
uv sync --extra llm

# 2. Download and preprocess Amazon data (10k users — fast for first experiments)
bash scripts/download_data.sh amazon
bash scripts/preprocess_amazon.sh 10000

# 3. Train MEMOIR
bash scripts/train_memoir.sh configs/memoir.yaml --dataset amazon
```

> **First run note:** With `freeze_llm: true` (default), MEMOIR pre-computes TinyLlama embeddings for all window texts and caches them to disk. This takes ~10–80 min depending on dataset size and hardware, but only happens **once**. Subsequent runs load the cache in seconds.

---

## Installation

### Requirements

- Python 3.11+
- macOS (MPS) or Linux (CUDA) — CPU fallback available

```bash
# Core dependencies
uv sync

# LLM features (required for MEMOIR)
uv sync --extra llm

# Optional: experiment tracking + plots
uv sync --extra experiment

# Optional: FAISS for fast retrieval
uv sync --extra search

# Optional: development tools
uv sync --extra dev
```

### Dependencies

| Group | Packages |
|-------|----------|
| Core | PyTorch 2.1+, NumPy, Pandas, PyArrow, scikit-learn |
| LLM | Transformers, PEFT, SentenceTransformers |
| Experiment | wandb, matplotlib |

---

## Dataset Preparation

### Amazon Reviews

```bash
# Download raw JSONL files (Electronics + Clothing_Shoes_and_Jewelry by default)
bash scripts/download_data.sh amazon

# Or download all 32 Amazon Reviews 2023 categories
bash scripts/download_data.sh amazon full
```

> **Note:** `scripts/preprocess_amazon.py` currently hardcodes `categories=['Electronics',
> 'Clothing_Shoes_and_Jewelry']`, so preprocessing only reads those two regardless of which
> raw files you downloaded above. Downloading with `full` does not by itself change what
> gets preprocessed — update that `categories` list too if you want the additional
> categories included.

```bash
# Preprocess — choose max_users based on your goal
bash scripts/preprocess_amazon.sh 10000    # ~10 min, good for ablations (~99k users max)
bash scripts/preprocess_amazon.sh 50000    # ~1 hour, medium scale
bash scripts/preprocess_amazon.sh          # up to 100k users (default cap), several hours

# --max-users is equivalent to the positional form above
bash scripts/preprocess_amazon.sh --max-users 10000
bash scripts/preprocess_amazon.sh --max-users 10000 --force-regenerate
```

The preprocessor creates temporal behavior windows and caches results intelligently:
- Re-running with the same parameters loads from cache instantly
- Changing parameters (e.g. `num_windows`) invalidates and rebuilds the cache
- Use `--force-regenerate` to rebuild everything from scratch

```bash
bash scripts/preprocess_amazon.sh --force-regenerate
```

Output structure:
```
data/processed/amazon/
├── train_samples.parquet
├── val_samples.parquet
├── test_samples.parquet
├── windows.parquet             # behavioral window texts per user
├── _metadata.json              # parameter hash for cache validation
└── pooled_cache_*.pt            # TinyLlama pooled hidden-state cache, pre-projection (built at training time)
```

### MovieLens

```bash
bash scripts/download_data.sh movielens
PYTHONPATH=. uv run python baselines/preprocess_data.py \
    --dataset-type movielens \
    --data-dir ./data/raw/movielens \
    --processed-dir ./data/processed/movielens
```

### MIND (News)

```bash
bash scripts/download_data.sh mind
bash scripts/preprocess_mind.sh
```

---

## Training

### Recommended workflow for paper experiments

**Step 1 — Validate on a small subset:**
```bash
bash scripts/preprocess_amazon.sh 10000
bash scripts/train_memoir.sh configs/memoir.yaml --dataset amazon
# ~10 min preprocessing + ~5 min/epoch on T4, ~35 min/epoch on MPS
```

**Step 2 — Full-scale training:**
```bash
bash scripts/preprocess_amazon.sh        # full ~99k users
bash scripts/train_memoir.sh configs/memoir.yaml --dataset amazon
```

### Training baselines

```bash
# Train a single baseline
bash scripts/train.sh sasrec --config configs/default.yaml --dataset amazon

# Run all baselines for comparison table
bash scripts/run_all_baselines.sh amazon
```

### Direct Python invocation

```bash
PYTHONPATH=. uv run python train.py \
    --config configs/memoir.yaml \
    --dataset amazon
```

---

## Configuration Reference

`configs/default.yaml` and `configs/memoir.yaml` share the same structure but are **not
interchangeable**: `train_baseline.py` (all seven baselines) reads `configs/default.yaml`,
while `train.py` (MEMOIR) defaults to `configs/memoir.yaml`. They intentionally diverge on
three regularization values validated specifically for MEMOIR's full-catalog training
objective — do not copy these into `configs/default.yaml`, or you will silently change every
baseline's training regime too:

| Key | `configs/default.yaml` (baselines) | `configs/memoir.yaml` (MEMOIR) |
|-----|-----|-----|
| `model.dropout` | 0.1 | 0.2 |
| `training.weight_decay` | 0.01 | 0.05 |
| `training.label_smoothing` | 0.0 (unused by baselines) | 0.1 |

Everything else (data paths, eval protocol, `embedding_dim`, `num_memory_windows`, etc.) is
kept identical between the two files. If you regenerate a GPU-tuned variant (e.g. for
Vast.ai — see [Cloud Training](#cloud-training)), generate one from *each* base file rather
than sharing a single `vastai.yaml` across both training scripts.

### Key settings in `configs/memoir.yaml`

```yaml
model:
  llm_name: TinyLlama/TinyLlama-1.1B-Chat-v1.0
  llm_load_in_4bit: false  # true = 4-bit quantization; requires NVIDIA GPU + bitsandbytes
  freeze_llm: true         # true = freeze LLM, pre-cache embeddings (recommended)
                           # false = fine-tune LLM with LoRA (slower, may improve quality)
  embedding_dim: 256
  num_memory_windows: 6
  temperature: 0.07        # contrastive temperature τ
  dropout: 0.2              # MEMOIR-specific; configs/default.yaml uses 0.1 for baselines

training:
  epochs: 100                      # max epochs; early stopping usually triggers earlier
  early_stopping_patience: 15      # stop if val NDCG@10 doesn't improve for N epochs
  batch_size: 16                   # MPS: 8–16 | T4: 64 | A100: 128
  lr: 0.0001
  llm_lr: 0.00001                  # only used when freeze_llm: false
  gradient_accumulation_steps: 8   # effective batch = batch_size × steps
  fp16: true
  weight_decay: 0.05        # MEMOIR-specific; configs/default.yaml uses 0.01 for baselines
  label_smoothing: 0.1      # MEMOIR-specific; validated against overfitting on the
                             # ~245k-way full-catalog softmax, unused by baselines
  lambda_evo: 0.5          # λ1: evolution contrastive loss weight
  lambda_consistency: 0.3  # λ2: behavioral consistency loss weight
  lambda_extrap: 0.3       # λ3: trajectory extrapolation loss weight

eval:
  eval_seed: 42            # fixed seed for reproducible negative sampling
```

### Config files at a glance

| Config | Used by | Dataset | Max Epochs | Batch | Use Case |
|--------|---------|---------|-----------|-------|----------|
| `configs/default.yaml` | `train_baseline.py` | amazon | 100 (early stop) | 16 | Baseline training |
| `configs/memoir.yaml` | `train.py` | amazon | 100 (early stop) | 16 | MEMOIR training |

### `freeze_llm` explained

MEMOIR uses TinyLlama to encode behavioral window texts (e.g. *"User highly rated AirPods Pro; viewed Phone Case"*) into pooled hidden states. Since window texts are fixed throughout training, a frozen LLM always produces the same pooled output — so that part can be pre-computed once and looked up instantly during training. The trainable projection MLP that maps this pooled state to the final embedding is still applied fresh on every forward pass (with gradients), so caching does not stop it from training.

| `freeze_llm` | LLM during training | Speed | Notes |
|---|---|---|---|
| `true` | Dict lookup (μs/step) + live projection | Fast | Recommended for all experiments |
| `false` | Full LoRA forward+backward | Slow | Use as ablation to measure LoRA benefit |

The pooled hidden-state cache is saved at:
```
data/processed/{dataset}/pooled_cache_{llm_name}.pt
```
It is reused across all runs with the same LLM. Delete it to force recomputation (e.g. after changing `max_length`).

---

## Performance Guide

### Training speed by device (full Amazon dataset, ~360k train samples)

| Device | `freeze_llm: true` | `freeze_llm: false` | Notes |
|--------|-------------------|---------------------|-------|
| Apple Silicon MPS | ~35 min/epoch | ~6 h/epoch | fp16 supported |
| Colab / Kaggle T4 | ~7 min/epoch | ~1 h/epoch | use `llm_load_in_4bit: true` |
| Colab A100 | ~2 min/epoch | ~15 min/epoch | use `llm_load_in_4bit: true` |
| CPU | ~4–6 h/epoch | not practical | fallback only |

**Estimated total (early stopping, freeze_llm: true, typically 50–80 epochs):**

| Device | Cache build (once) | Training | Total |
|--------|-------------------|----------|-------|
| MPS | ~80 min | ~20–30 h | ~22–32 h |
| T4 | ~35 min | ~4–8 h | ~4.5–8.5 h |
| A100 | ~10 min | ~1–2 h | ~1–2 h |

### Tips for faster iteration

```bash
# Smaller dataset — ~10x fewer samples
bash scripts/preprocess_amazon.sh 10000
```

```yaml
# Fewer epochs for early experiments
training:
  epochs: 10

# Fewer windows per user
model:
  num_memory_windows: 3     # default: 6

# On MPS: smaller batch to avoid OOM
training:
  batch_size: 8
  gradient_accumulation_steps: 16   # keeps effective batch at 128
```

---

## Cloud Training

Ready-to-use notebooks are included (split into two sessions to stay within Kaggle's 12-hour limit):

| Notebook | Contents | Est. time | Platform |
|----------|----------|-----------|----------|
| `memoir_kaggle_session1.ipynb` | 7 baselines + MEMOIR main | ~5 h | Kaggle T4 x2 |
| `memoir_kaggle_session2.ipynb` | 5 ablations + sensitivity | ~8 h | Kaggle T4 x2 |

### Kaggle

1. Notebook Settings → Accelerator → **GPU T4 x2**, Internet → **On**
2. Create a Kaggle Dataset from your `data/processed/amazon/` folder:
   - Upload the **`amazon/` folder itself** (not just its contents) — the code expects the path `<processed_dir>/amazon/`
   - Suggested dataset name: **`memoir-amazon-processed`** → mounts at `/kaggle/input/memoir-amazon-processed/amazon/`
   - Include `pooled_cache_TinyLlama_*.pt` if under the 20 GB limit — saves ~35 min of pre-computation on first run
3. Open `memoir_kaggle_session1.ipynb`, attach your dataset, set `GITHUB_REPO`, run all cells
4. Click **Save Version** → training runs in the background with no browser required
5. For Session 2: **Add Data** → add Session 1 output as `memoir-session1-output`, then run `memoir_kaggle_session2.ipynb`

### Cloud config (auto-generated by notebooks)

```yaml
model:
  llm_load_in_4bit: true   # 4-bit quantization (CUDA only, not supported on MPS/CPU)
  freeze_llm: true
training:
  batch_size: 64
  gradient_accumulation_steps: 2
  fp16: true
```

**Resuming after disconnect:** the notebook includes a resume cell that finds the latest checkpoint and continues from the correct epoch automatically.

### Vast.ai (recommended for single uninterrupted run)

Vast.ai is the cheapest option for a full experiment run (~$3–7 total) with no session timeout.

**Recommended GPU:** RTX 4090 (24 GB VRAM, ~5h total) or RTX 3090 (24 GB VRAM, ~13h total)

**1. Rent an instance**

- Go to [vast.ai](https://vast.ai) → Search → filter GPU RAM ≥ 16 GB
- Select template: **PyTorch (Vast)**
- Set Container Size to **50 GB**
- Click Rent

**2. Add your SSH key** (Account → SSH Keys → Add SSH Key):

```bash
cat ~/.ssh/id_ed25519.pub   # copy this output into Vast.ai
```

**3. Transfer preprocessed data** from your local machine:

```bash
ssh -p <PORT> root@<IP> "mkdir -p /workspace/MEMOIR/data/processed/amazon"

rsync -avz --progress \
  data/processed/amazon/ \
  -e "ssh -p <PORT>" root@<IP>:/workspace/MEMOIR/data/processed/amazon/
```

Replace `<PORT>` and `<IP>` with the values from the key icon (🔑) in your instance dashboard.

**4. Set up repo and config** (in the Vast.ai terminal via JupyterLab → Terminal):

```bash
git clone https://github.com/louiezzang/MEMOIR.git /workspace/MEMOIR
cd /workspace/MEMOIR

pip install transformers>=4.40.0 peft>=0.10.0 sentence-transformers>=3.0.0 \
    datasets>=2.19.0 bitsandbytes pandas pyarrow scikit-learn scipy pyyaml tqdm

mkdir -p checkpoints logs

python -c "
import yaml
# configs/default.yaml (baselines) and configs/memoir.yaml (MEMOIR) hold different
# regularization values that must not cross-apply — see 'Config files at a glance'
# below. Generate a GPU-tuned variant of each rather than sharing one vastai.yaml.
for src, dst in [('configs/default.yaml', 'configs/vastai.yaml'),
                  ('configs/memoir.yaml', 'configs/vastai_memoir.yaml')]:
    with open(src) as f:
        c = yaml.safe_load(f)
    c['data']['processed_dir'] = '/workspace/MEMOIR/data/processed'
    c['logging']['save_dir'] = '/workspace/MEMOIR/checkpoints'
    c['logging']['log_dir'] = '/workspace/MEMOIR/logs'
    c['training']['batch_size'] = 128
    c['training']['gradient_accumulation_steps'] = 1
    c['training']['fp16'] = True
    with open(dst, 'w') as f:
        yaml.dump(c, f)
"
```

**5. Run all experiments in background** (survives browser close):

```bash
nohup bash -c '
  export PYTHONPATH=/workspace/MEMOIR
  cd /workspace/MEMOIR

  # Session 1: baselines (configs/vastai.yaml) + MEMOIR main (configs/vastai_memoir.yaml)
  for model in gru4rec sasrec cl4srec duorec sracl sasrec_text unisrec; do
    python baselines/train_baseline.py --config configs/vastai.yaml --model $model --dataset amazon
  done
  python train.py --config configs/vastai_memoir.yaml --dataset amazon

  # Session 2: ablations
  python train.py --config configs/vastai_memoir.yaml --dataset amazon --no-evo-cl --log-suffix _no_evo_cl
  python train.py --config configs/vastai_memoir.yaml --dataset amazon --no-dir-loss --log-suffix _no_dir_loss
  python train.py --config configs/vastai_memoir.yaml --dataset amazon --no-temporal --log-suffix _no_temporal
  python train.py --config configs/vastai_memoir.yaml --dataset amazon --override model.item_encoder.type=random --log-suffix _random_items

  # Session 2: sensitivity
  python train.py --config configs/vastai_memoir.yaml --dataset amazon --override model.temperature=0.03 --log-suffix _tau0.03
  python train.py --config configs/vastai_memoir.yaml --dataset amazon --override model.temperature=0.2 --log-suffix _tau0.2
  python train.py --config configs/vastai_memoir.yaml --dataset amazon --override model.num_memory_windows=3 --log-suffix _W3
  python train.py --config configs/vastai_memoir.yaml --dataset amazon --override model.num_memory_windows=12 --log-suffix _W12
' > /root/train.log 2>&1 &

tail -f /root/train.log
```

**6. Download results** when done:

```bash
rsync -avz -e "ssh -p <PORT>" root@<IP>:/workspace/MEMOIR/logs/ ./logs/
rsync -avz -e "ssh -p <PORT>" root@<IP>:/workspace/MEMOIR/checkpoints/ ./checkpoints/
```

Then stop the instance from the Vast.ai dashboard to stop billing.

---

## Baseline Models

| Model | Type | Requires `--extra llm` |
|-------|------|:----------------------:|
| `gru4rec` | GRU-based sequential | |
| `sasrec` | Self-attentive sequential | |
| `bert4rec` | BERT-based sequential | |
| `cl4srec` | Contrastive learning | |
| `duorec` | Dual-view contrastive | |
| `cllmrec` | LLM-augmented | |
| `sracl` | Semantic retrieval contrastive | ✓ |
| `sasrec_text` | SASRec with MiniLM text embeddings | ✓ |
| `unisrec` | Whitening + MoE adaptor + SASRec (KDD 2022) | ✓ |
| `tallrec` | LoRA fine-tuning | |
| `llm_ranker` | Zero-shot LLM ranking | |

```bash
bash scripts/train.sh <model_name> --config configs/default.yaml --dataset amazon
bash scripts/run_all_baselines.sh amazon
```

---

## Preference Drift Analysis

MEMOIR is motivated by the observation that user preferences evolve over time. The `analyze_drift.py` script quantifies this drift per user, produces statistics for paper motivation, and can evaluate all models stratified by drift group.

### Step 1 — Compute drift scores

```bash
PYTHONPATH=. uv run python analyze_drift.py --dataset amazon --plot
```

Three metrics are computed per user:

| Metric | Description |
|--------|-------------|
| **Category JSD** | Jensen-Shannon divergence between adjacent windows' category distributions — how much the *types* of items change window-to-window |
| **Rating drift** | Variance in mean rating across windows — how much a user's satisfaction level fluctuates |
| **Novelty rate** | Fraction of items in each window not seen in prior windows — how exploratory the user is |

Users are classified into `high / medium / low` drift groups (top-25% / middle / bottom-25% by composite score).

**Amazon dataset results:**

| Metric | Mean | Median |
|--------|------|--------|
| Category JSD | 0.795 | 0.806 |
| Novelty rate | 0.922 | 1.000 |
| Composite drift score | 0.695 | 0.700 |

92% of items per window are items the user has never interacted with before (median novelty = 1.0), and even low-drift users score 0.60 — preference evolution is pervasive across all users, not just an edge case.

Output: `data/processed/amazon/user_drift_analysis.parquet`

### Step 2 — Evaluate models by drift group

After training MEMOIR and baselines, run stratified evaluation:

```bash
PYTHONPATH=. uv run python analyze_drift.py --dataset amazon --evaluate \
  --models memoir,gru4rec,sasrec,cl4srec,duorec,sracl,sasrec_text,unisrec
```

This loads each model's `best_model.pt` checkpoint, runs test-set evaluation separately for high / medium / low drift users, and prints a comparison table:

```
Model           Group     HR@5    NDCG@5    HR@10   NDCG@10     MRR
------------------------------------------------------------------------
memoir          high    0.XXXX    0.XXXX   0.XXXX    0.XXXX  0.XXXX
                medium  0.XXXX    0.XXXX   0.XXXX    0.XXXX  0.XXXX
                low     0.XXXX    0.XXXX   0.XXXX    0.XXXX  0.XXXX
------------------------------------------------------------------------
sasrec          high    ...
```

Results are saved to `data/processed/amazon/drift_eval_results.parquet` **incrementally, after each model finishes** — a crash partway through only costs you the model that was in progress, not the ones already evaluated. On the next run, previously-saved models are automatically loaded back in and included in the final table. Add `--skip-done` to explicitly skip re-evaluating any model already present in `drift_eval_results.parquet`:

```bash
PYTHONPATH=. uv run python analyze_drift.py --dataset amazon --evaluate \
  --models memoir,gru4rec,sasrec,cl4srec,duorec,sracl,sasrec_text,unisrec \
  --skip-done
```

### Use in my paper

- **Introduction / Motivation**: cite the dataset drift statistics (JSD = 0.795, novelty = 0.922) to show that preference evolution is real and prevalent
- **Experiments**: include the drift-stratified table — MEMOIR's advantage over baselines should be largest for high-drift users, directly validating the evolution-aware design
- **Dataset statistics table**: include drift group counts (high/medium/low: 25%/50%/25%) alongside standard dataset statistics

---

## Collecting Results

After training completes (locally or on Kaggle), use `scripts/collect_results.py` to summarize all experiment metrics and optionally fill the paper's LaTeX tables.

```bash
# Print summary table for all models, ablations, and sensitivity runs
python scripts/collect_results.py

# Use a custom log directory (e.g. downloaded output)
python scripts/collect_results.py --log-dir /path/to/logs

# Fill placeholder values (--) in paper/main.tex tables with actual results
python scripts/collect_results.py --update-paper

# Both options together
python scripts/collect_results.py --log-dir /path/to/logs --update-paper
```

The `--update-paper` mode:
- Scans `logs/` for `metrics.json` files from each experiment
- Replaces `--` placeholders in the main results, ablation, and sensitivity tables
- Applies `\textbf{}` to best values and `\underline{}` to second-best per column
- Skips any model that has no results yet

---

## Project Structure

```
memoir/
├── configs/
│   └── default.yaml              # Main config (MPS-optimized defaults)
│
├── data/
│   ├── amazon.py                 # Amazon Reviews dataset + preprocessing
│   ├── movielens.py              # MovieLens dataset
│   ├── mind.py                   # MIND news dataset
│   ├── collator.py               # Batch collation + contrastive pair sampling
│   └── temporal.py               # Temporal window segmentation
│
├── model/
│   ├── memoir.py                 # Full MEMOIR model
│   ├── memory_encoder.py         # LLM memory encoder (freeze / LoRA)
│   ├── item_encoder.py           # Item embedding (MiniLM)
│   ├── trajectory.py             # GRU trajectory predictor
│   ├── evolution_contrastive.py  # Evolution contrastive loss
│   ├── retriever.py              # ANN candidate retrieval
│   └── reasoning.py              # Candidate re-ranking layer
│
├── baselines/
│   ├── train_baseline.py         # Baseline training script
│   ├── preprocess_data.py        # Generic preprocessing
│   ├── filter_dataset.py         # Filter to top-K items
│   ├── sample_dataset.py         # Sample top-N users
│   └── *.py                      # Baseline model implementations
│
├── scripts/
│   ├── download_data.sh          # Download raw datasets
│   ├── preprocess_amazon.sh      # Amazon preprocessing
│   ├── train_memoir.sh           # MEMOIR training launcher
│   ├── train.sh                  # Baseline training launcher
│   ├── run_all_baselines.sh      # Run all baselines
│   └── collect_results.py        # Collect metrics & update paper tables
│
├── memoir_kaggle_session1.ipynb   # Kaggle Session 1: baselines + MEMOIR main
├── memoir_kaggle_session2.ipynb   # Kaggle Session 2: ablations + sensitivity
├── train.py                      # MEMOIR training entry point
├── eval.py                       # Evaluation script
├── analyze_drift.py              # Preference drift analysis per user
├── losses.py                     # Loss functions
├── logger.py                     # Experiment logging
└── utils.py                      # Device / autocast utilities
```

---

## Troubleshooting

**Out of memory on MPS (Apple Silicon)**

`batch_size: 32` is often still too large on MPS. Start with 8 and compensate with gradient accumulation:
```yaml
training:
  batch_size: 8
  gradient_accumulation_steps: 16   # effective batch = 128
```

**Embedding pre-caching takes too long**

This is a one-time cost per LLM. Speed it up by using a smaller dataset for development:
```bash
bash scripts/preprocess_amazon.sh 10000   # cache builds in ~10 min
```
The cache at `data/processed/amazon/pooled_cache_*.pt` is reused on every subsequent run.

**`windows.parquet` not found**

Run the MEMOIR-specific preprocessing — generic baseline preprocessing does not create window data:
```bash
bash scripts/preprocess_amazon.sh
```

**`llm_load_in_4bit: true` crashes**

4-bit quantization requires an NVIDIA GPU and `bitsandbytes`. On MPS or CPU, set:
```yaml
model:
  llm_load_in_4bit: false
```

**Pooled hidden-state cache is stale after changing the LLM**

Cache filenames include the model name (e.g. `pooled_cache_TinyLlama_TinyLlama-1.1B-Chat-v1.0.pt`). Switching LLMs automatically uses a separate cache. Delete old `.pt` files to free disk space.

**Training loss is NaN**

Try lowering the learning rate, or disable fp16 on CPU:
```yaml
training:
  lr: 0.00001
  fp16: false   # on CPU only
```

**Kaggle session disconnected mid-training**

Re-run the same notebook — each training cell checks for existing results and skips completed models automatically.

---

## Citation

If you use this code in your research, please cite:

```bibtex
@article{bae2026memoir,
  title={MEMOIR: Temporal Behavioral Memory for Recommendation Across the Preference-Drift Spectrum},
  author={Bae, Younggue},
  journal={arXiv preprint arXiv:xxxx.xxxxx},
  year={2026}
}
```

## License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.
