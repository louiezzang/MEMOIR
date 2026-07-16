"""Collect best eval metrics from all experiment logs.

Usage:
    python scripts/collect_results.py                    # print summary table
    python scripts/collect_results.py --update-paper     # fill paper/main.tex tables
    python scripts/collect_results.py --log-dir /path    # use custom log directory
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path


MODELS = [
    ("gru4rec",     "GRU4Rec",       "gru4rec_amazon"),
    ("sasrec",      "SASRec",        "sasrec_amazon"),
    ("cl4srec",     "CL4SRec",       "cl4srec_amazon"),
    ("duorec",      "DuoRec",        "duorec_amazon"),
    ("sasrec_text", "SASRec-Text",   "sasrec_text_amazon"),
    ("unisrec",     "UniSRec",       "unisrec_amazon"),
    ("sracl",       "SRA-CL",        "sracl_amazon"),
    ("memoir",      "MEMOIR (Ours)", "memoir_amazon"),
]

ABLATIONS = [
    ("memoir_full",       "MEMOIR (Full)",       "memoir_amazon"),
    ("memoir_no_evo_cl",  "w/o Evolution CL",    "memoir_amazon_no_evo_cl"),
    ("memoir_no_dir_loss","w/o Direction Loss",   "memoir_amazon_no_dir_loss"),
    ("memoir_no_temporal","w/o Temporal Seg.",    "memoir_amazon_no_temporal"),
    ("memoir_random",     "w/ Random-Init Items", "memoir_amazon_random_items"),
]

SENSITIVITY = [
    ("tau0.03", "temperature=0.03", "memoir_amazon_tau0.03"),
    ("tau0.07", "temperature=0.07", "memoir_amazon"),
    ("tau0.2",  "temperature=0.2",  "memoir_amazon_tau0.2"),
    ("W3",      "num_windows=3",    "memoir_amazon_W3"),
    ("W6",      "num_windows=6",    "memoir_amazon"),
    ("W12",     "num_windows=12",   "memoir_amazon_W12"),
]


def load_best(log_dir: Path, metric: str = "ndcg@10") -> dict | None:
    p = log_dir / "metrics.json"
    if not p.exists():
        return None
    try:
        data = json.loads(p.read_text())
        evals = [x for x in data if x.get("type") == "eval"]
        if not evals:
            return None
        test_evals = [x for x in evals if x.get("epoch") == "test"]
        if test_evals:
            return test_evals[-1]
        return max(evals, key=lambda x: x.get(metric, 0))
    except Exception:
        return None


def fmt(v: float | None) -> str:
    return f"{v:.4f}" if v is not None else "  --  "


# ---------------------------------------------------------------------------
# Terminal output
# ---------------------------------------------------------------------------

def print_table(rows: list[tuple[str, str, str]], title: str, log_root: Path):
    print(f"\n{'='*70}")
    print(f"  {title}")
    print(f"{'='*70}")
    print(f"{'Model':<22} {'HR@10':>7} {'NDCG@10':>9} {'HR@20':>7} {'MRR':>7}")
    print("-" * 56)
    best_ndcg = max(
        (load_best(log_root / d) or {}).get("ndcg@10", 0) for _, _, d in rows
    )
    for _, label, log_sub in rows:
        best = load_best(log_root / log_sub)
        if best:
            ndcg = best.get("ndcg@10")
            marker = " *" if ndcg and abs(ndcg - best_ndcg) < 1e-8 else "  "
            print(f"{label:<22} {fmt(best.get('hr@10')):>7} {fmt(ndcg):>9} "
                  f"{fmt(best.get('hr@20')):>7} {fmt(best.get('mrr')):>7}{marker}")
        else:
            print(f"{label:<22} {'(not run)':>34}")


def print_all(log_root: Path):
    print_table(MODELS, "Main Results — Amazon Reviews", log_root)
    print_table(ABLATIONS, "Ablation Study", log_root)

    print(f"\n{'='*70}")
    print("  Hyperparameter Sensitivity (HR@10)")
    print(f"{'='*70}")
    prev_group = None
    for _, label, log_sub in SENSITIVITY:
        group = label.split("=")[0]
        if group != prev_group:
            print()
            prev_group = group
        best = load_best(log_root / log_sub)
        hr10 = fmt(best.get("hr@10") if best else None)
        default_mark = " (default)" if log_sub == "memoir_amazon" else ""
        print(f"  {label:<30} HR@10={hr10}{default_mark}")


# ---------------------------------------------------------------------------
# LaTeX paper update
# ---------------------------------------------------------------------------

# Map LaTeX row identifiers → log subdirectory
_MAIN_TABLE_MAP: dict[str, str] = {
    "GRU4Rec":     "gru4rec_amazon",
    "SASRec":      "sasrec_amazon",
    "CL4SRec":     "cl4srec_amazon",
    "DuoRec":      "duorec_amazon",
    "SASRec-Text": "sasrec_text_amazon",
    "UniSRec":     "unisrec_amazon",
    "SRA-CL":      "sracl_amazon",
    "MEMOIR":      "memoir_amazon",
}

_ABLATION_TABLE_MAP: dict[str, str] = {
    "MEMOIR (Full)":        "memoir_amazon",
    "w/o Evolution CL":     "memoir_amazon_no_evo_cl",
    "w/o Direction Loss":   "memoir_amazon_no_dir_loss",
    "w/o Temporal Seg.":    "memoir_amazon_no_temporal",
    "w/ Random-Init Items": "memoir_amazon_random_items",
}

_SENSITIVITY_TABLE_MAP: dict[str, str] = {
    "0.03": "memoir_amazon_tau0.03",
    "0.07": "memoir_amazon",
    "0.2":  "memoir_amazon_tau0.2",
    "3":    "memoir_amazon_W3",
    "6":    "memoir_amazon",
    "12":   "memoir_amazon_W12",
}


def _fmt_tex(v: float | None) -> str:
    return f"{v:.4f}" if v is not None else "--"


def _apply_best_underline(lines: list[str], col_indices: list[int]):
    """Apply \\textbf to best and \\underline to second-best per column.

    Operates on data rows (lines containing '&' and '\\\\').
    col_indices specifies which &-separated fields hold numeric values.
    """
    data_rows = []
    for i, line in enumerate(lines):
        if "&" in line and "\\\\" in line:
            parts = [p.strip() for p in line.rstrip("\\").rstrip().split("&")]
            vals = []
            for ci in col_indices:
                if ci < len(parts):
                    raw = re.sub(r"\\textbf\{([^}]*)\}", r"\1", parts[ci])
                    raw = re.sub(r"\\underline\{([^}]*)\}", r"\1", raw)
                    raw = raw.strip()
                    try:
                        vals.append(float(raw))
                    except ValueError:
                        vals.append(None)
                else:
                    vals.append(None)
            data_rows.append((i, parts, vals))

    if not data_rows:
        return

    for j, ci in enumerate(col_indices):
        col_vals = [(idx, rows[j]) for idx, _, rows in data_rows if rows[j] is not None]
        if len(col_vals) < 2:
            continue
        col_vals.sort(key=lambda x: x[1], reverse=True)
        best_idx, _ = col_vals[0]
        second_idx, _ = col_vals[1]

        for row_i, parts, _ in data_rows:
            if ci >= len(parts):
                continue
            raw = re.sub(r"\\textbf\{([^}]*)\}", r"\1", parts[ci])
            raw = re.sub(r"\\underline\{([^}]*)\}", r"\1", raw).strip()
            if row_i == best_idx:
                parts[ci] = f"\\textbf{{{raw}}}"
            elif row_i == second_idx:
                parts[ci] = f"\\underline{{{raw}}}"
            else:
                parts[ci] = raw

        for row_i, parts, _ in data_rows:
            lines[row_i] = " & ".join(parts) + " \\\\"


def _find_table_region(lines: list[str], label: str) -> tuple[int, int] | None:
    start = None
    for i, line in enumerate(lines):
        if label in line:
            start = i
        if start is not None and "\\bottomrule" in line:
            return (start, i)
    return None


def _update_main_table(lines: list[str], region: tuple[int, int],
                       log_root: Path) -> tuple[int, int]:
    start, end = region
    updated = missing = 0
    for i in range(start, end + 1):
        line = lines[i]
        if "&" not in line or "\\\\" not in line:
            continue
        if "--" not in line:
            continue
        for tex_name, log_sub in _MAIN_TABLE_MAP.items():
            if tex_name == "MEMOIR":
                if "\\textbf{MEMOIR" not in line:
                    continue
            elif tex_name == "SASRec-Text":
                if "SASRec-Text" not in line:
                    continue
            elif tex_name == "SASRec":
                if "SASRec" not in line or "SASRec-Text" in line:
                    continue
            else:
                if tex_name not in line:
                    continue
            best = load_best(log_root / log_sub)
            if not best:
                missing += 1
                break
            hr10 = _fmt_tex(best.get("hr@10"))
            ndcg10 = _fmt_tex(best.get("ndcg@10"))
            hr20 = _fmt_tex(best.get("hr@20"))
            mrr_val = _fmt_tex(best.get("mrr"))
            if tex_name == "MEMOIR":
                lines[i] = (f"\\textbf{{MEMOIR (Ours)}} & "
                            f"\\textbf{{{hr10}}} & \\textbf{{{ndcg10}}} & "
                            f"\\textbf{{{hr20}}} & \\textbf{{{mrr_val}}} \\\\")
            else:
                lines[i] = f"{tex_name:<14} & {hr10} & {ndcg10} & {hr20} & {mrr_val} \\\\"
            updated += 1
            break
    _apply_best_underline(lines[start:end + 1], [1, 2, 3, 4])
    return updated, missing


def _update_ablation_table(lines: list[str], region: tuple[int, int],
                           log_root: Path) -> tuple[int, int]:
    start, end = region
    updated = missing = 0
    for i in range(start, end + 1):
        line = lines[i]
        if "&" not in line or "\\\\" not in line:
            continue
        if "--" not in line:
            continue
        for tex_name, log_sub in _ABLATION_TABLE_MAP.items():
            if tex_name not in line:
                continue
            best = load_best(log_root / log_sub)
            if not best:
                missing += 1
                break
            hr10 = _fmt_tex(best.get("hr@10"))
            ndcg10 = _fmt_tex(best.get("ndcg@10"))
            if tex_name == "MEMOIR (Full)":
                lines[i] = f"MEMOIR (Full)               & \\textbf{{{hr10}}} & \\textbf{{{ndcg10}}} \\\\"
            else:
                lines[i] = f"\\quad {tex_name:<23} & {hr10} & {ndcg10} \\\\"
            updated += 1
            break
    return updated, missing


def _update_sensitivity_table(lines: list[str], region: tuple[int, int],
                              log_root: Path) -> tuple[int, int]:
    start, end = region
    updated = missing = 0
    for i in range(start, end + 1):
        line = lines[i]
        if "&" not in line or "\\\\" not in line:
            continue
        if "--" not in line:
            continue
        for param_val, log_sub in _SENSITIVITY_TABLE_MAP.items():
            pat = rf"&\s*\\textbf\{{{re.escape(param_val)}\}}|&\s*{re.escape(param_val)}\s"
            if not re.search(pat, line):
                continue
            best = load_best(log_root / log_sub)
            if not best:
                missing += 1
                break
            hr10 = _fmt_tex(best.get("hr@10"))
            is_default = param_val in ("0.07", "6")
            if is_default:
                lines[i] = f"  & \\textbf{{{param_val}}} & \\textbf{{{hr10}}} \\\\"
            else:
                lines[i] = f"  & {param_val} & {hr10} \\\\"
            updated += 1
            break
    return updated, missing


def update_paper(log_root: Path, paper_path: Path):
    if not paper_path.exists():
        print(f"ERROR: {paper_path} not found")
        return

    text = paper_path.read_text()
    lines = text.split("\n")
    updated = missing = 0

    for label, updater in [
        ("tab:main_results", _update_main_table),
        ("tab:ablation", _update_ablation_table),
        ("tab:sensitivity", _update_sensitivity_table),
    ]:
        region = _find_table_region(lines, label)
        if region is None:
            print(f"WARNING: table {label} not found in {paper_path}")
            continue
        u, m = updater(lines, region, log_root)
        updated += u
        missing += m

    new_text = "\n".join(lines)
    if new_text != text:
        paper_path.write_text(new_text)
        print(f"Updated {paper_path}: {updated} values filled, {missing} missing")
    else:
        print(f"No changes to {paper_path} ({missing} models have no results)")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Collect MEMOIR experiment results")
    parser.add_argument("--log-dir", type=str, default="logs",
                        help="Root directory containing experiment logs")
    parser.add_argument("--update-paper", action="store_true",
                        help="Fill placeholder values in paper/main.tex")
    parser.add_argument("--paper-path", type=str, default="paper/main.tex",
                        help="Path to main.tex (default: paper/main.tex)")
    args = parser.parse_args()

    log_root = Path(args.log_dir)
    print_all(log_root)

    if args.update_paper:
        print()
        update_paper(log_root, Path(args.paper_path))


if __name__ == "__main__":
    main()
