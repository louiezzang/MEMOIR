"""Experiment logger with wandb + JSON file support."""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

try:
    import wandb

    _HAS_WANDB = True
except ImportError:
    _HAS_WANDB = False


class ExperimentLogger:
    def __init__(self, config: dict, run_name: str | None = None):
        log_cfg = config.get("logging", {})
        self.log_dir = Path(log_cfg.get("log_dir", "./logs"))
        self.project = log_cfg.get("project", "memoir")
        self.wandb_enabled = log_cfg.get("wandb_enabled", False) and _HAS_WANDB

        if run_name is None:
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            dataset = config.get("data", {}).get("dataset", "unknown")
            run_name = f"{dataset}_{ts}"
        self.run_name = run_name

        self.run_dir = self.log_dir / run_name
        self.run_dir.mkdir(parents=True, exist_ok=True)

        self._records: list[dict] = []
        self._debug_path = self.run_dir / "debug.log"

        # Save config
        with open(self.run_dir / "config.json", "w") as f:
            json.dump(config, f, indent=2, default=str)

        if self.wandb_enabled:
            wandb.init(project=self.project, name=run_name, config=config)
            print(f"[logger] wandb run: {wandb.run.url}")
        else:
            if log_cfg.get("wandb_enabled", False) and not _HAS_WANDB:
                print("[logger] wandb not installed, falling back to file logging")
            print(f"[logger] logging to {self.run_dir}")

    def log_debug(self, message: str):
        ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        line = f"[{ts}] {message}\n"
        with open(self._debug_path, "a") as f:
            f.write(line)
        print(f"[debug] {message}")

    def log_train(self, epoch: int, losses: dict[str, float]):
        record = {"type": "train", "epoch": epoch, **losses}
        self._records.append(record)
        self._flush()

        if self.wandb_enabled:
            wandb.log({f"train/{k}": v for k, v in losses.items()}, step=epoch)

    def log_eval(self, epoch: int, metrics: dict[str, float]):
        record = {"type": "eval", "epoch": epoch, **metrics}
        self._records.append(record)
        self._flush()

        if self.wandb_enabled:
            wandb.log({f"val/{k}": v for k, v in metrics.items()}, step=epoch)

    def finish(self):
        self._flush()
        if self.wandb_enabled:
            wandb.finish()
        print(f"[logger] logs saved to {self.run_dir / 'metrics.json'}")

    def _flush(self):
        with open(self.run_dir / "metrics.json", "w") as f:
            json.dump(self._records, f, indent=2)
