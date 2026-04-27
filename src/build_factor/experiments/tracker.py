"""
experiments/tracker.py
-----------------------
ExperimentTracker — persistent, append-only experiment log.

Each run is saved as an individual JSON file (immutable record) and a
row is appended to summary.csv (fast comparison across runs).

File layout
-----------
  experiments/results/
    summary.csv                    ← one row per run, all metrics flat
    baseline_20260427_143201.json  ← full detail: fold metrics, config, etc.
    returns_20260427_143210.json
    ...

JSON schema
-----------
{
  "run_id"        : "baseline_20260427_143201",
  "timestamp"     : "2026-04-27T14:32:01",
  "experiment"    : "baseline",
  "stages"        : ["baseline"],
  "model"         : "RidgeModel",
  "model_params"  : {"alpha": 1.0},
  "validator"     : {"method": "expanding", "train_size": 252, ...},
  "symbol"        : "VCB",
  "n_rows"        : 1200,
  "feature_count" : 1,
  "features"      : ["open_gap"],
  "fold_metrics"  : [{...}, {...}, ...],   # one dict per fold
  "aggregate"     : {                       # mean/std across folds
      "dir_acc_mean": 0.524, "dir_acc_std": 0.031,
      "ic_mean"     : 0.041, "ic_std"     : 0.018,
      "mse_mean"    : 0.0004, ...
  }
}
"""

from __future__ import annotations

import json
import os
from datetime import datetime

import pandas as pd

from evaluation.metrics import aggregate_fold_metrics

_RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results")
_SUMMARY_CSV = os.path.join(_RESULTS_DIR, "summary.csv")


class ExperimentTracker:
    """
    Log and persist experiment results.

    Parameters
    ----------
    results_dir : str
        Directory where JSON files and summary.csv are stored.
        Defaults to experiments/results/ relative to this file.
    """

    def __init__(self, results_dir: str = _RESULTS_DIR) -> None:
        self.results_dir = results_dir
        os.makedirs(results_dir, exist_ok=True)
        self._summary_path = os.path.join(results_dir, "summary.csv")

    def log(
        self,
        experiment: str,
        stages: list[str],
        features: list[str],
        model_name: str,
        model_params: dict,
        validator_config: dict,
        fold_metrics: list[dict],
        symbol: str = "unknown",
        n_rows: int = 0,
    ) -> str:
        """
        Persist one experiment run.

        Returns
        -------
        str : run_id (use to find the JSON file)
        """
        ts     = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_id = f"{experiment}_{ts}"

        aggregate = aggregate_fold_metrics(fold_metrics)

        record = {
            "run_id":        run_id,
            "timestamp":     datetime.now().isoformat(timespec="seconds"),
            "experiment":    experiment,
            "stages":        stages,
            "model":         model_name,
            "model_params":  model_params,
            "validator":     validator_config,
            "symbol":        symbol,
            "n_rows":        n_rows,
            "feature_count": len(features),
            "features":      features,
            "fold_metrics":  fold_metrics,
            "aggregate":     aggregate,
        }

        # Save full JSON
        json_path = os.path.join(self.results_dir, f"{run_id}.json")
        with open(json_path, "w") as fh:
            json.dump(record, fh, indent=2, default=str)

        # Append summary row
        summary_row = {
            "run_id":        run_id,
            "timestamp":     record["timestamp"],
            "experiment":    experiment,
            "stages":        "|".join(stages),
            "symbol":        symbol,
            "n_rows":        n_rows,
            "feature_count": len(features),
            "model":         model_name,
            **{k: v for k, v in aggregate.items()},
        }
        self._append_summary(summary_row)

        return run_id

    def _append_summary(self, row: dict) -> None:
        df_new = pd.DataFrame([row])
        if os.path.exists(self._summary_path):
            existing = pd.read_csv(self._summary_path)
            combined = pd.concat([existing, df_new], ignore_index=True)
        else:
            combined = df_new
        combined.to_csv(self._summary_path, index=False)

    def load_all(self) -> pd.DataFrame:
        """Return summary.csv as a DataFrame (empty DataFrame if no runs yet)."""
        if not os.path.exists(self._summary_path):
            return pd.DataFrame()
        return pd.read_csv(self._summary_path)

    def load_run(self, run_id: str) -> dict:
        """Load the full JSON record for a given run_id."""
        path = os.path.join(self.results_dir, f"{run_id}.json")
        if not os.path.exists(path):
            raise FileNotFoundError(f"No record found for run_id={run_id!r}")
        with open(path) as fh:
            return json.load(fh)
