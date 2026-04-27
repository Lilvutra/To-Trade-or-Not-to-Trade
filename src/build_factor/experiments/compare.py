"""
experiments/compare.py
-----------------------
Load the experiment summary and produce a formatted comparison table.

Typical workflow
----------------
After running several experiments via run_experiment.py:

    from experiments.compare import compare_experiments, print_comparison

    df = compare_experiments()   # loads experiments/results/summary.csv
    print_comparison(df)         # formatted table sorted by dir_acc_mean

The table shows, for each run:
  - stage progression (what features were active)
  - directional accuracy mean ± std
  - IC mean ± std
  - MSE mean
  - delta_dir_acc vs the previous run (how much did this stage add?)
"""

from __future__ import annotations

import os
import pandas as pd
import numpy as np

_RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results")


def compare_experiments(
    results_dir: str = _RESULTS_DIR,
    sort_by: str = "dir_acc_mean",
    ascending: bool = False,
) -> pd.DataFrame:
    """
    Load all logged experiments and return a comparison DataFrame.

    Parameters
    ----------
    results_dir : path to the experiments/results/ directory
    sort_by     : column to sort by (default: dir_acc_mean)
    ascending   : sort order

    Returns
    -------
    pd.DataFrame with one row per experiment run, sorted by sort_by.
    """
    summary_path = os.path.join(results_dir, "summary.csv")
    if not os.path.exists(summary_path):
        print(f"[compare] No summary found at {summary_path}. Run experiments first.")
        return pd.DataFrame()

    df = pd.read_csv(summary_path)
    if df.empty:
        return df

    # Sort
    if sort_by in df.columns:
        df = df.sort_values(sort_by, ascending=ascending).reset_index(drop=True)

    # Add delta_dir_acc: improvement over the row above (chronological order)
    if "dir_acc_mean" in df.columns:
        chrono = df.sort_values("timestamp").reset_index(drop=True)
        chrono["delta_dir_acc"] = chrono["dir_acc_mean"].diff()
        df = df.merge(chrono[["run_id", "delta_dir_acc"]], on="run_id", how="left")

    return df


def print_comparison(df: pd.DataFrame, top_n: int | None = None) -> None:
    """
    Print a human-readable comparison table to stdout.

    Parameters
    ----------
    df    : DataFrame from compare_experiments()
    top_n : show only the top N rows (default: all)
    """
    if df.empty:
        print("No experiments to compare.")
        return

    display_cols = [
        "run_id", "stages", "feature_count",
        "dir_acc_mean", "dir_acc_std",
        "ic_mean", "ic_std",
        "mse_mean",
        "n_folds", "n_test_total",
    ]
    if "delta_dir_acc" in df.columns:
        display_cols.append("delta_dir_acc")

    available = [c for c in display_cols if c in df.columns]
    view = df[available].copy()

    if top_n:
        view = view.head(top_n)

    # Format floats for readability
    float_cols = [c for c in view.columns if view[c].dtype == float]
    for col in float_cols:
        if "dir_acc" in col or "ic" in col:
            view[col] = view[col].map(lambda x: f"{x:.4f}" if pd.notna(x) else "nan")
        elif "mse" in col:
            view[col] = view[col].map(lambda x: f"{x:.6f}" if pd.notna(x) else "nan")
        elif "delta" in col:
            view[col] = view[col].map(
                lambda x: f"+{x:.4f}" if (pd.notna(x) and x > 0) else (f"{x:.4f}" if pd.notna(x) else "—")
            )

    sep = "─" * 120
    print(f"\n{'Experiment Comparison':^120}")
    print(sep)
    print(view.to_string(index=False))
    print(sep)
    print(
        f"  dir_acc > 0.53 is considered useful  |  "
        f"IC > 0.05 is considered useful  |  "
        f"delta_dir_acc > 0 means the stage helped\n"
    )


def feature_lift_table(results_dir: str = _RESULTS_DIR) -> pd.DataFrame:
    """
    Return a table showing how each stage addition changed dir_acc_mean.
    Assumes experiments were run in stage order: baseline → returns → volume → technical.
    """
    df = compare_experiments(results_dir, sort_by="timestamp", ascending=True)
    if df.empty:
        return df

    lift_rows = []
    prev_acc = None
    for _, row in df.iterrows():
        lift = (row["dir_acc_mean"] - prev_acc) if prev_acc is not None else None
        lift_rows.append({
            "experiment":    row.get("experiment", ""),
            "stages":        row.get("stages", ""),
            "feature_count": row.get("feature_count", 0),
            "dir_acc_mean":  row.get("dir_acc_mean", float("nan")),
            "ic_mean":       row.get("ic_mean", float("nan")),
            "lift_vs_prev":  lift,
        })
        prev_acc = row.get("dir_acc_mean", prev_acc)

    return pd.DataFrame(lift_rows)
