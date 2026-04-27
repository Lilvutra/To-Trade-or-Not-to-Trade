"""
evaluation/metrics.py
---------------------
All evaluation metrics for feature experiments.

Primary metric   : directional_accuracy — does the model predict the sign
                   of the next return correctly? This is the most actionable
                   metric for a long/short strategy.

Secondary metrics: MSE, MAE (magnitude accuracy), IC (rank correlation).
                   IC (Information Coefficient) is the standard quant metric:
                   Spearman rank correlation between predicted and realised
                   returns. IC > 0.05 is considered useful in practice.
"""

from __future__ import annotations

import numpy as np
from scipy import stats as scipy_stats


def directional_accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Fraction of predictions with the correct sign."""
    mask = y_true != 0
    if mask.sum() == 0:
        return float("nan")
    return float(np.mean(np.sign(y_true[mask]) == np.sign(y_pred[mask])))


def information_coefficient(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Spearman rank correlation (IC). NaN if constant input."""
    if len(y_true) < 3:
        return float("nan")
    rho, _ = scipy_stats.spearmanr(y_true, y_pred, nan_policy="omit")
    return float(rho) if not np.isnan(rho) else float("nan")


def mse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean((y_true - y_pred) ** 2))


def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean(np.abs(y_true - y_pred)))


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    """
    Return all metrics in a single dict.

    Keys: mse, mae, dir_acc, ic
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    return {
        "mse":     mse(y_true, y_pred),
        "mae":     mae(y_true, y_pred),
        "dir_acc": directional_accuracy(y_true, y_pred),
        "ic":      information_coefficient(y_true, y_pred),
    }


def aggregate_fold_metrics(folds: list[dict]) -> dict[str, float]:
    """
    Average metrics across walk-forward folds.

    Parameters
    ----------
    folds : list of dicts, each with keys from compute_metrics + 'n_test'

    Returns
    -------
    Dict with mean and std per metric, e.g. {"dir_acc_mean": 0.53, "dir_acc_std": 0.04, ...}
    """
    if not folds:
        return {}

    keys = [k for k in folds[0] if k != "n_test"]
    result: dict[str, float] = {}
    for k in keys:
        vals = [f[k] for f in folds if not np.isnan(f.get(k, float("nan")))]
        if vals:
            result[f"{k}_mean"] = float(np.mean(vals))
            result[f"{k}_std"]  = float(np.std(vals))
        else:
            result[f"{k}_mean"] = float("nan")
            result[f"{k}_std"]  = float("nan")

    result["n_folds"]     = len(folds)
    result["n_test_total"] = int(sum(f.get("n_test", 0) for f in folds))
    return result
