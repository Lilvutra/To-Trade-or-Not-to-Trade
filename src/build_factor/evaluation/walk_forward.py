"""
evaluation/walk_forward.py
--------------------------
Time-series aware validation. Never randomly splits data — always respects
the temporal order so there is no look-ahead contamination.

Two validation strategies
--------------------------
rolling   : Fixed-size training window slides forward.
            Simulates a model retrained on the most recent N days only.
            Good for detecting whether a feature is regime-stable.

expanding : Training window grows from a minimum size.
            Simulates using all available history at each refit.
            Good for detecting whether a feature has a persistent edge.

            ┌─────────────────────────── time ──────────────────────────────►
 rolling   │ [train₁][test₁]
            │         [train₂][test₂]
            │                 [train₃][test₃]
            │
 expanding  │ [────train₁────][test₁]
            │ [──────train₂──────][test₂]
            │ [────────train₃────────][test₃]
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from typing import Iterator

from .metrics import compute_metrics


def _rolling_windows(
    n: int,
    train_size: int,
    test_size: int,
    step: int,
) -> Iterator[tuple[np.ndarray, np.ndarray]]:
    """Yield (train_indices, test_indices) for rolling windows."""
    start = 0
    while start + train_size + test_size <= n:
        train_idx = np.arange(start, start + train_size)
        test_idx  = np.arange(start + train_size, start + train_size + test_size)
        yield train_idx, test_idx
        start += step


def _expanding_windows(
    n: int,
    min_train_size: int,
    test_size: int,
    step: int,
) -> Iterator[tuple[np.ndarray, np.ndarray]]:
    """Yield (train_indices, test_indices) for expanding windows."""
    train_end = min_train_size
    while train_end + test_size <= n:
        train_idx = np.arange(0, train_end)
        test_idx  = np.arange(train_end, train_end + test_size)
        yield train_idx, test_idx
        train_end += step


class WalkForwardValidator:
    """
    Run walk-forward cross-validation for a model class over (X, y).

    Parameters
    ----------
    model_cls : callable → BaseModel
        Factory that returns a fresh model instance per fold.
    method : "rolling" | "expanding"
    train_size : int
        Training window size (rolling) or minimum training size (expanding).
    test_size : int
        Number of test observations per fold.
    step : int
        How many observations to advance the window each fold.
        step = test_size  → non-overlapping test folds (default).
        step < test_size  → overlapping test folds.
    """

    def __init__(
        self,
        model_cls,
        method: str = "expanding",
        train_size: int = 252,
        test_size: int = 63,
        step: int | None = None,
    ) -> None:
        if method not in ("rolling", "expanding"):
            raise ValueError("method must be 'rolling' or 'expanding'")
        self.model_cls  = model_cls
        self.method     = method
        self.train_size = train_size
        self.test_size  = test_size
        self.step       = step if step is not None else test_size

    def validate(
        self,
        X: pd.DataFrame,
        y: pd.Series,
    ) -> list[dict]:
        """
        Run all folds. Return list of per-fold metric dicts.

        Each dict has keys from compute_metrics plus 'n_train' and 'n_test'.
        """
        X_arr = X.values.astype(float)
        y_arr = y.values.astype(float)
        n     = len(X_arr)

        window_fn = _rolling_windows if self.method == "rolling" else _expanding_windows
        windows   = list(window_fn(n, self.train_size, self.test_size, self.step))

        if not windows:
            raise ValueError(
                f"No valid folds: n={n}, train_size={self.train_size}, "
                f"test_size={self.test_size}. Need at least {self.train_size + self.test_size} rows."
            )

        fold_results = []
        for train_idx, test_idx in windows:
            X_train, y_train = X_arr[train_idx], y_arr[train_idx]
            X_test,  y_test  = X_arr[test_idx],  y_arr[test_idx]

            # Fresh model per fold — no state leaks between folds
            model = self.model_cls()
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)

            metrics = compute_metrics(y_test, y_pred)
            metrics["n_train"] = len(train_idx)
            metrics["n_test"]  = len(test_idx)
            fold_results.append(metrics)

        return fold_results

    @property
    def config(self) -> dict:
        return {
            "method":     self.method,
            "train_size": self.train_size,
            "test_size":  self.test_size,
            "step":       self.step,
        }
