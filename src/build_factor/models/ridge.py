"""
models/ridge.py
---------------
Ridge regression wrapper — the default baseline model for feature experiments.

Why Ridge and not something fancier?
  - Linear + L2 gives interpretable coefficients: you can inspect which
    features have large betas to understand what the model learned.
  - Low variance: won't overfit on small training windows during walk-forward.
  - Fast: hundreds of folds run in seconds, keeping iteration tight.
  - Easy to beat: if a feature set can't improve over Ridge, it is not worth
    carrying into the heavier MoE / LSTM pipeline.
"""

from __future__ import annotations

import numpy as np
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from .base import BaseModel


class RidgeModel(BaseModel):
    """
    Standardised Ridge regression.

    Features are z-scored per fold (fit on train, transform on test) so
    coefficient magnitudes are comparable across features and folds.

    Parameters
    ----------
    alpha : float
        L2 regularisation strength. Default 1.0.
    """

    def __init__(self, alpha: float = 1.0) -> None:
        self._alpha   = alpha
        self._scaler  = StandardScaler()
        self._model   = Ridge(alpha=alpha)
        self._fitted  = False

    @property
    def params(self) -> dict:
        return {"alpha": self._alpha}

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        X_scaled = self._scaler.fit_transform(X)
        self._model.fit(X_scaled, y)
        self._fitted = True

    def predict(self, X: np.ndarray) -> np.ndarray:
        if not self._fitted:
            raise RuntimeError("Call fit() before predict().")
        X_scaled = self._scaler.transform(X)
        return self._model.predict(X_scaled)

    def feature_importance(self, feature_names: list[str]) -> dict[str, float]:
        """Return absolute coefficient per feature (after standardisation)."""
        if not self._fitted:
            raise RuntimeError("Call fit() before feature_importance().")
        return dict(zip(feature_names, self._model.coef_.tolist()))
