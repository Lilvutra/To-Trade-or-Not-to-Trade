"""
models/base.py
--------------
Minimal interface every model in build_factor must implement.

Keeping this thin means we can swap Ridge for LightGBM, MLP, or the MoE
from build/ without touching evaluation or experiment code.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
import numpy as np


class BaseModel(ABC):

    @abstractmethod
    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        """Train on (X, y). Called once per walk-forward fold."""
        ...

    @abstractmethod
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Return predicted values (regression, not probabilities)."""
        ...

    @property
    @abstractmethod
    def params(self) -> dict:
        """Hyperparameters — logged verbatim to experiment JSON."""
        ...

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}({self.params})"
