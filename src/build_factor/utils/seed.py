"""utils/seed.py — reproducibility helper."""

from __future__ import annotations
import random
import numpy as np


def set_seed(seed: int = 42) -> None:
    """Set Python, NumPy (and optionally PyTorch) seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass
