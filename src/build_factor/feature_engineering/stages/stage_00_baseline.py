"""
Stage 00 — Baseline
--------------------
Single feature: the overnight gap (open vs prior close).

Rationale
---------
The opening gap is the market's first opinion on fair value. It is the
minimal informative signal derivable from OHLCV data and serves as the
anchor for all subsequent stage comparisons. A model trained on gap alone
sets the floor — any later stage must beat this or it adds no value.

  open_gap = (Open - Close.shift(1)) / Close.shift(1)

Range: typically −0.07 to +0.07 on HOSE (bounded by price limits).
"""

from __future__ import annotations

import pandas as pd
from ..base import FeatureStage
from ..registry import register


@register
class BaselineStage(FeatureStage):

    @property
    def name(self) -> str:
        return "baseline"

    @property
    def output_columns(self) -> list[str]:
        return ["open_gap"]

    def build(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        prev_close = df["Close"].shift(1)
        df["open_gap"] = (df["Open"] - prev_close) / (prev_close.abs() + 1e-8)
        return df
