"""
Stage 02 — Volume
------------------
Volume-derived features encoding the conviction behind price moves.

Features
--------
  vol_spike     Volume / 20-day rolling mean.
                > 1.5 = abnormal activity (retail herding or institution move).

  vol_accel     Day-over-day change in volume_pct_change — is volume
                accelerating or decelerating into the move?

  vol_relative  5-day vol std / 20-day vol mean — short-term volatility of
                volume itself; spikes here precede regime transitions.

  conviction    close_pos × log(1 + vol_spike) — combines price direction
                with volume intensity. The closest OHLCV proxy for order flow.
                Requires ReturnsStage (close_pos) to run first.

Why these
---------
Vietnam's retail-dominated market exhibits systematic volume patterns:
  - Vol spike on down-day → retail panic (Obs 1 in get_regime.py)
  - Vol spike on up-day   → institutional accumulation or FOMO (Obs 4)
  - Decelerating volume into rally → rally losing fuel (volume_price divergence)
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from ..base import FeatureStage
from ..registry import register

EPS = 1e-8


@register
class VolumeStage(FeatureStage):

    @property
    def name(self) -> str:
        return "volume"

    @property
    def output_columns(self) -> list[str]:
        return ["vol_spike", "vol_accel", "vol_relative", "conviction"]

    def build(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        vol = df["Volume"].astype(float)

        vol_mean20       = vol.rolling(20).mean()
        df["vol_spike"]  = vol / (vol_mean20 + EPS)
        df["vol_accel"]  = vol.pct_change().diff()
        df["vol_relative"] = vol.rolling(5).std() / (vol_mean20 + EPS)

        # conviction requires close_pos from ReturnsStage
        if "close_pos" in df.columns:
            df["conviction"] = df["close_pos"] * np.log1p(df["vol_spike"])
        else:
            df["conviction"] = np.nan

        return df
