"""
Stage 03 — Technical Indicators
---------------------------------
Classic momentum and trend indicators derived from price history.

Features
--------
  rsi_14          RSI over 14 periods. > 70 = overbought, < 30 = oversold.
  macd_hist       MACD histogram (12/26/9). Positive = bullish momentum.
  macd_hist_slope Day-over-day change of MACD histogram — inflection detector.
                  Turns positive before price crosses MA; fires early.
  dist_ma5        (Close − MA5) / MA5 — distance from short-term trend.
                  Negative = oversold relative to recent average.
  dist_ma20       (Close − MA20) / MA20 — distance from medium-term trend.
  z_5             5-day z-score: (Close − MA5) / rolling5_std.
                  Standardised deviation useful as an overbought/oversold filter.
  bb_pct          Bollinger Band %B: position within the 20-day, 2-std band.
                  0 = at lower band, 1 = at upper band.

Why these
---------
These are the standard inputs to the regime-aware models in build/.
Adding them in this stage lets us measure precisely how much incremental
lift they provide over raw returns + volume.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from ..base import FeatureStage
from ..registry import register

EPS = 1e-8


def _rsi(close: pd.Series, window: int = 14) -> pd.Series:
    delta = close.diff()
    gain  = delta.clip(lower=0).rolling(window).mean()
    loss  = (-delta.clip(upper=0)).rolling(window).mean()
    rs    = gain / (loss + EPS)
    return 100 - 100 / (1 + rs)


@register
class TechnicalStage(FeatureStage):

    @property
    def name(self) -> str:
        return "technical"

    @property
    def output_columns(self) -> list[str]:
        return [
            "rsi_14",
            "macd_hist",
            "macd_hist_slope",
            "dist_ma5",
            "dist_ma20",
            "z_5",
            "bb_pct",
        ]

    def build(self, df: pd.DataFrame) -> pd.DataFrame:
        df    = df.copy()
        close = df["Close"].astype(float)

        # RSI
        df["rsi_14"] = _rsi(close, 14)

        # MACD
        ema12 = close.ewm(span=12, adjust=False).mean()
        ema26 = close.ewm(span=26, adjust=False).mean()
        macd_line        = ema12 - ema26
        signal_line      = macd_line.ewm(span=9, adjust=False).mean()
        df["macd_hist"]  = macd_line - signal_line
        df["macd_hist_slope"] = df["macd_hist"].diff()

        # Distance from moving averages
        ma5  = close.rolling(5).mean()
        ma20 = close.rolling(20).mean()
        df["dist_ma5"]  = (close - ma5)  / (ma5  + EPS)
        df["dist_ma20"] = (close - ma20) / (ma20 + EPS)

        # 5-day z-score
        std5       = close.rolling(5).std()
        df["z_5"]  = (close - ma5) / (std5 + EPS)

        # Bollinger Band %B (20-day, 2 std)
        std20        = close.rolling(20).std()
        upper        = ma20 + 2 * std20
        lower        = ma20 - 2 * std20
        df["bb_pct"] = (close - lower) / (upper - lower + EPS)

        return df
