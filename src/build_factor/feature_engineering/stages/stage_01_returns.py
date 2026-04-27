"""
Stage 01 — Returns
-------------------
Price-return features across multiple horizons plus the log return.

Features
--------
  ret_1      1-day simple return  (momentum / reversal signal)
  ret_5      5-day simple return  (weekly trend)
  ret_20     20-day simple return (monthly trend)
  log_ret    log(Close / Close.shift(1)) — more stable for vol estimation
  close_pos  Intraday close position: (Close−Low) / (High−Low)
             1.0 = closed at high (buyers won the session)
             0.0 = closed at low  (sellers won)

Why these
---------
Short-horizon returns capture mean-reversion in QUIET_BEAR and momentum
in trending regimes. close_pos is the OHLCV proxy for order flow direction
without needing bid/ask data.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from ..base import FeatureStage
from ..registry import register

EPS = 1e-8


@register
class ReturnsStage(FeatureStage):

    @property
    def name(self) -> str:
        return "returns"

    @property
    def output_columns(self) -> list[str]:
        return ["ret_1", "ret_5", "ret_20", "log_ret", "close_pos"]

    def build(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        close = df["Close"]
        df["ret_1"]     = close.pct_change(1)
        df["ret_5"]     = close.pct_change(5)
        df["ret_20"]    = close.pct_change(20)
        df["log_ret"]   = np.log(close / close.shift(1))
        df["close_pos"] = (close - df["Low"]) / (df["High"] - df["Low"] + EPS)
        return df
