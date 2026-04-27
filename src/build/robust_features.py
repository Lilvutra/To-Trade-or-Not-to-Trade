"""
robust_features.py
------------------
Percentile-rank-based feature engineering for Vietnam equities.

Why percentile rank instead of mean/std normalization
------------------------------------------------------
Vietnam equity data violates the assumptions behind z-scores and ratio
normalization in three structural ways:

  1. Heavy tails from limit mechanics
     A single ±7% limit day inflates rolling std for the next 20 sessions,
     compressing every subsequent z-score toward zero — exactly when the
     signal should be strongest.  Percentile rank is unaffected by outlier
     magnitude; only rank order matters.

  2. Skewed volume distribution
     Volume spikes during retail herding episodes are 5-10x normal and
     right-skew the rolling mean used in volume_spike = vol / vol_mean20.
     After one spike, the mean stays elevated and the next genuine spike
     reads as "normal".  Rolling percentile rank re-anchors after each
     observation.

  3. Regime shifts change the mean
     A z-score calibrated in QUIET_BULL is meaningless in PANIC_BEAR.
     Percentile rank is self-calibrating: a "high" reading always means
     the same thing — this value is in the Nth percentile of its own
     recent history — regardless of the regime's level or scale.

Output contract
---------------
All rank features are in [0, 1]:
  0.0 = lowest value seen in the lookback window (oversold / quiet / weak)
  1.0 = highest value seen in the lookback window (overbought / extreme / strong)

This makes them directly comparable across features and stocks without
any further scaling, which benefits both Fama-MacBeth regression (stable
betas) and MoE classifiers (no feature dominates due to scale).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from typing import Optional

EPS = 1e-8
DEFAULT_WINDOW = 60   # ~3 months: enough history, short enough to adapt to regime shifts


# ─────────────────────────────────────────────────────────────────────────────
# Core primitives
# ─────────────────────────────────────────────────────────────────────────────

def rolling_pct_rank(
    series: pd.Series,
    window: int = DEFAULT_WINDOW,
    min_periods: Optional[int] = None,
) -> pd.Series:
    """
    For each day T, compute: fraction of days in [T-window, T-1] where value < value[T].

    This is a strict historical rank — today is NOT included in its own denominator,
    so there is zero look-ahead.  Output is in [0, 1].

    Edge cases
    ----------
    - Fewer than min_periods observations → NaN
    - All tied values → 0.5 (midpoint convention)
    - NaN in series → NaN propagated
    """
    if min_periods is None:
        min_periods = max(window // 4, 5)

    def _rank(arr: np.ndarray) -> float:
        if len(arr) < 2:
            return np.nan
        today = arr[-1]
        if np.isnan(today):
            return np.nan
        hist = arr[:-1]
        hist = hist[~np.isnan(hist)]
        if len(hist) == 0:
            return np.nan
        below = (hist < today).sum()
        equal = (hist == today).sum()
        # mid-rank for ties: (below + 0.5 × equal) / total
        return (below + 0.5 * equal) / len(hist)

    return series.rolling(window + 1, min_periods=min_periods + 1).apply(
        _rank, raw=True
    )


def cross_sectional_rank(
    df: pd.DataFrame,
    feature_col: str,
    date_col: str = "TradingDate",
) -> pd.Series:
    """
    On each date, rank feature_col across all stocks.  Output in [0, 1].
    Used for Fama-MacBeth cross-sectional factor construction.
    """
    return df.groupby(date_col)[feature_col].rank(pct=True)


def winsorize_pct(series: pd.Series, lower: float = 0.01, upper: float = 0.99) -> pd.Series:
    """Clip at rolling [lower, upper] quantile before ranking — handles extreme outliers."""
    lo = series.expanding(min_periods=20).quantile(lower)
    hi = series.expanding(min_periods=20).quantile(upper)
    return series.clip(lower=lo, upper=hi)


# ─────────────────────────────────────────────────────────────────────────────
# Before vs After: feature-by-feature redesign
# ─────────────────────────────────────────────────────────────────────────────
#
# BEFORE                          AFTER                         WHY
# ──────────────────────────────────────────────────────────────────────────
# zscore_return = ret/std(20)     ret_rank = prank(ret, 60)     std spikes on limit days
# vol_relative  = std/mean        vol_rank  = prank(std5, 60)   mean ≈ 0 near flat periods
# volume_spike  = vol/vol_mean    vol_prank = prank(vol, 60)    mean skewed by herding spikes
# mom_5         = close/close[-5] mom_rank  = prank(5d_ret, 60) raw % level-dependent
# rsi_14        = EMA RSI         rsi_rank  = prank(rsi, 60)    threshold 30/70 not universal
# z_window      = (p-ma5)/std5    dist_rank = prank(dist_ma,60) std5 from 5 pts = noisy
# range_expan   = range/mean_rng  range_rank= prank(range,60)   mean inflated by panic days
# gap           = raw gap %       gap_rank  = prank(gap, 60)    outlier-sensitive
# ──────────────────────────────────────────────────────────────────────────


def build_robust_features(
    df: pd.DataFrame,
    close_col:    str = "Close",
    open_col:     str = "Open",
    high_col:     str = "High",
    low_col:      str = "Low",
    vol_col:      str = "Volume",
    time_col:     str = "TradingDate",
    window:       int = DEFAULT_WINDOW,
    limit_thresh: float = 0.07,
) -> pd.DataFrame:
    """
    Drop-in replacement for _build_shared_base + regime feature builders.
    Returns a DataFrame with all robust features appended.

    Parameters
    ----------
    window        : lookback for rolling percentile rank (default 60 = ~3 months)
    limit_thresh  : circuit-breaker threshold (0.07 HOSE, 0.10 HNX, 0.15 UPCOM)
    """
    g = df.copy().sort_values(time_col).reset_index(drop=True)

    close  = g[close_col]
    open_  = g[open_col]
    high   = g[high_col]
    low    = g[low_col]
    volume = g[vol_col]

    # ── Raw building blocks (not features — used to compute ranks) ────────────
    ret       = close.pct_change()
    log_ret   = np.log(close + EPS).diff()
    daily_ret = close / close.shift(1) - 1

    ma5       = close.rolling(5).mean()
    ma20      = close.rolling(20).mean()
    std5      = close.rolling(5).std()
    std20     = ret.rolling(20).std()

    ema12     = close.ewm(span=12, adjust=False).mean()
    ema26     = close.ewm(span=26, adjust=False).mean()
    macd_line = ema12 - ema26
    macd_hist = macd_line - macd_line.ewm(span=9, adjust=False).mean()

    vol_ma20   = volume.rolling(20).mean()
    close_pos  = (close - low) / (high - low + EPS)
    gap_raw    = (open_ - close.shift(1)) / (close.shift(1) + EPS)
    daily_range= (high - low) / (close.shift(1) + EPS)

    rsi = _rsi_raw(close, window=14)

    dist_ma5   = (close - ma5) / (ma5 + EPS)
    mom_5_raw  = close / close.shift(5) - 1
    vol_chg    = volume.pct_change()

    # ── Limit flags (binary — percentile rank not meaningful for binary) ───────
    limit_up   = (daily_ret >  limit_thresh).astype(int)
    limit_down = (daily_ret < -limit_thresh).astype(int)

    def _streak(s: pd.Series) -> pd.Series:
        return s * (s.groupby((s != s.shift()).cumsum()).cumcount() + 1)

    g["limit_up"]          = limit_up
    g["limit_down"]        = limit_down
    g["limit_up_streak"]   = _streak(limit_up)
    g["limit_down_streak"] = _streak(limit_down)

    # ── Percentile-rank features ──────────────────────────────────────────────

    # 1. ret_rank  (replaces zscore_return)
    #    Where does today's return rank in its own 60-day history?
    #    Immune to std inflation from limit days.
    #    0 = worst return in window (extreme sell), 1 = best (extreme buy)
    g["ret_rank"] = rolling_pct_rank(ret, window)

    # 2. vol_rank  (replaces vol_relative = std/mean)
    #    Rolling rank of 5-day realised volatility.
    #    High rank = elevated volatility episode (regime transition signal).
    #    Not distorted by near-zero means during flat markets.
    g["vol_rank"] = rolling_pct_rank(std5, window)

    # 3. volume_rank  (replaces volume_spike = vol/vol_mean)
    #    Rank of today's volume in its 60-day history.
    #    Herding spikes in Vietnam elevate the mean for weeks after; rank is immune.
    g["volume_rank"] = rolling_pct_rank(volume, window)

    # 4. mom_rank  (replaces mom_5 = raw 5-day return)
    #    Rank of 5-day momentum in its history. Comparable across price levels
    #    and between HOSE large-caps and UPCOM small-caps.
    g["mom_rank"] = rolling_pct_rank(mom_5_raw, window)

    # 5. rsi_rank  (replaces raw rsi_14)
    #    Rank of RSI in its own history. The classic 30/70 thresholds assume
    #    a stationary distribution — Vietnam RSI stays elevated for months in
    #    FOMO periods. Rank re-calibrates automatically.
    g["rsi_rank"] = rolling_pct_rank(rsi, window)

    # 6. dist_rank  (replaces dist_ma and z_window)
    #    Rank of distance from MA5. Combines the roles of dist_ma (mean-reversion
    #    signal) and z_window (overbought filter) into one robust feature.
    g["dist_rank"] = rolling_pct_rank(dist_ma5, window)

    # 7. range_rank  (replaces range_expansion = range/range_mean)
    #    Rank of today's intraday range. Panic days create extreme range values
    #    that inflate rolling mean for months; rank avoids this anchor effect.
    g["range_rank"] = rolling_pct_rank(daily_range, window)

    # 8. gap_rank  (replaces raw gap %)
    #    Rank of overnight gap. Large negative gaps (index circuit breaker,
    #    macro shock) are extreme outliers; raw gap % is dominated by a few events.
    g["gap_rank"] = rolling_pct_rank(gap_raw, window)

    # 9. macd_rank  (replaces macd_hist_slope)
    #    Rank of MACD histogram slope. Captures momentum inflection point
    #    without sensitivity to absolute MACD level (which varies by price).
    g["macd_rank"] = rolling_pct_rank(macd_hist.diff(), window)

    # 10. vol_accel_rank  (replaces vol_accel = raw vol_chg.diff())
    #     Rank of volume acceleration. Detects accumulation buildup before
    #     breakout without being distorted by absolute volume levels.
    g["vol_accel_rank"] = rolling_pct_rank(vol_chg.diff(), window)

    # ── Composite features (robust building blocks combined) ──────────────────

    # conviction_close_r  (robust version of conviction_close)
    # Original: close_pos × log1p(vol/vol_mean)  — vol_mean distorted by spikes
    # New:      close_pos × volume_rank           — both components in [0,1]
    # Interpretation: high close_pos + high volume_rank = buyers held with conviction
    g["conviction_close_r"] = close_pos * g["volume_rank"]

    # retail_exhaustion_r  (robust version of retail_exhaustion)
    # Original: (1 - close_pos) × vol/vol_mean
    # New:      (1 - close_pos) × volume_rank
    # Interpretation: closed near low + high volume = retail panic / capitulation
    g["retail_exhaustion_r"] = (1.0 - close_pos) * g["volume_rank"]

    # smart_money_up_r / smart_money_down_r
    # Original: direction_flag × vol_spike (vol_spike = vol/mean)
    # New:      direction_flag × volume_rank
    # Makes the signal comparable across illiquid small-caps and large-caps.
    g["smart_money_up_r"]   = (ret > 0).astype(float) * g["volume_rank"]
    g["smart_money_down_r"] = (ret < 0).astype(float) * g["volume_rank"]

    # limit_up_conviction_r / limit_down_conviction_r
    # Binary limit flag × volume rank — more stable than × vol_mean ratio
    g["limit_up_conviction_r"]   = limit_up   * g["volume_rank"]
    g["limit_down_conviction_r"] = limit_down * g["volume_rank"]

    # intraday_distribution_r
    # Up day but closed near low — ATC institutional selling
    # New: weighted by how extreme the close_pos is (closer to 0 = stronger signal)
    g["intraday_distribution_r"] = (ret > 0).astype(float) * (1.0 - close_pos)

    # Vietnam T+2 features — structure unchanged, raw trigger replaced with rank
    # t2_forced_selling_r: sharp drop 2 days ago × today's volume rank
    #   Original used: (ret.shift(2) < -0.04) — fixed threshold, regime-blind
    #   New uses rank: ret.shift(2) in bottom 15% of its history (adaptive threshold)
    sharp_drop_2d_r           = (rolling_pct_rank(ret, window).shift(2) < 0.15).astype(float)
    g["t2_forced_selling_r"]  = sharp_drop_2d_r * g["volume_rank"]

    # t2_cascade_r: 3-day cumulative drawdown rank × volume rank
    cum_drop_3d               = close / close.shift(3) - 1
    g["t2_cascade_r"]         = (rolling_pct_rank(cum_drop_3d, window) < 0.10).astype(float) * g["volume_rank"]

    # herd_momentum_rank: rank of 10-day up-fraction (replaces raw fraction)
    daily_up                  = (close.diff() > 0).astype(float)
    herd_10                   = daily_up.rolling(10).mean()
    g["herd_momentum_rank"]   = rolling_pct_rank(herd_10, window)

    # margin_cascade_duration: unchanged — it's already a count (0-3), not a ratio
    extreme_down              = (rolling_pct_rank(ret, window) < 0.05).astype(int)
    g["margin_cascade_duration"] = extreme_down.rolling(3, min_periods=1).sum()

    # rsi_divergence: direction disagreement between price and RSI
    # Structure unchanged — it's already a discrete signal (−1, 0, +1)
    price_up                  = (close.diff() > 0).astype(int)
    rsi_up                    = (rsi.diff() > 0).astype(int)
    g["rsi_divergence"]       = price_up - rsi_up

    # limit_open_reversal_r: hit limit-up yesterday + closed near low today
    g["limit_open_reversal_r"] = limit_up.shift(1).fillna(0) * (1.0 - close_pos)

    # vol_price_divergence_r: price up but volume falling (rank-based)
    # Original: (ret > 0) × (-vol_chg)  — raw pct change
    # New: up_flag × (1 - volume_rank) — volume in bottom of range = divergence
    g["vol_price_divergence_r"] = (ret > 0).astype(float) * (1.0 - g["volume_rank"])

    # delta_dist_rank: velocity of price vs MA5 — rank of the daily change in dist_ma5
    g["delta_dist_rank"] = rolling_pct_rank(dist_ma5.diff(), window)

    return g


# ─────────────────────────────────────────────────────────────────────────────
# RSI helper (pure numpy, no dependency on build_features)
# ─────────────────────────────────────────────────────────────────────────────

def _rsi_raw(close: pd.Series, window: int = 14) -> pd.Series:
    delta = close.diff()
    gain  = delta.clip(lower=0)
    loss  = (-delta).clip(lower=0)
    avg_gain = gain.ewm(com=window - 1, min_periods=window).mean()
    avg_loss = loss.ewm(com=window - 1, min_periods=window).mean()
    rs = avg_gain / (avg_loss + EPS)
    return 100 - (100 / (1 + rs))


# ─────────────────────────────────────────────────────────────────────────────
# Mapping: old feature name → new robust replacement
# ─────────────────────────────────────────────────────────────────────────────

FEATURE_MAP = {
    # old                       new robust replacement
    "zscore_return":             "ret_rank",
    "vol_relative":              "vol_rank",
    "volume_spike":              "volume_rank",
    "mom_5":                     "mom_rank",
    "rsi_14":                    "rsi_rank",
    "dist_ma":                   "dist_rank",
    "z_window":                  "dist_rank",
    "range_expansion":           "range_rank",
    "gap":                       "gap_rank",
    "macd_hist_slope":           "macd_rank",
    "vol_accel":                 "vol_accel_rank",
    "delta_dist":                "delta_dist_rank",
    "conviction_close":          "conviction_close_r",
    "retail_exhaustion":         "retail_exhaustion_r",
    "smart_money_up":            "smart_money_up_r",
    "smart_money_down":          "smart_money_down_r",
    "limit_up_conviction":       "limit_up_conviction_r",
    "limit_down_conviction":     "limit_down_conviction_r",
    "intraday_distribution":     "intraday_distribution_r",
    "t2_forced_selling":         "t2_forced_selling_r",
    "t2_cascade":                "t2_cascade_r",
    "herd_momentum_10":          "herd_momentum_rank",
    "limit_open_reversal":       "limit_open_reversal_r",
    "volume_price_divergence":   "vol_price_divergence_r",
    # unchanged (already robust or binary)
    "limit_up":                  "limit_up",
    "limit_down":                "limit_down",
    "limit_up_streak":           "limit_up_streak",
    "limit_down_streak":         "limit_down_streak",
    "rsi_divergence":            "rsi_divergence",
    "margin_cascade_duration":   "margin_cascade_duration",
    "close_position":            "close_position",     # already [0,1]
}


def translate_feature_list(old_features: list[str]) -> list[str]:
    """Convert a list of old feature names to their robust replacements."""
    return [FEATURE_MAP.get(f, f) for f in old_features]


# ─────────────────────────────────────────────────────────────────────────────
# Optional: hybrid features (percentile rank + raw signal clipped)
# ─────────────────────────────────────────────────────────────────────────────
#
# For some features the raw direction still matters (sign of gap, sign of ret),
# but the magnitude should be rank-normalized.  A hybrid captures both:
#
#   hybrid_gap = sign(gap) × gap_rank
#
# This preserves the directional information (positive/negative gap) while
# normalizing the magnitude via rank.  Useful for Fama-MacBeth where you want
# signed betas.

def build_hybrid_features(g: pd.DataFrame) -> pd.DataFrame:
    """
    Add signed-rank hybrid features for use in Fama-MacBeth regressions
    where the sign of the factor matters, not just the magnitude.
    """
    if "gap_rank" not in g.columns or "ret_rank" not in g.columns:
        raise ValueError("Run build_robust_features first.")

    gap_raw = (g["Open"] - g["Close"].shift(1)) / (g["Close"].shift(1) + EPS)
    ret_raw = g["Close"].pct_change()

    # signed rank: [-1, +1] — preserves direction, normalizes size
    g["gap_signed_rank"]  = np.sign(gap_raw)  * g["gap_rank"]
    g["ret_signed_rank"]  = np.sign(ret_raw)  * g["ret_rank"]
    g["mom_signed_rank"]  = np.sign(g["Close"] / g["Close"].shift(5) - 1) * g["mom_rank"]

    return g


# ─────────────────────────────────────────────────────────────────────────────
# Smoke test
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import os

    data_path = "./data/data-vn-20230228/stock-historical-data/VCB-VNINDEX-History.csv"
    raw = pd.read_csv(data_path)
    raw["TradingDate"] = pd.to_datetime(raw["TradingDate"])

    out = build_robust_features(raw)
    out = build_hybrid_features(out)

    rank_cols = [c for c in out.columns if c.endswith("_rank") or c.endswith("_r")]
    print(f"Rows: {len(out)}  |  Robust features: {len(rank_cols)}")
    print("\nFeature ranges (should all be in [0, 1] for rank features):")
    print(out[rank_cols].agg(["min", "max", "mean"]).round(3).T.to_string())

    print("\nNaN counts per feature:")
    nan_counts = out[rank_cols].isna().sum()
    print(nan_counts[nan_counts > 0].to_string())
