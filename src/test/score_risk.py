"""
score_risk.py
-------------
Risk scoring model for Vietnamese equities.

Methodology
-----------
Each stock is scored across six risk dimensions, each normalised to [0, 1]
(0 = safest, 1 = riskiest).  A composite risk score is the weighted average.
Stocks above RISK_THRESHOLD are flagged for exclusion from the portfolio.

Risk dimensions
───────────────

1. Volatility Risk (weight 0.25)
   annual_vol = rolling_60d_std(daily_ret) × √252
   Risk ∝ annual_vol ranked within universe.
   Rationale: high realised vol amplifies drawdown risk and reduces the
   precision of the factor score prediction (noisier returns).

2. Drawdown Risk (weight 0.25)
   max_dd = 1 − close / max(close, window=252)
   The maximum loss from a trailing-year peak.  Stocks in deep drawdowns
   are often in distress — earnings deterioration, forced selling, or
   structural decline.  In Vietnam, drawdowns can persist due to thin
   order books and T+2 settlement overhang.

3. Tail Risk — CVaR (weight 0.20)
   cvar_5pct = mean of worst 5% daily returns (trailing 252d)
   Conditional Value-at-Risk at 5%.  Captures asymmetric downside that
   variance misses: a stock with several −6% days (near the HOSE limit)
   has much worse CVaR than one with a smooth −0.3%/day decline even if
   their standard deviation is similar.

4. Liquidity Risk (weight 0.15)
   liq_ratio = avg_volume_20 / median(avg_volume_20, universe)
   Low relative volume = higher execution cost and market-impact risk.
   In Vietnam's retail-dominated market, thinly traded stocks can gap
   violently on any institutional order flow.  Stocks below 20% of the
   universe median volume are penalised heavily.

5. Trend Risk (weight 0.10)
   Composite of:
     a. Detected stock-level regime (PANIC_BEAR=1.0, QUIET_BEAR=0.7,
        QUIET_BULL=0.2, VOLATILE_BULL=0.4)
     b. fraction_up_20 < 0.45  (more down-days than up-days recently)
   Stocks in confirmed downtrends carry adverse momentum — the same factor
   signals that work in bear markets often take several sessions to play
   out, increasing the probability of being trapped in a continued decline.

6. Vietnam Structural Risk (weight 0.05)
   margin_risk proxy: consecutive_down_days_near_limit (T+2 cascade).
   When a stock has experienced multiple near-limit-down sessions, brokers
   are likely to issue margin calls at the next open, mechanically
   amplifying any further drop.  This is a pure execution/liquidity risk
   specific to Vietnam's settlement structure.

Composite score
   risk_score = Σ_k weight_k × normalised_score_k

Exclusion threshold
   Default: risk_score ≥ 0.60  →  excluded from portfolio
   In PANIC_BEAR regime the threshold tightens to 0.45 to be more
   conservative when systemic risk is elevated.
"""

from __future__ import annotations

import os
import sys
import glob

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "build"))

from get_regime import build_conditional_features, REGIME_NAME, REGIME_ID

# ─────────────────────────────────────────────────────────────────────────────
# Config
# ─────────────────────────────────────────────────────────────────────────────

DATA_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..", "data",
                 "data-vn-20230228", "stock-historical-data")
)

N_STOCKS         = 150
RISK_THRESHOLD   = 0.60     # default exclusion threshold
RISK_THRESH_PANIC = 0.45    # stricter threshold in PANIC_BEAR regime
VOL_WINDOW       = 60       # days for realised vol
DD_WINDOW        = 252      # days for max drawdown
LIQRATIO_FLOOR   = 0.20     # stocks below 20% of median volume are high-risk

RISK_WEIGHTS = {
    "volatility":  0.25,
    "drawdown":    0.25,
    "tail_risk":   0.20,
    "liquidity":   0.15,
    "trend":       0.10,
    "structural":  0.05,
}

# Regime risk contribution (0 = safe, 1 = very risky)
REGIME_RISK_SCORE = {
    "QUIET_BULL":    0.20,
    "VOLATILE_BULL": 0.40,
    "QUIET_BEAR":    0.70,
    "PANIC_BEAR":    1.00,
}


# ─────────────────────────────────────────────────────────────────────────────
# Per-stock risk metric computation
# ─────────────────────────────────────────────────────────────────────────────

def _compute_stock_risk_metrics(raw: pd.DataFrame, ticker: str) -> dict:
    """
    Compute raw (un-normalised) risk metrics for a single stock's OHLCV history.
    """
    raw = raw.sort_values("TradingDate").copy()
    close  = raw["Close"].values.astype(float)
    volume = raw["Volume"].values.astype(float)

    daily_ret = np.diff(np.log(close + 1e-6))

    # 1. Realised volatility (60d trailing)
    if len(daily_ret) >= VOL_WINDOW:
        annual_vol = np.std(daily_ret[-VOL_WINDOW:]) * np.sqrt(252)
    elif len(daily_ret) > 5:
        annual_vol = np.std(daily_ret) * np.sqrt(252)
    else:
        annual_vol = np.nan

    # 2. Max drawdown (252d trailing)
    window = min(DD_WINDOW, len(close))
    close_w = close[-window:]
    running_max = np.maximum.accumulate(close_w)
    drawdowns   = 1.0 - close_w / (running_max + 1e-6)
    max_dd      = float(np.max(drawdowns))

    # 3. CVaR at 5% (252d trailing)
    ret_w = daily_ret[-min(DD_WINDOW, len(daily_ret)):]
    if len(ret_w) >= 20:
        threshold = np.percentile(ret_w, 5)
        cvar = float(np.mean(ret_w[ret_w <= threshold]))  # negative number
    else:
        cvar = float(np.min(ret_w)) if len(ret_w) > 0 else np.nan

    # 4. Average volume (20d)
    avg_vol_20 = float(np.mean(volume[-20:])) if len(volume) >= 20 else float(np.mean(volume))

    # 5. Trend: fraction of up-days in last 20 sessions
    fraction_up = float(np.mean(daily_ret[-20:] > 0)) if len(daily_ret) >= 20 else 0.5

    # 6. Structural (Vietnam): consecutive sessions within 5% of limit-down (−7%)
    limit_zone = -0.065   # flag sessions where daily return ≤ -6.5%
    recent_ret = daily_ret[-10:] if len(daily_ret) >= 10 else daily_ret
    limit_hits  = int(np.sum(recent_ret <= limit_zone))

    # Detected stock-level regime
    try:
        raw["return_1"] = pd.Series(close).pct_change().shift(-1).values
        raw["volatility"] = pd.Series(close).pct_change().rolling(20).std().values
        feat = build_conditional_features(raw, market_index="VNINDEX")
        last_regime = int(feat["regime"].dropna().iloc[-1])
        regime_name = REGIME_NAME.get(last_regime, "QUIET_BULL")
    except Exception:
        regime_name = "QUIET_BULL"

    return {
        "ticker":       ticker,
        "annual_vol":   annual_vol,
        "max_dd":       max_dd,
        "cvar_5pct":    cvar,      # negative → worse = more negative
        "avg_vol_20":   avg_vol_20,
        "fraction_up":  fraction_up,
        "limit_hits":   limit_hits,
        "regime_name":  regime_name,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Universe risk scoring
# ─────────────────────────────────────────────────────────────────────────────

def _rank_normalise(series: pd.Series) -> pd.Series:
    """Rank-normalise a series to [0, 1] (higher rank = higher raw value = riskier)."""
    return series.rank(pct=True, na_option="bottom")


def compute_risk_scores(
    metrics: pd.DataFrame,
    market_regime: str = "QUIET_BULL",
) -> pd.DataFrame:
    """
    Normalise raw metrics and compute composite risk scores.

    Args:
        metrics        : DataFrame output of load_risk_metrics()
        market_regime  : current market regime name (adjusts exclusion threshold)

    Returns:
        DataFrame with columns:
          [ticker, vol_score, dd_score, tail_score, liq_score, trend_score,
           struct_score, risk_score, risk_label, exclude]
    """
    df = metrics.copy()

    # ── 1. Volatility score ───────────────────────────────────────────────────
    df["vol_score"] = _rank_normalise(df["annual_vol"].fillna(df["annual_vol"].median()))

    # ── 2. Drawdown score ─────────────────────────────────────────────────────
    df["dd_score"] = _rank_normalise(df["max_dd"].fillna(df["max_dd"].median()))

    # ── 3. Tail risk score ────────────────────────────────────────────────────
    # CVaR is negative; more negative = worse → negate before ranking
    df["tail_score"] = _rank_normalise(-df["cvar_5pct"].fillna(df["cvar_5pct"].median()))

    # ── 4. Liquidity score ────────────────────────────────────────────────────
    # Low volume = high risk → invert (higher rank = higher risk = lower volume)
    vol_median = df["avg_vol_20"].median()
    liq_ratio  = df["avg_vol_20"] / (vol_median + 1e-6)
    # Map ratio to risk: ratio < LIQRATIO_FLOOR → risk=1.0, ratio=1.0 → risk=0.0 (floor), 2x+ → very low risk
    liq_risk   = np.clip(1.0 - (liq_ratio / 2.0), 0.0, 1.0)  # monotone decreasing with volume
    df["liq_score"] = liq_risk.values

    # ── 5. Trend score ────────────────────────────────────────────────────────
    regime_risk = df["regime_name"].map(REGIME_RISK_SCORE).fillna(0.5)
    trend_flag  = (df["fraction_up"] < 0.45).astype(float) * 0.3
    df["trend_score"] = (regime_risk * 0.7 + trend_flag).clip(0, 1)

    # ── 6. Structural (Vietnam) risk score ────────────────────────────────────
    df["struct_score"] = (df["limit_hits"] / 5.0).clip(0, 1)   # 5+ hits in 10d → score=1

    # ── Composite risk score ──────────────────────────────────────────────────
    w = RISK_WEIGHTS
    df["risk_score"] = (
        w["volatility"]  * df["vol_score"]    +
        w["drawdown"]    * df["dd_score"]      +
        w["tail_risk"]   * df["tail_score"]    +
        w["liquidity"]   * df["liq_score"]     +
        w["trend"]       * df["trend_score"]   +
        w["structural"]  * df["struct_score"]
    )

    # ── Risk label ────────────────────────────────────────────────────────────
    def _label(s):
        if s >= 0.80: return "VERY HIGH"
        if s >= 0.65: return "HIGH"
        if s >= 0.45: return "MEDIUM"
        if s >= 0.25: return "LOW"
        return "VERY LOW"

    df["risk_label"] = df["risk_score"].apply(_label)

    # ── Exclusion flag ────────────────────────────────────────────────────────
    threshold = RISK_THRESH_PANIC if market_regime == "PANIC_BEAR" else RISK_THRESHOLD
    df["exclude"]   = df["risk_score"] >= threshold
    df["threshold"] = threshold

    return df.sort_values("risk_score", ascending=False).reset_index(drop=True)


# ─────────────────────────────────────────────────────────────────────────────
# Universe loader
# ─────────────────────────────────────────────────────────────────────────────

def load_risk_metrics(n_stocks: int = N_STOCKS, verbose: bool = True) -> pd.DataFrame:
    """
    Load OHLCV history for each stock and compute raw risk metrics.
    """
    csv_files = sorted(glob.glob(os.path.join(DATA_DIR, "*-VNINDEX-History.csv")))[:n_stocks]
    rows = []
    for path in csv_files:
        ticker = os.path.basename(path).split("-")[0]
        raw = pd.read_csv(path)
        raw["TradingDate"] = pd.to_datetime(raw["TradingDate"])
        if len(raw) < 30:
            continue
        try:
            metrics = _compute_stock_risk_metrics(raw, ticker)
            rows.append(metrics)
        except Exception as e:
            if verbose:
                print(f"  [skip] {ticker}: {e}")

    df = pd.DataFrame(rows)
    if verbose:
        print(f"Risk metrics computed for {len(df)} stocks")
    return df


# ─────────────────────────────────────────────────────────────────────────────
# Public entry point
# ─────────────────────────────────────────────────────────────────────────────

def run_risk_scoring(
    n_stocks:      int  = N_STOCKS,
    market_regime: str  = "QUIET_BULL",
    verbose:       bool = True,
) -> pd.DataFrame:
    """
    Full risk scoring pipeline.

    Returns:
        DataFrame with risk scores and exclusion flags for all stocks,
        sorted by risk_score descending (riskiest first).
    """
    if verbose:
        print(f"Computing risk metrics for {n_stocks} VNINDEX stocks…")
    metrics = load_risk_metrics(n_stocks=n_stocks, verbose=verbose)

    if verbose:
        print(f"Scoring risks (market regime: {market_regime})…")
    scored = compute_risk_scores(metrics, market_regime=market_regime)

    n_excl = scored["exclude"].sum()
    if verbose:
        threshold = scored["threshold"].iloc[0]
        print(f"\nRisk summary (threshold={threshold:.2f}):")
        print(f"  Excluded (risk ≥ {threshold:.2f}): {n_excl} stocks")
        print(f"  Approved for portfolio:            {len(scored) - n_excl} stocks")
        print("\nTop 10 riskiest stocks:")
        cols = ["ticker", "risk_score", "risk_label", "annual_vol", "max_dd",
                "cvar_5pct", "regime_name", "exclude"]
        print(scored[cols].head(10).to_string(index=False))

    return scored
