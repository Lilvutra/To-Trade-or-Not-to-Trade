"""
build_portfolio.py
------------------
Portfolio construction module.

Strategy
--------
Combines profitability scores and risk scores into a tradeable portfolio
using a three-step process:

  Step 1 — Risk Filter
    Exclude any stock with risk_score ≥ threshold (set by score_risk.py,
    regime-adjusted).  This removes distressed, illiquid, or crashing stocks
    before any allocation decision.

  Step 2 — Profitability Ranking
    From the risk-approved universe, select the top TOP_N stocks by
    factor_score (from score_profitability.py).  Factor score is the
    t-stat-weighted composite of FM-validated signals in the current regime.
    This is the primary selection signal.

  Step 3 — Weight Optimisation
    Two weighting schemes available (configurable):

    A. Score-Proportional (default)
         w_i = max(score_i, ε) / Σ max(score_j, ε)
       Allocates more weight to stocks with stronger factor signals.
       Intuition: the model has more confidence in the top-ranked stocks,
       so they should carry more of the portfolio.

    B. Rank-Inverse (more conservative)
         w_i = (N+1 − rank_i) / Σ (N+1 − rank_j)
       Linear decay from top to bottom of the selected universe.
       Less sensitive to exact score values — useful when factor scores
       are noisy or regime detection is uncertain (e.g. transition periods).

  Position Constraints (both schemes)
    - Max weight per stock: MAX_WEIGHT (default 10%)
    - Min weight per stock: MIN_WEIGHT (default 2%)
    - After constraint enforcement, weights are re-normalised to sum to 1.

  Regime-Adjusted Gross Exposure
    The portfolio's total invested capital is scaled by REGIME_EXPOSURE:
      QUIET_BULL    → 100%  (full allocation, momentum-tilted)
      VOLATILE_BULL →  85%  (reduced; risk of sharp intraday reversals)
      QUIET_BEAR    →  50%  (half invested; mean-reversion is selective)
      PANIC_BEAR    →  30%  (heavy cash; systemic risk too high for full long)
    The remainder is held in cash or short-duration instruments.

Profit Estimation
-----------------
For each selected stock:
  proj_return_pct : from score_profitability (linear FM-factor projection over H days)
  expected_pnl    : proj_return_pct × weight × PORTFOLIO_SIZE

Note: proj_return_pct is a relative ranking score, not an absolute forecast.
Treat expected_pnl as indicative magnitude, not a committed prediction.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

# ─────────────────────────────────────────────────────────────────────────────
# Config
# ─────────────────────────────────────────────────────────────────────────────

TOP_N          = 20     # maximum number of stocks in the portfolio
MIN_WEIGHT     = 0.02   # minimum position size (2%)
MAX_WEIGHT     = 0.10   # maximum position size (10%)
PORTFOLIO_SIZE = 1_000_000_000   # VND — notional portfolio for PnL calc

REGIME_EXPOSURE = {
    "QUIET_BULL":    1.00,
    "VOLATILE_BULL": 0.85,
    "QUIET_BEAR":    0.50,
    "PANIC_BEAR":    0.30,
}

WEIGHTING_SCHEME = "score"   # "score" | "rank"


# ─────────────────────────────────────────────────────────────────────────────
# Weight computation
# ─────────────────────────────────────────────────────────────────────────────

def _score_proportional_weights(scores: np.ndarray) -> np.ndarray:
    """
    Allocate proportionally to factor score (floored at ε to avoid zero/negatives).
    """
    clipped = np.maximum(scores, 1e-6)
    return clipped / clipped.sum()


def _rank_inverse_weights(n: int) -> np.ndarray:
    """
    Rank-inverse: stock ranked 1st gets weight ∝ N, stock ranked last gets ∝ 1.
    """
    ranks = np.arange(n, 0, -1, dtype=float)
    return ranks / ranks.sum()


def _apply_constraints(weights: np.ndarray, min_w: float, max_w: float) -> np.ndarray:
    """
    Iteratively clamp weights to [min_w, max_w] and renormalise.
    Runs until convergence (max 20 iterations) to handle interactions.
    """
    w = weights.copy()
    for _ in range(20):
        w = np.clip(w, min_w, max_w)
        total = w.sum()
        if total < 1e-9:
            break
        w /= total
        if np.all((w >= min_w - 1e-9) & (w <= max_w + 1e-9)):
            break
    return w


# ─────────────────────────────────────────────────────────────────────────────
# Core builder
# ─────────────────────────────────────────────────────────────────────────────

def build_portfolio(
    profitability_scores: pd.DataFrame,
    risk_scores:          pd.DataFrame,
    regime:               str  = "QUIET_BULL",
    top_n:                int  = TOP_N,
    min_weight:           float = MIN_WEIGHT,
    max_weight:           float = MAX_WEIGHT,
    weighting:            str  = WEIGHTING_SCHEME,
    portfolio_size:       float = PORTFOLIO_SIZE,
    verbose:              bool  = True,
) -> pd.DataFrame:
    """
    Construct the portfolio from profitability and risk scores.

    Args:
        profitability_scores : output of score_profitability.run_profitability()
        risk_scores          : output of score_risk.run_risk_scoring()
        regime               : current market regime name
        top_n                : max stocks to include
        min_weight / max_weight : position size bounds
        weighting            : "score" (score-proportional) or "rank" (rank-inverse)
        portfolio_size       : notional capital in VND for expected PnL display

    Returns:
        DataFrame with columns:
          [ticker, factor_score, proj_return_pct, risk_score, risk_label,
           weight, weight_pct, exposure_pct, expected_pnl_vnd,
           regime, gross_exposure]
    """
    # ── Step 1: Risk filter ───────────────────────────────────────────────────
    approved = risk_scores[~risk_scores["exclude"]]["ticker"].tolist()
    n_excluded = risk_scores["exclude"].sum()
    if verbose:
        print(f"\n── Step 1: Risk Filter ──────────────────────────────────")
        print(f"  Universe:  {len(risk_scores)} stocks")
        print(f"  Excluded:  {n_excluded} stocks  (risk_score ≥ {risk_scores['threshold'].iloc[0]:.2f})")
        print(f"  Approved:  {len(approved)} stocks")

    # ── Step 2: Profitability ranking (within risk-approved universe) ─────────
    prof_filtered = (
        profitability_scores[profitability_scores["ticker"].isin(approved)]
        .sort_values("factor_score", ascending=False)
        .head(top_n)
        .reset_index(drop=True)
    )
    if verbose:
        print(f"\n── Step 2: Profitability Ranking ────────────────────────")
        print(f"  Selecting top {top_n} by factor_score in {regime}")
        print(f"  Selected: {len(prof_filtered)} stocks")

    if prof_filtered.empty:
        print("  WARNING: No stocks passed the combined filter. Returning empty portfolio.")
        return pd.DataFrame()

    # ── Step 3: Weight optimisation ───────────────────────────────────────────
    n = len(prof_filtered)
    scores_arr = prof_filtered["factor_score"].values.astype(float)

    if weighting == "score":
        raw_weights = _score_proportional_weights(scores_arr)
    else:
        raw_weights = _rank_inverse_weights(n)

    # Enforce min/max position constraints
    constrained = _apply_constraints(raw_weights, min_weight, max_weight)

    # Regime-adjusted gross exposure
    gross_exposure = REGIME_EXPOSURE.get(regime, 1.0)
    invested_weights = constrained * gross_exposure   # fraction of total capital

    if verbose:
        print(f"\n── Step 3: Weight Optimisation ──────────────────────────")
        print(f"  Scheme:         {weighting}-proportional")
        print(f"  Constraints:    [{min_weight:.0%}, {max_weight:.0%}] per stock")
        print(f"  Regime:         {regime}")
        print(f"  Gross exposure: {gross_exposure:.0%}  (remainder = cash)")

    # ── Merge risk info back ──────────────────────────────────────────────────
    risk_subset = risk_scores[["ticker", "risk_score", "risk_label", "annual_vol",
                                "max_dd", "cvar_5pct", "avg_vol_20"]].copy()
    portfolio = prof_filtered.merge(risk_subset, on="ticker", how="left")

    portfolio["weight"]         = constrained        # fraction of equity allocation
    portfolio["exposure_pct"]   = invested_weights * 100.0   # % of total capital
    portfolio["weight_pct"]     = constrained * 100.0

    # Expected PnL = capital × exposure × projected return
    portfolio["expected_pnl_vnd"] = (
        portfolio_size * invested_weights * portfolio["proj_return_pct"] / 100.0
    )

    portfolio["regime"]         = regime
    portfolio["gross_exposure"] = gross_exposure
    portfolio["rank"]           = np.arange(1, len(portfolio) + 1)

    return portfolio[[
        "rank", "ticker", "factor_score", "proj_return_pct",
        "risk_score", "risk_label", "annual_vol", "max_dd",
        "weight_pct", "exposure_pct", "expected_pnl_vnd",
        "avg_vol_20", "regime", "gross_exposure",
    ]]


# ─────────────────────────────────────────────────────────────────────────────
# Excluded stocks summary
# ─────────────────────────────────────────────────────────────────────────────

def summarise_excluded(risk_scores: pd.DataFrame) -> pd.DataFrame:
    """
    Return the excluded stocks with the primary reason for exclusion.

    Primary reason = the risk dimension with the highest sub-score.
    """
    excl = risk_scores[risk_scores["exclude"]].copy()
    if excl.empty:
        return excl

    sub_scores = {
        "Volatility":  excl["vol_score"],
        "Drawdown":    excl["dd_score"],
        "Tail Risk":   excl["tail_score"],
        "Liquidity":   excl["liq_score"],
        "Trend":       excl["trend_score"],
        "Structural":  excl["struct_score"],
    }
    sub_df      = pd.DataFrame(sub_scores)
    excl["primary_risk_driver"] = sub_df.idxmax(axis=1).values

    return excl[["ticker", "risk_score", "risk_label", "primary_risk_driver",
                  "annual_vol", "max_dd", "regime_name"]].reset_index(drop=True)
