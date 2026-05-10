"""
score_profitability.py
----------------------
Profitable stock selection module.

Methodology
-----------
We reuse the Fama-MacBeth (FM) output already produced by plot_cross_sectional.py.
The FM regression delivers, for each feature f:
  - mean_β_f  : average daily return earned by a stock that scores +1σ on feature f
  - t_stat_f  : how reliably that premium appeared across all cross-sections

Composite profitability score for stock i in regime r:
    score_i  = Σ_f  t_f  × z_f,i          (t-stat weighted, unitless ranking score)
    proj_ret = Σ_f  β_f  × z_f,i  × H     (expected return over H trading days)

where z_f,i is the cross-sectional z-score of feature f for stock i on the eval date.

Why t-stat weighting?
  A feature with β=0.003 but t=10 should outweigh one with β=0.005 but t=0.6.
  Using t-stats as weights folds in statistical confidence so noisy signals
  automatically shrink — no manual threshold needed.

Why cross-sectional z-score?
  FM was estimated on cross-sectionally standardised features.  The coefficient β
  carries units of "return per 1 σ cross-sectional deviation", so applying it to
  a raw feature value would give the wrong number.  Re-standardising the features
  at evaluation time puts us in the same units as the original regression.

Projected return caveat
  proj_ret = mean_β × z × H  is a linear extrapolation.
  It does NOT account for: regime changes, factor mean-reversion, or compounding.
  Treat it as a relative ranking tool, not an absolute return forecast.
"""

from __future__ import annotations

import os
import sys
import glob

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "build"))

from get_regime import build_conditional_features, REGIME_FEATURES, REGIME_NAME, REGIME_ID

# ─────────────────────────────────────────────────────────────────────────────
# Config
# ─────────────────────────────────────────────────────────────────────────────

DATA_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..", "data",
                 "data-vn-20230228", "stock-historical-data")
)
CS_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..", "data", "cross_sectional")
)

N_STOCKS     = 150   # how many VNINDEX stocks to evaluate
TSTAT_MIN    = 1.5   # minimum |t| to include a feature in the composite score
PROJ_HORIZON = 20    # default projection horizon in trading days
MAX_BETA_CAP = 0.005 # cap |mean_β| to avoid scale-outlier features dominating return projection

# Regime-adjusted gross exposure (fraction of capital deployed)
REGIME_EXPOSURE = {
    "QUIET_BULL":    1.00,
    "VOLATILE_BULL": 0.85,
    "QUIET_BEAR":    0.50,
    "PANIC_BEAR":    0.30,
}


# ─────────────────────────────────────────────────────────────────────────────
# FM coefficient loader
# ─────────────────────────────────────────────────────────────────────────────

def load_fm_coefficients() -> dict[str, pd.DataFrame]:
    """
    Load pre-computed FM summaries from data/cross_sectional/.

    Returns a dict keyed by regime name (e.g. 'QUIET_BULL') plus 'full'
    for the full-sample model.  Each value is a DataFrame with columns
    [mean, std, se_nw, t_stat].
    """
    coefs: dict[str, pd.DataFrame] = {}

    full_path = os.path.join(CS_DIR, "full_summary.csv")
    if os.path.exists(full_path):
        coefs["full"] = pd.read_csv(full_path, index_col=0)

    for name in REGIME_NAME.values():
        path = os.path.join(CS_DIR, f"regime_{name}.csv")
        if os.path.exists(path):
            coefs[name] = pd.read_csv(path, index_col=0)

    if not coefs:
        raise FileNotFoundError(
            f"No FM summary CSVs found in {CS_DIR}. "
            "Run plot_cross_sectional.py first."
        )
    return coefs


# ─────────────────────────────────────────────────────────────────────────────
# Current regime detector
# ─────────────────────────────────────────────────────────────────────────────

def detect_current_regime() -> str:
    """
    Detect the current market regime using VCB as a VNINDEX proxy.
    Returns the regime name string (e.g. 'QUIET_BULL').
    """
    vcb_path = os.path.join(DATA_DIR, "VCB-VNINDEX-History.csv")
    raw = pd.read_csv(vcb_path)
    raw["TradingDate"] = pd.to_datetime(raw["TradingDate"])
    feat = build_conditional_features(raw, market_index="VNINDEX")
    last_regime_id = int(feat["regime"].dropna().iloc[-1])
    return REGIME_NAME.get(last_regime_id, "QUIET_BULL")


# ─────────────────────────────────────────────────────────────────────────────
# Feature builder for the universe
# ─────────────────────────────────────────────────────────────────────────────

def load_universe(n_stocks: int = N_STOCKS) -> pd.DataFrame:
    """
    Load the most recent cross-section for all VNINDEX stocks.

    Returns a panel DataFrame with one row per stock (latest available date)
    plus all engineered features.
    """
    csv_files = sorted(glob.glob(os.path.join(DATA_DIR, "*-VNINDEX-History.csv")))[:n_stocks]
    rows = []
    for path in csv_files:
        ticker = os.path.basename(path).split("-")[0]
        raw = pd.read_csv(path)
        raw["TradingDate"] = pd.to_datetime(raw["TradingDate"])
        raw["return_1"] = raw["Close"].pct_change().shift(-1)
        raw["volatility"] = raw["Close"].pct_change().rolling(20).std()

        feat = build_conditional_features(raw, market_index="VNINDEX")
        last = feat.dropna(subset=["regime"]).tail(1).copy()
        if last.empty:
            continue
        last["ticker"] = ticker
        last["latest_close"] = raw["Close"].iloc[-1]
        last["avg_volume_20"] = raw["Volume"].rolling(20).mean().iloc[-1]
        rows.append(last)

    panel = pd.concat(rows, ignore_index=True)
    panel = panel.rename(columns={"TradingDate": "date"})
    return panel


# ─────────────────────────────────────────────────────────────────────────────
# Composite profitability score
# ─────────────────────────────────────────────────────────────────────────────

def compute_profitability_scores(
    panel:      pd.DataFrame,
    fm_coefs:   dict[str, pd.DataFrame],
    regime:     str,
    tstat_min:  float = TSTAT_MIN,
    horizon:    int   = PROJ_HORIZON,
) -> pd.DataFrame:
    """
    Compute two scores for each stock:

    factor_score  (unitless, used for ranking)
        Weighted sum of z-scored features where the weight = t-stat of the FM model.
        Sign of t-stat encodes direction: high positive score = strong buy signal.
        Only features with |t| >= tstat_min are included.

    proj_return  (in daily return units × horizon)
        Uses regime-specific FM mean_β coefficients (capped at MAX_BETA_CAP to
        avoid scale outliers from low-significance, large-coefficient features).
        Approximates H-day cumulative expected return from factor exposure.

    Args:
        panel     : latest cross-section of stocks with engineered features
        fm_coefs  : dict output of load_fm_coefficients()
        regime    : current regime name (e.g. 'QUIET_BULL')
        tstat_min : minimum |t| to include a feature
        horizon   : projection horizon in trading days

    Returns:
        DataFrame sorted by factor_score descending, with columns:
        [ticker, regime, factor_score, proj_return_pct, regime_features_used, ...]
    """
    # Pick the FM summary for the current regime; fall back to full-sample
    fm_regime = fm_coefs.get(regime, fm_coefs.get("full"))
    fm_full   = fm_coefs.get("full", fm_regime)

    # Features active in this regime (domain knowledge prior)
    regime_id     = REGIME_ID.get(regime, 2)
    regime_feats  = REGIME_FEATURES.get(regime_id, [])

    # Filter to features that exist in both the FM summary and the panel
    all_feats = [f for f in fm_regime.index if f in panel.columns and f != "const"]
    sig_feats = [f for f in all_feats
                 if abs(fm_regime.loc[f, "t_stat"]) >= tstat_min]

    if not sig_feats:
        sig_feats = all_feats  # fallback: use everything

    # Cross-sectional z-score: same transform applied in FM estimation
    # Replace ±inf with NaN first (vol_accel can be -inf when volume was 0)
    X = panel[sig_feats].copy().astype(float)
    X = X.replace([np.inf, -np.inf], np.nan)
    X_z = (X - X.mean()) / (X.std() + 1e-6)
    # Fill NaN with 0: stock gets neutral contribution for any missing feature
    X_z = X_z.fillna(0.0)

    # ── Factor score: Σ t_f × z_f,i  ─────────────────────────────────────────
    t_weights = fm_regime.loc[sig_feats, "t_stat"].values   # shape (K,)
    scores = X_z.values @ t_weights                          # shape (N,)

    # ── Projected return: Σ β_f × z_f,i × H  ────────────────────────────────
    # Use full-sample β for features not in the regime FM
    betas = []
    for f in sig_feats:
        if f in fm_full.index:
            b = float(fm_full.loc[f, "mean"])
        else:
            b = 0.0
        betas.append(np.clip(b, -MAX_BETA_CAP, MAX_BETA_CAP))
    betas = np.array(betas)

    proj_daily   = X_z.values @ betas    # expected return per day
    proj_total   = proj_daily * horizon   # over H days

    # ── Assemble result ───────────────────────────────────────────────────────
    result = panel[["ticker"]].copy()
    result["regime"]               = regime
    result["factor_score"]         = scores
    result["proj_return_pct"]      = proj_total * 100.0
    result["regime_features_used"] = len([f for f in sig_feats if f in regime_feats])
    result["n_features"]           = len(sig_feats)
    result["latest_close"]         = panel.get("latest_close", np.nan)
    result["avg_volume_20"]        = panel.get("avg_volume_20", np.nan)

    result = result.sort_values("factor_score", ascending=False).reset_index(drop=True)
    result["rank"] = result.index + 1
    return result


# ─────────────────────────────────────────────────────────────────────────────
# Public entry point
# ─────────────────────────────────────────────────────────────────────────────

def run_profitability(
    n_stocks:  int   = N_STOCKS,
    horizon:   int   = PROJ_HORIZON,
    tstat_min: float = TSTAT_MIN,
    verbose:   bool  = True,
) -> tuple[pd.DataFrame, str]:
    """
    Full profitability scoring pipeline.

    Returns:
        scores  : DataFrame of all stocks ranked by factor_score
        regime  : current detected regime name
    """
    if verbose:
        print("Loading FM coefficients…")
    fm_coefs = load_fm_coefficients()

    if verbose:
        print("Detecting current market regime…")
    regime = detect_current_regime()
    exposure = REGIME_EXPOSURE.get(regime, 1.0)
    if verbose:
        print(f"  Current regime: {regime}  (target gross exposure: {exposure:.0%})")

    if verbose:
        print(f"Loading {n_stocks} stock universe…")
    panel = load_universe(n_stocks)
    if verbose:
        print(f"  Loaded {len(panel)} stocks\n")

    scores = compute_profitability_scores(
        panel, fm_coefs, regime, tstat_min=tstat_min, horizon=horizon
    )

    if verbose:
        print(f"Profitability scores ({regime}, horizon={horizon}d):")
        print(scores[["rank", "ticker", "factor_score", "proj_return_pct"]].head(20).to_string(index=False))

    return scores, regime
