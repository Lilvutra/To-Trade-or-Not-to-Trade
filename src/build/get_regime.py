"""
----------------
Regime detection + regime-conditional feature engineering for Vietnam stocks.

Design philosophy
-----------------
Feature quality depends heavily on which market regime is active.
A momentum feature (delta_dist, macd_hist_slope) that works well in a
trending market becomes noise in a choppy sideways market.
A panic/exhaustion signal only fires meaningfully in high-volatility regimes.

Vietnam-specific structural observations encoded here
------------------------------------------------------
  Obs 1 – Retail dominance (~85-90% of daily volume)
    Herding and overreaction are systematic. Retail investors pile in on
    momentum and panic-sell on bad news. Mean-reversion after extremes is
    predictable within 3-5 sessions.
    → retail_exhaustion, herd_momentum_10, conviction_close

  Obs 2 – Price limits (±7% HOSE) create queue dynamics
    When a stock hits the ceiling, unmatched buy orders spill into the next
    open → structural continuation effect. Limit-down traps sellers → forced
    supply at tomorrow's open.
    → limit_up_conviction, limit_down_conviction, limit_open_reversal

  Obs 3 – T+2 settlement creates predictable forced-selling windows
    Investors who bought a falling stock cannot sell for 2 sessions.
    A sharp drop 2 days ago predicts elevated selling today when combined
    with current elevated volume.
    → t2_forced_selling, t2_cascade

  Obs 4 – Foreign investors act as smart money vs retail flow
    Without order-book data we approximate: up-day + high volume = someone
    absorbing retail selling (institution/foreign accumulation). Down-day +
    high volume = distribution by holders (smart money exiting).
    → smart_money_up, smart_money_down, volume_price_divergence

  Obs 5 – ATC session distorts end-of-day prices
    Institutional bulk orders at close frequently move price away from
    continuous-trading price. Proxy: where close lands in day's range,
    combined with volume, reveals ATC distribution or accumulation.
    → intraday_distribution, range_expansion, conviction_close

  Obs 6 – Two-phase margin-call cascade
    Sharp drops trigger broker margin calls. Forced liquidation at T+1 open
    causes index to gap further. After 2+ consecutive extreme days the
    cascade typically exhausts.
    → margin_risk, margin_cascade_duration

Regimes
-------
  0  QUIET_BEAR    low vol + downtrend   → mean-reversion signals dominate
  1  PANIC_BEAR    high vol + downtrend  → exhaustion / capitulation signals
  2  QUIET_BULL    low vol + uptrend     → trend-following / accumulation
  3  VOLATILE_BULL high vol + uptrend    → momentum / breakout signals

Universal feature (all regimes)
--------------------------------
  conviction_close = close_position × log(1 + volume_spike)
    High  → buyers held their ground with volume conviction (bullish)
    Low   → sellers dominated despite volume (bearish / capitulation)
  This is the closest OHLCV approximation to order-flow data.

Usage
-----
1. Run detect_regime() to assign a regime label to each row using only past
   information.
2. Build regime-specific features by slicing the DataFrame by regime and applying the relevant feature functions from REGIME_FEATURES.
   
"""

from __future__ import annotations

import os
import numpy as np
import pandas as pd

from vnstock_trade.features.build_features import _rsi


def infer_market_index_from_filename(path: str) -> str:
    """Infer market index from files named like ticker-INDEX-History.csv."""
    name = os.path.basename(path)
    if name.endswith("-History.csv"):
        core = name[: -len("-History.csv")]
        parts = core.split("-")
        if len(parts) >= 2:
            return "-".join(parts[1:])
    return "VNINDEX"

EPS = 1e-8

# ─────────────────────────────────────────────────────────────────────────────
# Regime identifiers
# ─────────────────────────────────────────────────────────────────────────────

REGIME_ID = {
    "QUIET_BEAR":    0,
    "PANIC_BEAR":    1,
    "QUIET_BULL":    2,
    "VOLATILE_BULL": 3,
}

REGIME_NAME = {v: k for k, v in REGIME_ID.items()} # reverse mapping for convenience

# Feature columns active for each regime
# The training pipeline should slice df by regime, then use only these columns.
REGIME_FEATURES: dict[int, list[str]] = {
    REGIME_ID["QUIET_BEAR"]: [
        # Core 
        "delta_dist",               # +8.1: MA velocity — positive = bounce momentum building
        "gap_down",                 # +3.26: gap-down only; cascade continuation signal
        # Supporting (|t| < 2 but directionally sound)
        "vol_accel",                # −1.76: volume deceleration = selling pressure fading
        "dist_ma",                  # +1.49: price below MA5 → snap-back predictor
        "seller_exhaustion_fresh",  # +1.45: isolated spike without cascade → near-term bounce
        "oversold_stable",          # +0.79: oversold composite gated by panic/cascade risk
    ],
    REGIME_ID["PANIC_BEAR"]: [
        # Core — joint FM dominant signals in PB (|t| ≥ 3.0)
        "smart_money_up",           # +12.7: institutions buying into panic = strongest bottom signal
        "range_expansion_up",       # −2.2: wide up-candle in panic = exhaustion reversal
        "dist_ma",                  # −1.4: price far below MA = mean-reversion setup
        "gap_down",                 # -0.8 gap-down only: cascade severity marker (no gap-up noise in panic)
        # Supporting
        "delta_dist",               # +1.4: MA velocity — acceleration into panic trough
        "seller_exhaustion_fresh",  # +5.2: isolated spike without cascade → near-term bounce
        # Context (Vietnam structural)
        "limit_down_conviction",    # -1.7 limit-down + high vol = trapped-seller queue overhang
    ],
    REGIME_ID["QUIET_BULL"]: [
        # Core signals (|t| ≥ 2.5 in joint FM)
        "smart_money_up",           # +11.97: institutional absorption = trend health indicator
        "dist_ma",                  # −6.03: universal; extended above MA in quiet bull → mean-reversion
        "range_expansion_up",       # −4.45: FOMO overshoot in uptrend → next-day fade (universal)
        "zscore_return_neg",        # −3.28: extreme down-day in bull = overreaction → fast bounce
        # Supporting (1.5 ≤ |t| < 2.5)
        "delta_dist",               # +1.88: MA velocity; confirms accumulation momentum
        "macd_hist_slope",          # slope of MACD histogram; rising = momentum building in uptrend
        # Context
        "conviction_close",         # −0.86: negative role — weak close in uptrend = distribution warning
    ],
    REGIME_ID["VOLATILE_BULL"]: [
        # Core signals (|t| ≥ 2.5 in joint FM)
        "smart_money_up",           # +10.81: institutional participation confirms momentum durability
        "dist_ma",                  # −5.54: universal reversion; large MA-extension = fade even in VB
        "range_expansion_up",       # −5.01: exhaustion guard; FOMO overshoot → reversal (universal)
        "zscore_return_neg",        # −4.93: extreme neg-z even in VB = sharp reversal signal
        "delta_dist",               # +3.71: MA velocity separates real breakout from retail-only pump
        # Supporting (1.0 ≤ |t| < 2.0)
        "vol_accel",                # −1.32: volume deceleration on up-moves = fuel running out
        "limit_up_streak",          # −1.33: consecutive limit-up streak (note: negative = exhaustion)
        # Context
        "conviction_close",         # +0.69: strong close confirms buyers held ground
    ],
}


# ─────────────────────────────────────────────────────────────────────────────
# Regime report metadata
# ─────────────────────────────────────────────────────────────────────────────

REGIME_DESCRIPTIONS: dict[int, dict] = {
    REGIME_ID["QUIET_BEAR"]: {
        "name":        "QUIET_BEAR",
        "label":       "0 — Quiet Bear",
        "conditions":  "Low volatility + downtrend (or negative momentum in sideways band)",
        "dominant_force": (
            "Retail drift downward without large panic spikes. "
            "Price bleeds slowly below its moving average as sellers outnumber buyers on low conviction."
        ),
        "signal_type": "Mean-reversion",
        "trading_edge": (
            "Oversold extremes are the opportunity. "
            "When zscore_return_neg and dist_ma reach extreme lows and seller_exhaustion_fresh spikes without cascade, "
            "a short-term bounce is likely within 3–5 sessions (delta_dist turning positive is the early confirmation)."
        ),
        "feature_descriptions": {
            "delta_dist":               "Day-over-day change in distance from MA5. Turning positive = bounce momentum building before price confirms.",
            "gap_down":                 "Negative overnight gap only (clipped at 0). Cascade / forced-selling continuation signal; positive β = larger gap-down → bigger snap-back.",
            "vol_accel":                "Volume acceleration (vol / lagged vol − 1). Negative = selling pressure fading; key signal that the bleed is running out of fuel.",
            "dist_ma":                  "Distance of close from MA5. Strongly negative = price stretched below trend; snap-back predictor in quiet regimes.",
            "seller_exhaustion_fresh":  "(1 − close_pos) × vol_spike × (1 − cascade_norm). High isolated selling spike without a cascade context → near-term bounce.",
            "oversold_stable":          "Composite: oversold strength × (1 − cascade) × (1 − vol_spike regime) × (1 − high down_ratio). Quantifies how oversold the stock is while gating out panic-driven extremes.",
        },
    },
    REGIME_ID["PANIC_BEAR"]: {
        "name":        "PANIC_BEAR",
        "label":       "1 — Panic Bear",
        "conditions":  "High volatility + downtrend (or negative momentum in high-vol sideways band)",
        "dominant_force": (
            "Margin cascades, T+2 trapped sellers, and limit-down queue dynamics. "
            "Two-phase forced liquidation: broker margin calls at T+1 gap-down, "
            "then retail panic selling into the void."
        ),
        "signal_type": "Exhaustion / capitulation bottom detection",
        "trading_edge": (
            "Identify when the cascade has run its course. "
            "smart_money_up appearing alongside seller_exhaustion_fresh (isolated spike, no ongoing cascade) "
            "= high-probability capitulation bottom; range_expansion_up confirms institutions stepped in."
        ),
        "feature_descriptions": {
            "smart_money_up":          "Up day + high volume. Institutions / foreigners absorbing panic selling; strongest capitulation bottom signal.",
            "range_expansion_up":      "Wide up-candle (high daily range) in a panic session = exhaustion reversal; price accepted back above recent lows with volume.",
            "dist_ma":                 "Distance of close from MA5. Extreme negative = price far below trend; mean-reversion setup after cascade exhaustion.",
            "gap_down":                "Negative overnight gap only. In panic, measures cascade severity; further gap-down after extreme drops can signal proximity to capitulation.",
            "delta_dist":              "Day-over-day change in distance from MA5. Turning positive inside a panic = buyers beginning to absorb selling.",
            "seller_exhaustion_fresh": "(1 − close_pos) × vol_spike × (1 − cascade_norm). Isolated high-volume selling spike not embedded in an ongoing cascade → near-term bounce signal.",
            "limit_down_conviction":   "limit_down × vol_spike. Large queue of trapped sellers → trapped-seller overhang; continuation risk or capitulation bottom depending on direction of follow-through.",
        },
    },
    REGIME_ID["QUIET_BULL"]: {
        "name":        "QUIET_BULL",
        "label":       "2 — Quiet Bull",
        "conditions":  "Low volatility + uptrend (or positive momentum in low-vol sideways band)",
        "dominant_force": (
            "Foreign / institutional accumulation on dips. "
            "Price drifts steadily above MA5 on moderate volume with occasional pullbacks absorbed quickly."
        ),
        "signal_type": "Trend-following / early acceleration detection",
        "trading_edge": (
            "Detect early signs of trend acceleration before price breaks away. "
            "delta_dist turning positive + smart_money_up on pullback days "
            "= momentum inflection before it is visible in price (Obs 4 — smart money accumulation)."
        ),
        "feature_descriptions": {
            "smart_money_up":      "Up day + high volume. Institutional accumulation on dips; strongest signal of trend health in quiet bull.",
            "dist_ma":             "Distance of close from MA5. Negative (pullback to MA) = entry point; positive (stretched) = reduce exposure.",
            "range_expansion_up":  "Wide up-candle relative to 20-day avg range. Confirms genuine institutional buying, not low-volume drift.",
            "zscore_return_neg":   "Negative-tail z-score of return. Small pullback days (mildly negative) precede best continuation entries; extreme lows are rare here.",
            "delta_dist":          "Day-over-day change in distance from MA5. Positive = price accelerating away from trend = early momentum inflection.",
            "macd_hist_slope":     "Day-over-day change in MACD histogram. Rising histogram = bullish momentum building; confirms trend before price crossover is visible.",
            "conviction_close":    "close_pos × log(1 + vol_spike). High = buyers held the close with volume; confirms accumulation. Declining = distribution beginning.",
        },
    },
    REGIME_ID["VOLATILE_BULL"]: {
        "name":        "VOLATILE_BULL",
        "label":       "3 — Volatile Bull",
        "conditions":  "High volatility + uptrend (or positive momentum in high-vol sideways band)",
        "dominant_force": (
            "Retail FOMO, limit-up queue momentum, and breakout continuation. "
            "Vietnam price limits create structural momentum: unmatched limit-up buy orders "
            "spill into the next open, compounding the move."
        ),
        "signal_type": "Momentum / breakout continuation",
        "trading_edge": (
            "Ride the burst while it has volume fuel; exit when FOMO exhaustion signals appear. "
            "limit_up_streak + vol_accel + smart_money_up = continuation confirmed. "
            "conviction_close dropping while price still up = distribution beginning (Obs 1 + 2)."
        ),
        "feature_descriptions": {
            "smart_money_up":    "Up day + high volume. Institutional participation distinguishes genuine breakouts from retail-only pumps; persistent = continuation.",
            "dist_ma":           "Distance of close from MA5. Moderately positive = healthy; extreme positive = overbought, fade risk rising.",
            "range_expansion_up":"Wide up-candle vs 20-day avg range. Confirms institutional conviction; combined with vol_accel = strong continuation signal.",
            "zscore_return_neg": "Negative-tail z-score. Small pullback after breakout session; negative z within VB = buy-the-dip, not reversal.",
            "delta_dist":        "Day-over-day velocity vs MA5. Very high = price stretching fast = breakout fuel remaining; turning flat = nearing exhaustion.",
            "vol_accel":         "Volume acceleration. Fuel gauge for the burst: rising vol on up-days = continuation; fading during limit-up streak = top approaching.",
            "limit_up_streak":   "Consecutive limit-up days. Vietnam FOMO burst creates structural momentum; unmatched buy queue spills to next open; longest streaks precede sharp reversals.",
            "conviction_close":  "close_pos × log(1 + vol_spike). High = buyers held their ground with volume = continuation. Sudden drop = distribution has begun.",
        },
    },
}


def regime_feature_report(regime_id: int | None = None) -> str:
    """
    Return a formatted plain-text report describing one or all regimes.

    Parameters
    ----------
    regime_id : int or None
        If given, report only that regime (0–3). If None, report all four.

    Returns
    -------
    str
        Multi-line report suitable for printing or embedding in a notebook.

    Example
    -------
    >>> print(regime_feature_report())          # all regimes
    >>> print(regime_feature_report(1))         # PANIC_BEAR only
    """
    ids = [regime_id] if regime_id is not None else sorted(REGIME_DESCRIPTIONS)
    lines: list[str] = []

    for rid in ids:
        d = REGIME_DESCRIPTIONS[rid]
        sep = "═" * 72
        lines += [
            "",
            sep,
            f"  {d['label']}",
            sep,
            f"  Conditions    : {d['conditions']}",
            f"  Dominant force: {d['dominant_force']}",
            f"  Signal type   : {d['signal_type']}",
            f"  Trading edge  : {d['trading_edge']}",
            "",
            "  Features",
            "  " + "─" * 68,
        ]
        for feat, desc in d["feature_descriptions"].items():
            lines.append(f"  {feat:<28} {desc}")
        lines.append("")

    return "\n".join(lines)


# ─────────────────────────────────────────────────────────────────────────────
# 1. REGIME DETECTION (no look-ahead)
# ─────────────────────────────────────────────────────────────────────────────

def detect_regime(
    g: pd.DataFrame,
    vol_window:     int   = 20,   # short-term volatility window
    vol_baseline:   int   = 252,  # long-run vol anchor (1 year)
    trend_window:   int   = 20,   # up-day fraction window
    bull_threshold: float = 0.55, # lowered from 0.60 — less strict uptrend requirement
    bear_threshold: float = 0.45, # raised  from 0.40 — less strict downtrend requirement
    mom_window:     int   = 20,   # momentum window for resolving the sideways zone
    close_col: str = "Close",
    time_col:  str = "TradingDate",
) -> pd.Series:
    """
    Assign a regime label to every row using only past information.

    Three axes (all strictly historical, no look-ahead):

    1. Volatility axis
         log_ret     = diff(log(close))
         current_vol = rolling(vol_window).std(log_ret)
         vol_median  = rolling(vol_baseline).median(current_vol)
         is_high_vol = current_vol > vol_median
       Compares current vol to its own long-run median so the threshold
       adapts to each stock's baseline volatility level.

    2. Trend axis  (up-day fraction)
         fraction_up  = rolling(trend_window).mean(close > close.shift(1))
         is_uptrend   = fraction_up > bull_threshold   (default 55%)
         is_downtrend = fraction_up < bear_threshold   (default 45%)
       The 45-55% band is "sideways" — many trending days in Vietnam land
       here because drift is smooth rather than monotone.

    3. Momentum axis  — resolves the sideways zone
         cum_ret = close / close.shift(mom_window) - 1
         is_up_momentum   = cum_ret > 0
         is_down_momentum = cum_ret < 0
       A day in the sideways band that has a positive cumulative return over
       the last mom_window days is trending UP, just not sharply enough to
       clear the fraction_up threshold.  Without this axis, all sideways
       days default to QUIET_BEAR even when price has risen 3-4% — the
       root cause of the previous 76% QUIET_BEAR over-classification.

    Regime matrix
    ─────────────────────────────────────────────────────
      is_uptrend   & is_high_vol              → VOLATILE_BULL (3)
      is_uptrend   & ~is_high_vol             → QUIET_BULL    (2)
      is_downtrend & is_high_vol              → PANIC_BEAR    (1)
      is_downtrend & ~is_high_vol             → QUIET_BEAR    (0)
      sideways     & is_high_vol  & up_mom    → VOLATILE_BULL (3)  ← new
      sideways     & ~is_high_vol & up_mom    → QUIET_BULL    (2)  ← new
      sideways     & is_high_vol  & down_mom  → PANIC_BEAR    (1)  ← new
      sideways     & ~is_high_vol & down_mom  → QUIET_BEAR    (0)  ← new
    ─────────────────────────────────────────────────────
    """
    g     = g.sort_values(time_col).copy()
    close = g[close_col]

    # ── Volatility axis ───────────────────────────────────────────────────────
    # Log returns are more stable than simple returns for vol estimation —
    # large daily moves (Vietnam limit-hits) are dampened by the log transform.
    log_ret     = np.log(close).diff()
    current_vol = log_ret.rolling(vol_window).std()
    vol_median  = current_vol.rolling(vol_baseline).median()
    vol_median  = vol_median.fillna(current_vol.expanding().median())
    is_high_vol = current_vol > vol_median

    # ── Trend axis (up-day fraction) ──────────────────────────────────────────
    daily_up    = (close.diff() > 0).astype(int)
    fraction_up = daily_up.rolling(trend_window).mean()
    fraction_up = fraction_up.fillna(daily_up.expanding().mean())

    is_uptrend   = fraction_up > bull_threshold
    is_downtrend = fraction_up < bear_threshold
    is_sideways  = ~is_uptrend & ~is_downtrend

    # ── Momentum axis (resolves sideways zone) ────────────────────────────────
    # cum_ret at row t = close[t] / close[t - mom_window] - 1
    # Fully historical — no look-ahead.
    cum_ret        = close.pct_change(mom_window)
    cum_ret        = cum_ret.fillna(0)
    is_up_momentum = cum_ret >= 0   # positive return → upward drift

    # ── Regime matrix ─────────────────────────────────────────────────────────
    # Start with all QUIET_BEAR, then overwrite in priority order.
    regime = pd.Series(REGIME_ID["QUIET_BEAR"], index=g.index, name="regime")

    # Clear trends — same as before
    regime[is_uptrend   &  is_high_vol] = REGIME_ID["VOLATILE_BULL"]
    regime[is_uptrend   & ~is_high_vol] = REGIME_ID["QUIET_BULL"]
    regime[is_downtrend &  is_high_vol] = REGIME_ID["PANIC_BEAR"]
    regime[is_downtrend & ~is_high_vol] = REGIME_ID["QUIET_BEAR"]

    # Sideways zone — resolved by momentum direction
    regime[is_sideways &  is_high_vol &  is_up_momentum] = REGIME_ID["VOLATILE_BULL"]
    regime[is_sideways & ~is_high_vol &  is_up_momentum] = REGIME_ID["QUIET_BULL"]
    regime[is_sideways &  is_high_vol & ~is_up_momentum] = REGIME_ID["PANIC_BEAR"]
    regime[is_sideways & ~is_high_vol & ~is_up_momentum] = REGIME_ID["QUIET_BEAR"]

    return regime


# ─────────────────────────────────────────────────────────────────────────────
# 2. SHARED BASE (computed once, reused by all regime builders)
# ─────────────────────────────────────────────────────────────────────────────

def _build_shared_base(
    g: pd.DataFrame,
    close_col: str = "Close",
    open_col:  str = "Open",
    high_col:  str = "High",
    low_col:   str = "Low",
    vol_col:   str = "Volume",
    market_index: str = "VNINDEX",
) -> pd.DataFrame:
    """
    Compute every building block that more than one regime builder needs.
    Stored as "_"-prefixed internals; regime builders read and combine them.
    Two public features live here because they apply to ALL regimes:
      conviction_close  — universal buyer-conviction composite
      retail_exhaustion — universal seller-pressure composite

    Parameters
    ----------
    market_index : str
        Market index type to determine limit thresholds:
        - "VNINDEX" or "HOSE" -> 7%
        - "UpcomIndex"         -> 10%
        - "HNXIndex"          -> 5%
    """
    close  = g[close_col]
    open_  = g[open_col]
    high   = g[high_col]
    low    = g[low_col]
    volume = g[vol_col]

    index_key = str(market_index).strip().upper() if market_index else "VNINDEX"
    if index_key in {"UPCOMINDEX", "UPCOM", "UPCOM-INDEX"}:
        limit_threshold = 0.095
    elif index_key in {"HNXINDEX", "HNX", "HNX-INDEX"}:
        limit_threshold = 0.045
    else:
        limit_threshold = 0.065

    # ── Returns ──────────────────────────────────────────────────────────────
    g["_ret"]       = close.pct_change()
    g["_log_ret"]   = np.log(close).diff()
    g["_vol_chg"]   = volume.pct_change()
    g["_daily_ret"] = close / close.shift(1) - 1   # for limit detection

    # ── Price structure ───────────────────────────────────────────────────────
    g["_ma5"]        = close.rolling(5).mean()
    g["_ma20"]       = close.rolling(20).mean()
    g["_roll_mean5"] = g["_ma5"]                   # alias for clarity in z-score calc
    g["_roll_std5"]  = close.rolling(5).std()

    # ── Momentum indicators ───────────────────────────────────────────────────
    g["_rsi14"]      = _rsi(close, window=14)

    _ema12           = close.ewm(span=12, adjust=False).mean() # adjust span specific to vietnam observation
    _ema26           = close.ewm(span=26, adjust=False).mean()
    g["_macd_line"]  = _ema12 - _ema26
    g["_macd_hist"]  = g["_macd_line"] - g["_macd_line"].ewm(span=9, adjust=False).mean()

    # ── Volume baseline ───────────────────────────────────────────────────────
    g["_vol_mean20"] = volume.rolling(20).mean()
    g["_vol_spike"]  = volume / (g["_vol_mean20"] + EPS)

    # ── Volatility ────────────────────────────────────────────────────────────
    g["_vol20_std"]  = g["_ret"].rolling(20).std()
    g["_zscore_ret"] = g["_ret"] / (g["_vol20_std"] + EPS)

    # ── zscore_return signed splits ───────────────────────────────────────────
    # Splitting the direction-agnostic zscore into three components so linear
    # models can assign different coefficients to each tail and to magnitude.
    
    # zscore_return_neg: oversold detector — 0 on neutral/up days, negative on
    #   extreme down days. Larger magnitude = more extreme sell-off.
    #   Use in QB/PB: high absolute value predicts mean-reversion bounce.
    #
    # zscore_return_pos: overbought / breakout size — 0 on neutral/down days,
    #   positive on extreme up days. Larger = stronger breakout move.
    #   Use in VB: confirms breakout when large, warns of exhaustion when very large.
    
    # zscore_return_abs: event-day flag — magnitude regardless of direction.
    #   Use as a volatility / regime-transition flag across all regimes.
    g["zscore_return_neg"] = g["_zscore_ret"].clip(upper=0)
    g["zscore_return_pos"] = g["_zscore_ret"].clip(lower=0)
    g["zscore_return_abs"] = g["_zscore_ret"].abs()

    # ── Intraday position ─────────────────────────────────────────────────────
    # 1.0 = closed at session high, 0.0 = closed at session low
    g["_close_pos"]  = (close - low) / (high - low + EPS)

    # ── Overnight gap — split by direction (same tail-domination logic as zscore_return)
    # gap_down (<0): cascade/forced-selling continuation → bear regimes
    # gap_up   (>0): FOMO/positive-sentiment continuation → bull regimes
    # Using a single coefficient collapses two distinct economic signals into one.
    g["_gap"]        = (open_ - close.shift(1)) / (close.shift(1) + EPS)
    g["gap_down"]    = g["_gap"].clip(upper=0)   # negative-only: gap-down magnitude
    g["gap_up"]      = g["_gap"].clip(lower=0)   # positive-only: gap-up magnitude

    # ── RSI divergence (reused in quiet_bear + panic_bear) ───────────────────
    _price_up        = (close.diff() > 0).astype(int)
    _rsi_up          = (g["_rsi14"].diff() > 0).astype(int)
    g["_rsi_div"]    = _price_up - _rsi_up

    # ── Vietnam limit flags (market-aware threshold) ─────────────────────────
    _limit_up        = (g["_daily_ret"] >  limit_threshold).astype(int)
    _limit_down      = (g["_daily_ret"] < -limit_threshold).astype(int)
    g["_limit_up"]   = _limit_up
    g["_limit_down"] = _limit_down

    # Limit streaks (consecutive limit hits)
    g["_limit_up_streak"] = _limit_up * (
        _limit_up.groupby((_limit_up != _limit_up.shift()).cumsum()).cumcount() + 1
    )
    g["_limit_down_streak"] = _limit_down * (
        _limit_down.groupby((_limit_down != _limit_down.shift()).cumsum()).cumcount() + 1
    )

    # ── Daily range (Obs 5: range expansion on breakout / panic days) ─────────
    g["_daily_range"]   = (high - low) / (close.shift(1) + EPS) # range relative to previous close
    g["_range_mean20"]  = g["_daily_range"].rolling(20).mean() # average range as volatility baseline

    # ── Store raw OHLCV for builders ──────────────────────────────────────────
    g["_close"]  = close
    g["_open"]   = open_
    g["_high"]   = high
    g["_low"]    = low
    g["_volume"] = volume

    # ═════════════════════════════════════════════════════════════════════════
    # Universal public features — valid in every regime
    # ═════════════════════════════════════════════════════════════════════════

    # conviction_close (Obs 5)
    # Combines intraday close position with volume intensity.
    # High → buyers held their ground with conviction throughout the session.
    # Low  → sellers dominated; price closed near the low despite high volume.
    # This is the closest OHLCV proxy to order-flow data.
    g["conviction_close"] = g["_close_pos"] * np.log1p(g["_vol_spike"])

    # retail_exhaustion (Obs 1) — kept for backward compatibility
    # Original composite: closed near low + high volume, regardless of day direction.
    g["retail_exhaustion"] = (1.0 - g["_close_pos"]) * g["_vol_spike"]

    # ── Cascade duration (shared internal — reused by panic bear builder) ────────
    # How many of the last 3 sessions had returns below −2 std?
    # 0 = no recent extreme selling, 3 = full cascade (all 3 days extreme).
    _extreme_down        = (g["_zscore_ret"] < -2.0).astype(float)
    g["_cascade_dur"]    = _extreme_down.rolling(3, min_periods=1).sum()   # range [0, 3]
    _cascade_norm        = g["_cascade_dur"] / 3.0                          # normalised [0, 1]

    # ── range_expansion splits ────────────────────────────────────────────────
    # Base: daily high-low range relative to 20-day average range.
    _range_expansion = g["_daily_range"] / (g["_range_mean20"] + EPS)

    # Bearish expansion — wide range on a down day = panic / forced liquidation still active.
    # FM result: strong CONTINUATION signal in PANIC_BEAR (t = −3.52).
    # High value means selling is ongoing, not exhausted → expect more downside.
    # Use in PANIC_BEAR only.
    g["range_expansion_down"] = (g["_ret"] < 0).astype(float) * _range_expansion

    # Bullish expansion — wide range on an up day = retail FOMO overshoot → mean-reversion.
    # FM result: universal mean-reversion signal across ALL regimes (t ≈ −3.9, no sign flips).
    # Wide up-day range in Vietnam reflects retail pile-in, not institutional conviction.
    # Moved to shared base so all regime models can use it.
    g["range_expansion_up"] = (g["_ret"] > 0).astype(float) * _range_expansion

    # seller_exhaustion — down-day only split of retail_exhaustion
    # Fires on: down day + closed near low + high volume = panic selling into close.
    _down_day = (g["_ret"] < 0).astype(float)
    g["seller_exhaustion"] = _down_day * (1.0 - g["_close_pos"]) * g["_vol_spike"]

    # seller_exhaustion_fresh — isolated selling with no cascade context
    # cascade_norm ≈ 0  → weight ≈ 1.0  (this is an unusual, lone selling event)
    # cascade_norm ≈ 1  → weight ≈ 0.0  (already in a cascade, signal muted)
    # Interpretation: sudden heavy selling in a calm market → sellers exhausted → bounce.
    # Use in QUIET_BEAR where cascade rarely builds.
    g["seller_exhaustion_fresh"] = g["seller_exhaustion"] * (1.0 - _cascade_norm)

    # iss — Isolated Selling Shock
    # Combines four conditions in one scalar:
    #   zscore_return_neg : drop is statistically unusual (negative tail only)
    #   (1 - cascade_norm): NOT part of an ongoing crash (isolated event)
    #   vol_relative      : attention spike — retail is reacting
    #   (1 - close_pos)   : sellers were aggressive (closed near low)
    # Interpretation: "retail panic in a calm market → exhaustion bounce likely"
    # Designed for QUIET_BEAR where cascades are rare and single-day spikes revert.
    _vol_rel = g["_roll_std5"] / (g["_roll_mean5"] + EPS)
    g["iss"] = (
        g["zscore_return_neg"]
        * (1.0 - _cascade_norm)
        * _vol_rel
        * (1.0 - g["_close_pos"])
    )

    # seller_exhaustion_late — selling pressure on top of an existing cascade
    # cascade_norm ≈ 0  → weight ≈ 0.0  (no cascade yet, not relevant)
    # cascade_norm ≈ 1  → weight ≈ 1.0  (3 consecutive extreme days → forced liq. exhausted)
    # Interpretation: continued panic selling after a multi-day cascade → capitulation bottom.
    # Use in PANIC_BEAR where cascade duration is the distinguishing context.
    g["seller_exhaustion_late"] = g["seller_exhaustion"] * _cascade_norm

    # distribution_pressure — up-day only split of retail_exhaustion
    # Fires on: up day + closed near low + high volume = institutions sold into rally at ATC.
    # In bull regimes this warns of a distribution top forming.
    # Predicted direction: bearish (high value → next-day weakness likely).
    _up_day = (g["_ret"] > 0).astype(float)
    g["distribution_pressure"] = _up_day * (1.0 - g["_close_pos"]) * g["_vol_spike"]

    # ── oversold_stable — gating composite feature ────────────────────────────
    # Formula:
    #   oversold_strength = (1 - z_pct) + (1 - dist_pct)   [how extreme the dip]
    #   × (1 - cascade_norm)                                [NOT already in cascade]
    #   × (1 - vol_regime_spike)                            [volatility NOT elevated]
    #   × (1 - (down_ratio > 0.6))                          [market NOT broadly weak]
    #
    # High value → very oversold in a stable market → strong bounce candidate
    # Low value → either not oversold OR system is in stress (gated out)
    #
    # Note: vol_regime_spike uses RETURN volatility (sigma), not volume.
    # _vol_spike above is a volume ratio — different concept.
    _z_pct    = g["_zscore_ret"].rolling(252, min_periods=63).rank(pct=True)
    _dist_ma  = (close - g["_ma5"]) / (g["_ma5"] + EPS)
    _dist_pct = _dist_ma.rolling(252, min_periods=63).rank(pct=True)

    _oversold_strength = (1.0 - _z_pct) + (1.0 - _dist_pct)   # range [0, 2]; 2 = max oversold

    _vol_regime_spike  = (
        g["_vol20_std"] > g["_vol20_std"].rolling(60, min_periods=20).quantile(0.8)
    ).astype(float)

    _down_ratio = (g["_ret"] < 0).rolling(10, min_periods=5).mean()

    g["oversold_stable"] = (
        _oversold_strength
        * (1.0 - _cascade_norm)
        * (1.0 - _vol_regime_spike)
        * (1.0 - (_down_ratio > 0.6).astype(float))
    )

    return g


# ─────────────────────────────────────────────────────────────────────────────
# 3. REGIME-SPECIFIC FEATURE BUILDERS to 
# ─────────────────────────────────────────────────────────────────────────────

def _build_quiet_bear_features(g: pd.DataFrame) -> pd.DataFrame:
    """
    REGIME 0 — QUIET_BEAR
    Dominant force: retail drift downward without large panic spikes.
    Signal type: mean-reversion. Oversold extremes are the opportunity.

    New features (Obs 1, 3, 5):
      retail_exhaustion   — already in shared base
      herd_momentum_10    — 10-day up-day fraction: measures bearish consensus
      t2_forced_selling   — T+2 settlement pressure proxy
      intraday_distribution — ATC selling into weak rally
    """
    close = g["_close"]

    # ── Core mean-reversion signals ───────────────────────────────────────────
    g["dist_ma"]        = (close - g["_ma5"]) / (g["_ma5"] + EPS)
    g["rsi_divergence"] = g["_rsi_div"]
    g["close_position"] = g["_close_pos"]
    g["gap"]            = g["_gap"]        # kept for backward compat; prefer splits below
    g["gap_down"]       = g["gap_down"]    # negative-only: cascade / forced-selling signal
    g["vol_relative"]   = g["_roll_std5"] / (g["_roll_mean5"] + EPS)

    # mean-reversion signal, since we already detect regime
    # for quiet bear, if oversold and exhaustion & bounce -> buy

   
    
    
    # ── Obs 1: herd momentum ──────────────────────────────────────────────────
    # Fraction of up-days in last 10 sessions.
    # Low value = bearish consensus building (retail all selling together).
    # Can precede a mean-reversion bounce when extreme.
    #daily_up               = (close.diff() > 0).astype(int)
    #g["herd_momentum_10"]  = daily_up.rolling(10).mean()

    # ── Obs 3: T+2 forced selling ─────────────────────────────────────────────
    # A sharp drop 2 sessions ago means those buyers' shares are settling today.
    # They are forced sellers. When combined with current elevated volume,
    # this predicts continuation of the decline.
    #sharp_drop_2d         = (g["_ret"].shift(2) < -0.04).astype(int)
    #g["t2_forced_selling"] = sharp_drop_2d * g["_vol_spike"]
    
    

    # ── Obs 5: ATC distribution ───────────────────────────────────────────────
    # Up day in continuous trading, but institutions dumped at ATC close.
    # Proxy: daily return > 0 but close landed near the LOW of the range.
    g["intraday_distribution"] = (g["_ret"] > 0).astype(int) * (1.0 - g["_close_pos"])

    # conviction_close and retail_exhaustion come from shared base
   
    
    
    

    return g


def _build_panic_bear_features(g: pd.DataFrame) -> pd.DataFrame:
    """
    REGIME 1 — PANIC_BEAR
    Dominant force: margin cascades, T+2 trapped sellers, limit-down queues.
    Signal type: exhaustion / capitulation bottom detection.

    New features (Obs 2, 3, 4, 5, 6):
      limit_down_conviction  — trapped-seller queue with conviction
      t2_cascade             — 3-day cumulative settlement pressure
      smart_money_down       — distribution by institutions / foreigners
      range_expansion        — panic days have abnormally wide ranges
      margin_cascade_duration— how long has the extreme selling lasted?
    """
    close = g["_close"]

    # ── Core exhaustion signals ───────────────────────────────────────────────
    g["volume_spike"]       = g["_vol_spike"]
    g["panic"]              = ((g["_ret"] < 0) & (g["_vol_spike"] > 1.5)).astype(int)
    g["limit_down"]         = g["_limit_down"]
    g["limit_down_streak"]  = g["_limit_down_streak"]
    g["down_vol_pressure"]  = (g["_log_ret"] < 0).astype(int) * g["_vol_chg"]
    g["rsi_divergence"]     = g["_rsi_div"]
    g["gap"]                = g["_gap"]

    # ── Obs 2: limit-down conviction ──────────────────────────────────────────
    # Hit the floor AND traded high volume = enormous trapped-seller queue
    # that will overhang the next session open → high probability of continuation.
    # When this signal appears after 2+ limit-down days it can also mark a
    # capitulation bottom if accompanied by rsi_divergence.
    g["limit_down_conviction"] = g["_limit_down"] * g["_vol_spike"]

    # ── Obs 3: T+2 settlement cascade ────────────────────────────────────────
    # Three consecutive cohorts of buyers are all stuck.
    # 3-day cumulative drop > 8% + today's elevated volume = all cohorts panicking.
    cum_drop_3d        = close / close.shift(3) - 1
    g["t2_cascade"]    = (cum_drop_3d < -0.08).astype(int) * g["_vol_spike"]

    # ── Obs 4: smart money distribution ──────────────────────────────────────
    # Down day + elevated volume = someone large is selling.
    # In a panic regime this is likely institutions/foreigners exiting into retail panic.
    g["smart_money_down"] = (g["_ret"] < 0).astype(int) * g["_vol_spike"]

    # ── Obs 6: margin cascade duration ───────────────────────────────────────
    # Reuse _cascade_dur computed in shared base (same formula, avoids duplication).
    # Value of 3 = forced liquidation likely exhausted; bounce imminent.
    g["margin_cascade_duration"] = g["_cascade_dur"]

    # conviction_close and retail_exhaustion come from shared base

    return g


def _build_quiet_bull_features(g: pd.DataFrame) -> pd.DataFrame:
    """
    REGIME 2 — QUIET_BULL
    Dominant force: foreign / institutional accumulation on dips.
    Signal type: trend-following. Detect early signs of acceleration.

    New features (Obs 2, 4, 5):
      smart_money_up         — up-day + volume: foreign buying into dip
      volume_price_divergence— price up but volume fading → rally losing fuel
      intraday_distribution  — up day but closed low → ATC institutional selling
      limit_open_reversal    — hit limit-up yesterday, sold off today → distribution top
    """
    close = g["_close"]

    # ── Core trend-following signals ──────────────────────────────────────────
    dist                     = (close - g["_ma5"]) / (g["_ma5"] + EPS)
    g["delta_dist"]          = dist.diff()
    g["macd_hist_slope"]     = g["_macd_hist"].diff()
    g["delta_macd_combined"] = g["delta_dist"] * g["macd_hist_slope"]
    g["mom_5"]               = close / close.shift(5) - 1
    g["vol_accel"]           = g["_vol_chg"].diff()
    g["rsi_14"]              = g["_rsi14"]

    # ── Obs 4: smart money accumulation ──────────────────────────────────────
    # Up day + elevated volume = someone large is absorbing retail selling.
    # In a quiet bull regime this is the key institutional accumulation signal.
    # Persistent smart_money_up days = distribution phase has not yet begun.
    g["smart_money_up"] = (g["_ret"] > 0).astype(int) * g["_vol_spike"]

    # ── Obs 4: volume/price divergence ───────────────────────────────────────
    # Price making new highs but on declining volume = rally losing institutional support.
    # A warning signal to reduce exposure before the trend reverses.
    g["volume_price_divergence"] = (g["_ret"] > 0).astype(int) * (-g["_vol_chg"])

    # ── Obs 5: intraday distribution ──────────────────────────────────────────
    # Up day in price but close landed near the session low = institutions used
    # the rally to sell (ATC distribution). Repeated occurrences = topping pattern.
    g["intraday_distribution"] = (g["_ret"] > 0).astype(int) * (1.0 - g["_close_pos"])

    # ── Obs 2: limit-open reversal ───────────────────────────────────────────
    # Yesterday the stock hit limit-up (unmatched buy queue).
    # Today it closed near the LOW = the queued buyers exhausted demand immediately
    # at open and sellers took over → strong distribution signal, potential top.
    prev_limit_up              = g["_limit_up"].shift(1).fillna(0)
    g["limit_open_reversal"]   = prev_limit_up * (1.0 - g["_close_pos"])

    # conviction_close and retail_exhaustion come from shared base

    return g


def _build_volatile_bull_features(g: pd.DataFrame) -> pd.DataFrame:
    """
    REGIME 3 — VOLATILE_BULL
    Dominant force: retail FOMO, limit-up queue momentum, breakout continuation.
    Signal type: momentum. Ride the burst while it has volume fuel.

    New features (Obs 1, 2, 4, 5):
      limit_up_conviction — limit-up + volume: unmatched queue is large → continuation
      range_expansion     — breakout days have wide ranges vs recent average
      smart_money_up      — institutional buying confirms the breakout
      retail_exhaustion   — (reused) high value here = FOMO top forming
    """
    close = g["_close"]

    # ── Core momentum / breakout signals ─────────────────────────────────────
    dist                 = (close - g["_ma5"]) / (g["_ma5"] + EPS)
    g["delta_dist"]      = dist.diff()
    g["macd_hist_slope"] = g["_macd_hist"].diff()
    g["gap"]             = g["_gap"]        # kept for backward compat
    g["gap_up"]          = g["gap_up"]      # positive-only: FOMO / bullish sentiment gap
    g["vol_accel"]       = g["_vol_chg"].diff()
    g["z_window"]        = (close - g["_roll_mean5"]) / (g["_roll_std5"] + EPS)
    g["limit_up"]        = g["_limit_up"]
    g["limit_up_streak"] = g["_limit_up_streak"]
    g["zscore_return"]   = g["_zscore_ret"]

    # ── Obs 2: limit-up with conviction volume ────────────────────────────────
    # Hit the ceiling AND high volume = enormous unmatched-buyer queue carried
    # into the next open. Strong statistical continuation effect in Vietnam.
    # The larger the volume during a limit-up day, the more trapped buyers exist
    # who cannot exit — they will push price higher at the next open.
    g["limit_up_conviction"] = g["_limit_up"] * g["_vol_spike"]


    # ── Obs 4: smart money buying ─────────────────────────────────────────────
    # Up day + elevated volume = institutions/foreigners participating in the move.
    # In a volatile bull this distinguishes genuine breakouts from retail-only pumps.
    g["smart_money_up"] = (g["_ret"] > 0).astype(int) * g["_vol_spike"]

    # conviction_close and retail_exhaustion come from shared base
    # Note on retail_exhaustion in this regime:
    # A HIGH value (closed near the low with high volume) signals that retail
    # FOMO buyers got trapped as institutions distributed into the spike → top forming.
    # The model should learn that high retail_exhaustion here is a negative predictor.

    return g


# Map regime ID → builder function
_REGIME_BUILDERS = {
    REGIME_ID["QUIET_BEAR"]:    _build_quiet_bear_features,
    REGIME_ID["PANIC_BEAR"]:    _build_panic_bear_features,
    REGIME_ID["QUIET_BULL"]:    _build_quiet_bull_features,
    REGIME_ID["VOLATILE_BULL"]: _build_volatile_bull_features,
}


# ─────────────────────────────────────────────────────────────────────────────
# 4. UNIFIED ENTRY POINT
# ─────────────────────────────────────────────────────────────────────────────

def build_conditional_features(
    df: pd.DataFrame,
    close_col:  str = "Close",
    open_col:   str = "Open",
    high_col:   str = "High",
    low_col:    str = "Low",
    vol_col:    str = "Volume",
    time_col:   str = "TradingDate",
    symbol_col: str | None = None,
    market_index: str | None = None,
) -> pd.DataFrame:
    """
    Main pipeline entry point with market-aware limit handling.

    Steps
    -----
    1. Detect regime per bar — no look-ahead, uses only past data. 
    2. Build shared base (OHLCV building blocks + universal features).
    3. Apply all four regime-specific builders to every row. 
       Each builder fills only the columns relevant to that regime;
       the training pipeline then slices by regime and uses REGIME_FEATURES.
    4. Drop "_"-prefixed internal columns.
    5. Return DataFrame with regime + regime_name + all feature columns.

    Parameters
    ----------
    symbol_col : if provided, all rolling windows are computed per-symbol
                 to prevent cross-contamination between tickers.
    market_index : Optional market index name used to select limit thresholds.
                   If not provided, defaults to VNINDEX.
    """
    df = df.copy()
    df[time_col] = pd.to_datetime(df[time_col])
    df["_market_index"] = market_index or "VNINDEX"

    def _process(g: pd.DataFrame) -> pd.DataFrame:
        g = g.sort_values(time_col).copy()

        # Step 1 — regime
        g["regime"]      = detect_regime(g, close_col=close_col, time_col=time_col)
        g["regime_name"] = g["regime"].map(REGIME_NAME)

        # Step 2 — shared base (also builds conviction_close + retail_exhaustion)
        g = _build_shared_base(
            g,
            close_col=close_col,
            open_col=open_col,
            high_col=high_col,
            low_col=low_col,
            vol_col=vol_col,
            market_index=g["_market_index"].iloc[0] if "_market_index" in g.columns else "VNINDEX",
        )

        # Step 3 — all regime-specific feature sets applied to all rows, we will slice by regime later in the training pipeline
        for builder in _REGIME_BUILDERS.values():
            g = builder(g)

        return g

    if symbol_col and symbol_col in df.columns:
        out = df.groupby(symbol_col, group_keys=False).apply(_process)
    else:
        out = _process(df)

    out = out.reset_index(drop=True)

    # Drop internal building-block columns
    internal = [c for c in out.columns if c.startswith("_")]
    out = out.drop(columns=internal)

    return out


# ─────────────────────────────────────────────────────────────────────────────
# 5. REGIME-AWARE TRAIN / TEST SPLITTER
# ─────────────────────────────────────────────────────────────────────────────

def get_regime_split(
    df: pd.DataFrame,
    regime_id: int,
    time_col: str = "TradingDate",
    train_frac: float = 0.70,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Filter df to a single regime and return a time-based train/test split.
    Used by the training pipeline to train one model per regime.
    """
    subset     = df[df["regime"] == regime_id].sort_values(time_col)
    split_date = subset[time_col].quantile(train_frac)
    return (
        subset[subset[time_col] <= split_date],
        subset[subset[time_col] >  split_date],
    )

# ─────────────────────────────────────────────────────────────────────────────
# 6. DIAGNOSTICS
# ─────────────────────────────────────────────────────────────────────────────

def regime_summary(df: pd.DataFrame, time_col: str = "TradingDate") -> pd.DataFrame:
    """Print and return regime distribution and date ranges."""
    summary = (
        df.groupby("regime")
        .agg(
            name=("regime_name", "first"),
            count=("regime", "count"),
            first_date=(time_col, "min"),
            last_date=(time_col, "max"),
        )
        .assign(pct=lambda x: (x["count"] / x["count"].sum() * 100).round(1))
    )
    print("\nRegime distribution:")
    print(summary.to_string())
    return summary


# ─────────────────────────────────────────────────────────────────────────────
# CLI smoke test
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    data_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "data", "data-vn-20230228", "stock-historical-data"))
    sample_file = os.path.join(data_dir, "VCB-VNINDEX-History.csv")

    print(f"Loading {sample_file}")
    raw = pd.read_csv(sample_file)
    raw["TradingDate"] = pd.to_datetime(raw["TradingDate"])

    market_idx = "VNINDEX"
    out = build_conditional_features(raw, market_index=market_idx)
    summary = regime_summary(out)

    print(f"\nOutput shape : {out.shape}")
    print(f"Columns      : {list(out.columns)}")

    print("\nFeature coverage per regime:")
    for rid, feats in REGIME_FEATURES.items():
        name      = REGIME_NAME[rid]
        available = [f for f in feats if f in out.columns]
        missing   = [f for f in feats if f not in out.columns]
        print(f"  {name} ({rid}): {len(available)} features")
        if missing:
            print(f"    MISSING: {missing}")

    print("\nSample rows per regime (regime-specific features only):")
    for rid in REGIME_ID.values():
        sample = out[out["regime"] == rid].head(3) # show only first 2 rows per regime for brevity
        if len(sample):
            feats = ["conviction_close", "retail_exhaustion"] + REGIME_FEATURES[rid]
            cols  = ["TradingDate", "regime_name"] + [f for f in feats if f in sample.columns] 
            # show date start and date end of the regime
            first_date = sample["TradingDate"].min()
            last_date = sample["TradingDate"].max()
            print(f"\n  {REGIME_NAME[rid]}:")
            print(f"    Date range: {first_date} to {last_date}")
            print(sample[cols].to_string(index=False))
