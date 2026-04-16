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
    from vnstock_trade.features.detect_regime import build_conditional_features

    df = pd.read_csv("data/VNM-VNINDEX-History.csv")
    df["TradingDate"] = pd.to_datetime(df["TradingDate"])
    result = build_conditional_features(df)
    # result has columns: regime, regime_name, conviction_close, + regime features
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

# Feature columns active for each regime.
# The training pipeline should slice df by regime, then use only these columns.
REGIME_FEATURES: dict[int, list[str]] = {
    REGIME_ID["QUIET_BEAR"]: [
        # Core mean-reversion signals
        "zscore_return",          # how extreme is today's return vs recent vol
        "rsi_divergence",         # price ↑ but RSI ↓ (or vice versa) → reversal setup
        "dist_ma",                # distance from MA5 (oversold when very negative)
        "close_position",         # where close landed in today's high-low range
        "gap",                    # overnight gap: sentiment before MA reacts
        "vol_relative",           # relative vol spike even in quiet regime = warning
        # Obs 1: retail exhaustion
        "retail_exhaustion",      # closed near low + high volume = retail panic selling
        "herd_momentum_10",       # 10-day fraction of up-days: low = bearish consensus building
        # Obs 3: T+2 settlement
        "t2_forced_selling",      # sharp drop 2 days ago × current volume spike
        # Obs 5: ATC distortion
        "intraday_distribution",  # up day but closed near low = institutional ATC selling
        # Universal
        "conviction_close",       # low value confirms weakness; reversal if it turns up
    ],
    REGIME_ID["PANIC_BEAR"]: [
        # Core exhaustion / capitulation signals
        "zscore_return",          # magnitude of the panic move
        "volume_spike",           # abnormal volume → exhaustion signal
        "panic",                  # composite: down-day + volume spike > 1.5×
        "limit_down_streak",      # consecutive circuit-breaker down days (Vietnam-specific)
        "down_vol_pressure",      # selling pressure on down days
        "rsi_divergence",         # RSI makes higher low while price lower low → bullish divergence
        "gap",                    # large negative gap = continuation; small = potential exhaustion
        # Obs 2: limit queue dynamics
        "limit_down_conviction",  # limit-down + high volume = strong trapped-seller queue
        # Obs 3: T+2 settlement cascade
        "t2_cascade",             # 3-day cumulative drop > 8% × volume: all cohorts stuck
        # Obs 4: smart money proxy
        "smart_money_down",       # down-day + high volume = distribution by institutions
        # Obs 5: range expansion
        "range_expansion",        # panic days have wider ranges than normal
        # Obs 6: margin cascade
        "margin_cascade_duration",# how many of last 3 days were extreme-down (>2 std)
        # Obs 1: retail exhaustion as capitulation detector
        "retail_exhaustion",      # high value = sellers dominated → potential exhaustion bottom
        # Universal
        "conviction_close",       # very low = max fear; turning up = potential reversal
    ],
    REGIME_ID["QUIET_BULL"]: [
        # Core trend-following / accumulation signals
        "delta_dist",             # velocity of price vs MA5 (fires before crossover)
        "macd_hist_slope",        # MACD histogram turning point detector
        "delta_macd_combined",    # delta_dist × macd_hist_slope: confluence of both
        "mom_5",                  # 5-day momentum
        "vol_accel",              # volume accelerating: quiet accumulation before breakout
        "rsi_14",                 # RSI confirms trend strength in steady uptrend
        # Obs 4: foreign / smart money accumulation
        "smart_money_up",         # up-day + high volume = institutions absorbing retail selling
        "volume_price_divergence",# price ↑ but volume fading → rally losing fuel, likely fade
        # Obs 5: ATC distribution detection
        "intraday_distribution",  # up day + closed near low = institutional ATC selling into rally
        # Obs 2: limit-open reversal (caution signal)
        "limit_open_reversal",    # hit limit-up yesterday + closed near low today = distribution
        # Universal
        "conviction_close",       # high = accumulation with volume; rising trend is healthy
    ],
    REGIME_ID["VOLATILE_BULL"]: [
        # Core momentum / breakout signals
        "delta_dist",             # price vs MA velocity
        "macd_hist_slope",        # momentum inflection detector
        "gap",                    # overnight gap on breakout days: very informative
        "vol_accel",              # volume surge: fuel for continuation
        "z_window",               # 5-day z-score: overbought filter (when very high = caution)
        "limit_up_streak",        # consecutive limit-up days: Vietnam FOMO momentum burst
        "zscore_return",          # size of today's move vs recent vol
        # Obs 2: limit queue dynamics
        "limit_up_conviction",    # limit-up + high volume = strong unmatched-buyer queue
        # Obs 5: range expansion
        "range_expansion",        # breakout days have wider ranges than recent average
        # Obs 4: smart money
        "smart_money_up",         # up-day + high volume = institutional buying into breakout
        # Obs 1: retail exhaustion as OVERBOUGHT filter
        "retail_exhaustion",      # closed near low + vol spike = distribution top forming
        # Universal
        "conviction_close",       # high = buyers held ground: continuation likely
    ],
}


# ─────────────────────────────────────────────────────────────────────────────
# 1. REGIME DETECTION (no look-ahead)
# ─────────────────────────────────────────────────────────────────────────────

def detect_regime(
    g: pd.DataFrame,
    vol_window: int = 20, # short-term volatility window for regime classification (e.g. 20-day)
    vol_baseline: int = 252, # long-term volatility baseline for regime classification (e.g. 1-year trading days)
    trend_window: int = 20, # window for calculating up-day fraction to determine trend (e.g. 20-day)
    bull_threshold: float = 0.60,  # fraction of up-days above which we classify as uptrend (e.g. >60% up-days)
    bear_threshold: float = 0.40, # fraction of up-days below which we classify as downtrend (e.g. <40% up-days)
    close_col: str = "Close", 
    time_col: str = "TradingDate",
) -> pd.Series:
    """
    Assign a regime label to every row using only past information.

    Two independent axes — both use only data available at each bar:

    Volatility axis
      current_vol = rolling(vol_window).std(log_ret)
      vol_median  = rolling(vol_baseline).median(current_vol)  ← long-run anchor
      is_high_vol = current_vol > vol_median

    Trend axis
      fraction_up  = rolling(trend_window).mean(daily_up_flag)
      is_uptrend   = fraction_up > bull_threshold   (default >60% up-days)
      is_downtrend = fraction_up < bear_threshold   (default <40% up-days)
      sideways = between thresholds → classified by vol only → QUIET_BEAR

    Regime matrix
      is_uptrend  &  is_high_vol  →  VOLATILE_BULL (3)
      is_uptrend  & ~is_high_vol  →  QUIET_BULL    (2)
      is_downtrend & is_high_vol  →  PANIC_BEAR    (1)
      otherwise                   →  QUIET_BEAR    (0)
    """
    g = g.sort_values(time_col).copy()
    close = g[close_col]

    # Volatility axis
    log_ret     = np.log(close).diff() # log return is more stable for volatility estimation than simple returns, especially in a market like Vietnam with occasional large jumps. The rolling std of log returns over vol_window gives us a measure of current volatility. We then compare this to a long-term median to determine if we're in a high-volatility regime. 
    current_vol = log_ret.rolling(vol_window).std()
    vol_median  = current_vol.rolling(vol_baseline).median()
    vol_median  = vol_median.fillna(current_vol.expanding().median())
    is_high_vol = current_vol > vol_median

    # Trend axis
    daily_up    = (close.diff() > 0).astype(int)
    fraction_up = daily_up.rolling(trend_window).mean()
    fraction_up = fraction_up.fillna(daily_up.expanding().mean())

    is_uptrend   = fraction_up > bull_threshold
    is_downtrend = fraction_up < bear_threshold

    regime = pd.Series(REGIME_ID["QUIET_BEAR"], index=g.index, name="regime")
    regime[is_uptrend   &  is_high_vol] = REGIME_ID["VOLATILE_BULL"]
    regime[is_uptrend   & ~is_high_vol] = REGIME_ID["QUIET_BULL"]
    regime[is_downtrend &  is_high_vol] = REGIME_ID["PANIC_BEAR"]

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

    _ema12           = close.ewm(span=12, adjust=False).mean() # 
    _ema26           = close.ewm(span=26, adjust=False).mean()
    g["_macd_line"]  = _ema12 - _ema26
    g["_macd_hist"]  = g["_macd_line"] - g["_macd_line"].ewm(span=9, adjust=False).mean()

    # ── Volume baseline ───────────────────────────────────────────────────────
    g["_vol_mean20"] = volume.rolling(20).mean()
    g["_vol_spike"]  = volume / (g["_vol_mean20"] + EPS)

    # ── Volatility ────────────────────────────────────────────────────────────
    g["_vol20_std"]  = g["_ret"].rolling(20).std() # 
    g["_zscore_ret"] = g["_ret"] / (g["_vol20_std"] + EPS)

    # ── Intraday position ─────────────────────────────────────────────────────
    # 1.0 = closed at session high, 0.0 = closed at session low
    g["_close_pos"]  = (close - low) / (high - low + EPS)

    # ── Overnight gap ─────────────────────────────────────────────────────────
    g["_gap"]        = (open_ - close.shift(1)) / (close.shift(1) + EPS)

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

    # retail_exhaustion (Obs 1)
    # Closed near the LOW with elevated volume = retail panic / capitulation.
    # In bear regimes: high value is a sell signal or capitulation bottom.
    # In bull regimes: high value warns of distribution (sellers taking over).
    g["retail_exhaustion"] = (1.0 - g["_close_pos"]) * g["_vol_spike"]

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
    g["zscore_return"]  = g["_zscore_ret"] # 
    g["rsi_divergence"] = g["_rsi_div"]
    g["close_position"] = g["_close_pos"]
    g["gap"]            = g["_gap"]
    g["vol_relative"]   = g["_roll_std5"] / (g["_roll_mean5"] + EPS)

    # ── Obs 1: herd momentum ──────────────────────────────────────────────────
    # Fraction of up-days in last 10 sessions.
    # Low value = bearish consensus building (retail all selling together).
    # Can precede a mean-reversion bounce when extreme.
    daily_up               = (close.diff() > 0).astype(int)
    g["herd_momentum_10"]  = daily_up.rolling(10).mean()

    # ── Obs 3: T+2 forced selling ─────────────────────────────────────────────
    # A sharp drop 2 sessions ago means those buyers' shares are settling today.
    # They are forced sellers. When combined with current elevated volume,
    # this predicts continuation of the decline.
    sharp_drop_2d         = (g["_ret"].shift(2) < -0.04).astype(int)
    g["t2_forced_selling"] = sharp_drop_2d * g["_vol_spike"]

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
    g["zscore_return"]      = g["_zscore_ret"]
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

    # ── Obs 5: range expansion ────────────────────────────────────────────────
    # Panic sessions have high-low ranges far above recent average.
    # Very high range_expansion = maximum fear = potential exhaustion point.
    g["range_expansion"] = g["_daily_range"] / (g["_range_mean20"] + EPS)

    # ── Obs 6: margin cascade duration ───────────────────────────────────────
    # How many of the last 3 sessions had returns more extreme than −2 std?
    # Cascade of 3 = forced liquidation likely exhausted; bounce imminent.
    extreme_down                   = (g["_zscore_ret"] < -2.0).astype(int)
    g["margin_cascade_duration"]   = extreme_down.rolling(3, min_periods=1).sum()

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
    g["gap"]             = g["_gap"]
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

    # ── Obs 5: range expansion ────────────────────────────────────────────────
    # Breakout days tend to have high-low ranges well above recent average.
    # Large range_expansion + positive return = real breakout (not a fake).
    # Small range despite high volume = absorption at resistance → be cautious.
    g["range_expansion"] = g["_daily_range"] / (g["_range_mean20"] + EPS)

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
