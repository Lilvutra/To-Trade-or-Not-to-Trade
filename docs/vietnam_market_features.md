# Vietnam Market: Microstructure Observations & Feature Engineering

## 1. Market Structure Overview

Vietnam's equity market (HOSE, HNX, UPCoM) operates under a set of structural constraints that make its price dynamics materially different from developed markets. Features designed for US or EU equities frequently fail here because the dominant behavioral and institutional forces are unique. All features in this pipeline are grounded in six observed structural properties.

---

## 2. Microstructure Observations

### Obs 1 — Retail Dominance (~85–90% of daily volume)
Retail investors account for the overwhelming majority of daily turnover. This produces systematic herding and overreaction: retail participants pile into momentum and panic-sell on bad news. The consequence is that **mean-reversion after extremes is predictable within 3–5 sessions** — the overreaction is large enough to be exploitable.

> Features derived: `retail_exhaustion`, `seller_exhaustion_fresh`, `conviction_close`

---

### Obs 2 — Price Limits (±7% HOSE) Create Queue Dynamics
HOSE imposes a daily price movement limit of ±7% (±10% UPCoM, ±5% HNX). When a stock hits the ceiling (limit-up), unmatched buy orders cannot be executed and spill into the next session's open — creating a **structural continuation effect**. Conversely, limit-down traps sellers: forced supply accumulates in the queue and overhangs the next open.

These are not soft equilibria — they are mechanical, predictable queue effects.

> Features derived: `limit_up_conviction`, `limit_down_conviction`, `limit_up_streak`, `limit_open_reversal`

---

### Obs 3 — T+2 Settlement Creates Forced-Selling Windows
Vietnam operates on T+2 settlement. Investors who bought a falling stock cannot sell for 2 sessions. A sharp drop 2 days ago predicts elevated selling today when combined with current elevated volume — three consecutive cohorts of trapped buyers create a **cascade of forced liquidations**.

> Features derived: `t2_forced_selling`, `t2_cascade`

---

### Obs 4 — Foreign / Institutional Flow as Smart Money Proxy
Without order-book data, institutional flow is approximated via price-volume interaction:
- **Up day + high volume** = someone large absorbed retail selling → institutional/foreign accumulation
- **Down day + high volume** = distribution by large holders → smart money exiting

This proxy works because retail investors in Vietnam are largely momentum-driven; elevated volume on an up day in a down-trending period signals contra-trend buying by a non-retail actor.

> Features derived: `smart_money_up`, `smart_money_down`, `volume_price_divergence`

---

### Obs 5 — ATC Session Distorts End-of-Day Prices
Vietnam's At-the-Close (ATC) session allows institutional bulk orders to moe price away from the continuous-trading equilibrium. Where the close lands in the day's range, combined with volume, reveals whether ATC activity was accumulation (closed near high) or distribution (closed near low despite price up).

> Features derived: `conviction_close`, `intraday_distribution`, `range_expansion_up/down`

---

### Obs 6 — Two-Phase Margin-Call Cascade
Sharp drops trigger broker margin calls. Forced liquidation at T+1 open causes further index gaps. After 2–3 consecutive extreme sessions, the cascade typically exhausts as the weakest holders are cleared out.

> Features derived: `margin_cascade_duration`, `seller_exhaustion_late`

---

## 3. Regime Detection

Each bar is assigned to one of four regimes using three backward-looking axes only — no look-ahead.

| Regime | ID | Condition | Dominant Dynamic |
|---|---|---|---|
| QUIET_BEAR | 0 | Low vol + downtrend | Retail drift, slow bleed |
| PANIC_BEAR | 1 | High vol + downtrend | Margin cascades, forced liquidation |
| QUIET_BULL | 2 | Low vol + uptrend | Institutional accumulation, orderly drift |
| VOLATILE_BULL | 3 | High vol + uptrend | Retail FOMO, limit-up queue momentum |

**Three detection axes:**

**Volatility axis** — compares current 20-day log-return std against its own 252-day rolling median. Adapts to each stock's baseline rather than using a universal threshold.
```
is_high_vol = rolling(20).std(log_ret) > rolling(252).median(rolling(20).std(log_ret))
```

**Trend axis** — fraction of up-days over the past 20 sessions.
```
is_uptrend   = fraction_up > 0.55
is_downtrend = fraction_up < 0.45
```
The 45–55% band is "sideways" — resolved by the momentum axis.

**Momentum axis** — 20-day cumulative return resolves the sideways zone.
```
is_up_momentum = close / close.shift(20) - 1 >= 0
```

---

## 4. Feature Engineering

Features are split into three layers:
1. **Shared base internals** — OHLCV building blocks computed once, prefixed `_`
2. **Universal public features** — valid in all regimes
3. **Regime-specific features** — built for each regime's dominant dynamic

---

### 4.1 Shared Base Internals

These are not model inputs directly but are combined into public features.

| Internal | Formula | Purpose |
|---|---|---|
| `_ret` | `close.pct_change()` | Simple daily return |
| `_log_ret` | `diff(log(close))` | Log return — dampens limit-hit extremes |
| `_ma5` | `close.rolling(5).mean()` | Short-term trend anchor |
| `_vol_spike` | `volume / rolling(20).mean(volume)` | Volume relative to 20-day average |
| `_vol20_std` | `rolling(20).std(_ret)` | Short-term return volatility |
| `_zscore_ret` | `_ret / _vol20_std` | Standardised return |
| `_close_pos` | `(close - low) / (high - low)` | Where close landed in day's range: 1=high, 0=low |
| `_gap` | `(open - close.shift(1)) / close.shift(1)` | Overnight gap as fraction of prior close |
| `_cascade_dur` | `rolling(3).sum(zscore_ret < -2)` | Count of extreme down-days in last 3 sessions |
| `_daily_range` | `(high - low) / close.shift(1)` | Day's range relative to prior close |
| `_range_mean20` | `rolling(20).mean(_daily_range)` | Baseline range for expansion detection |
| `_limit_up` | `_daily_ret > limit_threshold` | Hit ceiling (0.065 for HOSE) |
| `_limit_down` | `_daily_ret < -limit_threshold` | Hit floor |
| `_limit_up_streak` | consecutive limit-up count | Days consecutively at ceiling |

---

### 4.2 Universal Features (All Regimes)

#### `conviction_close` — Buyer Conviction Composite (Obs 5)
```
conviction_close = close_pos × log(1 + vol_spike)
```
Combines where the close landed in the day's range with volume intensity. High value = buyers held ground with conviction throughout the session and into ATC. Low value = sellers dominated; price closed near the low despite volume. This is the closest OHLCV proxy to order-flow data — the ATC session reveals who had the final say.

**FM results:** negative in QUIET_BULL (t=−2.49) and VOLATILE_BULL (t=−2.60 in core) — in bull regimes, a weak close warns of distribution beginning.

---

#### `seller_exhaustion_fresh` — Isolated Selling Shock (Obs 1)
```
seller_exhaustion_fresh = down_day × (1 - close_pos) × vol_spike × (1 - cascade_norm)
```
Fires on: down day + closed near low + high volume + NOT in an ongoing cascade. The `(1 - cascade_norm)` gate is the key innovation: it distinguishes a sudden lone selling event (retail overreaction → predictable bounce) from continuous cascade selling where exhaustion is not yet near.

**FM results:** PANIC_BEAR t=+5.2 (core). The signal is genuinely meaningful for identifying near-term bounces when selling is isolated.

---

#### `range_expansion_up` — Bullish Range Expansion / FOMO Overshoot (Obs 5)
```
range_expansion_up = (ret > 0) × (daily_range / range_mean20)
```
Wide up-candle relative to 20-day average range, on a positive-return day. In Vietnam, wide up-days reflect retail FOMO pile-in at ATC, not institutional conviction — institutions buy steadily, not in wide-range spikes. Consistently negative predictor across all regimes.

**FM results:** universal mean-reversion signal — QUIET_BEAR t=−4.45, VOLATILE_BULL t=−5.01. Wide FOMO candle predicts next-day fade.

---

#### `range_expansion_down` — Bearish Range Expansion / Panic Continuation
```
range_expansion_down = (ret < 0) × (daily_range / range_mean20)
```
Wide down-candle relative to average range, on a negative-return day. In PANIC_BEAR, this signals that forced liquidation is ongoing — the range is wide because sellers are urgent, not because buyers are stepping in. Predicts continuation of decline.

---

#### `gap_down` — Negative Overnight Gap (Obs 3, 6)
```
gap_down = clip(gap, upper=0)   # negative values only
```
Split from the bidirectional gap to allow a separate coefficient. In bear regimes, gap-down measures cascade severity: further overnight gaps after extreme drops signal proximity to capitulation (or continuation). Positive β (larger gap-down → bigger bounce) reflects the exhaustion interpretation.

**FM results:** QUIET_BEAR t=+3.26, PANIC_BEAR sign flip in core (−0.81 Pass 1 → +3.28 Pass 2).

---

#### `gap_up` — Positive Overnight Gap (Obs 2, FOMO)
```
gap_up = clip(gap, upper=0, lower=0)   # positive values only
```
In VOLATILE_BULL, unmatched limit-up orders from the prior session spill into the next open as a gap-up. Measures the continuation momentum from Obs 2.

---

#### `zscore_return_neg` — Negative-Tail Z-Score
```
zscore_return_neg = clip(ret / vol20_std, upper=0)
```
Zero on neutral/up days; increasingly negative on extreme down-days. Used as an oversold detector: high absolute value = the day's drop is statistically unusual for this stock. In QUIET_BULL and VOLATILE_BULL this identifies overreaction pullbacks — "small pullback day" = best continuation entry.

**FM results:** QUIET_BULL t=−3.28, VOLATILE_BULL t=−4.93 (agreement core). Stronger after filtering.

---

### 4.3 Regime-Specific Features

#### `smart_money_up` — Institutional Accumulation Proxy (Obs 4)
```
smart_money_up = (ret > 0) × vol_spike
```
Up day combined with volume above the 20-day average. In a falling or sideways market this signals contra-trend buying by a large participant (institutional or foreign). In a bull regime it confirms trend health — institutions are still net buyers, not distributing.

**FM results:** strongest feature across all regimes — PANIC_BEAR t=+12.7, QUIET_BULL t=+11.97, VOLATILE_BULL t=+10.81 (all core). Most reliable predictor in the pipeline.

---

#### `dist_ma` — Distance from 5-Day Moving Average (Obs 5)
```
dist_ma = (close - ma5) / ma5
```
Price displacement from the short-term trend anchor. Negative = oversold relative to recent trend; positive = overbought. In bear regimes, extreme negative values predict snap-back. In bull regimes, extreme positive values predict fade. Universal mean-reversion signal.

**FM results:** QUIET_BULL t=−6.03, VOLATILE_BULL t=−5.54, PANIC_BEAR t=−3.74 (core). Consistently negative across all regimes — the further from MA, the stronger the reversion pull.

---

#### `delta_dist` — MA Distance Velocity
```
delta_dist = diff(dist_ma)   # day-over-day change in distance from MA5
```
How fast price is moving relative to its short-term trend. Positive = price accelerating away from MA (upward momentum building). Negative = price decelerating toward MA (pullback beginning). Used as an early momentum inflection signal — often turns positive 1–2 days before price itself confirms.

**FM results:** QUIET_BEAR t=+8.1 (strongest in regime). In VOLATILE_BULL t=+3.71 (real breakout vs retail pump discriminator).

---

#### `vol_accel` — Volume Acceleration
```
vol_accel = diff(vol_chg)   # second derivative of volume
```
Rate of change in volume change. Rising = volume is accelerating → fuel for continuation. Falling = volume is decelerating → momentum losing power. Negative β in VOLATILE_BULL (vol deceleration = approaching exhaustion). In QUIET_BEAR: negative β means declining selling volume = bear weakening.

**FM results:** Interesting non-monotonic pass behavior in QUIET_BEAR — weakens P1→P2 (transition artifact) but strengthens P2→P3 (genuine regime signal on confirmed quiet bear days).

---

#### `macd_hist_slope` — MACD Histogram Momentum
```
macd_hist = (ema12 - ema26) - ema9(ema12 - ema26)
macd_hist_slope = diff(macd_hist)
```
Day-over-day change in the MACD histogram. Rising histogram = bullish momentum accelerating. Used in QUIET_BULL and VOLATILE_BULL to confirm trend before a price crossover is visible. The slope (not the level) is used because the level has level-dependency across different price scales.

Note: EMA spans (12/26/9) are standard but the signal is adapted — Vietnam's retail momentum tends to build faster than global markets, so the histogram slope is more informative than the crossover itself.

---

#### `limit_up_streak` — Consecutive Limit-Up Days (Obs 2)
```
limit_up_streak = consecutive count of days where daily_ret > limit_threshold
```
Counts how many consecutive sessions the stock has hit the ceiling. In VOLATILE_BULL, limit-up streaks create structural momentum: unmatched buy queues accumulate across sessions. However, the longest streaks precede the sharpest reversals — when the queue finally exhausts, the collapse is rapid. Negative β = exhaustion signal.

**FM results:** VOLATILE_BULL t=+1.0 (Pass 1) → t=+6.70 (Pass 2 core). One of the largest Pass 1→Pass 2 jumps — the signal is almost entirely a core-regime phenomenon, absent in transition days.

---

#### `limit_down_conviction` — Trapped-Seller Queue Overhang (Obs 2, 6)
```
limit_down_conviction = limit_down × vol_spike
```
Hit the floor AND traded high volume = an enormous queue of trapped sellers overhangs the next session. In PANIC_BEAR this signals continuation risk (sellers will dump at open) or, after 2+ limit-down days, potential capitulation bottom if accompanied by RSI divergence.

**FM results:** PANIC_BEAR t=−1.7 (core). The negative sign reflects the continuation interpretation — high trapped-seller overhang predicts further weakness.

---

#### `oversold_stable` — Gated Oversold Composite (Obs 1)
```
oversold_strength = (1 - z_pct) + (1 - dist_pct)   # how extreme the dip
oversold_stable   = oversold_strength
                  × (1 - cascade_norm)               # NOT in cascade
                  × (1 - vol_regime_spike)            # vol NOT elevated
                  × (1 - (down_ratio > 0.6))          # market NOT broadly weak
```
Where `z_pct` and `dist_pct` are 252-day percentile ranks of zscore_ret and dist_ma. A high value means the stock is very oversold *and* none of the stress gates have fired — a calm isolated dip rather than a systemic decline. Designed to identify genuine retail overreaction bounces in QUIET_BEAR.

**Structural limitation:** The gates screen for panic signals but cannot distinguish a genuine calm dip from a thin-market slow structural decline where there are simply no buyers. This explains the feature's sign flip across the sample period: it became predictive only after Vietnam's retail base grew large enough to create genuine mean-reversion after calm dips (post-2020).

---

### 4.4 Feature-to-Regime Mapping Summary

| Feature | QB | PB | QBull | VBull | Direction |
|---|---|---|---|---|---|
| `smart_money_up` | | ✓ | ✓ | ✓ | Positive — institutional buying |
| `dist_ma` | ✓ | ✓ | ✓ | ✓ | Negative — universal mean-reversion |
| `range_expansion_up` | | ✓ | ✓ | ✓ | Negative — FOMO overshoot fade |
| `zscore_return_neg` | | | ✓ | ✓ | Negative — oversold bounce |
| `delta_dist` | ✓ | ✓ | ✓ | ✓ | Positive — momentum velocity |
| `gap_down` | ✓ | ✓ | | | Positive — exhaustion bounce |
| `seller_exhaustion_fresh` | ✓ | ✓ | | | Positive — isolated spike bounce |
| `vol_accel` | ✓ | | | ✓ | Negative — deceleration = fading fuel |
| `macd_hist_slope` | | | ✓ | | Positive — momentum building |
| `conviction_close` | | | ✓ | ✓ | Mixed — distribution warning |
| `limit_up_streak` | | | | ✓ | Negative — exhaustion after streak |
| `limit_down_conviction` | | ✓ | | | Negative — trapped-seller overhang |
| `oversold_stable` | ✓ | | | | Positive — calm dip bounce |

QB = QUIET_BEAR, PB = PANIC_BEAR, QBull = QUIET_BULL, VBull = VOLATILE_BULL

---

## 5. Feature Quality Notes from Fama-MacBeth Analysis

**Most robust features** (significant in core AND agreement-only):
- `smart_money_up` — strongest signal across all regimes, robust to all filtering
- `dist_ma` — universal, stable mean-reversion, strengthens in core
- `delta_dist` — genuine regime-core signal, not transition-driven

**Transition artifacts** (weaken significantly Pass 1 → Pass 2):
- `vol_accel` in VOLATILE_BULL (t=−1.63 → −0.37): fires at regime boundaries
- `limit_up_streak` in VOLATILE_BULL (t=+1.0 → +6.70 in core): opposite direction — pure core signal, invisible in transitions

**Misclassification artifacts** (weaken Pass 2 → Pass 3):
- `gap_down` in QUIET_BEAR: borrows from adjacent PANIC_BEAR days mislabelled as QUIET_BEAR
- `seller_exhaustion_fresh` in QUIET_BEAR: fires on mis-detected days; weakens on confirmed clean QUIET_BEAR days

**Confirmed genuine regime signals** (strengthen in agreement-only):
- `vol_accel` in QUIET_BEAR: non-monotonic — weakens removing transitions, strengthens removing misclassification — true QUIET_BEAR core feature
- `zscore_return_neg` in VOLATILE_BULL: consistently strengthens across all passes
