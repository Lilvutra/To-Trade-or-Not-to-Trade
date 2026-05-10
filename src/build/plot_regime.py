"""
plot_regime.py
--------------
Two functions:

  label_true_regime(df, window, bull_thresh, bear_thresh)
    Classifies each day into one of 4 regimes using the actual
    price return over `window` trading days.  No model — purely
    based on observed price movement.  Use this as a ground truth
    to validate detect_regime().

  plot_regime(data_path, ...)
    4-panel chart:
      1. Price with detected-regime background shading
      2. Detected regime bar   (from detect_regime / build_conditional_features)
      3. True regime bar       (from label_true_regime)
      4. Agreement heatmap     (detected vs true confusion matrix)

    Also prints agreement rate and a per-regime breakdown.

Vietnam market self-defined conventions used
--------------------------------
  Window    : 20 trading days ≈ 1 calendar month
  Thresholds: bull_thresh = bear_thresh = 0.04  (4%)
  Regimes:
    return >= +6%        → VOLATILE_BULL  (3)
     0% ≤ return < +6%  → QUIET_BULL     (2)
    -5% < return < 0%   → QUIET_BEAR     (0)
    return ≤ -5%        → PANIC_BEAR     (1)

  These thresholds are calibrated to Vietnam's typical monthly
  volatility.  A 4% move in a month is meaningful for HOSE stocks;
  adjust bull_thresh / bear_thresh for different markets.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.colors as mcolors
from sklearn.metrics import confusion_matrix

from get_regime import (
    build_conditional_features,
    infer_market_index_from_filename,
    REGIME_NAME,
)

# ── Shared colour scheme ──────────────────────────────────────────────────────
REGIME_COLORS = {
    "QUIET_BEAR":    "#ef4444",   # red
    "PANIC_BEAR":    "#7f1d1d",   # dark red
    "QUIET_BULL":    "#22c55e",   # green
    "VOLATILE_BULL": "#f59e0b",   # amber
}

# Regime IDs in canonical order (matches get_regime.REGIME_ID)
REGIME_ORDER = ["QUIET_BEAR", "PANIC_BEAR", "QUIET_BULL", "VOLATILE_BULL"]
REGIME_IDS   = {name: i for i, name in enumerate(REGIME_ORDER)}


# ─────────────────────────────────────────────────────────────────────────────
# 1. True regime labeler
# ─────────────────────────────────────────────────────────────────────────────

def label_true_regime(
    df:           pd.DataFrame,
    window:       int = 20,    # trading days — horizon for both direction and vol
    vol_baseline: int = 252,   # past rolling window for vol_median anchor
    close_col:    str = "Close",
) -> pd.DataFrame:
    """
    Assign a zero-lag ground-truth regime to every row using forward-looking axes.

    Both axes look FORWARD from day t — what actually happened over the next
    `window` trading days.  This eliminates the detection lag that makes a
    backward-looking label compare two lagged signals instead of detector vs oracle.

    The last `window` rows are set to NaN (no future data available).

    Two axes
    --------
    1. Direction axis  (forward `window`-day realised return)
         future_ret[t] = close[t+window] / close[t] - 1
         is_bull       = future_ret >= 0
         is_bear       = future_ret <  0
       Answers: "did price end higher or lower over the NEXT month?"
       Implementation: close.pct_change(window).shift(-window)

    2. Volatility axis  (forward realised vol vs past long-run median)
         log_ret      = diff(log(close))
         future_vol[t] = std(log_ret[t+1 .. t+window])   ← forward
         vol_median[t] = rolling(vol_baseline).median(past log_ret std)  ← past
         is_high_vol   = future_vol > vol_median
       Implementation: log_ret.rolling(window).std().shift(-window)

       vol_median stays backward-looking — it is the historical baseline that
       existed at time t.  Comparing future_vol against it answers:
       "was the NEXT month more turbulent than this stock's normal level?"

    Why forward-looking fixes the vol_median contamination problem
    --------------------------------------------------------------
    Backward label_true_regime: direction axis uses past 20 days, which still
    reflects the prior regime at a transition point (e.g. March 1 2020 looks
    QUIET_BULL because the past 20 days were mostly pre-crash). The forward
    version immediately sees the crash: the NEXT 20 days from March 1 contain
    the full COVID drawdown → PANIC_BEAR on day 1 of the crash.

    The vol_median contamination (post-crash baseline stays elevated for ~252
    days) still affects the vol axis, but the direction axis is now always
    correct.  For regime detection purposes, direction correctness matters more:
    PANIC_BEAR vs QUIET_BEAR confusion is more damaging than PANIC_BEAR vs
    VOLATILE_BULL.

    Returns
    -------
    df with two new columns:
      true_regime      : float (NaN for last `window` rows)
                         [0=QUIET_BEAR, 1=PANIC_BEAR, 2=QUIET_BULL, 3=VOLATILE_BULL]
      true_regime_name : str  (NaN for last `window` rows)
    """
    df    = df.copy().reset_index(drop=True)
    close = df[close_col]
    log_ret = np.log(close).diff()

    # ── Direction axis — FORWARD-LOOKING ─────────────────────────────────────
    # close.pct_change(window) at index i = close[i]/close[i-window] - 1
    # .shift(-window) pulls that value window steps earlier, giving:
    #   future_ret[t] = close[t+window] / close[t] - 1
    future_ret = close.pct_change(window).shift(-window)
    is_bull    = future_ret >= 0
    is_bear    = future_ret <  0

    # ── Volatility axis — forward vol vs past baseline ────────────────────────
    # log_ret.rolling(window).std() at index i = std of log_ret[i-window+1 .. i]
    # .shift(-window) shifts to i+window, giving std of log_ret[i+1 .. i+window]
    past_vol   = log_ret.rolling(window).std()
    vol_median = past_vol.rolling(vol_baseline).median()
    vol_median = vol_median.fillna(past_vol.expanding().median())
    future_vol  = past_vol.shift(-window)
    is_high_vol = future_vol > vol_median

    # ── Regime matrix ─────────────────────────────────────────────────────────
    regime = pd.Series(
        np.where(is_bear & ~is_high_vol, float(REGIME_IDS["QUIET_BEAR"]),
        np.where(is_bull & ~is_high_vol, float(REGIME_IDS["QUIET_BULL"]),
        np.where(is_bull &  is_high_vol, float(REGIME_IDS["VOLATILE_BULL"]),
        np.where(is_bear &  is_high_vol, float(REGIME_IDS["PANIC_BEAR"]),
                 np.nan)))),
        index=df.index,
        name="true_regime",
    )

    # Last `window` rows have no forward data
    regime.iloc[-window:] = np.nan

    df["true_regime"]      = regime
    df["true_regime_name"] = df["true_regime"].map(
        lambda x: REGIME_NAME.get(int(x)) if pd.notna(x) else np.nan
    )

    return df


# ─────────────────────────────────────────────────────────────────────────────
# 2. Drawdown-based true regime labeler
# ─────────────────────────────────────────────────────────────────────────────

def _find_peaks_troughs(close: pd.Series, min_move: float) -> list:
    """
    Zigzag algorithm.  Returns [(int_position, 'peak'|'trough'), ...].

    A peak   is confirmed when price subsequently falls  >= min_move from it.
    A trough is confirmed when price subsequently rises  >= min_move from it.
    The trailing unconfirmed extreme is NOT returned (caller marks it NaN).
    """
    vals   = close.values
    n      = len(vals)
    events = []

    # Track running high and low to establish first direction
    running_high = vals[0]; running_high_pos = 0
    running_low  = vals[0]; running_low_pos  = 0
    direction    = None
    extreme_val  = vals[0]; extreme_pos = 0

    for i in range(1, n):
        v = vals[i]

        if direction is None:
            if v > running_high:
                running_high = v; running_high_pos = i
            if v < running_low:
                running_low  = v; running_low_pos  = i

            drop = (running_high - v) / running_high if running_high > 0 else 0
            rise = (v - running_low)  / running_low  if running_low  > 0 else 0

            if drop >= min_move and drop >= rise:
                events.append((running_high_pos, "peak"))
                direction = "down"; extreme_val = v; extreme_pos = i
            elif rise >= min_move:
                events.append((running_low_pos, "trough"))
                direction = "up";   extreme_val = v; extreme_pos = i

        elif direction == "up":
            if v > extreme_val:
                extreme_val = v; extreme_pos = i
            elif extreme_val > 0 and (extreme_val - v) / extreme_val >= min_move:
                events.append((extreme_pos, "peak"))
                direction = "down"; extreme_val = v; extreme_pos = i

        else:  # direction == "down"
            if v < extreme_val:
                extreme_val = v; extreme_pos = i
            elif extreme_val > 0 and (v - extreme_val) / extreme_val >= min_move:
                events.append((extreme_pos, "trough"))
                direction = "up";   extreme_val = v; extreme_pos = i

    return events


def label_regime_drawdown(
    df:           pd.DataFrame,
    min_move:     float = 0.05,   # minimum % price move to confirm peak/trough
    panic_thresh: float = 0.07,   # peak-to-trough drawdown threshold: PANIC_BEAR vs QUIET_BEAR
    vol_window:   int   = 20,     # rolling window for realized vol (bull classification)
    vol_baseline: int   = 252,    # long-run median baseline for vol comparison
    close_col:    str   = "Close",
) -> pd.DataFrame:
    """
    Assign ground-truth regime using drawdown-based peak/trough detection.

    Hybrid two-axis classification
    ------------------------------
    Direction axis — locked per confirmed zigzag segment:
        Peak → Trough : BEAR direction
        Trough → Peak : BULL direction

    Volatility axis — computed per day within each segment:
        realized_vol[t] > vol_median[t] → high vol
        realized_vol[t] ≤ vol_median[t] → low vol

    Combined per day:
        BEAR + high vol → PANIC_BEAR
        BEAR + low  vol → QUIET_BEAR
        BULL + high vol → VOLATILE_BULL
        BULL + low  vol → QUIET_BULL

    Direction is fixed by the confirmed zigzag (no early-transition bias,
    no endpoint-only bias).  Vol varies day by day so sharp vol spikes
    inside a calm trend are captured without needing a new peak/trough.

    The trailing unconfirmed segment after the last peak/trough is left NaN.
    """
    df       = df.copy().reset_index(drop=True)
    close    = df[close_col]
    log_ret  = np.log(close).diff()

    realized_vol = log_ret.rolling(vol_window).std()
    vol_median   = realized_vol.rolling(vol_baseline).median()
    vol_median   = vol_median.fillna(realized_vol.expanding().median())
    is_high_vol  = realized_vol > vol_median   # per-day boolean Series

    events = _find_peaks_troughs(close, min_move)
    regime = pd.Series(np.nan, index=df.index, name="true_regime")

    if not events:
        df["true_regime"]      = np.nan
        df["true_regime_name"] = np.nan
        return df

    def _label_segment(s: int, e: int, is_bear: bool) -> None:
        for t in range(s, e):
            high_vol = bool(is_high_vol.iloc[t]) if pd.notna(is_high_vol.iloc[t]) else False
            if is_bear:
                regime.iloc[t] = float(REGIME_IDS["PANIC_BEAR" if high_vol else "QUIET_BEAR"])
            else:
                regime.iloc[t] = float(REGIME_IDS["VOLATILE_BULL" if high_vol else "QUIET_BULL"])

    # ── Initial segment before first confirmed event ──────────────────────────
    first_pos, first_type = events[0]
    _label_segment(0, first_pos, is_bear=(first_type == "trough"))

    # ── Confirmed segments between consecutive events ─────────────────────────
    for k in range(len(events) - 1):
        pos_start, etype_start = events[k]
        pos_end, _             = events[k + 1]
        _label_segment(pos_start, pos_end, is_bear=(etype_start == "peak"))

    # ── Trailing unconfirmed segment → NaN ───────────────────────────────────
    regime.iloc[events[-1][0]:] = np.nan

    df["true_regime"]      = regime
    df["true_regime_name"] = df["true_regime"].map(
        lambda x: REGIME_NAME.get(int(x)) if pd.notna(x) else np.nan
    )
    return df


# ─────────────────────────────────────────────────────────────────────────────
# 3. Agreement statistics
# ─────────────────────────────────────────────────────────────────────────────

def print_regime_agreement(
    df:     pd.DataFrame,
    window: int = 20,
) -> None:
    """
    Print agreement rate between detect_regime() and label_true_regime().
    Rows with NaN true_regime (first `window` rows) are excluded.
    """
    valid = df.dropna(subset=["true_regime_name"])

    detected = valid["regime_name"].values
    true     = valid["true_regime_name"].values

    overall = (detected == true).mean()
    print(f"\n{'─'*54}")
    print(f"  Regime agreement: {overall*100:.1f}%  ({(detected==true).sum()}/{len(true)} days)")
    print(f"  Window: {window} trading days  |  axes: return direction + rolling vol vs median")
    print(f"{'─'*54}")

    print(f"\n  {'Regime':<18s}  {'n_true':>7}  {'agree%':>7}  {'detected_as (top 2)':>22}")
    print(f"  {'─'*62}")
    for name in REGIME_ORDER:
        mask = (true == name)
        if mask.sum() == 0:
            continue
        agree  = (detected[mask] == name).mean()
        # top-2 detected labels when true label is `name`
        det_counts = pd.Series(detected[mask]).value_counts()
        top2 = "  ".join(
            f"{n}({c})" for n, c in det_counts.head(2).items()
        )
        print(f"  {name:<18s}  {mask.sum():>7d}  {agree*100:>6.1f}%  {top2}")


# ─────────────────────────────────────────────────────────────────────────────
# 3. Plot helpers
# ─────────────────────────────────────────────────────────────────────────────

def _shade_regimes(ax: plt.Axes, df: pd.DataFrame, col: str) -> None:
    """Fill background of ax with regime colours for contiguous blocks."""
    df = df.reset_index(drop=True)
    change = (df[col] != df[col].shift()).cumsum()
    for _, block in df.groupby(change):
        name  = block[col].iloc[0]
        start = block["TradingDate"].iloc[0]
        end   = block["TradingDate"].iloc[-1]
        color = REGIME_COLORS.get(name, "#94a3b8")
        ax.axvspan(start, end, color=color, alpha=0.13, zorder=1)


def _regime_bar(ax: plt.Axes, df: pd.DataFrame, col: str) -> None:
    """Draw a solid colour bar (no y axis) coloured by regime per day."""
    dates = df["TradingDate"].values
    for i in range(len(df) - 1):
        name  = df[col].iloc[i]
        color = REGIME_COLORS.get(name, "#94a3b8")
        ax.axvspan(dates[i], dates[i + 1], color=color, alpha=0.9)
    ax.set_yticks([])


def _confusion_heatmap(ax: plt.Axes, df: pd.DataFrame) -> None:
    """
    Confusion matrix: rows = true regime, cols = detected regime.
    Normalised by row (recall perspective): diagonal = agreement rate.
    """
    valid    = df.dropna(subset=["true_regime_name"])
    labels   = REGIME_ORDER
    cm       = confusion_matrix(
        valid["true_regime_name"],
        valid["regime_name"],
        labels=labels,
        normalize="true",
    )

    im = ax.imshow(cm, cmap="RdYlGn", vmin=0, vmax=1, aspect="auto")
    ax.set_xticks(range(len(labels)))
    ax.set_yticks(range(len(labels)))

    short = {"QUIET_BEAR": "Q.Bear", "PANIC_BEAR": "P.Bear",
             "QUIET_BULL": "Q.Bull", "VOLATILE_BULL": "V.Bull"}
    ax.set_xticklabels([short[l] for l in labels], fontsize=8)
    ax.set_yticklabels([short[l] for l in labels], fontsize=8)
    ax.set_xlabel("Detected →",  fontsize=8)
    ax.set_ylabel("← True",      fontsize=8)
    ax.set_title("Agreement (row-normalised)", fontsize=9)

    for i in range(len(labels)):
        for j in range(len(labels)):
            ax.text(
                j, i, f"{cm[i,j]:.2f}",
                ha="center", va="center", fontsize=8,
                color="white" if cm[i, j] < 0.4 or cm[i, j] > 0.75 else "#1e293b",
            )

    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)


# ─────────────────────────────────────────────────────────────────────────────
# 4. Main plot
# ─────────────────────────────────────────────────────────────────────────────

def plot_regime(
    data_path:    str,
    window:       int   = 20,       # window for label_true_regime (ignored for drawdown)
    vol_baseline: int   = 252,
    label_method: str   = "window", # "window" | "drawdown"
    min_move:     float = 0.15,     # drawdown method: min % move to confirm peak/trough
    panic_thresh: float = 0.20,     # drawdown method: drawdown threshold for PANIC_BEAR
    save_path:    str | None = None,
) -> None:
    """
    4-panel chart comparing detect_regime() against a ground-truth labeler.

    label_method="window"   → label_true_regime  (forward rolling-window)
    label_method="drawdown" → label_regime_drawdown (zigzag peak/trough)

    Panel 1 (tall) : Close price + detected-regime background shading
    Panel 2        : Detected regime bar  (model)
    Panel 3        : True regime bar      (ground truth)
    Panel 4        : Confusion matrix heatmap (agreement breakdown)
    """
    raw = pd.read_csv(data_path)
    raw["TradingDate"] = pd.to_datetime(raw["TradingDate"])
    market_index = infer_market_index_from_filename(data_path)

    df = build_conditional_features(raw, market_index=market_index)
    df = df.sort_values("TradingDate").reset_index(drop=True)

    if label_method == "drawdown":
        df = label_regime_drawdown(
            df, min_move=min_move, panic_thresh=panic_thresh,
            vol_window=window, vol_baseline=vol_baseline,
        )
        method_label = f"drawdown(min_move={min_move}, panic={panic_thresh})"
    else:
        df = label_true_regime(df, window=window, vol_baseline=vol_baseline)
        method_label = f"window={window}d"

    ticker = os.path.basename(data_path).split("-")[0]

    # ── Layout: 4 rows, last panel shares no x-axis (it's a matrix) ──────────
    fig = plt.figure(figsize=(18, 11))
    gs  = fig.add_gridspec(
        4, 2,
        height_ratios=[4, 0.7, 0.7, 2.5],
        width_ratios=[3, 1],
        hspace=0.08, wspace=0.25,
    )

    ax_price    = fig.add_subplot(gs[0, 0])
    ax_detected = fig.add_subplot(gs[1, 0], sharex=ax_price)
    ax_true     = fig.add_subplot(gs[2, 0], sharex=ax_price)
    ax_stats    = fig.add_subplot(gs[3, 0])   # agreement stats text
    ax_heatmap  = fig.add_subplot(gs[:, 1])   # confusion matrix on the right

    fig.suptitle(
        f"{ticker}  —  Detected Regime vs True Regime  "
        f"({method_label}, vol_baseline={vol_baseline}d)",
        fontsize=13, fontweight="bold",
    )

    # ── Panel 1: price ────────────────────────────────────────────────────────
    ax_price.plot(df["TradingDate"], df["Close"],
                  color="#1e293b", linewidth=1.0, zorder=3)
    ax_price.set_ylabel("Close Price", fontsize=10)
    ax_price.grid(axis="y", linestyle="--", alpha=0.35)
    _shade_regimes(ax_price, df, "regime_name")
    plt.setp(ax_price.get_xticklabels(), visible=False)

    # ── Panel 2: detected regime bar ─────────────────────────────────────────
    _regime_bar(ax_detected, df, "regime_name")
    ax_detected.set_ylabel("Detected", fontsize=9, labelpad=2)
    plt.setp(ax_detected.get_xticklabels(), visible=False)

    # ── Panel 3: true regime bar ──────────────────────────────────────────────
    _regime_bar(ax_true, df, "true_regime_name")
    ax_true.set_ylabel("True", fontsize=9, labelpad=2)
    ax_true.set_xlabel("Date", fontsize=10)

    # ── Panel 4 (bottom-left): agreement stats text ───────────────────────────
    ax_stats.axis("off")
    valid   = df.dropna(subset=["true_regime_name"])
    overall = (valid["regime_name"] == valid["true_regime_name"]).mean()

    lines = [f"Overall agreement: {overall*100:.1f}%\n"]
    lines.append(f"{'Regime':<16} {'n':>5}  {'agree':>7}  {'most detected as':>18}")
    lines.append("─" * 52)
    for name in REGIME_ORDER:
        mask = (valid["true_regime_name"] == name)
        if mask.sum() == 0:
            continue
        agree = (valid.loc[mask, "regime_name"] == name).mean()
        top   = valid.loc[mask, "regime_name"].value_counts().index[0]
        lines.append(
            f"{name:<16} {mask.sum():>5d}  {agree*100:>6.1f}%  {top:>18}"
        )
    ax_stats.text(
        0.02, 0.95, "\n".join(lines),
        transform=ax_stats.transAxes,
        fontsize=8.5, fontfamily="monospace",
        verticalalignment="top",
    )

    # ── Panel right: confusion heatmap ────────────────────────────────────────
    _confusion_heatmap(ax_heatmap, df)

    # ── Legend ────────────────────────────────────────────────────────────────
    patches = [
        mpatches.Patch(color=c, label=name)
        for name, c in REGIME_COLORS.items()
    ]
    ax_price.legend(handles=patches, loc="upper left", fontsize=8.5, framealpha=0.7)

    # ── Duration stats subtitle under detected bar ────────────────────────────
    counts = df["regime_name"].value_counts()
    total  = len(df)
    stats  = "  |  ".join(
        f"{n}: {counts.get(n, 0)}d ({counts.get(n, 0)/total*100:.0f}%)"
        for n in REGIME_ORDER
    )
    ax_detected.set_title(f"Detected → {stats}", fontsize=7.5, color="#475569", pad=3)

    counts_t = valid["true_regime_name"].value_counts()
    total_t  = len(valid)
    stats_t  = "  |  ".join(
        f"{n}: {counts_t.get(n, 0)}d ({counts_t.get(n, 0)/total_t*100:.0f}%)"
        for n in REGIME_ORDER
    )
    ax_true.set_title(f"True →  {stats_t}", fontsize=7.5, color="#475569", pad=3)

    # ── Print to console ──────────────────────────────────────────────────────
    print_regime_agreement(df, window=window)

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved → {save_path}")
    else:
        plt.show()


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    DATA_PATH = "./data/data-vn-20230228/stock-historical-data/VHM-VNINDEX-History.csv"

    # Window-based ground truth
    plot_regime(
        DATA_PATH,
        window=20,
        vol_baseline=252,
        label_method="window",
        save_path="./data/regime_comparison_VHM_window.png",
    )

    # Drawdown-based ground truth — short-term (catches intra-trend swings)
    plot_regime(
        DATA_PATH,
        vol_baseline=252,
        label_method="drawdown",
        min_move=0.07,
        panic_thresh=0.10,
        save_path="./data/regime_comparison_VHM_drawdown.png",
    )
