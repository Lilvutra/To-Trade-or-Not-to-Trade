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

Vietnam market conventions used
--------------------------------
  Window    : 20 trading days ≈ 1 calendar month
  Thresholds: bull_thresh = bear_thresh = 0.04  (4%)
  Regimes:
    return >= +4%        → VOLATILE_BULL  (3)
     0% ≤ return < +4%  → QUIET_BULL     (2)
    -4% < return < 0%   → QUIET_BEAR     (0)
    return ≤ -4%        → PANIC_BEAR     (1)

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
    df:          pd.DataFrame,
    window:      int   = 20,    # trading days ≈ 1 calendar month in Vietnam
    bull_thresh: float = 0.04,  # +4% over window → VOLATILE_BULL
    bear_thresh: float = 0.04,  # -4% over window → PANIC_BEAR
    close_col:   str   = "Close",
) -> pd.DataFrame:
    """
    Assign a "ground truth" regime to every row using the actual price
    return over the past `window` trading days.

    This is the simplest possible regime classifier:
      - no volatility, no trend indicators, no rolling stats
      - just: where did price go over the last month?

    Why use this?
    -------------
    detect_regime() uses volatility + trend direction, which is predictive
    but may mislabel transition periods.  label_true_regime() uses realised
    returns, which is unambiguous but only observable in hindsight (for
    production you'd use the previous month's return, which is also historical
    and look-ahead-free at prediction time).

    Returns
    -------
    df with two new columns:
      true_regime      : int  [0=QUIET_BEAR, 1=PANIC_BEAR, 2=QUIET_BULL, 3=VOLATILE_BULL]
      true_regime_name : str
    """
    df = df.copy().reset_index(drop=True)

    # pct_change(window) at row t = close[t] / close[t-window] - 1
    # entirely historical — no look-ahead at any row t
    monthly_ret = df[close_col].pct_change(window)

    # Default: QUIET_BEAR 
    regime = pd.Series(REGIME_IDS["QUIET_BEAR"], index=df.index, name="true_regime")

    regime[monthly_ret >= bull_thresh]                            = REGIME_IDS["VOLATILE_BULL"]
    regime[(monthly_ret >= 0) & (monthly_ret < bull_thresh)]      = REGIME_IDS["QUIET_BULL"]
    regime[(monthly_ret < 0)  & (monthly_ret > -bear_thresh)]     = REGIME_IDS["QUIET_BEAR"]
    regime[monthly_ret <= -bear_thresh]                           = REGIME_IDS["PANIC_BEAR"]

    df["true_regime"]      = regime
    df["true_regime_name"] = df["true_regime"].map(REGIME_NAME)

    return df


# ─────────────────────────────────────────────────────────────────────────────
# 2. Agreement statistics
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
    print(f"  Window: {window} trading days  |  thresholds: ±4%")
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
    data_path:   str,
    window:      int   = 30,    # true-regime window in trading days
    bull_thresh: float = 0.06,  # 4 % → VOLATILE_BULL
    bear_thresh: float = 0.05,  # -5 % → PANIC_BEAR
    save_path:   str | None = None,
) -> None:
    """
    4-panel chart comparing detect_regime() against label_true_regime().

    Panel 1 (tall) : Close price + detected-regime background shading
    Panel 2        : Detected regime bar  (model)
    Panel 3        : True regime bar      (actual monthly return)
    Panel 4        : Confusion matrix heatmap (agreement breakdown)
    """
    raw = pd.read_csv(data_path)
    raw["TradingDate"] = pd.to_datetime(raw["TradingDate"])
    market_index = infer_market_index_from_filename(data_path)

    # detect_regime via build_conditional_features
    df = build_conditional_features(raw, market_index=market_index)
    df = df.sort_values("TradingDate").reset_index(drop=True)

    # true regime via actual monthly return
    df = label_true_regime(df, window=window,
                           bull_thresh=bull_thresh, bear_thresh=bear_thresh)

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
        f"(window={window}d, thresh=±{bull_thresh*100:.0f}%)",
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
    DATA_PATH = "./data/data-vn-20230228/stock-historical-data/VCB-VNINDEX-History.csv"
    plot_regime(
        DATA_PATH,
        window=30,
        bull_thresh=0.06,
        bear_thresh=0.05,
        save_path="./data/regime_comparison_VCB.png",
    )
