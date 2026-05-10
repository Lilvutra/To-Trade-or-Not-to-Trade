"""
plot_cross_sectional.py
-----------------------
Run the Fama-MacBeth cross-sectional analysis from run_cross_sectional.py,
save results to CSV, and produce four charts:

  1. Full-sample factor t-stats   — which features reliably predict returns
  2. Cumulative factor returns     — how each factor's edge accumulates over time
  3. Regime heatmap (t-stats)      — which factors matter in which market regime
  4. Factor Sharpe ratios          — mean/std per factor (return vs noise)

Outputs (all written to data/cross_sectional/):
  full_summary.csv
  regime_<name>.csv          (one per regime)
  factor_returns.csv         (raw per-date coefficients)
  plot_tstats.png
  plot_cumulative.png
  plot_regime_heatmap.png
  plot_sharpe.png
"""

import os
import sys
import glob

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.cm as cm

from get_regime import build_conditional_features, REGIME_FEATURES, REGIME_NAME
from plot_regime import label_true_regime
from run_cross_sectional import (
    run_cross_sectional_regression,
    fama_macbeth_summary,
    regime_fama_macbeth,
    univariate_fama_macbeth,
    regime_fm_proper,
)

# ─────────────────────────────────────────────────────────────────────────────
# Config
# ─────────────────────────────────────────────────────────────────────────────

DATA_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..", "data",
                 "data-vn-20230228", "stock-historical-data")
)
OUT_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..", "data", "cross_sectional")
)
os.makedirs(OUT_DIR, exist_ok=True)

N_STOCKS         = 150   # number of VNINDEX stocks to include
MIN_OBS          = 30    # minimum stocks per cross-section
TSTAT_THRESH     = 2.0   # |t| ≥ 2 → statistically significant
TOP_N_CUMULATIVE = 8     # how many factors to show in the cumulative plot

REGIME_COLORS = {
    "QUIET_BEAR":    "#ef4444",
    "PANIC_BEAR":    "#7f1d1d",
    "QUIET_BULL":    "#22c55e",
    "VOLATILE_BULL": "#f59e0b",
}


# ─────────────────────────────────────────────────────────────────────────────
# 1. Load data and run analysis
# ─────────────────────────────────────────────────────────────────────────────

def load_panel() -> tuple[pd.DataFrame, list[str]]:
    csv_files = sorted(glob.glob(os.path.join(DATA_DIR, "*-VNINDEX-History.csv")))[:N_STOCKS]
    print(f"Loading {len(csv_files)} stocks…")

    panels = []
    for path in csv_files:
        ticker = os.path.basename(path).split("-")[0]
        raw = pd.read_csv(path)
        raw["TradingDate"] = pd.to_datetime(raw["TradingDate"])
        raw["return_1"]    = raw["Close"].pct_change().shift(-1)
        raw["volatility"]  = raw["Close"].pct_change().rolling(20).std()

        out = build_conditional_features(raw, market_index="VNINDEX")
        out["ticker"] = ticker
        out = out.rename(columns={"TradingDate": "date"})
        panels.append(out)

    df = pd.concat(panels, ignore_index=True)
    df["date"] = pd.to_datetime(df["date"])
    print(f"Panel: {df.shape}  |  dates: {df['date'].nunique()}  |  tickers: {df['ticker'].nunique()}")

    seen: set = set()
    ordered: list = []
    for feats in REGIME_FEATURES.values():
        for f in feats:
            if f not in seen:
                seen.add(f); ordered.append(f)
    features = [f for f in ordered if f in df.columns]
    print(f"Features ({len(features)}): {features}\n")
    return df, features


def get_regime_map() -> pd.Series:
    """Market-level detected regime from VCB (large-cap VNINDEX proxy)."""
    vcb_path = os.path.join(DATA_DIR, "VCB-VNINDEX-History.csv")
    vcb_raw  = pd.read_csv(vcb_path)
    vcb_raw["TradingDate"] = pd.to_datetime(vcb_raw["TradingDate"])
    vcb_feat = build_conditional_features(vcb_raw, market_index="VNINDEX")
    vcb_feat = vcb_feat.rename(columns={"TradingDate": "date"})
    vcb_feat["date"] = pd.to_datetime(vcb_feat["date"])
    return vcb_feat.set_index("date")["regime"].rename("regime")


def get_true_regime_map(window: int = 20, vol_baseline: int = 252) -> pd.Series:
    """Forward-looking ground-truth regime from VCB using label_true_regime."""
    vcb_path = os.path.join(DATA_DIR, "VCB-VNINDEX-History.csv")
    vcb_raw  = pd.read_csv(vcb_path)
    vcb_raw["TradingDate"] = pd.to_datetime(vcb_raw["TradingDate"])
    vcb_feat = build_conditional_features(vcb_raw, market_index="VNINDEX")
    vcb_feat = label_true_regime(vcb_feat, window=window, vol_baseline=vol_baseline)
    vcb_feat = vcb_feat.rename(columns={"TradingDate": "date"})
    vcb_feat["date"] = pd.to_datetime(vcb_feat["date"])
    return vcb_feat.set_index("date")["true_regime"].rename("true_regime")


def run_analysis() -> tuple:
    df, features = load_panel()

    # Joint cross-sectional regression — all features, all stocks, all dates
    factor_df = run_cross_sectional_regression(
        df, features, use_wls=True, neutralize_industry=False
    )
    print(f"Factor returns: {factor_df.shape}\n")

    full_summary = fama_macbeth_summary(factor_df)

    regime_map            = get_regime_map()
    factor_df_with_regime = factor_df.join(regime_map)
    regime_results        = regime_fama_macbeth(factor_df_with_regime)

    # Univariate FM — each feature alone, all stocks, all dates
    print("\nRunning univariate FM…")
    uni_summary = univariate_fama_macbeth(df, features)

    # days_in_regime from the VCB market proxy (same method as run_cross_sectional __main__)
    regime_change    = (regime_map != regime_map.shift()).cumsum()
    days_map         = regime_map.groupby(regime_change).cumcount()
    df["days_in_regime"] = df["date"].map(days_map)

    # Regime-proper FM — Pass 1: all regime rows (including transitions)
    print("\nRunning regime-proper FM  [Pass 1 — all rows]…")
    proper_pass1 = regime_fm_proper(df, regime_map)

    # Regime-proper FM — Pass 2: core regime rows only (days_in_regime >= 10)
    df_core = df[df["days_in_regime"] >= 10].copy()
    n_drop  = len(df) - len(df_core)
    print(f"\nRunning regime-proper FM  [Pass 2 — core only, dropped {n_drop:,} transition rows]…")
    proper_pass2 = regime_fm_proper(df_core, regime_map)

    # Regime-proper FM — Pass 3: mutual agreement only (detected == true regime)
    true_regime_map = get_true_regime_map()
    df["_true_regime"] = df["date"].map(true_regime_map)
    df["_detected"]    = df["date"].map(regime_map)
    agree_mask = (df["_true_regime"] == df["_detected"]) & df["_true_regime"].notna()
    df_agree   = df[agree_mask].drop(columns=["_true_regime", "_detected"])
    n_agree    = agree_mask.sum()
    print(f"\nRunning regime-proper FM  [Pass 3 — agreement only, {n_agree:,} rows "
          f"({n_agree/len(df)*100:.1f}% of panel)]…")
    proper_pass3 = regime_fm_proper(df_agree, regime_map)

    return (factor_df, full_summary, regime_results, uni_summary,
            proper_pass1, proper_pass2, proper_pass3, regime_map)


# ─────────────────────────────────────────────────────────────────────────────
# 2. Save results to CSV
# ─────────────────────────────────────────────────────────────────────────────

def save_results(
    factor_df:      pd.DataFrame,
    full_summary:   pd.DataFrame,
    regime_results: dict,
) -> None:
    factor_df.to_csv(os.path.join(OUT_DIR, "factor_returns.csv"))
    full_summary.to_csv(os.path.join(OUT_DIR, "full_summary.csv"))

    for regime_id, result in regime_results.items():
        name = REGIME_NAME.get(regime_id, str(regime_id))
        result.to_csv(os.path.join(OUT_DIR, f"regime_{name}.csv"))

    print(f"Results saved to {OUT_DIR}/")
    print(f"  full_summary.csv  ({len(full_summary)} factors)")
    print(f"  factor_returns.csv  ({factor_df.shape[0]} dates × {factor_df.shape[1]} factors)")
    for rid, res in regime_results.items():
        print(f"  regime_{REGIME_NAME.get(rid, rid)}.csv  ({len(res)} factors)")


# ─────────────────────────────────────────────────────────────────────────────
# 3. Plot 1 — Full-sample factor t-statistics
# ─────────────────────────────────────────────────────────────────────────────

def plot_tstats(full_summary: pd.DataFrame, save_path: str) -> None:
    """
    Horizontal bar chart of Newey-West t-statistics for each factor.
    Bars are coloured by sign and significance:
      dark green  : t > +2   (significantly positive)
      light green : 0 < t < 2
      light red   : -2 < t < 0
      dark red    : t < -2   (significantly negative)
    """
    # Drop const; sort by t-stat for readability
    df = full_summary.drop(index="const", errors="ignore").copy()
    df = df.sort_values("t_stat")

    fig, ax = plt.subplots(figsize=(10, max(5, len(df) * 0.35)))

    colors = []
    for t in df["t_stat"]:
        if t >= TSTAT_THRESH:
            colors.append("#15803d")   # dark green
        elif t > 0:
            colors.append("#86efac")   # light green
        elif t > -TSTAT_THRESH:
            colors.append("#fca5a5")   # light red
        else:
            colors.append("#991b1b")   # dark red

    bars = ax.barh(df.index, df["t_stat"], color=colors, edgecolor="white", linewidth=0.4)

    # ±2 significance lines
    ax.axvline( TSTAT_THRESH, color="#475569", linestyle="--", linewidth=0.9, label=f"|t|={TSTAT_THRESH}")
    ax.axvline(-TSTAT_THRESH, color="#475569", linestyle="--", linewidth=0.9)
    ax.axvline(0, color="#1e293b", linewidth=0.7)

    # Value labels on bars
    for bar, t in zip(bars, df["t_stat"]):
        offset = 0.05 if t >= 0 else -0.05
        ax.text(t + offset, bar.get_y() + bar.get_height() / 2,
                f"{t:.2f}", va="center", ha="left" if t >= 0 else "right",
                fontsize=7.5, color="#1e293b")

    ax.set_xlabel("Newey-West t-statistic", fontsize=10)
    ax.set_title("Full-sample Fama-MacBeth: Factor t-statistics\n"
                 f"(|t| ≥ {TSTAT_THRESH} = significant, N stocks/date ≥ {MIN_OBS})",
                 fontsize=11, fontweight="bold")
    ax.legend(fontsize=8)
    ax.grid(axis="x", alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved → {save_path}")


# ─────────────────────────────────────────────────────────────────────────────
# 4. Plot 2 — Cumulative factor returns over time
# ─────────────────────────────────────────────────────────────────────────────

def plot_cumulative(
    factor_df:    pd.DataFrame,
    full_summary: pd.DataFrame,
    save_path:    str,
) -> None:
    """
    Line chart of cumulative mean factor return for the top-N factors
    ranked by |t-stat|.  A factor that consistently earns positive return
    shows an upward-sloping line.  Reversal (downslope) shows the factor
    stopped working in that period.
    """
    df = full_summary.drop(index="const", errors="ignore")
    top_factors = df["t_stat"].abs().nlargest(TOP_N_CUMULATIVE).index.tolist()

    # Only keep factor return columns (exclude 'const')
    plot_df = factor_df[top_factors].dropna(how="all")

    cmap   = cm.get_cmap("tab10", len(top_factors))
    fig, ax = plt.subplots(figsize=(14, 6))

    for i, feat in enumerate(top_factors):
        series = plot_df[feat].dropna()
        cumret = series.cumsum()
        t      = df.loc[feat, "t_stat"]
        label  = f"{feat}  (t={t:.2f})"
        ax.plot(cumret.index, cumret.values, label=label,
                color=cmap(i), linewidth=1.4)

    ax.axhline(0, color="#94a3b8", linewidth=0.8, linestyle="--")
    ax.set_title(f"Cumulative Factor Returns — top {TOP_N_CUMULATIVE} by |t-stat|",
                 fontsize=11, fontweight="bold")
    ax.set_ylabel("Cumulative coefficient sum", fontsize=9)
    ax.set_xlabel("Date", fontsize=9)
    ax.legend(fontsize=7.5, ncol=2, loc="upper left")
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved → {save_path}")


# ─────────────────────────────────────────────────────────────────────────────
# 5. Plot 3 — Regime heatmap of t-statistics
# ─────────────────────────────────────────────────────────────────────────────

def plot_regime_heatmap(
    full_summary:   pd.DataFrame,
    regime_results: dict,
    save_path:      str,
) -> None:
    """
    Heatmap: rows = regimes, columns = factors, cell = Newey-West t-statistic.
    Red = negative factor return, green = positive.
    Factors are ordered by full-sample |t-stat| so the most important ones
    appear on the left.

    This is the key diagnostic: a factor that matters only in QUIET_BEAR (e.g.
    mean-reversion signals) will show a strong cell there and weak/opposite
    cells in VOLATILE_BULL.
    """
    # Build matrix: rows=regimes, cols=factors
    fs = full_summary.drop(index="const", errors="ignore")
    factor_order = fs["t_stat"].abs().sort_values(ascending=False).index.tolist()

    regime_ids   = sorted(regime_results.keys())
    regime_names = [REGIME_NAME.get(r, str(r)) for r in regime_ids]

    matrix = pd.DataFrame(index=regime_names, columns=factor_order, dtype=float)
    for rid in regime_ids:
        name = REGIME_NAME.get(rid, str(rid))
        res  = regime_results[rid].drop(index="const", errors="ignore")
        for feat in factor_order:
            matrix.loc[name, feat] = res.loc[feat, "t_stat"] if feat in res.index else np.nan

    fig, ax = plt.subplots(figsize=(max(14, len(factor_order) * 0.55), 4.5))

    # Diverging colormap centred at 0, saturates at ±4
    norm = mcolors.TwoSlopeNorm(vmin=-4, vcenter=0, vmax=4)
    im   = ax.imshow(matrix.values.astype(float), cmap="RdYlGn", norm=norm, aspect="auto")

    ax.set_xticks(range(len(factor_order)))
    ax.set_xticklabels(factor_order, rotation=55, ha="right", fontsize=7.5)
    ax.set_yticks(range(len(regime_names)))
    ax.set_yticklabels(regime_names, fontsize=9)

    # Annotate cells and mark significance
    for i, rname in enumerate(regime_names):
        for j, feat in enumerate(factor_order):
            val = matrix.loc[rname, feat]
            if pd.isna(val):
                continue
            text_color = "white" if abs(val) > 2.5 else "#1e293b"
            ax.text(j, i, f"{val:.1f}", ha="center", va="center",
                    fontsize=7, color=text_color)
            # Box around significant cells
            if abs(val) >= TSTAT_THRESH:
                ax.add_patch(plt.Rectangle(
                    (j - 0.48, i - 0.48), 0.96, 0.96,
                    fill=False, edgecolor="black", linewidth=1.2
                ))

    plt.colorbar(im, ax=ax, fraction=0.015, pad=0.01,
                 label=f"NW t-statistic  (boxed = |t| ≥ {TSTAT_THRESH})")
    ax.set_title(
        "Regime-Conditional Fama-MacBeth: Factor t-statistics by Market Regime\n"
        "(columns ordered by full-sample |t|, red=negative, green=positive)",
        fontsize=10, fontweight="bold", pad=10,
    )
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved → {save_path}")


# ─────────────────────────────────────────────────────────────────────────────
# 6. Plot 4 — Factor Sharpe ratios (mean / std of factor return series)
# ─────────────────────────────────────────────────────────────────────────────

def plot_sharpe(full_summary: pd.DataFrame, save_path: str) -> None:
    """
    Bar chart of factor Sharpe ratios = mean factor return / std factor return.
    High Sharpe → the factor earns consistently, not just occasionally.
    Low Sharpe despite significant t-stat → the factor has one big period
    driving the average (regime-dependent or structural break).
    """
    df = full_summary.drop(index="const", errors="ignore").copy()
    df["sharpe"] = df["mean"] / (df["std"] + 1e-10)
    df = df.sort_values("sharpe")

    colors = ["#15803d" if s > 0 else "#991b1b" for s in df["sharpe"]]

    fig, ax = plt.subplots(figsize=(10, max(5, len(df) * 0.35)))
    bars = ax.barh(df.index, df["sharpe"], color=colors, edgecolor="white", linewidth=0.4)

    ax.axvline(0, color="#1e293b", linewidth=0.7)
    for bar, s in zip(bars, df["sharpe"]):
        offset = 0.005 if s >= 0 else -0.005
        ax.text(s + offset, bar.get_y() + bar.get_height() / 2,
                f"{s:.3f}", va="center", ha="left" if s >= 0 else "right",
                fontsize=7.5, color="#1e293b")

    ax.set_xlabel("Factor Sharpe ratio  (mean / std of daily factor return)", fontsize=9)
    ax.set_title("Factor Sharpe Ratios — consistency of factor return\n"
                 "(high Sharpe = factor earns reliably, not driven by one regime)",
                 fontsize=11, fontweight="bold")
    ax.grid(axis="x", alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved → {save_path}")


# ─────────────────────────────────────────────────────────────────────────────
# 7. Plot 5 — Univariate FM t-stats
# ─────────────────────────────────────────────────────────────────────────────

def plot_univariate_tstats(uni_summary: pd.DataFrame, save_path: str) -> None:
    """
    Standalone signal strength: each feature regressed alone (no multicollinearity).
    Same bar chart format as plot_tstats for direct comparison.
    """
    df = uni_summary.drop(index="const", errors="ignore").copy()
    df = df.sort_values("t_stat")

    colors = []
    for t in df["t_stat"]:
        if t >= TSTAT_THRESH:    colors.append("#15803d")
        elif t > 0:              colors.append("#86efac")
        elif t > -TSTAT_THRESH:  colors.append("#fca5a5")
        else:                    colors.append("#991b1b")

    fig, ax = plt.subplots(figsize=(10, max(5, len(df) * 0.35)))
    bars = ax.barh(df.index, df["t_stat"], color=colors, edgecolor="white", linewidth=0.4)

    ax.axvline( TSTAT_THRESH, color="#475569", linestyle="--", linewidth=0.9, label=f"|t|={TSTAT_THRESH}")
    ax.axvline(-TSTAT_THRESH, color="#475569", linestyle="--", linewidth=0.9)
    ax.axvline(0, color="#1e293b", linewidth=0.7)

    for bar, t in zip(bars, df["t_stat"]):
        n = int(df.loc[df["t_stat"] == t, "n_dates"].iloc[0]) if "n_dates" in df.columns else ""
        label = f"{t:.2f}  (n={n})" if n else f"{t:.2f}"
        offset = 0.1 if t >= 0 else -0.1
        ax.text(t + offset, bar.get_y() + bar.get_height() / 2,
                label, va="center", ha="left" if t >= 0 else "right",
                fontsize=7.5, color="#1e293b")

    ax.set_xlabel("Newey-West t-statistic", fontsize=10)
    ax.set_title("Univariate FM: Standalone Signal Strength per Feature\n"
                 "(each feature regressed alone — no multicollinearity)",
                 fontsize=11, fontweight="bold")
    ax.legend(fontsize=8)
    ax.grid(axis="x", alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved → {save_path}")


# ─────────────────────────────────────────────────────────────────────────────
# 8. Plot 6 — Regime-proper FM heatmap
# ─────────────────────────────────────────────────────────────────────────────

def plot_regime_proper_heatmap(regime_proper_results: dict, save_path: str) -> None:
    """
    Heatmap of regime-proper FM t-stats.
    Each regime's row shows only its own feature subset — the correct way to
    evaluate features as the training pipeline will use them.
    NaN cells = feature not used in that regime.
    """
    # Collect all features across all regimes, ordered by regime then feature
    all_features: list = []
    seen: set = set()
    for rid in sorted(regime_proper_results.keys()):
        feats = REGIME_FEATURES.get(rid, [])
        for f in feats:
            if f not in seen:
                seen.add(f); all_features.append(f)

    regime_ids   = sorted(regime_proper_results.keys())
    regime_names = [REGIME_NAME.get(r, str(r)) for r in regime_ids]

    # Build matrix — NaN where a feature is not in the regime's subset
    matrix = pd.DataFrame(np.nan, index=regime_names, columns=all_features)
    for rid in regime_ids:
        name  = REGIME_NAME.get(rid, str(rid))
        res   = regime_proper_results[rid].drop(index="const", errors="ignore")
        feats = REGIME_FEATURES.get(rid, [])
        for feat in feats:
            if feat in res.index:
                matrix.loc[name, feat] = res.loc[feat, "t_stat"]

    # Sort columns by max |t| across regimes
    col_order = matrix.abs().max(axis=0).sort_values(ascending=False).index.tolist()
    matrix = matrix[col_order]

    fig, ax = plt.subplots(figsize=(max(14, len(col_order) * 0.65), 4.5))

    norm = mcolors.TwoSlopeNorm(vmin=-6, vcenter=0, vmax=6)
    data = matrix.values.astype(float)
    im   = ax.imshow(data, cmap="RdYlGn", norm=norm, aspect="auto")

    ax.set_xticks(range(len(col_order)))
    ax.set_xticklabels(col_order, rotation=55, ha="right", fontsize=7.5)
    ax.set_yticks(range(len(regime_names)))
    ax.set_yticklabels(regime_names, fontsize=9)

    for i, rname in enumerate(regime_names):
        for j, feat in enumerate(col_order):
            val = matrix.loc[rname, feat]
            if pd.isna(val):
                # Grey out unused cells
                ax.add_patch(plt.Rectangle(
                    (j - 0.5, i - 0.5), 1.0, 1.0,
                    color="#e2e8f0", zorder=0
                ))
                ax.text(j, i, "—", ha="center", va="center", fontsize=7, color="#94a3b8")
                continue
            text_color = "white" if abs(val) > 3.5 else "#1e293b"
            ax.text(j, i, f"{val:.1f}", ha="center", va="center",
                    fontsize=7.5, color=text_color, fontweight="bold" if abs(val) >= TSTAT_THRESH else "normal")
            if abs(val) >= TSTAT_THRESH:
                ax.add_patch(plt.Rectangle(
                    (j - 0.48, i - 0.48), 0.96, 0.96,
                    fill=False, edgecolor="black", linewidth=1.4
                ))

    plt.colorbar(im, ax=ax, fraction=0.015, pad=0.01,
                 label=f"NW t-statistic  (boxed = |t| ≥ {TSTAT_THRESH}, grey = not used in regime)")
    ax.set_title(
        "Regime-Proper FM: Per-Regime Joint Regression (Regime's Features Only)\n"
        "(mirrors training pipeline — one model per regime, correct feature subset)",
        fontsize=10, fontweight="bold", pad=10,
    )
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved → {save_path}")


# ─────────────────────────────────────────────────────────────────────────────
# 9. Plot 7 — Factor return time series (one panel per feature)
# ─────────────────────────────────────────────────────────────────────────────

def plot_factor_return_series(
    factor_df:      pd.DataFrame,
    full_summary:   pd.DataFrame,
    regime_map:     pd.Series,
    save_path:      str,
    rolling_window: int = 63,   # ~3 months; window for rolling t-stat
    ncols:          int = 3,
) -> None:
    """
    One subplot per feature showing:
      - Grey area  : raw daily factor return (Stage 1 coefficient per day)
      - Solid line : cumulative factor return — slope reveals which periods drive the average
      - Dashed line: rolling t-stat (secondary axis) — structural break detector
                     A flat or declining rolling t-stat means the signal stopped working
      - Background : regime shading (QB/PB/QBull/VBull) so you can see if the
                     cumulative line spikes only during crash or bull periods

    How to read:
      - Upward-sloping cumulative line everywhere → robust signal, not crash-driven
      - Cumulative line flat then one big jump → signal dominated by a single episode
      - Rolling t-stat crosses ±2 → signal became/stopped being significant in that window
      - Cumulative line correlated with a single regime colour → regime-specific factor
    """
    features = [c for c in factor_df.columns if c != "const"]
    # Sort by full-sample |t-stat| — most important first
    fs = full_summary.drop(index="const", errors="ignore")
    features = sorted(features, key=lambda f: -abs(fs.loc[f, "t_stat"]) if f in fs.index else 0)

    nrows = int(np.ceil(len(features) / ncols))
    fig, axes = plt.subplots(
        nrows, ncols,
        figsize=(ncols * 6, nrows * 3.2),
        sharex=False,
    )
    axes = np.array(axes).flatten()

    # Pre-compute regime spans for shading
    regime_series = regime_map.reindex(factor_df.index).ffill()
    regime_spans: list[tuple] = []   # (start, end, regime_id)
    if not regime_series.empty:
        prev_rid  = regime_series.iloc[0]
        span_start = regime_series.index[0]
        for date, rid in regime_series.items():
            if rid != prev_rid:
                regime_spans.append((span_start, date, prev_rid))
                span_start = date
                prev_rid   = rid
        regime_spans.append((span_start, regime_series.index[-1], prev_rid))

    # Map regime int → colour with low alpha for background
    _REGIME_ALPHA = {
        0: ("#ef4444", 0.10),   # QUIET_BEAR  — light red
        1: ("#7f1d1d", 0.15),   # PANIC_BEAR  — dark red
        2: ("#22c55e", 0.08),   # QUIET_BULL  — light green
        3: ("#f59e0b", 0.10),   # VOLATILE_BULL — amber
    }

    for ax_idx, feat in enumerate(features):
        ax   = axes[ax_idx]
        ax2  = ax.twinx()

        series = factor_df[feat].dropna()
        cumret = series.cumsum()
        t_full = fs.loc[feat, "t_stat"] if feat in fs.index else float("nan")

        # ── Rolling t-stat ────────────────────────────────────────────────────
        roll_mean = series.rolling(rolling_window, min_periods=rolling_window // 2).mean()
        roll_std  = series.rolling(rolling_window, min_periods=rolling_window // 2).std()
        roll_tstat = roll_mean / (roll_std / np.sqrt(rolling_window) + 1e-10)

        # ── Regime shading ────────────────────────────────────────────────────
        for (s, e, rid) in regime_spans:
            color, alpha = _REGIME_ALPHA.get(int(rid) if pd.notna(rid) else -1, ("#cccccc", 0.05))
            ax.axvspan(s, e, color=color, alpha=alpha, linewidth=0)

        # ── Daily factor return (grey area) ───────────────────────────────────
        ax.fill_between(series.index, series.values, 0,
                        color="#94a3b8", alpha=0.25, linewidth=0)
        ax.axhline(0, color="#94a3b8", linewidth=0.6, linestyle="--")

        # ── Cumulative return (coloured by sign of full-sample t-stat) ────────
        line_color = "#15803d" if t_full > 0 else "#991b1b"
        ax.plot(cumret.index, cumret.values, color=line_color,
                linewidth=1.6, label="cumulative")
        ax.set_ylabel("Cumul. factor ret.", fontsize=7, color="#334155")
        ax.tick_params(axis="y", labelsize=6)

        # ── Rolling t-stat on secondary axis ─────────────────────────────────
        ax2.plot(roll_tstat.index, roll_tstat.values,
                 color="#6366f1", linewidth=1.0, linestyle="--",
                 alpha=0.85, label=f"rolling t ({rolling_window}d)")
        ax2.axhline( TSTAT_THRESH, color="#6366f1", linewidth=0.5, linestyle=":", alpha=0.6)
        ax2.axhline(-TSTAT_THRESH, color="#6366f1", linewidth=0.5, linestyle=":", alpha=0.6)
        ax2.axhline(0, color="#6366f1", linewidth=0.4, alpha=0.3)
        ax2.set_ylabel(f"rolling t ({rolling_window}d)", fontsize=6, color="#6366f1")
        ax2.tick_params(axis="y", labelsize=6, colors="#6366f1")
        # Clamp y-axis so one outlier doesn't flatten the rest
        ax2.set_ylim(-8, 8)

        ax.set_title(f"{feat}   (t={t_full:+.2f})", fontsize=8.5, fontweight="bold", pad=3)
        ax.tick_params(axis="x", labelsize=6, rotation=30)
        ax.grid(axis="x", alpha=0.15)

    # ── Legend (one shared legend at figure level) ────────────────────────────
    legend_patches = [
        plt.Line2D([0], [0], color="#15803d", linewidth=1.8, label="cumulative (pos t)"),
        plt.Line2D([0], [0], color="#991b1b", linewidth=1.8, label="cumulative (neg t)"),
        plt.Line2D([0], [0], color="#6366f1", linewidth=1.2, linestyle="--",
                   label=f"rolling t ({rolling_window}d)"),
        plt.Rectangle((0, 0), 1, 1, fc="#ef4444", alpha=0.25, label="QUIET_BEAR"),
        plt.Rectangle((0, 0), 1, 1, fc="#7f1d1d", alpha=0.35, label="PANIC_BEAR"),
        plt.Rectangle((0, 0), 1, 1, fc="#22c55e", alpha=0.20, label="QUIET_BULL"),
        plt.Rectangle((0, 0), 1, 1, fc="#f59e0b", alpha=0.25, label="VOLATILE_BULL"),
    ]
    fig.legend(handles=legend_patches, loc="lower center",
               ncol=len(legend_patches), fontsize=7.5,
               bbox_to_anchor=(0.5, -0.01), framealpha=0.9)

    # Hide unused axes
    for ax_idx in range(len(features), len(axes)):
        axes[ax_idx].set_visible(False)

    fig.suptitle(
        f"Factor Return Time Series — daily coefficient from Stage 1 FM\n"
        f"(sorted by full-sample |t|, rolling t window = {rolling_window} days, "
        f"background = market regime)",
        fontsize=10, fontweight="bold", y=1.01,
    )
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved → {save_path}")


# ─────────────────────────────────────────────────────────────────────────────
# 10. Plot 8 — Regime-Proper Pass 1 vs Pass 2 dumbbell chart
# ─────────────────────────────────────────────────────────────────────────────

def plot_pass_comparison(
    proper_pass1: dict,
    proper_pass2: dict,
    proper_pass3: dict,
    save_path:    str,
) -> None:
    """
    Three-dot dumbbell chart: one panel per regime, one row per feature.

    Dot 1 (hollow circle)  = Pass 1: all regime rows incl. transitions
    Dot 2 (filled circle)  = Pass 2: core only (days_in_regime >= 10)
    Dot 3 (filled diamond) = Pass 3: mutual agreement only (detected == true)

    Segment colours:
      P1→P2  green = strengthened removing transitions  |  red = weakened
      P2→P3  green = strengthened on agreement days     |  red = weakened
      grey   = little change (|Δt| < 0.5)

    How to read:
      P1→P2 green, P2→P3 green → robust signal that concentrates in clean regime periods
      P1→P2 green, P2→P3 red  → core improves over transitions, but needs correct detection
      P1→P2 red,   P2→P3 red  → transition artifact that also only fires when detector is wrong
      Dot crosses ±2           → significance threshold crossed between passes
    """
    regime_ids = sorted(
        set(proper_pass1.keys()) | set(proper_pass2.keys()) | set(proper_pass3.keys())
    )
    ncols = 2
    nrows = int(np.ceil(len(regime_ids) / ncols))

    fig, axes = plt.subplots(nrows, ncols,
                             figsize=(ncols * 9, nrows * 5.5),
                             squeeze=False)
    axes = axes.flatten()

    def _seg_color(ta, tb):
        if pd.isna(ta) or pd.isna(tb):
            return "#94a3b8"
        if abs(tb) - abs(ta) > 0.5:
            return "#15803d"
        if abs(ta) - abs(tb) > 0.5:
            return "#dc2626"
        return "#94a3b8"

    for ax_i, rid in enumerate(regime_ids):
        ax   = axes[ax_i]
        name = REGIME_NAME.get(rid, str(rid))
        res1 = proper_pass1.get(rid, pd.DataFrame()).drop(index="const", errors="ignore")
        res2 = proper_pass2.get(rid, pd.DataFrame()).drop(index="const", errors="ignore")
        res3 = proper_pass3.get(rid, pd.DataFrame()).drop(index="const", errors="ignore")

        feats = sorted(
            set(res1.index) | set(res2.index) | set(res3.index),
            key=lambda f: -abs(res3.loc[f, "t_stat"]) if f in res3.index
                          else (-abs(res2.loc[f, "t_stat"]) if f in res2.index else 0),
        )
        y_pos = np.arange(len(feats))

        for yi, feat in enumerate(feats):
            t1 = res1.loc[feat, "t_stat"] if feat in res1.index else np.nan
            t2 = res2.loc[feat, "t_stat"] if feat in res2.index else np.nan
            t3 = res3.loc[feat, "t_stat"] if feat in res3.index else np.nan

            c12 = _seg_color(t1, t2)
            c23 = _seg_color(t2, t3)

            if pd.notna(t1) and pd.notna(t2):
                ax.plot([t1, t2], [yi, yi], color=c12, linewidth=1.6, alpha=0.75)
            if pd.notna(t2) and pd.notna(t3):
                ax.plot([t2, t3], [yi, yi], color=c23, linewidth=1.6,
                        alpha=0.75, linestyle="--")

            if pd.notna(t1):
                ax.scatter(t1, yi, s=60, zorder=5,
                           facecolors="white", edgecolors=c12, linewidths=1.8)
            if pd.notna(t2):
                ax.scatter(t2, yi, s=60, zorder=5, color=c12)
            if pd.notna(t3):
                ax.scatter(t3, yi, s=70, zorder=5,
                           color=c23, marker="D")

        ax.set_yticks(y_pos)
        ax.set_yticklabels(feats, fontsize=8)
        ax.axvline( TSTAT_THRESH, color="#475569", linestyle="--", linewidth=0.8, alpha=0.5)
        ax.axvline(-TSTAT_THRESH, color="#475569", linestyle="--", linewidth=0.8, alpha=0.5)
        ax.axvline(0, color="#1e293b", linewidth=0.7)
        ax.set_xlabel("Newey-West t-stat", fontsize=8)
        ax.set_title(name, fontsize=10, fontweight="bold",
                     color=REGIME_COLORS.get(name, "#1e293b"))
        ax.grid(axis="x", alpha=0.2)

    for ax_i in range(len(regime_ids), len(axes)):
        axes[ax_i].set_visible(False)

    legend_items = [
        plt.Line2D([0], [0], color="#15803d", linewidth=2, label="Strengthened (|t| ↑)"),
        plt.Line2D([0], [0], color="#dc2626", linewidth=2, label="Weakened (|t| ↓)"),
        plt.Line2D([0], [0], color="#94a3b8", linewidth=2, label="Little change"),
        plt.scatter([], [], s=60, facecolors="white", edgecolors="#475569",
                    linewidths=1.8, label="Pass 1 — all rows incl. transitions"),
        plt.scatter([], [], s=60, color="#475569",
                    label="Pass 2 — core (days_in_regime ≥ 10)"),
        plt.scatter([], [], s=70, color="#475569", marker="D",
                    label="Pass 3 — agreement only (detected == true)"),
    ]
    fig.legend(handles=legend_items, loc="lower center", ncol=3,
               fontsize=8, bbox_to_anchor=(0.5, -0.03), framealpha=0.9)

    fig.suptitle(
        "Regime-Proper FM: Pass 1 → Pass 2 → Pass 3\n"
        "solid segment = P1→P2 (remove transitions)   "
        "dashed segment = P2→P3 (agreement days only)",
        fontsize=11, fontweight="bold",
    )
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved → {save_path}")


# ─────────────────────────────────────────────────────────────────────────────
# 11. Console summary
# ─────────────────────────────────────────────────────────────────────────────

def print_summary(full_summary: pd.DataFrame, regime_results: dict) -> None:
    print("\n" + "═" * 60)
    print("  FULL-SAMPLE FAMA-MacBETH SUMMARY")
    print("═" * 60)
    print(full_summary.round(4).to_string())

    sig = full_summary.drop(index="const", errors="ignore")
    sig = sig[sig["t_stat"].abs() >= TSTAT_THRESH]
    if len(sig):
        print(f"\n  Significant factors (|t| ≥ {TSTAT_THRESH}):")
        for feat, row in sig.iterrows():
            direction = "↑ positive" if row["t_stat"] > 0 else "↓ negative"
            print(f"    {feat:<28s}  t={row['t_stat']:+.2f}  {direction}")
    else:
        print(f"\n  No factors significant at |t| ≥ {TSTAT_THRESH}")

    for rid, res in sorted(regime_results.items()):
        label = REGIME_NAME.get(rid, str(rid))
        print(f"\n{'─'*60}")
        print(f"  REGIME {rid}: {label}  ({len(res)} factors)")
        print("─" * 60)
        print(res.round(4).to_string())


# ─────────────────────────────────────────────────────────────────────────────
# 8. Entry point
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    (factor_df, full_summary, regime_results, uni_summary,
     proper_pass1, proper_pass2, proper_pass3, regime_map) = run_analysis()

    save_results(factor_df, full_summary, regime_results)
    print_summary(full_summary, regime_results)

    print("\nGenerating plots…")
    plot_tstats(
        full_summary,
        save_path=os.path.join(OUT_DIR, "plot_tstats.png"),
    )
    plot_cumulative(
        factor_df, full_summary,
        save_path=os.path.join(OUT_DIR, "plot_cumulative.png"),
    )
    plot_regime_heatmap(
        full_summary, regime_results,
        save_path=os.path.join(OUT_DIR, "plot_regime_heatmap.png"),
    )
    plot_sharpe(
        full_summary,
        save_path=os.path.join(OUT_DIR, "plot_sharpe.png"),
    )
    plot_univariate_tstats(
        uni_summary,
        save_path=os.path.join(OUT_DIR, "plot_univariate_tstats.png"),
    )
    plot_regime_proper_heatmap(
        proper_pass2,
        save_path=os.path.join(OUT_DIR, "plot_regime_proper_heatmap.png"),
    )
    plot_factor_return_series(
        factor_df, full_summary, regime_map,
        save_path=os.path.join(OUT_DIR, "plot_factor_return_series.png"),
    )
    plot_pass_comparison(
        proper_pass1, proper_pass2, proper_pass3,
        save_path=os.path.join(OUT_DIR, "plot_pass_comparison.png"),
    )

    print(f"\nAll outputs written to {OUT_DIR}/")


if __name__ == "__main__":
    main()
