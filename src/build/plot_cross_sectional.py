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
from run_cross_sectional import (
    run_cross_sectional_regression,
    fama_macbeth_summary,
    regime_fama_macbeth,
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
    """Market-level regime from VCB (large-cap VNINDEX proxy)."""
    vcb_path = os.path.join(DATA_DIR, "VCB-VNINDEX-History.csv")
    vcb_raw  = pd.read_csv(vcb_path)
    vcb_raw["TradingDate"] = pd.to_datetime(vcb_raw["TradingDate"])
    vcb_feat = build_conditional_features(vcb_raw, market_index="VNINDEX")
    vcb_feat = vcb_feat.rename(columns={"TradingDate": "date"})
    vcb_feat["date"] = pd.to_datetime(vcb_feat["date"])
    return vcb_feat.set_index("date")["regime"].rename("regime")


def run_analysis() -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    df, features = load_panel()

    # run fama macbeth regression
    factor_df = run_cross_sectional_regression(
        df, features, use_wls=True, neutralize_industry=False
    )
    print(f"Factor returns: {factor_df.shape}\n")

    full_summary = fama_macbeth_summary(factor_df)

    regime_map              = get_regime_map()
    factor_df_with_regime   = factor_df.join(regime_map)
    regime_results          = regime_fama_macbeth(factor_df_with_regime)

    return factor_df, full_summary, regime_results


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
# 7. Console summary
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
    factor_df, full_summary, regime_results = run_analysis()

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

    print(f"\nAll outputs written to {OUT_DIR}/")


if __name__ == "__main__":
    main()
