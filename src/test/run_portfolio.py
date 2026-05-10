"""
run_portfolio.py
----------------
Entry point for the full portfolio construction pipeline.

Pipeline
--------
  1. score_profitability  → rank stocks by FM factor score in current regime
  2. score_risk           → score each stock on 6 risk dimensions, flag exclusions
  3. build_portfolio      → risk-filter → top-N select → weight optimise

Outputs (all written to data/portfolio/)
  portfolio.csv           — final portfolio with weights and projected PnL
  excluded.csv            — excluded stocks with primary risk driver
  profitability_all.csv   — factor scores for entire universe
  risk_all.csv            — risk scores for entire universe

  plot_portfolio.png      — portfolio weight bar chart + regime label
  plot_risk_return.png    — risk-vs-score scatter (portfolio highlighted)
  plot_exclusions.png     — breakdown of why stocks were excluded
"""

from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from score_profitability import run_profitability, REGIME_EXPOSURE
from score_risk import run_risk_scoring
from build_portfolio import build_portfolio, summarise_excluded

# ─────────────────────────────────────────────────────────────────────────────
# Config
# ─────────────────────────────────────────────────────────────────────────────

OUT_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..", "data", "portfolio")
)
os.makedirs(OUT_DIR, exist_ok=True)

N_STOCKS     = 150
PROJ_HORIZON = 20    # trading days
TOP_N        = 20
WEIGHTING    = "score"   # "score" | "rank"
PORTFOLIO_SIZE = 1_000_000_000   # 1 billion VND

REGIME_COLORS = {
    "QUIET_BEAR":    "#ef4444",
    "PANIC_BEAR":    "#7f1d1d",
    "QUIET_BULL":    "#22c55e",
    "VOLATILE_BULL": "#f59e0b",
}


# ─────────────────────────────────────────────────────────────────────────────
# Plots
# ─────────────────────────────────────────────────────────────────────────────

def plot_portfolio(portfolio: pd.DataFrame, save_path: str) -> None:
    """
    Horizontal bar chart of portfolio weights, colour-coded by risk label.
    Includes a gross exposure annotation and regime label.
    """
    df = portfolio.sort_values("weight_pct", ascending=True).copy()
    regime       = df["regime"].iloc[0]
    gross_exp    = df["gross_exposure"].iloc[0]
    regime_color = REGIME_COLORS.get(regime, "#64748b")

    risk_palette = {
        "VERY LOW":  "#15803d",
        "LOW":       "#86efac",
        "MEDIUM":    "#fbbf24",
        "HIGH":      "#ef4444",
        "VERY HIGH": "#7f1d1d",
    }
    bar_colors = [risk_palette.get(r, "#94a3b8") for r in df["risk_label"]]

    fig, ax = plt.subplots(figsize=(11, max(6, len(df) * 0.42)))

    bars = ax.barh(df["ticker"], df["weight_pct"], color=bar_colors,
                   edgecolor="white", linewidth=0.4)

    for bar, row in zip(bars, df.itertuples()):
        ax.text(
            bar.get_width() + 0.15, bar.get_y() + bar.get_height() / 2,
            f"{row.weight_pct:.1f}%  ({row.proj_return_pct:+.1f}% proj, "
            f"risk={row.risk_score:.2f})",
            va="center", ha="left", fontsize=7.5, color="#1e293b"
        )

    ax.set_xlabel("Portfolio weight (%)", fontsize=9)
    ax.set_xlim(0, df["weight_pct"].max() * 1.55)
    ax.set_title(
        f"Portfolio Composition  —  Regime: {regime}  "
        f"(gross exposure: {gross_exp:.0%}, {len(df)} stocks)\n"
        f"Projection horizon: {PROJ_HORIZON} trading days  |  "
        f"Weighting: {WEIGHTING}-proportional",
        fontsize=10, fontweight="bold",
        color=regime_color,
    )

    # Risk legend
    legend_patches = [
        mpatches.Patch(color=c, label=l)
        for l, c in risk_palette.items()
    ]
    ax.legend(handles=legend_patches, title="Risk label",
              loc="lower right", fontsize=7.5)
    ax.grid(axis="x", alpha=0.25)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved → {save_path}")


def plot_risk_return(
    profitability: pd.DataFrame,
    risk:          pd.DataFrame,
    portfolio:     pd.DataFrame,
    save_path:     str,
) -> None:
    """
    Scatter: x = risk_score, y = factor_score.
    All universe stocks in grey; portfolio stocks highlighted in colour;
    excluded stocks with red X marker.
    """
    merged = profitability.merge(
        risk[["ticker", "risk_score", "risk_label", "exclude"]], on="ticker", how="left"
    )
    port_tickers = set(portfolio["ticker"].tolist())
    excl_mask    = merged["exclude"].fillna(False)

    fig, ax = plt.subplots(figsize=(11, 7))

    # All stocks
    ax.scatter(merged["risk_score"], merged["factor_score"],
               c="#cbd5e1", s=25, alpha=0.4, label="Universe", zorder=1)

    # Excluded
    excl = merged[excl_mask]
    ax.scatter(excl["risk_score"], excl["factor_score"],
               c="#ef4444", s=35, alpha=0.5, marker="x", linewidths=1.2,
               label="Excluded (high risk)", zorder=2)

    # Portfolio
    port = merged[merged["ticker"].isin(port_tickers)]
    ax.scatter(port["risk_score"], port["factor_score"],
               c="#15803d", s=90, alpha=0.9, label="Portfolio", zorder=3,
               edgecolors="white", linewidths=0.8)

    for _, row in port.iterrows():
        ax.annotate(
            row["ticker"],
            (row["risk_score"], row["factor_score"]),
            textcoords="offset points", xytext=(5, 3),
            fontsize=6.5, color="#065f46",
        )

    # Reference lines
    threshold = risk["threshold"].iloc[0] if "threshold" in risk.columns else 0.6
    ax.axvline(threshold, color="#ef4444", linewidth=0.9, linestyle="--",
               label=f"Risk threshold ({threshold:.2f})")
    ax.axhline(0, color="#94a3b8", linewidth=0.7, linestyle="--")

    regime = portfolio["regime"].iloc[0] if not portfolio.empty else ""
    ax.set_xlabel("Risk Score (0=safe, 1=risky)", fontsize=9)
    ax.set_ylabel("Factor Score (higher = stronger buy signal)", fontsize=9)
    ax.set_title(
        f"Risk vs Profitability — Universe ({len(merged)} stocks)  |  Regime: {regime}\n"
        "Green dots = selected portfolio  |  Red × = excluded  |  Grey = not selected",
        fontsize=10, fontweight="bold",
    )
    ax.legend(fontsize=8)
    ax.grid(alpha=0.2)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved → {save_path}")


def plot_exclusions(excluded: pd.DataFrame, save_path: str) -> None:
    """
    Bar chart of primary risk driver breakdown among excluded stocks.
    """
    if excluded.empty:
        return

    counts = excluded["primary_risk_driver"].value_counts()

    colors = {
        "Volatility": "#f59e0b",
        "Drawdown":   "#ef4444",
        "Tail Risk":  "#7f1d1d",
        "Liquidity":  "#6366f1",
        "Trend":      "#94a3b8",
        "Structural": "#0ea5e9",
    }
    bar_colors = [colors.get(c, "#64748b") for c in counts.index]

    fig, ax = plt.subplots(figsize=(8, 4))
    bars = ax.bar(counts.index, counts.values, color=bar_colors,
                  edgecolor="white", linewidth=0.5)

    for bar, v in zip(bars, counts.values):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.3,
                str(v), ha="center", va="bottom", fontsize=9)

    ax.set_ylabel("Number of excluded stocks", fontsize=9)
    ax.set_title(
        f"Excluded Stocks — Primary Risk Driver  ({len(excluded)} stocks excluded)\n"
        "(the dimension with the highest sub-score for each excluded stock)",
        fontsize=10, fontweight="bold",
    )
    ax.grid(axis="y", alpha=0.25)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved → {save_path}")


# ─────────────────────────────────────────────────────────────────────────────
# Console summary
# ─────────────────────────────────────────────────────────────────────────────

def print_portfolio_summary(portfolio: pd.DataFrame, excluded: pd.DataFrame) -> None:
    regime     = portfolio["regime"].iloc[0]
    gross_exp  = portfolio["gross_exposure"].iloc[0]
    total_pnl  = portfolio["expected_pnl_vnd"].sum()
    total_proj = (portfolio["proj_return_pct"] * portfolio["weight_pct"] / 100).sum()

    print("\n" + "═" * 65)
    print("  PORTFOLIO SUMMARY")
    print("═" * 65)
    print(f"  Market Regime         : {regime}")
    print(f"  Gross Exposure        : {gross_exp:.0%}  ({gross_exp*100:.0f}% invested, "
          f"{(1-gross_exp)*100:.0f}% cash)")
    print(f"  Holdings              : {len(portfolio)} stocks")
    print(f"  Projection Horizon    : {PROJ_HORIZON} trading days")
    print(f"  Weighted Proj. Return : {total_proj:+.2f}%  (weighted avg)")
    print(f"  Expected Portfolio PnL: {total_pnl/1e6:+.1f}M VND  "
          f"(on {PORTFOLIO_SIZE/1e9:.0f}B VND notional)")

    print(f"\n  {'Rank':<5} {'Ticker':<8} {'Score':>8} {'Proj Ret':>9} "
          f"{'Risk':>6} {'Weight':>7} {'Exposure':>9}")
    print("  " + "─" * 60)
    for _, r in portfolio.iterrows():
        print(f"  {int(r['rank']):<5} {r['ticker']:<8} {r['factor_score']:>8.2f} "
              f"{r['proj_return_pct']:>+8.1f}% {r['risk_score']:>6.2f} "
              f"{r['weight_pct']:>6.1f}% {r['exposure_pct']:>8.1f}%")

    if not excluded.empty:
        print(f"\n  Excluded: {len(excluded)} stocks")
        reasons = excluded["primary_risk_driver"].value_counts()
        for reason, count in reasons.items():
            print(f"    {reason:<14}: {count}")
    print("═" * 65)


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    print("=" * 65)
    print("  Vietnam Equity Portfolio Construction")
    print("=" * 65)

    # ── 1. Profitability scoring ──────────────────────────────────────────────
    print("\n[1/3] Profitability Scoring")
    prof_scores, regime = run_profitability(
        n_stocks=N_STOCKS, horizon=PROJ_HORIZON, verbose=True
    )

    # ── 2. Risk scoring ───────────────────────────────────────────────────────
    print("\n[2/3] Risk Scoring")
    risk_scores = run_risk_scoring(
        n_stocks=N_STOCKS, market_regime=regime, verbose=True
    )

    # ── 3. Portfolio construction ─────────────────────────────────────────────
    print("\n[3/3] Portfolio Construction")
    portfolio = build_portfolio(
        profitability_scores=prof_scores,
        risk_scores=risk_scores,
        regime=regime,
        top_n=TOP_N,
        weighting=WEIGHTING,
        portfolio_size=PORTFOLIO_SIZE,
        verbose=True,
    )

    excluded = summarise_excluded(risk_scores)

    # ── Console summary ───────────────────────────────────────────────────────
    if not portfolio.empty:
        print_portfolio_summary(portfolio, excluded)

    # ── Save CSVs ─────────────────────────────────────────────────────────────
    portfolio.to_csv(os.path.join(OUT_DIR, "portfolio.csv"), index=False)
    excluded.to_csv(os.path.join(OUT_DIR, "excluded.csv"), index=False)
    prof_scores.to_csv(os.path.join(OUT_DIR, "profitability_all.csv"), index=False)
    risk_scores.to_csv(os.path.join(OUT_DIR, "risk_all.csv"), index=False)
    print(f"\nCSVs saved to {OUT_DIR}/")

    # ── Plots ─────────────────────────────────────────────────────────────────
    print("\nGenerating plots…")
    if not portfolio.empty:
        plot_portfolio(
            portfolio,
            save_path=os.path.join(OUT_DIR, "plot_portfolio.png"),
        )
    plot_risk_return(
        prof_scores, risk_scores, portfolio,
        save_path=os.path.join(OUT_DIR, "plot_risk_return.png"),
    )
    if not excluded.empty:
        plot_exclusions(
            excluded,
            save_path=os.path.join(OUT_DIR, "plot_exclusions.png"),
        )

    print(f"\nAll outputs written to {OUT_DIR}/")


if __name__ == "__main__":
    main()
