import pandas as pd
import numpy as np
import statsmodels.api as sm

# =============================================================================
# NEUTRALIZE BY INDUSTRY vs NOT — KEY DIFFERENCE
# =============================================================================
#
# WITHOUT neutralization — regression sees raw values:
#
#   Stock  Industry  mom_5   return_1
#     A     bank     +8%      +2%      ← banks rose because rates fell
#     B     bank     +6%      +1%
#     C     steel    -3%      -1%      ← steel fell on commodity news
#     D     steel    -5%      -2%
#
#   The model finds: high mom_5 → high return.
#   But the real driver is industry membership, not the signal.
#   Coefficient is CONTAMINATED by the industry event.
#
# WITH neutralization — regression sees within-industry deviations:
#
#   Stock  Industry  mom_5 residual   return_1 residual
#     A     bank        +1%               +0.5%    ← A beat B within banks
#     B     bank        -1%               -0.5%
#     C     steel       +1%               +0.5%    ← C beat D within steel
#     D     steel       -1%               -0.5%
#
#   Industry mean is removed from both sides. The coefficient now measures
#   pure stock-selection skill — does a stock beat its own sector peers?
#
# WHEN TO USE EACH:
#   neutralize_industry=True  → building a sector-neutral stock-picker
#                                (long best bank, short worst bank)
#   neutralize_industry=False → industry rotation is part of the strategy,
#                                let the model capture sector-level moves
#
# NOTE: current default is False because raw CSVs have no industry column.
#       Add industry labels to the panel and flip the flag to enable it.
# =============================================================================


def neutralize_by_industry(group, features, return_col="return_1"):
    """
    Industry-neutralize features and returns within a single cross-section.

    For each variable, regress on industry dummies and replace values with
    residuals. This removes the common industry component before factor
    estimation, so factor returns reflect within-industry variation only.

    Args:
        group      : DataFrame for one date. Must contain an 'industry' column.
        features   : list of feature column names to neutralize.
        return_col : name of the return column to neutralize.

    Returns:
        Copy of group with neutralized features and return column.
    """
    if "industry" not in group.columns:
        raise ValueError("DataFrame must contain an 'industry' column for industry neutralization.")

    # Work on a copy so the caller's DataFrame is not mutated.
    group = group.copy()

    # Convert the industry labels into binary (0/1) columns.
    # drop_first=True drops one industry (the reference category) to avoid
    # perfect multicollinearity: if you have bank/steel/energy, you only need
    # bank and steel — a stock with both = 0 is implicitly "energy".
    industry_dummies = pd.get_dummies(group["industry"], drop_first=True, dtype=float)

    # Neutralize every feature AND the return column.
    # Purpose: remove the part of each variable explained by industry membership,
    # so the FM regression only sees within-industry variation.
    for col in features + [return_col]:
        y = group[col]

        # has_constant="add" forces an explicit intercept column even though
        # statsmodels could detect one is already present.
        # The intercept absorbs the mean of the reference industry (the dropped
        # dummy). Without it, OLS is forced through zero and cannot correctly
        # remove the reference industry's mean, leaving biased residuals.
        X = sm.add_constant(industry_dummies, has_constant="add")

        try:
            # Fit y = β₀ + β₁·industry₁ + β₂·industry₂ + … + ε
            # The residual ε is the within-industry deviation — how this stock
            # differs from its own industry average on this date.
            residuals = sm.OLS(y, X).fit().resid
            # Replace the raw value with the industry-demeaned residual.
            group[col] = residuals.values
        except Exception:
            # A rank-deficient matrix (e.g. only one stock in an industry)
            # makes OLS fail. Skip neutralization for this column; the raw
            # value passes through unchanged rather than crashing the pipeline.
            pass

    return group


def run_cross_sectional_regression(
    df,
    features,
    return_col="return_1",
    min_obs=30,
    use_wls=False,
    neutralize_industry=False,
):
    """
    First stage of Fama-MacBeth: period-by-period cross-sectional OLS/WLS.

    For each date, regresses return_col on cross-sectionally standardized
    features and records the estimated coefficients (factor returns).

    Args:
        df                  : panel DataFrame with columns [date, return_col, *features,
                              optionally volatility, optionally industry].
        features            : list of feature column names.
        return_col          : dependent variable column (default "return_1").
        min_obs             : minimum stocks required to run a cross-section.
        use_wls             : if True, weight by 1/volatility (requires 'volatility' col).
        neutralize_industry : if True, remove industry effects before regression.

    Returns:
        DataFrame of factor returns indexed by date, one column per feature
        plus a 'const' column.
    """
    # Collect one row of coefficients per date; assembled into a DataFrame at the end.
    factor_returns = []

    # Each iteration is one cross-section: all stocks available on a single date.
    for date, group in df.groupby("date"):

        # Drop rows where any feature or the return is NaN — OLS cannot handle them.
        group = group.dropna(subset=features + [return_col])

        # dropna does not catch ±inf. Remove those too, as statsmodels will raise
        # MissingDataError if the design matrix contains infinite values.
        finite_mask = np.isfinite(group[features + [return_col]]).all(axis=1)
        group = group[finite_mask]

        # Skip dates with too few stocks — the regression would be unreliable
        # (more features than observations, or near-singular design matrix).
        if len(group) < min_obs:
            continue

        # Remove the common industry component from every feature and the return
        # so the regression measures within-industry alpha, not industry tilts.
        if neutralize_industry:
            group = neutralize_by_industry(group, features, return_col=return_col)

        # Cross-sectional z-score standardization.
        # Purpose: put all features on the same scale so their coefficients
        # (factor returns) are comparable in magnitude across features and dates.
        # Done AFTER neutralization so we standardize the already-demeaned values.
        X = group[features].copy()
        X = (X - X.mean()) / (X.std() + 1e-6)  # 1e-6 guards against zero-std columns

        y = group[return_col]

        # Add an intercept to capture the cross-sectional mean return on this date
        # (the "market return" component that is unrelated to any feature).
        X = sm.add_constant(X, has_constant="add")

        # OLS: equal weight to every stock.
        # WLS: weight each stock by 1/volatility so that low-volatility stocks
        # (whose returns are more precisely measured) influence the coefficients more.
        # This reduces noise from highly volatile small-caps dominating the estimate.
        if use_wls:
            weights = 1.0 / (group["volatility"] + 1e-6)
            model = sm.WLS(y, X, weights=weights).fit()
        else:
            model = sm.OLS(y, X).fit()

        # model.params is a Series of {feature_name: coefficient}.
        # Each coefficient is the "factor return" — how much a one-standard-deviation
        # increase in that feature predicted next-day return on this specific date.
        params = model.params.copy()
        params["date"] = date
        factor_returns.append(params)

    # Stack all per-date coefficient rows into a (T × K) DataFrame.
    # Rows = dates, columns = features + 'const'.
    factor_df = pd.DataFrame(factor_returns).set_index("date")
    return factor_df


def fama_macbeth_summary(factor_df):
    """
    Second stage of Fama-MacBeth: time-series average of factor returns.

    Uses Newey-West HAC standard errors (lags = floor(T^0.25)) to account
    for serial correlation in factor return series.

    Args:
        factor_df : DataFrame of per-period factor returns (output of
                    run_cross_sectional_regression, without a 'regime' column).

    Returns:
        DataFrame with columns [mean, std, se_nw, t_stat] sorted by |t_stat|.
    """
    T = len(factor_df)

    # Newey-West lag length rule of thumb: T^0.25.
    # Accounts for autocorrelation in factor return series up to this many lags.
    # E.g. T=1000 → lags=5; T=3000 → lags=7.
    lags = max(1, int(T ** 0.25))

    means = {}
    stds = {}
    nw_se = {}

    for col in factor_df.columns:
        series = factor_df[col].dropna()
        n = len(series)

        # Time-series mean of the per-date coefficient = the "average factor return".
        # A consistently positive mean means this feature reliably predicts returns.
        means[col] = series.mean()

        # Time-series std measures how stable the factor return is across dates.
        # High std relative to mean = the factor is noisy / regime-dependent.
        stds[col] = series.std()

        # Fall back to plain std error when too few observations for HAC.
        if n < 4:
            nw_se[col] = series.std() / np.sqrt(n) if n > 0 else np.nan
            continue

        # Newey-West standard error via a constant-only OLS trick.
        # Regressing the factor return series on a vector of ones is equivalent
        # to estimating the mean; the HAC-robust standard error of that estimate
        # corrects for autocorrelation (factor returns on adjacent days are not
        # independent), giving valid t-statistics.
        model = sm.OLS(series.values, np.ones(n)).fit(
            cov_type="HAC", cov_kwds={"maxlags": lags}
        )
        nw_se[col] = model.bse[0]  # bse[0] = standard error of the intercept = NW-SE of the mean

    summary = pd.DataFrame(
        {
            "mean":   pd.Series(means),
            "std":    pd.Series(stds),
            "se_nw":  pd.Series(nw_se),
            # t_stat > |2| is the conventional threshold for statistical significance.
            # Using NW standard errors makes this robust to serial correlation.
            "t_stat": pd.Series(means) / pd.Series(nw_se),
        }
    )

    # Sort by absolute t-stat so the most statistically significant factors appear first.
    return summary.sort_values("t_stat", key=abs, ascending=False)


def regime_fama_macbeth(factor_df, min_periods=20):
    """
    Run Fama-MacBeth summary separately for each market regime.

    Args:
        factor_df   : factor return DataFrame that already has a 'regime' column
                      joined in (e.g. via factor_df.join(regime_map)).
        min_periods : minimum number of periods required per regime.

    Returns:
        dict mapping regime label → fama_macbeth_summary DataFrame.
    """
    results = {}

    for regime, sub_df in factor_df.groupby("regime"):
        # Remove the regime label column before passing to fama_macbeth_summary,
        # which expects only factor return columns (no categorical columns).
        sub_df = sub_df.drop(columns=["regime"])

        # Skip regimes with too few dates — the NW standard errors and t-stats
        # would be unreliable with a tiny time series.
        if len(sub_df) < min_periods:
            continue

        # Run the full FM second stage within this regime's dates only.
        # The result shows which factors were significant when the market
        # was in this specific regime (e.g. mean-reversion dominates in QUIET_BEAR,
        # momentum dominates in VOLATILE_BULL).
        results[regime] = fama_macbeth_summary(sub_df)

    return results


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import glob
    import os
    import sys

    sys.path.insert(0, os.path.dirname(__file__))
    from get_regime import build_conditional_features, REGIME_FEATURES, REGIME_NAME

    DATA_DIR = os.path.abspath(
        os.path.join(
            os.path.dirname(__file__), "..", "..", "data",
            "data-vn-20230228", "stock-historical-data",
        )
    )

    # ------------------------------------------------------------------
    # 1. Load and featurise a sample of VNINDEX stocks
    # ------------------------------------------------------------------
    csv_files = sorted(glob.glob(os.path.join(DATA_DIR, "*-VNINDEX-History.csv")))[:150]
    print(f"Loading {len(csv_files)} stocks…")

    panels = []
    for path in csv_files:
        ticker = os.path.basename(path).split("-")[0]
        raw = pd.read_csv(path)
        raw["TradingDate"] = pd.to_datetime(raw["TradingDate"])

        # Attach forward return and volatility BEFORE build_conditional_features
        # so they survive the copy (non-"_" columns are preserved).
        raw["return_1"]   = raw["Close"].pct_change().shift(-1)   # 1-day forward return
        raw["volatility"] = raw["Close"].pct_change().rolling(20).std()

        out = build_conditional_features(raw, market_index="VNINDEX")
        out["ticker"] = ticker
        out = out.rename(columns={"TradingDate": "date"})
        panels.append(out)

    df = pd.concat(panels, ignore_index=True)
    df["date"] = pd.to_datetime(df["date"])
    print(f"Panel shape: {df.shape}  |  dates: {df['date'].nunique()}  |  tickers: {df['ticker'].nunique()}")

    # ------------------------------------------------------------------
    # 2. Feature list: union of all regime-specific features present in df
    # ------------------------------------------------------------------
    seen: set = set()
    all_feat: list = []
    for feats in REGIME_FEATURES.values():
        for f in feats:
            if f not in seen:
                seen.add(f)
                all_feat.append(f)

    features = [f for f in all_feat if f in df.columns]
    print(f"\nFeatures ({len(features)}): {features}")

    # ------------------------------------------------------------------
    # 3. Cross-sectional regression (WLS by volatility, no industry col)
    # ------------------------------------------------------------------
    factor_df = run_cross_sectional_regression(
        df,
        features,
        use_wls=True,
        neutralize_industry=False,   # raw data has no industry column
    )
    print(f"\nFactor returns shape: {factor_df.shape}")

    # ------------------------------------------------------------------
    # 4. Full-sample Fama-MacBeth summary
    # ------------------------------------------------------------------
    full_summary = fama_macbeth_summary(factor_df)
    print("\n=== Full-sample Fama-MacBeth ===")
    print(full_summary.to_string())

    # ------------------------------------------------------------------
    # 5. Regime-conditional Fama-MacBeth
    #    Derive a single market-level regime from VCB (large-cap VNINDEX proxy).
    #    This avoids the per-stock mode collapsing everything to QUIET_BEAR.
    # ------------------------------------------------------------------
    vcb_path = os.path.join(DATA_DIR, "VCB-VNINDEX-History.csv")
    vcb_raw = pd.read_csv(vcb_path)
    vcb_raw["TradingDate"] = pd.to_datetime(vcb_raw["TradingDate"])
    vcb_feat = build_conditional_features(vcb_raw, market_index="VNINDEX")
    vcb_feat = vcb_feat.rename(columns={"TradingDate": "date"})
    vcb_feat["date"] = pd.to_datetime(vcb_feat["date"])
    regime_map = vcb_feat.set_index("date")["regime"].rename("regime")

    factor_df_with_regime = factor_df.join(regime_map)
    print("\nMarket-regime distribution (from VCB proxy):")
    print(factor_df_with_regime["regime"].value_counts().sort_index())

    regime_results = regime_fama_macbeth(factor_df_with_regime)

    for regime_id, result in regime_results.items():
        label = REGIME_NAME.get(regime_id, regime_id)
        print(f"\n=== Regime {regime_id}: {label} ===")
        print(result.to_string())
