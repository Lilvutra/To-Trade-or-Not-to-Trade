"""
Feature-only generation for Vietnam regime-based OHLCV features.

This module reuses the same feature engineering pipeline from get_regime.py,
but returns only the generated numerical features so the output can be fed
straight into a deep learning model.
"""

from __future__ import annotations

import os
import pandas as pd

from get_regime import build_conditional_features, infer_market_index_from_filename


def build_feature_matrix(
    df: pd.DataFrame,
    close_col: str = "Close",
    open_col: str = "Open",
    high_col: str = "High",
    low_col: str = "Low",
    vol_col: str = "Volume",
    time_col: str = "TradingDate",
    symbol_col: str | None = None,
    market_index: str | None = None,
    drop_time: bool = True,
) -> pd.DataFrame:
    """Build only the feature matrix from raw OHLCV input."""
    df = df.copy()
    df[time_col] = pd.to_datetime(df[time_col])

    if market_index is None and isinstance(df, pd.DataFrame) and "filename" in df.attrs:
        market_index = infer_market_index_from_filename(df.attrs["filename"])

    out = build_conditional_features(
        df,
        close_col=close_col,
        open_col=open_col,
        high_col=high_col,
        low_col=low_col,
        vol_col=vol_col,
        time_col=time_col,
        symbol_col=symbol_col,
        market_index=market_index,
    )

    drop_cols = ["regime", "regime_name", 'Unnamed: 0', ''] # drop regime labels and any unnamed index columns
    if drop_time and time_col in out.columns: # drop_time is True by default since most models won't use the timestamp directly, but we keep it as an option for debugging or if some time-based features are needed later
        drop_cols.append(time_col)
    if symbol_col and symbol_col in out.columns: 
        drop_cols.append(symbol_col)

    features = out.drop(columns=[c for c in drop_cols if c in out.columns])

    return features


def load_csv_and_build_features(
    path: str,
    close_col: str = "Close",
    open_col: str = "Open",
    high_col: str = "High",
    low_col: str = "Low",
    vol_col: str = "Volume",
    time_col: str = "TradingDate",
    symbol_col: str | None = None,
    drop_time: bool = True,
) -> pd.DataFrame:
    """Load a CSV file and build the feature-only DataFrame."""
    df = pd.read_csv(path)
    df[time_col] = pd.to_datetime(df[time_col])
    df.attrs["filename"] = path
    market_index = infer_market_index_from_filename(path)
    return build_feature_matrix(
        df,
        close_col=close_col,
        open_col=open_col,
        high_col=high_col,
        low_col=low_col,
        vol_col=vol_col,
        time_col=time_col,
        symbol_col=symbol_col,
        market_index=market_index,
        drop_time=drop_time,
    )


if __name__ == "__main__":
    data_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "data", "data-vn-20230228", "stock-historical-data"))
    sample_file = os.path.join(data_dir, "VCB-VNINDEX-History.csv")

    if not os.path.exists(sample_file):
        raise FileNotFoundError(f"Sample file not found: {sample_file}")

    print(f"Loading sample file: {sample_file}")
    features = load_csv_and_build_features(sample_file)
    print(f"Feature matrix shape: {features.shape}")
    print("Columns:")
    print(list(features.columns))
