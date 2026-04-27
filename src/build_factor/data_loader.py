"""
data_loader.py
--------------
Thin wrapper around raw OHLCV CSV files.

Column contract — all downstream modules expect these names:
  TradingDate : datetime64[ns]
  Open        : float64
  High        : float64
  Low         : float64
  Close       : float64
  Volume      : float64
  Symbol      : str  (injected from filename if not present)
"""

from __future__ import annotations

import os
import glob
import pandas as pd

REQUIRED_COLS = {"Open", "High", "Low", "Close", "Volume"}
DATE_COL = "TradingDate"


def load_csv(
    path: str,
    symbol: str | None = None,
    date_col: str = DATE_COL,
) -> pd.DataFrame:
    """
    Load a single OHLCV CSV. Infers symbol from filename if not given.

    Filename convention expected: <TICKER>-<INDEX>-History.csv
    e.g. VCB-VNINDEX-History.csv → symbol = "VCB"
    """
    df = pd.read_csv(path)
    df[date_col] = pd.to_datetime(df[date_col])
    df = df.sort_values(date_col).reset_index(drop=True)

    missing = REQUIRED_COLS - set(df.columns)
    if missing:
        raise ValueError(f"{path}: missing columns {missing}")

    if "Symbol" not in df.columns:
        inferred = _symbol_from_path(path) if symbol is None else symbol
        df["Symbol"] = inferred

    for col in ["Open", "High", "Low", "Close", "Volume"]:
        df[col] = df[col].astype(float)

    return df


def load_directory(
    directory: str,
    pattern: str = "*-History.csv",
    limit: int | None = None,
) -> pd.DataFrame:
    """
    Load and concatenate all matching CSV files in a directory.
    Each file gets a Symbol column inferred from its filename.
    """
    files = sorted(glob.glob(os.path.join(directory, pattern)))
    if not files:
        raise FileNotFoundError(f"No files matching '{pattern}' in {directory}")

    if limit:
        files = files[:limit]

    frames = []
    for path in files:
        try:
            frames.append(load_csv(path))
        except Exception as exc:
            print(f"  [skip] {os.path.basename(path)}: {exc}")

    if not frames:
        raise RuntimeError("No files loaded successfully.")

    return pd.concat(frames, ignore_index=True)


def _symbol_from_path(path: str) -> str:
    name = os.path.basename(path)
    if name.endswith("-History.csv"):
        return name.split("-")[0]
    return name.replace(".csv", "")
