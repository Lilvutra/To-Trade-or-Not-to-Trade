"""
feature_engineering/base.py
----------------------------
Defines the FeatureStage protocol and the FeaturePipeline that chains stages.

Design contract
---------------
Each FeatureStage:
  - reads only columns it declares in `required_columns`
  - writes only columns it declares in `output_columns`
  - never modifies existing columns (append-only)
  - is stateless between calls (reproducible on the same input)

FeaturePipeline:
  - chains stages in order
  - builds the target column (next-day return)
  - drops rows with NaN (from rolling windows and the forward target)
  - returns (X: DataFrame, y: Series)
"""

from __future__ import annotations

from abc import ABC, abstractmethod
import numpy as np
import pandas as pd

DATE_COL  = "TradingDate"
TARGET_COL = "target_ret_1"  # forward 1-day return — what we predict


class FeatureStage(ABC):
    """
    Abstract base for a single feature-engineering step.

    Subclass and implement `build(df)`. The method receives the full DataFrame
    accumulated so far and must return it with new columns appended.
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """Unique identifier used in experiment logs."""
        ...

    @property
    @abstractmethod
    def output_columns(self) -> list[str]:
        """Columns this stage adds. Used for documentation and slicing."""
        ...

    @abstractmethod
    def build(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Append feature columns to df and return it.
        Do not modify or drop any existing column.
        """
        ...

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(name={self.name!r})"


class FeaturePipeline:
    """
    Chain multiple FeatureStage objects and produce (X, y) for modelling.

    Parameters
    ----------
    stages : list of FeatureStage
        Applied in order. Later stages can use columns built by earlier ones.
    target_horizon : int
        Number of days ahead to predict. Default = 1 (next-day return).
    close_col : str
        Column name for closing price.
    drop_na : bool
        If True, drop rows where any feature or target is NaN.
    """

    def __init__(
        self,
        stages: list[FeatureStage],
        target_horizon: int = 1,
        close_col: str = "Close",
        drop_na: bool = True,
    ) -> None:
        self.stages          = stages
        self.target_horizon  = target_horizon
        self.close_col       = close_col
        self.drop_na         = drop_na

    @property
    def stage_names(self) -> list[str]:
        return [s.name for s in self.stages]

    @property
    def feature_columns(self) -> list[str]:
        cols: list[str] = []
        for stage in self.stages:
            cols.extend(stage.output_columns)
        return cols

    def build(self, df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
        """
        Run all stages and return (X, y).

        X : DataFrame with feature columns only
        y : Series of forward returns (target_horizon days ahead)
        """
        df = df.copy().sort_values(DATE_COL).reset_index(drop=True)

        for stage in self.stages:
            df = stage.build(df)

        # Target: forward return over target_horizon sessions
        fwd_ret = df[self.close_col].pct_change(self.target_horizon).shift(-self.target_horizon)
        df[TARGET_COL] = fwd_ret

        available = [c for c in self.feature_columns if c in df.columns]
        X = df[available]
        y = df[TARGET_COL]

        if self.drop_na:
            mask = X.notna().all(axis=1) & y.notna()
            X = X[mask].reset_index(drop=True)
            y = y[mask].reset_index(drop=True)

        return X, y

    def describe(self) -> None:
        """Print the pipeline configuration."""
        print(f"FeaturePipeline — {len(self.stages)} stage(s)")
        for i, stage in enumerate(self.stages):
            print(f"  [{i}] {stage.name}: {stage.output_columns}")
        print(f"  Target: {TARGET_COL} (horizon={self.target_horizon}d)")
