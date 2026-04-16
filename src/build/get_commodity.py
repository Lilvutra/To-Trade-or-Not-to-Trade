"""
Fetch commodity data using vnstock_data based on the repository docs.

This module wraps the documented `vnstock_data.CommodityPrice` API and provides
convenient fetch methods for gold, oil, gas, steel, fertilizer, pork, and other
commodity series.

Reference:
  - docs/vnstock-data/10-commodity.md
  - docs/vnstock-data/README.md
"""

from __future__ import annotations

import warnings
import pandas as pd

warnings.filterwarnings("ignore")


class CommodityFeatures:
    """Commodity data helper for vnstock_data."""

    def __init__(self, start: str | None = None, end: str | None = None, length: str | int = "1Y", show_log: bool = False):
        self.start = start
        self.end = end
        self.length = length
        self.show_log = show_log
        self.commodity = self._init_commodity_api()

    def _init_commodity_api(self):
        try:
            from vnstock_data import CommodityPrice
            return CommodityPrice(start=self.start, end=self.end, length=self.length, show_log=self.show_log)
        except ImportError as exc:
            raise ImportError(
                "vnstock_data is required for commodity features. Install with `pip install vnstock_data`."
            ) from exc

    def gold_vn(self, **kwargs) -> pd.DataFrame:
        return self.commodity.gold_vn(**self._build_kwargs(kwargs))

    def gold_global(self, **kwargs) -> pd.DataFrame:
        return self.commodity.gold_global(**self._build_kwargs(kwargs))

    def gas_vn(self, **kwargs) -> pd.DataFrame:
        return self.commodity.gas_vn(**self._build_kwargs(kwargs))

    def oil_crude(self, **kwargs) -> pd.DataFrame:
        return self.commodity.oil_crude(**self._build_kwargs(kwargs))

    def gas_natural(self, **kwargs) -> pd.DataFrame:
        return self.commodity.gas_natural(**self._build_kwargs(kwargs))

    def coke(self, **kwargs) -> pd.DataFrame:
        return self.commodity.coke(**self._build_kwargs(kwargs))

    def steel_d10(self, **kwargs) -> pd.DataFrame:
        return self.commodity.steel_d10(**self._build_kwargs(kwargs))

    def iron_ore(self, **kwargs) -> pd.DataFrame:
        return self.commodity.iron_ore(**self._build_kwargs(kwargs))

    def steel_hrc(self, **kwargs) -> pd.DataFrame:
        return self.commodity.steel_hrc(**self._build_kwargs(kwargs))

    def fertilizer_ure(self, **kwargs) -> pd.DataFrame:
        return self.commodity.fertilizer_ure(**self._build_kwargs(kwargs))

    def soybean(self, **kwargs) -> pd.DataFrame:
        return self.commodity.soybean(**self._build_kwargs(kwargs))

    def corn(self, **kwargs) -> pd.DataFrame:
        return self.commodity.corn(**self._build_kwargs(kwargs))

    def sugar(self, **kwargs) -> pd.DataFrame:
        return self.commodity.sugar(**self._build_kwargs(kwargs))

    def pork_north_vn(self, **kwargs) -> pd.DataFrame:
        return self.commodity.pork_north_vn(**self._build_kwargs(kwargs))

    def pork_china(self, **kwargs) -> pd.DataFrame:
        return self.commodity.pork_china(**self._build_kwargs(kwargs))

    def _build_kwargs(self, override: dict[str, object]) -> dict[str, object]:
        result = {
            "start": self.start,
            "end": self.end,
            "length": self.length,
        }
        result.update({k: v for k, v in override.items() if v is not None})
        return result

    def fetch_all(self) -> dict[str, pd.DataFrame]:
        """Fetch a set of commodity series for quick exploration."""
        return {
            "gold_vn": self.gold_vn(),
            "gold_global": self.gold_global(),
            "gas_vn": self.gas_vn(),
            "oil_crude": self.oil_crude(),
            "gas_natural": self.gas_natural(),
            "coke": self.coke(),
            "steel_d10": self.steel_d10(),
            "iron_ore": self.iron_ore(),
            "steel_hrc": self.steel_hrc(),
            "fertilizer_ure": self.fertilizer_ure(),
            "soybean": self.soybean(),
            "corn": self.corn(),
            "sugar": self.sugar(),
            "pork_north_vn": self.pork_north_vn(),
            "pork_china": self.pork_china(),
        }


def fetch_commodity_series(symbol: str, **kwargs) -> pd.DataFrame:
    """Fetch a named commodity series by shorthand symbol."""
    mapping = {
        "gold_vn": "gold_vn",
        "gold_global": "gold_global",
        "gas_vn": "gas_vn",
        "oil_crude": "oil_crude",
        "gas_natural": "gas_natural",
        "coke": "coke",
        "steel_d10": "steel_d10",
        "iron_ore": "iron_ore",
        "steel_hrc": "steel_hrc",
        "fertilizer_ure": "fertilizer_ure",
        "soybean": "soybean",
        "corn": "corn",
        "sugar": "sugar",
        "pork_north_vn": "pork_north_vn",
        "pork_china": "pork_china",
    }
    if symbol not in mapping:
        raise ValueError(f"Unknown commodity symbol: {symbol}")
    features = CommodityFeatures(**kwargs)
    return getattr(features, mapping[symbol])()


if __name__ == "__main__":
    try:
        comm = CommodityFeatures(length="3M")
        print("Loaded vnstock_data commodity API")

        for symbol in ["gold_vn", "oil_crude", "pork_north_vn"]:
            df = fetch_commodity_series(symbol, length="3M")
            print(f"{symbol}: {df.shape}")
            print(df.head(), "\n")
    except Exception as exc:
        print(f"Error: {exc}")
