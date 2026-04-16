"""
Extract and engineer macro-economic features for Vietnam stock market regime detection.

This module uses vnstock_data.Macro API to fetch:
  - GDP growth (quarterly)
  - CPI inflation (monthly)
  - FDI inflows (monthly)
  - Exchange rates (daily/monthly)
  - Interest rates (daily)
  - Money supply (monthly)
  - Industrial production (monthly)

Then engineers composite features for regime classification.

Reference: docs/vnstock-data/09-macro.md
"""

from __future__ import annotations

import os
import pandas as pd
import numpy as np
import warnings

warnings.filterwarnings('ignore')


class MacroFeatures:
    """
    Extract and engineer macro-economic features from vnstock_data.Macro.
    
    Features cover:
      1. Growth indicators: GDP, FDI, industrial production
      2. Inflation: CPI (monthly + YoY)
      3. Liquidity: Money supply growth, interest rates
      4. Currency: USD/VND exchange rate, trends
      5. Composite indicators: momentum, regimes, percentiles
    """

    def __init__(self, use_fallback: bool = True):
        """
        Initialize MacroFeatures.
        
        Args:
            use_fallback (bool): If True, use mock data when vnstock_data is unavailable
        """
        self.use_fallback = use_fallback
        self.macro = None
        self._init_macro_api()

    def _init_macro_api(self) -> None:
        """Initialize vnstock_data.Macro API if available."""
        try:
            from vnstock_data import Macro
            self.macro = Macro()
            print("✅ vnstock_data.Macro initialized successfully")
        except ImportError:
            print("⚠️ vnstock_data not available. Install with: pip install vnstock_data")
            if self.use_fallback:
                print("   Using mock data mode for demonstration.")

    def fetch_gdp_features(self, length: str = "2Y") -> pd.DataFrame | None:
        """
        Fetch GDP growth data (quarterly).
        
        Args:
            length (str): Time window (e.g., "2Y", "1Y")
            
        Returns:
            DataFrame with GDP growth or None if unavailable
        """
        if self.macro is None:
            return None
        try:
            df = self.macro.gdp(length=length, period="quarter")
            # Pivot to get separate columns
            gdp_total = df[df['name'] == 'Tổng GDP'][['value']].rename(columns={'value': 'gdp_yoy'})
            return gdp_total
        except Exception as e:
            print(f"⚠️ Error fetching GDP: {e}")
            return None

    def fetch_cpi_features(self, length: str = "2Y") -> pd.DataFrame | None:
        """
        Fetch CPI data (monthly).
        
        Args:
            length (str): Time window (e.g., "2Y", "1Y")
            
        Returns:
            DataFrame with CPI YoY
        """
        if self.macro is None:
            return None
        try:
            df = self.macro.cpi(length=length, period="month")
            cpi = df[df['name'] == 'Chỉ số giá tiêu dùng'][['value']].rename(columns={'value': 'cpi_yoy'})
            return cpi
        except Exception as e:
            print(f"⚠️ Error fetching CPI: {e}")
            return None

    def fetch_fdi_features(self, length: str = "2Y") -> pd.DataFrame | None:
        """
        Fetch FDI data (monthly, in billions USD).
        
        Args:
            length (str): Time window
            
        Returns:
            DataFrame with FDI registered and released
        """
        if self.macro is None:
            return None
        try:
            df = self.macro.fdi(length=length, period="month")
            fdi_registered = df[df['name'] == 'Đăng ký'][['value']].rename(columns={'value': 'fdi_registered_bn_usd'})
            fdi_released = df[df['name'] == 'Giải ngân'][['value']].rename(columns={'value': 'fdi_released_bn_usd'})
            result = pd.concat([fdi_registered, fdi_released], axis=1)
            return result
        except Exception as e:
            print(f"⚠️ Error fetching FDI: {e}")
            return None

    def fetch_exchange_rate_features(self, length: str = "2Y") -> pd.DataFrame | None:
        """
        Fetch USD/VND exchange rate (daily).
        
        Args:
            length (str): Time window
            
        Returns:
            DataFrame with USD/VND rate
        """
        if self.macro is None:
            return None
        try:
            df = self.macro.exchange_rate(length=length, period="day")
            # Filter for central rate
            rate = df[df['name'].str.contains('trung tâm', case=False, na=False)][['value']].rename(
                columns={'value': 'usd_vnd_rate'}
            )
            return rate
        except Exception as e:
            print(f"⚠️ Error fetching exchange rate: {e}")
            return None

    def fetch_interest_rate_features(self, length: str = "2Y") -> pd.DataFrame | None:
        """
        Fetch overnight interest rate (daily).
        
        Args:
            length (str): Time window
            
        Returns:
            DataFrame with overnight rate
        """
        if self.macro is None:
            return None
        try:
            df = self.macro.interest_rate(length=length, period="day", format="long")
            # Get overnight rate
            rate = df[df['name'] == 'Qua đêm'][['value']].rename(columns={'value': 'overnight_rate_pct'})
            return rate
        except Exception as e:
            print(f"⚠️ Error fetching interest rates: {e}")
            return None

    def fetch_money_supply_features(self, length: str = "2Y") -> pd.DataFrame | None:
        """
        Fetch money supply data (monthly).
        
        Args:
            length (str): Time window
            
        Returns:
            DataFrame with M1, M2, M3 growth
        """
        if self.macro is None:
            return None
        try:
            df = self.macro.money_supply(length=length, period="month")
            # Extract M2 as proxy for liquidity
            m2 = df[df['name'] == 'M2'][['value']].rename(columns={'value': 'money_supply_m2'}) # m2 is 
            return m2
        except Exception as e:
            print(f"⚠️ Error fetching money supply: {e}")
            return None

    def build_macro_features(self, length: str = "2Y") -> pd.DataFrame:
        """
        Combine all macro features into engineered feature set.
        
        Args:
            length (str): Time window
            
        Returns:
            DataFrame with engineered macro features
        """
        features_list = []

        # Fetch each category
        gdp = self.fetch_gdp_features(length)
        cpi = self.fetch_cpi_features(length)
        fdi = self.fetch_fdi_features(length)
        exr = self.fetch_exchange_rate_features(length)
        ir = self.fetch_interest_rate_features(length)
        ms = self.fetch_money_supply_features(length)

        # Combine what's available
        for df in [gdp, cpi, fdi, exr, ir, ms]:
            if df is not None and len(df) > 0:
                features_list.append(df)

        if not features_list:
            print("⚠️ No macro data fetched. Returning mock data.")
            return self.create_mock_macro_features(length)

        macro_df = pd.concat(features_list, axis=1)
        macro_df = macro_df.dropna(how='all')

        # Forward-fill missing values for daily features
        macro_df = macro_df.fillna(method='ffill').fillna(method='bfill')

        return self._engineer_features(macro_df)

    def _engineer_features(self, macro_df: pd.DataFrame) -> pd.DataFrame:
        """
        Engineer composite features from raw macro data.
        
        Args:
            macro_df (pd.DataFrame): Raw macro indicators
            
        Returns:
            DataFrame with engineered features
        """
        result = macro_df.copy()

        # 1. Growth momentum (14-day average of recent indicators)
        if 'gdp_yoy' in result.columns:
            result['gdp_momentum'] = result['gdp_yoy'].rolling(14, min_periods=1).mean()

        # 2. CPI trend (acceleration/deceleration)
        if 'cpi_yoy' in result.columns:
            result['cpi_acceleration'] = result['cpi_yoy'].diff().rolling(5, min_periods=1).mean()
            # Inflation regime
            result['inflation_regime'] = pd.cut(
                result['cpi_yoy'],
                bins=[-np.inf, 2, 5, np.inf],
                labels=['low', 'moderate', 'high']
            )

        # 3. FDI trend (12-month normalized)
        if 'fdi_released_bn_usd' in result.columns:
            result['fdi_12m_avg'] = result['fdi_released_bn_usd'].rolling(12, min_periods=1).mean()
            result['fdi_momentum'] = (
                result['fdi_released_bn_usd'] / result['fdi_12m_avg'].replace(0, np.nan)
            ).rolling(3, min_periods=1).mean()

        # 4. Exchange rate trend (momentum)
        if 'usd_vnd_rate' in result.columns:
            result['usd_vnd_ma20'] = result['usd_vnd_rate'].rolling(20, min_periods=1).mean()
            result['usd_vnd_momentum'] = (result['usd_vnd_rate'] / result['usd_vnd_ma20']) - 1

        # 5. Liquidity conditions
        if 'money_supply_m2' in result.columns:
            result['liquidity_growth'] = result['money_supply_m2'].pct_change(30).rolling(5, min_periods=1).mean()

        if 'overnight_rate_pct' in result.columns:
            result['rate_trend'] = result['overnight_rate_pct'].diff().rolling(5, min_periods=1).mean()

        # 6. Composite "Macro Strength" score (-1 to +1)
        scores = []
        for col in ['gdp_momentum', 'cpi_momentum', 'fdi_momentum', 'usd_vnd_momentum', 'liquidity_growth']:
            if col in result.columns:
                # Normalize to [0, 1]
                normalized = (result[col] - result[col].min()) / (result[col].max() - result[col].min() + 1e-8)
                scores.append(normalized)

        if scores:
            result['macro_strength_score'] = pd.concat(scores, axis=1).mean(axis=1) * 2 - 1

        return result

    def create_mock_macro_features(self, length: str = "2Y") -> pd.DataFrame:
        """
        Create mock macro features for demonstration when API unavailable.
        
        Args:
            length (str): Time window
            
        Returns:
            Mock DataFrame with realistic macro data
        """
        days = 252 * 2 if "2Y" in length else 252
        dates = pd.date_range(end=pd.Timestamp.now(), periods=days, freq='D')

        np.random.seed(42)

        # Realistic base values
        gdp_base = 7.0 
        cpi_base = 3.5
        fdi_base = 2.5
        usd_vnd_base = 24500

        macro_df = pd.DataFrame({
            'gdp_yoy': gdp_base + np.random.normal(0, 0.5, days),
            'cpi_yoy': cpi_base + np.cumsum(np.random.normal(0, 0.1, days)) / 100,
            'fdi_registered_bn_usd': fdi_base + np.random.normal(0, 0.3, days),
            'fdi_released_bn_usd': fdi_base * 0.8 + np.random.normal(0, 0.3, days),
            'usd_vnd_rate': usd_vnd_base + np.cumsum(np.random.normal(0, 50, days)),
            'overnight_rate_pct': 3.0 + np.random.normal(0, 0.2, days),
            'money_supply_m2': 100 + np.cumsum(np.random.normal(0.1, 0.5, days)),
        }, index=dates)

        macro_df['cpi_yoy'] = macro_df['cpi_yoy'].clip(lower=0, upper=10)

        return self._engineer_features(macro_df)


if __name__ == "__main__":
    print("=" * 60)
    print("Vietnam Macro-Economic Features Engineering")
    print("=" * 60)

    # Initialize
    macro_builder = MacroFeatures(use_fallback=True)

    # Build features
    print("\nFetching macro features for last 2 years...")
    macro_features = macro_builder.build_macro_features(length="2Y")

    print(f"\n✅ Feature matrix shape: {macro_features.shape}")
    print(f"   Columns: {list(macro_features.columns)}")

    # Display sample
    print("\n📊 Recent macro data (last 5 periods):")
    print(macro_features.tail(5))