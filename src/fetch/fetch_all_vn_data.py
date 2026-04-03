import pandas as pd
from vnstock import Listing, Quote, Finance, Company

# Get all symbols from HOSE, HNX, and UPCOM
listing = Listing(source="KBS")
all_symbols = listing.all_symbols(to_df=True)

print(all_symbols[:10])  # Print first 10 symbols for verification
print(f"Fetching data for {len(all_symbols)} stocks from HOSE, HNX, and UPCOM...")

# Initialize dataframes
ohlcv_df = pd.DataFrame()
ratios_df = pd.DataFrame()
dividends_df = pd.DataFrame()
industries_df = pd.DataFrame()

for symbol in all_symbols['symbol']:
    try:
        print(f"Processing {symbol}...")

        # Fetch OHLCV data
        quote = Quote(source="KBS", symbol=symbol)
        df_ohlcv = quote.history(start="2023-01-01", end="2024-12-31", interval="1D")
        df_ohlcv['symbol'] = symbol
        ohlcv_df = pd.concat([ohlcv_df, df_ohlcv], ignore_index=True)

        # Fetch financial ratios
        finance = Finance(source="KBS", symbol=symbol)
        ratios = finance.ratio(period="year")
        ratios['symbol'] = symbol
        ratios_df = pd.concat([ratios_df, ratios], ignore_index=True)

        # Fetch dividend history
        dividends = finance.dividend()
        if not dividends.empty:
            dividends['symbol'] = symbol
            dividends_df = pd.concat([dividends_df, dividends], ignore_index=True)

        # Fetch company industry
        company = Company(source="KBS", symbol=symbol)
        overview = company.overview()
        industry = overview.get('industry', 'Unknown')
        industries_df = pd.concat([industries_df, pd.DataFrame({'symbol': [symbol], 'industry': [industry]})], ignore_index=True)

    except Exception as e:
        print(f"Error fetching data for {symbol}: {e}")
        continue

# Save datasets
ohlcv_df.to_csv("/Users/admin/dl4ai/data/all_vn_stocks_ohlcv_2023_2024.csv", index=False)
ratios_df.to_csv("/Users/admin/dl4ai/data/all_vn_stocks_financial_ratios.csv", index=False)
dividends_df.to_csv("/Users/admin/dl4ai/data/all_vn_stocks_dividends.csv", index=False)
industries_df.to_csv("/Users/admin/dl4ai/data/all_vn_stocks_industries.csv", index=False)

print("All data fetched and saved to data/ directory.")
print(f"OHLCV shape: {ohlcv_df.shape}")
print(f"Ratios shape: {ratios_df.shape}")
print(f"Dividends shape: {dividends_df.shape}")
print(f"Industries shape: {industries_df.shape}")