import pandas as pd
import os
from pathlib import Path
from tqdm import tqdm

# Directory containing all stock CSV files
data_dir = "/Users/admin/dl4ai/data/data-vn-20230228/stock-historical-data"

# Get all CSV files
csv_files = sorted([f for f in os.listdir(data_dir) if f.endswith(".csv")])
print(f"Found {len(csv_files)} CSV files")

# List to store all dataframes
all_dfs = []

# Read and process each CSV file
for csv_file in tqdm(csv_files, desc="Loading files"):
    try:
        file_path = os.path.join(data_dir, csv_file)
        # Extract symbol from filename (e.g., "VHM-VNINDEX-History.csv" -> "VHM")
        symbol = csv_file.split("-")[0]
        
        # Read CSV file
        df = pd.read_csv(file_path, index_col=0)
        
        # Add symbol column
        df['Symbol'] = symbol
        
        # Add to list
        all_dfs.append(df)
    except Exception as e:
        print(f"Error loading {csv_file}: {e}")
        continue

# Concatenate all dataframes
print("\nCombining all dataframes...")
consolidated_df = pd.concat(all_dfs, ignore_index=True)

# Convert TradingDate to datetime if it's not already
consolidated_df['TradingDate'] = pd.to_datetime(consolidated_df['TradingDate'])

# Sort by TradingDate and Symbol (time series order)
consolidated_df = consolidated_df.sort_values(by=['TradingDate', 'Symbol']).reset_index(drop=True)

# Reorder columns to put Symbol first
cols = ['Symbol'] + [col for col in consolidated_df.columns if col != 'Symbol']
consolidated_df = consolidated_df[cols]

# Save to CSV
output_path = "/Users/admin/dl4ai/data/vn_stock_market_consolidated.csv"
consolidated_df.to_csv(output_path, index=False)

print(f"\nConsolidated dataset created!")
print(f"Total rows: {len(consolidated_df)}")
print(f"Total columns: {len(consolidated_df.columns)}")
print(f"Date range: {consolidated_df['TradingDate'].min()} to {consolidated_df['TradingDate'].max()}")
print(f"Unique symbols: {consolidated_df['Symbol'].nunique()}")
print(f"\nFile saved to: {output_path}")
print(f"File size: {os.path.getsize(output_path) / (1024**2):.2f} MB")

# Display sample
print("\nSample of consolidated data:")
print(consolidated_df.head(10))
