import pandas as pd
import pathlib
import numpy as np
from config_new import RESULTS_DIR, RAW_PRICES_DIR, TRAIN_SPLIT_DATE

def main():
    m1_path = RESULTS_DIR / "price_only_results.csv"
    m2_path = RESULTS_DIR / "dual_fusion_results.csv"
    
    dfs = {}
    if m1_path.exists(): 
        dfs['Model 1 (Price Only)'] = pd.read_csv(m1_path).set_index('ticker')
    if m2_path.exists(): 
        dfs['Model 2 (Dual Fusion)'] = pd.read_csv(m2_path).set_index('ticker')
    
    if not dfs:
        print("No results found. Please run 04_model_price_only.py and 05_model_dual_fusion.py first.")
        return

    print("\n--- FINAL RESEARCH BENCHMARK ---")

    all_tickers = sorted(set().union(*(df.index for df in dfs.values())))
    long_rows = []

    for ticker in all_tickers:
        print(f"\nTicker: {ticker}")

        ticker_tables = []
        for model_name, df in dfs.items():
            if ticker not in df.index:
                continue

            row = df.loc[ticker].copy()
            row.name = model_name
            ticker_tables.append(row)

            row_dict = row.to_dict()
            row_dict["ticker"] = ticker
            row_dict["model"] = model_name
            long_rows.append(row_dict)

        if not ticker_tables:
            print("No results found for this ticker.")
            continue

        ticker_summary = pd.DataFrame(ticker_tables)
        ticker_summary.index.name = None

        # Match the requested layout: model names as row labels, metrics as columns.
        if 'Balanced_Accuracy' in ticker_summary.columns:
            ticker_summary = ticker_summary.drop(columns=['Balanced_Accuracy'])

        print(ticker_summary.round(4).to_string())

    summary_path = RESULTS_DIR / "final_summary.csv"
    if long_rows:
        pd.DataFrame(long_rows).to_csv(summary_path, index=False)
        print(f"\nSummary saved to: {summary_path}")
    else:
        print("\nNo rows available to save.")

if __name__ == "__main__":
    main()
