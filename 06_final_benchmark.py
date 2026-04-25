import pandas as pd
import pathlib
import numpy as np
from config import RESULTS_DIR, RAW_PRICES_DIR, TRAIN_SPLIT_DATE

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
        
    # We aggregate metrics across all tickers (Mean)
    summary = pd.DataFrame({name: df.mean(numeric_only=True) for name, df in dfs.items()})
    
    print("\n--- FINAL RESEARCH BENCHMARK ---")
    print(summary.round(4).to_string())
    summary_path = RESULTS_DIR / "final_summary.csv"
    summary.to_csv(summary_path)
    print(f"\nSummary saved to: {summary_path}")

if __name__ == "__main__":
    main()
