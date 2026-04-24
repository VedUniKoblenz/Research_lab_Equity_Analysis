import pandas as pd
import pathlib
import numpy as np
from config import RESULTS_DIR, RAW_PRICES_DIR, TRAIN_SPLIT_DATE

def get_bh_metrics(ticker):
    """
    Calculates the 'Buy & Hold' benchmark for the test period.
    This provides a baseline: What if we just bought the stock and did nothing?
    """
    p_file = RAW_PRICES_DIR / f"{ticker}.parquet"
    if not p_file.exists(): return 0, 0
    df = pd.read_parquet(p_file)
    
    # Standardize 'Date' column location
    if 'Date' not in df.columns:
        df = df.reset_index()
    if 'Date' not in df.columns:
        for col in df.columns:
            if pd.api.types.is_datetime64_any_dtype(df[col]):
                df = df.rename(columns={col: 'Date'})
                break
    
    if 'Date' not in df.columns: return 0, 0

    close_col = 'Adj Close' if 'Adj Close' in df.columns else 'Close'
    df['Date'] = pd.to_datetime(df['Date'])
    test_df = df[df['Date'] >= pd.to_datetime(TRAIN_SPLIT_DATE)]
    
    # Calculate daily returns during the TEST window
    rets = test_df[close_col].pct_change().dropna()
    if len(rets) < 2 or rets.std() == 0: return 0, 0
    
    # Annualized Sharpe: (Mean / Std) * Sqrt(252 Trading Days)
    sharpe = (rets.mean() / rets.std()) * np.sqrt(252)
    
    # Max Drawdown calculation: Peak-to-Trough decline
    wealth_index = np.cumprod(1 + rets)
    previous_peaks = np.maximum.accumulate(wealth_index)
    drawdowns = (wealth_index - previous_peaks) / previous_peaks
    mdd = np.min(drawdowns)
    return sharpe, mdd

def main():
    m1_path = RESULTS_DIR / "price_only_results.csv"
    m2_path = RESULTS_DIR / "dual_fusion_results.csv"
    
    dfs = {}
    if m1_path.exists(): 
        dfs['Model 1 (Price Only)'] = pd.read_csv(m1_path).set_index('ticker')
    if m2_path.exists(): 
        dfs['Model 2 (Dual Fusion)'] = pd.read_csv(m2_path).set_index('ticker')
    
    if not dfs:
        print("No results found. Please run 06_model_price_only.py and 07_model_dual_fusion.py first.")
        return
        
    # We aggregate metrics across all tickers (Mean)
    summary = pd.DataFrame({name: df.mean(numeric_only=True) for name, df in dfs.items()})
    
    # Add B&H for reference
    bh_sharpe, bh_mdd = get_bh_metrics("AAPL")
    summary.loc['Sharpe', 'Buy & Hold'] = bh_sharpe
    summary.loc['MDD', 'Buy & Hold'] = bh_mdd

    print("\n--- FINAL RESEARCH BENCHMARK ---")
    
    # Create a display version of the summary
    display_summary = summary.drop(columns=['Buy & Hold'], errors='ignore').copy()
    display_summary = display_summary.round(4).astype(object)
    
    # Format percentage metrics
    pct_metrics = ['Accuracy', 'MDD']
    for m in pct_metrics:
        if m in display_summary.index:
            display_summary.loc[m] = display_summary.loc[m].apply(lambda x: f"{x*100:.2f}%" if pd.notnull(x) else x)
    
    # Manual table formatter (zero-dependency replacement for tabulate)
    def format_grid(df):
        rows = [[""] + list(df.columns)]
        for idx, row in df.iterrows():
            rows.append([idx] + list(row))
        widths = [max(len(str(r[i])) for r in rows) + 2 for i in range(len(rows[0]))]
        def make_line(left, mid, right, char): return left + mid.join(char * w for w in widths) + right
        top, sep, bot = make_line("╔", "╦", "╗", "═"), make_line("╠", "╬", "╣", "═"), make_line("╚", "╩", "╝", "═")
        lines = [top]
        for i, row in enumerate(rows):
            lines.append("║" + "║".join(str(val).center(widths[j]) for j, val in enumerate(row)) + "║")
            if i == 0: lines.append(sep)
            elif i < len(rows) - 1: lines.append(make_line("╟", "╫", "╢", "─"))
        lines.append(bot)
        return "\n".join(lines)

    print("\n" + format_grid(display_summary))
    summary_path = RESULTS_DIR / "final_summary.csv"
    summary.to_csv(summary_path)
    print(f"\nSummary saved to: {summary_path}")

if __name__ == "__main__":
    main()
