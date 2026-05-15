import pandas as pd
from config_new import RESULTS_DIR


def print_formatted_comparison(df: pd.DataFrame):
    """Print a beautifully formatted benchmark comparison table matching 08 style."""
    
    print("\n" + "=" * 130)
    print("FINAL BENCHMARK COMPARISON")
    print("=" * 130)
    print("\nPerformance metrics aggregated across all tickers and test periods\n")
    
    print("-" * 130)
    print(f"{'Model':<30} {'Sharpe':>12} {'MaxDD':>12} {'Mean IC':>12} "
          f"{'Std IC':>10} {'IC IR':>12} {'Mean RankIC':>14} "
          f"{'Hit Rate':>12} {'Turnover':>12} {'N Dates':>10}")
    print("-" * 130)
    
    for _, row in df.iterrows():
        print(
            f"{row['Model']:<30} "
            f"{row['Sharpe']:>12.6f} "
            f"{row['MaxDD']:>12.6f} "
            f"{row['Mean IC']:>12.6f} "
            f"{row['Std IC']:>10.6f} "
            f"{row['IC IR']:>12.6f} "
            f"{row['Mean RankIC']:>14.6f} "
            f"{row['Hit Rate']:>12.6f} "
            f"{row['Turnover']:>12.6f} "
            f"{int(row['N Dates']):>10}"
        )
    
    print("-" * 130)
    
    if len(df) == 2:
        price_row = df[df['Model'] == 'Model 1 (Price Only)'].iloc[0]
        dual_row = df[df['Model'] == 'Model 2 (Dual Fusion)'].iloc[0]
        
        delta_sharpe = dual_row['Sharpe'] - price_row['Sharpe']
        delta_mdd = dual_row['MaxDD'] - price_row['MaxDD']
        delta_ic = dual_row['Mean IC'] - price_row['Mean IC']
        delta_std_ic = dual_row['Std IC'] - price_row['Std IC']
        delta_ic_ir = dual_row['IC IR'] - price_row['IC IR']
        delta_rankic = dual_row['Mean RankIC'] - price_row['Mean RankIC']
        delta_hitrate = dual_row['Hit Rate'] - price_row['Hit Rate']
        delta_turnover = dual_row['Turnover'] - price_row['Turnover']
        
        print(f"\n{'Improvement (Dual - Price):':<30} "
              f"{delta_sharpe:>+12.6f} "
              f"{delta_mdd:>+12.6f} "
              f"{delta_ic:>+12.6f} "
              f"{delta_std_ic:>+10.6f} "
              f"{delta_ic_ir:>+12.6f} "
              f"{delta_rankic:>+14.6f} "
              f"{delta_hitrate:>+12.6f} "
              f"{delta_turnover:>+12.6f} "
              f"{'':>10}")
    
    print("\n" + "=" * 130)


def main():
    csv_path = RESULTS_DIR / "final_benchmark_comparison.csv"
    
    if csv_path.exists():
        print(f"Loading existing comparison from {csv_path}")
        comparison = pd.read_csv(csv_path)
    else:
        price_path = RESULTS_DIR / "price_only_summary.csv"
        dual_path = RESULTS_DIR / "dual_fusion_summary.csv"
        
        if not price_path.exists():
            raise FileNotFoundError(f"Missing {price_path}. Run 07_forward_test_price_only.py first.")
        if not dual_path.exists():
            raise FileNotFoundError(f"Missing {dual_path}. Run 07_forward_test_dual_fusion.py first.")
        
        price = pd.read_csv(price_path)
        dual = pd.read_csv(dual_path)
        
        comparison = pd.concat([price, dual], ignore_index=True)
        
        comparison['Model'] = ['Model 1 (Price Only)', 'Model 2 (Dual Fusion)']
        
        column_renames = {
            'Sharpe_annual': 'Sharpe',
            'MDD': 'MaxDD',
            'IC': 'Mean IC',
            'IC_std': 'Std IC',
            'IC_IR': 'IC IR',
            'RankIC': 'Mean RankIC',
            'hit_rate': 'Hit Rate',
            'turnover': 'Turnover',
            'N': 'N Dates'
        }
        comparison = comparison.rename(columns=column_renames)
        
        cols = ['Model'] + [c for c in comparison.columns if c != 'Model']
        comparison = comparison[cols]
        
        comparison.to_csv(csv_path, index=False)
    
    print_formatted_comparison(comparison)
    
    print(f"\nComparison data saved to {csv_path}")


if __name__ == "__main__":
    main()