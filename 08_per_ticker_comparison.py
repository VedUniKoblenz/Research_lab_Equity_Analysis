import pandas as pd
import numpy as np
from pathlib import Path
from config_new import RESULTS_DIR, TICKERS
from typing import Dict, List, Tuple

# Sector mappings
SECTOR_MAPPING = {
    # Technology (20 stocks)
    "AAPL": "Technology", "MSFT": "Technology", "GOOGL": "Technology",
    "META": "Technology", "NVDA": "Technology", "AMD": "Technology",
    "AMZN": "Technology", "INTC": "Technology", "CSCO": "Technology",
    "ORCL": "Technology", "IBM": "Technology", "ADBE": "Technology",
    "CRM": "Technology", "NFLX": "Technology", "QCOM": "Technology",
    "TXN": "Technology", "AVGO": "Technology", "ACN": "Technology",
    "NOW": "Technology", "SNOW": "Technology",
    
    # Finance (15 stocks)
    "JPM": "Finance", "GS": "Finance", "BAC": "Finance",
    "C": "Finance", "WFC": "Finance", "MS": "Finance",
    "BLK": "Finance", "V": "Finance", "MA": "Finance",
    "PYPL": "Finance", "AXP": "Finance", "COF": "Finance",
    "USB": "Finance", "PNC": "Finance", "SCHW": "Finance",
    
    # Healthcare (14 stocks)
    "PFE": "Healthcare", "JNJ": "Healthcare", "MRK": "Healthcare",
    "ABBV": "Healthcare", "AMGN": "Healthcare", "GILD": "Healthcare",
    "BIIB": "Healthcare", "REGN": "Healthcare", "VRTX": "Healthcare",
    "BMY": "Healthcare", "LLY": "Healthcare", "UNH": "Healthcare",
    "CVS": "Healthcare", "CI": "Healthcare",
    
    # Energy (10 stocks)
    "XOM": "Energy", "CVX": "Energy", "COP": "Energy",
    "EOG": "Energy", "SLB": "Energy", "PSX": "Energy",
    "VLO": "Energy", "MPC": "Energy", "OXY": "Energy",
    "HAL": "Energy",
    
    # Consumer (15 stocks)
    "KO": "Consumer", "PG": "Consumer", "COST": "Consumer",
    "WMT": "Consumer", "TGT": "Consumer", "HD": "Consumer",
    "LOW": "Consumer", "NKE": "Consumer", "SBUX": "Consumer",
    "MCD": "Consumer", "DIS": "Consumer", "PEP": "Consumer",
    "PM": "Consumer", "MO": "Consumer", "CL": "Consumer",
    
    # Industrials (10 stocks)
    "BA": "Industrials", "CAT": "Industrials", "GE": "Industrials",
    "HON": "Industrials", "MMM": "Industrials", "LMT": "Industrials",
    "RTX": "Industrials", "UPS": "Industrials", "FDX": "Industrials",
    "DE": "Industrials",
    
    # Telecommunications (5 stocks)
    "T": "Telecommunications", "VZ": "Telecommunications", "TMUS": "Telecommunications",
    "CHTR": "Telecommunications", "CMCSA": "Telecommunications",
    
    # Utilities (5 stocks)
    "NEE": "Utilities", "DUK": "Utilities", "SO": "Utilities",
    "D": "Utilities", "EXC": "Utilities",
    
    # Real Estate / REITs (5 stocks)
    "AMT": "Real Estate", "PLD": "Real Estate", "CCI": "Real Estate",
    "EQIX": "Real Estate", "SPG": "Real Estate",
    
    # Materials (5 stocks)
    "LIN": "Materials", "APD": "Materials", "SHW": "Materials",
    "FCX": "Materials", "NEM": "Materials",
    
    # Transportation (5 stocks)
    "UNP": "Transportation", "CSX": "Transportation", "NSC": "Transportation",
    "ODFL": "Transportation", "DAL": "Transportation",
}

# Define sector order for consistent output
SECTOR_ORDER = [
    "Technology", "Finance", "Healthcare", "Energy", "Consumer",
    "Industrials", "Telecommunications", "Utilities", "Real Estate", "Materials", "Transportation"
]

def load_per_ticker_results() -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Load per-ticker results for both models."""
    price_path = RESULTS_DIR / "price_only_per_ticker.csv"
    dual_path = RESULTS_DIR / "dual_fusion_per_ticker.csv"
    
    if not price_path.exists():
        raise FileNotFoundError(f"Missing {price_path}. Run 07_forward_test_price_only.py first.")
    if not dual_path.exists():
        raise FileNotFoundError(f"Missing {dual_path}. Run 07_forward_test_dual_fusion.py first.")
    
    price_df = pd.read_csv(price_path)
    dual_df = pd.read_csv(dual_path)
    
    return price_df, dual_df

def prepare_comparison_data(price_df: pd.DataFrame, dual_df: pd.DataFrame) -> pd.DataFrame:
    """
    Prepare comparison data by merging Price-Only and Dual Fusion results.
    Returns a DataFrame with columns for the comparison table.
    """
    # Add model identifier
    price_df = price_df.copy()
    price_df['Model'] = 'Price-Only'
    
    dual_df = dual_df.copy()
    dual_df['Model'] = 'Dual Fusion'
    
    # Combine data
    combined = pd.concat([price_df, dual_df], ignore_index=True)
    
    # Add sector information
    combined['Sector'] = combined['ticker'].map(SECTOR_MAPPING)
    
    # Filter only tickers that have sector mappings
    combined = combined.dropna(subset=['Sector'])
    
    return combined

def format_comparison_table(combined_df: pd.DataFrame) -> str:
    """
    Format the comparison data into the target table format.
    """
    lines = []
    
    # Group by sector and ticker
    sectors = [s for s in SECTOR_ORDER if s in combined_df['Sector'].values]
    
    for sector in sectors:
        sector_data = combined_df[combined_df['Sector'] == sector]
        tickers_in_sector = sorted(sector_data['ticker'].unique())
        
        # Add sector header
        lines.append(f"{'Sector':<15} {'Ticker':<8} {'Model':<15} {'Mean IC':>8} "
                     f"{'Mean RankIC':>12} "
                     f"{'Hit Rate':>10} {'Sharpe Ratio':>12} {'Max Drawdown':>12}")
        lines.append(f"{sector}")
        lines.append("-" * 120)
        
        for ticker in tickers_in_sector:
            ticker_data = sector_data[sector_data['ticker'] == ticker]
            
            # Get Price-Only metrics
            price_data = ticker_data[ticker_data['Model'] == 'Price-Only']
            dual_data = ticker_data[ticker_data['Model'] == 'Dual Fusion']
            
            if not price_data.empty:
                price_row = price_data.iloc[0]
                # Format Price-Only row
                lines.append(
                    f"{'':<15} {ticker:<8} {'Price-Only':<15} "
                    f"{price_row.get('IC', np.nan):>8.3f} "
                    f"{price_row.get('RankIC', np.nan):>12.3f} "
                    f"{price_row.get('hit_rate', 0):>10.2f} "
                    f"{price_row.get('Sharpe_annual', np.nan):>12.2f} "
                    f"{price_row.get('MDD', 0):>12.2f}"
                )
            
            if not dual_data.empty:
                dual_row = dual_data.iloc[0]
                # Format Dual Fusion row
                lines.append(
                    f"{'':<15} {ticker:<8} {'Dual Fusion':<15} "
                    f"{dual_row.get('IC', np.nan):>8.3f} "
                    f"{dual_row.get('RankIC', np.nan):>12.3f} "
                    f"{dual_row.get('hit_rate', 0):>10.2f} "
                    f"{dual_row.get('Sharpe_annual', np.nan):>12.2f} "
                    f"{dual_row.get('MDD', 0):>12.2f}"
                )
        
        lines.append("")  # Empty line between sectors
    
    return "\n".join(lines)

def create_structured_comparison(price_df: pd.DataFrame, dual_df: pd.DataFrame) -> pd.DataFrame:
    """
    Create a structured DataFrame with the exact format needed for the output.
    """
    records = []
    
    # Combine data with model labels
    price_df = price_df.copy()
    dual_df = dual_df.copy()
    
    price_df['Model'] = 'Price-Only'
    dual_df['Model'] = 'Dual Fusion'
    
    combined = pd.concat([price_df, dual_df], ignore_index=True)
    combined['Sector'] = combined['ticker'].map(SECTOR_MAPPING)
    combined = combined.dropna(subset=['Sector'])
    
    # Use columns that actually exist
    for _, row in combined.iterrows():
        records.append({
            'Sector': row['Sector'],
            'Ticker': row['ticker'],
            'Model': row['Model'],
            'Mean IC': row.get('IC', np.nan),
            'Mean RankIC': row.get('RankIC', np.nan),
            'Hit Rate': row.get('hit_rate', np.nan),
            'Sharpe Ratio': row.get('Sharpe_annual', np.nan),
            'Max Drawdown': row.get('MDD', np.nan),
        })
    
    result_df = pd.DataFrame(records)
    
    # Sort by sector order then ticker
    result_df['Sector_Order'] = result_df['Sector'].map({s: i for i, s in enumerate(SECTOR_ORDER)})
    result_df = result_df.sort_values(['Sector_Order', 'Ticker', 'Model']).drop('Sector_Order', axis=1)
    
    return result_df

def print_formatted_table(structured_df: pd.DataFrame):
    """
    Print the comparison table with:
    - Sector shown once per block (centered)
    - Ticker shown once per pair (centered)
    - Extra spacing between tickers
    - Wider sector column for long names
    """
    SECTOR_WIDTH = 22  
    TICKER_WIDTH = 10

    print("\n" + "=" * 110)
    print("MODEL COMPARISON BY SECTOR")
    print("=" * 110)

    for sector, sector_df in structured_df.groupby('Sector', sort=False):

        # Header per sector
        print("\n" + "-" * 110)
        print(f"{'Sector':<{SECTOR_WIDTH}} {'Ticker':<{TICKER_WIDTH}} {'Model':<15} {'Mean IC':>8} "
              f"{'Mean RankIC':>12} {'Hit Rate':>10} "
              f"{'Sharpe Ratio':>12} {'Max Drawdown':>12}")
        print("-" * 110)

        tickers = list(sector_df['Ticker'].unique())
        total_rows = len(sector_df)

        row_counter = 0

        for ticker in tickers:
            ticker_df = sector_df[sector_df['Ticker'] == ticker]

            n_rows = len(ticker_df)
            mid_ticker_idx = n_rows // 2

            for i, (_, row) in enumerate(ticker_df.iterrows()):

                if row_counter == total_rows // 2:
                    sector_str = f"{sector:<{SECTOR_WIDTH}}"
                else:
                    sector_str = " " * SECTOR_WIDTH

                if i == mid_ticker_idx:
                    ticker_str = f"{ticker:<{TICKER_WIDTH}}"
                else:
                    ticker_str = " " * TICKER_WIDTH

                print(f"{sector_str} {ticker_str} {row['Model']:<15} "
                      f"{row['Mean IC']:>8.3f} "
                      f"{row['Mean RankIC']:>12.3f} {row['Hit Rate']:>10.2f} "
                      f"{row['Sharpe Ratio']:>12.2f} "
                      f"{row['Max Drawdown']:>12.2f}")

                row_counter += 1

            print()

    print("\n" + "=" * 110)

def main():
    """Main function to generate the comparison table."""
    
    print("Loading per-ticker results...")
    price_df, dual_df = load_per_ticker_results()
    
    print("Creating structured comparison...")
    structured_df = create_structured_comparison(price_df, dual_df)
    
    print_formatted_table(structured_df)
    
    output_path = RESULTS_DIR / "model_comparison_by_sector.csv"
    structured_df.to_csv(output_path, index=False)
    print(f"\nStructured comparison saved to {output_path}")

if __name__ == "__main__":
    main()