import pandas as pd
import numpy as np
from pathlib import Path

RESULTS_DIR = Path("results")

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

SECTOR_ORDER = [
    "Technology", "Finance", "Healthcare", "Energy",
    "Consumer", "Industrials", "Telecommunications", "Utilities", 
    "Real Estate", "Materials", "Transportation"
]

SECTOR_WIDTH = 22
COL_WIDTHS = {
    'Avg ΔIC': 12,
    'Avg ΔRankIC': 14,
    'Avg ΔHitRate': 14,
    'Avg ΔSharpe': 14,
    'Avg ΔMDD': 12
}

def load_results():
    price_df = pd.read_csv(RESULTS_DIR / "price_only_per_ticker.csv")
    dual_df = pd.read_csv(RESULTS_DIR / "dual_fusion_per_ticker.csv")
    return price_df, dual_df

def build_sector_table(price_df, dual_df):
    price_df = price_df.copy()
    dual_df = dual_df.copy()
    price_df["Model"] = "price only"
    dual_df["Model"] = "dual fusion"

    combined = pd.concat([price_df, dual_df], ignore_index=True)
    combined["Sector"] = combined["ticker"].map(SECTOR_MAPPING)
    combined = combined.dropna(subset=["Sector"])

    pivot = combined.pivot_table(
        index=["Sector", "ticker"],
        columns="Model",
        values=["IC", "RankIC", "hit_rate", "Sharpe_annual", "MDD"],
        aggfunc="mean",
    )

    improvements = pd.DataFrame({
        "IC_imp": pivot["IC"]["dual fusion"] - pivot["IC"]["price only"],
        "Sharpe_imp": pivot["Sharpe_annual"]["dual fusion"] - pivot["Sharpe_annual"]["price only"],
        "RankIC_imp": pivot["RankIC"]["dual fusion"] - pivot["RankIC"]["price only"],
        "HitRate_imp": pivot["hit_rate"]["dual fusion"] - pivot["hit_rate"]["price only"],
        "MDD_imp": pivot["MDD"]["dual fusion"] - pivot["MDD"]["price only"],
    }).reset_index().dropna()

    improvements["dual_better_ic"] = (improvements["IC_imp"] > 0).astype(int)
    improvements["dual_better_sharpe"] = (improvements["Sharpe_imp"] > 0).astype(int)
    improvements["dual_better_rankic"] = (improvements["RankIC_imp"] > 0).astype(int)

    sector_summary = improvements.groupby("Sector", as_index=True).agg(
        Avg_Delta_IC=("IC_imp", "mean"),
        Avg_Delta_Sharpe=("Sharpe_imp", "mean"),
        Avg_Delta_RankIC=("RankIC_imp", "mean"),
        Avg_Delta_HitRate=("HitRate_imp", "mean"),
        Avg_Delta_MDD=("MDD_imp", "mean"),
    )

    rows = []
    for sector in SECTOR_ORDER:
        if sector not in sector_summary.index:
            continue
        s = sector_summary.loc[sector]
        rows.append({
            "Sector": sector,
            "Avg ΔIC": s["Avg_Delta_IC"],
            "Avg ΔRankIC": s["Avg_Delta_RankIC"],
            "Avg ΔHitRate": s["Avg_Delta_HitRate"],
            "Avg ΔSharpe": s["Avg_Delta_Sharpe"],
            "Avg ΔMDD": s["Avg_Delta_MDD"],
        })
    return pd.DataFrame(rows)

def print_formatted_table(df):
    """Print a beautifully formatted sector comparison table."""
    
    print("\n" + "=" * 95)
    print("SECTOR COMPARISON: DUAL FUSION VS PRICE-ONLY")
    print("(Positive Δ indicates improvement with Dual Fusion)")
    print("=" * 95)
    
    # Build header with proper alignment
    header = (
        f"{'Sector':<{SECTOR_WIDTH}} "
        f"{'Avg ΔIC':>{COL_WIDTHS['Avg ΔIC']}} "
        f"{'Avg ΔRankIC':>{COL_WIDTHS['Avg ΔRankIC']}} "
        f"{'Avg ΔHitRate':>{COL_WIDTHS['Avg ΔHitRate']}} "
        f"{'Avg ΔSharpe':>{COL_WIDTHS['Avg ΔSharpe']}} "
        f"{'Avg ΔMDD':>{COL_WIDTHS['Avg ΔMDD']}}"
    )
    
    print("\n" + "-" * 95)
    print(header)
    print("-" * 95)
    
  
    for _, row in df.iterrows():
        ic_val = row['Avg ΔIC']
        sharpe_val = row['Avg ΔSharpe']
        
        ic_str = f"{ic_val:>+{COL_WIDTHS['Avg ΔIC']}.3f}"
        rankic_str = f"{row['Avg ΔRankIC']:>+{COL_WIDTHS['Avg ΔRankIC']}.3f}"
        hitrate_str = f"{row['Avg ΔHitRate']:>+{COL_WIDTHS['Avg ΔHitRate']}.3f}"
        sharpe_str = f"{sharpe_val:>+{COL_WIDTHS['Avg ΔSharpe']}.3f}"
        mdd_str = f"{row['Avg ΔMDD']:>+{COL_WIDTHS['Avg ΔMDD']}.3f}"
        
        print(
            f"{row['Sector']:<{SECTOR_WIDTH}} "
            f"{ic_str} "
            f"{rankic_str} "
            f"{hitrate_str} "
            f"{sharpe_str} "
            f"{mdd_str}"
        )
    
    print("-" * 95)

def main():
    price_df, dual_df = load_results()
    sector_df = build_sector_table(price_df, dual_df)
    print_formatted_table(sector_df)

    # Optional save
    sector_df.to_csv(RESULTS_DIR / "sector_comparison_summary.csv", index=False)
    print(f"\nData saved to {RESULTS_DIR / 'sector_comparison_summary.csv'}")

if __name__ == "__main__":
    main()