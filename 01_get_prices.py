import pandas as pd
import yfinance as yf
from config_new import RAW_PRICES_DIR, TICKERS, START_DATE, END_DATE

def main():
    print(f"Downloading prices for {TICKERS}...")
    RAW_PRICES_DIR.mkdir(parents=True, exist_ok=True)

    data = yf.download(TICKERS, start=START_DATE, end=END_DATE, auto_adjust=True, group_by='ticker')
    for t in TICKERS:
        if len(TICKERS) > 1:
            ticker_data = data[t].copy()
        else:    
            if isinstance(data.columns, pd.MultiIndex):
                ticker_data = data[t].copy() if t in data.columns.levels[0] else data.copy()
            else:
                ticker_data = data.copy()
        
        if isinstance(ticker_data.columns, pd.MultiIndex):
            ticker_data.columns = ticker_data.columns.get_level_values(-1)
            
        ticker_data.dropna(how='all').to_parquet(RAW_PRICES_DIR / f"{t}.parquet")
    print("Prices downloaded to data/raw/prices.")

if __name__ == "__main__":
    main()
