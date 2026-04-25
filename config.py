# In config.py
import datetime
import pathlib
import os
import numpy as np
from dotenv import load_dotenv

BASE_DIR = pathlib.Path(__file__).parent
load_dotenv(BASE_DIR / ".env")

RAW_PRICES_DIR = BASE_DIR / "data/raw/prices"
RAW_SEC_DIR = BASE_DIR / "data/raw/edgar_downloads"
PROC_EMBED_DIR = BASE_DIR / "data/processed/embeddings"
RESULTS_DIR = BASE_DIR / "results"

CONTEXT_LEN = 256
N_SEC_PCA = 8
PCA_COMPONENTS = 4 
HORIZON = 15
THRESHOLD_MULTIPLIER_PRICE = 0.75
THRESHOLD_MULTIPLIER_DUAL = 0.75
BARRIER_MULTIPLIER = 0.5
TICKERS = ["AAPL"]
START_DATE = "2018-01-01"
END_DATE = datetime.datetime.now().strftime("%Y-%m-%d")
TRAIN_SPLIT_DATE = "2024-01-01"

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")


if not GEMINI_API_KEY:
    print("WARNING: GEMINI_API_KEY not found in environment.")
GEN_MODEL = "gemini-flash-lite-latest"
EMBED_MODEL = "gemini-embedding-001"
CHRONOS_MODEL = "amazon/chronos-2"

def compute_vol_series(df, close_col='Close'):
    return np.log(df[close_col] / df[close_col].shift(1)).rolling(20).std().shift(1)

def calculate_financial_metrics(y_true, y_pred, actual_returns, pred_returns=None):
    import scipy.stats
    strat_returns = np.array(y_pred) * np.array(actual_returns)
    sharpe = (np.mean(strat_returns) / np.std(strat_returns)) * np.sqrt(252) if len(strat_returns) > 0 and np.std(strat_returns) > 0 else 0
    mdd = 0
    if len(strat_returns) > 0:
        wealth_index = np.cumprod(1 + strat_returns)
        previous_peaks = np.maximum.accumulate(wealth_index)
        drawdowns = (wealth_index - previous_peaks) / previous_peaks
        mdd = np.min(drawdowns) if len(drawdowns) > 0 else 0
    ic = 0
    if pred_returns is not None and len(pred_returns) > 1:
        ic, _ = scipy.stats.pearsonr(pred_returns, actual_returns)
    return {'Sharpe': sharpe, 'MDD': mdd, 'IC': ic}

def triple_barrier_label_v2(price_path, initial_price, vol, threshold_multiplier, horizon=HORIZON):
    final_return = (price_path[-1] / initial_price) - 1
    threshold = vol * np.sqrt(horizon) * threshold_multiplier
    if final_return > threshold:
        return 1
    elif final_return < -threshold:
        return -1
    else:
        return 0

