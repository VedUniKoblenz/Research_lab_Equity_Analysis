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
PROC_NEWS_DIR = BASE_DIR / "data/processed/news_embeddings"
RESULTS_DIR = BASE_DIR / "results"

CONTEXT_LEN = 512
N_SEC_PCA = 16
HORIZON = 15
THRESHOLD_MULTIPLIER_PRICE = 0.15
THRESHOLD_MULTIPLIER_DUAL = 0.15
BARRIER_MULTIPLIER = 0.5
TICKERS = [
    
    # Technology (20 stocks)
    "AAPL", "MSFT", "GOOGL", "META", "NVDA", "AMD", "AMZN", "INTC", "CSCO", "ORCL",
    "IBM", "ADBE", "CRM", "NFLX", "QCOM", "TXN", "AVGO", "ACN", "NOW", "SNOW",
    
    # Finance (15 stocks)
    "JPM", "GS", "BAC", "C", "WFC", "MS", "BLK", "V", "MA", "PYPL",
    "AXP", "COF", "USB", "PNC", "SCHW",
    
    # Healthcare (14 stocks)
    "PFE", "JNJ", "MRK", "ABBV", "AMGN", "GILD", "BIIB", "REGN", "VRTX", "BMY",
    "LLY", "UNH", "CVS", "CI",
    
    # Energy (10 stocks)
    "XOM", "CVX", "COP", "EOG", "SLB", "PSX", "VLO", "MPC", "OXY", "HAL",
    
    # Consumer Discretionary & Staples (15 stocks)
    "KO", "PG", "COST", "WMT", "TGT", "HD", "LOW", "NKE", "SBUX", "MCD",
    "DIS", "PEP", "PM", "MO", "CL",
    
    # Industrials (10 stocks)
    "BA", "CAT", "GE", "HON", "MMM", "LMT", "RTX", "UPS", "FDX", "DE",
    
    # Telecommunications (5 stocks)
    "T", "VZ", "TMUS", "CHTR", "CMCSA",
    
    # Utilities (5 stocks)
    "NEE", "DUK", "SO", "D", "EXC",
    
    # Real Estate (REITs) (5 stocks)
    "AMT", "PLD", "CCI", "EQIX", "SPG",
    
    # Materials (5 stocks)
    "LIN", "APD", "SHW", "FCX", "NEM",
    
    # Transportation (5 stocks)
    "UNP", "CSX", "NSC", "ODFL", "DAL",
    
]

START_DATE = "2018-01-01"
END_DATE = datetime.datetime.now().strftime("%Y-%m-%d")
TRAIN_SPLIT_DATE = "2024-01-01"

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")


if not GEMINI_API_KEY:
    print("WARNING: GEMINI_API_KEY not found in environment.")
GEN_MODEL = "gemini-flash-latest"
EMBED_MODEL = "gemini-embedding-2"
CHRONOS_MODEL = "amazon/chronos-2"