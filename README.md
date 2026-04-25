# Research_lab_Equity_Analysis

A research pipeline that forecasts stock price direction using two approaches: a **price-only baseline** (Chronos-2 time-series transformer) and a **dual-fusion model** that augments price history with SEC 10-Q filing embeddings compressed via PCA.

---

## Overview

This project investigates whether fundamental information from SEC filings can meaningfully improve short-term stock forecasting when fused with a pre-trained time-series model. Signals are evaluated using the triple-barrier labeling method and standard financial metrics (Sharpe ratio, MDD, IC).

```
01_get_prices.py       → Download historical price data (Yahoo Finance)
02_get_sec.py          → Download 10-Q filings (SEC EDGAR)
03_process_sec.py      → Parse filings → Gemini AI → embeddings (.npy)
create_pca.py          → Fit PCA on embeddings, save reduced features
04_model_price_only.py → Baseline: Chronos-2 on prices only
05_model_dual_fusion.py → Dual-fusion: Chronos-2 + SEC PCA features
06_final_benchmark.py  → Aggregate and compare model results
```

---

## Architecture

### Model 1 — Price Only (Baseline)
Uses the [Chronos-2](https://huggingface.co/amazon/chronos-2) pre-trained time-series transformer to forecast a 15-day price path from a 256-day closing price context. A trading signal (buy / sell / hold) is generated when the predicted return exceeds a volatility-scaled threshold.

### Model 2 — Dual Fusion
Extends the baseline by prepending a **fundamental pseudo-history** derived from SEC filing embeddings to the price context. The most recent 10-Q filing (before the forecast date, to avoid lookahead bias) is retrieved, embedded via Gemini, compressed to `N_SEC_PCA` PCA components, and converted into a synthetic price trend that is concatenated with real prices before being passed to Chronos.

```
SEC Filing Text
    → Gemini (generate_content: momentum factors)
    → Gemini (embed_content)
    → PCA (8 components)
    → Anchor-scaled pseudo-history
    → [pseudo-history | real price context]
    → Chronos-2
    → Trading signal
```

---

## Project Structure

```
.
├── config.py                  # All paths, constants, shared helpers
├── covariate_builder.py       # SEC feature retrieval + covariate matrix builder
├── create_pca.py              # PCA fitting on SEC embeddings
├── 01_get_prices.py
├── 02_get_sec.py
├── 03_process_sec.py
├── 04_model_price_only.py
├── 05_model_dual_fusion.py
├── 06_final_benchmark.py
├── data/
│   ├── raw/
│   │   ├── prices/            # {TICKER}.parquet
│   │   └── edgar_downloads/   # Raw SEC filings
│   └── processed/
│       └── embeddings/        # {ticker}_{filing}.npy, manifest.csv, sec_pca.joblib
├── results/
│   ├── price_only_results.csv
│   ├── dual_fusion_results.csv
│   └── final_summary.csv
└── .env
```

---

## Setup

### Prerequisites

- Python 3.9+
- A [Gemini API key](https://aistudio.google.com/app/apikey)

### Installation

```bash
git clone https://github.com/your-org/your-repo.git
cd your-repo
pip install -r requirements.txt
```

**Key dependencies:**

| Package | Purpose |
|---|---|
| `yfinance` | Price data download |
| `sec-edgar-downloader` | SEC 10-Q filing download |
| `google-genai` | Gemini embedding + generation |
| `scikit-learn` | PCA |
| `chronos` (amazon/chronos-2) | Time-series forecasting |
| `torch` | Model inference |
| `joblib` | PCA model serialization |
| `beautifulsoup4`, `lxml` | HTML filing parsing |

### Environment Variables

Create a `.env` file in the project root:

```env
GEMINI_API_KEY=your_api_key_here
```

---

## Configuration

All settings are centralized in `config.py`:

| Constant | Default | Description |
|---|---|---|
| `TICKERS` | `["AAPL"]` | Stocks to analyze |
| `START_DATE` | `"2018-01-01"` | Price history start |
| `TRAIN_SPLIT_DATE` | `"2024-01-01"` | Train / test boundary |
| `CONTEXT_LEN` | `256` | Days of price history fed to Chronos |
| `HORIZON` | `15` | Forecast horizon (trading days) |
| `N_SEC_PCA` | `8` | PCA components from SEC embeddings |
| `THRESHOLD_MULTIPLIER_PRICE` | `0.75` | Conviction threshold (price-only model) |
| `THRESHOLD_MULTIPLIER_DUAL` | `0.75` | Conviction threshold (dual-fusion model) |
| `CHRONOS_MODEL` | `"amazon/chronos-2"` | HuggingFace model ID |
| `EMBED_MODEL` | `"gemini-embedding-001"` | Gemini embedding model |
| `GEN_MODEL` | `"gemini-flash-lite-latest"` | Gemini generation model |

---

## Usage

Run each step in order:

```bash
# 1. Download price data
python 01_get_prices.py

# 2. Download SEC 10-Q filings
python 02_get_sec.py

# 3. Process filings into embeddings (requires GEMINI_API_KEY)
python 03_process_sec.py

# 4. Fit PCA on embeddings
python create_pca.py

# 5. Run price-only baseline
python 04_model_price_only.py

# 6. Run dual-fusion model
python 05_model_dual_fusion.py

# 7. Compare results
python 06_final_benchmark.py
```

Results are saved to `results/final_summary.csv`.

---

## Evaluation Metrics

| Metric | Description |
|---|---|
| **Accuracy** | Fraction of signals matching triple-barrier labels |
| **Sharpe** | Annualized Sharpe ratio of the signal-weighted strategy |
| **MDD** | Maximum drawdown of cumulative strategy returns |
| **IC** | Information Coefficient (Pearson correlation of predicted vs. actual returns) |
| **RMSE** | Root mean square error of price path forecast |
| **N_Trades** | Number of non-zero (active) signals |
| **Avg_Conviction** | Mean absolute predicted return across all time steps |

### Triple-Barrier Labeling

Each sample is labeled using a simplified triple-barrier method: if the realized return over `HORIZON` days exceeds `vol × √HORIZON × threshold_multiplier`, the label is **1** (up); if below the negative threshold, **−1** (down); otherwise **0** (neutral).

---

## Lookahead Bias Prevention

- Volatility is computed with a 1-day shift (`shift(1)`) before use in labeling.
- SEC features are retrieved only from filings dated **strictly before** the current forecast date (`manifest_df['date'] < current_date`).
- The train/test split is enforced at `TRAIN_SPLIT_DATE`; model evaluation only runs on the held-out test period.

---

## Notes

- **Single-ticker default:** The pipeline is configured for `TICKERS = ["AAPL"]` by default. Add tickers to the list in `config.py` to extend coverage.
- **GPU support:** Chronos inference automatically uses CUDA if available, falling back to CPU.
- **Embedding cost:** Step 3 calls the Gemini API once per unprocessed filing. A manifest file tracks completed filings to allow safe reruns.
- **Fusion signal:** The dual-fusion model normalizes SEC PCA vectors by their standard deviation (not L2 norm) and scales by a fixed factor of `0.10` before constructing the pseudo-history.

---

## License

This project is for research purposes only and is not intended as financial advice.
