# Research_lab_Equity_Analysis

A **strictly out-of-sample** quantitative research pipeline that evaluates whether augmenting a time-series foundation model (Amazon Chronos-2) with SEC filing signals improves equity return forecasting and portfolio performance.

Two model variants are compared head-to-head:
- **Model 1 — Price Only**: Chronos-2 forecasts from historical prices alone
- **Model 2 — Dual Fusion**: Chronos-2 forecasts from prices + time-decayed, PCA-reduced SEC 10-Q embeddings

---

## Table of Contents

- [Overview](#overview)
- [Pipeline Architecture](#pipeline-architecture)
- [Directory Structure](#directory-structure)
- [Setup](#setup)
- [Configuration](#configuration)
- [Running the Pipeline](#running-the-pipeline)
- [Outputs](#outputs)
- [Key Design Decisions](#key-design-decisions)
- [Caveats & Limitations](#caveats--limitations)

---

## Overview

The core research question is: **do SEC 10-Q filings contain price-relevant information beyond what is already reflected in historical prices?**

To answer this, the pipeline:
1. Downloads historical prices and SEC 10-Q filings for ~100 large-cap U.S. equities
2. Summarizes SEC filings using Gemini and embeds the summaries into a vector space
3. Reduces embedding dimensionality via PCA (fit on training data only)
4. Runs a walk-forward forward test for both model variants
5. Produces aggregate and per-ticker performance metrics, sector breakdowns, and 14 diagnostic figures

---

## Pipeline Architecture

The pipeline is organized as a sequence of numbered scripts:

```
01_get_prices.py          →  Download historical OHLCV data (yfinance)
02_get_sec.py             →  Download SEC 10-Q filings (EDGAR)
03_summarize_sec.py       →  Summarize filings with Gemini (price-signal focused)
04_process_sec.py         →  Embed summaries → .npy vectors via Gemini Embeddings
05_create_pca.py          →  Fit PCA on train-period embeddings; transform all
06_forward_test_price_only.py   →  Walk-forward test: Model 1 (Price Only)
07_forward_test_dual_fusion.py  →  Walk-forward test: Model 2 (Dual Fusion)
08_per_ticker_comparison.py     →  Per-ticker metrics, sector-grouped table + CSV
09_per_sector_comparison_.py    →  Sector-level delta analysis (Dual − Price)
10_Aggregate_model_comparison.py →  Final aggregate benchmark table
11_visualizations.py            →  14-figure visualization suite
```

**Core modules** (imported by the scripts above):

| Module | Role |
|---|---|
| `config_new.py` | Central configuration (paths, hyperparameters, ticker universe) |
| `forward_test_engine.py` | `ForwardTestEngine` — orchestrates walk-forward evaluation |
| `signal_generator.py` | Cross-sectional IC/RankIC, portfolio construction, Sharpe/MDD/turnover |
| `covariate_builder.py` | Time-decayed SEC covariate matrices with look-ahead prevention |

---

## Directory Structure

```
project_root/
├── config_new.py
├── forward_test_engine.py
├── signal_generator.py
├── covariate_builder.py
│
├── 01_get_prices.py
├── 02_get_sec.py
├── 03_summarize_sec.py
├── 04_process_sec.py
├── 05_create_pca.py
├── 06_forward_test_price_only.py
├── 07_forward_test_dual_fusion.py
├── 08_per_ticker_comparison.py
├── 09_per_sector_comparison_.py
├── 10_Aggregate_model_comparison.py
├── 11_visualizations.py
│
├── data/
│   ├── raw/
│   │   ├── prices/                  # <TICKER>.parquet
│   │   └── edgar_downloads/         # raw SEC filings
│   └── processed/
│       ├── sec_summaries/           # Gemini-generated .txt summaries
│       └── embeddings/              # .npy vectors, manifest.csv, sec_pca.joblib
│
├── results/
│   ├── price_only_summary.csv
│   ├── price_only_per_ticker.csv
│   ├── price_only_returns.csv
│   ├── dual_fusion_summary.csv
│   ├── dual_fusion_per_ticker.csv
│   ├── dual_fusion_returns.csv
│   ├── model_comparison_by_sector.csv
│   ├── sector_comparison_summary.csv
│   ├── final_benchmark_comparison.csv
│   └── figures/                     # 14 PNG figures
│
└── .env                             # GEMINI_API_KEY (not committed)
```

---

## Setup

### 1. Clone the repository

```bash
git clone <your-repo-url>
cd <project-root>
```

### 2. Install dependencies

```bash
pip install pandas numpy scipy scikit-learn joblib yfinance \
            sec-edgar-downloader beautifulsoup4 lxml \
            google-genai python-dotenv torch matplotlib seaborn
```

> **Chronos-2** also needs to be installed. Follow the [Amazon Chronos installation guide](https://github.com/amazon-science/chronos-forecasting).

### 3. Configure environment variables

Create a `.env` file in the project root:

```env
GEMINI_API_KEY=your_real_key_here
```

---

## Configuration

All key parameters live in `config_new.py`:

| Parameter | Default | Description |
|---|---|---|
| `CONTEXT_LEN` | `512` | Historical context window fed to Chronos-2 |
| `HORIZON` | `15` | Forecast horizon in business days |
| `N_SEC_PCA` | `16` | PCA components for SEC embeddings |
| `TRAIN_SPLIT_DATE` | `"2024-01-01"` | Train/test cutoff date |
| `START_DATE` | `"2018-01-01"` | Start of price history download |
| `CHRONOS_MODEL` | `"amazon/chronos-2"` | Chronos model identifier |
| `GEN_MODEL` | `"gemini-flash-latest"` | Gemini model for SEC summarization |
| `EMBED_MODEL` | `"gemini-embedding-2"` | Gemini model for embedding |

The ticker universe (`TICKERS`) is a curated list of ~100 large-cap U.S. equities across Technology, Finance, Healthcare, Energy, Consumer, and Industrials sectors.

---

## Running the Pipeline

Run scripts in order. Each step depends on outputs from prior steps.

```bash
# Step 1: Download data
python 01_get_prices.py
python 02_get_sec.py

# Step 2: Process SEC filings
python 03_summarize_sec.py
python 04_process_sec.py
python 05_create_pca.py

# Step 3: Run forward tests (these are the longest-running steps)
python 06_forward_test_price_only.py
python 07_forward_test_dual_fusion.py

# Step 4: Analysis and visualization
python 08_per_ticker_comparison.py
python 09_per_sector_comparison_.py
python 10_Aggregate_model_comparison.py
python 11_visualizations.py
```

> **GPU recommended** for steps 6 and 7. The engine uses CUDA automatically if available (`torch.cuda.is_available()`).

---

## Outputs

### Aggregate metrics (`results/final_benchmark_comparison.csv`)

| Metric | Description |
|---|---|
| `Sharpe` | Annualized Sharpe ratio |
| `MaxDD` | Maximum drawdown |
| `Mean IC` | Mean cross-sectional Pearson IC |
| `Std IC` | Std dev of IC across signal dates |
| `IC IR` | Information ratio (Mean IC / Std IC) |
| `Mean RankIC` | Mean cross-sectional Spearman IC |
| `Hit Rate` | Mean fraction of correct directional predictions |
| `Turnover` | Mean one-way portfolio turnover per rebalance |
| `N Dates` | Number of signal dates evaluated |

### Per-ticker metrics (`results/*_per_ticker.csv`)

`ticker`, `n_signals`, `IC`, `RankIC`, `hit_rate`, `Sharpe_annual`, `MDD`

### Visualization suite (`results/figures/`)

14 PNG figures including cumulative return curves, drawdown overlays, rolling Sharpe, sector heatmaps, IC/RankIC scatter plots, radar charts, return distributions, and win/loss sector profiles.
