# Research_lab_Equity_Analysis
A quantitative research pipeline that benchmarks two trading signal models:

- **Model 1 (Price Only):** Chronos-2 time-series forecasting on historical prices alone.
- **Model 2 (Dual Fusion):** Chronos-2 augmented with SEC 10-Q filing embeddings (via Gemini) reduced through PCA and injected as a pseudo-history context.

Both models use triple-barrier labeling and volatility-scaled thresholds to generate long/short/hold signals, and are evaluated on Sharpe ratio, max drawdown, information coefficient, and RMSE.

---

## Pipeline Overview

```
01_get_prices.py          → Download price data (Yahoo Finance)
02_get_sec.py             → Download 10-Q filings (SEC EDGAR)
02b_summarize_sec.py      → Summarize filings with Gemini (alpha-extraction prompt)
03_process_sec_new.py     → Embed summaries with Gemini → .npy files + manifest.csv
create_pca.py             → Fit PCA on all embeddings → sec_pca.joblib + *_pca.npy
04_model_price_only.py    → Run Model 1 baseline → price_only_results.csv
05_model_dual_fusion.py   → Run Model 2 fusion → dual_fusion_results.csv
06_final_benchmark.py     → Compare models, print table, save final_summary.csv
```

---

## Project Structure

```
.
├── config_new.py                  # Central config: paths, tickers, constants, helpers
├── covariate_builder.py           # SEC PCA feature retrieval (no lookahead bias)
├── create_pca.py                  # Fit + save PCA on SEC embeddings
│
├── 01_get_prices.py
├── 02_get_sec.py
├── 02b_summarize_sec.py
├── 03_process_sec_new.py
├── 04_model_price_only.py
├── 05_model_dual_fusion.py
├── 06_final_benchmark.py
│
├── data/
│   ├── raw/
│   │   ├── prices/                # {ticker}.parquet files
│   │   └── edgar_downloads/       # sec-edgar-filings/<ticker>/10-Q/<accession>/
│   └── processed/
│       ├── sec_summaries/         # {ticker}_{accession}.txt (Gemini summaries)
│       └── embeddings/            # {ticker}_{accession}.npy, *_pca.npy, manifest.csv, sec_pca.joblib
│
└── results/
    ├── price_only_results.csv
    ├── dual_fusion_results.csv
    └── final_summary.csv
```

---

## Requirements

**Python 3.10+**

```bash
pip install yfinance pandas numpy joblib scikit-learn torch \
            chronos-forecasting sec-edgar-downloader \
            google-genai beautifulsoup4 lxml python-dotenv pyarrow
```

> Chronos-2 requires PyTorch. GPU recommended for inference speed but CPU works.

---

## Configuration

All shared settings live in `config_new.py`. Key parameters:

| Constant | Default | Description |
|---|---|---|
| `TICKERS` | `["AAPL","MSFT","GOOGL","AMZN","META","NVDA"]` | Tickers to process |
| `START_DATE` | `"2018-01-01"` | Price history start |
| `TRAIN_SPLIT_DATE` | `"2024-01-01"` | Test window start |
| `HORIZON` | `15` | Forecast horizon (trading days) |
| `CONTEXT_LEN` | `512` | Price context window length |
| `N_SEC_PCA` | `16` | PCA components for SEC embeddings |
| `BARRIER_MULTIPLIER` | `0.5` | Triple-barrier band width (× horizon vol) |
| `THRESHOLD_MULTIPLIER_*` | `0.15` | Trade signal threshold (× horizon vol) |
| `CHRONOS_MODEL` | `"amazon/chronos-2"` | Hugging Face model ID |
| `EMBED_MODEL` | `"gemini-embedding-2"` | Gemini embedding model |
| `GEN_MODEL` | `"gemini-flash-latest"` | Gemini generative model |

Create a `.env` file in the project root:

```
GEMINI_API_KEY=your_key_here
```

---

## Quickstart

```bash
# 1. Download prices
python 01_get_prices.py

# 2. Download SEC 10-Q filings
python 02_get_sec.py

# 3. Summarize filings (requires GEMINI_API_KEY)
python 02b_summarize_sec.py

# 4. Embed summaries
python 03_process_sec_new.py

# 5. Fit PCA on embeddings
python create_pca.py

# 6. Run baseline model
python 04_model_price_only.py

# 7. Run fusion model
python 05_model_dual_fusion.py

# 8. Compare results
python 06_final_benchmark.py
```

---

## How the Dual Fusion Works

Model 2 augments Chronos with SEC filing signals using a **pseudo-history injection** strategy:

1. For each inference date, the most recent 10-Q filing **before** that date is retrieved (no lookahead bias).
2. The SEC embedding is reduced to `N_SEC_PCA` dimensions via the pre-fitted PCA.
3. The PCA vector is rescaled and interpreted as a sequence of cumulative return adjustments anchored to the first price in the context window → `pre_history`.
4. `pre_history` is prepended to the real price context: `[pre_history | price_ctx]`.
5. Chronos-2 forecasts over this combined context, allowing fundamental filing signals to influence the predicted trajectory.

---

## Labeling & Signal Logic

**Triple-barrier labels** (ground truth):

| Label | Condition |
|---|---|
| `+1` | Price hits upper barrier (`initial × (1 + 0.5 × vol × √horizon)`) first |
| `-1` | Price hits lower barrier first |
| `0` | Neither barrier hit within horizon |

**Trade signal** (model prediction):

```
predicted_return = (median_forecast[-1] / current_price) - 1
threshold        = vol × √horizon × THRESHOLD_MULTIPLIER

signal = +1  if predicted_return > threshold
       = -1  if predicted_return < -threshold
       =  0  otherwise (hold)
```

---

## Output Metrics

Each model produces per-ticker results with the following columns:

| Metric | Description |
|---|---|
| `Accuracy` | Fraction of correct directional labels |
| `Sharpe` | Annualized Sharpe of strategy returns |
| `MDD` | Maximum drawdown of cumulative strategy |
| `IC` | Information coefficient (Pearson r, predicted vs actual returns) |
| `RMSE` | Root mean squared error of price forecast |
| `N_Trades` | Number of non-zero signals |
| `Avg_Conviction` | Mean absolute predicted return |

Final comparison is printed to console and saved to `results/final_summary.csv`.

---

## Key Design Decisions

**No lookahead bias in SEC features** — `covariate_builder.py` strictly filters filings to dates *before* the current inference date, then picks the most recent one.

**Volatility-gated signals** — all trading thresholds scale with realized rolling volatility, making the strategy adaptive across different market regimes.

**Gemini alpha-extraction prompt** — the summarization step uses a structured prompt targeting sentiment divergence, hidden operational shifts, forward guidance, atypical risk factors, KPI trajectory, and capital allocation signals rather than generic document summarization.

**PCA dimensionality reduction** — raw Gemini embeddings are reduced from ~3072 to 16 components via PCA fitted once across all available filings, enabling efficient fusion with price context.

---


