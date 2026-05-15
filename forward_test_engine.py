import torch
import numpy as np
import pandas as pd
import joblib
from pathlib import Path
from chronos import Chronos2Pipeline

from config_new import (
    RAW_PRICES_DIR, PROC_EMBED_DIR, RESULTS_DIR, TICKERS,
    CONTEXT_LEN, HORIZON, N_SEC_PCA, TRAIN_SPLIT_DATE,
    CHRONOS_MODEL, 
)
from covariate_builder import get_pca_features, get_decayed_sec_features
from signal_generator import (
    rank_and_size,
    compute_cross_sectional_ic,
    compute_cross_sectional_hit_rate,
    compute_portfolio_returns,
    compute_sharpe_ratio,
    compute_max_drawdown,
    compute_turnover,
)


def resolve_embedding_path(path_value):
    path_text = str(path_value).replace("\\", "/")
    path = Path(path_text)

    if path.is_absolute() or (path.parts and ":" in path.parts[0]):
        return path

    if len(path.parts) > 1:
        return Path(__file__).parent / path

    return PROC_EMBED_DIR / path.name

ID_COL = "item_id"
TS_COL = "timestamp"
TARGET_COL = "target"
SEC_COV_COLS = [f"sec_pca_{k}" for k in range(N_SEC_PCA)]


class ForwardTestEngine:
    """
    Forward testing engine with strict temporal ordering.
    
    Adapted for updated codebase with time-decayed SEC features.
    """
    
    def __init__(self, use_dual_fusion: bool = False):
        self.use_dual_fusion = use_dual_fusion
        self.model_name = "Model 2 (Dual Fusion)" if use_dual_fusion else "Model 1 (Price Only)"
        
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Loading Chronos-2 on {self.device}...")
        self.pipeline = Chronos2Pipeline.from_pretrained(
            CHRONOS_MODEL, device_map=self.device, dtype=torch.float32
        )
        
        if use_dual_fusion:
            self.sec_pca = joblib.load(PROC_EMBED_DIR / "sec_pca.joblib")
            self.manifest_df = pd.read_csv(
                PROC_EMBED_DIR / "manifest.csv",
                names=["ticker", "date", "path", "dim"]
            )
            self.manifest_df["date"] = pd.to_datetime(self.manifest_df["date"], format='mixed', dayfirst=True)

        self.per_ticker_metrics = {
            ticker: {
                "predicted_returns": [],
                "actual_returns": [],
                "signal_dates": [],
                "weights": [],
            }
            for ticker in TICKERS
        }
        self.per_ticker_daily_pnl = {ticker: [] for ticker in TICKERS}
        self.per_ticker_daily_dates = {ticker: [] for ticker in TICKERS}
        
        self.price_data = {}
        self.daily_returns = self._build_daily_returns()
        self.trading_dates = self._get_common_trading_dates()
        
        if len(self.trading_dates) == 0:
            raise ValueError(f"No trading dates found after {TRAIN_SPLIT_DATE}")
    
    def _build_daily_returns(self) -> pd.DataFrame:
        returns_dict = {}
        for ticker in TICKERS:
            df = self._load_and_prepare_price(ticker)
            if df is not None and len(df) > 0:
                close_col = "Adj Close" if "Adj Close" in df.columns else "Close"
                rets = df[close_col].pct_change()
                returns_dict[ticker] = rets
                self.price_data[ticker] = df
        return pd.DataFrame(returns_dict).sort_index()
    
    def _load_and_prepare_price(self, ticker: str) -> pd.DataFrame:
        p_file = RAW_PRICES_DIR / f"{ticker}.parquet"
        if not p_file.exists():
            return None
        df = pd.read_parquet(p_file)
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(-1)
        df.index = pd.to_datetime(df.index)
        df = df.sort_index()
        return df
    
    def _get_common_trading_dates(self) -> pd.DatetimeIndex:
        all_dates = self.daily_returns.index
        test_dates = all_dates[all_dates >= TRAIN_SPLIT_DATE]
        return test_dates
    
    def _is_ticker_available(self, ticker: str, date: pd.Timestamp) -> bool:
        """Check if ticker has sufficient history at date."""
        if ticker not in self.price_data:
            return False
        df = self.price_data[ticker]
        close_col = "Adj Close" if "Adj Close" in df.columns else "Close"
        prices = df.loc[:date, close_col].dropna()
        return len(prices) >= CONTEXT_LEN
    
    def _build_context_price_only(self, ticker: str, date: pd.Timestamp) -> pd.DataFrame:
        df = self.price_data[ticker]
        close_col = "Adj Close" if "Adj Close" in df.columns else "Close"
        prices = df.loc[:date, close_col].dropna()
        
        if len(prices) < CONTEXT_LEN:
            raise ValueError(f"Insufficient history: {len(prices)} < {CONTEXT_LEN}")
        
        ctx_prices = prices.iloc[-CONTEXT_LEN:]
       
        regular_dates = pd.date_range(end=date, periods=CONTEXT_LEN, freq="B")
        
        return pd.DataFrame({
            ID_COL: "ts",
            TS_COL: regular_dates,
            TARGET_COL: ctx_prices.values.astype(np.float32),
        })
    
    def _build_context_dual_fusion(self, ticker: str, date: pd.Timestamp) -> pd.DataFrame:
        """Build context with time-decayed SEC features."""
        df = self.price_data[ticker]
        close_col = "Adj Close" if "Adj Close" in df.columns else "Close"
        prices = df.loc[:date, close_col].dropna()
        
        if len(prices) < CONTEXT_LEN:
            raise ValueError(f"Insufficient history: {len(prices)} < {CONTEXT_LEN}")
        
        ctx_prices = prices.iloc[-CONTEXT_LEN:]
        
        regular_dates = pd.date_range(end=date, periods=CONTEXT_LEN, freq="B")

        sec_features = get_decayed_sec_features(
            ticker,
            current_date=date,
            sec_pca=self.sec_pca,
            manifest_df=self.manifest_df,
            n_components=N_SEC_PCA,
            context_length=CONTEXT_LEN,
            context_dates=regular_dates,
        )
        
        ctx = {
            ID_COL: "ts",
            TS_COL: regular_dates,
            TARGET_COL: ctx_prices.values.astype(np.float32),
        }
        
        for k in range(N_SEC_PCA):
            ctx[f"sec_pca_{k}"] = sec_features[:, k].astype(np.float32)
        
        return pd.DataFrame(ctx)
    
    def _extract_median_forecast(self, pred_df: pd.DataFrame) -> np.ndarray:
        ts = pred_df[pred_df[ID_COL] == "ts"].sort_values(TS_COL) 
        return ts["predictions"].values.astype(np.float32)
    
    def _get_forward_return(self, ticker: str, from_date: pd.Timestamp) -> float:
        rets = self.daily_returns.loc[from_date:, ticker].dropna()
        if len(rets) < HORIZON + 1:
            return np.nan
        forward_rets = rets.iloc[1:HORIZON+1]
        if len(forward_rets) < HORIZON:
            return np.nan
        return (1 + forward_rets).prod() - 1
    
    def _get_historical_vol(self, ticker: str, date: pd.Timestamp) -> float:
        rets = self.daily_returns.loc[:date, ticker].dropna()
        if len(rets) < 20:
            return np.nan
        rolling_vol = rets.rolling(20).std().iloc[-1]
        return rolling_vol * np.sqrt(252)
    
    def run(self) -> dict:
        print(f"\n{'='*60}")
        print(f"FORWARD TEST: {self.model_name}")
        print(f"{'='*60}")
        print(f"Test period: {self.trading_dates[0].date()} to {self.trading_dates[-1].date()}")
        print(f"Trading days in test: {len(self.trading_dates)}")
        
        signal_dates = []
        weights_by_date = {}
        all_ic = []
        all_rankic = []
        all_cs_hit_rates = []
        
        for i, date in enumerate(self.trading_dates):
            if i % 50 == 0:
                print(f"  {date.date()} ({i}/{len(self.trading_dates)})")
            
            available = [t for t in TICKERS if self._is_ticker_available(t, date)]
            if len(available) < 10:
                continue
            
            predicted_returns = {}
            vol_estimates = {}
            
            for ticker in available:
                try:
                    if self.use_dual_fusion:
                        context_df = self._build_context_dual_fusion(ticker, date)
                    else:
                        context_df = self._build_context_price_only(ticker, date)
                    
                    pred_df = self.pipeline.predict_df(
                        context_df,
                        prediction_length=HORIZON,
                        quantile_levels=[0.1, 0.5, 0.9],
                        id_column=ID_COL,
                        timestamp_column=TS_COL,
                        target=TARGET_COL,
                    )
                    
                    median_path = self._extract_median_forecast(pred_df)
                    
                    df = self.price_data[ticker]
                    close_col = "Adj Close" if "Adj Close" in df.columns else "Close"
                    current_price = df.loc[:date, close_col].dropna().iloc[-1]
                    
                    pred_return = (median_path[-1] / current_price) - 1
                    predicted_returns[ticker] = pred_return
                    
                    vol_estimates[ticker] = self._get_historical_vol(ticker, date)
                    
                except Exception:
                    continue
            
            if len(predicted_returns) < 10:
                continue
            
            signals_series = pd.Series(predicted_returns)
            vol_series = pd.Series(vol_estimates)
            
            valid = ~vol_series.isna()
            signals_series = signals_series[valid]
            vol_series = vol_series[valid]
            
            if len(signals_series) < 10:
                continue
            
            weights = rank_and_size(
                signals=signals_series,
                vol_estimates=vol_series,
                long_quantile=0.2,
                short_quantile=0.2,
                vol_target=0.15,
                dollar_neutral=True,
            )
            
            signal_dates.append(date)
            weights_by_date[date] = weights
            
            forward_returns = {}
            for ticker in signals_series.index:
                fwd = self._get_forward_return(ticker, date)
                if not np.isnan(fwd):
                    forward_returns[ticker] = fwd

            for ticker in signals_series.index:
                if ticker in weights.index and weights[ticker] != 0:
                    self.per_ticker_metrics[ticker]["predicted_returns"].append(
                        signals_series[ticker]
                    )
                    self.per_ticker_metrics[ticker]["actual_returns"].append(
                        forward_returns.get(ticker, np.nan)
                    )
            
            if len(forward_returns) >= 5:
                fwd_series = pd.Series(forward_returns).dropna()

                if len(fwd_series) >= 5:
                    signals_for_ic = signals_series.reindex(fwd_series.index).dropna()
                    fwd_series = fwd_series.reindex(signals_for_ic.index)

                    if len(signals_for_ic) >= 5:
                        ic_dict = compute_cross_sectional_ic(signals_for_ic, fwd_series)
                        all_ic.append(ic_dict['IC'])
                        all_rankic.append(ic_dict['RankIC'])
                        cs_hit_rate = compute_cross_sectional_hit_rate(signals_for_ic, fwd_series)
                        all_cs_hit_rates.append(cs_hit_rate)
        
        portfolio_returns = compute_portfolio_returns(
            signal_dates, weights_by_date, self.daily_returns
        )

        for date, weights in weights_by_date.items():
            date_idx = signal_dates.index(date)
            if date_idx < len(signal_dates) - 1:
                next_date = signal_dates[date_idx + 1]
            else:
                next_date = self.trading_dates[-1]

            holding_mask = (self.daily_returns.index > date) & (
                self.daily_returns.index <= next_date
            )
            holding_rets = self.daily_returns.loc[holding_mask]

            for ticker in weights.index:
                if ticker in holding_rets.columns:
                    weight = weights[ticker]
                    if weight != 0:
                        daily_pnl = weight * holding_rets[ticker]
                        self.per_ticker_daily_pnl[ticker].extend(daily_pnl.values)
                        self.per_ticker_daily_dates[ticker].extend(daily_pnl.index)

        if portfolio_returns.empty:
            benchmark_returns = pd.Series(dtype=float)
        else:
            benchmark_returns = self.daily_returns.mean(axis=1)
            benchmark_returns = benchmark_returns.loc[
                portfolio_returns.index.min():portfolio_returns.index.max()
            ]
        
        valid_ic = [x for x in all_ic if not np.isnan(x)]
        valid_rankic = [x for x in all_rankic if not np.isnan(x)]
        valid_cs_hit_rates = [x for x in all_cs_hit_rates if not np.isnan(x)]
        
        return {
            'model': self.model_name,
            'n_signal_dates': len(signal_dates),
            'sharpe_ratio': compute_sharpe_ratio(portfolio_returns),
            'max_drawdown': compute_max_drawdown(portfolio_returns),
            'mean_IC': np.mean(valid_ic) if valid_ic else np.nan,
            'std_IC': np.std(valid_ic, ddof=1) if len(valid_ic) > 1 else np.nan,
            'IC_IR': (np.mean(np.array(valid_ic)) / np.std(np.array(valid_ic), ddof=1)) if len(valid_ic) > 1 and np.std(np.array(valid_ic), ddof=1) > 0 else np.nan,
            'mean_RankIC': np.mean(valid_rankic) if valid_rankic else np.nan,
            'hit_rate': np.mean(all_cs_hit_rates) if all_cs_hit_rates else np.nan,  
            'turnover': compute_turnover(weights_by_date, signal_dates),
            'portfolio_returns': portfolio_returns,
            'per_ticker_metrics': self.compute_per_ticker_metrics(),
        }

    def compute_per_ticker_metrics(self) -> pd.DataFrame:

        import scipy.stats

        per_ticker_results = []

        for ticker, data in self.per_ticker_metrics.items():
            daily_pnl = pd.Series(
                self.per_ticker_daily_pnl[ticker],
                index=self.per_ticker_daily_dates[ticker]
            ).sort_index()

            daily_pnl = daily_pnl.groupby(daily_pnl.index).mean()

            if len(daily_pnl) < 20:
                continue

            pred_rets = np.array(data["predicted_returns"], dtype=float)
            actual_rets = np.array(data["actual_returns"], dtype=float)
            weights = np.array(data.get("weights", []), dtype=float)

            valid_mask = ~np.isnan(pred_rets) & ~np.isnan(actual_rets)
            pred_rets = pred_rets[valid_mask]
            actual_rets = actual_rets[valid_mask]
            weights = weights[valid_mask] if len(weights) == len(valid_mask) else weights

            if len(pred_rets) < 3:
                continue

            try:
                ic, _ = scipy.stats.pearsonr(pred_rets, actual_rets)
                rank_ic, _ = scipy.stats.spearmanr(pred_rets, actual_rets)
            except Exception:
                ic = np.nan
                rank_ic = np.nan

            sharpe_annual = compute_sharpe_ratio(daily_pnl)
            mdd = compute_max_drawdown(daily_pnl)

            per_ticker_results.append({
                "ticker": ticker,
                "n_signals": len(pred_rets),
                "IC": ic,
                "RankIC": rank_ic,
                "hit_rate": np.mean(np.sign(pred_rets) == np.sign(actual_rets)),
                "Sharpe_annual": sharpe_annual,
                "MDD": mdd,
            })

        return pd.DataFrame(per_ticker_results)

    
    