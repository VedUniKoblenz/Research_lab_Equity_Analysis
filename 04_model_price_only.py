import torch
import pandas as pd
import numpy as np
from chronos import Chronos2Pipeline
from config_new import compute_vol_series, CONTEXT_LEN, TRAIN_SPLIT_DATE, CHRONOS_MODEL, RAW_PRICES_DIR, RESULTS_DIR, TICKERS, THRESHOLD_MULTIPLIER_PRICE, calculate_financial_metrics, triple_barrier_label

HORIZON = 15


def extract_median_forecast(quantiles, batch_idx=0):
    """Extract the median forecast from Chronos-2 predict_quantiles output."""
    t = quantiles[batch_idx]
    if hasattr(t, "cpu"):
        t = t.cpu().numpy()
    else:
        t = np.array(t)

    if t.ndim == 3:
        # Chronos can return either (n_var, n_q, pred_len) or (n_var, pred_len, n_q)
        if t.shape[2] == 3:
            median = t[0, :, 1]
        elif t.shape[1] == 3:
            median = t[0, 1, :]
        else:
            raise ValueError(f"Unexpected 3D quantile shape: {t.shape}")
    elif t.ndim == 2:
        if t.shape[1] == 3:
            median = t[:, 1]
        elif t.shape[0] == 3:
            median = t[1, :]
        else:
            raise ValueError(f"Unexpected 2D quantile shape: {t.shape}")
    elif t.ndim == 1:
        median = t
    else:
        raise ValueError(f"Unexpected quantile tensor shape: {t.shape}")

    return median.astype(np.float32)

def main():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    pipeline = Chronos2Pipeline.from_pretrained(CHRONOS_MODEL, device_map=device, dtype=torch.float32)
    all_results = []

    for ticker in TICKERS:
        p_file = RAW_PRICES_DIR / f"{ticker}.parquet"
        if not p_file.exists():
            print(f"Skipping {ticker}, file not found.")
            continue
        print(f"Running Model 1 (Price Only) for {ticker}...")
        df = pd.read_parquet(p_file).reset_index()
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(-1)
        df['Date'] = pd.to_datetime(df['Date'])
        close_col = 'Adj Close' if 'Adj Close' in df.columns else 'Close'
        
        df['vol'] = compute_vol_series(df, close_col=close_col)
        df['forward_return'] = df[close_col].shift(-HORIZON) / df[close_col] - 1
        df = df.dropna(subset=['vol']).reset_index(drop=True)
        
        test_mask = df['Date'] >= TRAIN_SPLIT_DATE
        if not test_mask.any(): continue
        start_idx = max(df[test_mask].index[0], CONTEXT_LEN - 1)
        
        y_true, y_pred, actual_rets, pred_rets, rmses = [], [], [], [], []
        
        for i in range(start_idx, len(df) - HORIZON):
            if i % 100 == 0: print(f"  {ticker}: {i}/{len(df)-HORIZON}")
            row = df.iloc[i]
            if pd.isna(row['forward_return']): continue
            
            # 1. Prepare Context: Only the univariate price sequence
            price_ctx = df[close_col].values[i-CONTEXT_LEN+1:i+1].astype(np.float32)
            input_dict = {"target": price_ctx}

            # 2. Prediction: Generate HORIZON-day forward price paths using predict_quantiles
            quantiles, mean = pipeline.predict_quantiles(
                [input_dict],
                prediction_length=HORIZON,
                quantile_levels=[0.1, 0.5, 0.9]
            )
            median_path = extract_median_forecast(quantiles, batch_idx=0)
            
            # 3. Labeling: Use the Triple-Barrier Method
            # This checks if the price hits an Upper Barrier (Buy), Lower Barrier (Sell), 
            # or neither within the horizon (Hold).
            actual_path = df.iloc[i+1:i+1+HORIZON][close_col].values
            rmses.append(np.sqrt(np.mean((median_path - actual_path)**2)))
            yt = triple_barrier_label(actual_path, row[close_col], row['vol'], horizon=HORIZON)
            
            # 4. Decision: Calculate model's predicted return and compare to Volatility threshold
            fwd_ret = float((median_path[-1] / row[close_col]) - 1)
            
            # Extreme Conviction Gating: Only trade if predicted return is > 2x current volatility
            threshold = row['vol'] * np.sqrt(HORIZON) * THRESHOLD_MULTIPLIER_PRICE
            yp = 1 if fwd_ret > threshold else (-1 if fwd_ret < -threshold else 0)
            
            y_true.append(yt)
            y_pred.append(yp)
            actual_rets.append(row['forward_return'])
            pred_rets.append(fwd_ret)

        if y_true:
            metrics = calculate_financial_metrics(y_true, y_pred, actual_rets, pred_returns=pred_rets)
            n_trades = np.count_nonzero(y_pred)
            avg_conviction = np.mean(np.abs(pred_rets))
            all_results.append({
                'ticker': ticker,
                'Accuracy': np.mean(np.array(y_true)==np.array(y_pred)),
                'Sharpe': metrics['Sharpe'],
                'MDD': metrics['MDD'],
                'IC': metrics['IC'],
                'RMSE': np.mean(rmses),
                'N_Trades': n_trades,
                'Avg_Conviction': avg_conviction
            })
            
    pd.DataFrame(all_results).to_csv(RESULTS_DIR / "price_only_results.csv", index=False)

if __name__ == "__main__": main()
