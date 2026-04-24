# In 07_model_dual_fusion.py
import torch
import pandas as pd
import numpy as np
import joblib
from chronos.chronos2 import Chronos2Pipeline
from config import compute_vol_series, CONTEXT_LEN, TRAIN_SPLIT_DATE, CHRONOS_MODEL, RAW_PRICES_DIR, PROC_EMBED_DIR, RESULTS_DIR, TICKERS, N_SEC_PCA, triple_barrier_label, calculate_financial_metrics
from covariate_builder import get_pca_features

def main():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    pipeline = Chronos2Pipeline.from_pretrained(CHRONOS_MODEL, device_map=device, dtype=torch.float32)
    all_results = []
    
    sec_pca = joblib.load(PROC_EMBED_DIR / "sec_pca.joblib")
    manifest_df = pd.read_csv(PROC_EMBED_DIR / "manifest.csv", names=['ticker','date','path','dim'])
    manifest_df['date'] = pd.to_datetime(manifest_df['date'])

    for ticker in TICKERS:
        p_file = RAW_PRICES_DIR / f"{ticker}.parquet"
        if not p_file.exists():
            print(f"Skipping {ticker}, file not found.")
            continue
        print(f"Running Model 2 (Dual Fusion) for {ticker}...")
        df = pd.read_parquet(p_file).reset_index()
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(-1)
        df['Date'] = pd.to_datetime(df['Date'])
        close_col = 'Adj Close' if 'Adj Close' in df.columns else 'Close'
        df['vol'] = compute_vol_series(df, close_col=close_col)
        df['forward_return_15d'] = df[close_col].shift(-15) / df[close_col] - 1
        df = df.dropna(subset=['vol']).reset_index(drop=True)
        
        test_mask = df['Date'] >= TRAIN_SPLIT_DATE
        if not test_mask.any(): continue
        start_idx = df[test_mask].index[0]
        
        y_true, y_pred, actual_rets, pred_rets, rmses = [], [], [], [], []
        
        for i in range(start_idx, len(df) - 15):
            if i % 100 == 0: print(f"  {ticker}: {i}/{len(df)-15}")
            row = df.iloc[i]
            if pd.isna(row['forward_return_15d']): continue
            
            # 1. Prepare Context: Extract historical prices and fundamental SEC features
            price_ctx = df[close_col].values[i-CONTEXT_LEN+1:i+1].astype(np.float32)
            
            # get_pca_features retrieves the closest SEC filing embedding available BEFORE the current date
            sec_vec = get_pca_features(ticker, row['Date'], sec_pca, manifest_df, N_SEC_PCA, path_col_idx=2)
            
            # Compute confidence weight from the embedding's own magnitude.
            # A strong/clear filing has a larger norm; weak/ambiguous ones stay near zero.
            # This prevents low-quality embeddings from polluting the fusion signal.
            sec_norm = np.linalg.norm(sec_vec)
            sec_confidence = min(sec_norm / 5.0, 1.0)  # clip to [0, 1]

            # 2. Anchor Trend Scaling (FUSION LOGIC):
            # We take the first price in our window as the 'Anchor'.
            anchor_price = price_ctx[0]
            
            # Normalize the SEC PCA vector to a unit scale, then amplify it by 5% (0.05).
            # This turns raw embeddings into a 'Fundamental Trend' signal.
            # Replace norm-based scaling with std-based scaling
            sec_std = np.std(sec_vec) + 1e-6
            norm_sec = (sec_vec / sec_std) * 0.10  # scales by signal strength, not unit vector

            
            # We create a 'pseudo-history' by applying the fundamental trend to the anchor price.
            # This effectively tells Chronos: "Based on SEC filings, the company's baseline should be X."
            pre_history = anchor_price * (1 + np.cumsum(norm_sec))
            
            # 3. Concatenation: We prepend the fundamental pseudo-history to the real price history.
            # This creates a 'Multivariate' context window for the univariate Chronos model.
            combined_ctx = np.concatenate([pre_history, price_ctx])
            
            # Reshape for Chronos: (Batch, Channels, Length)
            context = torch.tensor(combined_ctx.astype(np.float32)).unsqueeze(0).unsqueeze(0)
            
            # 4. Prediction: Forecast 15 days ahead
            forecast = pipeline.predict(context, prediction_length=15)
            full_path = torch.quantile(forecast[0], 0.5, dim=0).cpu().numpy().flatten()
            median_path = full_path[-15:] # Extract only the future 15-day window
            
            # RMSE
            actual_path = df.iloc[i+1:i+16][close_col].values
            rmses.append(np.sqrt(np.mean((median_path - actual_path)**2)))
            
            yt = triple_barrier_label(actual_path, row[close_col], row['vol'])
            fwd_ret = float((median_path[-1] / row[close_col]) - 1)
            
            # Extreme Conviction Gating: 2.0x vol
            threshold = row['vol'] * np.sqrt(15) * 0.75
            yp = 1 if fwd_ret > threshold else (-1 if fwd_ret < -threshold else 0)
            
            y_true.append(yt)
            y_pred.append(yp)
            actual_rets.append(row['forward_return_15d'])
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
            
    pd.DataFrame(all_results).to_csv(RESULTS_DIR / "dual_fusion_results.csv", index=False)

if __name__ == "__main__": main()
