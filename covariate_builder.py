"""covariate_builder.py
Helpers to load SEC PCA features and produce time-decayed SEC feature matrices
used by the dual-fusion model. Includes conservative fallbacks when data
is missing to avoid look-ahead.
"""
import numpy as np
import pandas as pd
from pathlib import Path
from config_new import BASE_DIR, PROC_EMBED_DIR


def resolve_embedding_path(path_value):
    path_text = str(path_value).replace("\\", "/")
    path = Path(path_text)

    if path.is_absolute() or (path.parts and ":" in path.parts[0]):
        return path

    if len(path.parts) > 1:
        return BASE_DIR / path

    return PROC_EMBED_DIR / path.name

def get_pca_features(ticker, current_date, sec_pca, manifest_df, n_components, path_col_idx=2):
    """
    Retrieve the closest SEC filing PCA features before the current date.
    """
    manifest_df = manifest_df.copy()
    manifest_df['date'] = pd.to_datetime(manifest_df['date'], errors='coerce', format='mixed', dayfirst=True)

    lag = pd.tseries.offsets.BDay(1)
    mask = (manifest_df['ticker'] == ticker) & (manifest_df['date'] < pd.to_datetime(current_date) - lag)
    ticker_data = manifest_df[mask].copy()
    
    if len(ticker_data) == 0:
        return np.zeros(n_components, dtype=np.float32)
    
    most_recent = ticker_data.loc[ticker_data['date'].idxmax()]
    
    raw_path = resolve_embedding_path(most_recent.iloc[path_col_idx])
    
    pca_path = raw_path.with_name(raw_path.stem + "_pca.npy")
    
    try:
        if pca_path.exists():
            pca_features = np.load(pca_path)
        elif raw_path.exists():
            embedding = np.load(raw_path)
            pca_features = sec_pca.transform(embedding.reshape(1, -1))[0]
        else:
            raise FileNotFoundError(f"Neither {pca_path.name} nor {raw_path.name} found.")
        
        if pca_features.ndim == 2:
            pca_features = pca_features[0]
        
        if len(pca_features) < n_components:
            pca_features = np.pad(pca_features, (0, n_components - len(pca_features)))
        elif len(pca_features) > n_components:
            pca_features = pca_features[:n_components]
        
        return pca_features.astype(np.float32)
    
    except Exception as e:
        print(f"Warning: Could not load PCA features for {ticker} on {current_date}: {e}")
        return np.zeros(n_components, dtype=np.float32)





def get_decayed_sec_features(ticker, current_date, sec_pca, manifest_df, n_components,
                              context_length=20, decay_half_life=60, context_dates=None):
    import math

    manifest_df = manifest_df.copy()
    manifest_df['date'] = pd.to_datetime(manifest_df['date'], errors='coerce', format='mixed', dayfirst=True)

    lag = pd.tseries.offsets.BDay(1)
    cutoff = pd.to_datetime(current_date) - lag
    mask = (manifest_df['ticker'] == ticker) & (manifest_df['date'] < cutoff)
    ticker_data = manifest_df[mask].sort_values('date')

    if len(ticker_data) == 0:
        return np.zeros((context_length, n_components), dtype=np.float32)

    all_features = []
    all_dates = []
    for _, row in ticker_data.iterrows():
        vec = get_pca_features(ticker, row['date'], sec_pca, manifest_df, n_components)
        all_features.append(vec)
        all_dates.append(pd.to_datetime(row['date']))

    all_features = np.array(all_features)
    all_dates = np.array(all_dates)

    result = np.zeros((context_length, n_components), dtype=np.float32)
    if context_dates is not None:
        context_dates = pd.DatetimeIndex(context_dates)
    else:
        context_dates = pd.bdate_range(end=pd.to_datetime(current_date), periods=context_length)

    for day_idx in range(context_length):
        target_date = context_dates[day_idx]

        active_mask = all_dates < cutoff
        if not active_mask.any():
            continue
        active_idx = np.where(active_mask)[0][-1]

        days_since_filing = len(pd.bdate_range(all_dates[active_idx], target_date)) - 1
        weight = math.exp(-days_since_filing / decay_half_life)

        result[day_idx] = all_features[active_idx] * weight

    return result