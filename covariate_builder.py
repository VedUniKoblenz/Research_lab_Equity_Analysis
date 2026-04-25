# covariate_builder.py
import numpy as np
import pandas as pd
from pathlib import Path
from config import PROC_EMBED_DIR

def get_pca_features(ticker, current_date, sec_pca, manifest_df, n_components, path_col_idx=2):
    """
    Retrieve the closest SEC filing PCA features before the current date.
    """
    # Filter for this ticker and dates BEFORE current_date (no lookahead bias)
    mask = (manifest_df['ticker'] == ticker) & (manifest_df['date'] < current_date)
    ticker_data = manifest_df[mask].copy()
    
    if len(ticker_data) == 0:
        return np.zeros(n_components, dtype=np.float32)
    
    # Get the most recent filing before current_date
    most_recent = ticker_data.loc[ticker_data['date'].idxmax()]
    
    # Get the path to the PCA features
    raw_path = Path(most_recent.iloc[path_col_idx])
    if not raw_path.is_absolute():
        raw_path = PROC_EMBED_DIR / raw_path
    
    # We want the PCA version: stem + _pca.npy
    pca_path = raw_path.with_name(raw_path.stem + "_pca.npy")
    
    try:
        if pca_path.exists():
            pca_features = np.load(pca_path)
        elif raw_path.exists():
            # If PCA file doesn't exist but raw does, transform it on the fly
            embedding = np.load(raw_path)
            pca_features = sec_pca.transform(embedding.reshape(1, -1))[0]
        else:
            raise FileNotFoundError(f"Neither {pca_path.name} nor {raw_path.name} found.")
        
        # Handle different possible shapes
        if pca_features.ndim == 2:
            pca_features = pca_features[0]
        
        # Ensure correct dimensionality
        if len(pca_features) < n_components:
            pca_features = np.pad(pca_features, (0, n_components - len(pca_features)))
        elif len(pca_features) > n_components:
            pca_features = pca_features[:n_components]
        
        return pca_features.astype(np.float32)
    
    except Exception as e:
        print(f"Warning: Could not load PCA features for {ticker} on {current_date}: {e}")
        return np.zeros(n_components, dtype=np.float32)


def build_covariate_matrix(ticker, dates, sec_pca, manifest_df, n_components, path_col_idx=2):
    """
    Build a matrix of SEC covariate features for multiple dates.
    
    Parameters:
    -----------
    ticker : str
        Stock ticker symbol
    dates : array-like
        List of dates to get features for
    sec_pca : PCA object
        Fitted PCA transformer
    manifest_df : DataFrame
        Manifest file with SEC filing information
    n_components : int
        Number of PCA components to use
    path_col_idx : int
        Index of path column in manifest
    
    Returns:
    --------
    np.ndarray : Covariate matrix of shape (len(dates), n_components)
    """
    covariate_matrix = []
    
    for date in dates:
        features = get_pca_features(
            ticker, date, sec_pca, manifest_df, n_components, path_col_idx
        )
        covariate_matrix.append(features)
    
    return np.array(covariate_matrix, dtype=np.float32)