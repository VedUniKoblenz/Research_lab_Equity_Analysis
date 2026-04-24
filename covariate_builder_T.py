# covariate_builder.py
import numpy as np
import pandas as pd
from pathlib import Path

def get_pca_features(ticker, current_date, sec_pca, manifest_df, n_components, path_col_idx=2):
    """
    Retrieve the closest SEC filing PCA features before the current date.
    
    Parameters:
    -----------
    ticker : str
        Stock ticker symbol
    current_date : datetime
        Current prediction date (use SEC data ONLY from before this date)
    sec_pca : PCA object
        Fitted PCA transformer (loaded from sec_pca.joblib)
    manifest_df : DataFrame
        DataFrame with columns ['ticker', 'date', 'path', 'dim']
    n_components : int
        Number of PCA components to use (N_SEC_PCA from config)
    path_col_idx : int
        Index of the path column in manifest_df (default 2)
    
    Returns:
    --------
    np.ndarray : PCA features of shape (n_components,)
    """
    # Filter for this ticker and dates BEFORE current_date (no lookahead bias)
    mask = (manifest_df['ticker'] == ticker) & (manifest_df['date'] < current_date)
    ticker_data = manifest_df[mask].copy()
    
    if len(ticker_data) == 0:
        # No SEC data available before this date - return zeros
        # This is common for early dates before first filing
        return np.zeros(n_components, dtype=np.float32)
    
    # Get the most recent filing before current_date
    most_recent = ticker_data.loc[ticker_data['date'].idxmax()]
    
    # Get the path to the PCA features
    pca_path = Path(most_recent.iloc[path_col_idx])
    
    try:
        # Load PCA features
        if pca_path.suffix == '.npy':
            pca_features = np.load(pca_path)
        elif pca_path.suffix == '.joblib':
            import joblib
            pca_features = joblib.load(pca_path)
        else:
            # Try both formats
            if pca_path.with_suffix('.npy').exists():
                pca_features = np.load(pca_path.with_suffix('.npy'))
            elif pca_path.with_suffix('.joblib').exists():
                import joblib
                pca_features = joblib.load(pca_path.with_suffix('.joblib'))
            else:
                raise FileNotFoundError(f"PCA file not found: {pca_path}")
        
        # Handle different possible shapes
        if pca_features.ndim == 2:
            # If it's 2D, take the first row (assuming (n_samples, n_features))
            pca_features = pca_features[0]
        
        # Ensure correct dimensionality
        if len(pca_features) < n_components:
            # Pad with zeros if too short
            pca_features = np.pad(pca_features, (0, n_components - len(pca_features)))
        elif len(pca_features) > n_components:
            # Truncate if too long
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