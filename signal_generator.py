import numpy as np
import pandas as pd
from scipy.stats import spearmanr, pearsonr
from typing import Dict, List, Optional, Tuple


def compute_cross_sectional_ic(
    predicted_returns: pd.Series,
    actual_returns: pd.Series,
    min_samples: int = 5
) -> Dict[str, float]:

    # Align indices
    common_idx = predicted_returns.index.intersection(actual_returns.index)
    if len(common_idx) < min_samples:
        return {'IC': np.nan, 'RankIC': np.nan}
    
    pred = predicted_returns.loc[common_idx].astype(float)
    actual = actual_returns.loc[common_idx].astype(float)
    
    # Remove NaN/Inf values
    valid_mask = (
        ~pred.isna() & ~actual.isna() & 
        np.isfinite(pred) & np.isfinite(actual)
    )
    
    if valid_mask.sum() < min_samples:
        return {'IC': np.nan, 'RankIC': np.nan}
    
    pred = pred[valid_mask]
    actual = actual[valid_mask]
    
    # Pearson IC
    try:
        ic, _ = pearsonr(pred, actual)
    except (ValueError, RuntimeWarning):
        ic = np.nan
    
    # Spearman Rank IC
    try:
        rank_ic, _ = spearmanr(pred, actual)
    except (ValueError, RuntimeWarning):
        rank_ic = np.nan
    
    return {'IC': ic, 'RankIC': rank_ic}


def compute_cross_sectional_hit_rate(
    predicted_returns: pd.Series,
    actual_returns: pd.Series,
    min_samples: int = 5
) -> float:
    # Align indices
    common_idx = predicted_returns.index.intersection(actual_returns.index)
    if len(common_idx) < min_samples:
        return np.nan
    
    pred = predicted_returns.loc[common_idx]
    actual = actual_returns.loc[common_idx]
    
    # Direction agreement (positive vs negative)
    hit = ((pred > 0) & (actual > 0)) | ((pred < 0) & (actual < 0))
    
    return hit.mean()





def rank_and_size(
    signals: pd.Series,
    vol_estimates: Optional[pd.Series] = None,
    long_quantile: float = 0.2,
    short_quantile: float = 0.2,
    vol_target: float = 0.15,
    max_leverage: float = 1.5,
    dollar_neutral: bool = True,
    min_positions_per_leg: int = 3,
) -> pd.Series:
    #  Input validation 
    signals = signals.dropna()
    if len(signals) < (min_positions_per_leg * 2):
        return pd.Series(dtype=float)
    
    n = len(signals)
    n_long = max(min_positions_per_leg, int(np.ceil(n * long_quantile)))
    n_short = max(min_positions_per_leg, int(np.ceil(n * short_quantile)))
    
    # Ensure we don't take more than available
    actual_long_frac = min(n_long / n, 0.5)
    actual_short_frac = min(n_short / n, 0.5)
    
    n_long = int(np.ceil(n * actual_long_frac))
    n_short = int(np.ceil(n * actual_short_frac))
    
    #  Rank and select 
    ranked = signals.sort_values(ascending=False)
    long_tickers = ranked.head(n_long).index
    short_tickers = ranked.tail(n_short).index
    
    # Initialize zero weights for all tickers
    all_tickers = signals.index
    weights = pd.Series(0.0, index=all_tickers)
    
    #  Size positions within each leg 
    if vol_estimates is not None and not vol_estimates.isna().all():
        # Clip volatility to prevent extreme weights
        # Floor: 5% annualized, Ceiling: 100% annualized
        vol = vol_estimates.clip(lower=0.05, upper=1.0)
        
        # Inverse-vol weights
        long_inv_vol = 1.0 / vol.reindex(long_tickers).fillna(1.0 / 0.05)
        short_inv_vol = 1.0 / vol.reindex(short_tickers).fillna(1.0 / 0.05)
        
        long_sum = long_inv_vol.sum()
        short_sum = short_inv_vol.sum()
        
        if long_sum > 0:
            long_weights = long_inv_vol / long_sum
        else:
            long_weights = pd.Series(1.0 / len(long_tickers), index=long_tickers)
        
        if short_sum > 0:
            short_weights = -short_inv_vol / short_sum
        else:
            short_weights = pd.Series(-1.0 / len(short_tickers), index=short_tickers)
    else:
        # Equal weight within each leg
        long_weights = pd.Series(1.0 / len(long_tickers), index=long_tickers)
        short_weights = pd.Series(-1.0 / len(short_tickers), index=short_tickers)
    
    # Assign weights
    weights.loc[long_weights.index] = long_weights.values
    weights.loc[short_weights.index] = short_weights.values
    
    #  Enforce dollar neutrality 
    if dollar_neutral:
        gross_long = weights[weights > 0].sum()
        gross_short = abs(weights[weights < 0].sum())
        
        if gross_long > 0 and gross_short > 0:
            # Scale the smaller leg to match the larger
            if gross_long > gross_short:
                # Scale up shorts
                scale = gross_long / gross_short
                weights[weights < 0] *= scale
            else:
                # Scale up longs
                scale = gross_short / gross_long
                weights[weights > 0] *= scale
    
    #  Scale to volatility target 
    portfolio_vol = _estimate_portfolio_vol(weights, vol_estimates)
    
    if portfolio_vol > 0 and not np.isnan(portfolio_vol):
        scale = vol_target / portfolio_vol
        gross_leverage = abs(weights).sum()
        
        if gross_leverage > 0:
            # Cap leverage
            scale = min(scale, max_leverage / gross_leverage)
        
        weights = weights * scale
    
    return weights


def _estimate_portfolio_vol(
    weights: pd.Series,
    vol_estimates: Optional[pd.Series] = None
) -> float:
    
    if vol_estimates is None or vol_estimates.isna().all():
        return np.nan
    
    common = weights.index.intersection(vol_estimates.index)
    if len(common) == 0:
        return np.nan
    
    w = weights.loc[common].values
    v = vol_estimates.loc[common].values
    
    # Remove NaN
    valid = ~np.isnan(v)
    if valid.sum() == 0:
        return np.nan
    
    w = w[valid]
    v = v[valid]
    
    # Diagonal covariance: portfolio_var = sum(w_i^2 * sigma_i^2)
    portfolio_var = np.sum(w ** 2 * v ** 2)
    
    return np.sqrt(portfolio_var)


def compute_portfolio_returns(
    signal_dates: List[pd.Timestamp],
    weights_by_date: Dict[pd.Timestamp, pd.Series],
    daily_returns: pd.DataFrame,
    transaction_cost: float = 0.0,
) -> pd.Series:
    
    if len(signal_dates) == 0:
        return pd.Series(dtype=float)
    
    portfolio_returns = {}
    previous_weights = None
    
    for i, signal_date in enumerate(signal_dates):
        weights = weights_by_date.get(signal_date)
        if weights is None or len(weights) == 0:
            previous_weights = weights
            continue
        
        # Determine holding period end
        if i < len(signal_dates) - 1:
            next_signal = signal_dates[i + 1]
        else:
            next_signal = daily_returns.index[-1]
        
        # Get trading days in holding period (EXCLUDE entry day)
        mask = (daily_returns.index > signal_date) & (daily_returns.index <= next_signal)
        holding_returns = daily_returns.loc[mask]
        
        if len(holding_returns) == 0:
            previous_weights = weights
            continue
        
        # Find common tickers
        common_tickers = [t for t in weights.index if t in daily_returns.columns]
        if not common_tickers:
            previous_weights = weights
            continue
        
        w = weights.loc[common_tickers].values
        
        # Apply transaction cost if rebalancing
        if transaction_cost > 0 and previous_weights is not None:
            prev_w = previous_weights.reindex(common_tickers, fill_value=0).values
            turnover = 0.5 * np.sum(np.abs(w - prev_w))
            cost = turnover * transaction_cost
            if cost > 0 and len(holding_returns) > 0:
                # Amortize cost over holding period
                daily_cost = cost / len(holding_returns)
            else:
                daily_cost = 0
        else:
            daily_cost = 0
        
        # Compute daily portfolio returns
        for day in holding_returns.index:
            day_rets = holding_returns.loc[day, common_tickers].values
            port_ret = np.dot(w, day_rets) - daily_cost
            portfolio_returns[day] = port_ret
        
        previous_weights = weights
    
    if not portfolio_returns:
        return pd.Series(dtype=float)
    
    result = pd.Series(portfolio_returns, name='portfolio_return').sort_index()
    return result


def compute_sharpe_ratio(
    returns: pd.Series,
    risk_free_rate: float = 0.0,
    periods_per_year: int = 252
) -> float:
    
    returns = returns.dropna()
    if len(returns) < 2:
        return 0.0
    
    excess = returns - risk_free_rate / periods_per_year
    
    if excess.std() == 0:
        return 0.0
    
    return (excess.mean() / excess.std()) * np.sqrt(periods_per_year)





def compute_max_drawdown(returns: pd.Series) -> float:
    
    returns = returns.dropna()
    if len(returns) == 0:
        return 0.0
    
    wealth = (1 + returns).cumprod()
    peak = wealth.cummax()
    drawdown = (wealth - peak) / peak
    
    return float(drawdown.min())


def compute_turnover(
    weights_by_date: Dict[pd.Timestamp, pd.Series],
    signal_dates: List[pd.Timestamp]
) -> float:
    
    if len(signal_dates) < 2:
        return 0.0
    
    turnovers = []
    
    for i in range(1, len(signal_dates)):
        prev_date = signal_dates[i - 1]
        curr_date = signal_dates[i]
        
        w_prev = weights_by_date.get(prev_date, pd.Series(dtype=float))
        w_curr = weights_by_date.get(curr_date, pd.Series(dtype=float))
        
        if len(w_prev) == 0 and len(w_curr) == 0:
            continue
        
        # Align tickers
        all_tickers = w_prev.index.union(w_curr.index)
        w_prev = w_prev.reindex(all_tickers, fill_value=0.0)
        w_curr = w_curr.reindex(all_tickers, fill_value=0.0)
        
        # One-way turnover
        turnover = 0.5 * abs(w_curr - w_prev).sum()
        turnovers.append(turnover)
    
    return float(np.mean(turnovers)) if turnovers else 0.0
