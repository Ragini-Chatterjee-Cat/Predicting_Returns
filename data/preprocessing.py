"""
Data preprocessing utilities for financial time series.
"""

import numpy as np
import pandas as pd
from typing import Optional, Tuple


def calculate_returns(
    prices: pd.DataFrame,
    method: str = "log"
) -> pd.DataFrame:
    """
    Calculate returns from price data.

    Parameters
    ----------
    prices : pd.DataFrame
        Price data (can be single or multiple columns)
    method : str
        'log' for log returns, 'simple' for simple returns

    Returns
    -------
    pd.DataFrame
        Returns data
    """
    if method == "log":
        returns = np.log(prices / prices.shift(1))
    elif method == "simple":
        returns = prices.pct_change()
    else:
        raise ValueError(f"Unknown method: {method}. Use 'log' or 'simple'")

    return returns.dropna()


def calculate_realized_variance(
    returns: pd.Series,
    window: int = 21
) -> pd.Series:
    """
    Calculate realized variance over a rolling window.

    Parameters
    ----------
    returns : pd.Series
        Return series
    window : int
        Rolling window size (default 21 for monthly)

    Returns
    -------
    pd.Series
        Realized variance series
    """
    return returns.rolling(window=window).var()


def calculate_realized_volatility(
    returns: pd.Series,
    window: int = 21,
    annualize: bool = True,
    trading_days: int = 252
) -> pd.Series:
    """
    Calculate realized volatility over a rolling window.

    Parameters
    ----------
    returns : pd.Series
        Return series
    window : int
        Rolling window size
    annualize : bool
        Whether to annualize the volatility
    trading_days : int
        Number of trading days per year

    Returns
    -------
    pd.Series
        Realized volatility series
    """
    vol = returns.rolling(window=window).std()

    if annualize:
        vol = vol * np.sqrt(trading_days)

    return vol


def clean_data(
    data: pd.DataFrame,
    method: str = "ffill",
    max_missing_pct: float = 0.1
) -> pd.DataFrame:
    """
    Clean data by handling missing values.

    Parameters
    ----------
    data : pd.DataFrame
        Input data
    method : str
        Method for filling missing values ('ffill', 'bfill', 'interpolate', 'drop')
    max_missing_pct : float
        Maximum allowed percentage of missing values per column

    Returns
    -------
    pd.DataFrame
        Cleaned data
    """
    # Check missing data percentage
    missing_pct = data.isnull().sum() / len(data)
    cols_to_drop = missing_pct[missing_pct > max_missing_pct].index

    if len(cols_to_drop) > 0:
        print(f"Dropping columns with >{max_missing_pct*100}% missing: {list(cols_to_drop)}")
        data = data.drop(columns=cols_to_drop)

    # Handle remaining missing values
    if method == "ffill":
        data = data.ffill().bfill()
    elif method == "bfill":
        data = data.bfill().ffill()
    elif method == "interpolate":
        data = data.interpolate(method="time")
    elif method == "drop":
        data = data.dropna()
    else:
        raise ValueError(f"Unknown method: {method}")

    return data


def create_features(
    returns: pd.Series,
    lookback: int = 20
) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Create features for ML models from return series.

    Parameters
    ----------
    returns : pd.Series
        Return series
    lookback : int
        Number of lagged periods to include

    Returns
    -------
    Tuple[pd.DataFrame, pd.Series]
        Features DataFrame and target Series (squared returns as variance proxy)
    """
    df = pd.DataFrame(index=returns.index)

    # Lagged returns
    for i in range(1, lookback + 1):
        df[f"ret_lag_{i}"] = returns.shift(i)

    # Lagged squared returns (variance proxy)
    squared_returns = returns ** 2
    for i in range(1, lookback + 1):
        df[f"var_lag_{i}"] = squared_returns.shift(i)

    # Rolling statistics
    df["rolling_mean_5"] = returns.rolling(5).mean().shift(1)
    df["rolling_std_5"] = returns.rolling(5).std().shift(1)
    df["rolling_mean_20"] = returns.rolling(20).mean().shift(1)
    df["rolling_std_20"] = returns.rolling(20).std().shift(1)

    # Target: next period squared return (variance proxy)
    target = squared_returns

    # Align and drop NaN
    valid_idx = df.dropna().index

    return df.loc[valid_idx], target.loc[valid_idx]
