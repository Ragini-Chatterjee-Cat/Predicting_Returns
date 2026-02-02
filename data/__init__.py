"""
Data module for fetching and preprocessing stock data.
"""

from .fetcher import fetch_stock_data, fetch_multiple_stocks
from .preprocessing import calculate_returns, calculate_realized_variance, clean_data

__all__ = [
    "fetch_stock_data",
    "fetch_multiple_stocks",
    "calculate_returns",
    "calculate_realized_variance",
    "clean_data",
]
