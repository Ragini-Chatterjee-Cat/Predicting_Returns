"""
Yahoo Finance data fetching utilities.
"""

import yfinance as yf
import pandas as pd
from typing import Optional, List, Union


def fetch_stock_data(
    ticker: str,
    start_date: str,
    end_date: str,
    interval: str = "1d"
) -> pd.DataFrame:
    """
    Fetch OHLCV data for a single stock from Yahoo Finance.

    Parameters
    ----------
    ticker : str
        Stock ticker symbol (e.g., 'AAPL')
    start_date : str
        Start date in 'YYYY-MM-DD' format
    end_date : str
        End date in 'YYYY-MM-DD' format
    interval : str
        Data interval ('1d', '1h', '5m', etc.)

    Returns
    -------
    pd.DataFrame
        DataFrame with OHLCV data
    """
    stock = yf.Ticker(ticker)
    df = stock.history(start=start_date, end=end_date, interval=interval)

    if df.empty:
        raise ValueError(f"No data found for ticker {ticker}")

    # Standardize column names
    df.columns = [col.lower().replace(" ", "_") for col in df.columns]
    df.index.name = "date"

    return df


def fetch_multiple_stocks(
    tickers: List[str],
    start_date: str,
    end_date: str,
    interval: str = "1d",
    price_column: str = "close"
) -> pd.DataFrame:
    """
    Fetch price data for multiple stocks and return as a single DataFrame.

    Parameters
    ----------
    tickers : List[str]
        List of stock ticker symbols
    start_date : str
        Start date in 'YYYY-MM-DD' format
    end_date : str
        End date in 'YYYY-MM-DD' format
    interval : str
        Data interval
    price_column : str
        Which price to extract ('open', 'high', 'low', 'close')

    Returns
    -------
    pd.DataFrame
        DataFrame with price data for all tickers (columns = tickers)
    """
    data = yf.download(
        tickers,
        start=start_date,
        end=end_date,
        interval=interval,
        progress=False
    )

    if data.empty:
        raise ValueError("No data found for the specified tickers")

    # Handle both single and multiple tickers
    if len(tickers) == 1:
        prices = data[price_column.capitalize()].to_frame(name=tickers[0])
    else:
        prices = data[price_column.capitalize()]

    prices.index.name = "date"

    return prices
