"""
Configuration settings for the stock variance modeling project.
"""

# Tickers to analyze
TICKERS = ["AAPL", "MSFT", "GOOGL", "AMZN", "SPY"]

# Date range for historical data
START_DATE = "2018-01-01"
END_DATE = "2024-01-01"

# Model parameters
GARCH_P = 1
GARCH_Q = 1

# Rolling window settings for backtesting
TRAIN_WINDOW = 252 * 2  # 2 years of trading days
TEST_WINDOW = 21  # 1 month ahead forecast

# ML model parameters
LOOKBACK_PERIOD = 20  # Number of lagged features
LSTM_EPOCHS = 50
LSTM_BATCH_SIZE = 32

# Random seed for reproducibility
RANDOM_SEED = 42
