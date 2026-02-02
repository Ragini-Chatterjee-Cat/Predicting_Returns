# Stock Variance Modeling

A Python framework for comparing different volatility/variance forecasting models on stock data.

## Overview

This project implements and compares various approaches to forecasting stock return variance:

- **Traditional Econometric Models**: GARCH(1,1), EGARCH(1,1), GJR-GARCH(1,1)
- **Machine Learning Models**: Random Forest, XGBoost, LSTM
- **Stochastic Volatility Models**: Heston model

## Project Structure

```
Predicting_Returns/
├── config.py                    # Configuration (tickers, date ranges, etc.)
├── data/
│   ├── fetcher.py               # Yahoo Finance data fetching
│   └── preprocessing.py         # Returns calculation, feature engineering
├── models/
│   ├── base.py                  # Abstract base class for models
│   ├── garch.py                 # GARCH, EGARCH, GJR-GARCH models
│   ├── ml_models.py             # Random Forest, XGBoost, LSTM
│   └── stochastic_vol.py        # Heston model implementation
├── evaluation/
│   ├── metrics.py               # MSE, MAE, QLIKE, Mincer-Zarnowitz
│   └── backtesting.py           # Rolling window evaluation
├── notebooks/
│   ├── 01_exploratory_analysis.ipynb
│   └── 02_model_comparison.ipynb
└── tests/
    └── test_models.py
```

## Installation

```bash
# Clone the repository
git clone <repository-url>
cd Predicting_Returns

# Install dependencies
pip install -r requirements.txt
```

## Quick Start

### Fetch and Preprocess Data

```python
from data.fetcher import fetch_stock_data
from data.preprocessing import calculate_returns

# Fetch SPY data
data = fetch_stock_data('SPY', '2018-01-01', '2024-01-01')
returns = calculate_returns(data['close'].to_frame())
```

### Train a GARCH Model

```python
from models.garch import GARCHModel

model = GARCHModel(p=1, q=1)
model.fit(returns)

# Forecast variance for next 5 days
forecast = model.forecast_variance(steps=5)
print(forecast)
```

### Compare Multiple Models

```python
from models.garch import GARCHModel, EGARCHModel
from models.ml_models import RandomForestVolatility
from evaluation.backtesting import RollingWindowBacktest

models = {
    'GARCH': GARCHModel(p=1, q=1),
    'EGARCH': EGARCHModel(p=1, q=1),
    'RF': RandomForestVolatility(n_estimators=100)
}

backtest = RollingWindowBacktest(train_window=504, test_window=21)
results = backtest.run(returns, models)
metrics = backtest.evaluate(results)
print(metrics)
```

## Models

### GARCH Family (arch library)

- **GARCH(1,1)**: Standard Generalized Autoregressive Conditional Heteroskedasticity
- **EGARCH(1,1)**: Exponential GARCH, captures asymmetric volatility
- **GJR-GARCH(1,1)**: Captures leverage effect (negative returns → higher volatility)

### Machine Learning

- **Random Forest**: Ensemble of decision trees with lagged features
- **XGBoost**: Gradient boosting with lagged returns and variance
- **LSTM**: Recurrent neural network for sequence modeling

### Stochastic Volatility

- **Heston Model**: Mean-reverting variance process with correlation between price and variance

## Evaluation Metrics

- **MSE/RMSE/MAE**: Standard error metrics
- **QLIKE**: Quasi-likelihood loss (robust for variance forecasting)
- **Mincer-Zarnowitz**: Tests forecast unbiasedness (α=0, β=1)
- **Diebold-Mariano**: Statistical test for comparing forecast accuracy

## Running Tests

```bash
poetry run pytest tests/
```

## Streamlit Dashboard

Launch the interactive dashboard:

```bash
poetry run streamlit run app.py
```

The dashboard provides:
- **Data Overview**: Price charts, return distributions, summary statistics
- **Model Fitting**: Fit selected models and view conditional variance
- **Backtest Results**: Rolling window out-of-sample evaluation
- **Metrics Comparison**: Model rankings and performance metrics

## Configuration

Edit `config.py` to customize:

```python
TICKERS = ["AAPL", "MSFT", "GOOGL", "SPY"]
START_DATE = "2018-01-01"
END_DATE = "2024-01-01"
TRAIN_WINDOW = 504  # 2 years
TEST_WINDOW = 21    # 1 month
```

## Dependencies

Core dependencies (managed by Poetry):
- yfinance: Stock data fetching
- arch: GARCH models
- scikit-learn: ML models
- xgboost: Gradient boosting
- streamlit, plotly: Dashboard and visualization
- pandas, numpy, scipy: Data manipulation

Optional:
- tensorflow: LSTM neural network (install separately if needed)

## Installation with Poetry

```bash
# Install dependencies
poetry install

# Activate virtual environment
poetry shell
```

## License

MIT License
