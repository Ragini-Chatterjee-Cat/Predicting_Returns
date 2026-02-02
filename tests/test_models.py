"""
Tests for volatility models.
"""

import pytest
import numpy as np
import pandas as pd

from models.garch import GARCHModel, EGARCHModel, GJRGARCHModel
from models.ml_models import RandomForestVolatility, XGBoostVolatility, _check_tensorflow
from models.stochastic_vol import HestonModel
from evaluation.metrics import mse, rmse, mae, qlike, mincer_zarnowitz_regression

# Check if TensorFlow is available
HAS_TENSORFLOW = _check_tensorflow()


# Fixtures
@pytest.fixture
def sample_returns():
    """Generate sample returns for testing."""
    np.random.seed(42)
    n = 500
    returns = np.random.normal(0, 0.02, n)
    # Add some volatility clustering
    for i in range(1, n):
        if abs(returns[i-1]) > 0.03:
            returns[i] *= 1.5
    return pd.Series(returns, index=pd.date_range("2020-01-01", periods=n, freq="B"))


@pytest.fixture
def sample_variance():
    """Generate sample variance proxy."""
    np.random.seed(42)
    n = 100
    return np.random.exponential(0.0004, n)


# GARCH Model Tests
class TestGARCHModel:
    def test_fit(self, sample_returns):
        model = GARCHModel(p=1, q=1)
        model.fit(sample_returns)
        assert model.fitted is True
        assert model.params is not None

    def test_predict(self, sample_returns):
        model = GARCHModel(p=1, q=1)
        model.fit(sample_returns)
        forecast = model.predict(horizon=5)
        assert len(forecast) == 5
        assert all(f > 0 for f in forecast)

    def test_forecast_variance(self, sample_returns):
        model = GARCHModel(p=1, q=1)
        model.fit(sample_returns)
        forecasts = model.forecast_variance(steps=10)
        assert len(forecasts) == 10
        assert all(f > 0 for f in forecasts)

    def test_conditional_variance(self, sample_returns):
        model = GARCHModel(p=1, q=1)
        model.fit(sample_returns)
        cond_var = model.get_conditional_variance()
        assert cond_var is not None
        assert len(cond_var) == len(sample_returns)

    def test_unfitted_predict_raises(self):
        model = GARCHModel()
        with pytest.raises(ValueError):
            model.predict()


class TestEGARCHModel:
    def test_fit_and_predict(self, sample_returns):
        model = EGARCHModel(p=1, q=1)
        model.fit(sample_returns)
        assert model.fitted is True

        forecast = model.predict(horizon=5)
        assert len(forecast) == 5
        assert all(f > 0 for f in forecast)


class TestGJRGARCHModel:
    def test_fit_and_predict(self, sample_returns):
        model = GJRGARCHModel(p=1, o=1, q=1)
        model.fit(sample_returns)
        assert model.fitted is True

        forecast = model.predict(horizon=5)
        assert len(forecast) == 5
        assert all(f > 0 for f in forecast)


# ML Model Tests
class TestRandomForestVolatility:
    def test_fit(self, sample_returns):
        model = RandomForestVolatility(n_estimators=10, lookback=10)
        model.fit(sample_returns)
        assert model.fitted is True

    def test_predict(self, sample_returns):
        model = RandomForestVolatility(n_estimators=10, lookback=10)
        model.fit(sample_returns)
        forecast = model.predict(horizon=1)
        assert len(forecast) == 1
        assert forecast[0] >= 0

    def test_feature_importances(self, sample_returns):
        model = RandomForestVolatility(n_estimators=10, lookback=10)
        model.fit(sample_returns)
        importances = model.feature_importances
        assert importances is not None
        assert len(importances) > 0


class TestXGBoostVolatility:
    def test_fit_and_predict(self, sample_returns):
        model = XGBoostVolatility(n_estimators=10, lookback=10)
        model.fit(sample_returns)
        assert model.fitted is True

        forecast = model.predict(horizon=1)
        assert len(forecast) == 1
        assert forecast[0] >= 0


# Stochastic Volatility Tests
class TestHestonModel:
    def test_fit(self, sample_returns):
        model = HestonModel()
        model.fit(sample_returns)
        assert model.fitted is True
        assert model.params is not None

    def test_predict(self, sample_returns):
        model = HestonModel()
        model.fit(sample_returns)
        forecast = model.forecast_variance(steps=5)
        assert len(forecast) == 5
        assert all(f > 0 for f in forecast)

    def test_simulate(self, sample_returns):
        model = HestonModel()
        model.fit(sample_returns)
        returns, variance = model.simulate(n_steps=100, n_paths=10)
        assert returns.shape == (10, 100)
        assert variance.shape == (10, 100)


# Metrics Tests
class TestMetrics:
    def test_mse(self, sample_variance):
        predicted = sample_variance + np.random.normal(0, 0.0001, len(sample_variance))
        result = mse(sample_variance, predicted)
        assert result >= 0

    def test_rmse(self, sample_variance):
        predicted = sample_variance + np.random.normal(0, 0.0001, len(sample_variance))
        result = rmse(sample_variance, predicted)
        assert result >= 0
        assert result == np.sqrt(mse(sample_variance, predicted))

    def test_mae(self, sample_variance):
        predicted = sample_variance + np.random.normal(0, 0.0001, len(sample_variance))
        result = mae(sample_variance, predicted)
        assert result >= 0

    def test_qlike(self, sample_variance):
        predicted = sample_variance + np.random.normal(0, 0.0001, len(sample_variance))
        predicted = np.maximum(predicted, 1e-10)
        result = qlike(sample_variance, predicted)
        assert np.isfinite(result)

    def test_mincer_zarnowitz(self, sample_variance):
        # Perfect forecast should have alpha~0, beta~1
        predicted = sample_variance + np.random.normal(0, 0.00001, len(sample_variance))
        result = mincer_zarnowitz_regression(sample_variance, predicted)
        assert "alpha" in result
        assert "beta" in result
        assert "r_squared" in result


# Integration test
class TestIntegration:
    def test_full_pipeline(self, sample_returns):
        """Test the full pipeline from fitting to evaluation."""
        from evaluation.metrics import calculate_all_metrics

        # Fit GARCH model
        model = GARCHModel(p=1, q=1)
        model.fit(sample_returns)

        # Get conditional variance
        cond_var = model.get_conditional_variance()

        # Use squared returns as proxy
        realized = (sample_returns ** 2).values

        # Calculate metrics
        metrics = calculate_all_metrics(realized, cond_var)

        assert "mse" in metrics
        assert "rmse" in metrics
        assert "mae" in metrics
        assert "qlike" in metrics
        assert all(np.isfinite(v) for v in metrics.values() if isinstance(v, (int, float)))
