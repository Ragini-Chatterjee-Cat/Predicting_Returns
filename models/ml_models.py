"""
Machine learning models for volatility forecasting.
"""

import numpy as np
import pandas as pd
from typing import Optional, Union, Tuple
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler

from .base import VolatilityModel


class RandomForestVolatility(VolatilityModel):
    """
    Random Forest model for variance prediction.

    Uses lagged returns and lagged squared returns as features.
    """

    def __init__(
        self,
        n_estimators: int = 100,
        max_depth: Optional[int] = 10,
        lookback: int = 20,
        random_state: int = 42,
        name: str = "RandomForest"
    ):
        """
        Initialize Random Forest volatility model.

        Parameters
        ----------
        n_estimators : int
            Number of trees
        max_depth : Optional[int]
            Maximum depth of trees
        lookback : int
            Number of lagged periods for features
        random_state : int
            Random seed
        name : str
            Model name
        """
        super().__init__(name=name)
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.lookback = lookback
        self.random_state = random_state

        self._model = RandomForestRegressor(
            n_estimators=n_estimators,
            max_depth=max_depth,
            random_state=random_state,
            n_jobs=-1
        )
        self._scaler = StandardScaler()
        self._last_features = None

    def _create_features(self, returns: pd.Series) -> Tuple[np.ndarray, np.ndarray]:
        """Create features from returns."""
        n = len(returns)
        features = []

        # Lagged returns and squared returns
        for i in range(1, self.lookback + 1):
            features.append(returns.shift(i).values)
            features.append((returns ** 2).shift(i).values)

        # Rolling statistics
        features.append(returns.rolling(5).mean().shift(1).values)
        features.append(returns.rolling(5).std().shift(1).values)
        features.append(returns.rolling(20).mean().shift(1).values)
        features.append(returns.rolling(20).std().shift(1).values)

        X = np.column_stack(features)
        y = (returns ** 2).values  # Target: squared returns

        # Remove NaN rows
        valid_mask = ~np.isnan(X).any(axis=1) & ~np.isnan(y)

        return X[valid_mask], y[valid_mask]

    def fit(self, returns: Union[pd.Series, np.ndarray]) -> "RandomForestVolatility":
        """
        Fit the Random Forest model.

        Parameters
        ----------
        returns : Union[pd.Series, np.ndarray]
            Return series

        Returns
        -------
        RandomForestVolatility
            Fitted model
        """
        if isinstance(returns, np.ndarray):
            returns = pd.Series(returns)

        self._returns = returns
        X, y = self._create_features(returns)

        # Scale features
        X_scaled = self._scaler.fit_transform(X)

        self._model.fit(X_scaled, y)
        self._last_features = X_scaled[-1:]
        self.fitted = True

        return self

    def predict(self, horizon: int = 1) -> np.ndarray:
        """
        Predict variance for the next period.

        Parameters
        ----------
        horizon : int
            Forecast horizon (only horizon=1 supported for simple prediction)

        Returns
        -------
        np.ndarray
            Predicted variance
        """
        if not self.fitted:
            raise ValueError("Model must be fitted before prediction")

        if horizon > 1:
            return self.forecast_variance(horizon)

        return self._model.predict(self._last_features)

    def forecast_variance(self, steps: int = 1) -> np.ndarray:
        """
        Forecast variance for multiple steps using recursive forecasting.

        Parameters
        ----------
        steps : int
            Number of steps to forecast

        Returns
        -------
        np.ndarray
            Forecasted variances
        """
        if not self.fitted:
            raise ValueError("Model must be fitted before prediction")

        forecasts = []
        current_features = self._last_features.copy()

        for _ in range(steps):
            pred = self._model.predict(current_features)[0]
            forecasts.append(pred)

            # Update features for next step (simplified recursive approach)
            # Shift lagged values and insert new prediction
            current_features = np.roll(current_features, 1, axis=1)
            current_features[0, 0] = 0  # Assume zero return for simplicity
            current_features[0, 1] = pred  # Insert predicted variance

        return np.array(forecasts)

    @property
    def feature_importances(self) -> Optional[np.ndarray]:
        """Get feature importances."""
        if not self.fitted:
            return None
        return self._model.feature_importances_


class XGBoostVolatility(VolatilityModel):
    """
    XGBoost model for variance prediction.
    """

    def __init__(
        self,
        n_estimators: int = 100,
        max_depth: int = 6,
        learning_rate: float = 0.1,
        lookback: int = 20,
        random_state: int = 42,
        name: str = "XGBoost"
    ):
        """
        Initialize XGBoost volatility model.

        Parameters
        ----------
        n_estimators : int
            Number of boosting rounds
        max_depth : int
            Maximum tree depth
        learning_rate : float
            Learning rate
        lookback : int
            Number of lagged periods for features
        random_state : int
            Random seed
        name : str
            Model name
        """
        super().__init__(name=name)
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.learning_rate = learning_rate
        self.lookback = lookback
        self.random_state = random_state

        # Import here to avoid issues if xgboost not installed
        from xgboost import XGBRegressor
        self._model = XGBRegressor(
            n_estimators=n_estimators,
            max_depth=max_depth,
            learning_rate=learning_rate,
            random_state=random_state,
            verbosity=0
        )
        self._scaler = StandardScaler()
        self._last_features = None

    def _create_features(self, returns: pd.Series) -> Tuple[np.ndarray, np.ndarray]:
        """Create features from returns."""
        features = []

        for i in range(1, self.lookback + 1):
            features.append(returns.shift(i).values)
            features.append((returns ** 2).shift(i).values)

        features.append(returns.rolling(5).mean().shift(1).values)
        features.append(returns.rolling(5).std().shift(1).values)
        features.append(returns.rolling(20).mean().shift(1).values)
        features.append(returns.rolling(20).std().shift(1).values)

        X = np.column_stack(features)
        y = (returns ** 2).values

        valid_mask = ~np.isnan(X).any(axis=1) & ~np.isnan(y)

        return X[valid_mask], y[valid_mask]

    def fit(self, returns: Union[pd.Series, np.ndarray]) -> "XGBoostVolatility":
        """Fit the XGBoost model."""
        if isinstance(returns, np.ndarray):
            returns = pd.Series(returns)

        self._returns = returns
        X, y = self._create_features(returns)

        X_scaled = self._scaler.fit_transform(X)

        self._model.fit(X_scaled, y)
        self._last_features = X_scaled[-1:]
        self.fitted = True

        return self

    def predict(self, horizon: int = 1) -> np.ndarray:
        """Predict variance for the next period."""
        if not self.fitted:
            raise ValueError("Model must be fitted before prediction")

        if horizon > 1:
            return self.forecast_variance(horizon)

        return self._model.predict(self._last_features)

    def forecast_variance(self, steps: int = 1) -> np.ndarray:
        """Forecast variance for multiple steps."""
        if not self.fitted:
            raise ValueError("Model must be fitted before prediction")

        forecasts = []
        current_features = self._last_features.copy()

        for _ in range(steps):
            pred = self._model.predict(current_features)[0]
            forecasts.append(pred)
            current_features = np.roll(current_features, 1, axis=1)
            current_features[0, 0] = 0
            current_features[0, 1] = pred

        return np.array(forecasts)

    @property
    def feature_importances(self) -> Optional[np.ndarray]:
        """Get feature importances."""
        if not self.fitted:
            return None
        return self._model.feature_importances_


def _check_tensorflow():
    """Check if TensorFlow is available."""
    try:
        import tensorflow
        return True
    except ImportError:
        return False


class LSTMVolatility(VolatilityModel):
    """
    LSTM neural network for sequence-based variance prediction.

    Requires TensorFlow to be installed (optional dependency).
    Install with: pip install tensorflow
    """

    def __init__(
        self,
        lookback: int = 20,
        units: int = 50,
        epochs: int = 50,
        batch_size: int = 32,
        random_state: int = 42,
        name: str = "LSTM"
    ):
        """
        Initialize LSTM volatility model.

        Parameters
        ----------
        lookback : int
            Sequence length for LSTM input
        units : int
            Number of LSTM units
        epochs : int
            Training epochs
        batch_size : int
            Batch size
        random_state : int
            Random seed
        name : str
            Model name

        Raises
        ------
        ImportError
            If TensorFlow is not installed
        """
        if not _check_tensorflow():
            raise ImportError(
                "TensorFlow is required for LSTMVolatility. "
                "Install it with: pip install tensorflow"
            )

        super().__init__(name=name)
        self.lookback = lookback
        self.units = units
        self.epochs = epochs
        self.batch_size = batch_size
        self.random_state = random_state

        self._model = None
        self._scaler = StandardScaler()
        self._last_sequence = None

    def _build_model(self, input_shape: Tuple[int, int]):
        """Build the LSTM model architecture."""
        import tensorflow as tf
        tf.random.set_seed(self.random_state)

        model = tf.keras.Sequential([
            tf.keras.layers.LSTM(
                self.units,
                input_shape=input_shape,
                return_sequences=True
            ),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.LSTM(self.units // 2),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.Dense(1)
        ])

        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
            loss="mse"
        )

        return model

    def _create_sequences(
        self,
        returns: pd.Series
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Create sequences for LSTM training."""
        # Features: returns and squared returns
        data = np.column_stack([returns.values, (returns ** 2).values])
        data_scaled = self._scaler.fit_transform(data)

        X, y = [], []

        for i in range(self.lookback, len(data_scaled)):
            X.append(data_scaled[i - self.lookback:i])
            y.append(data_scaled[i, 1])  # Squared return as target

        return np.array(X), np.array(y)

    def fit(self, returns: Union[pd.Series, np.ndarray]) -> "LSTMVolatility":
        """Fit the LSTM model."""
        if isinstance(returns, np.ndarray):
            returns = pd.Series(returns)

        self._returns = returns

        X, y = self._create_sequences(returns)

        self._model = self._build_model((X.shape[1], X.shape[2]))

        self._model.fit(
            X, y,
            epochs=self.epochs,
            batch_size=self.batch_size,
            validation_split=0.1,
            verbose=0
        )

        self._last_sequence = X[-1:]
        self.fitted = True

        return self

    def predict(self, horizon: int = 1) -> np.ndarray:
        """Predict variance for the next period."""
        if not self.fitted:
            raise ValueError("Model must be fitted before prediction")

        if horizon > 1:
            return self.forecast_variance(horizon)

        pred_scaled = self._model.predict(self._last_sequence, verbose=0)
        # Inverse transform (approximate, as we only have partial info)
        pred = pred_scaled[0, 0] * self._scaler.scale_[1] + self._scaler.mean_[1]

        return np.array([max(pred, 0)])  # Ensure non-negative

    def forecast_variance(self, steps: int = 1) -> np.ndarray:
        """Forecast variance for multiple steps."""
        if not self.fitted:
            raise ValueError("Model must be fitted before prediction")

        forecasts = []
        current_seq = self._last_sequence.copy()

        for _ in range(steps):
            pred_scaled = self._model.predict(current_seq, verbose=0)[0, 0]
            pred = pred_scaled * self._scaler.scale_[1] + self._scaler.mean_[1]
            forecasts.append(max(pred, 0))

            # Update sequence
            new_row = np.array([[0, pred_scaled]])  # Assume zero return
            current_seq = np.concatenate([current_seq[:, 1:, :], new_row.reshape(1, 1, 2)], axis=1)

        return np.array(forecasts)
