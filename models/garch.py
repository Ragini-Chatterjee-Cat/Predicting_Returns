"""
GARCH family models for volatility forecasting.
"""

import numpy as np
import pandas as pd
from typing import Optional, Union
from arch import arch_model

from .base import VolatilityModel


class GARCHModel(VolatilityModel):
    """
    Standard GARCH(p,q) model.

    The variance equation is:
    σ²_t = ω + α * ε²_{t-1} + β * σ²_{t-1}
    """

    def __init__(self, p: int = 1, q: int = 1, name: str = "GARCH"):
        """
        Initialize GARCH model.

        Parameters
        ----------
        p : int
            Order of the GARCH term (lagged variance)
        q : int
            Order of the ARCH term (lagged squared residuals)
        name : str
            Model name
        """
        super().__init__(name=name)
        self.p = p
        self.q = q
        self._model = None
        self._result = None

    def fit(self, returns: Union[pd.Series, np.ndarray]) -> "GARCHModel":
        """
        Fit the GARCH model to returns.

        Parameters
        ----------
        returns : Union[pd.Series, np.ndarray]
            Return series (should be in percentage points, not decimals)

        Returns
        -------
        GARCHModel
            Fitted model
        """
        if isinstance(returns, np.ndarray):
            returns = pd.Series(returns)

        self._returns = returns

        # Scale returns to percentage points for numerical stability
        scaled_returns = returns * 100

        self._model = arch_model(
            scaled_returns,
            vol="Garch",
            p=self.p,
            q=self.q,
            mean="Constant",
            rescale=False
        )

        self._result = self._model.fit(disp="off")
        self.fitted = True

        return self

    def predict(self, horizon: int = 1) -> np.ndarray:
        """
        Predict variance for the next horizon periods.

        Parameters
        ----------
        horizon : int
            Forecast horizon

        Returns
        -------
        np.ndarray
            Forecasted variance (in decimal form, not percentage)
        """
        if not self.fitted:
            raise ValueError("Model must be fitted before prediction")

        forecasts = self._result.forecast(horizon=horizon)
        # Convert back from percentage points squared to decimal squared
        return forecasts.variance.iloc[-1].values / 10000

    def forecast_variance(self, steps: int = 1) -> np.ndarray:
        """
        Forecast variance for multiple steps.

        Parameters
        ----------
        steps : int
            Number of steps to forecast

        Returns
        -------
        np.ndarray
            Forecasted variances
        """
        return self.predict(horizon=steps)

    def get_conditional_variance(self) -> Optional[np.ndarray]:
        """
        Get in-sample conditional variance.

        Returns
        -------
        Optional[np.ndarray]
            Conditional variance series
        """
        if not self.fitted:
            return None
        # Convert back from percentage points squared
        return self._result.conditional_volatility.values ** 2 / 10000

    @property
    def params(self) -> Optional[pd.Series]:
        """Get fitted parameters."""
        if not self.fitted:
            return None
        return self._result.params


class EGARCHModel(VolatilityModel):
    """
    Exponential GARCH model that captures asymmetric volatility effects.

    The log variance equation is:
    log(σ²_t) = ω + α * |z_{t-1}| + γ * z_{t-1} + β * log(σ²_{t-1})

    where z_t = ε_t / σ_t (standardized residuals)
    """

    def __init__(self, p: int = 1, q: int = 1, name: str = "EGARCH"):
        """
        Initialize EGARCH model.

        Parameters
        ----------
        p : int
            Order of the GARCH term
        q : int
            Order of the ARCH term
        name : str
            Model name
        """
        super().__init__(name=name)
        self.p = p
        self.q = q
        self._model = None
        self._result = None

    def fit(self, returns: Union[pd.Series, np.ndarray]) -> "EGARCHModel":
        """
        Fit the EGARCH model.

        Parameters
        ----------
        returns : Union[pd.Series, np.ndarray]
            Return series

        Returns
        -------
        EGARCHModel
            Fitted model
        """
        if isinstance(returns, np.ndarray):
            returns = pd.Series(returns)

        self._returns = returns
        scaled_returns = returns * 100

        self._model = arch_model(
            scaled_returns,
            vol="EGARCH",
            p=self.p,
            q=self.q,
            mean="Constant",
            rescale=False
        )

        self._result = self._model.fit(disp="off")
        self.fitted = True

        return self

    def predict(self, horizon: int = 1) -> np.ndarray:
        """Predict variance for the next horizon periods."""
        if not self.fitted:
            raise ValueError("Model must be fitted before prediction")

        # EGARCH only supports analytic forecasts for horizon=1
        # Use simulation for longer horizons
        if horizon > 1:
            forecasts = self._result.forecast(horizon=horizon, method="simulation", simulations=1000)
        else:
            forecasts = self._result.forecast(horizon=horizon)
        return forecasts.variance.iloc[-1].values / 10000

    def forecast_variance(self, steps: int = 1) -> np.ndarray:
        """Forecast variance for multiple steps."""
        return self.predict(horizon=steps)

    def get_conditional_variance(self) -> Optional[np.ndarray]:
        """Get in-sample conditional variance."""
        if not self.fitted:
            return None
        return self._result.conditional_volatility.values ** 2 / 10000

    @property
    def params(self) -> Optional[pd.Series]:
        """Get fitted parameters."""
        if not self.fitted:
            return None
        return self._result.params


class GJRGARCHModel(VolatilityModel):
    """
    GJR-GARCH model (Glosten-Jagannathan-Runkle) that captures leverage effects.

    The variance equation is:
    σ²_t = ω + (α + γ * I_{t-1}) * ε²_{t-1} + β * σ²_{t-1}

    where I_{t-1} = 1 if ε_{t-1} < 0 (indicator for negative shocks)
    """

    def __init__(self, p: int = 1, o: int = 1, q: int = 1, name: str = "GJR-GARCH"):
        """
        Initialize GJR-GARCH model.

        Parameters
        ----------
        p : int
            Order of the GARCH term
        o : int
            Order of the asymmetric term
        q : int
            Order of the ARCH term
        name : str
            Model name
        """
        super().__init__(name=name)
        self.p = p
        self.o = o
        self.q = q
        self._model = None
        self._result = None

    def fit(self, returns: Union[pd.Series, np.ndarray]) -> "GJRGARCHModel":
        """
        Fit the GJR-GARCH model.

        Parameters
        ----------
        returns : Union[pd.Series, np.ndarray]
            Return series

        Returns
        -------
        GJRGARCHModel
            Fitted model
        """
        if isinstance(returns, np.ndarray):
            returns = pd.Series(returns)

        self._returns = returns
        scaled_returns = returns * 100

        self._model = arch_model(
            scaled_returns,
            vol="Garch",
            p=self.p,
            o=self.o,
            q=self.q,
            mean="Constant",
            rescale=False
        )

        self._result = self._model.fit(disp="off")
        self.fitted = True

        return self

    def predict(self, horizon: int = 1) -> np.ndarray:
        """Predict variance for the next horizon periods."""
        if not self.fitted:
            raise ValueError("Model must be fitted before prediction")

        forecasts = self._result.forecast(horizon=horizon)
        return forecasts.variance.iloc[-1].values / 10000

    def forecast_variance(self, steps: int = 1) -> np.ndarray:
        """Forecast variance for multiple steps."""
        return self.predict(horizon=steps)

    def get_conditional_variance(self) -> Optional[np.ndarray]:
        """Get in-sample conditional variance."""
        if not self.fitted:
            return None
        return self._result.conditional_volatility.values ** 2 / 10000

    @property
    def params(self) -> Optional[pd.Series]:
        """Get fitted parameters."""
        if not self.fitted:
            return None
        return self._result.params
