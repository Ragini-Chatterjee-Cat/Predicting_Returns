"""
Abstract base class for volatility models.
"""

from abc import ABC, abstractmethod
import numpy as np
import pandas as pd
from typing import Optional, Union


class VolatilityModel(ABC):
    """
    Abstract base class for all volatility models.

    All volatility models should inherit from this class and implement
    the required methods.
    """

    def __init__(self, name: str = "BaseModel"):
        """
        Initialize the volatility model.

        Parameters
        ----------
        name : str
            Name of the model
        """
        self.name = name
        self.fitted = False
        self._returns = None

    @abstractmethod
    def fit(self, returns: Union[pd.Series, np.ndarray]) -> "VolatilityModel":
        """
        Fit the model to historical returns.

        Parameters
        ----------
        returns : Union[pd.Series, np.ndarray]
            Historical return series

        Returns
        -------
        VolatilityModel
            Fitted model instance
        """
        pass

    @abstractmethod
    def predict(self, horizon: int = 1) -> np.ndarray:
        """
        Predict future volatility/variance.

        Parameters
        ----------
        horizon : int
            Number of steps ahead to forecast

        Returns
        -------
        np.ndarray
            Forecasted variance/volatility
        """
        pass

    @abstractmethod
    def forecast_variance(self, steps: int = 1) -> np.ndarray:
        """
        Forecast variance for multiple steps ahead.

        Parameters
        ----------
        steps : int
            Number of steps to forecast

        Returns
        -------
        np.ndarray
            Array of forecasted variances
        """
        pass

    def get_conditional_variance(self) -> Optional[np.ndarray]:
        """
        Get the in-sample conditional variance.

        Returns
        -------
        Optional[np.ndarray]
            Conditional variance series if available
        """
        return None

    def __repr__(self) -> str:
        status = "fitted" if self.fitted else "not fitted"
        return f"{self.__class__.__name__}(name='{self.name}', {status})"
