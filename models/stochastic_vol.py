"""
Stochastic volatility models.
"""

import numpy as np
import pandas as pd
from typing import Optional, Union, Tuple
from scipy.optimize import minimize
from scipy.stats import norm

from .base import VolatilityModel


class HestonModel(VolatilityModel):
    """
    Heston stochastic volatility model.

    The model is defined by:
    dS_t = μ * S_t * dt + √V_t * S_t * dW^S_t
    dV_t = κ * (θ - V_t) * dt + σ * √V_t * dW^V_t

    where:
    - κ (kappa): Speed of mean reversion
    - θ (theta): Long-run variance
    - σ (sigma): Volatility of volatility
    - ρ (rho): Correlation between price and variance innovations
    - V_0: Initial variance
    """

    def __init__(
        self,
        kappa: Optional[float] = None,
        theta: Optional[float] = None,
        sigma: Optional[float] = None,
        rho: Optional[float] = None,
        v0: Optional[float] = None,
        name: str = "Heston"
    ):
        """
        Initialize Heston model.

        Parameters
        ----------
        kappa : Optional[float]
            Speed of mean reversion (will be estimated if None)
        theta : Optional[float]
            Long-run variance (will be estimated if None)
        sigma : Optional[float]
            Volatility of volatility (will be estimated if None)
        rho : Optional[float]
            Correlation (will be estimated if None)
        v0 : Optional[float]
            Initial variance (will be estimated if None)
        name : str
            Model name
        """
        super().__init__(name=name)
        self.kappa = kappa
        self.theta = theta
        self.sigma = sigma
        self.rho = rho
        self.v0 = v0

        self._variance_path = None

    def _estimate_initial_params(self, returns: pd.Series) -> dict:
        """Estimate initial parameters using method of moments."""
        var_returns = returns.var()
        vol_returns = returns.std()

        # Simple initial estimates
        theta = var_returns  # Long-run variance
        v0 = var_returns  # Initial variance
        kappa = 2.0  # Mean reversion speed
        sigma = 0.3  # Vol of vol
        rho = -0.7  # Typically negative for equities

        return {
            "kappa": kappa,
            "theta": theta,
            "sigma": sigma,
            "rho": rho,
            "v0": v0
        }

    def _negative_log_likelihood(
        self,
        params: np.ndarray,
        returns: np.ndarray,
        dt: float = 1/252
    ) -> float:
        """
        Calculate negative log-likelihood for parameter estimation.

        Uses a simplified quasi-maximum likelihood approach.
        """
        kappa, theta, sigma, rho, v0 = params

        # Parameter constraints
        if kappa <= 0 or theta <= 0 or sigma <= 0 or v0 <= 0:
            return 1e10
        if abs(rho) >= 1:
            return 1e10
        # Feller condition: 2*kappa*theta > sigma^2
        if 2 * kappa * theta <= sigma ** 2:
            return 1e10

        n = len(returns)
        variance = np.zeros(n)
        variance[0] = v0

        log_likelihood = 0

        for t in range(1, n):
            # Euler discretization of variance process
            variance[t] = variance[t-1] + kappa * (theta - variance[t-1]) * dt
            variance[t] = max(variance[t], 1e-10)  # Ensure positive

            # Adjust for correlation effect
            std_t = np.sqrt(variance[t])
            if std_t > 0:
                log_likelihood += norm.logpdf(returns[t], loc=0, scale=std_t)

        return -log_likelihood

    def fit(self, returns: Union[pd.Series, np.ndarray]) -> "HestonModel":
        """
        Fit the Heston model using quasi-maximum likelihood.

        Parameters
        ----------
        returns : Union[pd.Series, np.ndarray]
            Return series

        Returns
        -------
        HestonModel
            Fitted model
        """
        if isinstance(returns, np.ndarray):
            returns = pd.Series(returns)

        self._returns = returns

        # Get initial parameters
        if all(p is not None for p in [self.kappa, self.theta, self.sigma, self.rho, self.v0]):
            init_params = [self.kappa, self.theta, self.sigma, self.rho, self.v0]
        else:
            init = self._estimate_initial_params(returns)
            init_params = [init["kappa"], init["theta"], init["sigma"], init["rho"], init["v0"]]

        # Bounds for parameters
        bounds = [
            (0.01, 20),      # kappa
            (1e-6, 0.1),     # theta
            (0.01, 2),       # sigma
            (-0.99, 0.99),   # rho
            (1e-6, 0.1)      # v0
        ]

        # Optimize
        result = minimize(
            self._negative_log_likelihood,
            init_params,
            args=(returns.values,),
            method="L-BFGS-B",
            bounds=bounds
        )

        self.kappa, self.theta, self.sigma, self.rho, self.v0 = result.x

        # Calculate variance path
        self._calculate_variance_path(returns.values)

        self.fitted = True

        return self

    def _calculate_variance_path(self, returns: np.ndarray, dt: float = 1/252):
        """Calculate the filtered variance path."""
        n = len(returns)
        self._variance_path = np.zeros(n)
        self._variance_path[0] = self.v0

        for t in range(1, n):
            self._variance_path[t] = self._variance_path[t-1] + \
                self.kappa * (self.theta - self._variance_path[t-1]) * dt
            self._variance_path[t] = max(self._variance_path[t], 1e-10)

    def predict(self, horizon: int = 1) -> np.ndarray:
        """
        Predict variance for the next period.

        Parameters
        ----------
        horizon : int
            Forecast horizon

        Returns
        -------
        np.ndarray
            Predicted variance
        """
        if not self.fitted:
            raise ValueError("Model must be fitted before prediction")

        return self.forecast_variance(horizon)

    def forecast_variance(self, steps: int = 1) -> np.ndarray:
        """
        Forecast variance for multiple steps ahead.

        Uses the analytical formula for conditional expected variance:
        E[V_t | V_0] = θ + (V_0 - θ) * exp(-κ * t)

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

        current_var = self._variance_path[-1]
        dt = 1 / 252  # Daily

        forecasts = []
        for t in range(1, steps + 1):
            # Analytical expectation formula
            expected_var = self.theta + (current_var - self.theta) * np.exp(-self.kappa * t * dt)
            forecasts.append(expected_var)

        return np.array(forecasts)

    def get_conditional_variance(self) -> Optional[np.ndarray]:
        """Get the filtered variance path."""
        if not self.fitted:
            return None
        return self._variance_path

    def simulate(
        self,
        n_steps: int,
        n_paths: int = 1000,
        dt: float = 1/252
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Simulate price and variance paths using the Heston model.

        Parameters
        ----------
        n_steps : int
            Number of time steps to simulate
        n_paths : int
            Number of simulation paths
        dt : float
            Time step size

        Returns
        -------
        Tuple[np.ndarray, np.ndarray]
            Simulated returns and variance paths (n_paths x n_steps)
        """
        if not self.fitted:
            raise ValueError("Model must be fitted before simulation")

        np.random.seed(42)

        returns = np.zeros((n_paths, n_steps))
        variance = np.zeros((n_paths, n_steps))
        variance[:, 0] = self._variance_path[-1]  # Start from last observed

        for t in range(1, n_steps):
            # Correlated random numbers
            z1 = np.random.standard_normal(n_paths)
            z2 = self.rho * z1 + np.sqrt(1 - self.rho**2) * np.random.standard_normal(n_paths)

            # Variance process (ensure non-negative using reflection)
            variance[:, t] = np.abs(
                variance[:, t-1] +
                self.kappa * (self.theta - variance[:, t-1]) * dt +
                self.sigma * np.sqrt(variance[:, t-1] * dt) * z2
            )

            # Return process
            returns[:, t] = np.sqrt(variance[:, t] * dt) * z1

        return returns, variance

    @property
    def params(self) -> Optional[dict]:
        """Get fitted parameters."""
        if not self.fitted:
            return None
        return {
            "kappa": self.kappa,
            "theta": self.theta,
            "sigma": self.sigma,
            "rho": self.rho,
            "v0": self.v0
        }
