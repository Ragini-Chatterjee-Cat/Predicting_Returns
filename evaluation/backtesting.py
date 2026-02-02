"""
Backtesting framework for volatility models.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Type, Union
from tqdm import tqdm

from models.base import VolatilityModel
from .metrics import calculate_all_metrics


class RollingWindowBacktest:
    """
    Rolling window backtesting framework for volatility models.

    Performs out-of-sample evaluation using expanding or rolling windows.
    """

    def __init__(
        self,
        train_window: int = 504,
        test_window: int = 21,
        step_size: int = 21,
        expanding: bool = False
    ):
        """
        Initialize the backtesting framework.

        Parameters
        ----------
        train_window : int
            Size of the training window (number of observations)
        test_window : int
            Size of the test window (forecast horizon)
        step_size : int
            Number of observations to move forward at each iteration
        expanding : bool
            If True, use expanding window; if False, use rolling window
        """
        self.train_window = train_window
        self.test_window = test_window
        self.step_size = step_size
        self.expanding = expanding

        self.results: Dict[str, Dict] = {}

    def run(
        self,
        returns: pd.Series,
        models: Dict[str, VolatilityModel],
        variance_proxy: Optional[pd.Series] = None,
        verbose: bool = True
    ) -> pd.DataFrame:
        """
        Run the backtest for all models.

        Parameters
        ----------
        returns : pd.Series
            Return series
        models : Dict[str, VolatilityModel]
            Dictionary of model instances {name: model}
        variance_proxy : Optional[pd.Series]
            Proxy for realized variance (default: squared returns)
        verbose : bool
            Whether to show progress bar

        Returns
        -------
        pd.DataFrame
            DataFrame with forecasts and realized values for each model
        """
        if variance_proxy is None:
            variance_proxy = returns ** 2

        n = len(returns)
        n_iterations = (n - self.train_window - self.test_window) // self.step_size + 1

        if n_iterations <= 0:
            raise ValueError(
                f"Not enough data for backtesting. Need at least "
                f"{self.train_window + self.test_window} observations, got {n}"
            )

        # Initialize results storage
        all_forecasts = {name: [] for name in models}
        all_realized = []
        all_dates = []

        # Progress bar
        iterator = range(n_iterations)
        if verbose:
            iterator = tqdm(iterator, desc="Backtesting")

        for i in iterator:
            # Define train/test split
            if self.expanding:
                train_start = 0
            else:
                train_start = i * self.step_size

            train_end = self.train_window + i * self.step_size
            test_end = train_end + self.test_window

            if test_end > n:
                break

            train_returns = returns.iloc[train_start:train_end]
            test_variance = variance_proxy.iloc[train_end:test_end]

            # Store realized variance
            all_realized.extend(test_variance.values)
            all_dates.extend(test_variance.index)

            # Fit and forecast each model
            for name, model in models.items():
                try:
                    # Create a fresh instance for each iteration
                    model_instance = model.__class__(
                        **{k: v for k, v in model.__dict__.items()
                           if not k.startswith("_") and k != "fitted"}
                    )
                    model_instance.fit(train_returns)
                    forecast = model_instance.forecast_variance(self.test_window)
                    all_forecasts[name].extend(forecast)
                except Exception as e:
                    if verbose:
                        print(f"Warning: {name} failed at iteration {i}: {e}")
                    all_forecasts[name].extend([np.nan] * self.test_window)

        # Create results DataFrame
        results_df = pd.DataFrame(index=all_dates)
        results_df["realized"] = all_realized

        for name in models:
            results_df[name] = all_forecasts[name][:len(all_realized)]

        self.results_df = results_df

        return results_df

    def evaluate(self, results_df: Optional[pd.DataFrame] = None) -> pd.DataFrame:
        """
        Evaluate all models using various metrics.

        Parameters
        ----------
        results_df : Optional[pd.DataFrame]
            Results DataFrame (uses stored results if None)

        Returns
        -------
        pd.DataFrame
            DataFrame with metrics for each model
        """
        if results_df is None:
            results_df = self.results_df

        realized = results_df["realized"].values
        model_names = [col for col in results_df.columns if col != "realized"]

        metrics_list = []

        for name in model_names:
            predicted = results_df[name].values

            # Remove NaN values
            valid_mask = ~(np.isnan(predicted) | np.isnan(realized))
            if valid_mask.sum() == 0:
                continue

            metrics = calculate_all_metrics(
                realized[valid_mask],
                predicted[valid_mask]
            )
            metrics["model"] = name
            metrics_list.append(metrics)

        metrics_df = pd.DataFrame(metrics_list)
        metrics_df = metrics_df.set_index("model")

        return metrics_df

    def plot_forecasts(
        self,
        results_df: Optional[pd.DataFrame] = None,
        models_to_plot: Optional[List[str]] = None,
        figsize: tuple = (14, 6)
    ):
        """
        Plot forecasted vs realized variance.

        Parameters
        ----------
        results_df : Optional[pd.DataFrame]
            Results DataFrame
        models_to_plot : Optional[List[str]]
            List of model names to plot (default: all)
        figsize : tuple
            Figure size
        """
        import matplotlib.pyplot as plt

        if results_df is None:
            results_df = self.results_df

        if models_to_plot is None:
            models_to_plot = [col for col in results_df.columns if col != "realized"]

        fig, ax = plt.subplots(figsize=figsize)

        # Plot realized variance
        ax.plot(
            results_df.index,
            results_df["realized"],
            label="Realized",
            color="black",
            linewidth=1.5,
            alpha=0.7
        )

        # Plot each model's forecasts
        colors = plt.cm.tab10(np.linspace(0, 1, len(models_to_plot)))
        for name, color in zip(models_to_plot, colors):
            ax.plot(
                results_df.index,
                results_df[name],
                label=name,
                color=color,
                alpha=0.7
            )

        ax.set_xlabel("Date")
        ax.set_ylabel("Variance")
        ax.set_title("Volatility Forecasts vs Realized Variance")
        ax.legend(loc="upper right")
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        return fig, ax

    def plot_cumulative_loss(
        self,
        results_df: Optional[pd.DataFrame] = None,
        loss_func: str = "mse",
        figsize: tuple = (14, 6)
    ):
        """
        Plot cumulative loss over time for model comparison.

        Parameters
        ----------
        results_df : Optional[pd.DataFrame]
            Results DataFrame
        loss_func : str
            Loss function ('mse', 'mae', 'qlike')
        figsize : tuple
            Figure size
        """
        import matplotlib.pyplot as plt

        if results_df is None:
            results_df = self.results_df

        model_names = [col for col in results_df.columns if col != "realized"]
        realized = results_df["realized"].values

        fig, ax = plt.subplots(figsize=figsize)

        colors = plt.cm.tab10(np.linspace(0, 1, len(model_names)))

        for name, color in zip(model_names, colors):
            predicted = results_df[name].values

            if loss_func == "mse":
                loss = (realized - predicted) ** 2
            elif loss_func == "mae":
                loss = np.abs(realized - predicted)
            elif loss_func == "qlike":
                predicted_safe = np.maximum(predicted, 1e-10)
                realized_safe = np.maximum(realized, 1e-10)
                loss = np.log(predicted_safe) + realized_safe / predicted_safe
            else:
                raise ValueError(f"Unknown loss function: {loss_func}")

            cumulative_loss = np.nancumsum(loss)
            ax.plot(
                results_df.index,
                cumulative_loss,
                label=name,
                color=color
            )

        ax.set_xlabel("Date")
        ax.set_ylabel(f"Cumulative {loss_func.upper()} Loss")
        ax.set_title(f"Cumulative {loss_func.upper()} Loss Over Time")
        ax.legend(loc="upper left")
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        return fig, ax
