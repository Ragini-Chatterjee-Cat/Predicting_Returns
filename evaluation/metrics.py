"""
Evaluation metrics for volatility forecasting models.
"""

import numpy as np
import pandas as pd
from typing import Dict, Tuple, Optional
from scipy import stats


def mse(actual: np.ndarray, predicted: np.ndarray) -> float:
    """
    Calculate Mean Squared Error.

    Parameters
    ----------
    actual : np.ndarray
        Actual (realized) variance
    predicted : np.ndarray
        Predicted variance

    Returns
    -------
    float
        MSE value
    """
    return np.mean((actual - predicted) ** 2)


def rmse(actual: np.ndarray, predicted: np.ndarray) -> float:
    """
    Calculate Root Mean Squared Error.

    Parameters
    ----------
    actual : np.ndarray
        Actual variance
    predicted : np.ndarray
        Predicted variance

    Returns
    -------
    float
        RMSE value
    """
    return np.sqrt(mse(actual, predicted))


def mae(actual: np.ndarray, predicted: np.ndarray) -> float:
    """
    Calculate Mean Absolute Error.

    Parameters
    ----------
    actual : np.ndarray
        Actual variance
    predicted : np.ndarray
        Predicted variance

    Returns
    -------
    float
        MAE value
    """
    return np.mean(np.abs(actual - predicted))


def qlike(actual: np.ndarray, predicted: np.ndarray) -> float:
    """
    Calculate QLIKE (Quasi-Likelihood) loss.

    This is a robust loss function for variance forecasting that is
    particularly appropriate when the true variance is unknown and
    a proxy (like squared returns) is used.

    QLIKE = E[log(h) + RV/h]

    where h is the forecast and RV is the realized variance proxy.

    Parameters
    ----------
    actual : np.ndarray
        Actual (realized) variance
    predicted : np.ndarray
        Predicted variance

    Returns
    -------
    float
        QLIKE loss value
    """
    # Ensure positive values to avoid log issues
    predicted = np.maximum(predicted, 1e-10)
    actual = np.maximum(actual, 1e-10)

    return np.mean(np.log(predicted) + actual / predicted)


def mape(actual: np.ndarray, predicted: np.ndarray) -> float:
    """
    Calculate Mean Absolute Percentage Error.

    Parameters
    ----------
    actual : np.ndarray
        Actual variance
    predicted : np.ndarray
        Predicted variance

    Returns
    -------
    float
        MAPE value (as a fraction, not percentage)
    """
    actual = np.maximum(actual, 1e-10)  # Avoid division by zero
    return np.mean(np.abs((actual - predicted) / actual))


def r_squared(actual: np.ndarray, predicted: np.ndarray) -> float:
    """
    Calculate R-squared (coefficient of determination).

    Parameters
    ----------
    actual : np.ndarray
        Actual variance
    predicted : np.ndarray
        Predicted variance

    Returns
    -------
    float
        R-squared value
    """
    ss_res = np.sum((actual - predicted) ** 2)
    ss_tot = np.sum((actual - np.mean(actual)) ** 2)

    if ss_tot == 0:
        return 0.0

    return 1 - (ss_res / ss_tot)


def mincer_zarnowitz_regression(
    actual: np.ndarray,
    predicted: np.ndarray
) -> Dict[str, float]:
    """
    Perform Mincer-Zarnowitz regression for forecast evaluation.

    The regression is: RV_t = α + β * h_t + ε_t

    Under the null hypothesis of unbiased and efficient forecasts:
    - α = 0 (no bias)
    - β = 1 (efficient use of information)

    Parameters
    ----------
    actual : np.ndarray
        Actual (realized) variance
    predicted : np.ndarray
        Predicted variance

    Returns
    -------
    Dict[str, float]
        Dictionary containing:
        - alpha: intercept
        - beta: slope
        - alpha_pvalue: p-value for α = 0
        - beta_pvalue: p-value for β = 1
        - r_squared: R-squared of the regression
        - joint_test_pvalue: p-value for joint test α=0, β=1
    """
    # Add constant for intercept
    X = np.column_stack([np.ones(len(predicted)), predicted])
    y = actual

    # OLS estimation
    beta_hat = np.linalg.lstsq(X, y, rcond=None)[0]
    alpha, beta = beta_hat

    # Residuals and variance
    residuals = y - X @ beta_hat
    n = len(y)
    k = 2  # number of parameters

    sigma_sq = np.sum(residuals ** 2) / (n - k)

    # Standard errors
    var_beta = sigma_sq * np.linalg.inv(X.T @ X)
    se_alpha = np.sqrt(var_beta[0, 0])
    se_beta = np.sqrt(var_beta[1, 1])

    # t-statistics
    t_alpha = alpha / se_alpha  # Test α = 0
    t_beta = (beta - 1) / se_beta  # Test β = 1

    # p-values (two-tailed)
    alpha_pvalue = 2 * (1 - stats.t.cdf(np.abs(t_alpha), df=n-k))
    beta_pvalue = 2 * (1 - stats.t.cdf(np.abs(t_beta), df=n-k))

    # R-squared
    ss_res = np.sum(residuals ** 2)
    ss_tot = np.sum((y - np.mean(y)) ** 2)
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0

    # Joint F-test for α=0, β=1
    R = np.array([[1, 0], [0, 1]])
    r = np.array([0, 1])
    diff = R @ beta_hat - r

    # F-statistic
    try:
        F_stat = (diff.T @ np.linalg.inv(R @ var_beta @ R.T) @ diff) / 2
        joint_pvalue = 1 - stats.f.cdf(F_stat, 2, n - k)
    except np.linalg.LinAlgError:
        joint_pvalue = np.nan

    return {
        "alpha": alpha,
        "beta": beta,
        "alpha_se": se_alpha,
        "beta_se": se_beta,
        "alpha_pvalue": alpha_pvalue,
        "beta_pvalue": beta_pvalue,
        "r_squared": r2,
        "joint_test_pvalue": joint_pvalue
    }


def diebold_mariano_test(
    actual: np.ndarray,
    pred1: np.ndarray,
    pred2: np.ndarray,
    loss_func: str = "mse"
) -> Tuple[float, float]:
    """
    Perform Diebold-Mariano test for comparing forecast accuracy.

    Tests the null hypothesis that two forecasts have equal predictive accuracy.

    Parameters
    ----------
    actual : np.ndarray
        Actual values
    pred1 : np.ndarray
        Predictions from model 1
    pred2 : np.ndarray
        Predictions from model 2
    loss_func : str
        Loss function to use ('mse', 'mae', 'qlike')

    Returns
    -------
    Tuple[float, float]
        DM statistic and p-value
    """
    # Calculate loss differentials
    if loss_func == "mse":
        d = (actual - pred1) ** 2 - (actual - pred2) ** 2
    elif loss_func == "mae":
        d = np.abs(actual - pred1) - np.abs(actual - pred2)
    elif loss_func == "qlike":
        pred1 = np.maximum(pred1, 1e-10)
        pred2 = np.maximum(pred2, 1e-10)
        actual_safe = np.maximum(actual, 1e-10)
        d = (np.log(pred1) + actual_safe / pred1) - (np.log(pred2) + actual_safe / pred2)
    else:
        raise ValueError(f"Unknown loss function: {loss_func}")

    # DM statistic
    d_bar = np.mean(d)
    n = len(d)

    # Newey-West standard error (with automatic lag selection)
    max_lag = int(np.floor(4 * (n / 100) ** (2/9)))
    gamma_0 = np.var(d, ddof=1)

    gamma_sum = 0
    for k in range(1, max_lag + 1):
        gamma_k = np.sum((d[k:] - d_bar) * (d[:-k] - d_bar)) / (n - 1)
        weight = 1 - k / (max_lag + 1)  # Bartlett kernel
        gamma_sum += 2 * weight * gamma_k

    var_d = (gamma_0 + gamma_sum) / n
    se_d = np.sqrt(max(var_d, 1e-10))

    dm_stat = d_bar / se_d
    p_value = 2 * (1 - stats.norm.cdf(np.abs(dm_stat)))

    return dm_stat, p_value


def calculate_all_metrics(
    actual: np.ndarray,
    predicted: np.ndarray
) -> Dict[str, float]:
    """
    Calculate all evaluation metrics.

    Parameters
    ----------
    actual : np.ndarray
        Actual variance
    predicted : np.ndarray
        Predicted variance

    Returns
    -------
    Dict[str, float]
        Dictionary of all metrics
    """
    mz = mincer_zarnowitz_regression(actual, predicted)

    return {
        "mse": mse(actual, predicted),
        "rmse": rmse(actual, predicted),
        "mae": mae(actual, predicted),
        "qlike": qlike(actual, predicted),
        "mape": mape(actual, predicted),
        "r_squared": r_squared(actual, predicted),
        "mz_alpha": mz["alpha"],
        "mz_beta": mz["beta"],
        "mz_r_squared": mz["r_squared"],
        "mz_joint_pvalue": mz["joint_test_pvalue"]
    }
