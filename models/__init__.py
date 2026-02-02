"""
Models module for volatility forecasting.
"""

from .base import VolatilityModel
from .garch import GARCHModel, EGARCHModel, GJRGARCHModel
from .ml_models import RandomForestVolatility, XGBoostVolatility
from .stochastic_vol import HestonModel

# LSTM requires TensorFlow (optional)
try:
    from .ml_models import LSTMVolatility
    _LSTM_AVAILABLE = True
except ImportError:
    _LSTM_AVAILABLE = False

__all__ = [
    "VolatilityModel",
    "GARCHModel",
    "EGARCHModel",
    "GJRGARCHModel",
    "RandomForestVolatility",
    "XGBoostVolatility",
    "HestonModel",
]

if _LSTM_AVAILABLE:
    __all__.append("LSTMVolatility")
