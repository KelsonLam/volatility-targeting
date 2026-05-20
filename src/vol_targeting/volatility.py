"""Estimating realized volatility from a daily return series.

Two estimators are offered:

    rolling   the standard deviation of the last ``window`` returns
    ewma      an exponentially weighted estimate in the spirit of RiskMetrics

Both return an annualized figure. Both are lookahead-safe: the estimate dated
on day ``t`` uses only returns up to and including day ``t``, so a position
sized from it and applied to day ``t+1`` never sees the future.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

TRADING_DAYS = 252


def rolling_volatility(returns: pd.Series, window: int = 21) -> pd.Series:
    """Annualized rolling standard deviation of returns."""
    if window < 2:
        raise ValueError("window must be at least 2.")
    daily = returns.rolling(window).std(ddof=1)
    return daily * np.sqrt(TRADING_DAYS)


def ewma_volatility(returns: pd.Series, lam: float = 0.94) -> pd.Series:
    """Annualized exponentially weighted volatility (RiskMetrics style).

    The variance recursion is::

        var[t] = lam * var[t-1] + (1 - lam) * return[t] ** 2

    A larger ``lam`` puts more weight on history and reacts more slowly.
    """
    if not 0.0 < lam < 1.0:
        raise ValueError("lam must be strictly between 0 and 1.")
    # pandas ewm with alpha = 1 - lam reproduces the RiskMetrics recursion on
    # squared returns. mean of squared returns is the variance estimate.
    var = (returns ** 2).ewm(alpha=1.0 - lam, adjust=False).mean()
    return np.sqrt(var * TRADING_DAYS)


def estimate_volatility(
    returns: pd.Series,
    method: str = "rolling",
    window: int = 21,
    lam: float = 0.94,
) -> pd.Series:
    """Dispatch to the chosen estimator by name."""
    method = method.lower()
    if method == "rolling":
        return rolling_volatility(returns, window=window)
    if method == "ewma":
        return ewma_volatility(returns, lam=lam)
    raise ValueError(f"Unknown method '{method}'. Use 'rolling' or 'ewma'.")
