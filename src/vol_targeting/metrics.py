"""Performance statistics for a daily return series.

Annualization assumes 252 trading days per year. The realized-volatility stat
is the one that matters most here: a volatility targeting strategy should land
close to its target, and this lets you check that it actually did.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

TRADING_DAYS = 252


def cagr(returns: pd.Series) -> float:
    if returns.empty:
        return float("nan")
    growth = (1.0 + returns).prod()
    years = len(returns) / TRADING_DAYS
    if years <= 0 or growth <= 0:
        return float("nan")
    return growth ** (1.0 / years) - 1.0


def annualized_volatility(returns: pd.Series) -> float:
    return returns.std(ddof=1) * np.sqrt(TRADING_DAYS)


def sharpe_ratio(returns: pd.Series, risk_free_rate: float = 0.0) -> float:
    rf_daily = (1.0 + risk_free_rate) ** (1.0 / TRADING_DAYS) - 1.0
    excess = returns - rf_daily
    vol = excess.std(ddof=1)
    if vol == 0 or np.isnan(vol):
        return float("nan")
    return (excess.mean() / vol) * np.sqrt(TRADING_DAYS)


def sortino_ratio(returns: pd.Series, risk_free_rate: float = 0.0) -> float:
    rf_daily = (1.0 + risk_free_rate) ** (1.0 / TRADING_DAYS) - 1.0
    excess = returns - rf_daily
    downside = excess[excess < 0]
    downside_dev = np.sqrt((downside ** 2).mean()) if len(downside) else np.nan
    if not downside_dev or np.isnan(downside_dev):
        return float("nan")
    return (excess.mean() / downside_dev) * np.sqrt(TRADING_DAYS)


def max_drawdown(returns: pd.Series) -> float:
    equity = (1.0 + returns).cumprod()
    drawdown = equity / equity.cummax() - 1.0
    return drawdown.min()


def calmar_ratio(returns: pd.Series) -> float:
    mdd = max_drawdown(returns)
    if mdd == 0 or np.isnan(mdd):
        return float("nan")
    return cagr(returns) / abs(mdd)


def summarize(returns: pd.Series, risk_free_rate: float = 0.0) -> dict[str, float]:
    return {
        "Total return": (1.0 + returns).prod() - 1.0,
        "CAGR": cagr(returns),
        "Realized volatility": annualized_volatility(returns),
        "Sharpe ratio": sharpe_ratio(returns, risk_free_rate),
        "Sortino ratio": sortino_ratio(returns, risk_free_rate),
        "Max drawdown": max_drawdown(returns),
        "Calmar ratio": calmar_ratio(returns),
        "Days": float(len(returns)),
    }


def format_summary(stats: dict[str, float]) -> str:
    percent_keys = {
        "Total return", "CAGR", "Realized volatility", "Max drawdown",
    }
    width = max(len(k) for k in stats)
    lines = []
    for key, value in stats.items():
        if key in percent_keys:
            shown = f"{value * 100:,.2f}%"
        elif key == "Days":
            shown = f"{value:,.0f}"
        else:
            shown = f"{value:,.2f}"
        lines.append(f"{key:<{width}}  {shown}")
    return "\n".join(lines)
