"""Charts for the volatility targeting strategy.

Three views tell the story: how the equity grew versus buy and hold, how the
exposure moved around, and how the volatility estimate behaved. Matplotlib
only, and each function returns the Figure so the caller can save or tweak it.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def plot_equity_curve(
    strategy_equity: pd.Series,
    buy_and_hold_returns: pd.Series,
    title: str = "Volatility targeting vs buy and hold",
):
    """Strategy equity against a buy-and-hold of the same asset."""
    bh_equity = (1.0 + buy_and_hold_returns).cumprod()
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(strategy_equity.index, strategy_equity.values, label="Vol targeted")
    ax.plot(bh_equity.index, bh_equity.values, label="Buy and hold", alpha=0.7)
    ax.set_title(title)
    ax.set_ylabel("Growth of 1 unit")
    ax.set_xlabel("Date")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    return fig


def plot_leverage(leverage: pd.Series, title: str = "Exposure over time"):
    """The leverage actually applied each day."""
    fig, ax = plt.subplots(figsize=(10, 3.5))
    ax.plot(leverage.index, leverage.values, color="tab:purple")
    ax.axhline(1.0, color="grey", linestyle="--", alpha=0.6, label="1x")
    ax.set_title(title)
    ax.set_ylabel("Leverage")
    ax.set_xlabel("Date")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    return fig


def plot_realized_vol(
    realized_vol: pd.Series, target: float, title: str = "Estimated volatility"
):
    """The volatility estimate used for sizing, against the target."""
    fig, ax = plt.subplots(figsize=(10, 3.5))
    ax.plot(realized_vol.index, realized_vol.values, label="Estimated vol")
    ax.axhline(target, color="tab:green", linestyle="--", label="Target")
    ax.set_title(title)
    ax.set_ylabel("Annualized volatility")
    ax.set_xlabel("Date")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    return fig


def save_figure(fig, path: Path | str) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=120)
    return path
