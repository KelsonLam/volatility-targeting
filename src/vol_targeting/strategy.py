"""Position sizing and the backtest.

The idea in one line: when markets are calm, hold more; when they get wild,
hold less, so that the portfolio's risk stays near a constant target.

For a target annual volatility ``v`` and an estimate of the asset's current
annual volatility ``sigma``, the exposure is::

    leverage = v / sigma

capped at ``max_leverage`` so a very quiet stretch does not push the position
to absurd size. The exposure decided at the close of day t is held through day
t+1, so the position is never sized using the very return it goes on to earn.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .volatility import estimate_volatility


@dataclass
class StrategyConfig:
    target_annual_vol: float = 0.10
    method: str = "rolling"
    window: int = 21
    ewma_lambda: float = 0.94
    max_leverage: float = 3.0
    transaction_cost_bps: float = 1.0
    risk_free_rate: float = 0.0

    def __post_init__(self) -> None:
        if self.target_annual_vol <= 0:
            raise ValueError("target_annual_vol must be positive.")
        if self.method not in ("rolling", "ewma"):
            raise ValueError("method must be 'rolling' or 'ewma'.")
        if self.window < 2:
            raise ValueError("window must be at least 2.")
        if self.max_leverage <= 0:
            raise ValueError("max_leverage must be positive.")
        if not 0.0 < self.ewma_lambda < 1.0:
            raise ValueError("ewma_lambda must be strictly between 0 and 1.")
        if self.transaction_cost_bps < 0:
            raise ValueError("transaction_cost_bps cannot be negative.")


@dataclass
class StrategyResult:
    returns: pd.Series          # net daily returns of the vol-targeted strategy
    gross_returns: pd.Series    # before transaction costs
    buy_and_hold: pd.Series     # daily returns of the underlying, for comparison
    leverage: pd.Series         # exposure applied each day
    realized_vol: pd.Series     # the volatility estimate used for sizing
    equity_curve: pd.Series     # growth of one unit, net of costs
    config: StrategyConfig


def run_strategy(
    asset_returns: pd.Series, config: StrategyConfig
) -> StrategyResult:
    """Run the volatility targeting backtest on a daily return series."""
    if asset_returns.empty:
        raise ValueError("asset_returns is empty. Nothing to run.")
    if len(asset_returns) <= config.window:
        raise ValueError(
            f"Need more than {config.window} returns to warm up the "
            f"volatility estimate, but only {len(asset_returns)} were given."
        )

    sigma = estimate_volatility(
        asset_returns,
        method=config.method,
        window=config.window,
        lam=config.ewma_lambda,
    )

    # Raw exposure, then clipped to keep it sane. Where the estimate is not yet
    # available (the warm-up), there is no position.
    leverage = (config.target_annual_vol / sigma).clip(
        lower=0.0, upper=config.max_leverage
    )

    # Exposure set at the close of day t earns day t+1's return.
    held = leverage.shift(1)
    gross_returns = held * asset_returns

    # Cost is proportional to how much the exposure moved, shifted to line up
    # with the same holding period as the return above.
    turnover = leverage.diff().abs()
    cost_rate = config.transaction_cost_bps / 10_000.0
    costs = turnover.shift(1) * cost_rate

    net_returns = gross_returns - costs

    # Drop the warm-up where there was no usable estimate.
    valid = net_returns.notna() & held.notna()
    net_returns = net_returns[valid]
    gross_returns = gross_returns[valid]
    buy_and_hold = asset_returns.loc[net_returns.index]
    leverage_out = leverage.loc[net_returns.index]
    realized_vol = sigma.loc[net_returns.index]

    equity_curve = (1.0 + net_returns).cumprod()

    return StrategyResult(
        returns=net_returns,
        gross_returns=gross_returns,
        buy_and_hold=buy_and_hold,
        leverage=leverage_out,
        realized_vol=realized_vol,
        equity_curve=equity_curve,
        config=config,
    )
