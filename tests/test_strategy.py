"""Tests for the volatility estimators and the targeting strategy.

Synthetic return paths keep these fast and offline.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

SRC = Path(__file__).resolve().parents[1] / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from vol_targeting.volatility import (
    rolling_volatility,
    ewma_volatility,
    estimate_volatility,
)
from vol_targeting.strategy import StrategyConfig, run_strategy
from vol_targeting import metrics


def _days(n: int) -> pd.DatetimeIndex:
    return pd.bdate_range("2015-01-01", periods=n)


def test_rolling_vol_annualizes():
    # Constant-magnitude returns make the std easy to reason about.
    rng = np.random.default_rng(0)
    r = pd.Series(rng.normal(0, 0.01, 500), index=_days(500))
    vol = rolling_volatility(r, window=21).dropna()
    # Annualized vol of ~1% daily should be roughly 0.01 * sqrt(252) ~ 0.16.
    assert 0.10 < vol.mean() < 0.22


def test_ewma_matches_recursion():
    r = pd.Series([0.02, -0.01, 0.015, -0.02, 0.01], index=_days(5))
    lam = 0.94
    vol = ewma_volatility(r, lam=lam)
    # Reproduce the recursion by hand. To match pandas ewm(adjust=False), the
    # series is seeded with the first squared return rather than with zero.
    squared = (r ** 2).tolist()
    var = squared[0]
    for x in squared[1:]:
        var = lam * var + (1 - lam) * x
    expected = np.sqrt(var * 252)
    assert vol.iloc[-1] == pytest.approx(expected)


def test_estimate_dispatch():
    r = pd.Series(np.random.default_rng(1).normal(0, 0.01, 100), index=_days(100))
    a = estimate_volatility(r, method="rolling", window=21)
    b = rolling_volatility(r, window=21)
    pd.testing.assert_series_equal(a, b)
    with pytest.raises(ValueError):
        estimate_volatility(r, method="nope")


def test_no_lookahead_uses_lagged_leverage():
    rng = np.random.default_rng(2)
    r = pd.Series(rng.normal(0.0003, 0.012, 600), index=_days(600))
    cfg = StrategyConfig(target_annual_vol=0.10, window=21)
    res = run_strategy(r, cfg)
    # Reconstruct gross returns from lagged leverage and the raw returns.
    expected = (res.leverage.shift(1) * r).loc[res.gross_returns.index]
    # The first point relies on a leverage value from before the kept window,
    # so compare from the second observation onward.
    pd.testing.assert_series_equal(
        res.gross_returns.iloc[1:], expected.iloc[1:], check_names=False
    )


def test_realized_vol_is_near_target():
    # On a stationary series, the targeted strategy should land near target.
    rng = np.random.default_rng(3)
    r = pd.Series(rng.normal(0.0, 0.012, 2500), index=_days(2500))
    cfg = StrategyConfig(target_annual_vol=0.10, window=21, max_leverage=10.0)
    res = run_strategy(r, cfg)
    realized = metrics.annualized_volatility(res.returns)
    # Estimation noise means it will not be exact, but it should be in the ballpark.
    assert 0.07 < realized < 0.13


def test_leverage_respects_cap():
    rng = np.random.default_rng(4)
    r = pd.Series(rng.normal(0.0, 0.005, 400), index=_days(400))
    cfg = StrategyConfig(target_annual_vol=0.50, window=21, max_leverage=2.0)
    res = run_strategy(r, cfg)
    assert res.leverage.max() <= 2.0 + 1e-9


def test_costs_reduce_returns():
    rng = np.random.default_rng(5)
    r = pd.Series(rng.normal(0.0003, 0.015, 800), index=_days(800))
    free = run_strategy(r, StrategyConfig(transaction_cost_bps=0.0))
    costly = run_strategy(r, StrategyConfig(transaction_cost_bps=20.0))
    assert costly.returns.sum() < free.returns.sum()
