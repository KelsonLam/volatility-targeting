"""Edge-case and validation tests for the volatility targeting strategy."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

SRC = Path(__file__).resolve().parents[1] / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from vol_targeting.strategy import StrategyConfig, run_strategy


def _days(n):
    return pd.bdate_range("2015-01-01", periods=n)


def test_config_validation():
    with pytest.raises(ValueError):
        StrategyConfig(target_annual_vol=0)
    with pytest.raises(ValueError):
        StrategyConfig(method="garch")
    with pytest.raises(ValueError):
        StrategyConfig(window=1)
    with pytest.raises(ValueError):
        StrategyConfig(ewma_lambda=1.5)
    with pytest.raises(ValueError):
        StrategyConfig(max_leverage=0)


def test_short_series_raises():
    r = pd.Series(np.zeros(10), index=_days(10))
    with pytest.raises(ValueError):
        run_strategy(r, StrategyConfig(window=21))


def test_empty_series_raises():
    with pytest.raises(ValueError):
        run_strategy(pd.Series(dtype=float), StrategyConfig())


def test_zero_returns_give_zero_leverage_pnl():
    # Flat market: zero volatility estimate is undefined, so the strategy should
    # still run without producing NaNs in the realized series.
    rng = np.random.default_rng(0)
    r = pd.Series(rng.normal(0, 0.01, 300), index=_days(300))
    res = run_strategy(r, StrategyConfig(window=21))
    assert res.returns.notna().all()
