"""Tests for the no-trade band rebalancing helper."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

SRC = Path(__file__).resolve().parents[1] / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from vol_targeting.rebalance import apply_rebalance_threshold, turnover


def _series(values):
    return pd.Series(values, index=pd.RangeIndex(len(values)))


def test_zero_threshold_is_identity():
    s = _series([1.0, 1.1, 0.9, 1.3, 0.7])
    out = apply_rebalance_threshold(s, threshold=0.0)
    pd.testing.assert_series_equal(out, s)


def test_threshold_reduces_turnover():
    rng = np.random.default_rng(0)
    s = _series(1.0 + rng.normal(0, 0.05, 500))
    lazy = apply_rebalance_threshold(s, threshold=0.10)
    assert turnover(lazy) < turnover(s)


def test_small_moves_are_ignored():
    # Each step is a 2% move; with a 10% band nothing should trade after the first.
    s = _series([1.0, 1.02, 1.04, 1.0608])
    out = apply_rebalance_threshold(s, threshold=0.10)
    assert (out == 1.0).all()


def test_big_move_triggers_rebalance():
    s = _series([1.0, 1.0, 1.5])   # a 50% jump clears any reasonable band
    out = apply_rebalance_threshold(s, threshold=0.10)
    assert out.iloc[-1] == pytest.approx(1.5)


def test_negative_threshold_raises():
    with pytest.raises(ValueError):
        apply_rebalance_threshold(_series([1.0, 1.1]), threshold=-0.1)
