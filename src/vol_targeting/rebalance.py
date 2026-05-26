"""A no-trade band to cut turnover.

Volatility targeting recomputes a target exposure every day, and chasing every
tiny change racks up trading costs for almost no risk benefit. A no-trade band
fixes that: only move the position when the new target differs from what you are
already holding by more than a threshold. It is the single most common practical
tweak to a vol-targeting rule, trading a little tracking error for a lot less
turnover.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def apply_rebalance_threshold(
    target_leverage: pd.Series, threshold: float = 0.10
) -> pd.Series:
    """Hold the current exposure until the target moves more than ``threshold``.

    ``threshold`` is a relative move (0.10 means rebalance only when the target
    is at least 10% away from the current holding). A threshold of 0 reproduces
    the original series exactly.
    """
    if threshold < 0:
        raise ValueError("threshold cannot be negative.")

    values = target_leverage.to_numpy(dtype=float)
    held = np.empty_like(values)
    current = np.nan

    for i, target in enumerate(values):
        if np.isnan(target):
            held[i] = current
            continue
        if np.isnan(current):
            current = target            # establish the first real position
        else:
            ref = abs(current) if abs(current) > 1e-9 else 1.0
            if abs(target - current) / ref > threshold:
                current = target        # the move is big enough to act on
        held[i] = current

    return pd.Series(held, index=target_leverage.index)


def turnover(leverage: pd.Series) -> float:
    """Total absolute change in exposure, a simple turnover proxy."""
    return float(leverage.diff().abs().sum())
