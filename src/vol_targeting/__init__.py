"""Volatility targeting: scaling exposure to steer toward a constant risk level.

Modules:

    data        load prices behind a vendor-agnostic loader, with caching
    volatility  rolling and EWMA estimates of realized volatility
    strategy    turn a vol estimate into a position size and run the backtest
    metrics     daily-frequency performance statistics
    plotting    equity curve, leverage, and rolling-volatility charts
"""

__version__ = "0.1.0"
