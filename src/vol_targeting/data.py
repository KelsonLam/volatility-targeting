"""Price loading behind a small, vendor-agnostic interface.

The strategy only ever asks for a single series of adjusted close prices. It
never talks to a data vendor directly, so swapping yfinance for a paid feed is
a one-class change. Downloads are cached to parquet, keyed by the request, so
repeat runs are instant and work offline once the cache is warm.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import pandas as pd

DEFAULT_CACHE_DIR = Path("data/cache")


def _cache_key(ticker: str, start: str, end: str) -> str:
    raw = f"{ticker}|{start}|{end}"
    digest = hashlib.sha1(raw.encode()).hexdigest()[:12]
    return f"price_{ticker}_{digest}.parquet"


class YFinanceLoader:
    """Adjusted close prices for one ticker from Yahoo Finance.

    Pass ``use_cache=False`` to force a fresh download.
    """

    def __init__(
        self, cache_dir: Path | str = DEFAULT_CACHE_DIR, use_cache: bool = True
    ) -> None:
        self.cache_dir = Path(cache_dir)
        self.use_cache = use_cache

    def load(self, ticker: str, start: str, end: str) -> pd.Series:
        if not ticker or not str(ticker).strip():
            raise ValueError("ticker must be a non-empty symbol.")
        if pd.Timestamp(start) >= pd.Timestamp(end):
            raise ValueError(f"start ({start}) must be earlier than end ({end}).")
        cache_path = self.cache_dir / _cache_key(ticker, start, end)

        if self.use_cache and cache_path.exists():
            return pd.read_parquet(cache_path)[ticker]

        prices = self._download(ticker, start, end)

        self.cache_dir.mkdir(parents=True, exist_ok=True)
        prices.to_frame(name=ticker).to_parquet(cache_path)
        return prices

    @staticmethod
    def _download(ticker: str, start: str, end: str) -> pd.Series:
        import yfinance as yf

        raw = yf.download(
            ticker, start=start, end=end, auto_adjust=True, progress=False
        )
        if raw.empty:
            raise ValueError(
                f"No data returned for {ticker}. Check the ticker and dates."
            )

        # With auto_adjust=True the adjusted price is the "Close" column.
        close = raw["Close"]
        if isinstance(close, pd.DataFrame):
            close = close.iloc[:, 0]
        close.index = pd.to_datetime(close.index)
        close = close.sort_index().dropna()
        close.name = ticker
        return close


def daily_returns(prices: pd.Series) -> pd.Series:
    """Simple daily returns from a price series."""
    return prices.pct_change().dropna()
