"""
data_fetcher - Real historical tick data fetching for backtests.

This module provides tools to download and normalize real market data
from Binance Vision public data dumps.
"""

from .binance_vision import BinanceVisionFetcher, FetchConfig

__all__ = ["BinanceVisionFetcher", "FetchConfig"]
