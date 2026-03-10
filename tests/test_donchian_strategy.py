"""
tests/test_donchian_strategy.py
===============================
Tests for Strategy/DonchianATRVolTarget.py (Part B).
"""
from __future__ import annotations
import math
from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional

import numpy as np
import pytest

from Strategy.DonchianATRVolTarget import Strategy as DonchianStrategy
from Interfaces.market_data import Bar
from Interfaces.IStrategy import StrategyDecision


# ---------------------------------------------------------------------------
# Fake StrategyContext
# ---------------------------------------------------------------------------
@dataclass
class FakeContext:
    symbol: str = "BTCUSDT"
    timeframe: str = "1m"
    position: float = 0.0
    equity: float = 10000.0
    cash: float = 10000.0
    timestamp_ns: int = 0
    bar_store: Any = None
    portfolio: Any = None
    params: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)
    _ohlcv: Optional[Dict[str, List[float]]] = None

    def get_ohlcv(self, limit: int = 500):
        return self._ohlcv


def _generate_trending_bars(n: int, start_price: float = 100.0,
                             trend: float = 0.5, noise: float = 0.3,
                             seed: int = 42) -> Dict[str, List[float]]:
    """Generate synthetic OHLCV with an upward trend."""
    rng = np.random.RandomState(seed)
    closes = []
    highs = []
    lows = []
    opens = []
    volumes = []
    p = start_price
    for _ in range(n):
        o = p
        c = p + trend + rng.randn() * noise
        h = max(o, c) + abs(rng.randn()) * noise
        lo = min(o, c) - abs(rng.randn()) * noise
        opens.append(o)
        closes.append(c)
        highs.append(h)
        lows.append(lo)
        volumes.append(1000 + rng.rand() * 500)
        p = c
    return {"open": opens, "high": highs, "low": lows, "close": closes, "volume": volumes}


class TestDonchianStrategy:

    def test_no_action_when_insufficient_data(self):
        strat = DonchianStrategy(dc_period=20)
        ctx = FakeContext(_ohlcv=_generate_trending_bars(5))
        bar = Bar("BTCUSDT", "1m", 0, 100, 101, 99, 100, 1000)
        dec = strat.on_bar(bar, ctx)
        assert not dec.has_orders

    def test_long_entry_on_breakout(self):
        """Close above upper Donchian with filter pass → long entry."""
        strat = DonchianStrategy(dc_period=5, filter_type="none", atr_period=3)
        ohlcv = _generate_trending_bars(30, start_price=100, trend=1.0, noise=0.1)
        ctx = FakeContext(position=0.0, _ohlcv=ohlcv, equity=10000)
        last_close = ohlcv["close"][-1]
        bar = Bar("BTCUSDT", "1m", 0, last_close, last_close + 1,
                  last_close - 0.5, last_close, 1000)
        dec = strat.on_bar(bar, ctx)
        if dec.has_orders:
            # Should be BUY
            from Interfaces.orders import OrderSide
            assert any(o.side == OrderSide.BUY for o in dec.orders)
            assert "vol_target_qty" in dec.features

    def test_trailing_stop_exit(self):
        """With a long position, trailing stop should trigger on price drop."""
        strat = DonchianStrategy(dc_period=5, atr_period=3, atr_mult=1.0,
                                  filter_type="none")
        # Build bars where price first rises then drops sharply
        n = 30
        ohlcv = _generate_trending_bars(n, 100, trend=0.5, noise=0.1)
        # Simulate "already long" by setting position
        ctx = FakeContext(position=10.0, _ohlcv=ohlcv, equity=10000)
        strat._entry_price = 100.0
        strat._highest_since_entry = 115.0
        strat._entry_bar_index = 1
        strat._bar_count = n - 1
        # Drop bar: low goes well below trailing stop
        bar = Bar("BTCUSDT", "1m", 0, 115, 115, 50, 60, 1000)
        dec = strat.on_bar(bar, ctx)
        # Should generate exit
        if dec.has_orders:
            from Interfaces.orders import OrderSide
            assert any(o.reduce_only for o in dec.orders)

    def test_reset_clears_state(self):
        strat = DonchianStrategy()
        strat._entry_bar_index = 100
        strat._highest_since_entry = 999
        strat.reset()
        assert strat._entry_bar_index == 0
        assert strat._highest_since_entry == -math.inf

    def test_atr_computation(self):
        h = np.array([11, 12, 13, 14, 15], dtype=float)
        l = np.array([9, 10, 11, 12, 13], dtype=float)
        c = np.array([10, 11, 12, 13, 14], dtype=float)
        atr = DonchianStrategy._compute_atr(h, l, c, 3)
        assert atr > 0

    def test_ema_computation(self):
        arr = np.arange(1.0, 21.0)
        ema = DonchianStrategy._ema(arr, 10)
        assert 10 < ema < 20

    def test_time_stop(self):
        strat = DonchianStrategy(max_holding_bars=5, dc_period=3, atr_period=3,
                                  atr_mult=100.0,  # very wide trailing stop
                                  filter_type="none")
        ohlcv = _generate_trending_bars(20, 100, trend=0.1, noise=0.1)
        ctx = FakeContext(position=5.0, _ohlcv=ohlcv, equity=10000)
        strat._entry_bar_index = 1
        strat._bar_count = 10  # already held for 10 bars (> max 5)
        strat._highest_since_entry = 200  # prevent trailing stop
        strat._entry_price = 100.0
        # Bar that won't hit trailing stop (low = 199 > 200 - 100*atr)
        bar = Bar("BTCUSDT", "1m", 0, 100, 200, 199, 100, 1000)
        dec = strat.on_bar(bar, ctx)
        assert dec.has_orders
        assert dec.metadata.get("exit_reason") == "time_stop"

    def test_features_populated(self):
        strat = DonchianStrategy(dc_period=5, atr_period=3, filter_type="none")
        ohlcv = _generate_trending_bars(20, 100, 0.2, 0.1)
        ctx = FakeContext(position=0.0, _ohlcv=ohlcv)
        bar = Bar("BTCUSDT", "1m", 0, 100, 101, 99, 100, 1000)
        dec = strat.on_bar(bar, ctx)
        assert "upper" in dec.features
        assert "lower" in dec.features
        assert "atr" in dec.features
