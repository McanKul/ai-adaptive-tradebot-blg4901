"""
tests/test_ema_macd_atr_strategy.py
====================================
Unit tests for Strategy/EMA_MACD_ATR.py — EMA + MACD + ATR trend-following.
"""
from __future__ import annotations
import math
import os
import sys
from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional

# Add project root to path (same convention as other test files)
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
import pytest

from Strategy.EMA_MACD_ATR import Strategy as EMA_MACD_ATR_Strategy
from Interfaces.market_data import Bar
from Interfaces.IStrategy import StrategyDecision
from Interfaces.orders import OrderSide


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


# ---------------------------------------------------------------------------
# Data generators
# ---------------------------------------------------------------------------
def _generate_uptrend(n: int, start: float = 100.0, trend: float = 0.5,
                       noise: float = 0.3, seed: int = 42) -> Dict[str, List[float]]:
    """Generate OHLCV with a clear upward trend (EMA50 > EMA200 after warmup)."""
    rng = np.random.RandomState(seed)
    opens, highs, lows, closes, volumes = [], [], [], [], []
    p = start
    for _ in range(n):
        o = p
        c = p + trend + rng.randn() * noise
        h = max(o, c) + abs(rng.randn()) * noise
        lo = min(o, c) - abs(rng.randn()) * noise
        opens.append(o); highs.append(h); lows.append(lo); closes.append(c)
        volumes.append(1000 + rng.rand() * 500)
        p = c
    return {"open": opens, "high": highs, "low": lows, "close": closes, "volume": volumes}


def _generate_downtrend(n: int, start: float = 200.0, trend: float = -0.5,
                          noise: float = 0.3, seed: int = 42) -> Dict[str, List[float]]:
    """Generate OHLCV with a clear downward trend (EMA50 < EMA200 after warmup)."""
    return _generate_uptrend(n, start=start, trend=trend, noise=noise, seed=seed)


def _generate_flat(n: int, price: float = 100.0,
                    noise: float = 0.3, seed: int = 42) -> Dict[str, List[float]]:
    """Generate flat/sideways OHLCV."""
    return _generate_uptrend(n, start=price, trend=0.0, noise=noise, seed=seed)


def _make_bar(close: float = 100.0, symbol: str = "BTCUSDT") -> Bar:
    return Bar(symbol, "1m", 0, close, close + 1, close - 1, close, 1000)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------
class TestEMA_MACD_ATR_Strategy:

    def test_no_action_insufficient_data(self):
        """With fewer bars than EMA slow period, no orders should be generated."""
        strat = EMA_MACD_ATR_Strategy(ema_slow=200)
        ohlcv = _generate_uptrend(50)  # Only 50 bars, need 200+
        ctx = FakeContext(_ohlcv=ohlcv)
        bar = _make_bar()
        dec = strat.on_bar(bar, ctx)
        assert not dec.has_orders

    def test_long_entry_on_uptrend_macd_crossover(self):
        """Uptrend + MACD crossover → should generate a BUY order."""
        # Use small EMA periods so uptrend is established quickly
        strat = EMA_MACD_ATR_Strategy(
            ema_fast=5, ema_slow=10,
            macd_fast=3, macd_slow=6, macd_signal=3,
            atr_period=3, atr_mult=2.0,
        )
        # Generate strong uptrend with enough bars
        ohlcv = _generate_uptrend(50, start=100, trend=1.0, noise=0.1)

        # Feed bars one at a time to let MACD crossover develop
        entry_found = False
        for i in range(15, len(ohlcv["close"])):
            sliced = {k: v[:i+1] for k, v in ohlcv.items()}
            ctx = FakeContext(position=0.0, _ohlcv=sliced, equity=10000)
            bar = Bar("BTCUSDT", "1m", 0,
                      ohlcv["open"][i], ohlcv["high"][i],
                      ohlcv["low"][i], ohlcv["close"][i], 1000)
            dec = strat.on_bar(bar, ctx)
            if dec.has_orders:
                assert any(o.side == OrderSide.BUY for o in dec.orders)
                assert dec.metadata.get("entry_side") == "long"
                entry_found = True
                break

        # In a strong uptrend the strategy should eventually enter long
        assert entry_found, "Strategy should have entered long in a strong uptrend"

    def test_short_entry_on_downtrend_macd_crossover(self):
        """Downtrend + MACD crossover → should generate a SELL order."""
        strat = EMA_MACD_ATR_Strategy(
            ema_fast=5, ema_slow=10,
            macd_fast=3, macd_slow=6, macd_signal=3,
            atr_period=3, atr_mult=2.0,
        )
        ohlcv = _generate_downtrend(50, start=200, trend=-1.0, noise=0.1)

        entry_found = False
        for i in range(15, len(ohlcv["close"])):
            sliced = {k: v[:i+1] for k, v in ohlcv.items()}
            ctx = FakeContext(position=0.0, _ohlcv=sliced, equity=10000)
            bar = Bar("BTCUSDT", "1m", 0,
                      ohlcv["open"][i], ohlcv["high"][i],
                      ohlcv["low"][i], ohlcv["close"][i], 1000)
            dec = strat.on_bar(bar, ctx)
            if dec.has_orders:
                assert any(o.side == OrderSide.SELL and not o.reduce_only for o in dec.orders)
                assert dec.metadata.get("entry_side") == "short"
                entry_found = True
                break

        assert entry_found, "Strategy should have entered short in a strong downtrend"

    def test_no_entry_without_ema_confirmation(self):
        """In flat market, even if MACD crosses, no entry should happen
        because EMA fast ≈ EMA slow (no clear trend)."""
        strat = EMA_MACD_ATR_Strategy(
            ema_fast=5, ema_slow=10,
            macd_fast=3, macd_slow=6, macd_signal=3,
            atr_period=3,
        )
        # Generate flat data — EMAs should be very close
        ohlcv = _generate_flat(60, price=100.0, noise=0.01, seed=99)
        ctx = FakeContext(position=0.0, _ohlcv=ohlcv, equity=10000)
        bar = _make_bar(100.0)

        # Call on_bar multiple times; entries should be rare/none in flat market
        entry_count = 0
        for i in range(15, len(ohlcv["close"])):
            sliced = {k: v[:i+1] for k, v in ohlcv.items()}
            ctx = FakeContext(position=0.0, _ohlcv=sliced, equity=10000)
            bar = Bar("BTCUSDT", "1m", 0,
                      ohlcv["open"][i], ohlcv["high"][i],
                      ohlcv["low"][i], ohlcv["close"][i], 1000)
            dec = strat.on_bar(bar, ctx)
            if dec.has_orders:
                entry_count += 1

        # In a truly flat market, we expect very few or zero entries
        assert entry_count <= 3, f"Too many entries ({entry_count}) in a flat market"

    def test_trailing_stop_exit(self):
        """With a long position, ATR trailing stop should trigger on drop."""
        strat = EMA_MACD_ATR_Strategy(
            ema_fast=5, ema_slow=10,
            macd_fast=3, macd_slow=6, macd_signal=3,
            atr_period=3, atr_mult=1.0,
        )
        ohlcv = _generate_uptrend(30, 100, trend=0.5, noise=0.1)

        # Simulate "already long"
        ctx = FakeContext(position=10.0, _ohlcv=ohlcv, equity=10000)
        strat._entry_price = 100.0
        strat._highest_since_entry = 115.0
        strat._entry_bar_index = 1
        strat._bar_count = 29
        strat._prev_macd_hist = 0.5  # need valid MACD state

        # Big drop bar: low goes well below trailing stop
        bar = Bar("BTCUSDT", "1m", 0, 115, 115, 50, 60, 1000)
        dec = strat.on_bar(bar, ctx)
        assert dec.has_orders
        assert any(o.reduce_only for o in dec.orders)
        assert dec.metadata.get("exit_reason") == "trailing_stop_long"

    def test_trailing_stop_short_exit(self):
        """With a short position, ATR trailing stop should trigger on rise."""
        strat = EMA_MACD_ATR_Strategy(
            ema_fast=5, ema_slow=10,
            macd_fast=3, macd_slow=6, macd_signal=3,
            atr_period=3, atr_mult=1.0,
        )
        ohlcv = _generate_downtrend(30, 200, trend=-0.5, noise=0.1)

        ctx = FakeContext(position=-10.0, _ohlcv=ohlcv, equity=10000)
        strat._entry_price = 200.0
        strat._lowest_since_entry = 185.0
        strat._entry_bar_index = 1
        strat._bar_count = 29
        strat._prev_macd_hist = -0.5

        # Big rise bar: high goes well above trailing stop
        bar = Bar("BTCUSDT", "1m", 0, 185, 250, 184, 240, 1000)
        dec = strat.on_bar(bar, ctx)
        assert dec.has_orders
        assert any(o.reduce_only for o in dec.orders)
        assert dec.metadata.get("exit_reason") == "trailing_stop_short"

    def test_time_stop(self):
        """Time stop should trigger after max_holding_bars."""
        strat = EMA_MACD_ATR_Strategy(
            ema_fast=5, ema_slow=10,
            macd_fast=3, macd_slow=6, macd_signal=3,
            atr_period=3, atr_mult=100.0,  # very wide trailing stop
            max_holding_bars=5,
        )
        ohlcv = _generate_uptrend(30, 100, trend=0.1, noise=0.1)
        ctx = FakeContext(position=5.0, _ohlcv=ohlcv, equity=10000)
        strat._entry_bar_index = 1
        strat._bar_count = 10  # held for 10 bars > max 5
        strat._highest_since_entry = 200  # prevent trailing stop
        strat._entry_price = 100.0
        strat._prev_macd_hist = 0.1

        bar = Bar("BTCUSDT", "1m", 0, 100, 200, 199, 100, 1000)
        dec = strat.on_bar(bar, ctx)
        assert dec.has_orders
        assert dec.metadata.get("exit_reason") == "time_stop"

    def test_reset_clears_state(self):
        strat = EMA_MACD_ATR_Strategy()
        strat._entry_bar_index = 100
        strat._highest_since_entry = 999
        strat._prev_macd_hist = 1.5
        strat._bar_count = 50
        strat.reset()
        assert strat._entry_bar_index == 0
        assert strat._highest_since_entry == -math.inf
        assert strat._prev_macd_hist is None
        assert strat._bar_count == 0

    def test_features_populated(self):
        """Decision features should contain all indicator values."""
        strat = EMA_MACD_ATR_Strategy(
            ema_fast=5, ema_slow=10,
            macd_fast=3, macd_slow=6, macd_signal=3,
            atr_period=3,
        )
        ohlcv = _generate_uptrend(30, 100, 0.2, 0.1)
        ctx = FakeContext(position=0.0, _ohlcv=ohlcv)
        bar = _make_bar(100)
        dec = strat.on_bar(bar, ctx)

        expected_keys = [
            "ema_fast", "ema_slow", "macd_line", "signal_line",
            "macd_histogram", "atr", "uptrend", "downtrend",
            "macd_cross_up", "macd_cross_down", "stop_distance", "vol_target_qty",
        ]
        for key in expected_keys:
            assert key in dec.features, f"Missing feature: {key}"

    def test_indicator_computations(self):
        """Verify indicator helper methods produce sensible values."""
        arr = np.arange(1.0, 21.0)
        ema = EMA_MACD_ATR_Strategy._ema(arr, 10)
        assert 10 < ema < 20, f"EMA should be between 10 and 20, got {ema}"

        h = np.array([11, 12, 13, 14, 15], dtype=float)
        l = np.array([9, 10, 11, 12, 13], dtype=float)
        c = np.array([10, 11, 12, 13, 14], dtype=float)
        atr = EMA_MACD_ATR_Strategy._compute_atr(h, l, c, 3)
        assert atr > 0, f"ATR should be positive, got {atr}"

        closes = np.arange(1.0, 40.0)
        macd, signal, hist = EMA_MACD_ATR_Strategy._compute_macd(closes, 12, 26, 9)
        assert isinstance(macd, float)
        assert isinstance(signal, float)
        assert isinstance(hist, float)
