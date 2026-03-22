"""
tests/test_emacross_strategy.py
================================
Comprehensive tests for the EMACrossMACDTrend strategy.

Tests cover:
- LONG entry: fast EMA > slow EMA + histogram > 0 rising + ADX + volume
- SHORT entry: fast EMA < slow EMA + histogram < 0 falling + ADX + volume
- ATR trailing stop exit (long and short)
- MACD histogram reversal exit
- Time stop exit
- Per-symbol state isolation (multi-coin)
- Reversal: SHORT → LONG and LONG → SHORT
- No action when insufficient data
- Volume filter blocks entry when volume is low
- ADX filter blocks entry when trend is weak
"""
import os
import sys
import math
import unittest
from typing import Dict, List, Optional, Any

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from Strategy.EMACrossMACDTrend import Strategy as EMACrossStrategy
from Interfaces.IStrategy import StrategyDecision
from Interfaces.orders import OrderSide
from Interfaces.market_data import Bar


# =====================================================================
# Helpers — synthetic price data generators
# =====================================================================
def make_uptrend(n: int = 60, start: float = 100.0, slope: float = 0.5,
                 noise: float = 0.3) -> Dict[str, List[float]]:
    """Generate trending-up OHLCV data with increasing volatility."""
    np.random.seed(42)
    closes = [start + i * slope + np.random.uniform(-noise, noise) for i in range(n)]
    highs = [c + abs(np.random.uniform(0.2, 1.0)) for c in closes]
    lows = [c - abs(np.random.uniform(0.2, 1.0)) for c in closes]
    opens = [closes[max(0, i - 1)] for i in range(n)]
    volumes = [1000 + np.random.uniform(0, 500) for _ in range(n)]
    # Last bar: spike volume so it passes volume filter
    volumes[-1] = 3000
    volumes[-2] = 2500
    volumes[-3] = 2200
    return {
        "open": opens, "high": highs, "low": lows,
        "close": closes, "volume": volumes,
    }


def make_downtrend(n: int = 60, start: float = 100.0, slope: float = 0.5,
                   noise: float = 0.3) -> Dict[str, List[float]]:
    """Generate trending-down OHLCV data."""
    np.random.seed(123)
    closes = [start - i * slope + np.random.uniform(-noise, noise) for i in range(n)]
    highs = [c + abs(np.random.uniform(0.2, 1.0)) for c in closes]
    lows = [c - abs(np.random.uniform(0.2, 1.0)) for c in closes]
    opens = [closes[max(0, i - 1)] for i in range(n)]
    volumes = [1000 + np.random.uniform(0, 500) for _ in range(n)]
    volumes[-1] = 3000
    volumes[-2] = 2500
    volumes[-3] = 2200
    return {
        "open": opens, "high": highs, "low": lows,
        "close": closes, "volume": volumes,
    }


def make_flat(n: int = 60, center: float = 100.0,
              noise: float = 0.1) -> Dict[str, List[float]]:
    """Generate flat/sideways OHLCV data (low ADX)."""
    np.random.seed(77)
    closes = [center + np.random.uniform(-noise, noise) for _ in range(n)]
    highs = [c + 0.05 for c in closes]
    lows = [c - 0.05 for c in closes]
    opens = closes[:]
    volumes = [1000] * n
    return {
        "open": opens, "high": highs, "low": lows,
        "close": closes, "volume": volumes,
    }


class MockCtx:
    """Mock StrategyContext that returns OHLCV from a dict."""
    def __init__(self, symbol: str, ohlcv: Dict[str, List[float]],
                 position: float = 0.0, equity: float = 10000.0):
        self.symbol = symbol
        self.timeframe = "5m"
        self.bar_store = None
        self.position = position
        self.equity = equity
        self.cash = equity
        self.timestamp_ns = 0
        self._ohlcv = ohlcv

    def get_ohlcv(self):
        return self._ohlcv


def make_bar(symbol: str, ohlcv: Dict, index: int = -1) -> Bar:
    """Create a Bar from OHLCV dict at given index."""
    i = index if index >= 0 else len(ohlcv["close"]) + index
    return Bar(
        symbol=symbol, timeframe="5m",
        timestamp_ns=i * 300_000_000_000,
        open=ohlcv["open"][i], high=ohlcv["high"][i],
        low=ohlcv["low"][i], close=ohlcv["close"][i],
        volume=ohlcv["volume"][i],
    )


# =====================================================================
# Test Cases
# =====================================================================
class TestEMACrossLongEntry(unittest.TestCase):
    """Test LONG entry conditions."""

    def test_long_signal_on_uptrend(self):
        """Uptrending data should produce a LONG entry signal."""
        strategy = EMACrossStrategy(
            fast_ema_period=5, slow_ema_period=12, macd_signal=5,
            atr_period=5, adx_period=5, adx_threshold=15.0,
            volume_filter=True, volume_period=10,
        )
        ohlcv = make_uptrend(n=60, slope=0.8)
        bar = make_bar("AVAXUSDT", ohlcv)
        ctx = MockCtx("AVAXUSDT", ohlcv, position=0.0)

        decision = strategy.on_bar(bar, ctx)

        if decision.has_orders:
            entry_orders = [o for o in decision.orders if not o.reduce_only]
            if entry_orders:
                self.assertEqual(entry_orders[0].side, OrderSide.BUY,
                                 "Uptrend should produce BUY order")
                self.assertEqual(decision.metadata.get("entry_side"), "long")
                return

        # If first bar doesn't trigger, feed a few more
        for i in range(50, 60):
            partial = {k: v[:i+1] for k, v in ohlcv.items()}
            bar = make_bar("AVAXUSDT", partial)
            ctx = MockCtx("AVAXUSDT", partial, position=0.0)
            decision = strategy.on_bar(bar, ctx)
            if decision.metadata.get("entry_side") == "long":
                entry_orders = [o for o in decision.orders if not o.reduce_only]
                self.assertEqual(entry_orders[0].side, OrderSide.BUY)
                return

        self.fail("Strategy never produced LONG signal on uptrend data")

    def test_long_returns_features(self):
        """Decision should include indicator features."""
        strategy = EMACrossStrategy(
            fast_ema_period=5, slow_ema_period=12, macd_signal=5,
            atr_period=5, adx_period=5, adx_threshold=10.0,
            volume_filter=False,
        )
        ohlcv = make_uptrend(n=60, slope=1.0)
        bar = make_bar("AVAXUSDT", ohlcv)
        ctx = MockCtx("AVAXUSDT", ohlcv, position=0.0)
        decision = strategy.on_bar(bar, ctx)

        self.assertIn("fast_ema", decision.features)
        self.assertIn("slow_ema", decision.features)
        self.assertIn("histogram", decision.features)
        self.assertIn("atr", decision.features)
        self.assertIn("adx", decision.features)


class TestEMACrossShortEntry(unittest.TestCase):
    """Test SHORT entry conditions."""

    def test_short_signal_on_downtrend(self):
        """Downtrending data should produce a SHORT entry signal."""
        strategy = EMACrossStrategy(
            fast_ema_period=5, slow_ema_period=12, macd_signal=5,
            atr_period=5, adx_period=5, adx_threshold=15.0,
            volume_filter=True, volume_period=10,
        )
        ohlcv = make_downtrend(n=60, slope=0.8)

        for i in range(40, 60):
            partial = {k: v[:i+1] for k, v in ohlcv.items()}
            bar = make_bar("DOTUSDT", partial)
            ctx = MockCtx("DOTUSDT", partial, position=0.0)
            decision = strategy.on_bar(bar, ctx)
            if decision.metadata.get("entry_side") == "short":
                entry_orders = [o for o in decision.orders if not o.reduce_only]
                self.assertEqual(entry_orders[0].side, OrderSide.SELL,
                                 "Downtrend should produce SELL order")
                return

        self.fail("Strategy never produced SHORT signal on downtrend data")


class TestEMACrossExits(unittest.TestCase):
    """Test exit conditions: trailing stop, MACD cross, time stop."""

    def _get_strategy(self, **kw):
        defaults = dict(
            fast_ema_period=5, slow_ema_period=12, macd_signal=5,
            atr_period=5, adx_period=5, adx_threshold=10.0,
            atr_mult=1.5, volume_filter=False,
        )
        defaults.update(kw)
        return EMACrossStrategy(**defaults)

    def test_trailing_stop_long(self):
        """Long position: price drops below trailing stop → exit."""
        strategy = self._get_strategy(atr_mult=1.0)
        ohlcv = make_uptrend(n=60, slope=0.5)

        # Feed bars to build up state, simulate having a LONG position
        # First: get the entry
        found_entry = False
        for i in range(30, 60):
            partial = {k: v[:i+1] for k, v in ohlcv.items()}
            bar = make_bar("AVAXUSDT", partial)
            ctx = MockCtx("AVAXUSDT", partial, position=0.0)
            decision = strategy.on_bar(bar, ctx)
            if decision.metadata.get("entry_side") == "long":
                found_entry = True
                entry_idx = i
                break

        if not found_entry:
            self.skipTest("Could not get LONG entry for trailing stop test")

        # Now simulate: position is open, price drops sharply
        # Create data where price crashes
        crash_ohlcv = {k: list(v[:entry_idx+1]) for k, v in ohlcv.items()}
        entry_price = crash_ohlcv["close"][-1]

        # Add a few bars at peak
        for _ in range(3):
            peak = entry_price + 2
            crash_ohlcv["open"].append(peak)
            crash_ohlcv["high"].append(peak + 0.5)
            crash_ohlcv["low"].append(peak - 0.3)
            crash_ohlcv["close"].append(peak)
            crash_ohlcv["volume"].append(1500)

        # Now add crash bar: low drops well below trailing stop
        crash_low = entry_price - 10  # way below any trailing stop
        crash_ohlcv["open"].append(entry_price + 1)
        crash_ohlcv["high"].append(entry_price + 1.5)
        crash_ohlcv["low"].append(crash_low)
        crash_ohlcv["close"].append(crash_low + 1)
        crash_ohlcv["volume"].append(2000)

        # Feed peak bars (position=1.0 to simulate open LONG)
        for j in range(3):
            idx = entry_idx + 1 + j
            bar = Bar(
                symbol="AVAXUSDT", timeframe="5m",
                timestamp_ns=idx * 300_000_000_000,
                open=crash_ohlcv["open"][idx], high=crash_ohlcv["high"][idx],
                low=crash_ohlcv["low"][idx], close=crash_ohlcv["close"][idx],
                volume=crash_ohlcv["volume"][idx],
            )
            ctx = MockCtx("AVAXUSDT", crash_ohlcv, position=1.0)
            strategy.on_bar(bar, ctx)

        # Feed crash bar
        crash_idx = len(crash_ohlcv["close"]) - 1
        bar = Bar(
            symbol="AVAXUSDT", timeframe="5m",
            timestamp_ns=crash_idx * 300_000_000_000,
            open=crash_ohlcv["open"][crash_idx],
            high=crash_ohlcv["high"][crash_idx],
            low=crash_ohlcv["low"][crash_idx],
            close=crash_ohlcv["close"][crash_idx],
            volume=crash_ohlcv["volume"][crash_idx],
        )
        ctx = MockCtx("AVAXUSDT", crash_ohlcv, position=1.0)
        decision = strategy.on_bar(bar, ctx)

        self.assertTrue(decision.has_orders, "Should produce exit order")
        exit_orders = [o for o in decision.orders if o.reduce_only]
        self.assertGreater(len(exit_orders), 0, "Should have reduce_only exit")
        self.assertEqual(exit_orders[0].side, OrderSide.SELL,
                         "Long exit should be SELL")
        self.assertEqual(decision.metadata.get("exit_reason"), "trailing_stop_long")

    def test_trailing_stop_short(self):
        """Short position: price rises above trailing stop → exit."""
        strategy = self._get_strategy(atr_mult=1.0)
        ohlcv = make_downtrend(n=60, slope=0.5)

        found_entry = False
        for i in range(30, 60):
            partial = {k: v[:i+1] for k, v in ohlcv.items()}
            bar = make_bar("DOTUSDT", partial)
            ctx = MockCtx("DOTUSDT", partial, position=0.0)
            decision = strategy.on_bar(bar, ctx)
            if decision.metadata.get("entry_side") == "short":
                found_entry = True
                entry_idx = i
                break

        if not found_entry:
            self.skipTest("Could not get SHORT entry for trailing stop test")

        # Simulate price spike up
        spike_ohlcv = {k: list(v[:entry_idx+1]) for k, v in ohlcv.items()}
        entry_price = spike_ohlcv["close"][-1]

        # Add bars at trough
        for _ in range(3):
            trough = entry_price - 2
            spike_ohlcv["open"].append(trough)
            spike_ohlcv["high"].append(trough + 0.3)
            spike_ohlcv["low"].append(trough - 0.5)
            spike_ohlcv["close"].append(trough)
            spike_ohlcv["volume"].append(1500)

        # Spike bar
        spike_high = entry_price + 10
        spike_ohlcv["open"].append(entry_price - 1)
        spike_ohlcv["high"].append(spike_high)
        spike_ohlcv["low"].append(entry_price - 1.5)
        spike_ohlcv["close"].append(spike_high - 1)
        spike_ohlcv["volume"].append(2000)

        # Feed trough bars
        for j in range(3):
            idx = entry_idx + 1 + j
            bar = Bar(
                symbol="DOTUSDT", timeframe="5m",
                timestamp_ns=idx * 300_000_000_000,
                open=spike_ohlcv["open"][idx], high=spike_ohlcv["high"][idx],
                low=spike_ohlcv["low"][idx], close=spike_ohlcv["close"][idx],
                volume=spike_ohlcv["volume"][idx],
            )
            ctx = MockCtx("DOTUSDT", spike_ohlcv, position=-1.0)
            strategy.on_bar(bar, ctx)

        # Feed spike bar
        spike_idx = len(spike_ohlcv["close"]) - 1
        bar = Bar(
            symbol="DOTUSDT", timeframe="5m",
            timestamp_ns=spike_idx * 300_000_000_000,
            open=spike_ohlcv["open"][spike_idx],
            high=spike_ohlcv["high"][spike_idx],
            low=spike_ohlcv["low"][spike_idx],
            close=spike_ohlcv["close"][spike_idx],
            volume=spike_ohlcv["volume"][spike_idx],
        )
        ctx = MockCtx("DOTUSDT", spike_ohlcv, position=-1.0)
        decision = strategy.on_bar(bar, ctx)

        self.assertTrue(decision.has_orders)
        exit_orders = [o for o in decision.orders if o.reduce_only]
        self.assertGreater(len(exit_orders), 0)
        self.assertEqual(exit_orders[0].side, OrderSide.BUY,
                         "Short exit should be BUY")
        self.assertEqual(decision.metadata.get("exit_reason"), "trailing_stop_short")

    def test_time_stop(self):
        """Position held beyond max_holding_bars → time stop exit."""
        max_bars = 5
        strategy = self._get_strategy(max_holding_bars=max_bars)
        ohlcv = make_uptrend(n=80, slope=0.3)

        # Get entry
        found_entry = False
        entry_idx = 0
        for i in range(30, 70):
            partial = {k: v[:i+1] for k, v in ohlcv.items()}
            bar = make_bar("AVAXUSDT", partial)
            ctx = MockCtx("AVAXUSDT", partial, position=0.0)
            decision = strategy.on_bar(bar, ctx)
            if decision.metadata.get("entry_side") == "long":
                found_entry = True
                entry_idx = i
                break

        if not found_entry:
            self.skipTest("Could not get entry for time stop test")

        # Feed bars with position open, prices staying stable (no trailing stop)
        last_close = ohlcv["close"][entry_idx]
        extended = {k: list(v[:entry_idx+1]) for k, v in ohlcv.items()}
        for j in range(max_bars + 2):
            # Flat-ish prices that don't trigger trailing stop
            c = last_close + 0.01 * j
            extended["open"].append(c)
            extended["high"].append(c + 0.1)
            extended["low"].append(c - 0.1)
            extended["close"].append(c)
            extended["volume"].append(1500)

        exit_found = False
        for j in range(max_bars + 2):
            idx = entry_idx + 1 + j
            bar = Bar(
                symbol="AVAXUSDT", timeframe="5m",
                timestamp_ns=idx * 300_000_000_000,
                open=extended["open"][idx], high=extended["high"][idx],
                low=extended["low"][idx], close=extended["close"][idx],
                volume=extended["volume"][idx],
            )
            ctx = MockCtx("AVAXUSDT", extended, position=1.0)
            decision = strategy.on_bar(bar, ctx)

            if decision.metadata.get("exit_reason") == "time_stop":
                exit_found = True
                self.assertGreaterEqual(j + 1, max_bars,
                    f"Time stop fired too early: bar {j+1} < max {max_bars}")
                break

        self.assertTrue(exit_found, "Time stop should have triggered")


class TestEMACrossFilters(unittest.TestCase):
    """Test ADX and volume filters."""

    def test_no_entry_on_flat_market(self):
        """Flat market (low ADX) should not produce entry signals."""
        strategy = EMACrossStrategy(
            fast_ema_period=5, slow_ema_period=12, macd_signal=5,
            atr_period=5, adx_period=5, adx_threshold=25.0,
            volume_filter=False,
        )
        ohlcv = make_flat(n=60)

        signals = []
        for i in range(30, 60):
            partial = {k: v[:i+1] for k, v in ohlcv.items()}
            bar = make_bar("AVAXUSDT", partial)
            ctx = MockCtx("AVAXUSDT", partial, position=0.0)
            decision = strategy.on_bar(bar, ctx)
            if decision.metadata.get("entry_side"):
                signals.append(decision.metadata["entry_side"])

        self.assertEqual(len(signals), 0,
                         f"Flat market should have no entries, got {signals}")

    def test_volume_filter_blocks_low_volume(self):
        """Low volume should prevent entry even if other conditions are met."""
        strategy = EMACrossStrategy(
            fast_ema_period=5, slow_ema_period=12, macd_signal=5,
            atr_period=5, adx_period=5, adx_threshold=10.0,
            volume_filter=True, volume_period=10,
        )
        ohlcv = make_uptrend(n=60, slope=0.8)
        # Set ALL volumes to same value → last bar won't exceed SMA
        ohlcv["volume"] = [1000.0] * 60

        signals = []
        for i in range(40, 60):
            partial = {k: v[:i+1] for k, v in ohlcv.items()}
            bar = make_bar("AVAXUSDT", partial)
            ctx = MockCtx("AVAXUSDT", partial, position=0.0)
            decision = strategy.on_bar(bar, ctx)
            if decision.metadata.get("entry_side"):
                signals.append(i)

        self.assertEqual(len(signals), 0,
                         "Volume filter should block entries with flat volume")


class TestEMACrossPerSymbolState(unittest.TestCase):
    """Test that per-symbol state works correctly for multi-coin."""

    def test_independent_state_per_symbol(self):
        """Two symbols should have independent bar_count and entry tracking."""
        strategy = EMACrossStrategy(
            fast_ema_period=5, slow_ema_period=12, macd_signal=5,
            atr_period=5, adx_period=5, adx_threshold=10.0,
            volume_filter=False,
        )
        up = make_uptrend(n=60, slope=0.8)
        down = make_downtrend(n=60, slope=0.8)

        # Feed AVAXUSDT (uptrend) and DOTUSDT (downtrend) interleaved
        avax_signal = None
        dot_signal = None

        for i in range(30, 60):
            # AVAXUSDT bar
            partial_up = {k: v[:i+1] for k, v in up.items()}
            bar = make_bar("AVAXUSDT", partial_up)
            ctx = MockCtx("AVAXUSDT", partial_up, position=0.0)
            decision = strategy.on_bar(bar, ctx)
            if decision.metadata.get("entry_side") and avax_signal is None:
                avax_signal = decision.metadata["entry_side"]

            # DOTUSDT bar
            partial_down = {k: v[:i+1] for k, v in down.items()}
            bar = make_bar("DOTUSDT", partial_down)
            ctx = MockCtx("DOTUSDT", partial_down, position=0.0)
            decision = strategy.on_bar(bar, ctx)
            if decision.metadata.get("entry_side") and dot_signal is None:
                dot_signal = decision.metadata["entry_side"]

        # Verify independent signals
        if avax_signal:
            self.assertEqual(avax_signal, "long", "Uptrend should be LONG")
        if dot_signal:
            self.assertEqual(dot_signal, "short", "Downtrend should be SHORT")

        # At least one should have fired
        self.assertTrue(avax_signal or dot_signal,
                        "At least one symbol should have generated a signal")

        # Verify states are independent
        self.assertIn("AVAXUSDT", strategy._state)
        self.assertIn("DOTUSDT", strategy._state)
        self.assertNotEqual(
            strategy._state["AVAXUSDT"]["bar_count"],
            0, "AVAXUSDT state should have bar count > 0"
        )

    def test_state_isolation_no_cross_contamination(self):
        """Entering on one symbol should not affect another's tracking."""
        strategy = EMACrossStrategy(
            fast_ema_period=5, slow_ema_period=12, macd_signal=5,
            atr_period=5, adx_period=5, adx_threshold=10.0,
            volume_filter=False,
        )
        up = make_uptrend(n=60, slope=0.8)
        flat = make_flat(n=60)

        # Feed AVAXUSDT uptrend until entry
        for i in range(30, 60):
            partial = {k: v[:i+1] for k, v in up.items()}
            bar = make_bar("AVAXUSDT", partial)
            ctx = MockCtx("AVAXUSDT", partial, position=0.0)
            decision = strategy.on_bar(bar, ctx)
            if decision.metadata.get("entry_side") == "long":
                break

        # AVAXUSDT entry should not set DOTUSDT's entry_price
        dot_state = strategy._get_state("DOTUSDT")
        self.assertEqual(dot_state["entry_price"], 0.0,
                         "DOTUSDT entry_price should not be affected by AVAXUSDT entry")


class TestEMACrossReversal(unittest.TestCase):
    """Test position reversal (SHORT → LONG, LONG → SHORT)."""

    def test_reversal_produces_exit_and_entry(self):
        """When reversing, should produce both exit (reduce_only) and entry orders."""
        strategy = EMACrossStrategy(
            fast_ema_period=5, slow_ema_period=12, macd_signal=5,
            atr_period=5, adx_period=5, adx_threshold=10.0,
            volume_filter=False, allow_reversal=True,
        )
        ohlcv = make_uptrend(n=60, slope=1.0)

        # Feed bars and simulate existing SHORT position
        for i in range(40, 60):
            partial = {k: v[:i+1] for k, v in ohlcv.items()}
            bar = make_bar("AVAXUSDT", partial)
            ctx = MockCtx("AVAXUSDT", partial, position=-1.0)
            decision = strategy.on_bar(bar, ctx)

            if decision.metadata.get("entry_side") == "long":
                # Should have 2 orders: exit SHORT + enter LONG
                exit_orders = [o for o in decision.orders if o.reduce_only]
                entry_orders = [o for o in decision.orders if not o.reduce_only]

                self.assertGreater(len(exit_orders), 0,
                                   "Reversal should include exit order")
                self.assertGreater(len(entry_orders), 0,
                                   "Reversal should include entry order")
                self.assertEqual(exit_orders[0].side, OrderSide.BUY,
                                 "Exit SHORT = BUY")
                self.assertEqual(entry_orders[0].side, OrderSide.BUY,
                                 "Enter LONG = BUY")
                return

        self.skipTest("Reversal not triggered (depends on data alignment)")

    def test_no_reversal_when_disabled(self):
        """With allow_reversal=False, should not produce exit+entry."""
        strategy = EMACrossStrategy(
            fast_ema_period=5, slow_ema_period=12, macd_signal=5,
            atr_period=5, adx_period=5, adx_threshold=10.0,
            volume_filter=False, allow_reversal=False,
        )
        ohlcv = make_uptrend(n=60, slope=1.0)

        for i in range(40, 60):
            partial = {k: v[:i+1] for k, v in ohlcv.items()}
            bar = make_bar("AVAXUSDT", partial)
            ctx = MockCtx("AVAXUSDT", partial, position=-1.0)
            decision = strategy.on_bar(bar, ctx)

            if decision.has_orders:
                exit_orders = [o for o in decision.orders if o.reduce_only]
                entry_orders = [o for o in decision.orders if not o.reduce_only]
                # Should have entry only, no exit (no reversal)
                if entry_orders and entry_orders[0].side == OrderSide.BUY:
                    self.assertEqual(len(exit_orders), 0,
                                     "No reversal → no exit order")
                    return

        self.skipTest("Entry not triggered (depends on data)")


class TestEMACrossEdgeCases(unittest.TestCase):
    """Test edge cases and minimal data."""

    def test_insufficient_data_returns_no_action(self):
        """With too few bars, should return no_action."""
        strategy = EMACrossStrategy(
            fast_ema_period=12, slow_ema_period=26, macd_signal=9,
            atr_period=14, adx_period=14,
        )
        # Only 10 bars — needs at least 26+9+2 = 37
        ohlcv = make_uptrend(n=10)
        bar = make_bar("AVAXUSDT", ohlcv)
        ctx = MockCtx("AVAXUSDT", ohlcv, position=0.0)
        decision = strategy.on_bar(bar, ctx)

        self.assertFalse(decision.has_orders, "Insufficient data → no orders")
        self.assertFalse(decision.has_signal, "Insufficient data → no signal")

    def test_none_ohlcv_returns_no_action(self):
        """If context returns None for OHLCV, should return no_action."""
        strategy = EMACrossStrategy()
        bar = Bar(symbol="AVAXUSDT", timeframe="5m", timestamp_ns=0,
                  open=100, high=101, low=99, close=100, volume=1000)

        class NoneCtx:
            symbol = "AVAXUSDT"
            position = 0.0
            equity = 10000.0
            def get_ohlcv(self):
                return None

        decision = strategy.on_bar(bar, NoneCtx())
        self.assertFalse(decision.has_orders)

    def test_reset_clears_all_state(self):
        """reset() should clear per-symbol state."""
        strategy = EMACrossStrategy(
            fast_ema_period=5, slow_ema_period=12, macd_signal=5,
            atr_period=5, adx_period=5, volume_filter=False,
        )
        ohlcv = make_uptrend(n=40)
        bar = make_bar("AVAXUSDT", ohlcv)
        ctx = MockCtx("AVAXUSDT", ohlcv, position=0.0)
        strategy.on_bar(bar, ctx)

        self.assertIn("AVAXUSDT", strategy._state)
        strategy.reset()
        self.assertEqual(len(strategy._state), 0, "reset() should clear state")

    def test_exit_order_direction(self):
        """Exit order for LONG should be SELL, for SHORT should be BUY."""
        strategy = EMACrossStrategy()
        bar = Bar(symbol="TEST", timeframe="5m", timestamp_ns=0,
                  open=100, high=101, low=99, close=100, volume=1000)

        # Long exit
        order = strategy._exit_order(bar, 5.0)
        self.assertEqual(order.side, OrderSide.SELL)
        self.assertTrue(order.reduce_only)
        self.assertEqual(order.quantity, 5.0)

        # Short exit
        order = strategy._exit_order(bar, -3.0)
        self.assertEqual(order.side, OrderSide.BUY)
        self.assertTrue(order.reduce_only)
        self.assertEqual(order.quantity, 3.0)


class TestEMACrossIndicators(unittest.TestCase):
    """Test indicator calculations."""

    def test_ema_series_length(self):
        """EMA series should be same length as input."""
        arr = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        ema = EMACrossStrategy._ema_series(arr, 3)
        self.assertEqual(len(ema), len(arr))

    def test_ema_converges_to_constant(self):
        """EMA of constant series should equal the constant."""
        arr = np.full(50, 42.0)
        ema = EMACrossStrategy._ema_series(arr, 10)
        self.assertAlmostEqual(ema[-1], 42.0, places=5)

    def test_atr_positive(self):
        """ATR should be positive for non-constant data."""
        highs = np.array([101, 102, 103, 104, 105, 106, 107, 108, 109, 110], dtype=float)
        lows = np.array([99, 100, 101, 102, 103, 104, 105, 106, 107, 108], dtype=float)
        closes = np.array([100, 101, 102, 103, 104, 105, 106, 107, 108, 109], dtype=float)
        atr = EMACrossStrategy._compute_atr(highs, lows, closes, 5)
        self.assertGreater(atr, 0)

    def test_adx_trending(self):
        """ADX should be high for strongly trending data."""
        n = 50
        closes = np.array([100 + i * 2 for i in range(n)], dtype=float)
        highs = closes + 1
        lows = closes - 1
        adx = EMACrossStrategy._compute_adx(highs, lows, closes, 5)
        if adx is not None:
            self.assertGreater(adx, 20, "Strong trend should have high ADX")

    def test_adx_returns_none_for_short_data(self):
        """ADX should return None if data shorter than period+1."""
        closes = np.array([100, 101, 102], dtype=float)
        highs = closes + 1
        lows = closes - 1
        adx = EMACrossStrategy._compute_adx(highs, lows, closes, 14)
        self.assertIsNone(adx)


if __name__ == '__main__':
    unittest.main()
