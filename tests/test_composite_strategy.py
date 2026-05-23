"""
tests/test_composite_strategy.py
================================
Unit tests for CompositeStrategy + StrategySlot.

Covers:
- Multi-slot signal aggregation (orders from N children combined, tagged)
- Per-slot SizingConfig override (different qty per slot)
- entry_coefficient scales qty without rewriting child
- Per-slot ExitManager isolation (slot A's TP doesn't touch slot B)
- Regime gating (regime_gate policy filters by regime tags)
- Slot position tracking (eager update from emitted orders)
"""
from __future__ import annotations
import os
import sys
import unittest
from dataclasses import dataclass

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from Backtest.exit_manager import ExitConfig
from Interfaces.IStrategy import IStrategy, StrategyDecision
from Interfaces.market_data import Bar
from Interfaces.orders import Order, OrderSide, OrderType
from Interfaces.strategy_adapter import StrategyContext, SizingConfig, SizingMode
from Interfaces.strategy_slot import StrategySlot
from Strategy.composite_strategy import CompositeStrategy


# ---------------------------------------------------------------------------
# Helpers / mocks
# ---------------------------------------------------------------------------

def make_bar(close=100.0, ts=1_000_000_000, symbol="BTCUSDT"):
    return Bar(
        symbol=symbol, timeframe="1h",
        timestamp_ns=ts,
        open=close * 0.99, high=close * 1.02, low=close * 0.98,
        close=close, volume=1000.0,
    )


def make_ctx(symbol="BTCUSDT", position=0.0, equity=10_000.0):
    return StrategyContext(
        symbol=symbol, timeframe="1h",
        position=position, equity=equity, cash=equity,
        timestamp_ns=1_000_000_000,
    )


class _FixedSignalStrategy(IStrategy):
    """Strategy that always emits the same signal — for deterministic tests."""

    def __init__(self, signal: str | None = "+1"):
        self._signal = signal

    def generate_signal(self, symbol):
        return self._signal


class _OrderListStrategy(IStrategy):
    """Strategy that returns a fixed list of orders via on_bar."""

    def __init__(self, side=OrderSide.BUY, qty=1.0):
        self._side = side
        self._qty = qty

    def on_bar(self, bar, ctx):
        return StrategyDecision(orders=[Order(
            symbol=bar.symbol, side=self._side, order_type=OrderType.MARKET,
            quantity=self._qty, timestamp_ns=bar.timestamp_ns,
        )])


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestCompositeBasics(unittest.TestCase):

    def test_requires_non_empty_slots(self):
        with self.assertRaises(ValueError):
            CompositeStrategy(slots=[])

    def test_rejects_duplicate_slot_ids(self):
        s = StrategySlot(id="dup", strategy=_FixedSignalStrategy())
        with self.assertRaises(ValueError):
            CompositeStrategy(slots=[s, StrategySlot(id="dup", strategy=_FixedSignalStrategy())])

    def test_rejects_unknown_policy(self):
        s = StrategySlot(id="a", strategy=_FixedSignalStrategy())
        with self.assertRaises(ValueError):
            CompositeStrategy(slots=[s], policy="bogus")


class TestCompositeAggregation(unittest.TestCase):
    """Two children, opposite signals — net qty is buy − sell."""

    def test_orders_from_all_slots_aggregated(self):
        sizing = SizingConfig(
            mode=SizingMode.NOTIONAL_USD, notional_usd=1000.0, leverage=1.0,
        )
        slots = [
            StrategySlot(id="bull", strategy=_FixedSignalStrategy("+1"), sizing=sizing),
            StrategySlot(id="bear", strategy=_FixedSignalStrategy("-1"), sizing=sizing),
        ]
        comp = CompositeStrategy(slots=slots)
        bar = make_bar(close=100.0)
        ctx = make_ctx()

        decision = comp.on_bar(bar, ctx)

        self.assertEqual(len(decision.orders), 2)
        ids = {o.strategy_id for o in decision.orders}
        self.assertEqual(ids, {"bull", "bear"})
        # qty = 1000 / 100 = 10 for each
        for o in decision.orders:
            self.assertAlmostEqual(o.quantity, 10.0, places=6)


class TestPerSlotSizing(unittest.TestCase):
    """Same signal, different SizingConfig → different qty."""

    def test_each_slot_uses_own_sizing(self):
        big = SizingConfig(mode=SizingMode.NOTIONAL_USD, notional_usd=2000.0)
        small = SizingConfig(mode=SizingMode.NOTIONAL_USD, notional_usd=500.0)
        slots = [
            StrategySlot(id="big", strategy=_FixedSignalStrategy("+1"), sizing=big),
            StrategySlot(id="small", strategy=_FixedSignalStrategy("+1"), sizing=small),
        ]
        comp = CompositeStrategy(slots=slots)
        decision = comp.on_bar(make_bar(close=100.0), make_ctx())

        by_id = {o.strategy_id: o for o in decision.orders}
        self.assertAlmostEqual(by_id["big"].quantity, 20.0, places=6)
        self.assertAlmostEqual(by_id["small"].quantity, 5.0, places=6)

    def test_default_sizing_falls_back_when_slot_unset(self):
        default = SizingConfig(mode=SizingMode.NOTIONAL_USD, notional_usd=300.0)
        slots = [StrategySlot(id="s1", strategy=_FixedSignalStrategy("+1"))]
        comp = CompositeStrategy(slots=slots, default_sizing=default)
        decision = comp.on_bar(make_bar(close=100.0), make_ctx())
        self.assertAlmostEqual(decision.orders[0].quantity, 3.0, places=6)


class TestEntryCoefficient(unittest.TestCase):
    """entry_coefficient scales qty without touching the child."""

    def test_half_coefficient_halves_qty(self):
        sizing = SizingConfig(mode=SizingMode.NOTIONAL_USD, notional_usd=1000.0)
        slots = [
            StrategySlot(id="full", strategy=_FixedSignalStrategy("+1"),
                         sizing=sizing, entry_coefficient=1.0),
            StrategySlot(id="half", strategy=_FixedSignalStrategy("+1"),
                         sizing=sizing, entry_coefficient=0.5),
        ]
        comp = CompositeStrategy(slots=slots)
        decision = comp.on_bar(make_bar(close=100.0), make_ctx())
        by_id = {o.strategy_id: o for o in decision.orders}
        # full = 10, half = 5
        self.assertAlmostEqual(by_id["full"].quantity, 10.0, places=6)
        self.assertAlmostEqual(by_id["half"].quantity, 5.0, places=6)


class TestRegimeGating(unittest.TestCase):
    """regime_gate policy: only slots whose regimes intersect tags fire."""

    def _detector(self, tags):
        @dataclass
        class _Det:
            _tags: list

            def detect(self, bar, ctx):
                @dataclass
                class _S:
                    tags: list
                return _S(tags=self._tags)

        return _Det(tags)

    def test_only_trend_slot_fires_when_trend_up(self):
        sizing = SizingConfig(mode=SizingMode.NOTIONAL_USD, notional_usd=1000.0)
        slots = [
            StrategySlot(id="trend", strategy=_FixedSignalStrategy("+1"),
                         sizing=sizing, regimes=["trend_up", "trend_down"]),
            StrategySlot(id="range", strategy=_FixedSignalStrategy("+1"),
                         sizing=sizing, regimes=["range"]),
        ]
        comp = CompositeStrategy(
            slots=slots, regime_detector=self._detector(["trend_up"]),
            policy="regime_gate",
        )
        decision = comp.on_bar(make_bar(close=100.0), make_ctx())
        ids = {o.strategy_id for o in decision.orders}
        self.assertIn("trend", ids)
        self.assertNotIn("range", ids)
        self.assertEqual(decision.regime_tags, ["trend_up"])


class TestSlotPositionTracking(unittest.TestCase):
    """Composite tracks per-slot intended position from emitted orders."""

    def test_position_increments_after_buy(self):
        sizing = SizingConfig(mode=SizingMode.NOTIONAL_USD, notional_usd=500.0)
        slots = [
            StrategySlot(id="x", strategy=_FixedSignalStrategy("+1"), sizing=sizing),
        ]
        comp = CompositeStrategy(slots=slots)
        comp.on_bar(make_bar(close=100.0), make_ctx())
        # 500/100 = 5 → +5
        self.assertAlmostEqual(comp.get_slot_position("x"), 5.0, places=6)
        self.assertAlmostEqual(comp.get_slot_entry_price("x"), 100.0, places=6)


class TestPerSlotExitIsolation(unittest.TestCase):
    """Slot A reaching TP must not generate exits for slot B."""

    def test_slot_a_tp_does_not_touch_slot_b(self):
        sizing = SizingConfig(mode=SizingMode.NOTIONAL_USD, notional_usd=1000.0)
        # A: 5% TP (will trigger when price moves +6%)
        # B: 50% TP (won't trigger)
        a_exit = ExitConfig(take_profit_pct=0.05)
        b_exit = ExitConfig(take_profit_pct=0.50)
        slots = [
            StrategySlot(id="a", strategy=_OrderListStrategy(qty=1.0),
                         sizing=sizing, exit=a_exit),
            StrategySlot(id="b", strategy=_OrderListStrategy(qty=1.0),
                         sizing=sizing, exit=b_exit),
        ]
        comp = CompositeStrategy(slots=slots)

        # Bar 1 — both slots open positions
        comp.on_bar(make_bar(close=100.0, ts=1_000_000_000), make_ctx())
        a_pos_after_entry = comp.get_slot_position("a")
        b_pos_after_entry = comp.get_slot_position("b")
        self.assertGreater(a_pos_after_entry, 0)
        self.assertGreater(b_pos_after_entry, 0)

        # Strategies stop emitting (replace with no-signal so we only see exits)
        for slot in slots:
            slot.strategy = _FixedSignalStrategy(None)

        # Bar 2 — price up 6%; A should TP, B should not
        decision = comp.on_bar(make_bar(close=106.0, ts=2_000_000_000), make_ctx())
        ids = {o.strategy_id for o in decision.orders if o.reduce_only}
        self.assertIn("a", ids)
        self.assertNotIn("b", ids)


if __name__ == "__main__":
    unittest.main()
