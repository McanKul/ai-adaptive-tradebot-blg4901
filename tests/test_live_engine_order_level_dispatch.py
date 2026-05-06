"""
tests/test_live_engine_order_level_dispatch.py
================================================
Pin the order-level dispatch contract on LiveEngine.

Background
----------
Before this refactor LiveEngine collapsed `decision.orders` into a
single ``strategy_sig`` (last entry order's side wins, all reduce-only
orders folded into one ``has_exit_signal`` bool).  Composite strategies
that emit a per-slot order list lost their slot semantics there.

The new dispatcher iterates every order, runs reduce-only orders first
(prevents flip races), and uses ``order.strategy_id`` (slot.id) as the
PositionManager bucket key when a composite spec is loaded.

These tests use a stub LiveEngine instance with a mocked supervisor /
notifier / cfg / streamer so we exercise the dispatch logic without
spinning up a broker, websocket, or asyncio event loop in user code.
"""
import os
import sys
import types
import unittest
from types import SimpleNamespace
from unittest import mock
from unittest.mock import AsyncMock, MagicMock, patch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# Pre-mock binance + submodules before any project import.  The project
# reaches into `binance.enums` (position_manager) and the top-level
# `BinanceSocketManager` symbol (streamer) at import time; the bare
# MagicMock stubs cover both surfaces without dragging in real binance.
_binance_stub = types.SimpleNamespace(
    BinanceSocketManager=MagicMock(return_value=MagicMock()),
    AsyncClient=MagicMock(),
    Client=MagicMock(),
)
sys.modules["binance"] = _binance_stub
sys.modules["binance.exceptions"] = types.SimpleNamespace(
    BinanceAPIException=type("BinanceAPIException", (Exception,), {}),
)
sys.modules["binance.enums"] = types.SimpleNamespace(
    SIDE_BUY="BUY",
    SIDE_SELL="SELL",
    FUTURE_ORDER_TYPE_MARKET="MARKET",
    FUTURE_ORDER_TYPE_STOP_MARKET="STOP_MARKET",
    FUTURE_ORDER_TYPE_TAKE_PROFIT_MARKET="TAKE_PROFIT_MARKET",
)

from live.live_engine import LiveEngine  # noqa: E402
from Interfaces.market_data import Bar  # noqa: E402
from Interfaces.IStrategy import StrategyDecision  # noqa: E402
from Interfaces.orders import Order, OrderSide, OrderType  # noqa: E402


def _bare_engine() -> LiveEngine:
    """Build a LiveEngine with __init__ bypassed so we can wire in the
    minimum number of mocks each test needs."""
    with patch.object(LiveEngine, "__init__", lambda self, *a, **kw: None):
        engine = LiveEngine()
    engine.cfg = SimpleNamespace(
        name="canary_v1",
        composite_spec=None,
        execution=SimpleNamespace(max_tick_age_seconds=0.0),
        leverage_for=lambda sym: 2,
    )
    engine.supervisor = MagicMock()
    engine.supervisor.position_qty_for = MagicMock(return_value=0.0)
    engine.supervisor.position_qty = MagicMock(return_value=0.0)
    engine.supervisor.open_position = AsyncMock(return_value=True)
    engine.supervisor.close_position = AsyncMock()
    engine.notifier = MagicMock()
    engine.notifier.position_opened = AsyncMock()
    engine.notifier.kill_switch = AsyncMock()
    engine.streamer = None
    engine.news_engine = None
    engine.signal_combiner = None
    return engine


def _order(side: OrderSide, *, reduce_only: bool, strategy_id: str = "X") -> Order:
    return Order(
        symbol="BTCUSDT", side=side, order_type=OrderType.MARKET,
        quantity=1.0, timestamp_ns=0, strategy_id=strategy_id,
        reduce_only=reduce_only,
    )


def _decision(orders, exit_reason: str = "trailing_stop_long") -> StrategyDecision:
    return StrategyDecision(
        orders=orders, features={}, metadata={"exit_reason": exit_reason},
    )


class TestOrdersForExecutionOrdering(unittest.TestCase):
    """`_orders_for_execution`: reduce-only first, FIFO within priority."""

    def test_reduce_only_runs_before_entry(self):
        entry = _order(OrderSide.BUY, reduce_only=False)
        exit_ = _order(OrderSide.SELL, reduce_only=True)
        decision = _decision([entry, exit_])
        ordered = LiveEngine._orders_for_execution(decision)
        self.assertIs(ordered[0], exit_)
        self.assertIs(ordered[1], entry)

    def test_already_correct_order_is_stable(self):
        exit_ = _order(OrderSide.SELL, reduce_only=True)
        entry = _order(OrderSide.BUY, reduce_only=False)
        decision = _decision([exit_, entry])
        ordered = LiveEngine._orders_for_execution(decision)
        self.assertEqual([exit_, entry], ordered)

    def test_two_entries_keep_emit_order(self):
        slot_a = _order(OrderSide.BUY, reduce_only=False, strategy_id="slot_a")
        slot_b = _order(OrderSide.SELL, reduce_only=False, strategy_id="slot_b")
        decision = _decision([slot_a, slot_b])
        ordered = LiveEngine._orders_for_execution(decision)
        # Same priority → FIFO preserved.
        self.assertEqual(["slot_a", "slot_b"], [o.strategy_id for o in ordered])


class TestStrategyNameForOrder(unittest.TestCase):
    """Composite path uses slot.id; single-strategy path uses cfg.name."""

    def test_single_strategy_uses_cfg_name(self):
        engine = _bare_engine()
        engine.cfg.composite_spec = None
        order = _order(OrderSide.BUY, reduce_only=False, strategy_id="EMACROSS_ENTRY")
        self.assertEqual("canary_v1", engine._strategy_name_for_order(order))

    def test_composite_uses_strategy_id(self):
        engine = _bare_engine()
        engine.cfg.composite_spec = "config/composite_example.yaml"
        order = _order(OrderSide.BUY, reduce_only=False, strategy_id="trend_v1")
        self.assertEqual("trend_v1", engine._strategy_name_for_order(order))

    def test_composite_falls_back_to_cfg_name_when_strategy_id_missing(self):
        engine = _bare_engine()
        engine.cfg.composite_spec = "config/composite_example.yaml"
        order = _order(OrderSide.BUY, reduce_only=False, strategy_id=None)
        self.assertEqual("canary_v1", engine._strategy_name_for_order(order))


class TestHandleExitOrder(unittest.IsolatedAsyncioTestCase):
    """Exit handler must be idempotent and pass slot-aware
    (symbol, strategy_name) keys to the supervisor."""

    async def test_idempotent_when_no_position_open(self):
        engine = _bare_engine()
        engine.supervisor.position_qty_for.return_value = 0.0
        order = _order(OrderSide.SELL, reduce_only=True, strategy_id="slot_a")
        decision = _decision([order])
        await engine._handle_exit_order("BTCUSDT", order, "slot_a", decision)
        engine.supervisor.close_position.assert_not_called()

    async def test_closes_with_slot_strategy_name(self):
        engine = _bare_engine()
        engine.supervisor.position_qty_for.return_value = 1.0
        order = _order(OrderSide.SELL, reduce_only=True, strategy_id="slot_a")
        decision = _decision([order], exit_reason="trailing_stop_long")
        await engine._handle_exit_order("BTCUSDT", order, "slot_a", decision)
        engine.supervisor.close_position.assert_awaited_once_with(
            symbol="BTCUSDT",
            strategy_name="slot_a",
            exit_type="trailing_stop_long",
        )


class TestHandleEntryOrder(unittest.IsolatedAsyncioTestCase):
    """Entry handler must honour risk gate, sentiment combine, stale-feed
    gate, and pass slot-aware strategy_name to the supervisor."""

    async def _patch_combine(self, engine, return_value):
        # Bind the real method via __get__, then have it return the
        # provided sentiment-combined value.
        async def _combined(symbol, raw):
            return return_value
        engine._get_combined_signal = _combined

    async def test_risk_block_short_circuits(self):
        engine = _bare_engine()
        order = _order(OrderSide.BUY, reduce_only=False, strategy_id="slot_a")
        await engine._handle_entry_order(
            "BTCUSDT", order, "slot_a", {"c": 50_000.0}, "15m",
            risk_block_entries=True,
        )
        engine.supervisor.open_position.assert_not_called()

    async def test_no_signal_after_combine_short_circuits(self):
        engine = _bare_engine()
        await self._patch_combine(engine, return_value=None)
        order = _order(OrderSide.BUY, reduce_only=False, strategy_id="slot_a")
        await engine._handle_entry_order(
            "BTCUSDT", order, "slot_a", {"c": 50_000.0}, "15m",
            risk_block_entries=False,
        )
        engine.supervisor.open_position.assert_not_called()

    async def test_passes_slot_strategy_name_to_supervisor(self):
        engine = _bare_engine()
        engine.supervisor.position_qty_for.return_value = 0.5
        await self._patch_combine(engine, return_value=1)
        order = _order(OrderSide.BUY, reduce_only=False, strategy_id="trend_v1")
        await engine._handle_entry_order(
            "BTCUSDT", order, "trend_v1", {"c": 50_000.0}, "15m",
            risk_block_entries=False,
        )
        engine.supervisor.open_position.assert_awaited_once_with(
            symbol="BTCUSDT",
            side=1,
            strategy_name="trend_v1",
            leverage=2,
            timeframe="15m",
        )
        engine.notifier.position_opened.assert_awaited_once()

    async def test_short_entry_passes_negative_side(self):
        engine = _bare_engine()
        engine.supervisor.position_qty_for.return_value = 0.5
        await self._patch_combine(engine, return_value=-1)
        order = _order(OrderSide.SELL, reduce_only=False, strategy_id="meanrev_v1")
        await engine._handle_entry_order(
            "BTCUSDT", order, "meanrev_v1", {"c": 50_000.0}, "15m",
            risk_block_entries=False,
        )
        kwargs = engine.supervisor.open_position.await_args.kwargs
        self.assertEqual(kwargs["side"], -1)
        self.assertEqual(kwargs["strategy_name"], "meanrev_v1")


class TestFeedStaleGate(unittest.IsolatedAsyncioTestCase):
    """Stale-feed gate must signal `feed_stale` to the dispatcher and
    leave existing positions untouched (entries blocked, exits not)."""

    async def test_no_streamer_means_not_stale(self):
        engine = _bare_engine()
        engine.cfg.execution.max_tick_age_seconds = 30
        engine.streamer = None
        self.assertFalse(await engine._feed_too_stale_for_entries("BTCUSDT"))

    async def test_disabled_when_max_age_zero(self):
        engine = _bare_engine()
        engine.cfg.execution.max_tick_age_seconds = 0
        engine.streamer = MagicMock(seconds_since_last_message=999)
        self.assertFalse(await engine._feed_too_stale_for_entries("BTCUSDT"))

    async def test_blocks_when_feed_old(self):
        engine = _bare_engine()
        engine.cfg.execution.max_tick_age_seconds = 30
        engine.streamer = MagicMock(seconds_since_last_message=120)
        result = await engine._feed_too_stale_for_entries("BTCUSDT")
        self.assertTrue(result)
        engine.notifier.kill_switch.assert_awaited_once()


class TestEndToEndDispatch(unittest.IsolatedAsyncioTestCase):
    """Composite multi-order scenario: 2 slots emit (entry, entry, exit)
    in a single decision.  The dispatcher must:
      * close the existing slot_a position FIRST (reduce-only first),
      * then open positions for slot_a (re-entry) and slot_b,
      * keying every supervisor call by the slot's strategy_id.
    """

    async def test_two_slot_decision_dispatches_correctly(self):
        engine = _bare_engine()
        engine.cfg.composite_spec = "fake/composite.yaml"

        # Slot A had a long open; the strategy now wants to flip with a
        # reduce-only exit and a fresh entry.  Slot B opens a short.
        engine.supervisor.position_qty_for.side_effect = lambda sym, sname: (
            1.0 if (sym, sname) == ("BTCUSDT", "slot_a") else 0.0
        )

        async def _combined(symbol, raw):
            return int(raw)
        engine._get_combined_signal = _combined

        slot_a_exit = _order(OrderSide.SELL, reduce_only=True, strategy_id="slot_a")
        slot_a_entry = _order(OrderSide.BUY, reduce_only=False, strategy_id="slot_a")
        slot_b_entry = _order(OrderSide.SELL, reduce_only=False, strategy_id="slot_b")
        decision = _decision(
            [slot_a_entry, slot_b_entry, slot_a_exit],  # arbitrary emit order
            exit_reason="macd_cross_exit_long",
        )

        # Run the dispatch loop body manually — same code path as the
        # main run() while-loop, but without the surrounding bar setup.
        for order in engine._orders_for_execution(decision):
            strategy_name = engine._strategy_name_for_order(order)
            if order.reduce_only:
                await engine._handle_exit_order(
                    "BTCUSDT", order, strategy_name, decision,
                )
            else:
                await engine._handle_entry_order(
                    "BTCUSDT", order, strategy_name,
                    {"c": 50_000.0}, "15m",
                    risk_block_entries=False,
                )

        # close_position called exactly once, for slot_a, with the
        # exit_reason from decision metadata.
        engine.supervisor.close_position.assert_awaited_once_with(
            symbol="BTCUSDT",
            strategy_name="slot_a",
            exit_type="macd_cross_exit_long",
        )

        # open_position called twice — slot_a re-entry (long) and
        # slot_b new entry (short).  Order: slot_a_entry first because
        # it was emitted first; slot_b_entry second.  Reduce-only ran
        # before either of these.
        self.assertEqual(2, engine.supervisor.open_position.await_count)
        first_call = engine.supervisor.open_position.await_args_list[0].kwargs
        second_call = engine.supervisor.open_position.await_args_list[1].kwargs
        self.assertEqual(first_call["strategy_name"], "slot_a")
        self.assertEqual(first_call["side"], 1)
        self.assertEqual(second_call["strategy_name"], "slot_b")
        self.assertEqual(second_call["side"], -1)


if __name__ == "__main__":
    unittest.main()
