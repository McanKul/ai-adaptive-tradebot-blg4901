"""
tests/test_live_engine_full.py
==============================
Full integration tests for the LiveEngine with mocked broker/streamer.

Tests cover:
- LONG and SHORT position lifecycle (open → strategy exit → closed)
- Global position limit enforcement (max 2 concurrent)
- bars_held increments correctly (1 per bar, not N per bar)
- MAX_BARS safety net triggers at correct time
- Risk block allows strategy exits for open positions
- Kill switch does NOT fire for position limit (only drawdown/loss)
"""
import asyncio
import os
import sys
import tempfile
import time
import unittest
from unittest.mock import AsyncMock, MagicMock, patch
from typing import Optional, List, Dict, Any

# ── Mock binance before any project imports ──────────────────────────
sys.modules["binance"] = MagicMock()
sys.modules["binance.client"] = MagicMock()
sys.modules["binance.exceptions"] = MagicMock()
sys.modules["binance.enums"] = MagicMock()
sys.modules["binance.enums"].SIDE_BUY = "BUY"
sys.modules["binance.enums"].SIDE_SELL = "SELL"
sys.modules["binance.enums"].FUTURE_ORDER_TYPE_MARKET = "MARKET"
sys.modules["binance.enums"].FUTURE_ORDER_TYPE_STOP_MARKET = "STOP_MARKET"
sys.modules["binance.enums"].FUTURE_ORDER_TYPE_TAKE_PROFIT_MARKET = "TAKE_PROFIT_MARKET"

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from live.live_engine import LiveEngine
from live.live_config import (
    LiveConfig, SizingConfig, ExitConfig, RiskConfig,
    GlobalRiskConfig, ExecutionConfig,
)
from live.global_risk import LiveGlobalRisk
from live.position_manager import LiveSupervisor, Position
from Interfaces.IStrategy import IStrategy, StrategyDecision
from Interfaces.orders import Order, OrderType, OrderSide
from Interfaces.market_data import Bar


# =====================================================================
# Test strategy — deterministic signals based on price
# =====================================================================
class PriceBasedStrategy(IStrategy):
    """
    Deterministic strategy for testing:
    - close > 100 and rising → BUY
    - close < 100 and falling → SELL
    - close == 100 → no action

    Also produces exit signals (reduce_only) when:
    - Long position and close < 95 → exit
    - Short position and close > 105 → exit
    """
    def __init__(self, **kw):
        self._prev_close: Dict[str, float] = {}

    def on_bar(self, bar: Bar, ctx: Any) -> StrategyDecision:
        sym = bar.symbol
        prev = self._prev_close.get(sym, 100.0)  # default=100 (neutral)
        self._prev_close[sym] = bar.close
        position = getattr(ctx, "position", 0.0) or 0.0
        orders: List[Order] = []

        # Exit logic
        if position > 0.01 and bar.close < 95:
            orders.append(Order(
                symbol=sym, side=OrderSide.SELL,
                order_type=OrderType.MARKET, quantity=abs(position),
                strategy_id="TEST_EXIT", reduce_only=True,
            ))
            return StrategyDecision(
                orders=orders, metadata={"exit_reason": "price_drop_exit"}
            )
        if position < -0.01 and bar.close > 105:
            orders.append(Order(
                symbol=sym, side=OrderSide.BUY,
                order_type=OrderType.MARKET, quantity=abs(position),
                strategy_id="TEST_EXIT", reduce_only=True,
            ))
            return StrategyDecision(
                orders=orders, metadata={"exit_reason": "price_rise_exit"}
            )

        # Entry logic
        if bar.close > 100 and bar.close > prev:
            orders.append(Order(
                symbol=sym, side=OrderSide.BUY,
                order_type=OrderType.MARKET, quantity=1.0,
                strategy_id="TEST_ENTRY",
            ))
            return StrategyDecision(orders=orders, metadata={"entry_side": "long"})

        if bar.close < 100 and bar.close < prev:
            orders.append(Order(
                symbol=sym, side=OrderSide.SELL,
                order_type=OrderType.MARKET, quantity=1.0,
                strategy_id="TEST_ENTRY",
            ))
            return StrategyDecision(orders=orders, metadata={"entry_side": "short"})

        return StrategyDecision.no_action()

    def reset(self):
        self._prev_close.clear()


# =====================================================================
# Helpers
# =====================================================================
def make_config(
    symbols=None,
    max_concurrent=2,
    max_correlated=2,
    max_holding_bars=16,
) -> LiveConfig:
    return LiveConfig(
        strategy_class="PriceBasedStrategy",
        strategy_params={},
        symbols=symbols or ["AVAXUSDT", "DOTUSDT", "LINKUSDT"],
        timeframe="5m",
        name="TestEMACross",
        sizing=SizingConfig(leverage=10, margin_usd=5.0),
        exit=ExitConfig(max_holding_bars=max_holding_bars),
        risk=RiskConfig(max_concurrent_positions=max_concurrent),
        execution=ExecutionConfig(preload_bars=10),
        global_risk=GlobalRiskConfig(
            max_correlated_positions=max_correlated,
            persist_path=tempfile.mktemp(suffix=".json"),
        ),
    )


def make_broker(mark_prices: Optional[Dict[str, float]] = None) -> AsyncMock:
    prices = mark_prices or {"AVAXUSDT": 9.0, "DOTUSDT": 1.4, "LINKUSDT": 8.7}
    broker = AsyncMock()
    broker.client = AsyncMock()
    broker.client.futures_exchange_info = AsyncMock(return_value={
        "symbols": [
            {"symbol": s, "quoteAsset": "USDT", "status": "TRADING"}
            for s in prices
        ]
    })
    broker.balance = AsyncMock(return_value=100.0)

    async def _get_mark_price(symbol):
        return prices.get(symbol, 10.0)
    broker.get_mark_price = AsyncMock(side_effect=_get_mark_price)

    broker.ensure_isolated_margin = AsyncMock()
    broker.set_leverage = AsyncMock()
    broker.market_order = AsyncMock()
    broker.place_stop_market = AsyncMock(return_value=None)
    broker.place_take_profit = AsyncMock(return_value=None)
    broker.cancel_order = AsyncMock()
    broker.close_position = AsyncMock()

    # Track open positions so position_amt returns correct values
    broker._open_positions: Dict[str, float] = {}
    _orig_market_order = broker.market_order

    async def _market_order(symbol, side, qty):
        if side == "BUY":
            broker._open_positions[symbol] = broker._open_positions.get(symbol, 0) + qty
        else:
            broker._open_positions[symbol] = broker._open_positions.get(symbol, 0) - qty
    broker.market_order = AsyncMock(side_effect=_market_order)

    async def _close_position(symbol):
        broker._open_positions[symbol] = 0.0
    broker.close_position = AsyncMock(side_effect=_close_position)

    async def _position_amt(symbol):
        return broker._open_positions.get(symbol, 0.0)
    broker.position_amt = AsyncMock(side_effect=_position_amt)

    async def _exchange_info():
        return {
            "symbols": [{
                "symbol": s,
                "filters": [
                    {"filterType": "LOT_SIZE", "stepSize": "0.1"},
                    {"filterType": "PRICE_FILTER", "tickSize": "0.0001"},
                ],
            } for s in prices]
        }
    broker.exchange_info = AsyncMock(side_effect=_exchange_info)
    return broker


def make_bar_event(symbol: str, close: float, tf: str = "5m") -> dict:
    """Create a bar event dict like the Streamer produces."""
    ts = int(time.time() * 1000)
    return {
        "s": symbol,
        "k": {
            "i": tf,
            "t": ts,
            "o": str(close),
            "h": str(close + 1),
            "l": str(close - 1),
            "c": str(close),
            "v": "1000",
            "x": True,
        },
    }


def setup_engine_state(engine: LiveEngine, cfg: LiveConfig):
    """Manually set up engine state (skip run() startup sequence)."""
    engine.symbols = cfg.symbols
    for sym in cfg.symbols:
        engine.supervisor.register_symbol(
            symbol=sym,
            sizing_cfg=cfg.sizing_for(sym),
            exit_cfg=cfg.exit_for(sym),
            max_concurrent=cfg.risk.max_concurrent_positions,
        )
        # Populate BarStore with minimal data
        for i in range(30):
            engine.bar_store.add_bar(sym, cfg.timeframe, {
                "t": 1000 + i * 300000,
                "o": "100", "h": "101", "l": "99",
                "c": "100", "v": "500", "x": True, "i": cfg.timeframe,
            })
    engine.global_risk.set_start_equity(100.0)
    engine._last_equity = 100.0
    engine._equity_ts = time.monotonic()


async def feed_bar(engine: LiveEngine, bar_event: dict):
    """Simulate one bar event through the engine's processing logic."""
    sym = bar_event["s"]
    tf = bar_event["k"]["i"]

    if tf != engine.cfg.timeframe or sym not in engine.symbols:
        return

    equity = await engine._refresh_equity()
    engine.metrics.update_equity(equity)

    try:
        open_count = len(engine.supervisor.open_positions)
        total_exposure = sum(
            pos.entry * pos.qty
            for pos in engine.supervisor.open_positions.values()
        )
        risk_ok, risk_reason = engine.global_risk.check_account_risk(
            current_equity=equity,
            total_exposure_usd=total_exposure,
            open_position_count=open_count,
        )
        has_open_pos = engine.supervisor.position_qty(sym) != 0.0
        if not risk_ok and not has_open_pos:
            await engine.supervisor.update_symbol(sym)
            return
        risk_block_entries = not risk_ok
        if risk_ok:
            engine._risk_block_logged = False
    except Exception:
        risk_block_entries = False

    bar_dict = bar_event["k"]
    bar = Bar(
        symbol=sym, timeframe=tf,
        timestamp_ns=int(bar_dict.get("t", 0)) * 1_000_000,
        open=float(bar_dict["o"]), high=float(bar_dict["h"]),
        low=float(bar_dict["l"]), close=float(bar_dict["c"]),
        volume=float(bar_dict["v"]),
    )

    current_pos = engine.supervisor.position_qty(sym)
    ctx = type('Ctx', (), {
        'symbol': sym, 'timeframe': tf, 'bar_store': engine.bar_store,
        'position': current_pos, 'equity': equity, 'cash': equity,
        'timestamp_ns': bar.timestamp_ns,
        'get_ohlcv': lambda: engine.bar_store.get_ohlcv(sym, tf),
    })()

    decision = engine.strategy.on_bar(bar, ctx)

    strategy_sig = None
    has_exit_signal = False
    exit_reason = None

    if decision.has_orders:
        for order in decision.orders:
            if getattr(order, 'reduce_only', False):
                has_exit_signal = True
                exit_reason = (decision.metadata or {}).get(
                    "exit_reason", "STRATEGY_EXIT"
                )
            else:
                strategy_sig = "+1" if order.side == OrderSide.BUY else "-1"
    elif decision.has_signal:
        strategy_sig = decision.signal

    prev_history_len = len(engine.supervisor.history)

    if has_exit_signal and engine.supervisor.position_qty(sym) != 0.0:
        await engine.supervisor.close_position(
            symbol=sym, strategy_name=engine.cfg.name,
            exit_type=exit_reason or "STRATEGY_EXIT",
        )

    final_sig = await engine._get_combined_signal(sym, strategy_sig)
    if final_sig and not risk_block_entries:
        leverage = engine.cfg.leverage_for(sym)
        await engine.supervisor.open_position(
            symbol=sym, side=final_sig,
            strategy_name=engine.cfg.name,
            leverage=leverage, timeframe=tf,
        )

    await engine.supervisor.update_symbol(sym)

    # Add bar to store for strategy's historical data
    engine.bar_store.add_bar(sym, tf, bar_dict)


# =====================================================================
# Test Cases
# =====================================================================
class TestLiveEngineLongShort(unittest.IsolatedAsyncioTestCase):
    """Test LONG and SHORT position lifecycle."""

    async def test_long_position_opens_and_exits(self):
        """Price > 100 rising → LONG opens; price drops < 95 → exits."""
        cfg = make_config(symbols=["AVAXUSDT"])
        broker = make_broker()
        risk = LiveGlobalRisk(cfg.global_risk)
        engine = LiveEngine(cfg, broker, PriceBasedStrategy, risk)
        setup_engine_state(engine, cfg)

        # Bar 1: close=101, prev was 100 → rising above 100 → BUY
        await feed_bar(engine, make_bar_event("AVAXUSDT", 101))

        pos_qty = engine.supervisor.position_qty("AVAXUSDT")
        self.assertGreater(pos_qty, 0, "Should have opened LONG position")
        broker.market_order.assert_called()

        # Bar 2: close=103, still above 95 → no exit
        await feed_bar(engine, make_bar_event("AVAXUSDT", 103))
        pos_qty = engine.supervisor.position_qty("AVAXUSDT")
        self.assertGreater(pos_qty, 0, "LONG should still be open")

        # Bar 3: close=93, below 95 → strategy exit signal
        await feed_bar(engine, make_bar_event("AVAXUSDT", 93))
        pos_qty = engine.supervisor.position_qty("AVAXUSDT")
        self.assertEqual(pos_qty, 0.0, "LONG should be closed by strategy exit")

        # Check history recorded exit
        exits = [p for p in engine.supervisor.history if p.exit_type == "price_drop_exit"]
        self.assertGreater(len(exits), 0, "Should have exit record")

    async def test_short_position_opens_and_exits(self):
        """Price < 100 falling → SHORT opens; price rises > 105 → exits."""
        cfg = make_config(symbols=["DOTUSDT"])
        broker = make_broker()
        risk = LiveGlobalRisk(cfg.global_risk)
        engine = LiveEngine(cfg, broker, PriceBasedStrategy, risk)
        setup_engine_state(engine, cfg)

        # Bar 1: close=99, prev was 100 → falling below 100 → SELL
        await feed_bar(engine, make_bar_event("DOTUSDT", 99))

        pos_qty = engine.supervisor.position_qty("DOTUSDT")
        self.assertLess(pos_qty, 0, "Should have opened SHORT position")

        # Bar 2: close=97, still below 105 → no exit
        await feed_bar(engine, make_bar_event("DOTUSDT", 97))
        pos_qty = engine.supervisor.position_qty("DOTUSDT")
        self.assertLess(pos_qty, 0, "SHORT should still be open")

        # Bar 3: close=106, above 105 → strategy exit signal
        await feed_bar(engine, make_bar_event("DOTUSDT", 106))
        pos_qty = engine.supervisor.position_qty("DOTUSDT")
        self.assertEqual(pos_qty, 0.0, "SHORT should be closed by strategy exit")

        exits = [p for p in engine.supervisor.history if p.exit_type == "price_rise_exit"]
        self.assertGreater(len(exits), 0, "Should have exit record")


class TestGlobalPositionLimit(unittest.IsolatedAsyncioTestCase):
    """Test that max 2 concurrent positions are enforced."""

    async def test_third_position_blocked(self):
        """With max_concurrent=2, third symbol should be blocked."""
        cfg = make_config(
            symbols=["AVAXUSDT", "DOTUSDT", "LINKUSDT"],
            max_concurrent=2, max_correlated=2,
        )
        broker = make_broker()
        risk = LiveGlobalRisk(cfg.global_risk)
        engine = LiveEngine(cfg, broker, PriceBasedStrategy, risk)
        setup_engine_state(engine, cfg)

        # Open 2 positions (LONG on rising prices)
        await feed_bar(engine, make_bar_event("AVAXUSDT", 101))
        await feed_bar(engine, make_bar_event("DOTUSDT", 101))

        # Verify 2 open
        total_open = len(engine.supervisor.open_positions)
        self.assertEqual(total_open, 2, "Should have 2 open positions")

        # Third should be blocked
        await feed_bar(engine, make_bar_event("LINKUSDT", 101))
        total_open = len(engine.supervisor.open_positions)
        self.assertEqual(total_open, 2, "Third position should be blocked")

        # LINKUSDT should have no position
        link_qty = engine.supervisor.position_qty("LINKUSDT")
        self.assertEqual(link_qty, 0.0, "LINKUSDT should not have a position")

    async def test_position_limit_not_exceeded_after_fix(self):
        """Global risk >= check should prevent exactly N+1."""
        cfg = make_config(max_concurrent=2, max_correlated=2)
        broker = make_broker()
        risk = LiveGlobalRisk(cfg.global_risk)
        engine = LiveEngine(cfg, broker, PriceBasedStrategy, risk)
        setup_engine_state(engine, cfg)

        # Open 2
        await feed_bar(engine, make_bar_event("AVAXUSDT", 101))
        await feed_bar(engine, make_bar_event("DOTUSDT", 101))

        # Try 3rd via different path
        await feed_bar(engine, make_bar_event("LINKUSDT", 101))

        # Count actual positions
        total = sum(len(pm.open_positions) for pm in engine.supervisor._managers.values())
        self.assertLessEqual(total, 2, "Never more than 2 positions")


class TestBarsHeldCounter(unittest.IsolatedAsyncioTestCase):
    """Test that bars_held increments correctly with update_symbol."""

    async def test_bars_held_increments_once_per_bar(self):
        """Each bar event for a symbol should increment bars_held by 1, not N."""
        cfg = make_config(
            symbols=["AVAXUSDT", "DOTUSDT", "LINKUSDT"],
            max_concurrent=2, max_correlated=2,
            max_holding_bars=16,
        )
        broker = make_broker()
        risk = LiveGlobalRisk(cfg.global_risk)
        engine = LiveEngine(cfg, broker, PriceBasedStrategy, risk)
        setup_engine_state(engine, cfg)

        # Open AVAXUSDT position
        await feed_bar(engine, make_bar_event("AVAXUSDT", 101))
        self.assertGreater(engine.supervisor.position_qty("AVAXUSDT"), 0)

        # Now simulate 5 bar cycles (3 symbols each)
        for i in range(5):
            await feed_bar(engine, make_bar_event("AVAXUSDT", 102))
            await feed_bar(engine, make_bar_event("DOTUSDT", 100))  # no signal
            await feed_bar(engine, make_bar_event("LINKUSDT", 100))  # no signal

        # Check bars_held on AVAXUSDT's position
        avax_positions = engine.supervisor._managers["AVAXUSDT"].open_positions
        if avax_positions:
            pos = list(avax_positions.values())[0]
            # Initial bar + 5 more = 6 bars (first bar increments too)
            self.assertLessEqual(
                pos.bars_held, 7,
                f"bars_held should be ~6, got {pos.bars_held} "
                f"(would be 18+ with old update_all bug)"
            )
            self.assertGreater(pos.bars_held, 0)

    async def test_max_bars_triggers_at_correct_time(self):
        """MAX_BARS should trigger after exactly max_holding_bars bars."""
        max_bars = 5
        cfg = make_config(
            symbols=["AVAXUSDT"],
            max_concurrent=2, max_correlated=2,
            max_holding_bars=max_bars,
        )
        broker = make_broker()
        risk = LiveGlobalRisk(cfg.global_risk)
        engine = LiveEngine(cfg, broker, PriceBasedStrategy, risk)
        setup_engine_state(engine, cfg)

        # Open position
        await feed_bar(engine, make_bar_event("AVAXUSDT", 101))
        self.assertGreater(engine.supervisor.position_qty("AVAXUSDT"), 0)

        # Feed bars that DON'T trigger strategy exit (close=102, no exit)
        for i in range(max_bars - 1):
            await feed_bar(engine, make_bar_event("AVAXUSDT", 102))
            if engine.supervisor.position_qty("AVAXUSDT") == 0:
                self.fail(
                    f"Position closed after {i+1} bars, expected {max_bars}"
                )

        # One more bar should trigger MAX_BARS
        await feed_bar(engine, make_bar_event("AVAXUSDT", 102))
        pos_qty = engine.supervisor.position_qty("AVAXUSDT")
        self.assertEqual(pos_qty, 0.0, "MAX_BARS should have closed position")


class TestRiskBlockAllowsExits(unittest.IsolatedAsyncioTestCase):
    """Test that risk block allows strategy exits for open positions."""

    async def test_strategy_exit_works_when_risk_blocked(self):
        """
        With 2 positions open (risk blocked), strategy should still
        be able to exit positions via reduce_only orders.
        """
        cfg = make_config(
            symbols=["AVAXUSDT", "DOTUSDT", "LINKUSDT"],
            max_concurrent=2, max_correlated=2,
        )
        broker = make_broker()
        risk = LiveGlobalRisk(cfg.global_risk)
        engine = LiveEngine(cfg, broker, PriceBasedStrategy, risk)
        setup_engine_state(engine, cfg)

        # Open 2 LONG positions
        await feed_bar(engine, make_bar_event("AVAXUSDT", 101))
        await feed_bar(engine, make_bar_event("DOTUSDT", 101))
        self.assertEqual(len(engine.supervisor.open_positions), 2)

        # Now AVAXUSDT drops to 93 → strategy exit should work
        # even though risk says 2 >= 2
        await feed_bar(engine, make_bar_event("AVAXUSDT", 93))

        avax_qty = engine.supervisor.position_qty("AVAXUSDT")
        self.assertEqual(avax_qty, 0.0, "AVAXUSDT should be closed despite risk block")

        # After exit, only 1 position remains
        self.assertEqual(len(engine.supervisor.open_positions), 1)


class TestSupervisorGlobalLimit(unittest.IsolatedAsyncioTestCase):
    """Test LiveSupervisor's _max_global_positions hard gate."""

    async def test_supervisor_blocks_excess_positions(self):
        """Supervisor should block open_position when at limit."""
        broker = make_broker()
        supervisor = LiveSupervisor(broker, max_global_positions=2)

        sizing = SizingConfig(leverage=10, margin_usd=5.0)
        exit_cfg = ExitConfig()

        for sym in ["AVAXUSDT", "DOTUSDT", "LINKUSDT"]:
            supervisor.register_symbol(sym, sizing, exit_cfg, max_concurrent=2)

        # Open 2
        ok1 = await supervisor.open_position("AVAXUSDT", -1, "test", 10, "5m")
        ok2 = await supervisor.open_position("DOTUSDT", 1, "test", 10, "5m")
        self.assertTrue(ok1)
        self.assertTrue(ok2)

        # Third should fail
        ok3 = await supervisor.open_position("LINKUSDT", -1, "test", 10, "5m")
        self.assertFalse(ok3, "Third position should be blocked by global limit")

        # Total open should be 2
        total = sum(len(pm.open_positions) for pm in supervisor._managers.values())
        self.assertEqual(total, 2)


class TestGlobalRiskOffByOne(unittest.IsolatedAsyncioTestCase):
    """Test the >= fix in global risk check."""

    def test_correlated_positions_gte(self):
        """With max=2, count=2 should be blocked (>=), not allowed (>)."""
        risk = LiveGlobalRisk(GlobalRiskConfig(max_correlated_positions=2))
        risk.set_start_equity(1000.0)

        # 0 positions → OK
        ok, _ = risk.check_account_risk(1000.0, 0.0, 0)
        self.assertTrue(ok)

        # 1 position → OK
        ok, _ = risk.check_account_risk(1000.0, 50.0, 1)
        self.assertTrue(ok)

        # 2 positions → BLOCKED (the fix: >= instead of >)
        ok, reason = risk.check_account_risk(1000.0, 100.0, 2)
        self.assertFalse(ok, "2 >= 2 should be blocked")
        self.assertIn("correlated", reason)

        # 3 positions → also BLOCKED
        ok, _ = risk.check_account_risk(1000.0, 150.0, 3)
        self.assertFalse(ok)


class TestMixedLongShort(unittest.IsolatedAsyncioTestCase):
    """Test that both LONG and SHORT positions can coexist."""

    async def test_long_and_short_simultaneously(self):
        """One symbol LONG, another SHORT, both tracked correctly."""
        cfg = make_config(
            symbols=["AVAXUSDT", "DOTUSDT"],
            max_concurrent=2, max_correlated=2,
        )
        broker = make_broker()
        risk = LiveGlobalRisk(cfg.global_risk)
        engine = LiveEngine(cfg, broker, PriceBasedStrategy, risk)
        setup_engine_state(engine, cfg)

        # AVAXUSDT: close=101 rising → LONG
        await feed_bar(engine, make_bar_event("AVAXUSDT", 101))
        # DOTUSDT: close=99 falling → SHORT
        await feed_bar(engine, make_bar_event("DOTUSDT", 99))

        avax_qty = engine.supervisor.position_qty("AVAXUSDT")
        dot_qty = engine.supervisor.position_qty("DOTUSDT")

        self.assertGreater(avax_qty, 0, "AVAXUSDT should be LONG")
        self.assertLess(dot_qty, 0, "DOTUSDT should be SHORT")

        # Total 2 positions
        self.assertEqual(len(engine.supervisor.open_positions), 2)


if __name__ == '__main__':
    unittest.main()
