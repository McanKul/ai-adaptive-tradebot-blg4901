"""
tests/test_tick_exits.py
========================
Tests for the live tick-exit pipeline:

* MarkPriceTickStreamer payload parsing
* PositionManager.check_tick_exits — trailing / USD / time-based
* LiveSupervisor.on_tick wiring

The streamer's network loop is not exercised here (BinanceSocketManager
is heavy to mock); we test the message-extraction and callback dispatch
in isolation, plus the position-manager logic with synthetic tick prices.
"""
from __future__ import annotations
import asyncio
import os
import sys
import time
import unittest
from unittest.mock import AsyncMock, MagicMock

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from live.position_manager import LiveSupervisor, Position, PositionManager
from live.tick_stream import MarkPriceTickStreamer
from Interfaces.strategy_adapter import SizingConfig, SizingMode
from Backtest.exit_manager import ExitConfig


# ---------------------------------------------------------------------------
# MarkPriceTickStreamer message extraction
# ---------------------------------------------------------------------------

class TestTickStreamExtract(unittest.TestCase):

    def _make_streamer(self, source="mark"):
        client = MagicMock()
        client.raw_client = MagicMock()
        # We never call start()/stop() in these tests; just exercise _extract
        return MarkPriceTickStreamer(
            client=client, symbols=["BTCUSDT"],
            on_tick=AsyncMock(), source=source,
        )

    def test_mark_payload_parsed(self):
        s = self._make_streamer("mark")
        symbol, price, ts = s._extract({
            "e": "markPriceUpdate", "E": 1626265555000,
            "s": "BTCUSDT", "p": "33500.50",
        })
        self.assertEqual(symbol, "BTCUSDT")
        self.assertAlmostEqual(price, 33500.50)
        self.assertEqual(ts, 1626265555000)

    def test_book_payload_uses_mid(self):
        s = self._make_streamer("book")
        symbol, price, _ = s._extract({
            "u": 100, "s": "BTCUSDT", "b": "100.0", "a": "102.0",
        })
        self.assertEqual(symbol, "BTCUSDT")
        self.assertAlmostEqual(price, 101.0)

    def test_malformed_payload_returns_none(self):
        s = self._make_streamer("mark")
        symbol, price, _ = s._extract({"foo": "bar"})
        self.assertIsNone(symbol)
        self.assertIsNone(price)

    def test_unknown_source_rejected(self):
        client = MagicMock()
        client.raw_client = MagicMock()
        with self.assertRaises(ValueError):
            MarkPriceTickStreamer(
                client=client, symbols=["X"], on_tick=AsyncMock(), source="bogus",
            )


# ---------------------------------------------------------------------------
# PositionManager.check_tick_exits
# ---------------------------------------------------------------------------

def _make_pm(exit_cfg, sizing_cfg=None):
    """Build a PositionManager with stubbed broker."""
    broker = MagicMock()
    broker.close_position = AsyncMock(return_value=True)
    broker.cancel_order = AsyncMock(return_value=True)
    sizing = sizing_cfg or SizingConfig(
        mode=SizingMode.MARGIN_USD, margin_usd=100.0, leverage=1.0,
    )
    pm = PositionManager(
        broker=broker, sizing_cfg=sizing, exit_cfg=exit_cfg,
        max_concurrent=1, symbol="BTCUSDT",
    )
    return pm, broker


def _open_position(pm, side="BUY", entry=100.0, qty=1.0, sl=None, tp=None):
    pos = Position(
        symbol="BTCUSDT", side=side, qty=qty,
        entry_price=entry, sl_price=sl, tp_price=tp,
        opened_ts=time.time() - 1.0,  # 1s ago
        strategy="test", timeframe="1m",
    )
    pm.open_positions[("BTCUSDT", "test")] = pos
    return pos


class TestCheckTickExits(unittest.IsolatedAsyncioTestCase):

    async def test_no_exit_when_rules_silent(self):
        pm, broker = _make_pm(ExitConfig())  # no rules
        _open_position(pm, entry=100.0)
        closed = await pm.check_tick_exits("BTCUSDT", 105.0)
        self.assertEqual(closed, [])
        broker.close_position.assert_not_called()
        # Position still open
        self.assertEqual(len(pm.open_positions), 1)

    async def test_trailing_stop_fires_on_tick(self):
        # 5% trailing stop, 1x leverage → fires when drawdown from peak ≥ 5%
        pm, broker = _make_pm(ExitConfig(trailing_stop_pct=0.05))
        pos = _open_position(pm, entry=100.0, qty=1.0)
        # Peak at 110 (10% gain)
        await pm.check_tick_exits("BTCUSDT", 110.0)
        self.assertEqual(len(pm.open_positions), 1)  # still open
        # Pull back to 104.4 → drawdown from peak ~5.1% → triggers
        closed = await pm.check_tick_exits("BTCUSDT", 104.0)
        self.assertEqual(closed, ["test"])
        broker.close_position.assert_called_once_with("BTCUSDT")
        self.assertEqual(pos.exit_type, "TRAILING")

    async def test_usd_target_take_profit(self):
        pm, broker = _make_pm(ExitConfig(take_profit_usd=10.0))
        pos = _open_position(pm, entry=100.0, qty=1.0)
        # +9 USD: not enough
        await pm.check_tick_exits("BTCUSDT", 109.0)
        self.assertEqual(len(pm.open_positions), 1)
        # +12 USD: triggers
        closed = await pm.check_tick_exits("BTCUSDT", 112.0)
        self.assertEqual(closed, ["test"])
        self.assertEqual(pos.exit_type, "TP_USD")

    async def test_usd_target_stop_loss(self):
        pm, broker = _make_pm(ExitConfig(stop_loss_usd=5.0))
        pos = _open_position(pm, entry=100.0, qty=1.0)
        closed = await pm.check_tick_exits("BTCUSDT", 94.0)  # -6 USD
        self.assertEqual(closed, ["test"])
        self.assertEqual(pos.exit_type, "SL_USD")

    async def test_max_holding_seconds(self):
        # The Backtest ExitConfig dataclass doesn't have max_holding_seconds
        # but our tick checker reads it via getattr — simulate by patching
        cfg = ExitConfig()
        cfg.max_holding_seconds = 0.5  # 500ms
        pm, broker = _make_pm(cfg)
        pos = _open_position(pm, entry=100.0, qty=1.0)
        # Fudge open_ts to the past
        pos.open_ts = time.time() - 1.0
        closed = await pm.check_tick_exits("BTCUSDT", 100.0)
        self.assertEqual(closed, ["test"])
        self.assertEqual(pos.exit_type, "MAX_HOLD_S")

    async def test_flat_tp_sl_NOT_triggered_by_tick(self):
        # Server-side STOP_MARKET / TAKE_PROFIT_MARKET own these.  The
        # tick checker must NOT race them.
        pm, broker = _make_pm(ExitConfig(
            take_profit_pct=0.02,  # set on local config, but we still skip
            stop_loss_pct=0.02,
        ))
        _open_position(pm, entry=100.0, qty=1.0, sl=98.0, tp=102.0)
        # Price hits TP zone — local check should NOT fire
        await pm.check_tick_exits("BTCUSDT", 102.5)
        self.assertEqual(len(pm.open_positions), 1)
        broker.close_position.assert_not_called()

    async def test_already_closed_position_ignored(self):
        pm, broker = _make_pm(ExitConfig(trailing_stop_pct=0.05))
        pos = _open_position(pm, entry=100.0, qty=1.0)
        pos.closed = True  # mark closed without removing from dict
        await pm.check_tick_exits("BTCUSDT", 104.0)
        broker.close_position.assert_not_called()


# ---------------------------------------------------------------------------
# LiveSupervisor.on_tick wiring
# ---------------------------------------------------------------------------

class TestSupervisorOnTick(unittest.IsolatedAsyncioTestCase):

    async def test_dispatches_to_per_symbol_manager(self):
        broker = MagicMock()
        sup = LiveSupervisor(broker, persist_path="/tmp/_test_ticks.json")
        # Substitute a fake PM that records tick calls
        fake_pm = MagicMock()
        fake_pm.check_tick_exits = AsyncMock(return_value=[])
        sup._managers["BTCUSDT"] = fake_pm

        await sup.on_tick("BTCUSDT", 100.0, ts_ms=123)
        fake_pm.check_tick_exits.assert_awaited_once_with("BTCUSDT", 100.0)

    async def test_unknown_symbol_is_noop(self):
        broker = MagicMock()
        sup = LiveSupervisor(broker, persist_path="/tmp/_test_ticks2.json")
        # No managers registered — must not raise
        await sup.on_tick("UNKNOWN", 1.0, ts_ms=0)


if __name__ == "__main__":
    unittest.main()
