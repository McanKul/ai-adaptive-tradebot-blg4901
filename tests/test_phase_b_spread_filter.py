"""
tests/test_phase_b_spread_filter.py
===================================
Phase B1 — entry spread filter built on the bookTicker streamer cache.

Covers:
* ``MarkPriceTickStreamer`` cache + ``get_spread_bps`` math.
* ``PositionManager.open_position`` rejects when spread is too wide,
  rejects when no tick has arrived (default-deny), and admits when
  the spread is below the threshold.
* Filter is bypassed when ``execution_cfg`` / ``book_streamer`` are
  ``None`` (backwards compatibility for backtest fixtures and old
  configs that never set them).
"""
from __future__ import annotations
import os
import sys
import unittest
from unittest.mock import AsyncMock, MagicMock

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from Backtest.exit_manager import ExitConfig
from Interfaces.strategy_adapter import SizingConfig, SizingMode
from live.live_config import ExecutionConfig
from live.position_manager import PositionManager
from live.tick_stream import MarkPriceTickStreamer


# ---------------------------------------------------------------------------
# Streamer cache + spread math
# ---------------------------------------------------------------------------

def _book_streamer():
    client = MagicMock()
    client.raw_client = MagicMock()
    return MarkPriceTickStreamer(
        client=client, symbols=["BTCUSDT"],
        on_tick=AsyncMock(), source="book",
    )


class TestStreamerSpreadCache(unittest.TestCase):

    def test_no_data_returns_none(self):
        s = _book_streamer()
        self.assertIsNone(s.get_book("BTCUSDT"))
        self.assertIsNone(s.get_spread_bps("BTCUSDT"))

    def test_bid_ask_cached_after_extract(self):
        s = _book_streamer()
        # Simulate a bookTicker payload through _extract
        sym, price, ts = s._extract({
            "u": 100, "s": "BTCUSDT", "b": "100.00", "a": "100.10",
        })
        self.assertEqual(sym, "BTCUSDT")
        bid, ask, _ = s.get_book("BTCUSDT")
        self.assertAlmostEqual(bid, 100.0)
        self.assertAlmostEqual(ask, 100.1)
        # Spread: 0.10 / 100.05 * 1e4 ≈ 9.99 bps
        self.assertAlmostEqual(s.get_spread_bps("BTCUSDT"), 9.995, places=2)

    def test_mark_streamer_returns_none_for_book(self):
        client = MagicMock()
        client.raw_client = MagicMock()
        s = MarkPriceTickStreamer(
            client=client, symbols=["BTCUSDT"],
            on_tick=AsyncMock(), source="mark",
        )
        self.assertIsNone(s.get_book("BTCUSDT"))
        self.assertIsNone(s.get_spread_bps("BTCUSDT"))

    def test_invalid_book_inputs(self):
        s = _book_streamer()
        # Crossed/flat book → None
        s._book_cache["BTCUSDT"] = (100.0, 100.0, 0)
        self.assertIsNone(s.get_spread_bps("BTCUSDT"))


# ---------------------------------------------------------------------------
# PositionManager.open_position spread filter
# ---------------------------------------------------------------------------

def _make_pm(max_spread_bps=5.0, book_streamer=None):
    broker = MagicMock()
    broker.market_order = AsyncMock(return_value={"orderId": 1})
    broker.close_position = AsyncMock(return_value=True)
    broker.cancel_order = AsyncMock(return_value=True)
    broker.get_mark_price = AsyncMock(return_value=100.0)
    broker.place_stop_market = AsyncMock(return_value=42)
    broker.place_take_profit = AsyncMock(return_value=43)
    broker.set_leverage = AsyncMock()
    broker.ensure_isolated_margin = AsyncMock()
    sizing = SizingConfig(
        mode=SizingMode.MARGIN_USD, margin_usd=100.0, leverage=1.0,
    )
    exit_cfg = ExitConfig()
    exit_cfg.use_exchange_orders = False  # avoid SL/TP path noise
    exec_cfg = ExecutionConfig(max_entry_spread_bps=max_spread_bps)
    pm = PositionManager(
        broker=broker, sizing_cfg=sizing, exit_cfg=exit_cfg,
        max_concurrent=1, symbol="BTCUSDT",
        execution_cfg=exec_cfg, book_streamer=book_streamer,
    )
    return pm, broker


class _StubStreamer:
    def __init__(self, spread_bps):
        self._spread = spread_bps

    def get_spread_bps(self, symbol):
        return self._spread


class TestSpreadFilterEntry(unittest.IsolatedAsyncioTestCase):

    async def test_rejected_when_no_tick_yet(self):
        # Streamer present but no data → spread filter denies (default-deny)
        pm, broker = _make_pm(max_spread_bps=5.0,
                               book_streamer=_StubStreamer(None))
        # Force _symbol_filters to pass — patch it
        async def _filt(sym, qty):
            return qty, 0.01, None  # qty, tick, min_notional
        pm._symbol_filters = _filt
        ok = await pm.open_position(
            "BTCUSDT", side=1, strategy_name="t", leverage=1, timeframe="1m",
        )
        self.assertFalse(ok)
        broker.market_order.assert_not_called()

    async def test_rejected_when_spread_too_wide(self):
        pm, broker = _make_pm(max_spread_bps=3.0,
                               book_streamer=_StubStreamer(10.0))
        async def _filt(sym, qty):
            return qty, 0.01, None  # qty, tick, min_notional
        pm._symbol_filters = _filt
        ok = await pm.open_position(
            "BTCUSDT", side=1, strategy_name="t", leverage=1, timeframe="1m",
        )
        self.assertFalse(ok)
        broker.market_order.assert_not_called()

    async def test_admitted_when_spread_below_threshold(self):
        pm, broker = _make_pm(max_spread_bps=5.0,
                               book_streamer=_StubStreamer(2.0))
        async def _filt(sym, qty):
            return qty, 0.01, None  # qty, tick, min_notional
        pm._symbol_filters = _filt
        ok = await pm.open_position(
            "BTCUSDT", side=1, strategy_name="t", leverage=1, timeframe="1m",
        )
        self.assertTrue(ok)
        broker.market_order.assert_awaited_once()

    async def test_disabled_when_max_is_zero(self):
        # max_entry_spread_bps=0 → filter off, even with no streamer
        pm, broker = _make_pm(max_spread_bps=0.0, book_streamer=None)
        async def _filt(sym, qty):
            return qty, 0.01, None  # qty, tick, min_notional
        pm._symbol_filters = _filt
        ok = await pm.open_position(
            "BTCUSDT", side=1, strategy_name="t", leverage=1, timeframe="1m",
        )
        self.assertTrue(ok)

    async def test_no_streamer_with_filter_on_blocks(self):
        # max_spread_bps>0 with no book streamer used to silently bypass
        # the filter ("allow all"); real money cannot tolerate that.
        # We now default-deny: missing streamer with the filter armed
        # rejects the entry.  Tests that want to skip the filter must
        # set ``max_entry_spread_bps=0`` (see test_disabled_when_max_is_zero).
        pm, broker = _make_pm(max_spread_bps=5.0, book_streamer=None)
        async def _filt(sym, qty):
            return qty, 0.01, None  # qty, tick, min_notional
        pm._symbol_filters = _filt
        ok = await pm.open_position(
            "BTCUSDT", side=1, strategy_name="t", leverage=1, timeframe="1m",
        )
        self.assertFalse(ok)


if __name__ == "__main__":
    unittest.main()
