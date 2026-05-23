"""
tests/test_phase_b_min_notional.py
==================================
Phase B3 — MIN_NOTIONAL filter rejects under-sized orders before they
hit the exchange.  Also covers Phase B2 plumbing of intended_price /
fill_price / slippage_bps onto the Position dataclass + CSV columns.
"""
from __future__ import annotations
import csv
import os
import sys
import tempfile
import time
import unittest
from unittest.mock import AsyncMock, MagicMock

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from Backtest.exit_manager import ExitConfig
from Interfaces.strategy_adapter import SizingConfig, SizingMode
from live.live_config import ExecutionConfig
from live.live_metrics import LiveMetrics, _CSV_HEADER
from live.position_manager import PositionManager, Position


def _ex_info_with_filters(symbol, step, tick, min_notional):
    """Synthesize a Binance-shaped exchange_info payload."""
    filters = [
        {"filterType": "LOT_SIZE", "stepSize": str(step)},
        {"filterType": "PRICE_FILTER", "tickSize": str(tick)},
    ]
    if min_notional is not None:
        filters.append({"filterType": "MIN_NOTIONAL", "notional": str(min_notional)})
    return {"symbols": [{"symbol": symbol, "filters": filters}]}


def _make_pm(min_notional, mark_price=100.0, qty_per_size=0.01):
    broker = MagicMock()
    broker.exchange_info = AsyncMock(return_value=_ex_info_with_filters(
        "BTCUSDT", step=0.001, tick=0.01, min_notional=min_notional,
    ))
    broker.get_mark_price = AsyncMock(return_value=mark_price)
    broker.market_order = AsyncMock(return_value={"orderId": 1})
    broker.close_position = AsyncMock(return_value=True)
    broker.cancel_order = AsyncMock(return_value=True)
    broker.set_leverage = AsyncMock()
    broker.ensure_isolated_margin = AsyncMock()
    broker.place_stop_market = AsyncMock(return_value=42)
    broker.place_take_profit = AsyncMock(return_value=43)

    sizing = SizingConfig(
        mode=SizingMode.NOTIONAL_USD, notional_usd=qty_per_size * mark_price,
        leverage=1.0,
    )
    exit_cfg = ExitConfig()
    exit_cfg.use_exchange_orders = False
    pm = PositionManager(
        broker=broker, sizing_cfg=sizing, exit_cfg=exit_cfg,
        max_concurrent=1, symbol="BTCUSDT",
        execution_cfg=ExecutionConfig(max_entry_spread_bps=0.0),
    )
    return pm, broker


# ---------------------------------------------------------------------------
# _symbol_filters extracts MIN_NOTIONAL
# ---------------------------------------------------------------------------

class TestSymbolFiltersMinNotional(unittest.IsolatedAsyncioTestCase):

    async def test_min_notional_extracted(self):
        pm, _ = _make_pm(min_notional=10.0)
        qty, tick, mn = await pm._symbol_filters("BTCUSDT", 0.05)
        self.assertAlmostEqual(qty, 0.05, places=6)
        self.assertAlmostEqual(tick, 0.01, places=6)
        self.assertAlmostEqual(mn, 10.0, places=6)

    async def test_no_filter_returns_none(self):
        pm, _ = _make_pm(min_notional=None)
        _, _, mn = await pm._symbol_filters("BTCUSDT", 0.05)
        self.assertIsNone(mn)


# ---------------------------------------------------------------------------
# open_position rejects when notional < min_notional
# ---------------------------------------------------------------------------

class TestMinNotionalEntry(unittest.IsolatedAsyncioTestCase):

    async def test_below_min_notional_rejected(self):
        # qty 0.01 * mark 100 = 1 USD; min_notional 10 USD → reject
        pm, broker = _make_pm(min_notional=10.0, mark_price=100.0,
                               qty_per_size=0.01)
        ok = await pm.open_position(
            "BTCUSDT", side=1, strategy_name="t", leverage=1, timeframe="1m",
        )
        self.assertFalse(ok)
        broker.market_order.assert_not_called()

    async def test_above_min_notional_admitted(self):
        # qty 0.5 * mark 100 = 50 USD; min_notional 10 USD → ok
        pm, broker = _make_pm(min_notional=10.0, mark_price=100.0,
                               qty_per_size=0.5)
        ok = await pm.open_position(
            "BTCUSDT", side=1, strategy_name="t", leverage=1, timeframe="1m",
        )
        self.assertTrue(ok)
        broker.market_order.assert_awaited_once()

    async def test_no_min_notional_skips_check(self):
        # When the filter is absent the order proceeds regardless
        pm, broker = _make_pm(min_notional=None, mark_price=100.0,
                               qty_per_size=0.01)
        ok = await pm.open_position(
            "BTCUSDT", side=1, strategy_name="t", leverage=1, timeframe="1m",
        )
        self.assertTrue(ok)


# ---------------------------------------------------------------------------
# Position has Phase B2 fields, CSV header carries them
# ---------------------------------------------------------------------------

class TestPositionExecutionFields(unittest.TestCase):

    def test_position_initial_state(self):
        pos = Position(
            symbol="BTCUSDT", side="BUY", qty=1.0,
            entry_price=100.0, opened_ts=time.time(),
            strategy="t", timeframe="1m",
        )
        self.assertEqual(pos.intended_price, 100.0)
        self.assertIsNone(pos.fill_price)
        self.assertIsNone(pos.slippage_bps)

    def test_csv_header_includes_b2_columns(self):
        for col in ("intended_price", "fill_price", "slippage_bps"):
            self.assertIn(col, _CSV_HEADER)

    def test_csv_record_writes_b2_when_populated(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "trades.csv")
            m = LiveMetrics(csv_path=path, start_equity=1_000.0)
            pos = Position(
                symbol="BTCUSDT", side="BUY", qty=1.0,
                entry_price=100.0, opened_ts=time.time(),
                strategy="t", timeframe="1m",
            )
            pos.exit = 110.0
            pos.exit_ts = time.time() + 60
            pos.exit_type = "TP"
            pos.intended_price = 100.0
            pos.fill_price = 100.05
            pos.slippage_bps = 5.0
            m.record(pos)
            with open(path, "r", newline="", encoding="utf-8") as f:
                rows = list(csv.DictReader(f))
            self.assertEqual(len(rows), 1)
            self.assertEqual(float(rows[0]["intended_price"]), 100.0)
            self.assertEqual(float(rows[0]["fill_price"]), 100.05)
            self.assertEqual(float(rows[0]["slippage_bps"]), 5.0)


if __name__ == "__main__":
    unittest.main()
