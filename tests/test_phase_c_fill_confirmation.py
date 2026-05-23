"""
tests/test_phase_c_fill_confirmation.py
=======================================
Phase C1 — fill confirmation polling integrated into ``open_position``,
plus the slippage abort branch from Phase B2 that depends on it.

Covers:
* ``BinanceBroker.wait_for_fill`` polls until ``status==FILLED``.
* ``DryBroker.wait_for_fill`` echoes the synthetic fill instantly.
* ``PositionManager.open_position`` honors ``executed_qty``, records
  ``fill_price`` and ``slippage_bps`` on the Position, and aborts when
  realised slippage exceeds ``max_slippage_bps``.
"""
from __future__ import annotations
import asyncio
import os
import sys
import unittest
from unittest.mock import AsyncMock, MagicMock

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from Backtest.exit_manager import ExitConfig
from Interfaces.strategy_adapter import SizingConfig, SizingMode
from live.dry_broker import DryBroker
from live.live_config import ExecutionConfig
from live.position_manager import PositionManager


# ---------------------------------------------------------------------------
# DryBroker.wait_for_fill round-trip
# ---------------------------------------------------------------------------

class TestDryBrokerWaitForFill(unittest.IsolatedAsyncioTestCase):

    async def test_market_order_then_fill(self):
        broker = DryBroker()
        broker.set_price("BTCUSDT", 100.0)
        resp = await broker.market_order("BTCUSDT", "BUY", 1.0)
        oid = resp["orderId"]
        fill = await broker.wait_for_fill("BTCUSDT", oid, timeout=1.0)
        self.assertEqual(fill["status"], "FILLED")
        self.assertAlmostEqual(fill["executed_qty"], 1.0)
        self.assertAlmostEqual(fill["avg_price"], 100.0)

    async def test_unknown_order_id_times_out(self):
        broker = DryBroker()
        fill = await broker.wait_for_fill("BTCUSDT", 999, timeout=0.1)
        self.assertEqual(fill["status"], "TIMEOUT")
        self.assertEqual(fill["executed_qty"], 0.0)


# ---------------------------------------------------------------------------
# BinanceBroker.wait_for_fill polling loop (mock the async client)
# ---------------------------------------------------------------------------

class TestBinanceBrokerWaitForFill(unittest.IsolatedAsyncioTestCase):

    async def _broker(self, sequence):
        """Build a broker where futures_get_order returns successive items
        from *sequence* (one per call)."""
        from live.broker_binance import BinanceBroker

        client = MagicMock()
        client.futures_get_order = AsyncMock(side_effect=sequence)
        broker = BinanceBroker(client)
        return broker

    async def test_immediate_fill(self):
        broker = await self._broker([{
            "status": "FILLED", "executedQty": "1.0", "avgPrice": "100.5",
            "fills": [{"commission": "0.04"}],
        }])
        fill = await broker.wait_for_fill("BTCUSDT", 1, timeout=1.0)
        self.assertEqual(fill["status"], "FILLED")
        self.assertAlmostEqual(fill["executed_qty"], 1.0)
        self.assertAlmostEqual(fill["avg_price"], 100.5)
        self.assertAlmostEqual(fill["commission_usd"], 0.04)

    async def test_pending_then_filled(self):
        broker = await self._broker([
            {"status": "NEW", "executedQty": "0", "avgPrice": "0"},
            {"status": "FILLED", "executedQty": "1.0", "avgPrice": "100.0", "fills": []},
        ])
        fill = await broker.wait_for_fill("BTCUSDT", 1, timeout=2.0)
        self.assertEqual(fill["status"], "FILLED")

    async def test_canceled_short_circuits(self):
        broker = await self._broker([
            {"status": "CANCELED", "executedQty": "0", "avgPrice": "0"},
        ])
        fill = await broker.wait_for_fill("BTCUSDT", 1, timeout=2.0)
        self.assertEqual(fill["status"], "CANCELED")


# ---------------------------------------------------------------------------
# open_position: slippage abort + fill price persistence
# ---------------------------------------------------------------------------

def _ex_info(symbol="BTCUSDT", step=0.001, tick=0.01, min_notional=None):
    f = [
        {"filterType": "LOT_SIZE", "stepSize": str(step)},
        {"filterType": "PRICE_FILTER", "tickSize": str(tick)},
    ]
    if min_notional is not None:
        f.append({"filterType": "MIN_NOTIONAL", "notional": str(min_notional)})
    return {"symbols": [{"symbol": symbol, "filters": f}]}


def _make_pm_with_fill(fill_price, max_slippage_bps=15.0):
    broker = MagicMock()
    broker.exchange_info = AsyncMock(return_value=_ex_info())
    broker.get_mark_price = AsyncMock(return_value=100.0)
    broker.market_order = AsyncMock(return_value={"orderId": 11})
    broker.wait_for_fill = AsyncMock(return_value={
        "status": "FILLED", "executed_qty": 0.5,
        "avg_price": fill_price, "commission_usd": 0.02, "raw": {},
    })
    broker.close_position = AsyncMock(return_value=True)
    broker.cancel_order = AsyncMock(return_value=True)
    broker.set_leverage = AsyncMock()
    broker.ensure_isolated_margin = AsyncMock()
    broker.place_stop_market = AsyncMock(return_value=42)
    broker.place_take_profit = AsyncMock(return_value=43)

    sizing = SizingConfig(
        mode=SizingMode.NOTIONAL_USD, notional_usd=50.0, leverage=1.0,
    )
    exit_cfg = ExitConfig()
    exit_cfg.use_exchange_orders = False
    exec_cfg = ExecutionConfig(
        max_entry_spread_bps=0.0,  # off
        max_slippage_bps=max_slippage_bps,
    )
    pm = PositionManager(
        broker=broker, sizing_cfg=sizing, exit_cfg=exit_cfg,
        max_concurrent=1, symbol="BTCUSDT",
        execution_cfg=exec_cfg,
    )
    return pm, broker


class TestSlippageAbort(unittest.IsolatedAsyncioTestCase):

    async def test_within_budget_admitted_with_slippage_recorded(self):
        # 5 bps adverse on a long: fill 100.05 vs intended 100.0
        pm, broker = _make_pm_with_fill(fill_price=100.05, max_slippage_bps=15.0)
        ok = await pm.open_position(
            "BTCUSDT", side=1, strategy_name="t", leverage=1, timeframe="1m",
        )
        self.assertTrue(ok)
        pos = pm.open_positions[("BTCUSDT", "t")]
        self.assertAlmostEqual(pos.intended_price, 100.0)
        self.assertAlmostEqual(pos.fill_price, 100.05)
        self.assertAlmostEqual(pos.slippage_bps, 5.0, places=2)
        self.assertAlmostEqual(pos.entry_fee_usd, 0.02)
        broker.close_position.assert_not_called()

    async def test_over_budget_aborts(self):
        # 50 bps adverse on a long: 100.50 vs 100.0; max=15 bps
        pm, broker = _make_pm_with_fill(fill_price=100.50, max_slippage_bps=15.0)
        ok = await pm.open_position(
            "BTCUSDT", side=1, strategy_name="t", leverage=1, timeframe="1m",
        )
        self.assertFalse(ok)
        broker.close_position.assert_awaited_once_with("BTCUSDT")
        self.assertNotIn(("BTCUSDT", "t"), pm.open_positions)

    async def test_short_side_signed_correctly(self):
        # SHORT: adverse means price went UP from intended.  Fill 100.10
        # on a sell at intended 100.0 → +10 bps adverse, abort at 5 bps.
        pm, broker = _make_pm_with_fill(fill_price=100.10, max_slippage_bps=5.0)
        ok = await pm.open_position(
            "BTCUSDT", side=-1, strategy_name="t", leverage=1, timeframe="1m",
        )
        self.assertFalse(ok)
        broker.close_position.assert_awaited_once()


if __name__ == "__main__":
    unittest.main()
