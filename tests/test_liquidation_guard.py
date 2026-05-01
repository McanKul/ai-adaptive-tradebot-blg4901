"""
tests/test_liquidation_guard.py
===============================
Tests for the WebSocket-driven liquidation early-warning guard.

Covers:
* ``PositionManager.compute_liq_price`` — formula correctness for long
  and short at varied leverages.
* ``assign_liq_price`` integration on a freshly opened position.
* ``check_liquidation_warning`` — pre-emptive close fires only when the
  buffer is breached, respects ``action="alarm"`` (no close), and
  correctly handles short positions.
* ``LiquidationGuardConfig`` round-trip through ``LiveConfig.from_dict``.
* ``LiveSupervisor.on_tick`` runs the guard BEFORE local-exit checks.
"""
from __future__ import annotations
import os
import sys
import tempfile
import unittest
from unittest.mock import AsyncMock, MagicMock

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from Backtest.exit_manager import ExitConfig
from Interfaces.strategy_adapter import SizingConfig, SizingMode
from live.live_config import LiquidationGuardConfig, LiveConfig
from live.position_manager import LiveSupervisor, Position, PositionManager


# ---------------------------------------------------------------------------
# compute_liq_price
# ---------------------------------------------------------------------------

class TestComputeLiqPrice(unittest.TestCase):

    def test_long_10x_leverage(self):
        # Long 100 USDT, 10×, MMR=0.005 → liq ≈ 100 * (1 - 0.1 + 0.005) = 90.5
        liq = PositionManager.compute_liq_price("BUY", 100.0, 10.0, 0.005)
        self.assertAlmostEqual(liq, 90.5, places=4)

    def test_short_10x_leverage(self):
        # Short 100 USDT, 10×, MMR=0.005 → liq ≈ 100 * (1 + 0.1 - 0.005) = 109.5
        liq = PositionManager.compute_liq_price("SELL", 100.0, 10.0, 0.005)
        self.assertAlmostEqual(liq, 109.5, places=4)

    def test_long_3x_leverage(self):
        # 3×: cushion ≈ 1/3, liq ≈ 100 * (1 - 0.333 + 0.01) = ~67.67
        liq = PositionManager.compute_liq_price("BUY", 100.0, 3.0, 0.01)
        self.assertAlmostEqual(liq, 67.6667, places=3)

    def test_long_1x_leverage_clamps_at_zero(self):
        # 1× spot: 1 - 1/1 + mmr = mmr → liq = entry * mmr
        liq = PositionManager.compute_liq_price("BUY", 100.0, 1.0, 0.01)
        self.assertAlmostEqual(liq, 1.0, places=4)

    def test_invalid_inputs_return_zero(self):
        self.assertEqual(PositionManager.compute_liq_price("BUY", 100.0, 0.0, 0.01), 0.0)
        self.assertEqual(PositionManager.compute_liq_price("BUY", 0.0, 10.0, 0.01), 0.0)


# ---------------------------------------------------------------------------
# assign_liq_price + check_liquidation_warning
# ---------------------------------------------------------------------------

def _make_pm(liq_cfg=None, leverage=10.0):
    broker = MagicMock()
    broker.close_position = AsyncMock(return_value=True)
    broker.cancel_order = AsyncMock(return_value=True)
    sizing = SizingConfig(
        mode=SizingMode.MARGIN_USD, margin_usd=100.0, leverage=leverage,
    )
    pm = PositionManager(
        broker=broker, sizing_cfg=sizing, exit_cfg=ExitConfig(),
        max_concurrent=1, symbol="BTCUSDT",
        liq_guard_cfg=liq_cfg,
    )
    return pm, broker


def _open_position(pm, side="BUY", entry=100.0, qty=1.0, strategy="test"):
    pos = Position(
        symbol="BTCUSDT", side=side, qty=qty, entry_price=entry,
        opened_ts=1000.0, strategy=strategy, timeframe="1m",
    )
    pm.assign_liq_price(pos)
    pm.open_positions[("BTCUSDT", strategy)] = pos
    return pos


class TestAssignLiqPrice(unittest.TestCase):

    def test_no_assignment_when_guard_disabled(self):
        pm, _ = _make_pm(liq_cfg=None, leverage=10.0)
        pos = _open_position(pm)
        self.assertIsNone(pos.liq_price)

    def test_long_assigned_below_entry(self):
        cfg = LiquidationGuardConfig(enabled=True, maintenance_margin_ratio=0.005)
        pm, _ = _make_pm(liq_cfg=cfg, leverage=10.0)
        pos = _open_position(pm, side="BUY", entry=100.0)
        self.assertIsNotNone(pos.liq_price)
        self.assertLess(pos.liq_price, pos.entry)
        self.assertAlmostEqual(pos.liq_price, 90.5, places=4)

    def test_short_assigned_above_entry(self):
        cfg = LiquidationGuardConfig(enabled=True, maintenance_margin_ratio=0.005)
        pm, _ = _make_pm(liq_cfg=cfg, leverage=10.0)
        pos = _open_position(pm, side="SELL", entry=100.0)
        self.assertGreater(pos.liq_price, pos.entry)


class TestCheckLiquidationWarning(unittest.IsolatedAsyncioTestCase):

    async def test_disabled_guard_is_noop(self):
        cfg = LiquidationGuardConfig(enabled=False)
        pm, broker = _make_pm(liq_cfg=cfg, leverage=10.0)
        _open_position(pm)
        # Even at price 90.5 (right at liq) nothing fires
        closed = await pm.check_liquidation_warning("BTCUSDT", 90.5)
        self.assertEqual(closed, [])
        broker.close_position.assert_not_called()

    async def test_long_safe_distance_no_close(self):
        # Long at 100, liq~90.5, cushion=9.5. buffer=0.20 → trigger when
        # remaining < 1.9 (price < 92.4).  At 95 we're still safe.
        cfg = LiquidationGuardConfig(enabled=True, buffer_pct=0.20,
                                     maintenance_margin_ratio=0.005)
        pm, broker = _make_pm(liq_cfg=cfg, leverage=10.0)
        _open_position(pm, side="BUY", entry=100.0)
        closed = await pm.check_liquidation_warning("BTCUSDT", 95.0)
        self.assertEqual(closed, [])
        broker.close_position.assert_not_called()

    async def test_long_breaches_buffer_closes(self):
        cfg = LiquidationGuardConfig(enabled=True, buffer_pct=0.20,
                                     maintenance_margin_ratio=0.005)
        pm, broker = _make_pm(liq_cfg=cfg, leverage=10.0)
        pos = _open_position(pm, side="BUY", entry=100.0)
        # Price at 92.0 → distance=1.5, cushion=9.5, ratio=0.158 < 0.20 → fires
        closed = await pm.check_liquidation_warning("BTCUSDT", 92.0)
        self.assertEqual(closed, ["test"])
        broker.close_position.assert_called_once_with("BTCUSDT")
        self.assertEqual(pos.exit_type, "LIQ_GUARD")
        self.assertNotIn(("BTCUSDT", "test"), pm.open_positions)

    async def test_short_breaches_buffer_closes(self):
        cfg = LiquidationGuardConfig(enabled=True, buffer_pct=0.20,
                                     maintenance_margin_ratio=0.005)
        pm, broker = _make_pm(liq_cfg=cfg, leverage=10.0)
        pos = _open_position(pm, side="SELL", entry=100.0)
        # Short liq~109.5, cushion=9.5. At 108 distance=1.5, ratio=0.158 → fires
        closed = await pm.check_liquidation_warning("BTCUSDT", 108.0)
        self.assertEqual(closed, ["test"])
        self.assertEqual(pos.exit_type, "LIQ_GUARD")

    async def test_alarm_action_logs_only(self):
        cfg = LiquidationGuardConfig(enabled=True, buffer_pct=0.20,
                                     maintenance_margin_ratio=0.005,
                                     action="alarm")
        pm, broker = _make_pm(liq_cfg=cfg, leverage=10.0)
        _open_position(pm, side="BUY", entry=100.0)
        closed = await pm.check_liquidation_warning("BTCUSDT", 91.0)
        self.assertEqual(closed, [])
        broker.close_position.assert_not_called()

    async def test_position_without_liq_price_skipped(self):
        cfg = LiquidationGuardConfig(enabled=True, buffer_pct=0.20)
        pm, broker = _make_pm(liq_cfg=cfg, leverage=10.0)
        pos = _open_position(pm, side="BUY", entry=100.0)
        pos.liq_price = None  # adoption case — we never computed it
        closed = await pm.check_liquidation_warning("BTCUSDT", 91.0)
        self.assertEqual(closed, [])


# ---------------------------------------------------------------------------
# Supervisor.on_tick runs guard BEFORE tick exits
# ---------------------------------------------------------------------------

class TestSupervisorOrdering(unittest.IsolatedAsyncioTestCase):

    async def test_liq_guard_fires_before_tick_exits(self):
        with tempfile.TemporaryDirectory() as tmp:
            cfg = LiquidationGuardConfig(enabled=True, buffer_pct=0.20,
                                         maintenance_margin_ratio=0.005)
            broker = MagicMock()
            broker.close_position = AsyncMock(return_value=True)
            broker.cancel_order = AsyncMock(return_value=True)
            sup = LiveSupervisor(
                broker,
                persist_path=os.path.join(tmp, "p.json"),
                liq_guard_cfg=cfg,
            )
            sup.register_symbol(
                "BTCUSDT",
                sizing_cfg=SizingConfig(
                    mode=SizingMode.MARGIN_USD, margin_usd=100.0, leverage=10.0,
                ),
                exit_cfg=ExitConfig(trailing_stop_pct=0.05),
            )
            pm = sup.get("BTCUSDT")
            _open_position(pm, side="BUY", entry=100.0)

            calls = []
            real_liq = pm.check_liquidation_warning
            real_tick = pm.check_tick_exits

            async def spy_liq(*a, **kw):
                calls.append("liq")
                return await real_liq(*a, **kw)

            async def spy_tick(*a, **kw):
                calls.append("tick")
                return await real_tick(*a, **kw)

            pm.check_liquidation_warning = spy_liq
            pm.check_tick_exits = spy_tick
            await sup.on_tick("BTCUSDT", 92.0)
            self.assertEqual(calls, ["liq", "tick"])


# ---------------------------------------------------------------------------
# YAML round-trip
# ---------------------------------------------------------------------------

class TestLiquidationGuardYaml(unittest.TestCase):

    def test_yaml_overrides(self):
        cfg = LiveConfig.from_dict({
            "liquidation_guard": {
                "enabled": True,
                "buffer_pct": 0.10,
                "maintenance_margin_ratio": 0.008,
                "action": "alarm",
            },
        })
        g = cfg.liquidation_guard
        self.assertTrue(g.enabled)
        self.assertAlmostEqual(g.buffer_pct, 0.10)
        self.assertAlmostEqual(g.maintenance_margin_ratio, 0.008)
        self.assertEqual(g.action, "alarm")


if __name__ == "__main__":
    unittest.main()
