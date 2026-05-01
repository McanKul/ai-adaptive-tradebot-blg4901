"""
tests/test_position_reconciliation.py
=====================================
Tests for the periodic position reconciliation safety net.

Verifies:
* ``LiveSupervisor.detect_drift()`` correctly classifies all four
  drift kinds (in_sync / orphan_exchange / ghost_local / size_mismatch
  / side_mismatch).
* ``LiveGlobalRisk.trip_kill_switch()`` is idempotent and persists.
* The drift response policy (``alarm`` / ``halt`` / ``force_flat``)
  routes to the right side-effects.
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
from live.global_risk import LiveGlobalRisk
from live.live_config import GlobalRiskConfig, ReconciliationConfig
from live.position_manager import LiveSupervisor, Position, PositionManager


def _make_supervisor(tmpdir, exchange_qty: float = 0.0):
    broker = MagicMock()
    broker.position_amt = AsyncMock(return_value=exchange_qty)
    broker.close_position = AsyncMock(return_value=True)
    broker.cancel_order = AsyncMock(return_value=True)
    sup = LiveSupervisor(
        broker, persist_path=os.path.join(tmpdir, "pos.json"),
    )
    return sup, broker


def _register_symbol(sup, symbol="BTCUSDT"):
    sup.register_symbol(
        symbol=symbol,
        sizing_cfg=SizingConfig(mode=SizingMode.MARGIN_USD,
                                 margin_usd=100.0, leverage=1.0),
        exit_cfg=ExitConfig(),
    )
    return sup.get(symbol)


def _open_local(pm: PositionManager, side="BUY", qty=0.5,
                symbol="BTCUSDT", strategy="test"):
    pos = Position(
        symbol=symbol, side=side, qty=qty,
        entry_price=100.0, strategy=strategy, timeframe="1m",
    )
    pm.open_positions[(symbol, strategy)] = pos
    return pos


# ---------------------------------------------------------------------------
# detect_drift
# ---------------------------------------------------------------------------

class TestDetectDrift(unittest.IsolatedAsyncioTestCase):

    async def test_in_sync_returns_empty(self):
        with tempfile.TemporaryDirectory() as tmp:
            sup, _ = _make_supervisor(tmp, exchange_qty=0.5)
            pm = _register_symbol(sup)
            _open_local(pm, side="BUY", qty=0.5)
            drifts = await sup.detect_drift()
            self.assertEqual(drifts, [])

    async def test_orphan_exchange(self):
        # Local has nothing; exchange holds 0.3 BTC long
        with tempfile.TemporaryDirectory() as tmp:
            sup, _ = _make_supervisor(tmp, exchange_qty=0.3)
            _register_symbol(sup)
            drifts = await sup.detect_drift()
            self.assertEqual(len(drifts), 1)
            self.assertEqual(drifts[0]["kind"], "orphan_exchange")
            self.assertEqual(drifts[0]["symbol"], "BTCUSDT")
            self.assertAlmostEqual(drifts[0]["abs_diff"], 0.3)

    async def test_ghost_local(self):
        # Local thinks 0.5 long; exchange flat (server-side SL fired silently)
        with tempfile.TemporaryDirectory() as tmp:
            sup, _ = _make_supervisor(tmp, exchange_qty=0.0)
            pm = _register_symbol(sup)
            _open_local(pm, side="BUY", qty=0.5)
            drifts = await sup.detect_drift()
            self.assertEqual(len(drifts), 1)
            self.assertEqual(drifts[0]["kind"], "ghost_local")

    async def test_size_mismatch(self):
        # Local 0.5, exchange 0.3 — partial fill scenario
        with tempfile.TemporaryDirectory() as tmp:
            sup, _ = _make_supervisor(tmp, exchange_qty=0.3)
            pm = _register_symbol(sup)
            _open_local(pm, side="BUY", qty=0.5)
            drifts = await sup.detect_drift()
            self.assertEqual(len(drifts), 1)
            self.assertEqual(drifts[0]["kind"], "size_mismatch")
            self.assertAlmostEqual(drifts[0]["abs_diff"], 0.2)

    async def test_side_mismatch(self):
        # Local long 0.5, exchange short 0.5 — catastrophic
        with tempfile.TemporaryDirectory() as tmp:
            sup, _ = _make_supervisor(tmp, exchange_qty=-0.5)
            pm = _register_symbol(sup)
            _open_local(pm, side="BUY", qty=0.5)
            drifts = await sup.detect_drift()
            self.assertEqual(len(drifts), 1)
            self.assertEqual(drifts[0]["kind"], "side_mismatch")

    async def test_short_in_sync(self):
        # Local short 0.4, exchange short 0.4 → no drift
        with tempfile.TemporaryDirectory() as tmp:
            sup, _ = _make_supervisor(tmp, exchange_qty=-0.4)
            pm = _register_symbol(sup)
            _open_local(pm, side="SELL", qty=0.4)
            drifts = await sup.detect_drift()
            self.assertEqual(drifts, [])

    async def test_tolerance_absorbs_rounding(self):
        # Local 0.5000001, exchange 0.5 — within tol=1e-6
        with tempfile.TemporaryDirectory() as tmp:
            sup, _ = _make_supervisor(tmp, exchange_qty=0.5)
            pm = _register_symbol(sup)
            _open_local(pm, side="BUY", qty=0.5000001)
            drifts = await sup.detect_drift(qty_tolerance=1e-5)
            self.assertEqual(drifts, [])

    async def test_multi_strategy_qty_summed_per_symbol(self):
        # Two strategies open on same symbol: 0.3 + 0.2 long → 0.5 net.
        # Exchange holds exactly 0.5 → no drift.
        with tempfile.TemporaryDirectory() as tmp:
            sup, _ = _make_supervisor(tmp, exchange_qty=0.5)
            pm = _register_symbol(sup)
            _open_local(pm, side="BUY", qty=0.3, strategy="trend")
            _open_local(pm, side="BUY", qty=0.2, strategy="meanrev")
            drifts = await sup.detect_drift()
            self.assertEqual(drifts, [])


# ---------------------------------------------------------------------------
# LiveGlobalRisk.trip_kill_switch
# ---------------------------------------------------------------------------

class TestKillSwitchExternalTrip(unittest.TestCase):

    def test_trips_and_persists(self):
        with tempfile.TemporaryDirectory() as tmp:
            cfg = GlobalRiskConfig(persist_path=os.path.join(tmp, "risk.json"))
            gr = LiveGlobalRisk(cfg)
            self.assertFalse(gr.is_kill_switch_active)
            gr.trip_kill_switch("drift detected")
            self.assertTrue(gr.is_kill_switch_active)
            ok, reason = gr.check_account_risk(
                current_equity=10_000, total_exposure_usd=0, open_position_count=0,
            )
            self.assertFalse(ok)
            self.assertIn("drift detected", reason)

    def test_idempotent_keeps_first_reason(self):
        with tempfile.TemporaryDirectory() as tmp:
            cfg = GlobalRiskConfig(persist_path=os.path.join(tmp, "risk.json"))
            gr = LiveGlobalRisk(cfg)
            gr.trip_kill_switch("first")
            gr.trip_kill_switch("second")
            ok, reason = gr.check_account_risk(
                current_equity=10_000, total_exposure_usd=0, open_position_count=0,
            )
            self.assertIn("first", reason)
            self.assertNotIn("second", reason)


# ---------------------------------------------------------------------------
# Reconciliation action policy (engine-level handler)
# ---------------------------------------------------------------------------

class TestDriftHandler(unittest.IsolatedAsyncioTestCase):
    """Test the action policy without spinning up the full LiveEngine."""

    async def test_alarm_does_not_trip_kill_switch(self):
        # Build a minimal engine-like object with the bits _handle_drift uses
        from live.live_engine import LiveEngine
        with tempfile.TemporaryDirectory() as tmp:
            cfg = GlobalRiskConfig(persist_path=os.path.join(tmp, "r.json"))
            gr = LiveGlobalRisk(cfg)

            engine = MagicMock(spec=LiveEngine)
            engine.global_risk = gr
            engine.notifier = MagicMock()
            engine.notifier.kill_switch = AsyncMock()
            engine.broker = MagicMock()
            engine.broker.close_position = AsyncMock()

            # Bind the real method and call it
            await LiveEngine._handle_drift(
                engine,
                [{"symbol": "BTCUSDT", "kind": "ghost_local",
                  "local_qty": 0.5, "exchange_qty": 0.0, "abs_diff": 0.5}],
                action="alarm",
            )
            self.assertFalse(gr.is_kill_switch_active)
            engine.notifier.kill_switch.assert_awaited_once()

    async def test_halt_trips_kill_switch(self):
        from live.live_engine import LiveEngine
        with tempfile.TemporaryDirectory() as tmp:
            cfg = GlobalRiskConfig(persist_path=os.path.join(tmp, "r.json"))
            gr = LiveGlobalRisk(cfg)

            engine = MagicMock(spec=LiveEngine)
            engine.global_risk = gr
            engine.notifier = MagicMock()
            engine.notifier.kill_switch = AsyncMock()
            engine.broker = MagicMock()
            engine.broker.close_position = AsyncMock()

            await LiveEngine._handle_drift(
                engine,
                [{"symbol": "BTCUSDT", "kind": "side_mismatch",
                  "local_qty": 0.5, "exchange_qty": -0.5, "abs_diff": 1.0}],
                action="halt",
            )
            self.assertTrue(gr.is_kill_switch_active)

    async def test_force_flat_closes_and_trips(self):
        from live.live_engine import LiveEngine
        with tempfile.TemporaryDirectory() as tmp:
            cfg = GlobalRiskConfig(persist_path=os.path.join(tmp, "r.json"))
            gr = LiveGlobalRisk(cfg)

            engine = MagicMock(spec=LiveEngine)
            engine.global_risk = gr
            engine.notifier = MagicMock()
            engine.notifier.kill_switch = AsyncMock()
            engine.broker = MagicMock()
            engine.broker.close_position = AsyncMock()

            await LiveEngine._handle_drift(
                engine,
                [{"symbol": "BTCUSDT", "kind": "ghost_local",
                  "local_qty": 0.5, "exchange_qty": 0.0, "abs_diff": 0.5},
                 {"symbol": "ETHUSDT", "kind": "size_mismatch",
                  "local_qty": 1.0, "exchange_qty": 0.5, "abs_diff": 0.5}],
                action="force_flat",
            )
            self.assertTrue(gr.is_kill_switch_active)
            self.assertEqual(engine.broker.close_position.call_count, 2)


# ---------------------------------------------------------------------------
# ReconciliationConfig wired through LiveConfig YAML
# ---------------------------------------------------------------------------

class TestReconciliationConfigYaml(unittest.TestCase):

    def test_default_disabled(self):
        from live.live_config import LiveConfig
        cfg = LiveConfig()
        self.assertFalse(cfg.reconciliation.enabled)
        self.assertEqual(cfg.reconciliation.action, "halt")

    def test_yaml_overrides_picked_up(self):
        from live.live_config import LiveConfig
        cfg = LiveConfig.from_dict({
            "reconciliation": {
                "enabled": True,
                "interval_seconds": 15.0,
                "qty_tolerance": 1e-5,
                "action": "force_flat",
            },
        })
        self.assertTrue(cfg.reconciliation.enabled)
        self.assertEqual(cfg.reconciliation.interval_seconds, 15.0)
        self.assertEqual(cfg.reconciliation.action, "force_flat")


if __name__ == "__main__":
    unittest.main()
