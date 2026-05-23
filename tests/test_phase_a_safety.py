"""
tests/test_phase_a_safety.py
============================
Phase A regression tests:

* A1 — ``max_daily_loss`` USD and pct circuit breakers actually trip
  the kill switch from inside ``check_account_risk``.
* A2 — Missing-stop guard at entry refuses to track an unprotected
  position; tick-time recheck flattens any position whose
  ``sl_order_id`` becomes ``None``.
* A3 — Consecutive-loss cooldown blocks while the timer is alive and
  resumes automatically when ``time.time() >= cooldown_until_ts``.
* A4 — ``GlobalRiskConfig`` accepts both the new
  ``max_concurrent_positions`` field AND the legacy
  ``max_correlated_positions`` alias from YAML.
"""
from __future__ import annotations
import os
import sys
import tempfile
import time
import unittest
from unittest.mock import AsyncMock, MagicMock

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from Backtest.exit_manager import ExitConfig
from Interfaces.strategy_adapter import SizingConfig, SizingMode
from live.global_risk import LiveGlobalRisk
from live.live_config import GlobalRiskConfig, LiveConfig
from live.position_manager import LiveSupervisor, Position, PositionManager


def _gr(tmpdir, **overrides):
    """Build a LiveGlobalRisk with a temp persist file and overrides."""
    cfg = GlobalRiskConfig(
        persist_path=os.path.join(tmpdir, "r.json"),
        **overrides,
    )
    gr = LiveGlobalRisk(cfg)
    gr.set_start_equity(1_000.0)
    return gr, cfg


# ---------------------------------------------------------------------------
# A1 — daily-loss enforcement
# ---------------------------------------------------------------------------

class TestDailyLossEnforce(unittest.TestCase):

    def test_usd_threshold_trips_kill_switch(self):
        with tempfile.TemporaryDirectory() as tmp:
            gr, _ = _gr(tmp, max_daily_loss=30.0, max_daily_loss_pct=0.0)
            gr.record_pnl(-15.0)
            ok, _ = gr.check_account_risk(1_000.0, 0.0, 0)
            self.assertTrue(ok)
            gr.record_pnl(-20.0)  # cumulative -35 ≥ 30
            ok, reason = gr.check_account_risk(1_000.0, 0.0, 0)
            self.assertFalse(ok)
            self.assertIn("daily loss", reason.lower())
            self.assertTrue(gr.is_kill_switch_active)

    def test_pct_threshold_trips_kill_switch(self):
        with tempfile.TemporaryDirectory() as tmp:
            gr, _ = _gr(tmp, max_daily_loss=10_000.0, max_daily_loss_pct=0.05)
            gr.record_pnl(-60.0)  # 6% of 1000 → over
            ok, reason = gr.check_account_risk(940.0, 0.0, 0)
            self.assertFalse(ok)
            self.assertIn("daily loss", reason.lower())

    def test_zero_thresholds_disable_check(self):
        with tempfile.TemporaryDirectory() as tmp:
            gr, _ = _gr(tmp, max_daily_loss=0.0, max_daily_loss_pct=0.0,
                        max_account_drawdown_pct=0.99)
            gr.record_pnl(-500.0)
            ok, _ = gr.check_account_risk(1_000.0, 0.0, 0)
            # No daily-loss trip; drawdown threshold is 99% so equity 1000 vs
            # peak 1000 stays inside it
            self.assertTrue(ok)
            self.assertFalse(gr.is_kill_switch_active)

    def test_kill_switch_persists_across_reinit(self):
        with tempfile.TemporaryDirectory() as tmp:
            gr1, cfg = _gr(tmp, max_daily_loss=10.0, max_daily_loss_pct=0.0)
            gr1.record_pnl(-15.0)
            gr1.check_account_risk(1_000.0, 0.0, 0)
            self.assertTrue(gr1.is_kill_switch_active)
            # Re-instantiate
            gr2 = LiveGlobalRisk(cfg)
            self.assertTrue(gr2.is_kill_switch_active)


# ---------------------------------------------------------------------------
# A3 — cooldown
# ---------------------------------------------------------------------------

class TestCooldown(unittest.TestCase):

    def test_three_losses_arm_cooldown(self):
        with tempfile.TemporaryDirectory() as tmp:
            gr, _ = _gr(tmp, max_daily_loss=10_000.0, max_daily_loss_pct=0.0,
                        cooldown_after_losses=3, cooldown_seconds=600)
            gr.record_pnl(-1.0)
            gr.record_pnl(-1.0)
            ok, _ = gr.check_account_risk(1_000.0, 0.0, 0)
            self.assertTrue(ok)  # only 2 losses, no cooldown
            gr.record_pnl(-1.0)
            ok, reason = gr.check_account_risk(1_000.0, 0.0, 0)
            self.assertFalse(ok)
            self.assertIn("cooldown", reason.lower())

    def test_winner_resets_streak(self):
        with tempfile.TemporaryDirectory() as tmp:
            gr, _ = _gr(tmp, max_daily_loss=10_000.0, max_daily_loss_pct=0.0,
                        cooldown_after_losses=3, cooldown_seconds=600)
            gr.record_pnl(-1.0)
            gr.record_pnl(-1.0)
            gr.record_pnl(+5.0)
            gr.record_pnl(-1.0)
            ok, _ = gr.check_account_risk(1_000.0, 0.0, 0)
            self.assertTrue(ok)  # streak was reset, only 1 loss now

    def test_auto_resume_when_timer_expires(self):
        with tempfile.TemporaryDirectory() as tmp:
            gr, _ = _gr(tmp, max_daily_loss=10_000.0, max_daily_loss_pct=0.0,
                        cooldown_after_losses=2, cooldown_seconds=300)
            gr.record_pnl(-1.0)
            gr.record_pnl(-1.0)
            ok, _ = gr.check_account_risk(1_000.0, 0.0, 0)
            self.assertFalse(ok)
            # Force-expire the cooldown
            gr._cooldown_until_ts = time.time() - 1.0
            ok, _ = gr.check_account_risk(1_000.0, 0.0, 0)
            self.assertTrue(ok)
            # Kill switch was NOT tripped — cooldown is a soft block
            self.assertFalse(gr.is_kill_switch_active)

    def test_disabled_when_threshold_zero(self):
        with tempfile.TemporaryDirectory() as tmp:
            gr, _ = _gr(tmp, max_daily_loss=10_000.0, max_daily_loss_pct=0.0,
                        cooldown_after_losses=0)
            for _ in range(5):
                gr.record_pnl(-1.0)
            ok, _ = gr.check_account_risk(1_000.0, 0.0, 0)
            self.assertTrue(ok)

    def test_cooldown_persisted(self):
        with tempfile.TemporaryDirectory() as tmp:
            gr1, cfg = _gr(tmp, max_daily_loss=10_000.0, max_daily_loss_pct=0.0,
                           cooldown_after_losses=2, cooldown_seconds=600)
            gr1.record_pnl(-1.0)
            gr1.record_pnl(-1.0)
            gr2 = LiveGlobalRisk(cfg)
            ok, _ = gr2.check_account_risk(1_000.0, 0.0, 0)
            self.assertFalse(ok)


# ---------------------------------------------------------------------------
# A4 — config alias
# ---------------------------------------------------------------------------

class TestConfigAlias(unittest.TestCase):

    def test_legacy_correlated_key_maps_to_concurrent(self):
        cfg = LiveConfig.from_dict({
            "global_risk": {
                "max_correlated_positions": 7,  # legacy spelling
            },
        })
        self.assertEqual(cfg.global_risk.max_concurrent_positions, 7)

    def test_new_key_takes_precedence(self):
        cfg = LiveConfig.from_dict({
            "global_risk": {
                "max_correlated_positions": 7,
                "max_concurrent_positions": 4,  # both present, new wins
            },
        })
        self.assertEqual(cfg.global_risk.max_concurrent_positions, 4)


# ---------------------------------------------------------------------------
# A2 — missing-stop guard
# ---------------------------------------------------------------------------

def _make_pm():
    broker = MagicMock()
    broker.market_order = AsyncMock(return_value={"orderId": 1})
    broker.close_position = AsyncMock(return_value=True)
    broker.cancel_order = AsyncMock(return_value=True)
    broker.place_stop_market = AsyncMock()  # caller decides return
    broker.place_take_profit = AsyncMock()
    sizing = SizingConfig(
        mode=SizingMode.MARGIN_USD, margin_usd=100.0, leverage=1.0,
    )
    exit_cfg = ExitConfig()
    exit_cfg.use_exchange_orders = True
    pm = PositionManager(
        broker=broker, sizing_cfg=sizing, exit_cfg=exit_cfg,
        max_concurrent=1, symbol="BTCUSDT",
    )
    return pm, broker


class TestMissingStopTickGuard(unittest.IsolatedAsyncioTestCase):

    async def test_missing_stop_at_tick_force_closes(self):
        pm, broker = _make_pm()
        pos = Position(
            symbol="BTCUSDT", side="BUY", qty=1.0, entry_price=100.0,
            sl_price=95.0, opened_ts=time.time(),
            strategy="test", timeframe="1m",
        )
        pos.sl_order_id = None  # the guard's trigger condition
        pos.tp_order_id = 42
        pm.open_positions[("BTCUSDT", "test")] = pos

        closed = await pm.check_missing_stop("BTCUSDT", 100.0)
        self.assertEqual(closed, ["test"])
        broker.close_position.assert_awaited_once_with("BTCUSDT")
        self.assertNotIn(("BTCUSDT", "test"), pm.open_positions)
        self.assertEqual(pos.exit_type, "MISSING_STOP")

    async def test_present_stop_not_touched(self):
        pm, broker = _make_pm()
        pos = Position(
            symbol="BTCUSDT", side="BUY", qty=1.0, entry_price=100.0,
            sl_price=95.0, opened_ts=time.time(),
            strategy="test", timeframe="1m",
        )
        pos.sl_order_id = 1234  # healthy
        pm.open_positions[("BTCUSDT", "test")] = pos

        closed = await pm.check_missing_stop("BTCUSDT", 100.0)
        self.assertEqual(closed, [])
        broker.close_position.assert_not_called()

    async def test_disabled_when_use_exchange_orders_false(self):
        pm, broker = _make_pm()
        pm.exit_cfg.use_exchange_orders = False
        pos = Position(
            symbol="BTCUSDT", side="BUY", qty=1.0, entry_price=100.0,
            opened_ts=time.time(), strategy="test", timeframe="1m",
        )
        pos.sl_order_id = None
        pm.open_positions[("BTCUSDT", "test")] = pos
        closed = await pm.check_missing_stop("BTCUSDT", 100.0)
        self.assertEqual(closed, [])
        broker.close_position.assert_not_called()


if __name__ == "__main__":
    unittest.main()
