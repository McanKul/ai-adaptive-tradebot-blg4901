"""
tests/test_live_service_safety.py
==================================
Pin the live-mode default-deny guard.

`_validate_live_strategy_safety` refuses to start LIVE for strategies
whose live execution path is not implemented (or known to diverge from
backtest).  Dry-run is always permitted because the divergence cost is
zero — we are not signing real orders.

The helper is imported directly so this test file does not need a live
broker, websocket client, or async event loop to run.
"""
import os
import sys
import unittest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from core.services.live_service import _validate_live_strategy_safety
from live.live_config import LiveConfig


class TestCompositeLivePermitted(unittest.TestCase):
    """Composite live execution is now wired through LiveEngine's
    order-level dispatcher (each slot.id gets its own PositionManager
    bucket).  The guard that used to refuse it has been removed; these
    tests pin the *new* behaviour so a future regression cannot re-add
    the guard silently.
    """

    def test_composite_spec_set_live_passes(self):
        cfg = LiveConfig.from_dict({
            "strategy": {
                "class": "EMACrossMACDTrend",
                "composite_spec": "config/composite_example.yaml",
            },
        })
        self.assertEqual(cfg.composite_spec, "config/composite_example.yaml")
        # Returns None — composite live is allowed as of the order-level
        # dispatch refactor.  FundingArb is the only remaining refuse.
        self.assertIsNone(_validate_live_strategy_safety(cfg, dry_run=False))

    def test_composite_spec_set_dry_run_passes(self):
        cfg = LiveConfig.from_dict({
            "strategy": {
                "class": "EMACrossMACDTrend",
                "composite_spec": "config/composite_example.yaml",
            },
        })
        self.assertIsNone(_validate_live_strategy_safety(cfg, dry_run=True))


class TestFundingArbLiveGuard(unittest.TestCase):
    def test_funding_arb_live_raises(self):
        cfg = LiveConfig.from_dict({
            "strategy": {"class": "FundingRateArbStrategy"},
        })
        with self.assertRaises(RuntimeError) as ctx:
            _validate_live_strategy_safety(cfg, dry_run=False)
        self.assertIn("FundingRateArbStrategy", str(ctx.exception))
        self.assertIn("research/backtest-only", str(ctx.exception))

    def test_funding_arb_dry_run_passes(self):
        cfg = LiveConfig.from_dict({
            "strategy": {"class": "FundingRateArbStrategy"},
        })
        self.assertIsNone(_validate_live_strategy_safety(cfg, dry_run=True))


class TestSafeStrategyLivePasses(unittest.TestCase):
    def test_emacross_live_passes(self):
        cfg = LiveConfig.from_dict({
            "strategy": {"class": "EMACrossMACDTrend"},
        })
        self.assertIsNone(_validate_live_strategy_safety(cfg, dry_run=False))

    def test_donchian_live_passes(self):
        cfg = LiveConfig.from_dict({
            "strategy": {"class": "DonchianATRVolTarget"},
        })
        self.assertIsNone(_validate_live_strategy_safety(cfg, dry_run=False))

    def test_rsi_threshold_live_passes(self):
        # The default LiveConfig strategy_class — must not trip the guard.
        cfg = LiveConfig.from_dict({})
        self.assertIsNone(_validate_live_strategy_safety(cfg, dry_run=False))


if __name__ == "__main__":
    unittest.main()
