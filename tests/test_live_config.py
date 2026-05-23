"""
tests/test_live_config.py
==========================
Pin LiveConfig.from_dict's `testnet` parsing precedence.

Real-money risk: a silent bug here can route an order to mainnet when
the YAML clearly intended testnet (or vice versa).  Both nested
`api.testnet` and a top-level `testnet` must be honoured; nested wins
when both are present.
"""
import os
import sys
import unittest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from live.live_config import LiveConfig


class TestLiveConfigTestnetParsing(unittest.TestCase):
    def test_testnet_top_level_true(self):
        cfg = LiveConfig.from_dict({"testnet": True})
        self.assertIs(cfg.testnet, True)

    def test_testnet_nested_api_true(self):
        cfg = LiveConfig.from_dict({"api": {"testnet": True}})
        self.assertIs(cfg.testnet, True)

    def test_testnet_nested_overrides_top_level(self):
        cfg = LiveConfig.from_dict({
            "testnet": False,
            "api": {"testnet": True},
        })
        self.assertIs(cfg.testnet, True)

    def test_testnet_top_level_false_with_no_api(self):
        cfg = LiveConfig.from_dict({"testnet": False})
        self.assertIs(cfg.testnet, False)

    def test_testnet_default_when_neither_present(self):
        cfg = LiveConfig.from_dict({})
        self.assertIs(cfg.testnet, False)

    def test_testnet_string_true_coerced_to_bool(self):
        # YAML can yield non-bool truthy values; we explicitly coerce.
        cfg = LiveConfig.from_dict({"testnet": "yes"})
        self.assertIs(cfg.testnet, True)
        cfg = LiveConfig.from_dict({"testnet": ""})
        self.assertIs(cfg.testnet, False)


class TestReconciliationConfigNumericCoercion(unittest.TestCase):
    """PyYAML 1.1 schema turns `1e-6` (unquoted) into a *string*; the
    live engine's reconciliation log uses `%g` which then raises
    `TypeError: must be real number, not str` at runtime.  Coerce
    numeric fields at dataclass __post_init__ so the runtime path is
    safe regardless of how YAML parsed the literal.
    """

    def test_string_qty_tolerance_coerced_to_float(self):
        cfg = LiveConfig.from_dict({"reconciliation": {"qty_tolerance": "1e-6"}})
        self.assertIsInstance(cfg.reconciliation.qty_tolerance, float)
        self.assertAlmostEqual(cfg.reconciliation.qty_tolerance, 1e-6)

    def test_string_interval_seconds_coerced_to_float(self):
        cfg = LiveConfig.from_dict({"reconciliation": {"interval_seconds": "30"}})
        self.assertIsInstance(cfg.reconciliation.interval_seconds, float)
        self.assertEqual(cfg.reconciliation.interval_seconds, 30.0)

    def test_log_format_does_not_raise_after_coercion(self):
        # The exact log statement that was crashing in production.
        cfg = LiveConfig.from_dict({"reconciliation": {"qty_tolerance": "1e-6"}})
        msg = "tol=%g action=%s interval=%.1fs" % (
            cfg.reconciliation.qty_tolerance,
            cfg.reconciliation.action,
            cfg.reconciliation.interval_seconds,
        )
        self.assertIn("1e-06", msg)


if __name__ == "__main__":
    unittest.main()
