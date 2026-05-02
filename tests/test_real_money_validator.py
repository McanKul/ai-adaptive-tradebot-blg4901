"""
tests/test_real_money_validator.py
==================================
``ConfigValidator(real_money=True)`` is the canary contract — it
catches every config-level failure mode the canary checklist warns
about.  These tests document the contract; if any rule is added,
removed, or relaxed the test must move with it.
"""
from __future__ import annotations
import os
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from core.config_validator import ConfigValidator


def _write_yaml(content: str) -> str:
    f = tempfile.NamedTemporaryFile(
        mode="w", suffix=".yaml", delete=False, encoding="utf-8",
    )
    f.write(content)
    f.close()
    return f.name


_CANARY_CLEAN = """\
strategy:
  class: EMACrossMACDTrend
  params:
    fast_ema_period: 12
    slow_ema_period: 26
    macd_signal: 9
    allow_reversal: false
symbols: [BTCUSDT, ETHUSDT, SOLUSDT, AVAXUSDT, LINKUSDT]
min_24h_volume_usd: 100000000
timeframe: 15m
sizing: {mode: margin_usd, margin_usd: 50, leverage: 3, leverage_mode: margin}
risk: {max_concurrent_positions: 1, max_daily_loss: 30, max_daily_loss_pct: 0.02}
global_risk:
  max_concurrent_positions: 1
  max_daily_loss: 30
  max_daily_loss_pct: 0.02
  max_account_drawdown_pct: 0.05
liquidation_guard: {enabled: true, buffer_pct: 0.30}
reconciliation: {enabled: true, interval_seconds: 30, action: halt}
news: {enabled: false}
testnet: false
"""


class TestRealMoneyValidator(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        # The validator delegates to StrategyFactory; without the
        # default registrations every test would trip on "cannot
        # resolve strategy".  Bootstrap once.
        from core.bootstrap import register_defaults
        register_defaults()

    def test_clean_canary_passes(self):
        path = _write_yaml(_CANARY_CLEAN)
        try:
            errors = ConfigValidator().validate(path, real_money=True)
            self.assertEqual(errors, [], msg=f"unexpected: {errors}")
        finally:
            os.unlink(path)

    def test_leverage_above_cap_fails(self):
        bad = _CANARY_CLEAN.replace("leverage: 3", "leverage: 15")
        path = _write_yaml(bad)
        try:
            errors = ConfigValidator().validate(path, real_money=True)
            self.assertTrue(any("leverage" in e and "15" in e for e in errors))
        finally:
            os.unlink(path)

    def test_volume_gate_below_minimum_fails(self):
        bad = _CANARY_CLEAN.replace(
            "min_24h_volume_usd: 100000000",
            "min_24h_volume_usd: 0",
        )
        path = _write_yaml(bad)
        try:
            errors = ConfigValidator().validate(path, real_money=True)
            self.assertTrue(any("min_24h_volume_usd" in e for e in errors))
        finally:
            os.unlink(path)

    def test_deprecated_ftm_symbol_fails(self):
        bad = _CANARY_CLEAN.replace(
            "[BTCUSDT, ETHUSDT, SOLUSDT, AVAXUSDT, LINKUSDT]",
            "[BTCUSDT, FTMUSDT]",
        )
        path = _write_yaml(bad)
        try:
            errors = ConfigValidator().validate(path, real_money=True)
            self.assertTrue(any("FTMUSDT" in e for e in errors))
            self.assertTrue(any("renamed" in e.lower() for e in errors))
        finally:
            os.unlink(path)

    def test_deprecated_matic_symbol_fails(self):
        bad = _CANARY_CLEAN.replace(
            "[BTCUSDT, ETHUSDT, SOLUSDT, AVAXUSDT, LINKUSDT]",
            "[BTCUSDT, MATICUSDT]",
        )
        path = _write_yaml(bad)
        try:
            errors = ConfigValidator().validate(path, real_money=True)
            self.assertTrue(any("MATICUSDT" in e for e in errors))
        finally:
            os.unlink(path)

    def test_too_many_symbols_warns(self):
        many = ",".join(f"COIN{i}USDT" for i in range(10))
        bad = _CANARY_CLEAN.replace(
            "[BTCUSDT, ETHUSDT, SOLUSDT, AVAXUSDT, LINKUSDT]",
            f"[{many}]",
        )
        path = _write_yaml(bad)
        try:
            errors = ConfigValidator().validate(path, real_money=True)
            self.assertTrue(any("symbol count" in e.lower() for e in errors))
        finally:
            os.unlink(path)

    def test_allow_reversal_true_blocks(self):
        bad = _CANARY_CLEAN.replace(
            "allow_reversal: false", "allow_reversal: true",
        )
        path = _write_yaml(bad)
        try:
            errors = ConfigValidator().validate(path, real_money=True)
            self.assertTrue(any("allow_reversal" in e for e in errors))
        finally:
            os.unlink(path)

    def test_daily_loss_disabled_blocks(self):
        bad = _CANARY_CLEAN.replace(
            "max_daily_loss: 30\n  max_daily_loss_pct: 0.02",
            "max_daily_loss: 0\n  max_daily_loss_pct: 0",
        )
        # Replace inside global_risk only — risk: section also has these keys
        # but the validator reads global_risk for the daily-loss check.
        path = _write_yaml(bad)
        try:
            errors = ConfigValidator().validate(path, real_money=True)
            self.assertTrue(any("daily-loss" in e.lower() for e in errors))
        finally:
            os.unlink(path)

    def test_liquidation_guard_off_blocks(self):
        bad = _CANARY_CLEAN.replace(
            "liquidation_guard: {enabled: true, buffer_pct: 0.30}",
            "liquidation_guard: {enabled: false}",
        )
        path = _write_yaml(bad)
        try:
            errors = ConfigValidator().validate(path, real_money=True)
            self.assertTrue(any("liquidation_guard" in e for e in errors))
        finally:
            os.unlink(path)

    def test_reconciliation_off_blocks(self):
        bad = _CANARY_CLEAN.replace(
            "reconciliation: {enabled: true, interval_seconds: 30, action: halt}",
            "reconciliation: {enabled: false}",
        )
        path = _write_yaml(bad)
        try:
            errors = ConfigValidator().validate(path, real_money=True)
            self.assertTrue(any("reconciliation" in e for e in errors))
        finally:
            os.unlink(path)

    def test_basic_mode_skips_real_money_checks(self):
        # Same bad config — basic mode should NOT flag the real-money rules
        bad = _CANARY_CLEAN.replace("leverage: 3", "leverage: 15")
        path = _write_yaml(bad)
        try:
            errors = ConfigValidator().validate(path, real_money=False)
            # No "[real-money]" flagged failures in basic mode
            self.assertFalse(any("[real-money]" in e for e in errors))
        finally:
            os.unlink(path)


if __name__ == "__main__":
    unittest.main()
