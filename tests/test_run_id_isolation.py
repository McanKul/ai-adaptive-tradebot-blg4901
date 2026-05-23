"""
tests/test_run_id_isolation.py
==============================
Verify that ``LiveConfig.run_id`` produces isolated log/state paths so two
parallel dry-runs (sentiment ON vs OFF demo) do not stomp on each other,
and that ``tools/compare_dry_runs`` aggregates two CSVs correctly.
"""
from __future__ import annotations
import csv
import os
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from live.live_config import LiveConfig
from tools.compare_dry_runs import _read_run


class TestRunIdNamespacing(unittest.TestCase):

    def test_default_run_id_falls_back_to_name(self):
        cfg = LiveConfig(name="MyStrat")
        self.assertEqual(cfg.effective_run_id(), "MyStrat")

    def test_explicit_run_id_overrides_name(self):
        cfg = LiveConfig(name="MyStrat", run_id="sentiment_on")
        self.assertEqual(cfg.effective_run_id(), "sentiment_on")

    def test_paths_are_namespaced_by_run_id(self):
        a = LiveConfig(run_id="sent_off")
        b = LiveConfig(run_id="sent_on")
        # CSV trade log
        self.assertNotEqual(a.trade_log_path(), b.trade_log_path())
        self.assertIn("sent_off", a.trade_log_path())
        self.assertIn("sent_on", b.trade_log_path())
        # Position store
        self.assertNotEqual(a.positions_state_path(), b.positions_state_path())
        # Risk state
        self.assertNotEqual(a.risk_state_path(), b.risk_state_path())

    def test_explicit_persist_path_in_global_risk_wins(self):
        # If user customised the path, run_id should NOT override
        a = LiveConfig(run_id="sent_off")
        a.global_risk.persist_path = "logs/custom.json"
        self.assertEqual(a.risk_state_path(), "logs/custom.json")


class TestCompareDryRuns(unittest.TestCase):

    def _write_trades(self, path, rows):
        with open(path, "w", encoding="utf-8", newline="") as f:
            w = csv.writer(f)
            w.writerow([
                "timestamp", "symbol", "side", "entry_price", "exit_price",
                "qty", "pnl_usd", "pnl_pct", "bars_held", "hold_seconds",
                "exit_type", "strategy",
            ])
            for r in rows:
                w.writerow(r)

    def test_compare_aggregates_correctly(self):
        with tempfile.TemporaryDirectory() as tmp:
            a_path = os.path.join(tmp, "a.csv")
            b_path = os.path.join(tmp, "b.csv")
            # Baseline: 3 trades, +1, -2, +5 → net +4, 2 wins
            self._write_trades(a_path, [
                ("t", "BTC", "long", 1, 2, 1, 1.0, 1.0, 1, 60, "TP", ""),
                ("t", "BTC", "long", 2, 1, 1, -2.0, -2.0, 1, 60, "SL", ""),
                ("t", "ETH", "long", 1, 6, 1, 5.0, 5.0, 1, 60, "TP", ""),
            ])
            # Variant: 1 trade, +10, 1 win
            self._write_trades(b_path, [
                ("t", "BTC", "long", 1, 11, 1, 10.0, 10.0, 1, 60, "TP", ""),
            ])
            a = _read_run(a_path, "off")
            b = _read_run(b_path, "on")
            self.assertEqual(a.n_trades, 3)
            self.assertEqual(a.n_wins, 2)
            self.assertEqual(a.n_losses, 1)
            self.assertAlmostEqual(a.pnl_total, 4.0)
            self.assertAlmostEqual(a.pnl_best, 5.0)
            self.assertAlmostEqual(a.pnl_worst, -2.0)
            self.assertEqual(b.n_trades, 1)
            self.assertAlmostEqual(b.pnl_total, 10.0)
            # Profit factor: wins=10, losses=0 → inf
            import math
            self.assertEqual(b.profit_factor, math.inf)

    def test_compare_handles_missing_files(self):
        s = _read_run("/does/not/exist.csv", "x")
        self.assertEqual(s.n_trades, 0)


if __name__ == "__main__":
    unittest.main()
