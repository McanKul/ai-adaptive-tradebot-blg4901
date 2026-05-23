"""
tests/test_export_trades_flag.py
================================
Verify the ``--export-trades`` plumbing closes the gap between the
backtest engine and the divergence harness.

Covers:
* ``app.py backtest --export-trades`` is exposed in the CLI parser.
* ``BacktestService.export_trades`` writes a JSON payload that the
  divergence harness can consume round-trip.
"""
from __future__ import annotations
import json
import os
import subprocess
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

APP_PY = os.path.join(os.path.dirname(__file__), "..", "app.py")


# ---------------------------------------------------------------------------
# CLI flag presence
# ---------------------------------------------------------------------------

class TestExportTradesCli(unittest.TestCase):

    def test_flag_listed_in_help(self):
        result = subprocess.run(
            [sys.executable, APP_PY, "backtest", "--help"],
            capture_output=True, text=True, timeout=30,
        )
        self.assertEqual(result.returncode, 0)
        self.assertIn("--export-trades", result.stdout)


# ---------------------------------------------------------------------------
# Round-trip via divergence harness loader
# ---------------------------------------------------------------------------

class TestExportTradesRoundTrip(unittest.TestCase):

    def _make_result(self):
        from Interfaces.metrics_interface import BacktestResult
        return BacktestResult(
            strategy_name="EMACrossMACDTrend",
            params={"fast_ema_period": 12, "slow_ema_period": 26},
            initial_capital=10_000.0,
            final_equity=10_350.0,
            total_return=350.0,
            total_return_pct=3.5,
            max_drawdown=0.04,
            sharpe_ratio=1.3,
            total_trades=2,
            win_rate=0.5,
            metadata={},
            trades=[
                {"symbol": "BTCUSDT", "entry_side": "LONG",
                 "entry_price": 100.0, "exit_price": 110.0,
                 "quantity": 1.0,
                 "entry_timestamp_ns": 1_000_000_000_000_000_000,
                 "exit_timestamp_ns": 1_000_000_001_000_000_000,
                 "pnl": 10.0},
                {"symbol": "BTCUSDT", "entry_side": "SHORT",
                 "entry_price": 110.0, "exit_price": 105.0,
                 "quantity": 1.0,
                 "entry_timestamp_ns": 1_000_000_002_000_000_000,
                 "exit_timestamp_ns": 1_000_000_003_000_000_000,
                 "pnl": 5.0},
            ],
        )

    def test_export_writes_expected_shape(self):
        from core.services.backtest_service import BacktestService
        result = self._make_result()
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "trades.json")
            BacktestService.export_trades(
                result, path, strategy_name="EMACrossMACDTrend",
                symbol="BTCUSDT",
            )
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
        self.assertEqual(data["strategy_name"], "EMACrossMACDTrend")
        self.assertEqual(data["symbol"], "BTCUSDT")
        self.assertIn("trades", data)
        self.assertEqual(len(data["trades"]), 2)
        self.assertIn("summary", data)
        self.assertAlmostEqual(data["summary"]["total_return_pct"], 3.5)

    def test_divergence_loader_consumes_export(self):
        from core.services.backtest_service import BacktestService
        from tools.compare_backtest_live import load_backtest_trades

        result = self._make_result()
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "trades.json")
            BacktestService.export_trades(
                result, path, strategy_name="X", symbol="BTCUSDT",
            )
            canonical = load_backtest_trades(path)
        self.assertEqual(len(canonical), 2)
        self.assertEqual(canonical[0].symbol, "BTCUSDT")
        self.assertEqual(canonical[0].side, "LONG")
        self.assertEqual(canonical[1].side, "SHORT")


if __name__ == "__main__":
    unittest.main()
