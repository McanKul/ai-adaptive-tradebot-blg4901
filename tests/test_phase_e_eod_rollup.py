"""
tests/test_phase_e_eod_rollup.py
================================
Phase E2 + E3 — notifier coverage and end-of-day rollup.

* ``TelegramNotifier`` exposes the new safety-event coroutines and
  short-circuits gracefully when the env vars are missing.
* ``tools/eod_rollup.aggregate`` produces the summary dict from a
  synthetic LiveMetrics CSV correctly.
* ``write_eod`` emits a single-row CSV with the documented columns.
"""
from __future__ import annotations
import asyncio
import csv
import math
import os
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from live.notifier import TelegramNotifier
from tools.eod_rollup import aggregate, write_eod, _read_rows


# ---------------------------------------------------------------------------
# Notifier coverage
# ---------------------------------------------------------------------------

class TestNotifierEvents(unittest.IsolatedAsyncioTestCase):

    async def test_new_methods_are_callable_and_graceful(self):
        # Without TELEGRAM_BOT_TOKEN/CHAT_ID the notifier is disabled
        # and every coroutine should return False without raising.
        os.environ.pop("TELEGRAM_BOT_TOKEN", None)
        os.environ.pop("TELEGRAM_CHAT_ID", None)
        n = TelegramNotifier()

        for coro in (
            n.notify_drift("BTCUSDT", "ghost_local", 0.5),
            n.notify_missing_stop("BTCUSDT", "trend"),
            n.notify_rejection_storm("rate_limit", 5),
            n.notify_daily_loss_breach(-30.0, 30.0),
            n.notify_heartbeat_lost(45.0),
            n.notify_blocked("spread"),
        ):
            res = await coro
            self.assertFalse(res, "disabled notifier should return False")


# ---------------------------------------------------------------------------
# EOD rollup aggregation
# ---------------------------------------------------------------------------

def _write_trade_csv(rows):
    """Emit a CSV that mirrors the LiveMetrics schema."""
    f = tempfile.NamedTemporaryFile(
        mode="w", suffix=".csv", delete=False, encoding="utf-8", newline="",
    )
    fieldnames = [
        "timestamp", "symbol", "side", "entry_price", "exit_price",
        "qty", "pnl_usd", "pnl_pct", "bars_held", "hold_seconds",
        "exit_type", "strategy",
        "intended_price", "fill_price", "slippage_bps",
        "entry_fee_usd", "exit_fee_usd", "funding_usd",
        "pnl_gross_usd", "pnl_net_usd",
    ]
    w = csv.DictWriter(f, fieldnames=fieldnames)
    w.writeheader()
    for r in rows:
        full = {k: "" for k in fieldnames}
        full.update(r)
        w.writerow(full)
    f.close()
    return f.name


class TestEodAggregate(unittest.TestCase):

    def test_empty_rows_emits_zero_summary(self):
        s = aggregate([])
        self.assertEqual(s["total_trades"], 0)
        self.assertEqual(s["profit_factor"], 0.0)
        self.assertEqual(s["per_symbol_pnl"], {})

    def test_summary_math(self):
        path = _write_trade_csv([
            {"timestamp": "2026-05-02 12:00:00", "symbol": "BTCUSDT",
             "side": "BUY", "pnl_usd": "10.0", "pnl_gross_usd": "10.0",
             "pnl_net_usd": "9.5", "entry_fee_usd": "0.25", "exit_fee_usd": "0.25",
             "funding_usd": "0.0", "slippage_bps": "3.0", "hold_seconds": "60"},
            {"timestamp": "2026-05-02 12:30:00", "symbol": "BTCUSDT",
             "side": "BUY", "pnl_usd": "-5.0", "pnl_gross_usd": "-5.0",
             "pnl_net_usd": "-5.4", "entry_fee_usd": "0.20", "exit_fee_usd": "0.20",
             "funding_usd": "0.0", "slippage_bps": "-7.0", "hold_seconds": "30"},
            {"timestamp": "2026-05-02 13:00:00", "symbol": "ETHUSDT",
             "side": "SELL", "pnl_usd": "8.0", "pnl_gross_usd": "8.0",
             "pnl_net_usd": "7.6", "entry_fee_usd": "0.20", "exit_fee_usd": "0.20",
             "funding_usd": "0.0", "slippage_bps": "2.0", "hold_seconds": "120"},
        ])
        try:
            rows = _read_rows(path, "2026-05-02")
            s = aggregate(rows)
            self.assertEqual(s["total_trades"], 3)
            self.assertEqual(s["winning_trades"], 2)
            self.assertEqual(s["losing_trades"], 1)
            self.assertAlmostEqual(s["win_rate"], 2 / 3, places=4)
            # PF = wins(18) / losses(5) = 3.6
            self.assertAlmostEqual(s["profit_factor"], 18.0 / 5.0, places=4)
            self.assertAlmostEqual(s["total_pnl_gross"], 13.0, places=4)
            self.assertAlmostEqual(s["total_pnl_net"], 11.7, places=4)
            self.assertAlmostEqual(s["total_fees"], 1.30, places=4)
            self.assertAlmostEqual(s["avg_slippage_bps"], 4.0, places=4)
            self.assertEqual(s["consecutive_losses_max"], 1)
            self.assertAlmostEqual(s["longest_hold_seconds"], 120.0)
            self.assertIn("BTCUSDT", s["per_symbol_pnl"])
            self.assertIn("ETHUSDT", s["per_symbol_pnl"])
        finally:
            os.unlink(path)

    def test_filter_by_date(self):
        path = _write_trade_csv([
            {"timestamp": "2026-05-01 10:00:00", "symbol": "BTC", "pnl_usd": "5.0"},
            {"timestamp": "2026-05-02 10:00:00", "symbol": "BTC", "pnl_usd": "-3.0"},
        ])
        try:
            rows = _read_rows(path, "2026-05-02")
            self.assertEqual(len(rows), 1)
        finally:
            os.unlink(path)


class TestEodWriteCsv(unittest.TestCase):

    def test_round_trip(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = os.path.join(tmp, "eod_2026-05-02_canary.csv")
            summary = aggregate([
                {"timestamp": "2026-05-02 12:00:00", "symbol": "BTCUSDT",
                 "pnl_usd": "10.0", "pnl_gross_usd": "10.0",
                 "pnl_net_usd": "9.5", "entry_fee_usd": "0.25",
                 "exit_fee_usd": "0.25", "funding_usd": "0.0",
                 "slippage_bps": "3.0", "hold_seconds": "60"},
            ])
            write_eod(out, "2026-05-02", "canary", summary)
            with open(out, "r", encoding="utf-8", newline="") as f:
                rows = list(csv.DictReader(f))
            self.assertEqual(len(rows), 1)
            self.assertEqual(rows[0]["date"], "2026-05-02")
            self.assertEqual(rows[0]["run_id"], "canary")
            self.assertEqual(int(rows[0]["total_trades"]), 1)


if __name__ == "__main__":
    unittest.main()
