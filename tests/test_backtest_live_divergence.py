"""
tests/test_backtest_live_divergence.py
======================================
Tests for ``tools/compare_backtest_live.py`` — the backtest-vs-live
divergence harness.

Covers:
* Side normalization (LONG/SHORT vs BUY/SELL).
* Backtest JSON loader handles both array and {"trades": [...]} shapes.
* Live CSV loader recovers entry timestamps via ``hold_seconds``.
* Bipartite matching prefers the closest entry_ts within tolerance and
  doesn't double-consume live trades.
* Side mismatches show up in the report but stay in ``matched``.
* Time-window filter drops trades outside the requested range.
* Aggregate metrics (match_rate, pnl_diff, exit histograms) match
  hand-computed values.
* Per-symbol breakdown is correct.
"""
from __future__ import annotations
import csv
import json
import os
import sys
import tempfile
import unittest
from datetime import datetime, timezone

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from tools.compare_backtest_live import (
    CanonicalTrade,
    aggregate,
    filter_window,
    load_backtest_trades,
    load_live_trades,
    match_trades,
    _normalize_side,
    _parse_datetime,
)


# ---------------------------------------------------------------------------
# Side normalization
# ---------------------------------------------------------------------------

class TestNormalizeSide(unittest.TestCase):

    def test_long_aliases(self):
        for s in ("LONG", "long", "BUY", "buy", "+1"):
            self.assertEqual(_normalize_side(s), "LONG")

    def test_short_aliases(self):
        for s in ("SHORT", "SELL", "-1"):
            self.assertEqual(_normalize_side(s), "SHORT")

    def test_unknown_returned_uppercase(self):
        self.assertEqual(_normalize_side("foo"), "FOO")
        self.assertEqual(_normalize_side(""), "UNKNOWN")


# ---------------------------------------------------------------------------
# Loaders
# ---------------------------------------------------------------------------

class TestBacktestLoader(unittest.TestCase):

    def _write(self, payload):
        f = tempfile.NamedTemporaryFile(
            mode="w", suffix=".json", delete=False, encoding="utf-8",
        )
        json.dump(payload, f)
        f.close()
        return f.name

    def test_array_shape(self):
        path = self._write([
            {"symbol": "BTCUSDT", "entry_side": "LONG",
             "entry_price": 100.0, "exit_price": 110.0,
             "quantity": 1.0,
             "entry_timestamp_ns": 1_000_000_000_000_000_000,
             "exit_timestamp_ns": 1_000_000_001_000_000_000,
             "pnl": 10.0},
        ])
        try:
            trades = load_backtest_trades(path)
            self.assertEqual(len(trades), 1)
            self.assertEqual(trades[0].symbol, "BTCUSDT")
            self.assertEqual(trades[0].side, "LONG")
            self.assertAlmostEqual(trades[0].pnl, 10.0)
            self.assertAlmostEqual(trades[0].entry_ts, 1_000_000_000.0)
        finally:
            os.unlink(path)

    def test_object_with_trades_field(self):
        path = self._write({"strategy_name": "X", "trades": [
            {"symbol": "ETHUSDT", "entry_side": "SHORT",
             "entry_price": 50.0, "exit_price": 45.0, "quantity": 2.0,
             "entry_timestamp_ns": 1_000_000_000_000_000_000,
             "exit_timestamp_ns": 1_000_000_010_000_000_000,
             "pnl": 10.0},
        ]})
        try:
            trades = load_backtest_trades(path)
            self.assertEqual(len(trades), 1)
            self.assertEqual(trades[0].side, "SHORT")
        finally:
            os.unlink(path)

    def test_invalid_root_raises(self):
        path = self._write("just a string")
        try:
            with self.assertRaises(ValueError):
                load_backtest_trades(path)
        finally:
            os.unlink(path)


class TestLiveLoader(unittest.TestCase):

    def _write_csv(self, rows):
        f = tempfile.NamedTemporaryFile(
            mode="w", suffix=".csv", delete=False, newline="", encoding="utf-8",
        )
        w = csv.writer(f)
        w.writerow([
            "timestamp", "symbol", "side", "entry_price", "exit_price",
            "qty", "pnl_usd", "pnl_pct", "bars_held", "hold_seconds",
            "exit_type", "strategy",
        ])
        for r in rows:
            w.writerow(r)
        f.close()
        return f.name

    def test_recovers_entry_ts_from_hold_seconds(self):
        # exit at 2026-04-15 12:00:00 UTC, held 60s → entry 11:59:00
        path = self._write_csv([
            ("2026-04-15 12:00:00", "BTCUSDT", "BUY",
             "100", "110", "1.0", "10.0", "10.0", "1", "60", "TP", "test"),
        ])
        try:
            trades = load_live_trades(path)
            self.assertEqual(len(trades), 1)
            self.assertEqual(trades[0].side, "LONG")
            # exit_ts - entry_ts == 60.0
            self.assertAlmostEqual(trades[0].exit_ts - trades[0].entry_ts, 60.0)
            self.assertEqual(trades[0].exit_type, "TP")
        finally:
            os.unlink(path)

    def test_missing_file_returns_empty(self):
        self.assertEqual(load_live_trades("/does/not/exist.csv"), [])


# ---------------------------------------------------------------------------
# Time-window filter
# ---------------------------------------------------------------------------

class TestFilterWindow(unittest.TestCase):

    def _make(self, ts, sym="BTCUSDT"):
        return CanonicalTrade(
            source="live", symbol=sym, side="LONG",
            entry_ts=ts, exit_ts=ts + 1,
            entry_price=1, exit_price=1, quantity=1, pnl=0,
        )

    def test_window_drops_outside_range(self):
        trades = [self._make(100), self._make(200), self._make(300)]
        out = filter_window(trades, start_ts=150, end_ts=250)
        self.assertEqual([t.entry_ts for t in out], [200])

    def test_no_window_passthrough(self):
        trades = [self._make(100), self._make(200)]
        out = filter_window(trades, start_ts=None, end_ts=None)
        self.assertEqual(len(out), 2)


# ---------------------------------------------------------------------------
# Matching
# ---------------------------------------------------------------------------

def _trade(source, sym, side, entry, pnl=0, exit_ts=None, exit_type="TP"):
    return CanonicalTrade(
        source=source, symbol=sym, side=side,
        entry_ts=entry, exit_ts=exit_ts or entry + 60,
        entry_price=100, exit_price=100, quantity=1, pnl=pnl,
        exit_type=exit_type,
    )


class TestMatchTrades(unittest.TestCase):

    def test_exact_match(self):
        bt = [_trade("backtest", "BTCUSDT", "LONG", entry=1000)]
        lv = [_trade("live", "BTCUSDT", "LONG", entry=1000)]
        res = match_trades(bt, lv, tolerance_seconds=60)
        self.assertEqual(len(res.matched), 1)
        self.assertFalse(res.only_backtest)
        self.assertFalse(res.only_live)

    def test_within_tolerance_matches(self):
        bt = [_trade("backtest", "BTCUSDT", "LONG", entry=1000)]
        lv = [_trade("live", "BTCUSDT", "LONG", entry=1030)]  # 30s diff
        res = match_trades(bt, lv, tolerance_seconds=60)
        self.assertEqual(len(res.matched), 1)

    def test_outside_tolerance_unmatched(self):
        bt = [_trade("backtest", "BTCUSDT", "LONG", entry=1000)]
        lv = [_trade("live", "BTCUSDT", "LONG", entry=1500)]  # 500s diff
        res = match_trades(bt, lv, tolerance_seconds=60)
        self.assertEqual(res.only_backtest, bt)
        self.assertEqual(res.only_live, lv)

    def test_closest_chosen(self):
        bt = [_trade("backtest", "BTCUSDT", "LONG", entry=1000)]
        lv = [
            _trade("live", "BTCUSDT", "LONG", entry=1050, pnl=1.0),
            _trade("live", "BTCUSDT", "LONG", entry=1010, pnl=2.0),  # closer
        ]
        res = match_trades(bt, lv, tolerance_seconds=60)
        self.assertEqual(len(res.matched), 1)
        self.assertAlmostEqual(res.matched[0][1].pnl, 2.0)
        self.assertEqual(len(res.only_live), 1)

    def test_each_live_consumed_once(self):
        bt = [
            _trade("backtest", "BTCUSDT", "LONG", entry=1000),
            _trade("backtest", "BTCUSDT", "LONG", entry=1010),
        ]
        lv = [_trade("live", "BTCUSDT", "LONG", entry=1005)]
        res = match_trades(bt, lv, tolerance_seconds=60)
        self.assertEqual(len(res.matched), 1)
        self.assertEqual(len(res.only_backtest), 1)

    def test_different_symbols_dont_match(self):
        bt = [_trade("backtest", "BTCUSDT", "LONG", entry=1000)]
        lv = [_trade("live", "ETHUSDT", "LONG", entry=1000)]
        res = match_trades(bt, lv, tolerance_seconds=60)
        self.assertEqual(len(res.matched), 0)
        self.assertEqual(len(res.only_backtest), 1)
        self.assertEqual(len(res.only_live), 1)

    def test_side_mismatch_still_matches_but_flagged(self):
        bt = [_trade("backtest", "BTCUSDT", "LONG", entry=1000)]
        lv = [_trade("live", "BTCUSDT", "SHORT", entry=1000)]
        res = match_trades(bt, lv, tolerance_seconds=60)
        self.assertEqual(len(res.matched), 1)
        report = aggregate(res)
        self.assertEqual(report.side_mismatch_count, 1)


# ---------------------------------------------------------------------------
# Aggregate
# ---------------------------------------------------------------------------

class TestAggregate(unittest.TestCase):

    def test_pnl_diff_and_match_rate(self):
        bt = [
            _trade("backtest", "BTCUSDT", "LONG", entry=1000, pnl=10.0,
                   exit_type="TP"),
            _trade("backtest", "BTCUSDT", "LONG", entry=2000, pnl=-5.0,
                   exit_type="SL"),
        ]
        lv = [
            _trade("live", "BTCUSDT", "LONG", entry=1010, pnl=8.0,
                   exit_type="TP"),  # matches bt[0]
            _trade("live", "BTCUSDT", "LONG", entry=2010, pnl=-7.0,
                   exit_type="SL"),  # matches bt[1]
            _trade("live", "BTCUSDT", "LONG", entry=3000, pnl=15.0,
                   exit_type="TRAILING"),  # only-live
        ]
        res = match_trades(bt, lv, tolerance_seconds=60)
        report = aggregate(res, by_symbol=True)
        self.assertEqual(report.matched_count, 2)
        self.assertEqual(report.only_backtest_count, 0)
        self.assertEqual(report.only_live_count, 1)
        # max(bt_count=2, live_count=3) = 3 → match_rate 2/3
        self.assertAlmostEqual(report.match_rate, 2 / 3, places=4)
        # PnL diffs on matched: (8 - 10) + (-7 - -5) = -2 + -2 = -4
        self.assertAlmostEqual(report.pnl_diff_avg_matched, -2.0)
        self.assertEqual(report.exit_hist_backtest, {"TP": 1, "SL": 1})
        # Live exit_hist: matched live (TP, SL) + only-live (TRAILING)
        self.assertEqual(report.exit_hist_live, {"TP": 1, "SL": 1, "TRAILING": 1})
        # Per-symbol breakdown
        self.assertIn("BTCUSDT", report.by_symbol)
        self.assertEqual(report.by_symbol["BTCUSDT"]["matched"], 2)
        self.assertEqual(report.by_symbol["BTCUSDT"]["only_live"], 1)


# ---------------------------------------------------------------------------
# Date parsing
# ---------------------------------------------------------------------------

class TestParseDatetime(unittest.TestCase):

    def test_iso_format(self):
        ts = _parse_datetime("2026-04-15T12:00:00")
        ref = datetime(2026, 4, 15, 12, 0, 0, tzinfo=timezone.utc).timestamp()
        self.assertAlmostEqual(ts, ref)

    def test_space_format(self):
        ts = _parse_datetime("2026-04-15 12:00:00")
        ref = datetime(2026, 4, 15, 12, 0, 0, tzinfo=timezone.utc).timestamp()
        self.assertAlmostEqual(ts, ref)

    def test_garbage_returns_zero(self):
        self.assertEqual(_parse_datetime("not a date"), 0.0)


if __name__ == "__main__":
    unittest.main()
