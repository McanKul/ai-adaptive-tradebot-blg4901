"""
tests/test_phase_b_fee_accounting.py
====================================
Phase B4 — fee / funding accounting in the live trade CSV.

Covers:
* ``Position`` carries ``entry_fee_usd`` / ``exit_fee_usd`` /
  ``funding_usd`` initialised to 0.0.
* ``LiveMetrics.record`` writes every Phase-B4 column and computes
  ``pnl_gross_usd`` and ``pnl_net_usd`` correctly.
* Running stats aggregate fees / funding / slippage so the
  end-of-session summary is net-of-cost.
"""
from __future__ import annotations
import csv
import os
import sys
import tempfile
import time
import unittest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from live.live_metrics import LiveMetrics, _CSV_HEADER
from live.position_manager import Position


def _build_pos(*, entry=100.0, exit_=110.0, qty=1.0, side="BUY",
               entry_fee=0.0, exit_fee=0.0, funding=0.0,
               slippage_bps=None, fill_price=None,
               intended_price=None):
    pos = Position(
        symbol="BTCUSDT", side=side, qty=qty,
        entry_price=entry, opened_ts=time.time(),
        strategy="t", timeframe="1m",
    )
    pos.exit = exit_
    pos.exit_ts = time.time() + 60
    pos.exit_type = "TP"
    pos.entry_fee_usd = entry_fee
    pos.exit_fee_usd = exit_fee
    pos.funding_usd = funding
    pos.slippage_bps = slippage_bps
    pos.fill_price = fill_price
    if intended_price is not None:
        pos.intended_price = intended_price
    return pos


# ---------------------------------------------------------------------------
# Position defaults
# ---------------------------------------------------------------------------

class TestPositionFeeFields(unittest.TestCase):

    def test_initial_fee_state_zero(self):
        pos = Position(
            symbol="BTCUSDT", side="BUY", qty=1.0,
            entry_price=100.0, opened_ts=time.time(),
            strategy="t", timeframe="1m",
        )
        self.assertEqual(pos.entry_fee_usd, 0.0)
        self.assertEqual(pos.exit_fee_usd, 0.0)
        self.assertEqual(pos.funding_usd, 0.0)


# ---------------------------------------------------------------------------
# CSV header coverage
# ---------------------------------------------------------------------------

class TestCsvHeader(unittest.TestCase):

    def test_b4_columns_present(self):
        for col in (
            "entry_fee_usd", "exit_fee_usd", "funding_usd",
            "pnl_gross_usd", "pnl_net_usd",
        ):
            self.assertIn(col, _CSV_HEADER)


# ---------------------------------------------------------------------------
# record() math
# ---------------------------------------------------------------------------

class TestRecordNetPnl(unittest.TestCase):

    def _write_one(self, pos):
        tmp = tempfile.mkdtemp()
        path = os.path.join(tmp, "trades.csv")
        m = LiveMetrics(csv_path=path, start_equity=1_000.0)
        m.record(pos)
        with open(path, "r", newline="", encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
        return rows[0], m

    def test_zero_costs_gross_equals_net(self):
        pos = _build_pos(entry=100.0, exit_=110.0, qty=1.0)
        row, _ = self._write_one(pos)
        self.assertAlmostEqual(float(row["pnl_gross_usd"]), 10.0)
        self.assertAlmostEqual(float(row["pnl_net_usd"]), 10.0)
        self.assertAlmostEqual(float(row["entry_fee_usd"]), 0.0)
        self.assertAlmostEqual(float(row["exit_fee_usd"]), 0.0)
        self.assertAlmostEqual(float(row["funding_usd"]), 0.0)

    def test_fees_subtract_from_gross(self):
        pos = _build_pos(entry=100.0, exit_=110.0, qty=1.0,
                         entry_fee=0.04, exit_fee=0.05, funding=0.01)
        row, _ = self._write_one(pos)
        # gross=10, fees=0.09, funding=0.01 → net=9.90
        self.assertAlmostEqual(float(row["pnl_gross_usd"]), 10.0)
        self.assertAlmostEqual(float(row["pnl_net_usd"]), 9.90, places=4)

    def test_running_stats_aggregate(self):
        pos1 = _build_pos(entry=100.0, exit_=110.0, qty=1.0,
                          entry_fee=0.05, exit_fee=0.05, funding=0.10,
                          slippage_bps=3.0, fill_price=100.03)
        pos2 = _build_pos(entry=200.0, exit_=190.0, qty=1.0, side="SELL",
                          entry_fee=0.10, exit_fee=0.08, funding=0.0,
                          slippage_bps=-2.0, fill_price=199.96)
        # SELL math: pnl = (entry - exit) * qty = 10
        with tempfile.TemporaryDirectory() as tmp:
            m = LiveMetrics(csv_path=os.path.join(tmp, "x.csv"))
            m.record(pos1)
            m.record(pos2)
        s = m._stats
        self.assertEqual(s.total_trades, 2)
        # Fees: 0.10 + 0.18 = 0.28
        self.assertAlmostEqual(s.total_fees, 0.28, places=4)
        self.assertAlmostEqual(s.total_funding, 0.10, places=4)
        # Both gross 10 each → 20 gross; fees+funding=0.38 → net=19.62
        self.assertAlmostEqual(s.total_pnl_net, 19.62, places=4)
        # Slippage: |3| + |2| = 5 over 2 samples → avg 2.5
        self.assertEqual(s.slippage_count, 2)
        self.assertAlmostEqual(s.total_slippage_bps_abs, 5.0, places=4)


if __name__ == "__main__":
    unittest.main()
