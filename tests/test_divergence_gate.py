"""
tests/test_divergence_gate.py
=============================
Strict-mode gate for ``tools/compare_backtest_live.py``.

Exit-code-driven gate is the difference between an "informational
tool" and an actual promotion gate.  Without these checks the
backtest-live divergence harness can be ignored by accident; with
them, the calling shell pipeline can refuse to flip the canary to
live trading.
"""
from __future__ import annotations
import os
import sys
import unittest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from tools.compare_backtest_live import (
    DivergenceReport,
    GateResult,
    evaluate_gate,
)


def _report(
    *,
    matched=20,
    backtest_count=20,
    live_count=20,
    only_backtest=0,
    only_live=0,
    side_mismatch=0,
    pnl_total_bt=0.0,
    pnl_total_lv=0.0,
    pnl_diff_total=0.0,
    pnl_diff_avg=0.0,
    pnl_diff_max_abs=0.0,
):
    return DivergenceReport(
        backtest_count=backtest_count,
        live_count=live_count,
        matched_count=matched,
        only_backtest_count=only_backtest,
        only_live_count=only_live,
        side_mismatch_count=side_mismatch,
        pnl_total_backtest=pnl_total_bt,
        pnl_total_live=pnl_total_lv,
        pnl_diff_total=pnl_diff_total,
        pnl_diff_avg_matched=pnl_diff_avg,
        pnl_diff_max_abs=pnl_diff_max_abs,
        exit_hist_backtest={},
        exit_hist_live={},
        by_symbol={},
    )


class TestEvaluateGate(unittest.TestCase):

    def test_no_thresholds_means_pass(self):
        r = _report()
        out = evaluate_gate(r)
        self.assertTrue(out.passed)
        self.assertEqual(out.failures, [])

    def test_match_rate_below_threshold_fails(self):
        # match_rate = matched / max(bt, live) = 5 / 20 = 0.25
        r = _report(matched=5, backtest_count=20, live_count=20,
                    only_backtest=15, only_live=15)
        out = evaluate_gate(r, require_match_rate=0.80)
        self.assertFalse(out.passed)
        self.assertTrue(any("match_rate" in f for f in out.failures))

    def test_side_mismatch_blocks_promotion(self):
        r = _report(side_mismatch=1)
        out = evaluate_gate(r, max_side_mismatch=0)
        self.assertFalse(out.passed)
        self.assertTrue(any("side_mismatch" in f for f in out.failures))

    def test_pnl_drift_blocks(self):
        r = _report(pnl_diff_avg=2.5)
        out = evaluate_gate(r, max_pnl_diff_avg=1.0)
        self.assertFalse(out.passed)

    def test_pnl_max_abs_outlier_blocks(self):
        r = _report(pnl_diff_max_abs=15.0)
        out = evaluate_gate(r, max_pnl_diff_max_abs=5.0)
        self.assertFalse(out.passed)

    def test_min_matched_guards_against_empty_window(self):
        # 3 matched trades is statistically meaningless; refuse.
        r = _report(matched=3, backtest_count=3, live_count=3)
        out = evaluate_gate(r, require_min_matched=10)
        self.assertFalse(out.passed)
        self.assertTrue(any("matched_count" in f for f in out.failures))

    def test_all_thresholds_combined_pass(self):
        r = _report(
            matched=50, backtest_count=55, live_count=52,
            side_mismatch=0, pnl_diff_avg=0.1, pnl_diff_max_abs=2.0,
        )
        out = evaluate_gate(
            r, require_match_rate=0.80, max_side_mismatch=0,
            max_pnl_diff_avg=1.0, max_pnl_diff_max_abs=5.0,
            require_min_matched=10,
        )
        self.assertTrue(out.passed, msg=f"unexpected failures: {out.failures}")

    def test_failure_messages_human_readable(self):
        r = _report(matched=2, backtest_count=20, live_count=20,
                    only_backtest=18, side_mismatch=3,
                    pnl_diff_avg=10.0)
        out = evaluate_gate(
            r, require_match_rate=0.80, max_side_mismatch=0,
            max_pnl_diff_avg=1.0, require_min_matched=10,
        )
        self.assertFalse(out.passed)
        # Each independent threshold should appear in the failure list
        self.assertEqual(len(out.failures), 4)


if __name__ == "__main__":
    unittest.main()
