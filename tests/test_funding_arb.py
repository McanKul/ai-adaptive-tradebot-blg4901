"""
tests/test_funding_arb.py
=========================
Tests for FundingRateArbStrategy + BarSyncBuffer.
"""
from __future__ import annotations
import csv
import os
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from Backtest.multi_symbol_sync import BarSyncBuffer, MultiBar
from Interfaces.market_data import Bar
from Interfaces.orders import OrderSide
from Strategy.arb.funding_arb_strategy import FundingRateArbStrategy


def make_bar(symbol, close, ts, hl_amp=0.001):
    return Bar(
        symbol=symbol, timeframe="1h",
        timestamp_ns=ts,
        open=close * (1 - hl_amp), high=close * (1 + hl_amp),
        low=close * (1 - hl_amp), close=close, volume=1000.0,
    )


class _FakeCtx:
    pass


# ---------------------------------------------------------------------------
# BarSyncBuffer
# ---------------------------------------------------------------------------

class TestBarSyncBuffer(unittest.TestCase):

    def test_emits_only_when_all_symbols_present(self):
        buf = BarSyncBuffer(["A", "B"], mode="strict")
        buf.update(make_bar("A", 100.0, 1_000))
        self.assertIsNone(buf.try_emit())
        buf.update(make_bar("B", 50.0, 1_000))
        mb = buf.try_emit()
        self.assertIsNotNone(mb)
        self.assertEqual(set(mb.bars.keys()), {"A", "B"})

    def test_strict_drops_misaligned(self):
        buf = BarSyncBuffer(["A", "B"], mode="strict", tolerance_ns=0)
        buf.update(make_bar("A", 100.0, 1_000))
        buf.update(make_bar("B", 50.0, 2_000))
        self.assertIsNone(buf.try_emit())
        # B catches up
        buf.update(make_bar("B", 51.0, 1_000))  # back-fill
        # Now they align at ts=1000
        # (Note: strict requires exact alignment)
        # If B's last is now ts=1_000 (we overwrote previous), max=min=1000
        mb = buf.try_emit()
        self.assertIsNotNone(mb)

    def test_strict_tolerance_allows_jitter(self):
        buf = BarSyncBuffer(["A", "B"], mode="strict", tolerance_ns=500)
        buf.update(make_bar("A", 100.0, 1_000))
        buf.update(make_bar("B", 50.0, 1_400))  # within tolerance
        mb = buf.try_emit()
        self.assertIsNotNone(mb)

    def test_no_double_emit_for_same_alignment(self):
        buf = BarSyncBuffer(["A", "B"], mode="strict")
        buf.update(make_bar("A", 100.0, 1_000))
        buf.update(make_bar("B", 50.0, 1_000))
        self.assertIsNotNone(buf.try_emit())
        # Calling again without new updates — no emit
        self.assertIsNone(buf.try_emit())
        self.assertEqual(buf.dropped, 1)

    def test_forward_fill_emits_with_staleness(self):
        buf = BarSyncBuffer(["A", "B"], mode="forward_fill")
        buf.update(make_bar("A", 100.0, 1_000))
        buf.update(make_bar("B", 50.0, 2_000))
        mb = buf.try_emit()
        self.assertIsNotNone(mb)
        self.assertEqual(mb.timestamp_ns, 2_000)
        # A's bar is stale by 1000 ns
        self.assertEqual(mb.staleness_ns["A"], 1_000)

    def test_invalid_mode_rejected(self):
        with self.assertRaises(ValueError):
            BarSyncBuffer(["A"], mode="bogus")


# ---------------------------------------------------------------------------
# FundingRateArbStrategy
# ---------------------------------------------------------------------------

class TestFundingArbEntry(unittest.TestCase):

    def test_no_action_below_threshold(self):
        strat = FundingRateArbStrategy(
            spot_symbol="BTC_SPOT", perp_symbol="BTCUSDT",
            funding_threshold_bps=10.0, funding_constant_bps=2.0,
        )
        ctx = _FakeCtx()
        ts = 1_000_000_000
        # Both legs aligned but funding too small
        strat.on_bar(make_bar("BTC_SPOT", 50_000.0, ts), ctx)
        decision = strat.on_bar(make_bar("BTCUSDT", 50_010.0, ts), ctx)
        self.assertEqual(decision.orders, [])
        self.assertIsNone(strat.position)

    def test_positive_funding_enters_long_spot_short_perp(self):
        strat = FundingRateArbStrategy(
            spot_symbol="BTC_SPOT", perp_symbol="BTCUSDT",
            funding_threshold_bps=5.0, funding_constant_bps=15.0,
            notional_per_leg_usd=1_000.0,
        )
        ctx = _FakeCtx()
        ts = 1_000_000_000
        strat.on_bar(make_bar("BTC_SPOT", 50_000.0, ts), ctx)
        decision = strat.on_bar(make_bar("BTCUSDT", 50_050.0, ts), ctx)

        # Should emit two orders, opposite sides
        self.assertEqual(len(decision.orders), 2)
        by_leg = {o.metadata["leg"]: o for o in decision.orders}
        self.assertEqual(by_leg["spot"].side, OrderSide.BUY)
        self.assertEqual(by_leg["perp"].side, OrderSide.SELL)
        self.assertAlmostEqual(by_leg["spot"].quantity, 1_000 / 50_000.0, places=8)
        self.assertAlmostEqual(by_leg["perp"].quantity, 1_000 / 50_050.0, places=8)
        self.assertIsNotNone(strat.position)
        self.assertEqual(strat.position.direction, 1)

    def test_negative_funding_enters_short_spot_long_perp(self):
        strat = FundingRateArbStrategy(
            spot_symbol="BTC_SPOT", perp_symbol="BTCUSDT",
            funding_threshold_bps=5.0, funding_constant_bps=-15.0,
        )
        ctx = _FakeCtx()
        ts = 1_000_000_000
        strat.on_bar(make_bar("BTC_SPOT", 50_000.0, ts), ctx)
        decision = strat.on_bar(make_bar("BTCUSDT", 49_950.0, ts), ctx)
        by_leg = {o.metadata["leg"]: o for o in decision.orders}
        self.assertEqual(by_leg["spot"].side, OrderSide.SELL)
        self.assertEqual(by_leg["perp"].side, OrderSide.BUY)


class TestFundingArbExit(unittest.TestCase):

    def _enter_position(self, strat, ts, spot=50_000.0, perp=50_050.0):
        ctx = _FakeCtx()
        strat.on_bar(make_bar("BTC_SPOT", spot, ts), ctx)
        strat.on_bar(make_bar("BTCUSDT", perp, ts), ctx)

    def test_exit_on_funding_flip(self):
        strat = FundingRateArbStrategy(
            spot_symbol="BTC_SPOT", perp_symbol="BTCUSDT",
            funding_threshold_bps=5.0, funding_constant_bps=15.0,
            delever_on_basis_pct=0.0,  # disable basis exit
        )
        ts = 1_000_000_000
        self._enter_position(strat, ts)
        self.assertIsNotNone(strat.position)

        # Flip funding to negative — should unwind
        strat.funding_constant_bps = -10.0
        ts2 = ts + 60 * 60 * 1_000_000_000  # +1h
        ctx = _FakeCtx()
        strat.on_bar(make_bar("BTC_SPOT", 50_100.0, ts2), ctx)
        decision = strat.on_bar(make_bar("BTCUSDT", 50_120.0, ts2), ctx)
        self.assertEqual(len(decision.orders), 2)
        self.assertTrue(all(o.reduce_only for o in decision.orders))
        self.assertIsNone(strat.position)
        self.assertEqual(decision.metadata.get("exit_reason"), "funding_flip")

    def test_exit_on_basis_converged(self):
        strat = FundingRateArbStrategy(
            spot_symbol="BTC_SPOT", perp_symbol="BTCUSDT",
            funding_threshold_bps=5.0, funding_constant_bps=15.0,
            delever_on_basis_pct=0.0005,
        )
        ts = 1_000_000_000
        # Enter with 0.1% basis (above threshold)
        self._enter_position(strat, ts, spot=50_000.0, perp=50_050.0)
        # Now basis is essentially zero (within delever_on_basis_pct)
        ts2 = ts + 60 * 60 * 1_000_000_000
        ctx = _FakeCtx()
        strat.on_bar(make_bar("BTC_SPOT", 50_000.0, ts2), ctx)
        decision = strat.on_bar(make_bar("BTCUSDT", 50_010.0, ts2), ctx)  # 0.02% basis
        self.assertEqual(decision.metadata.get("exit_reason"), "basis_converged")
        self.assertIsNone(strat.position)

    def test_exit_on_max_holding(self):
        strat = FundingRateArbStrategy(
            spot_symbol="BTC_SPOT", perp_symbol="BTCUSDT",
            funding_threshold_bps=5.0, funding_constant_bps=15.0,
            funding_interval_hours=1,
            max_holding_intervals=2,
            delever_on_basis_pct=0.0,
        )
        ts = 1_000_000_000
        self._enter_position(strat, ts)

        # 3 hours later — beyond max_holding=2
        ts2 = ts + 3 * 60 * 60 * 1_000_000_000
        ctx = _FakeCtx()
        strat.on_bar(make_bar("BTC_SPOT", 50_100.0, ts2), ctx)
        decision = strat.on_bar(make_bar("BTCUSDT", 50_150.0, ts2), ctx)
        self.assertEqual(decision.metadata.get("exit_reason"), "max_holding")


class TestFundingCSV(unittest.TestCase):

    def test_csv_funding_lookup(self):
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".csv", delete=False, newline="",
        ) as f:
            csv_path = f.name
            writer = csv.writer(f)
            writer.writerow(["timestamp", "funding_rate"])
            # ts in ms; rate as decimal (e.g. 0.0010 = 10 bps)
            writer.writerow([1_000, 0.0010])    # 10 bps
            writer.writerow([2_000, 0.0020])    # 20 bps
            writer.writerow([3_000, -0.0005])   # -5 bps

        try:
            strat = FundingRateArbStrategy(
                spot_symbol="BTC_SPOT", perp_symbol="BTCUSDT",
                funding_series_csv=csv_path,
                funding_timestamp_unit="ms",
            )
            # 1.5 sec → between row 1 (1000 ms) and row 2 (2000 ms) → 10 bps
            self.assertAlmostEqual(strat._funding_at(1_500 * 1_000_000), 10.0, places=4)
            # 2.5 sec → 20 bps
            self.assertAlmostEqual(strat._funding_at(2_500 * 1_000_000), 20.0, places=4)
            # 4 sec → -5 bps
            self.assertAlmostEqual(strat._funding_at(4_000 * 1_000_000), -5.0, places=4)
            # Before first row → constant fallback (0.0)
            self.assertAlmostEqual(strat._funding_at(500 * 1_000_000), 0.0, places=4)
        finally:
            os.unlink(csv_path)


class TestFundingArbViaFactory(unittest.TestCase):

    def test_factory_resolves_funding_arb(self):
        from core.bootstrap import register_defaults
        from core.factories.strategy_factory import StrategyFactory
        register_defaults()
        cls = StrategyFactory.resolve_class("FundingRateArbStrategy")
        self.assertIs(cls, FundingRateArbStrategy)


if __name__ == "__main__":
    unittest.main()
