"""
tests/test_regime_detectors.py
==============================
Tests for ADXRegimeDetector, ATRPercentileRegime, CompositeRegimeDetector,
RegimeFactory, and CompositeStrategy regime gating end-to-end.
"""
from __future__ import annotations
import os
import sys
import unittest

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from Interfaces.IRegimeDetector import RegimeState
from Interfaces.market_data import Bar
from Interfaces.strategy_adapter import StrategyContext
from Strategy.regime.adx_regime import ADXRegimeDetector
from Strategy.regime.atr_vol_regime import ATRPercentileRegime
from Strategy.regime.composite_regime import CompositeRegimeDetector


def make_bar(close=100.0, high=None, low=None, ts=1_000_000_000):
    return Bar(
        symbol="BTCUSDT", timeframe="1h",
        timestamp_ns=ts,
        open=close * 0.999, high=high or close * 1.005, low=low or close * 0.995,
        close=close, volume=1000.0,
    )


class _FakeCtx:
    """Duck-typed context exposing get_ohlcv()."""

    def __init__(self, ohlcv):
        self._ohlcv = ohlcv

    def get_ohlcv(self, limit=500):
        return self._ohlcv


def _ohlcv_from_close(closes, hl_amp=0.005):
    closes = np.asarray(closes, dtype=np.float64)
    return {
        "close": closes.tolist(),
        "high": (closes * (1 + hl_amp)).tolist(),
        "low": (closes * (1 - hl_amp)).tolist(),
        "open": closes.tolist(),
        "volume": [1000.0] * len(closes),
    }


# ---------------------------------------------------------------------------
# ADX
# ---------------------------------------------------------------------------

class TestADXRegime(unittest.TestCase):

    def test_warmup_returns_empty(self):
        det = ADXRegimeDetector(period=14)
        ctx = _FakeCtx(_ohlcv_from_close([100.0] * 5))
        state = det.detect(make_bar(), ctx)
        self.assertEqual(state.tags, [])

    def test_uptrend_emits_trend_up(self):
        det = ADXRegimeDetector(period=14, trend_threshold=20.0, di_gap=2.0)
        # Strong steady uptrend (5% per bar)
        closes = [100.0 * (1.05 ** i) for i in range(80)]
        state = det.detect(make_bar(close=closes[-1]), _FakeCtx(_ohlcv_from_close(closes)))
        self.assertIn("trend_up", state.tags)
        self.assertGreater(state.score.get("adx", 0), 20.0)

    def test_downtrend_emits_trend_down(self):
        det = ADXRegimeDetector(period=14, trend_threshold=20.0, di_gap=2.0)
        closes = [100.0 * (0.96 ** i) for i in range(80)]
        state = det.detect(make_bar(close=closes[-1]), _FakeCtx(_ohlcv_from_close(closes)))
        self.assertIn("trend_down", state.tags)

    def test_choppy_emits_range(self):
        det = ADXRegimeDetector(period=14, trend_threshold=25.0)
        rng = np.random.default_rng(42)
        # Mean-reverting random walk around 100, no drift
        closes = 100 + np.cumsum(rng.normal(0, 0.05, size=200)) - np.cumsum(rng.normal(0, 0.05, size=200))
        state = det.detect(make_bar(close=float(closes[-1])),
                            _FakeCtx(_ohlcv_from_close(closes.tolist())))
        # Should be range OR (rarely) empty during indeterminate gaps
        self.assertTrue("range" in state.tags or state.tags == [])


# ---------------------------------------------------------------------------
# ATR percentile
# ---------------------------------------------------------------------------

class TestATRPercentileRegime(unittest.TestCase):

    def test_warmup_returns_empty(self):
        det = ATRPercentileRegime(window=200)
        ctx = _FakeCtx(_ohlcv_from_close([100.0] * 50))
        state = det.detect(make_bar(), ctx)
        self.assertEqual(state.tags, [])

    def test_high_vol_burst_emits_vol_high(self):
        det = ATRPercentileRegime(window=200, hi_pct=0.8, lo_pct=0.2)
        # 200 quiet bars + 30 wild bars
        rng = np.random.default_rng(1)
        quiet = 100 + np.cumsum(rng.normal(0, 0.05, size=200))
        wild = quiet[-1] + np.cumsum(rng.normal(0, 5.0, size=30))
        closes = np.concatenate([quiet, wild])
        ohlcv = {
            "close": closes.tolist(),
            "high": (closes + 2.0).tolist(),
            "low": (closes - 2.0).tolist(),
            "open": closes.tolist(),
            "volume": [1.0] * len(closes),
        }
        state = det.detect(make_bar(close=float(closes[-1])), _FakeCtx(ohlcv))
        self.assertIn("vol_high", state.tags)

    def test_low_vol_emits_vol_low(self):
        det = ATRPercentileRegime(window=200, hi_pct=0.8, lo_pct=0.2)
        # 200 wild bars + 30 quiet bars (current ATR low vs trailing)
        rng = np.random.default_rng(2)
        wild = 100 + np.cumsum(rng.normal(0, 2.0, size=200))
        quiet = wild[-1] + np.cumsum(rng.normal(0, 0.01, size=30))
        closes = np.concatenate([wild, quiet])
        ohlcv = {
            "close": closes.tolist(),
            "high": (closes * 1.001).tolist(),
            "low": (closes * 0.999).tolist(),
            "open": closes.tolist(),
            "volume": [1.0] * len(closes),
        }
        state = det.detect(make_bar(close=float(closes[-1])), _FakeCtx(ohlcv))
        # Should be vol_low or vol_mid (not vol_high)
        self.assertNotIn("vol_high", state.tags)

    def test_invalid_thresholds_raise(self):
        with self.assertRaises(ValueError):
            ATRPercentileRegime(hi_pct=0.5, lo_pct=0.6)


# ---------------------------------------------------------------------------
# Composite detector
# ---------------------------------------------------------------------------

class TestCompositeRegime(unittest.TestCase):

    def test_tags_merged_no_duplicates(self):
        class _D:
            def __init__(self, tags):
                self.tags = tags

            def detect(self, bar, ctx):
                return RegimeState(tags=list(self.tags))

        comp = CompositeRegimeDetector(detectors=[_D(["trend_up"]), _D(["vol_high", "trend_up"])])
        state = comp.detect(make_bar(), _FakeCtx({}))
        self.assertEqual(set(state.tags), {"trend_up", "vol_high"})
        self.assertEqual(len(state.tags), 2)  # no dupes

    def test_empty_detectors_rejected(self):
        with self.assertRaises(ValueError):
            CompositeRegimeDetector(detectors=[])


# ---------------------------------------------------------------------------
# RegimeFactory
# ---------------------------------------------------------------------------

class TestRegimeFactory(unittest.TestCase):

    def test_factory_resolves_registered(self):
        from core.bootstrap import register_defaults
        from core.factories.regime_factory import RegimeFactory
        register_defaults()
        det = RegimeFactory.create("ADXRegimeDetector", period=14)
        self.assertIsInstance(det, ADXRegimeDetector)

    def test_factory_from_list_spec_builds_composite(self):
        from core.bootstrap import register_defaults
        from core.factories.regime_factory import RegimeFactory
        register_defaults()
        det = RegimeFactory.from_spec([
            {"type": "ADXRegimeDetector", "params": {"period": 10}},
            {"type": "ATRPercentileRegime", "params": {"window": 50, "atr_period": 14}},
        ])
        self.assertIsInstance(det, CompositeRegimeDetector)
        self.assertEqual(len(det.detectors), 2)


# ---------------------------------------------------------------------------
# Composite + regime gating end-to-end
# ---------------------------------------------------------------------------

class TestCompositeRegimeGating(unittest.TestCase):

    def test_yaml_spec_with_regime_loads(self):
        from core.bootstrap import register_defaults
        from core.factories.composite_factory import CompositeFactory
        register_defaults()
        path = os.path.join(os.path.dirname(__file__), "..",
                            "config", "profiles", "multi_strategy_example.yaml")
        comp = CompositeFactory.from_path(path)
        self.assertIsNotNone(comp.regime_detector)
        # Composite regime detector wraps both ADX and ATR
        self.assertEqual(len(comp.regime_detector.detectors), 2)


if __name__ == "__main__":
    unittest.main()
