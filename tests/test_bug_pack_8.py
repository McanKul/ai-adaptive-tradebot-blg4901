"""
tests/test_bug_pack_8.py
========================
Regression tests for the 8-bug user-reported pack.

Each block names the bug it pins down so future regressions trace
back to the original report rather than feeling like noise.
"""
from __future__ import annotations
import json
import os
import sys
import tempfile
import unittest
from datetime import datetime
from unittest.mock import AsyncMock, MagicMock, patch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from data_fetcher.binance_vision import (
    BINANCE_VISION_BASE,
    BinanceVisionFetcher,
    FetchConfig,
    _build_path,
)


# ---------------------------------------------------------------------------
# Bug 2 + 3 — futures URL support, market_type honoured
# ---------------------------------------------------------------------------

class TestFuturesUrlSupport(unittest.TestCase):

    def test_build_path_spot(self):
        path = _build_path("spot", "daily", "aggTrades", "BTCUSDT")
        self.assertEqual(path, "data/spot/daily/aggTrades/BTCUSDT/")

    def test_build_path_um(self):
        path = _build_path("um", "daily", "aggTrades", "BTCUSDT")
        self.assertEqual(path, "data/futures/um/daily/aggTrades/BTCUSDT/")

    def test_build_path_cm(self):
        path = _build_path("cm", "monthly", "trades", "BTCUSD_PERP")
        self.assertEqual(path, "data/futures/cm/monthly/trades/BTCUSD_PERP/")

    def test_build_path_unknown_market_raises(self):
        with self.assertRaises(ValueError):
            _build_path("perp", "daily", "aggTrades", "BTCUSDT")

    def test_get_daily_url_um(self):
        fetcher = BinanceVisionFetcher(FetchConfig(market_type="um"))
        url = fetcher.get_daily_url("BTCUSDT", datetime(2024, 3, 15))
        self.assertEqual(
            url,
            BINANCE_VISION_BASE
            + "data/futures/um/daily/aggTrades/BTCUSDT/"
            + "BTCUSDT-aggTrades-2024-03-15.zip",
        )

    def test_get_monthly_url_um(self):
        fetcher = BinanceVisionFetcher(FetchConfig(market_type="um"))
        url = fetcher.get_monthly_url("BTCUSDT", 2024, 4)
        self.assertIn("data/futures/um/monthly/aggTrades/BTCUSDT/", url)
        self.assertTrue(url.endswith("BTCUSDT-aggTrades-2024-04.zip"))

    def test_get_daily_url_uses_data_type(self):
        # When data_type="trades" the directory name and file name both update
        fetcher = BinanceVisionFetcher(FetchConfig(market_type="um", data_type="trades"))
        url = fetcher.get_daily_url("ETHUSDT", datetime(2024, 1, 5))
        self.assertIn("/trades/", url)
        self.assertIn("ETHUSDT-trades-2024-01-05.zip", url)


# ---------------------------------------------------------------------------
# Bug 1 — fetch_ticks.py multi-symbol crash
# ---------------------------------------------------------------------------

class TestFetchTicksMultiSymbol(unittest.TestCase):
    """The verify-loop used to call ``args.symbol.upper()`` even when
    only ``--symbols`` was supplied (single-symbol arg stayed None ->
    AttributeError).  After the fix the loop iterates the resolved
    ``symbols`` list."""

    def _read_source(self) -> str:
        path = os.path.join(os.path.dirname(__file__), "..", "tools",
                             "fetch_ticks.py")
        with open(path, "r", encoding="utf-8") as f:
            return f.read()

    def test_post_loop_uses_resolved_symbols_list(self):
        src = self._read_source()
        # Pin: the verify block at the bottom must walk the list, not
        # reach for the singular slot directly.
        self.assertIn("for sym in symbols:", src)
        self.assertIn("Path(args.output) / sym", src)
        # And ``args.output`` is no longer paired with ``args.symbol``
        # in the post-loop section (used to be ``args.output) / args.symbol.upper()``)
        self.assertNotIn("args.output) / args.symbol", src)

    def test_symbols_and_market_type_flags(self):
        src = self._read_source()
        self.assertIn("--symbols", src)
        self.assertIn("--market-type", src)
        self.assertIn("--data-type", src)


# ---------------------------------------------------------------------------
# Bug 4 — CV defaults to real tick range, not 0 → 1 year
# ---------------------------------------------------------------------------

class TestCvTickRangeInference(unittest.TestCase):

    def test_infer_returns_none_without_data(self):
        from Backtest.runner import BacktestConfig, DataConfig
        from Backtest.scoring.batch import BatchBacktest
        from Backtest.scoring.scorer import Scorer

        cfg = BacktestConfig(data=DataConfig(
            tick_data_dir="/does/not/exist", symbols=["BTCUSDT"],
        ))
        bb = BatchBacktest(
            config=cfg, strategy_factory=lambda p: None, scorer=Scorer(),
        )
        self.assertIsNone(bb._infer_tick_range())

    def test_infer_picks_up_partitioned_files(self):
        from Backtest.runner import BacktestConfig, DataConfig
        from Backtest.scoring.batch import BatchBacktest
        from Backtest.scoring.scorer import Scorer

        with tempfile.TemporaryDirectory() as tmp:
            sym_dir = os.path.join(tmp, "BTCUSDT")
            os.makedirs(sym_dir)
            for d in ("2026-01-05", "2026-01-06", "2026-01-08"):
                with open(os.path.join(sym_dir, f"{d}.csv"), "w") as f:
                    f.write("")
            cfg = BacktestConfig(data=DataConfig(
                tick_data_dir=tmp, symbols=["BTCUSDT"],
            ))
            bb = BatchBacktest(
                config=cfg, strategy_factory=lambda p: None, scorer=Scorer(),
            )
            inferred = bb._infer_tick_range()
            self.assertIsNotNone(inferred)
            start_ns, end_ns = inferred
            # First file at 2026-01-05 00:00 UTC, last at 2026-01-08 23:59:59.999999
            self.assertGreater(end_ns, start_ns)
            self.assertGreater(end_ns - start_ns, 3 * 86_400_000_000_000)

    def test_run_with_cv_raises_when_no_range_and_no_data(self):
        from Backtest.runner import BacktestConfig, DataConfig
        from Backtest.scoring.batch import BatchBacktest
        from Backtest.scoring.scorer import Scorer

        cfg = BacktestConfig(data=DataConfig(
            tick_data_dir="/does/not/exist", symbols=["BTCUSDT"],
        ))
        bb = BatchBacktest(
            config=cfg, strategy_factory=lambda p: None, scorer=Scorer(),
        )
        with self.assertRaises(ValueError) as ctx:
            bb.run_with_cv(param_space=[{}], split_mode="walk_forward")
        self.assertIn("explicit time range", str(ctx.exception))


# ---------------------------------------------------------------------------
# Bug 5 — _create_fold_config carries leverage + exit knobs
# ---------------------------------------------------------------------------

class TestCreateFoldConfigCarriesAllFields(unittest.TestCase):

    def test_leverage_and_exit_fields_propagate(self):
        from Backtest.runner import BacktestConfig, DataConfig
        from Backtest.scoring.batch import BatchBacktest
        from Backtest.scoring.scorer import Scorer
        from Backtest.scoring.splits import TimeRange

        cfg = BacktestConfig(
            data=DataConfig(tick_data_dir="x", symbols=["BTCUSDT"]),
            leverage_mode="margin",
            leverage=10,
            maintenance_margin_ratio=0.4,
            enable_tick_exit=True,
            tp_pct=0.025,
            sl_pct=0.012,
            trailing_stop_pct=0.008,
            max_holding_bars=96,
        )
        bb = BatchBacktest(
            config=cfg, strategy_factory=lambda p: None, scorer=Scorer(),
        )
        fold = bb._create_fold_config(
            TimeRange(start_ns=0, end_ns=86_400_000_000_000),
        )
        # Every leverage / exit knob must equal the parent config so
        # CV folds and the full-period run produce comparable results.
        self.assertEqual(fold.leverage_mode, "margin")
        self.assertEqual(fold.leverage, 10)
        self.assertAlmostEqual(fold.maintenance_margin_ratio, 0.4)
        self.assertTrue(fold.enable_tick_exit)
        self.assertAlmostEqual(fold.tp_pct, 0.025)
        self.assertAlmostEqual(fold.sl_pct, 0.012)
        self.assertAlmostEqual(fold.trailing_stop_pct, 0.008)
        self.assertEqual(fold.max_holding_bars, 96)


# ---------------------------------------------------------------------------
# Bug 6 — total_return_pct format unit (no double-percent)
# ---------------------------------------------------------------------------

class TestReturnPctFormatNoDoublePercent(unittest.TestCase):

    def test_engine_log_uses_percent_literal(self):
        path = os.path.join(os.path.dirname(__file__), "..", "Backtest", "engine.py")
        with open(path, "r", encoding="utf-8") as f:
            src = f.read()
        # The buggy form was ``total_return_pct:.2%`` (Python multiplies
        # by 100 again).  Confirm it's gone.
        self.assertNotIn("total_return_pct:.2%", src)
        # And the corrected form is present.
        self.assertIn("total_return_pct:+.2f}%", src)

    def test_runner_log_uses_percent_literal(self):
        path = os.path.join(os.path.dirname(__file__), "..", "Backtest", "runner.py")
        with open(path, "r", encoding="utf-8") as f:
            src = f.read()
        self.assertNotIn("total_return_pct:.2%", src)
        self.assertIn("total_return_pct:+.2f}%", src)


# ---------------------------------------------------------------------------
# Bug 7 — partial-fill CLI + realism mapping
# ---------------------------------------------------------------------------

class TestPartialFillsCliAndRealism(unittest.TestCase):

    def test_realism_config_carries_partial_fills(self):
        from Backtest.realism_config import RealismConfig
        cfg = RealismConfig.from_dict({
            "partial_fills": {
                "enabled": True, "liquidity_scale": 5.0, "min_fill_ratio": 0.2,
            },
        })
        self.assertTrue(cfg.partial_fills.enabled)
        self.assertAlmostEqual(cfg.partial_fills.liquidity_scale, 5.0)
        self.assertAlmostEqual(cfg.partial_fills.min_fill_ratio, 0.2)

    def test_realism_yaml_propagates_to_engine_config(self):
        # Build a service config purely in-memory; intercept the engine
        # to capture the EngineConfig it receives.
        from core.services.backtest_service import BacktestService
        from core.bootstrap import register_defaults
        register_defaults()

        with tempfile.TemporaryDirectory() as tmp:
            yaml_path = os.path.join(tmp, "realism.yaml")
            with open(yaml_path, "w") as f:
                f.write(
                    "partial_fills:\n"
                    "  enabled: true\n"
                    "  liquidity_scale: 7.5\n"
                    "  min_fill_ratio: 0.3\n"
                )

            svc = BacktestService()
            captured = {}

            class _DummyEngine:
                def __init__(self, config):
                    captured["cfg"] = config

                def run(self, strategy, sizing_config=None):
                    from Interfaces.metrics_interface import BacktestResult
                    return BacktestResult()

            with patch("core.services.backtest_service.BacktestEngine",
                       _DummyEngine):
                svc.run_single(
                    strategy_name="EMACrossMACDTrend",
                    strategy_params={},
                    symbol="BTCUSDT",
                    realism_config_path=yaml_path,
                    data_dir=tmp,
                )

        cfg = captured["cfg"]
        self.assertTrue(cfg.enable_partial_fills)
        self.assertAlmostEqual(cfg.liquidity_scale, 7.5)
        self.assertAlmostEqual(cfg.min_fill_ratio, 0.3)

    def test_cli_flag_overrides_realism(self):
        from core.services.backtest_service import BacktestService
        from core.bootstrap import register_defaults
        register_defaults()

        with tempfile.TemporaryDirectory() as tmp:
            yaml_path = os.path.join(tmp, "realism.yaml")
            with open(yaml_path, "w") as f:
                f.write(
                    "partial_fills:\n  enabled: false\n"
                )
            svc = BacktestService()
            captured = {}

            class _DummyEngine:
                def __init__(self, config):
                    captured["cfg"] = config

                def run(self, strategy, sizing_config=None):
                    from Interfaces.metrics_interface import BacktestResult
                    return BacktestResult()

            with patch("core.services.backtest_service.BacktestEngine",
                       _DummyEngine):
                svc.run_single(
                    strategy_name="EMACrossMACDTrend",
                    strategy_params={},
                    symbol="BTCUSDT",
                    realism_config_path=yaml_path,
                    enable_partial_fills=True,
                    liquidity_scale=4.0,
                    min_fill_ratio=0.05,
                    data_dir=tmp,
                )
        cfg = captured["cfg"]
        # CLI must win
        self.assertTrue(cfg.enable_partial_fills)
        self.assertAlmostEqual(cfg.liquidity_scale, 4.0)
        self.assertAlmostEqual(cfg.min_fill_ratio, 0.05)


# ---------------------------------------------------------------------------
# Bug 8 — export-trades carries runtime metadata
# ---------------------------------------------------------------------------

class TestExportTradesMetadata(unittest.TestCase):

    def test_summary_includes_run_context(self):
        from core.services.backtest_service import BacktestService
        from Interfaces.metrics_interface import BacktestResult

        result = BacktestResult(
            strategy_name="X", initial_capital=10_000, final_equity=10_500,
            total_return=500, total_return_pct=5.0, max_drawdown=0.04,
            sharpe_ratio=1.0, total_trades=1, win_rate=1.0,
            metadata={
                "partial_fills_enabled": True,
                "partial_fills": 3,
                "avg_fill_ratio": 0.92,
                "tick_exit_count": 5,
                "total_fee_cost": 0.50,
                "total_spread_cost": 0.25,
                "total_slippage_cost": 0.10,
            },
            trades=[{"symbol": "BTCUSDT", "entry_side": "LONG",
                     "entry_price": 100.0, "exit_price": 105.0,
                     "quantity": 1.0,
                     "entry_timestamp_ns": 1_700_000_000_000_000_000,
                     "exit_timestamp_ns": 1_700_000_001_000_000_000,
                     "pnl": 5.0}],
        )
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "out.json")
            BacktestService.export_trades(
                result, path, strategy_name="X", symbol="BTCUSDT",
                run_metadata={
                    "timeframe": "15m",
                    "data_dir": "./data/ticks",
                    "realism_config": "example_realism_config.yaml",
                    "tick_exit_enabled": True,
                    "partial_fills_cli": None,
                    "leverage": 5,
                },
            )
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)

        s = data["summary"]
        for key in (
            "partial_fills_enabled", "partial_fills_count",
            "avg_fill_ratio", "tick_exit_count",
            "total_fee_cost", "total_spread_cost", "total_slippage_cost",
            "timeframe", "data_dir", "realism_config",
            "tick_exit_enabled", "leverage",
        ):
            self.assertIn(key, s, msg=f"missing summary key: {key}")
        self.assertEqual(s["leverage"], 5)
        self.assertEqual(s["partial_fills_count"], 3)


if __name__ == "__main__":
    unittest.main()
