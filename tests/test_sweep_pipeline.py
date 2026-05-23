"""
tests/test_sweep_pipeline.py
============================
Integration tests for the unified sweep pipeline:

    SearchSpace (constraints)
       → ParameterGrid
       → BatchBacktest (optional CV: PurgedKFold / WalkForward / CPCV)
       → Selector (filters + tie-breakers + CV-stability penalty)

These tests stub the heavy parts (engine, runner) and exercise the
service-level wiring so we know the CLI flag → SweepService → batch →
selector chain stays correct as the codebase evolves.
"""
from __future__ import annotations
import os
import sys
import unittest
from unittest.mock import patch, MagicMock

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from Backtest.scoring.search_space import ParameterGrid, SearchSpace
from Backtest.scoring.selector import SelectionCriteria
from core.services.sweep_service import (
    SweepService,
    _build_search_space,
    _build_selection_criteria,
)


# ---------------------------------------------------------------------------
# _build_search_space
# ---------------------------------------------------------------------------

class TestBuildSearchSpace(unittest.TestCase):

    def test_no_constraints_returns_parameter_grid(self):
        grid = _build_search_space({"a": [1, 2], "b": [10, 20]}, constraints=None)
        self.assertIsInstance(grid, ParameterGrid)
        self.assertEqual(len(grid), 4)

    def test_less_than_constraint_filters_combos(self):
        space = _build_search_space(
            {"oversold": [20, 30, 40], "overbought": [60, 70, 80]},
            constraints=[{"type": "less_than", "a": "oversold", "b": "overbought"}],
        )
        self.assertIsInstance(space, SearchSpace)
        # Every combo should satisfy oversold < overbought; 3*3=9 unconstrained,
        # all 9 happen to satisfy here, but the constraint shape matters
        for params in space:
            self.assertLess(params["oversold"], params["overbought"])

    def test_less_than_filters_when_overlap(self):
        space = _build_search_space(
            {"a": [10, 20, 30], "b": [15, 25]},
            constraints=[{"type": "less_than", "a": "a", "b": "b"}],
        )
        # 3*2=6 unconstrained; valid: (10,15), (10,25), (20,25) = 3
        self.assertEqual(len(space), 3)

    def test_range_constraint(self):
        space = _build_search_space(
            {"x": [1, 5, 10, 50, 100]},
            constraints=[{"type": "range", "param": "x", "min": 5, "max": 50}],
        )
        self.assertEqual(len(space), 3)  # 5, 10, 50

    def test_unknown_constraint_type_raises(self):
        with self.assertRaises(ValueError):
            _build_search_space(
                {"a": [1, 2]},
                constraints=[{"type": "bogus"}],
            )


# ---------------------------------------------------------------------------
# _build_selection_criteria
# ---------------------------------------------------------------------------

class TestBuildSelectionCriteria(unittest.TestCase):

    def test_passes_only_known_fields(self):
        sc = _build_selection_criteria(
            min_trades=20, min_sharpe=0.5,
            unknown_field="ignored", another_unknown=999,
        )
        self.assertIsInstance(sc, SelectionCriteria)
        self.assertEqual(sc.min_trades, 20)
        self.assertEqual(sc.min_sharpe, 0.5)

    def test_drops_none_values(self):
        sc = _build_selection_criteria(
            min_trades=None, min_sharpe=0.3,
        )
        # Defaults preserved when None passed
        self.assertEqual(sc.min_trades, 10)  # SelectionCriteria default
        self.assertEqual(sc.min_sharpe, 0.3)


# ---------------------------------------------------------------------------
# SweepService.run_sweep with stubbed BatchBacktest
# ---------------------------------------------------------------------------

class TestSweepServiceWiring(unittest.TestCase):
    """Verify SweepService routes args correctly without running real engine."""

    def _make_fake_batch_result(self, n=2):
        """Build a BatchResult-like object with `n` synthetic combos."""
        from Backtest.scoring.batch import BatchResult
        from Interfaces.metrics_interface import BacktestResult

        results = []
        scores = []
        params_list = []
        for i in range(n):
            r = BacktestResult(
                strategy_name="X",
                params={"p": i},
                initial_capital=10_000.0,
                final_equity=10_000.0 + i * 100,
                total_return=i * 100.0,
                total_return_pct=i * 1.0,
                max_drawdown=0.05,
                sharpe_ratio=0.5 + i * 0.1,
                total_trades=20,
                win_rate=0.55,
                metadata={},
            )
            results.append(r)
            scores.append(0.5 + i * 0.1)
            params_list.append({"p": i})
        rankings = sorted(range(n), key=lambda i: scores[i], reverse=True)
        return BatchResult(
            results=results, scores=scores, params_list=params_list,
            rankings=rankings, total_time_seconds=0.1,
            trial_count=n, failed_count=0,
        )

    def test_run_sweep_no_cv_uses_run(self):
        from core.bootstrap import register_defaults
        register_defaults()

        with patch("core.services.sweep_service.BatchBacktest") as MockBB:
            instance = MagicMock()
            instance.run.return_value = self._make_fake_batch_result(n=3)
            instance.run_with_cv.side_effect = AssertionError("should NOT be called")
            MockBB.return_value = instance

            svc = SweepService()
            results = svc.run_sweep(
                strategy_name="RSIThreshold",
                param_grid={"rsi_period": [14, 21, 28]},
                symbol="BTCUSDT",
                cv_method="none",
            )
            instance.run.assert_called_once()
            instance.run_with_cv.assert_not_called()
            self.assertEqual(len(results), 3)

    def test_run_sweep_cv_uses_run_with_cv(self):
        from core.bootstrap import register_defaults
        register_defaults()

        with patch("core.services.sweep_service.BatchBacktest") as MockBB:
            instance = MagicMock()
            instance.run.side_effect = AssertionError("should NOT be called")
            instance.run_with_cv.return_value = self._make_fake_batch_result(n=2)
            MockBB.return_value = instance

            svc = SweepService()
            results = svc.run_sweep(
                strategy_name="RSIThreshold",
                param_grid={"rsi_period": [14, 21]},
                symbol="BTCUSDT",
                cv_method="walk_forward",
                cv_n_splits=3,
                cv_embargo_pct=0.02,
            )
            instance.run_with_cv.assert_called_once()
            kwargs = instance.run_with_cv.call_args.kwargs
            self.assertEqual(kwargs["split_mode"], "walk_forward")
            self.assertEqual(kwargs["n_splits"], 3)
            self.assertAlmostEqual(kwargs["embargo_pct"], 0.02)
            self.assertEqual(len(results), 2)

    def test_filter_min_trades_drops_low_trade_combos(self):
        from core.bootstrap import register_defaults
        from Backtest.scoring.batch import BatchResult
        from Interfaces.metrics_interface import BacktestResult
        register_defaults()

        # Build a result where one combo has 5 trades, another has 50
        fake = BatchResult(
            results=[
                BacktestResult(strategy_name="X", params={"p": 0},
                               initial_capital=10_000, final_equity=10_500,
                               total_return=500, total_return_pct=5,
                               max_drawdown=0.05, sharpe_ratio=1.0,
                               total_trades=5, win_rate=0.6, metadata={}),
                BacktestResult(strategy_name="X", params={"p": 1},
                               initial_capital=10_000, final_equity=11_000,
                               total_return=1_000, total_return_pct=10,
                               max_drawdown=0.05, sharpe_ratio=1.5,
                               total_trades=50, win_rate=0.6, metadata={}),
            ],
            scores=[1.0, 1.5],
            params_list=[{"p": 0}, {"p": 1}],
            rankings=[1, 0],
            total_time_seconds=0.1,
        )
        with patch("core.services.sweep_service.BatchBacktest") as MockBB:
            instance = MagicMock()
            instance.run.return_value = fake
            MockBB.return_value = instance

            svc = SweepService()
            results = svc.run_sweep(
                strategy_name="RSIThreshold",
                param_grid={"p": [0, 1]},
                symbol="BTCUSDT",
                min_trades=20,  # drops the 5-trade combo
            )
            # Only the 50-trade combo survives the filter
            self.assertEqual(len(results), 1)
            self.assertEqual(results[0][0], {"p": 1})


# ---------------------------------------------------------------------------
# YAML param-grid loader
# ---------------------------------------------------------------------------

class TestParamGridLoader(unittest.TestCase):

    def _write(self, content, suffix=".yaml"):
        import tempfile
        f = tempfile.NamedTemporaryFile(mode="w", suffix=suffix, delete=False)
        f.write(content)
        f.close()
        return f.name

    def test_flat_yaml_returns_grid_no_constraints(self):
        from app import _load_param_grid
        path = self._write("rsi_period: [14, 21]\nthreshold: [0.5, 0.7]\n")
        try:
            grid, constraints = _load_param_grid(path)
            self.assertEqual(grid, {"rsi_period": [14, 21], "threshold": [0.5, 0.7]})
            self.assertEqual(constraints, [])
        finally:
            os.unlink(path)

    def test_structured_yaml_returns_grid_and_constraints(self):
        from app import _load_param_grid
        yaml_text = (
            "grid:\n"
            "  oversold: [20, 30]\n"
            "  overbought: [60, 70]\n"
            "constraints:\n"
            "  - {type: less_than, a: oversold, b: overbought}\n"
        )
        path = self._write(yaml_text)
        try:
            grid, constraints = _load_param_grid(path)
            self.assertEqual(grid, {"oversold": [20, 30], "overbought": [60, 70]})
            self.assertEqual(len(constraints), 1)
            self.assertEqual(constraints[0]["type"], "less_than")
        finally:
            os.unlink(path)


if __name__ == "__main__":
    unittest.main()
