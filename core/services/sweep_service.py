"""
core/services/sweep_service.py
===============================
Thin orchestration wrapper for parameter sweeps.

Composes StrategyFactory + ParameterGrid + Scorer + BacktestEngine.
Unified parameter sweep service for any registered strategy.
"""
from __future__ import annotations

import csv
import logging
import os
import time
from typing import Any, Dict, List, Optional, Tuple

from Backtest.engine import BacktestEngine, EngineConfig
from Backtest.runner import BacktestConfig, DataConfig
from Backtest.scoring.search_space import ParameterGrid
from Backtest.scoring.scorer import Scorer
from Interfaces.metrics_interface import BacktestResult
from Interfaces.strategy_adapter import SizingConfig, SizingMode
from core.factories.strategy_factory import StrategyFactory

log = logging.getLogger(__name__)

# Type alias for ranked results
RankedResult = Tuple[Dict[str, Any], BacktestResult, float]


class SweepService:
    """Run a parameter sweep using factory-resolved strategies."""

    def run_sweep(
        self,
        strategy_name: str,
        param_grid: Dict[str, List[Any]],
        symbol: str,
        timeframe: str = "15m",
        capital: float = 10_000.0,
        leverage: float = 10.0,
        margin_usd: float = 100.0,
        data_dir: str = "./data/ticks",
        csv_output: Optional[str] = None,
        top_n: int = 15,
        tp_pct: Optional[float] = None,
        sl_pct: Optional[float] = None,
        **engine_overrides: Any,
    ) -> List[RankedResult]:
        """
        Run a parameter sweep over the given grid.

        Args:
            strategy_name: Registered strategy name.
            param_grid: ``{param_name: [values]}``.
            symbol: Trading pair.
            timeframe: Bar timeframe.
            capital / leverage / margin_usd: Sizing defaults.
            data_dir: Tick data directory.
            csv_output: Optional CSV output path.
            top_n: Number of top results to print.
            tp_pct / sl_pct: Engine-level exit params (None = disabled).
            **engine_overrides: Extra EngineConfig fields.

        Returns:
            Sorted list of ``(params, result, score)`` tuples (best first).
        """
        grid = ParameterGrid(param_grid)
        all_combos = list(grid)
        total = len(all_combos)

        strategy_cls = StrategyFactory.resolve_class(strategy_name)

        log.info("=" * 80)
        log.info("%s PARAMETER SWEEP", strategy_name)
        log.info("=" * 80)
        log.info("Symbol: %s | TF: %s | Leverage: %sx | Margin: $%s", symbol, timeframe, leverage, margin_usd)
        log.info("Total combinations: %d", total)
        for k, v in param_grid.items():
            log.info("  %s: %s", k, v)
        log.info("=" * 80)

        sizing = SizingConfig(
            mode=SizingMode.MARGIN_USD,
            margin_usd=margin_usd,
            leverage=float(leverage),
            leverage_mode="margin",
        )

        scorer = Scorer()
        results: List[RankedResult] = []
        t0 = time.time()

        for i, params in enumerate(all_combos):
            try:
                result = self._run_single(
                    strategy_cls=strategy_cls,
                    params=params,
                    symbol=symbol,
                    timeframe=timeframe,
                    capital=capital,
                    leverage=leverage,
                    sizing=sizing,
                    data_dir=data_dir,
                    tp_pct=tp_pct,
                    sl_pct=sl_pct,
                    **engine_overrides,
                )
                score = scorer.score(result)
                results.append((params, result, score))

                log.info(
                    "[%3d/%d] %s | ret=%+6.2f%% sharpe=%.2f dd=%.1f%% trades=%d wr=%.0f%% | score=%.3f",
                    i + 1, total,
                    " ".join(f"{k}={v}" for k, v in params.items()),
                    result.total_return_pct,
                    result.sharpe_ratio,
                    result.max_drawdown * 100,
                    result.total_trades,
                    result.win_rate * 100,
                    score,
                )
            except Exception as e:
                log.error("[%3d/%d] FAILED: %s | %s", i + 1, total, params, e)

        elapsed = time.time() - t0

        # Sort by score descending
        results.sort(key=lambda x: x[2], reverse=True)

        # CSV export
        if csv_output:
            self._write_csv(csv_output, results)
            log.info("Results saved to %s", csv_output)

        # Print
        self.print_top_results(results, top_n=top_n)

        if results:
            self._print_best(results[0])

        avg = elapsed / total if total else 0
        print(f"\nTotal time: {elapsed:.1f}s ({avg:.1f}s per combo, {total} combos)")
        if csv_output:
            print(f"Results CSV: {csv_output}")

        return results

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    @staticmethod
    def _run_single(
        strategy_cls,
        params: Dict[str, Any],
        symbol: str,
        timeframe: str,
        capital: float,
        leverage: float,
        sizing: SizingConfig,
        data_dir: str,
        tp_pct: Optional[float] = None,
        sl_pct: Optional[float] = None,
        **engine_overrides: Any,
    ) -> BacktestResult:
        data_cfg = DataConfig(
            tick_data_dir=data_dir,
            symbols=[symbol],
            bar_type="time",
            timeframe=timeframe,
        )
        bt_cfg = BacktestConfig(
            data=data_cfg,
            initial_capital=capital,
            leverage_mode="margin",
            leverage=int(leverage),
            maintenance_margin_ratio=engine_overrides.pop("maintenance_margin_ratio", 0.5),
            taker_fee_bps=engine_overrides.pop("taker_fee_bps", 4.0),
            maker_fee_bps=engine_overrides.pop("maker_fee_bps", 2.0),
            slippage_bps=engine_overrides.pop("slippage_bps", 1.0),
            spread_bps=engine_overrides.pop("spread_bps", 2.0),
            max_position_size=engine_overrides.pop("max_position_size", 1_000_000.0),
            max_position_notional=engine_overrides.pop("max_position_notional", 50_000.0),
            max_daily_loss=engine_overrides.pop("max_daily_loss", capital * 0.5),
            max_drawdown=engine_overrides.pop("max_drawdown", 0.9),
            enable_tick_exit=engine_overrides.pop("enable_tick_exit", True),
            bar_store_maxlen=engine_overrides.pop("bar_store_maxlen", 600),
            tp_pct=tp_pct,
            sl_pct=sl_pct,
        )

        engine_cfg = bt_cfg.to_engine_config()
        engine = BacktestEngine(engine_cfg)

        strategy = strategy_cls(**params)
        result = engine.run(strategy, sizing_config=sizing)
        result.params = params
        return result

    # ------------------------------------------------------------------
    # Presentation
    # ------------------------------------------------------------------

    @staticmethod
    def print_top_results(results: List[RankedResult], top_n: int = 15) -> None:
        if not results:
            print("No results.")
            return

        print("\n" + "=" * 110)
        print(f"TOP {min(top_n, len(results))} PARAMETER COMBINATIONS")
        print("=" * 110)

        # Dynamic header from first result's param keys
        param_keys = list(results[0][0].keys())
        param_header = " ".join(f"{k[:8]:>8}" for k in param_keys)
        print(
            f"{'#':<4} {param_header} | "
            f"{'Return':>8} {'Sharpe':>7} {'MaxDD':>6} {'Trades':>6} {'WinR':>5} {'PF':>5} | {'Score':>7}"
        )
        print("-" * 110)

        for rank, (params, result, score) in enumerate(results[:top_n], 1):
            pvals = " ".join(f"{str(v)[:8]:>8}" for v in params.values())
            print(
                f"{rank:<4} {pvals} | "
                f"{result.total_return_pct:>+7.2f}% {result.sharpe_ratio:>7.3f} "
                f"{result.max_drawdown * 100:>5.1f}% {result.total_trades:>6} "
                f"{result.win_rate * 100:>4.0f}% {result.profit_factor:>5.2f} | "
                f"{score:>7.3f}"
            )

    @staticmethod
    def _print_best(best: RankedResult) -> None:
        params, result, score = best
        print("\n" + "=" * 110)
        print("BEST PARAMETERS")
        print("=" * 110)
        for k, v in params.items():
            print(f"  {k}: {v}")
        print("  ---")
        print(f"  Return:         {result.total_return_pct:+.2f}%")
        print(f"  Sharpe:         {result.sharpe_ratio:.3f}")
        print(f"  Max Drawdown:   {result.max_drawdown * 100:.2f}%")
        print(f"  Total Trades:   {result.total_trades}")
        print(f"  Win Rate:       {result.win_rate * 100:.1f}%")
        print(f"  Profit Factor:  {result.profit_factor:.3f}")
        print(f"  Score:          {score:.4f}")

    @staticmethod
    def _write_csv(path: str, results: List[RankedResult]) -> None:
        if not results:
            return
        param_keys = list(results[0][0].keys())
        with open(path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(
                ["rank"] + param_keys +
                ["return_pct", "sharpe", "sortino", "calmar",
                 "max_dd_pct", "trades", "win_rate", "profit_factor", "score"]
            )
            for rank, (params, result, score) in enumerate(results, 1):
                writer.writerow(
                    [rank] + [params[k] for k in param_keys] +
                    [
                        f"{result.total_return_pct:.4f}",
                        f"{result.sharpe_ratio:.4f}",
                        f"{result.sortino_ratio:.4f}",
                        f"{result.calmar_ratio:.4f}",
                        f"{result.max_drawdown * 100:.2f}",
                        result.total_trades,
                        f"{result.win_rate * 100:.1f}",
                        f"{result.profit_factor:.4f}",
                        f"{score:.4f}",
                    ]
                )
