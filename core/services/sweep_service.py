"""
core/services/sweep_service.py
===============================
Parameter-search service.

Composes the existing scoring stack into a single CLI-callable flow:

    SearchSpace (constraints)
        → ParameterGrid
        → BatchBacktest (with optional CV: PurgedKFold / WalkForward / CPCV)
        → Selector (filters + tie-breakers + CV-stability penalty)
        → ranked top-N + CSV export

Previously the sweep had its own engine-config builder and a hand-rolled
loop, which duplicated ``BatchBacktest`` (no CV, no failure dummy, no
selection-bias warning).  This module is now a thin orchestrator over
:class:`BatchBacktest` so every fix that lands in the batch core also
lands here.
"""
from __future__ import annotations

import csv
import logging
import time
from typing import Any, Callable, Dict, List, Optional, Tuple

from Backtest.realism_config import RealismConfig
from Backtest.runner import BacktestConfig, DataConfig
from Backtest.scoring.batch import BatchBacktest, BatchResult
from Backtest.scoring.scorer import Scorer
from Backtest.scoring.search_space import ParameterGrid, SearchSpace
from Backtest.scoring.selector import Selector, SelectionCriteria
from Interfaces.metrics_interface import BacktestResult
from Interfaces.strategy_adapter import IBacktestStrategy, SizingConfig, SizingMode
from core.factories.strategy_factory import StrategyFactory

log = logging.getLogger(__name__)

# Type alias for ranked results
RankedResult = Tuple[Dict[str, Any], BacktestResult, float]


def _build_search_space(
    param_grid: Dict[str, List[Any]],
    constraints: Optional[List[Dict[str, Any]]] = None,
) -> SearchSpace | ParameterGrid:
    """Build a ParameterGrid (no constraints) or SearchSpace (with).

    ``constraints`` is a list of dicts, each having ``type`` and
    type-specific params:

    * ``{type: less_than,    a: <param>, b: <param>}``  → a < b
    * ``{type: less_equal,   a: <param>, b: <param>}``  → a ≤ b
    * ``{type: range,        param: <name>, min: x, max: y}``
    * ``{type: max_leverage, max: <int>}``
    """
    if not constraints:
        return ParameterGrid(param_grid)

    space = SearchSpace()
    for name, values in param_grid.items():
        space.add(name, values)
    for c in constraints:
        ctype = c.get("type")
        if ctype == "less_than":
            space.require_less_than(c["a"], c["b"])
        elif ctype == "less_equal":
            space.require_less_equal(c["a"], c["b"])
        elif ctype == "range":
            space.require_range(c["param"], c["min"], c["max"])
        elif ctype == "max_leverage":
            space.require_max_leverage(c["max"])
        else:
            raise ValueError(f"unknown constraint type '{ctype}'")
    return space


def _build_selection_criteria(**kwargs: Any) -> SelectionCriteria:
    """Filter caller kwargs through SelectionCriteria's known fields."""
    valid = SelectionCriteria.__dataclass_fields__.keys()
    return SelectionCriteria(**{k: v for k, v in kwargs.items()
                                if k in valid and v is not None})


class SweepService:
    """Run a parameter sweep using BatchBacktest underneath.

    All fields default to safe values; the CLI wires user choices through.
    """

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
        # NEW: CV
        cv_method: str = "none",
        cv_n_splits: int = 5,
        cv_embargo_pct: float = 0.01,
        cv_train_pct: float = 0.6,
        cv_n_test_splits: int = 2,
        cv_aggregate: str = "mean",
        cv_expanding: bool = False,
        # NEW: hyperband (successive halving)
        cv_hyperband: bool = False,
        cv_halving_factor: int = 2,
        cv_min_active: int = 2,
        # NEW: realism
        realism_config_path: Optional[str] = None,
        # NEW: SearchSpace constraints (list of dicts; see _build_search_space)
        constraints: Optional[List[Dict[str, Any]]] = None,
        # NEW: Selector / filters
        min_trades: Optional[int] = None,
        scorer_min_trades: int = 10,
        min_sharpe: Optional[float] = None,
        max_drawdown: Optional[float] = None,
        min_win_rate: Optional[float] = None,
        cv_stability_weight: Optional[float] = None,
        max_cv_std: Optional[float] = None,
        **engine_overrides: Any,
    ) -> List[RankedResult]:
        """Run a parameter sweep and return ranked results."""
        # 1) Search space
        space = _build_search_space(param_grid, constraints)
        n_combos = len(space)
        if n_combos == 0:
            log.warning("Search space is empty after constraints — nothing to sweep")
            return []

        log.info("=" * 80)
        log.info("%s PARAMETER SWEEP", strategy_name)
        log.info("=" * 80)
        log.info("Symbol: %s | TF: %s | Leverage: %sx | Margin: $%s",
                 symbol, timeframe, leverage, margin_usd)
        log.info("Grid: %d combinations (CV=%s)", n_combos, cv_method)
        for k, v in param_grid.items():
            log.info("  %s: %s", k, v)
        if constraints:
            log.info("Constraints: %d (filtered out %d combos)",
                     len(constraints),
                     space.total_unconstrained() - n_combos
                     if isinstance(space, SearchSpace) else 0)
        log.info("=" * 80)

        # 2) BatchBacktest config — reuse DataConfig + BacktestConfig
        data_cfg = DataConfig(
            tick_data_dir=data_dir,
            symbols=[symbol],
            bar_type="time",
            timeframe=timeframe,
        )
        realism = (
            RealismConfig.from_yaml(realism_config_path)
            if realism_config_path else RealismConfig()
        )
        # Sizing must flow through so CV folds compute actual qty.  Without
        # this, the engine keeps the strategy's qty=1.0 placeholder and
        # max_position_notional silently rejects every order on high-price
        # symbols (BTC at $70k × 1.0 = $70k > 50k limit → 0 trades).
        sizing = SizingConfig(
            mode=SizingMode.MARGIN_USD,
            margin_usd=float(margin_usd),
            leverage=float(leverage),
            leverage_mode="margin",
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
            close_positions_at_end=engine_overrides.pop("close_positions_at_end", True),
            random_seed=engine_overrides.pop("random_seed", 42),
            bar_store_maxlen=engine_overrides.pop("bar_store_maxlen", 600),
            enable_tick_exit=engine_overrides.pop("enable_tick_exit", True),
            tp_pct=tp_pct,
            sl_pct=sl_pct,
            realism=realism,
            sizing=sizing,
        )

        # 3) Strategy factory closure (BatchBacktest expects it)
        strategy_cls = StrategyFactory.resolve_class(strategy_name)

        def _factory(params: Dict[str, Any]) -> IBacktestStrategy:
            return strategy_cls(**params)

        batch = BatchBacktest(
            config=bt_cfg,
            strategy_factory=_factory,
            scorer=Scorer(min_trades=scorer_min_trades),
            strategy_name=strategy_name,
        )

        # 4) Run with or without CV
        t0 = time.time()
        if cv_method == "none":
            batch_result = batch.run(space)
        elif cv_hyperband:
            if cv_method == "cpcv":
                raise ValueError(
                    "cv_hyperband is incompatible with cv_method='cpcv'. "
                    "Use 'walk_forward' or 'purged_kfold'."
                )
            log.info(
                "Hyperband enabled: halving_factor=%d, min_active=%d",
                cv_halving_factor, cv_min_active,
            )
            batch_result = batch.run_with_cv_hyperband(
                param_space=space,
                split_mode=cv_method,
                n_splits=cv_n_splits,
                embargo_pct=cv_embargo_pct,
                train_pct=cv_train_pct,
                aggregate=cv_aggregate,
                halving_factor=cv_halving_factor,
                min_active=cv_min_active,
            )
        else:
            batch_result = batch.run_with_cv(
                param_space=space,
                split_mode=cv_method,
                n_splits=cv_n_splits,
                embargo_pct=cv_embargo_pct,
                train_pct=cv_train_pct,
                n_test_splits=cv_n_test_splits,
                aggregate=cv_aggregate,
            )
        elapsed = time.time() - t0

        # 5) Selector — filters + tie-breakers + CV stability penalty
        criteria = _build_selection_criteria(
            min_trades=min_trades,
            min_sharpe=min_sharpe,
            max_drawdown=max_drawdown,
            min_win_rate=min_win_rate,
            cv_stability_weight=cv_stability_weight or 0.0,
            max_cv_std=max_cv_std if max_cv_std is not None else float("inf"),
        )
        selector = Selector(criteria)
        # Always rank ALL combos for the CSV — filters only narrow the
        # printed top-N.  Earlier behaviour silently dropped the CSV when
        # every combo failed a filter (e.g. all kombolar 30 trade altında),
        # which left the user with nothing to inspect.
        any_filter = any(v is not None for v in
                         [min_trades, min_sharpe, max_drawdown, min_win_rate, max_cv_std])
        unfiltered = selector.select_top_k(
            batch_result, k=len(batch_result.results), apply_filters=False,
        )
        if any_filter:
            filtered = selector.select_top_k(
                batch_result, k=len(batch_result.results), apply_filters=True,
            )
        else:
            filtered = unfiltered

        # Reshape to (params, result, score) for backwards compatibility
        all_results: List[RankedResult] = [(p, r, s) for r, s, p in unfiltered]
        results: List[RankedResult] = [(p, r, s) for r, s, p in filtered]

        # 6) CSV export — always write the full ranking so failed-filter
        # runs leave a debuggable trail.
        if csv_output:
            self._write_csv(csv_output, all_results, cv_method=cv_method)
            log.info(
                "Results saved to %s (%d combos, %d passed filters)",
                csv_output, len(all_results), len(results),
            )

        self.print_top_results(results, top_n=top_n, cv_method=cv_method)
        if results:
            self._print_best(results[0], cv_method=cv_method)
        elif all_results:
            log.warning(
                "All %d combos were filtered out — printing best-of-unfiltered "
                "for inspection:", len(all_results),
            )
            self._print_best(all_results[0], cv_method=cv_method)

        # Selection-bias warning
        if batch_result.selection_bias_report:
            batch_result.print_selection_bias_warning()

        avg = elapsed / max(n_combos, 1)
        cv_label = f" (CV={cv_method})" if cv_method != "none" else ""
        print(f"\nTotal time: {elapsed:.1f}s "
              f"({avg:.1f}s per combo, {n_combos} combos{cv_label})")
        if csv_output:
            print(f"Results CSV: {csv_output}")
        if any_filter and len(results) < len(batch_result.results):
            print(f"Filter dropped {len(batch_result.results) - len(results)} combos")

        return results

    # ------------------------------------------------------------------
    # Presentation
    # ------------------------------------------------------------------

    @staticmethod
    def print_top_results(
        results: List[RankedResult],
        top_n: int = 15,
        cv_method: str = "none",
    ) -> None:
        if not results:
            print("No results passed the filters.")
            return

        cv_on = cv_method != "none"
        print("\n" + "=" * 120)
        print(f"TOP {min(top_n, len(results))} PARAMETER COMBINATIONS"
              f"{f'  (CV={cv_method})' if cv_on else ''}")
        print("=" * 120)

        param_keys = list(results[0][0].keys())
        param_header = " ".join(f"{k[:8]:>8}" for k in param_keys)
        cv_col = f" {'CV_std':>7}" if cv_on else ""
        print(
            f"{'#':<4} {param_header} | "
            f"{'Return':>8} {'Sharpe':>7} {'MaxDD':>6} {'Trades':>6} "
            f"{'WinR':>5} {'PF':>5}{cv_col} | {'Score':>7}"
        )
        print("-" * 120)

        for rank, (params, result, score) in enumerate(results[:top_n], 1):
            pvals = " ".join(f"{str(v)[:8]:>8}" for v in params.values())
            cv_std = (result.metadata or {}).get("cv_score_std", 0.0)
            cv_str = f" {cv_std:>7.3f}" if cv_on else ""
            print(
                f"{rank:<4} {pvals} | "
                f"{result.total_return_pct:>+7.2f}% {result.sharpe_ratio:>7.3f} "
                f"{result.max_drawdown * 100:>5.1f}% {result.total_trades:>6} "
                f"{result.win_rate * 100:>4.0f}% {result.profit_factor:>5.2f}"
                f"{cv_str} | {score:>7.3f}"
            )

    @staticmethod
    def _print_best(best: RankedResult, cv_method: str = "none") -> None:
        params, result, score = best
        meta = result.metadata or {}
        print("\n" + "=" * 120)
        print("BEST PARAMETERS")
        print("=" * 120)
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
        if cv_method != "none":
            print(f"  CV folds:       {meta.get('cv_n_folds', '?')}")
            print(f"  CV mean:        {meta.get('cv_score_mean', 0):.4f}")
            print(f"  CV std:         {meta.get('cv_score_std', 0):.4f}")
            print(f"  CV min/max:     {meta.get('cv_score_min', 0):.4f} / "
                  f"{meta.get('cv_score_max', 0):.4f}")

    @staticmethod
    def _write_csv(
        path: str,
        results: List[RankedResult],
        cv_method: str = "none",
    ) -> None:
        if not results:
            return
        param_keys = list(results[0][0].keys())
        cv_on = cv_method != "none"
        with open(path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            base = (
                ["rank"] + param_keys +
                ["return_pct", "sharpe", "sortino", "calmar",
                 "max_dd_pct", "trades", "win_rate", "profit_factor", "score"]
            )
            if cv_on:
                base += ["cv_score_mean", "cv_score_std",
                         "cv_score_min", "cv_score_max", "cv_n_folds"]
            writer.writerow(base)
            for rank, (params, result, score) in enumerate(results, 1):
                row = [rank] + [params[k] for k in param_keys] + [
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
                if cv_on:
                    meta = result.metadata or {}
                    row += [
                        f"{meta.get('cv_score_mean', 0):.4f}",
                        f"{meta.get('cv_score_std', 0):.4f}",
                        f"{meta.get('cv_score_min', 0):.4f}",
                        f"{meta.get('cv_score_max', 0):.4f}",
                        meta.get("cv_n_folds", 0),
                    ]
                writer.writerow(row)
