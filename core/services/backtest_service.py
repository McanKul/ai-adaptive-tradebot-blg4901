"""
core/services/backtest_service.py
==================================
Thin orchestration wrapper for running a single backtest.

Composes StrategyFactory + BacktestEngine.  Contains zero business logic.
"""
from __future__ import annotations

import logging
from typing import Any, Dict, Optional

from Backtest.engine import BacktestEngine, EngineConfig
from Backtest.realism_config import RealismConfig
from Backtest.runner import BacktestConfig, DataConfig
from Backtest.scoring.batch import BatchBacktest, BatchResult
from Interfaces.metrics_interface import BacktestResult
from Interfaces.strategy_adapter import SizingConfig, SizingMode
from core.factories.composite_factory import CompositeFactory
from core.factories.strategy_factory import StrategyFactory

log = logging.getLogger(__name__)


class BacktestService:
    """Run a single backtest using factory-resolved strategies."""

    def run_single(
        self,
        strategy_name: str,
        strategy_params: Dict[str, Any],
        symbol: str,
        timeframe: str = "15m",
        capital: float = 10_000.0,
        leverage: float = 10.0,
        leverage_mode: str = "margin",
        margin_usd: float = 100.0,
        notional_usd: Optional[float] = None,
        fixed_qty: Optional[float] = None,
        tp_pct: Optional[float] = None,
        sl_pct: Optional[float] = None,
        tick_exit: bool = True,
        data_dir: str = "./data/ticks",
        realism_config_path: Optional[str] = None,
        composite_spec_path: Optional[str] = None,
        **engine_overrides: Any,
    ) -> BacktestResult:
        """
        Run a single backtest.

        Args:
            strategy_name: Registered strategy name (e.g. ``"EMACrossMACDTrend"``).
            strategy_params: Kwargs forwarded to the strategy constructor.
            symbol: Trading pair (e.g. ``"AVAXUSDT"``).
            timeframe: Bar timeframe (default ``"15m"``).
            capital: Initial capital.
            leverage / leverage_mode: Leverage settings.
            margin_usd / notional_usd / fixed_qty: Position sizing
                (priority: fixed_qty > notional_usd > margin_usd).
            tp_pct / sl_pct: Take-profit / stop-loss percentages.
            tick_exit: Enable tick-level TP/SL checking.
            data_dir: Directory with tick CSV files.
            realism_config_path: Optional path to realism YAML.
            **engine_overrides: Extra fields forwarded to ``EngineConfig``.

        Returns:
            ``BacktestResult`` with metrics, equity curve, trades.
        """
        # -- Sizing ---------------------------------------------------------
        sizing_config = self._build_sizing(
            fixed_qty=fixed_qty,
            notional_usd=notional_usd,
            margin_usd=margin_usd,
            leverage=leverage,
            leverage_mode=leverage_mode,
        )

        target_notional = sizing_config.get_target_notional()
        max_position_notional = max(target_notional * 2, 10_000.0)

        # -- Engine config --------------------------------------------------
        config = EngineConfig(
            tick_data_dir=data_dir,
            symbols=[symbol],
            bar_type=engine_overrides.pop("bar_type", "time"),
            timeframe=timeframe,
            initial_capital=capital,
            leverage_mode=leverage_mode,
            leverage=leverage,
            maintenance_margin_ratio=engine_overrides.pop("maintenance_margin_ratio", 0.5),
            taker_fee_bps=engine_overrides.pop("taker_fee_bps", 4.0),
            maker_fee_bps=engine_overrides.pop("maker_fee_bps", 2.0),
            slippage_bps=engine_overrides.pop("slippage_bps", 1.0),
            spread_bps=engine_overrides.pop("spread_bps", 2.0),
            max_position_size=engine_overrides.pop("max_position_size", 1_000_000.0),
            max_position_notional=engine_overrides.pop("max_position_notional", max_position_notional),
            max_daily_loss=engine_overrides.pop("max_daily_loss", capital * 0.5),
            max_drawdown=engine_overrides.pop("max_drawdown", 0.9),
            close_positions_at_end=engine_overrides.pop("close_positions_at_end", False),
            enable_tick_exit=tick_exit,
            random_seed=engine_overrides.pop("random_seed", 42),
            bar_store_maxlen=engine_overrides.pop("bar_store_maxlen", 600),
            tp_pct=tp_pct,
            sl_pct=sl_pct,
        )

        if realism_config_path:
            config.realism = RealismConfig.from_yaml(realism_config_path)

        # -- Strategy (composite or single) ---------------------------------
        if composite_spec_path:
            strategy = CompositeFactory.from_path(composite_spec_path)
            engine_sizing = None  # composite owns sizing
            log.info(
                "Backtest: composite spec=%s symbol=%s tf=%s slots=%d policy=%s",
                composite_spec_path, symbol, timeframe,
                len(strategy.slots), strategy.policy,
            )
        else:
            strategy_cls = StrategyFactory.resolve_class(strategy_name)
            strategy = strategy_cls(**strategy_params)
            engine_sizing = sizing_config
            log.info(
                "Backtest: strategy=%s symbol=%s tf=%s sizing=%s",
                strategy_name, symbol, timeframe, sizing_config.mode.value,
            )

        # -- Run ------------------------------------------------------------
        engine = BacktestEngine(config)
        result = engine.run(strategy, sizing_config=engine_sizing)
        return result

    # ------------------------------------------------------------------
    # Cross-validation
    # ------------------------------------------------------------------

    def run_with_cv(
        self,
        strategy_name: str,
        strategy_params: Dict[str, Any],
        symbol: str,
        timeframe: str = "15m",
        capital: float = 10_000.0,
        leverage: float = 10.0,
        leverage_mode: str = "margin",
        margin_usd: float = 100.0,
        notional_usd: Optional[float] = None,
        fixed_qty: Optional[float] = None,
        tp_pct: Optional[float] = None,
        sl_pct: Optional[float] = None,
        tick_exit: bool = True,
        data_dir: str = "./data/ticks",
        realism_config_path: Optional[str] = None,
        cv_method: str = "purged_kfold",
        cv_n_splits: int = 5,
        cv_embargo_pct: float = 0.01,
        cv_train_pct: float = 0.6,
        cv_n_test_splits: int = 2,
        cv_aggregate: str = "mean",
        cv_expanding: bool = False,
        **engine_overrides: Any,
    ) -> BatchResult:
        """
        Run a single strategy through cross-validation.

        Wraps :class:`Backtest.scoring.batch.BatchBacktest` with a one-element
        parameter list (``strategy_params``).  Honors the ``cv_*`` knobs in
        ``RealismConfig`` if loaded from YAML; CLI args override.

        Args:
            cv_method: ``"purged_kfold"`` | ``"walk_forward"`` | ``"cpcv"``.
            cv_n_splits: Number of folds (purged_kfold / cpcv).
            cv_embargo_pct: Embargo gap as fraction of total period.
            cv_train_pct: Training-window fraction (walk_forward).
            cv_n_test_splits: Test folds per split (cpcv).
            cv_aggregate: ``"mean"`` | ``"median"`` | ``"min"``.
            cv_expanding: Expanding-window walk-forward.

        Returns:
            ``BatchResult`` with per-fold scores and aggregated metrics.
        """
        # -- Sizing (reused) -----------------------------------------------
        sizing_config = self._build_sizing(
            fixed_qty=fixed_qty,
            notional_usd=notional_usd,
            margin_usd=margin_usd,
            leverage=leverage,
            leverage_mode=leverage_mode,
        )
        target_notional = sizing_config.get_target_notional()
        max_position_notional = max(target_notional * 2, 10_000.0)

        # -- Realism --------------------------------------------------------
        realism = (
            RealismConfig.from_yaml(realism_config_path)
            if realism_config_path else RealismConfig()
        )
        # CLI overrides realism cv_* (CLI > YAML > default)
        realism.cv_enabled = True
        realism.cv_method = cv_method
        realism.cv_n_splits = cv_n_splits
        realism.cv_embargo_pct = cv_embargo_pct
        realism.cv_expanding = cv_expanding

        # -- BacktestConfig (used by BacktestRunner inside BatchBacktest) ---
        data_config = DataConfig(
            tick_data_dir=data_dir,
            symbols=[symbol],
            bar_type=engine_overrides.pop("bar_type", "time"),
            timeframe=timeframe,
        )
        config = BacktestConfig(
            data=data_config,
            initial_capital=capital,
            taker_fee_bps=engine_overrides.pop("taker_fee_bps", 4.0),
            maker_fee_bps=engine_overrides.pop("maker_fee_bps", 2.0),
            slippage_bps=engine_overrides.pop("slippage_bps", 1.0),
            spread_bps=engine_overrides.pop("spread_bps", 2.0),
            max_position_size=engine_overrides.pop("max_position_size", 1_000_000.0),
            max_position_notional=engine_overrides.pop("max_position_notional", max_position_notional),
            max_daily_loss=engine_overrides.pop("max_daily_loss", capital * 0.5),
            max_drawdown=engine_overrides.pop("max_drawdown", 0.9),
            close_positions_at_end=engine_overrides.pop("close_positions_at_end", True),
            random_seed=engine_overrides.pop("random_seed", 42),
            bar_store_maxlen=engine_overrides.pop("bar_store_maxlen", 600),
            leverage_mode=leverage_mode,
            leverage=int(leverage),
            maintenance_margin_ratio=engine_overrides.pop("maintenance_margin_ratio", 0.5),
            enable_tick_exit=tick_exit,
            tp_pct=tp_pct,
            sl_pct=sl_pct,
            realism=realism,
        )

        # -- Strategy factory closure --------------------------------------
        strategy_cls = StrategyFactory.resolve_class(strategy_name)

        def _factory(params: Dict[str, Any]):
            return strategy_cls(**params)

        # -- BatchBacktest with single-element grid -----------------------
        batch = BatchBacktest(
            config=config,
            strategy_factory=_factory,
            strategy_name=strategy_name,
        )

        log.info(
            "CV backtest: strategy=%s symbol=%s tf=%s method=%s n_splits=%d embargo=%.3f",
            strategy_name, symbol, timeframe, cv_method, cv_n_splits, cv_embargo_pct,
        )

        return batch.run_with_cv(
            param_space=[strategy_params],
            split_mode=cv_method,
            n_splits=cv_n_splits,
            embargo_pct=cv_embargo_pct,
            train_pct=cv_train_pct,
            n_test_splits=cv_n_test_splits,
            aggregate=cv_aggregate,
        )

    @staticmethod
    def print_cv_result(batch: BatchResult, symbol: str = "") -> None:
        """Print formatted CV results to stdout (per-fold + aggregate)."""
        if not batch.results:
            print("\nCV produced no results.")
            return

        result = batch.results[0]
        meta = result.metadata or {}
        fold_details = meta.get("cv_fold_details", []) or []

        print("\n" + "=" * 70)
        print(f"CV BACKTEST RESULTS  —  {symbol}  (method={batch.cv_method})")
        print("=" * 70)
        print(f"Strategy:           {result.strategy_name or 'N/A'}")
        print(f"Folds:              {meta.get('cv_n_folds', len(fold_details))}")
        print(f"Aggregate:          {meta.get('cv_aggregate', 'mean')}")
        print(f"Aggregated Score:   {batch.scores[0]:.4f}")
        print(f"Score Mean:         {meta.get('cv_score_mean', 0):.4f}")
        print(f"Score Std:          {meta.get('cv_score_std', 0):.4f}")
        print(f"Score Min / Max:    {meta.get('cv_score_min', 0):.4f} / "
              f"{meta.get('cv_score_max', 0):.4f}")
        print("=" * 70)

        if fold_details:
            print(f"\nPER-FOLD BREAKDOWN:")
            print(f"  {'fold':>4} {'score':>10} {'sharpe':>8} {'return%':>9} "
                  f"{'mdd':>7} {'trades':>7} {'costs':>10}")
            for d in fold_details:
                print(f"  {d.get('fold_idx', 0):>4} "
                      f"{d.get('score', 0):>10.4f} "
                      f"{d.get('sharpe', 0):>8.3f} "
                      f"{d.get('total_return_pct', 0):>9.2f} "
                      f"{d.get('max_drawdown', 0):>7.2%} "
                      f"{d.get('total_trades', 0):>7d} "
                      f"{d.get('total_costs', 0):>10.2f}")
        print("=" * 70)

    # ------------------------------------------------------------------
    # Presentation helpers
    # ------------------------------------------------------------------

    @staticmethod
    def print_result(result: BacktestResult, symbol: str = "") -> None:
        """Print formatted backtest results to stdout."""
        print("\n" + "=" * 70)
        print(f"BACKTEST RESULTS  —  {symbol}")
        print("=" * 70)
        print(f"Strategy:           {result.strategy_name or 'N/A'}")
        print(f"Initial Capital:    ${result.initial_capital:,.2f}")
        print(f"Final Equity:       ${result.final_equity:,.2f}")
        print(f"Total Return:       {result.total_return_pct:.2f}%")
        print(f"Annualized Return:  {result.annualized_return:.2%}")
        print(f"Max Drawdown:       {result.max_drawdown:.2%}")
        print(f"Sharpe Ratio:       {result.sharpe_ratio:.3f}")
        print(f"Sortino Ratio:      {result.sortino_ratio:.3f}")
        print(f"Calmar Ratio:       {result.calmar_ratio:.3f}")
        print(f"Total Trades:       {result.total_trades}")
        print(f"Win Rate:           {result.win_rate:.1%}")
        print(f"Profit Factor:      {result.profit_factor:.3f}")
        print(f"Avg Trade Return:   {result.avg_trade_return:.4%}")
        print(f"Turnover:           {result.turnover:.2f}")
        print(f"Total Fees:         ${result.total_fees:.4f}")
        print(f"Total Costs:        ${result.total_costs:.4f}")
        print("=" * 70)

        meta = result.metadata or {}
        print(f"\nDECOMPOSED COSTS:")
        print(f"  Fee:              ${meta.get('total_fee_cost', 0):.4f}")
        print(f"  Spread:           ${meta.get('total_spread_cost', 0):.4f}")
        print(f"  Slippage:         ${meta.get('total_slippage_cost', 0):.4f}")

        tick_exits = meta.get("tick_exit_count", 0)
        print(f"\nTICK-LEVEL EXIT STATS:")
        print(f"  Intrabar Exits:   {tick_exits}")

        print(f"\nEXECUTION STATS:")
        print(f"  Orders Submitted: {meta.get('order_count', 0)}")
        print(f"  Fills:            {meta.get('fill_count', 0)}")
        print(f"  Risk Rejected:    {meta.get('risk_rejected_orders', 0)}")
        print("=" * 70)

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    @staticmethod
    def _build_sizing(
        fixed_qty: Optional[float],
        notional_usd: Optional[float],
        margin_usd: float,
        leverage: float,
        leverage_mode: str,
    ) -> SizingConfig:
        lev = leverage if leverage_mode == "margin" else 1.0
        if fixed_qty is not None:
            return SizingConfig(
                mode=SizingMode.FIXED_QTY,
                fixed_qty=fixed_qty,
                leverage=lev,
                leverage_mode=leverage_mode,
            )
        if notional_usd is not None:
            return SizingConfig(
                mode=SizingMode.NOTIONAL_USD,
                notional_usd=notional_usd,
                leverage=lev,
                leverage_mode=leverage_mode,
            )
        return SizingConfig(
            mode=SizingMode.MARGIN_USD,
            margin_usd=margin_usd,
            leverage=lev,
            leverage_mode=leverage_mode,
        )
