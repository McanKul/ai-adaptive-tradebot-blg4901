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
from Interfaces.metrics_interface import BacktestResult
from Interfaces.strategy_adapter import SizingConfig, SizingMode
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

        # -- Strategy -------------------------------------------------------
        strategy_cls = StrategyFactory.resolve_class(strategy_name)
        strategy = strategy_cls(**strategy_params)

        log.info(
            "Backtest: strategy=%s symbol=%s tf=%s sizing=%s",
            strategy_name, symbol, timeframe, sizing_config.mode.value,
        )

        # -- Run ------------------------------------------------------------
        engine = BacktestEngine(config)
        result = engine.run(strategy, sizing_config=sizing_config)
        return result

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
