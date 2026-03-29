#!/usr/bin/env python3
"""
run_emacross_backtest.py
========================
EMACrossMACDTrend stratejisini AVAXUSDT üzerinde backtest eder.
"""
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from Backtest.engine import BacktestEngine, EngineConfig
from Interfaces.strategy_adapter import SizingConfig, SizingMode
from Strategy.EMACrossMACDTrend import Strategy as EMACrossStrategy

import logging
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
log = logging.getLogger(__name__)


def main():
    import warnings
    warnings.warn(
        "run_emacross_backtest.py is deprecated. Use:\n"
        "  python app.py backtest --strategy EMACrossMACDTrend --symbol AVAXUSDT "
        '--strategy-params \'{"fast_ema_period":12,"slow_ema_period":26,...}\'\n'
        "This script will be removed in a future version.",
        DeprecationWarning, stacklevel=2,
    )
    symbol = "AVAXUSDT"
    data_dir = "./data/ticks"
    timeframe = "15m"
    capital = 10_000.0
    leverage = 10.0
    margin_usd = 100.0

    sizing_config = SizingConfig(
        mode=SizingMode.MARGIN_USD,
        margin_usd=margin_usd,
        leverage=leverage,
        leverage_mode="margin",
    )

    target_notional = sizing_config.get_target_notional()

    config = EngineConfig(
        tick_data_dir=data_dir,
        symbols=[symbol],
        bar_type="time",
        timeframe=timeframe,
        initial_capital=capital,
        leverage_mode="margin",
        leverage=leverage,
        maintenance_margin_ratio=0.5,
        taker_fee_bps=4.0,
        maker_fee_bps=2.0,
        slippage_bps=1.0,
        spread_bps=2.0,
        max_position_size=1_000_000.0,
        max_position_notional=max(target_notional * 5, 50_000.0),
        max_daily_loss=capital * 0.5,
        max_drawdown=0.9,
        close_positions_at_end=False,
        enable_tick_exit=True,
        random_seed=42,
        bar_store_maxlen=600,
    )

    engine = BacktestEngine(config)

    strategy = EMACrossStrategy(
        fast_ema_period=12,
        slow_ema_period=26,
        macd_signal=9,
        atr_period=14,
        atr_mult=2.5,
        adx_period=14,
        adx_threshold=20.0,
        volume_filter=True,
        volume_period=20,
        risk_pct=0.005,
        position_size=1.0,
        allow_reversal=True,
        exit_on_macd_cross=True,
    )

    log.info(f"Starting EMACrossMACDTrend backtest on {symbol} ({timeframe})")

    result = engine.run(strategy, sizing_config=sizing_config)

    print("\n" + "=" * 70)
    print("EMACrossMACDTrend BACKTEST RESULTS  —  " + symbol)
    print("=" * 70)
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


if __name__ == "__main__":
    main()
