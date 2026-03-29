#!/usr/bin/env python3
"""
run_unified_backtest.py
=======================
Example backtest script demonstrating the unified strategy system.

This script shows how to run a backtest with:
- The unified IStrategy interface (same class works in live and backtest)
- Tick-level TP/SL exit checking (intrabar exits)
- Leverage/margin mode support
- Proper metrics reporting
- REALISTIC POSITION SIZING via --margin-usd or --notional-usd

USAGE:
======
    # Basic run with DOGEUSDT data (default: 100 USD margin * 10x leverage = $1000 notional)
    python run_unified_backtest.py
    
    # With custom margin (margin_usd * leverage = notional)
    python run_unified_backtest.py --symbol DOGEUSDT --leverage 10 --margin-usd 100 --tp-pct 0.02 --sl-pct 0.01
    
    # With explicit notional (ignores margin-usd)
    python run_unified_backtest.py --symbol DOGEUSDT --notional-usd 1000 --tp-pct 0.02 --sl-pct 0.01
    
    # With explicit quantity (ignores margin/notional)
    python run_unified_backtest.py --symbol DOGEUSDT --qty 10000 --tp-pct 0.02 --sl-pct 0.01

SIZING PRIORITY:
================
    1. --qty (explicit asset quantity, e.g. 10000 DOGE)
    2. --notional-usd (explicit USD notional, e.g. 1000 USD)
    3. --margin-usd (default: margin * leverage = notional)

REQUIREMENTS:
=============
- Tick data must exist in data/ticks/{SYMBOL}/
- To fetch data: python tools/fetch_ticks.py --symbol DOGEUSDT --start 2024-01-01 --end 2024-01-07
"""
import argparse
import sys
import os
from pathlib import Path

# Ensure project root is on path
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from Interfaces.market_data import Bar
from Interfaces.orders import Order, OrderType, OrderSide
from Interfaces.IStrategy import IStrategy, StrategyDecision

from Backtest.runner import BacktestRunner, BacktestConfig, DataConfig, create_runner
from Backtest.engine import BacktestEngine, EngineConfig
from Backtest.tick_store import TickStore, TickStoreConfig
from Backtest.realism_config import RealismConfig
from Interfaces.strategy_adapter import SizingConfig, SizingMode

from Strategy.RSIThreshold import Strategy as RSIStrategy

import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
log = logging.getLogger(__name__)


def _build_sizing_config(args) -> SizingConfig:
    """
    Build SizingConfig from CLI arguments.
    
    Priority order:
    1. --qty (or --position-size): Fixed asset quantity
    2. --notional-usd: Fixed USD notional, compute qty at entry
    3. --margin-usd: Margin * leverage = notional (default)
    
    Args:
        args: Parsed CLI arguments
        
    Returns:
        SizingConfig with appropriate mode and values
    """
    leverage = args.leverage if args.leverage_mode == "margin" else 1.0
    
    # Check for explicit qty (highest priority)
    fixed_qty = args.qty or args.position_size
    if fixed_qty is not None:
        return SizingConfig(
            mode=SizingMode.FIXED_QTY,
            fixed_qty=fixed_qty,
            leverage=leverage,
            leverage_mode=args.leverage_mode,
        )
    
    # Check for explicit notional
    if args.notional_usd is not None:
        return SizingConfig(
            mode=SizingMode.NOTIONAL_USD,
            notional_usd=args.notional_usd,
            leverage=leverage,
            leverage_mode=args.leverage_mode,
        )
    
    # Default: margin-based sizing
    return SizingConfig(
        mode=SizingMode.MARGIN_USD,
        margin_usd=args.margin_usd,
        leverage=leverage,
        leverage_mode=args.leverage_mode,
    )


def validate_data(data_dir: str, symbol: str) -> bool:
    """Check if tick data exists for the symbol."""
    tick_store = TickStore(TickStoreConfig(data_dir=data_dir))
    if tick_store.file_exists(symbol):
        return True
    
    print(f"""
================================================================================
ERROR: No tick data found for {symbol}
================================================================================

To run this backtest, you need tick data. Options:

1. Download real data:
   python tools/fetch_ticks.py --symbol {symbol} --start 2024-01-01 --end 2024-01-07

2. Use a different symbol that has data in {data_dir}/

Expected location: {data_dir}/{symbol}/YYYY-MM-DD.csv
================================================================================
""")
    return False


def run_backtest(args):
    """Run the backtest with specified parameters."""
    
    # Validate data exists
    data_dir = args.data_dir
    symbol = args.symbol
    
    if not validate_data(data_dir, symbol):
        sys.exit(1)
    
    # Determine sizing mode and compute target notional
    sizing_config = _build_sizing_config(args)
    
    print("\n" + "="*70)
    print("UNIFIED STRATEGY BACKTEST")
    print("="*70)
    print(f"Symbol:           {symbol}")
    print(f"Timeframe:        {args.timeframe}")
    print(f"Initial Capital:  ${args.capital:,.2f}")
    print(f"Leverage Mode:    {args.leverage_mode}")
    if args.leverage_mode == "margin":
        print(f"Leverage:         {args.leverage}x")
    print(f"Take Profit:      {args.tp_pct*100:.1f}%")
    print(f"Stop Loss:        {args.sl_pct*100:.1f}%")
    print(f"Tick-Level Exit:  {'Enabled' if args.tick_exit else 'Disabled'}")
    
    # Print sizing info
    print("-"*70)
    print("POSITION SIZING:")
    print(f"  Mode:           {sizing_config.mode.value}")
    if sizing_config.mode == SizingMode.FIXED_QTY:
        print(f"  Fixed Qty:      {sizing_config.fixed_qty:.6f} {symbol.replace('USDT', '')}")
    elif sizing_config.mode == SizingMode.NOTIONAL_USD:
        print(f"  Target Notional: ${sizing_config.notional_usd:,.2f}")
    elif sizing_config.mode == SizingMode.MARGIN_USD:
        print(f"  Margin USD:     ${sizing_config.margin_usd:,.2f}")
        print(f"  Leverage:       {sizing_config.leverage}x")
        estimated_notional = sizing_config.margin_usd * sizing_config.leverage
        print(f"  Est. Notional:  ${estimated_notional:,.2f}")
    print("="*70 + "\n")
    
    # Create engine config
    # Compute sensible risk limits based on sizing config
    target_notional = sizing_config.get_target_notional()
    max_position_notional = max(target_notional * 2, 10000.0)  # Allow 2x target notional
    
    config = EngineConfig(
        tick_data_dir=data_dir,
        symbols=[symbol],
        bar_type="time",
        timeframe=args.timeframe,
        initial_capital=args.capital,
        leverage_mode=args.leverage_mode,
        leverage=args.leverage,
        maintenance_margin_ratio=0.5,
        taker_fee_bps=0.0,
        maker_fee_bps=0.0,
        slippage_bps=0.0,
        spread_bps=5.0,
        max_position_size=1_000_000.0,  # 1M units (allow large DOGE positions)
        max_position_notional=max_position_notional,
        max_daily_loss=args.capital * 0.1,  # 10% of capital
        max_drawdown=0.5,
        close_positions_at_end=False,
        enable_tick_exit=args.tick_exit,
        random_seed=42,
        bar_store_maxlen=600,
    )
    
    # Load realism config from YAML if provided
    if args.realism_config:
        rc = RealismConfig.from_yaml(args.realism_config)
        config.realism = rc
        print(f"Loaded RealismConfig from: {args.realism_config}")
    
    # Create engine
    engine = BacktestEngine(config)
    
    # Create strategy with TP/SL parameters and sizing config
    # Strategy receives sizing_config; engine computes qty at entry time
    strategy = RSIStrategy(
        rsi_period=args.rsi_period,
        rsi_overbought=args.rsi_overbought,
        rsi_oversold=args.rsi_oversold,
        position_size=1.0,  # Placeholder - engine overrides via sizing_config
        sizing_config=sizing_config,
        take_profit_pct=args.tp_pct,
        stop_loss_pct=args.sl_pct,
        leverage=args.leverage,
    )
    
    log.info(f"Starting backtest with RSI strategy (period={args.rsi_period}, OB={args.rsi_overbought}, OS={args.rsi_oversold})")
    log.info(f"Sizing: {sizing_config.mode.value}, margin=${sizing_config.margin_usd}, leverage={sizing_config.leverage}x")
    
    # Run backtest with sizing config
    result = engine.run(strategy, sizing_config=sizing_config)
    
    # Print results
    print("\n" + "="*70)
    print("BACKTEST RESULTS")
    print("="*70)
    print(f"Strategy:           {result.strategy_name or 'RSIStrategy'}")
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
    print("="*70)
    
    # Decomposed costs
    meta = result.metadata or {}
    print(f"\nDECOMPOSED COSTS:")
    print(f"  Fee:              ${meta.get('total_fee_cost', 0):.4f}")
    print(f"  Spread:           ${meta.get('total_spread_cost', 0):.4f}")
    print(f"  Slippage:         ${meta.get('total_slippage_cost', 0):.4f}")
    print(f"  Funding:          ${meta.get('total_funding_cost', 0):.6f}")
    print(f"  Borrow:           ${meta.get('total_borrow_cost', 0):.6f}")
    
    # Tick exit stats
    tick_exits = meta.get("tick_exit_count", 0)
    if args.tick_exit:
        print(f"\nTICK-LEVEL EXIT STATS:")
        print(f"  Intrabar Exits:   {tick_exits}")
        print(f"  Tick Exit Mode:   Enabled")
    
    # Exit manager stats
    if strategy.exit_manager:
        exit_stats = strategy.exit_manager.get_stats()
        print(f"\nEXIT MANAGER STATS:")
        print(f"  Total Exits:      {exit_stats['exits_triggered']}")
        print(f"  Exit Reasons:     {exit_stats['exit_reasons']}")
    
    # Execution stats
    exec_stats = meta.get("execution_stats", {})
    print(f"\nEXECUTION STATS:")
    print(f"  Orders Submitted: {meta.get('order_count', 0)}")
    print(f"  Fills:            {meta.get('fill_count', 0)}")
    print(f"  Risk Rejected:    {meta.get('risk_rejected_orders', 0)}")
    
    print("\n" + "="*70)
    
    return result


def main():
    import warnings
    warnings.warn(
        "run_unified_backtest.py is deprecated. Use:\n"
        "  python app.py backtest --strategy RSIThreshold --symbol DOGEUSDT "
        '--strategy-params \'{"rsi_period":14}\'\n'
        "This script will be removed in a future version.",
        DeprecationWarning, stacklevel=2,
    )
    parser = argparse.ArgumentParser(
        description="Run unified strategy backtest with tick-level TP/SL"
    )
    
    # Data options
    parser.add_argument("--data-dir", type=str, default="./data/ticks",
                        help="Directory with tick data")
    parser.add_argument("--symbol", type=str, default="DOGEUSDT",
                        help="Trading symbol")
    parser.add_argument("--timeframe", type=str, default="15m",
                        help="Bar timeframe")
    
    # Capital/leverage
    parser.add_argument("--capital", type=float, default=10000.0,
                        help="Initial capital")
    parser.add_argument("--leverage-mode", type=str, default="margin",
                        choices=["spot", "margin"],
                        help="Leverage mode")
    parser.add_argument("--leverage", type=float, default=10.0,
                        help="Leverage multiplier (margin mode only)")
    
    # Strategy params
    parser.add_argument("--rsi-period", type=int, default=14,
                        help="RSI calculation period")
    parser.add_argument("--rsi-overbought", type=int, default=70,
                        help="RSI overbought threshold")
    parser.add_argument("--rsi-oversold", type=int, default=30,
                        help="RSI oversold threshold")
    
    # Position sizing options (priority: qty > notional-usd > margin-usd)
    sizing_group = parser.add_argument_group("Position Sizing", 
        "Control position size. Priority: --qty > --notional-usd > --margin-usd")
    sizing_group.add_argument("--margin-usd", type=float, default=100.0,
                        help="Margin in USD (default). Notional = margin * leverage. (default: 100)")
    sizing_group.add_argument("--notional-usd", type=float, default=None,
                        help="Explicit notional USD (overrides margin-usd). qty = notional / price")
    sizing_group.add_argument("--qty", type=float, default=None,
                        help="Explicit asset quantity (overrides all). E.g. 10000 = 10000 DOGE")
    # Legacy alias for backward compatibility
    parser.add_argument("--position-size", type=float, default=None,
                        help="(DEPRECATED) Alias for --qty. Use --qty instead.")
    
    # Exit params
    parser.add_argument("--tp-pct", type=float, default=0.05,
                        help="Take profit percentage (0.02 = 2%)")
    parser.add_argument("--sl-pct", type=float, default=0.05,
                        help="Stop loss percentage (0.01 = 1%)")
    
    # Tick exit
    parser.add_argument("--tick-exit", action="store_true", default=True,
                        help="Enable tick-level TP/SL checking")
    parser.add_argument("--no-tick-exit", action="store_false", dest="tick_exit",
                        help="Disable tick-level TP/SL checking")
    
    # Realism config
    parser.add_argument("--realism-config", type=str, default=None,
                        help="Path to realism YAML config file")
    
    args = parser.parse_args()
    
    run_backtest(args)


if __name__ == "__main__":
    main()
