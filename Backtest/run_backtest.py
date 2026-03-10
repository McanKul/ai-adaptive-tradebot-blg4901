#!/usr/bin/env python3
"""
Backtest/run_backtest.py
========================
CLI script for running backtests.

USAGE:
======
    # Single backtest (requires data)
    python -m Backtest.run_backtest --data-dir ./data/ticks --symbol DOGEUSDT
    
    # Parameter sweep
    python -m Backtest.run_backtest --mode sweep --data-dir ./data/ticks --symbol DOGEUSDT
    
    # With leverage mode (futures-style)
    python -m Backtest.run_backtest --leverage-mode margin --leverage 10 --data-dir ./data/ticks
    
    # Synthetic data for testing ONLY
    python -m Backtest.run_backtest --synthetic --data-dir ./data/ticks --symbol TEST

CRITICAL:
=========
- Backtest is DISK REPLAY ONLY (no live data)
- Strategy receives BARS, never TICKS
- If data is missing, script will fail with instructions to fetch data

To fetch real data:
    python tools/fetch_ticks.py --symbol DOGEUSDT --start 2024-01-01 --end 2024-01-07
"""
import argparse
import sys
import os
from pathlib import Path
from typing import List, Dict, Any

# Ensure project root is on path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from Interfaces.market_data import Bar
from Interfaces.orders import Order, OrderType, OrderSide
from Interfaces.strategy_adapter import StrategyContext, IBacktestStrategy

from Backtest.runner import BacktestRunner, BacktestConfig, DataConfig, create_runner
from Backtest.tick_store import TickStore, TickStoreConfig, TickDataNotFoundError
from Backtest.scoring.search_space import ParameterGrid, SearchSpace
from Backtest.scoring.batch import BatchBacktest
from Backtest.scoring.selector import Selector
from Backtest.scoring.splits import PurgedKFold, walk_forward_splits

from utils.logger import log


# --------------------------------------------------------------------------
# Execution Stats Printer
# --------------------------------------------------------------------------

def _print_execution_stats(result, config_args=None) -> None:
    """
    Print execution/fill statistics from backtest result.
    
    Only shows stats for features that were actually enabled.
    Includes: order counts, partial fills, latency, turnover, close-at-end.
    """
    meta = result.metadata or {}
    exec_stats = meta.get("execution_stats", {})
    
    # Determine what features were enabled
    latency_enabled = (config_args and getattr(config_args, 'latency_ms', 0) > 0) or meta.get("avg_latency_ns", 0) > 0
    partial_enabled = (config_args and getattr(config_args, 'partial_fills', False)) or meta.get("partial_fills_enabled", False)
    close_at_end = meta.get("close_positions_at_end", False)
    
    # Always show order/fill summary
    print("\n" + "-"*60)
    print("EXECUTION / FILL STATS")
    print("-"*60)
    
    # Order counts - include forced close orders in totals
    normal_orders = meta.get("order_count", 0)
    forced_close_orders = meta.get("forced_close_orders", 0)
    forced_close_fills = meta.get("forced_close_fills", 0)
    
    # Total orders = normal + forced close
    total_orders_submitted = normal_orders + forced_close_orders
    
    # Fill counts - exec_stats.total_fills should include forced close fills
    total_fills = meta.get("fill_count", exec_stats.get("total_fills", 0))
    
    rejected = meta.get("rejected_orders", exec_stats.get("rejected_orders", 0))
    risk_rejected = meta.get("risk_rejected_orders", 0)
    
    print(f"Orders Submitted:     {total_orders_submitted}")
    print(f"Orders Filled:        {total_fills}")
    
    # Consistency check: fills should never exceed submitted
    if total_fills > total_orders_submitted:
        print(f"  ⚠️ Counter mismatch: fills ({total_fills}) > submitted ({total_orders_submitted})")
    
    if rejected > 0:
        print(f"Orders Rejected:      {rejected} (liquidity)")
    if risk_rejected > 0:
        print(f"Orders Rejected:      {risk_rejected} (risk)")
    
    # Partial fill stats - ONLY if partial fills were enabled
    partial_fills = meta.get("partial_fills", exec_stats.get("partial_fills", 0))
    if partial_enabled and total_fills > 0:
        pct = (partial_fills / total_fills * 100) if total_fills > 0 else 0
        print(f"Partial Fills:        {partial_fills} / {total_fills} ({pct:.1f}%)")
        
        # Average fill ratio
        avg_fill_ratio = meta.get("avg_fill_ratio", exec_stats.get("avg_fill_ratio", 0))
        if avg_fill_ratio > 0:
            print(f"Avg Fill Ratio:       {avg_fill_ratio:.1%}")
        
        # Unfilled quantity
        unfilled_qty = meta.get("total_unfilled_qty", exec_stats.get("total_unfilled_qty", 0))
        if unfilled_qty > 0:
            print(f"Total Unfilled Qty:   {unfilled_qty:.6f}")
    
    # Latency stats - ONLY if latency was enabled
    avg_latency = meta.get("avg_latency_ns", exec_stats.get("avg_latency_ns", 0))
    max_latency = meta.get("max_latency_ns", exec_stats.get("max_latency_ns", 0))
    
    if latency_enabled and avg_latency > 0:
        def fmt_latency(ns):
            if ns >= 1_000_000_000:
                return f"{ns / 1e9:.2f}s"
            elif ns >= 1_000_000:
                return f"{ns / 1e6:.2f}ms"
            elif ns >= 1_000:
                return f"{ns / 1e3:.2f}µs"
            return f"{ns}ns"
        
        print(f"Avg Latency:          {fmt_latency(avg_latency)}")
        print(f"Max Latency:          {fmt_latency(max_latency)}")
    
    # Turnover (always show if available)
    turnover = getattr(result, 'turnover', 0) or 0
    if turnover > 0:
        # Compute total notional from fills
        total_notional = meta.get("total_traded_notional", 0)
        if total_notional > 0:
            print(f"Total Notional:       ${total_notional:,.2f}")
        print(f"Turnover Ratio:       {turnover:.2f}x")
    
    # Forced close stats - ONLY if enabled
    if close_at_end:
        print(f"Close At End:         Enabled")
        if forced_close_orders > 0 or forced_close_fills > 0:
            print(f"Forced Close Orders:  {forced_close_orders}")
            print(f"Forced Close Fills:   {forced_close_fills}")
    
    # Liquidation events (margin mode)
    liquidations = meta.get("liquidation_count", 0)
    if liquidations > 0:
        print(f"Liquidation Events:   {liquidations} ⚠️")
    
    print("-"*60)


# --------------------------------------------------------------------------
# Example Strategy
# --------------------------------------------------------------------------

class SimpleMomentumStrategy:
    """
    Simple momentum strategy for demonstration.
    
    Buys when price is above the moving average of closes.
    Sells when price is below the moving average.
    
    Parameters:
        lookback: Number of bars for moving average
        position_size: Size of each trade
    """
    
    def __init__(self, lookback: int = 20, position_size: float = 0.01):
        self.lookback = lookback
        self.position_size = position_size
        self.position = 0.0
    
    def on_bar(self, bar: Bar, ctx: StrategyContext) -> List[Order]:
        orders = []
        
        # Get historical closes
        ohlcv = ctx.get_ohlcv()
        closes = ohlcv.get("close", [])
        
        if len(closes) < self.lookback:
            return orders
        
        # Calculate simple moving average
        ma = sum(closes[-self.lookback:]) / self.lookback
        
        # Generate signals
        if bar.close > ma and self.position == 0:
            # Buy signal
            orders.append(Order(
                symbol=bar.symbol,
                side=OrderSide.BUY,
                order_type=OrderType.MARKET,
                quantity=self.position_size,
            ))
            self.position = self.position_size
        
        elif bar.close < ma and self.position > 0:
            # Sell signal
            orders.append(Order(
                symbol=bar.symbol,
                side=OrderSide.SELL,
                order_type=OrderType.MARKET,
                quantity=self.position,
            ))
            self.position = 0.0
        
        return orders
    
    def reset(self):
        self.position = 0.0


# --------------------------------------------------------------------------
# Data Validation
# --------------------------------------------------------------------------

def validate_data_exists(data_dir: str, symbol: str, allow_synthetic: bool = False) -> bool:
    """
    Check if tick data exists and provide clear instructions if missing.
    
    Returns:
        True if data exists (or synthetic is allowed)
        
    Raises:
        SystemExit with instructions if data missing and synthetic not allowed
    """
    tick_store = TickStore(TickStoreConfig(
        data_dir=data_dir,
        allow_synthetic=allow_synthetic,
    ))
    
    if tick_store.file_exists(symbol):
        layout = tick_store.get_storage_layout(symbol)
        log("INFO", f"Found tick data for {symbol} (layout: {layout})")
        return True
    
    if allow_synthetic:
        log("WARNING", f"No tick data for {symbol}, but --synthetic flag is set")
        return True
    
    # Data not found - provide helpful error
    print(f"""
================================================================================
ERROR: No tick data found for {symbol}
================================================================================

To run a backtest, you need historical tick data.

OPTION 1: Download real data from Binance Vision (RECOMMENDED):
    python tools/fetch_ticks.py --symbol {symbol} --start 2024-01-01 --end 2024-01-07

OPTION 2: Use synthetic data for testing ONLY (not for real analysis):
    Add --synthetic flag to this command

Expected data location: 
    {data_dir}/{symbol}/YYYY-MM-DD.csv  (partitioned format)
    OR
    {data_dir}/{symbol}_ticks.csv  (legacy format)

For more info, see docs/data_setup.md
================================================================================
""")
    sys.exit(1)


# --------------------------------------------------------------------------
# Main Functions
# --------------------------------------------------------------------------

def run_single_backtest(args):
    """Run a single backtest with default parameters."""
    log("INFO", f"Running single backtest on {args.symbol}")
    
    # Validate data exists
    validate_data_exists(args.data_dir, args.symbol, getattr(args, 'synthetic', False))
    
    leverage_mode = getattr(args, 'leverage_mode', 'spot')
    leverage = getattr(args, 'leverage', 1.0)
    close_at_end = getattr(args, 'close_positions_at_end', False)
    
    # Execution realism settings
    partial_fills = getattr(args, 'partial_fills', False)
    liquidity_scale = getattr(args, 'liquidity_scale', 10.0)
    latency_ms = getattr(args, 'latency_ms', 0.0)
    latency_jitter_ms = getattr(args, 'latency_jitter_ms', 0.0)
    spread_bps = getattr(args, 'spread_bps', 2.0)
    slippage_bps = getattr(args, 'slippage_bps', 1.0)
    
    # Convert ms to ns for internal use
    latency_ns = int(latency_ms * 1_000_000)
    latency_jitter_ns = int(latency_jitter_ms * 1_000_000)
    
    runner = create_runner(
        tick_data_dir=args.data_dir,
        symbols=[args.symbol],
        timeframe=args.timeframe,
        initial_capital=args.capital,
        random_seed=42,
        leverage_mode=leverage_mode,
        leverage=int(leverage),
        close_positions_at_end=close_at_end,
        # Execution realism
        enable_partial_fills=partial_fills,
        liquidity_scale=liquidity_scale,
        latency_ns=latency_ns,
        latency_jitter_ns=latency_jitter_ns,
        spread_bps=spread_bps,
        slippage_bps=slippage_bps,
    )
    
    params = {
        "lookback": 20,
        "position_size": 0.01,
    }
    
    result = runner.run_with_class(
        SimpleMomentumStrategy,
        params,
        strategy_name="SimpleMomentum",
    )
    
    print("\n" + "="*60)
    print("SINGLE BACKTEST RESULTS")
    print("="*60)
    print(f"Strategy:       {result.strategy_name}")
    print(f"Parameters:     {result.params}")
    print(f"Leverage Mode:  {leverage_mode}")
    if leverage_mode == "margin":
        print(f"Leverage:       {leverage}x")
    print(f"Initial:        ${result.initial_capital:,.2f}")
    print(f"Final Equity:   ${result.final_equity:,.2f}")
    
    # Compute return from displayed values for consistency check
    displayed_return_pct = result.total_return_pct  # Already a percentage
    expected_return_pct = ((result.final_equity / result.initial_capital) - 1.0) * 100.0 if result.initial_capital > 0 else 0.0
    
    # total_return_pct is ALREADY a percentage (e.g., -0.08 means -0.08%)
    # So we print it directly with f-string formatting, NOT with :.2%
    print(f"Total Return:   {displayed_return_pct:.2f}%")
    
    # Consistency check
    if abs(displayed_return_pct - expected_return_pct) > 0.01:
        print(f"  ⚠️ Return/Equity mismatch: displayed={displayed_return_pct:.4f}%, computed={expected_return_pct:.4f}%")
    
    # Max drawdown is a FRACTION (0.1 = 10%), format with %
    print(f"Max Drawdown:   {result.max_drawdown:.2%}")
    print(f"Sharpe Ratio:   {result.sharpe_ratio:.3f}")
    print(f"Total Trades:   {result.total_trades}")
    
    # Fee reporting with model info
    meta = result.metadata or {}
    fee_config = meta.get("fee_config", {})
    taker_bps = fee_config.get("taker_fee_bps", 4.0)
    maker_bps = fee_config.get("maker_fee_bps", 2.0)
    
    if result.total_fees < 0.01:
        # Small fees - show with more precision and model info
        print(f"Total Fees:     ${result.total_fees:.4f} (taker: {taker_bps:.1f} bps, maker: {maker_bps:.1f} bps)")
    else:
        print(f"Total Fees:     ${result.total_fees:,.2f} (taker: {taker_bps:.1f} bps)")
    print("="*60)
    
    # Print execution/fill stats if available
    _print_execution_stats(result, args)
    
    return result


def run_parameter_sweep(args):
    """Run parameter sweep to find optimal parameters."""
    log("INFO", f"Running parameter sweep on {args.symbol}")
    
    # Validate data exists
    validate_data_exists(args.data_dir, args.symbol, getattr(args, 'synthetic', False))
    
    leverage_mode = getattr(args, 'leverage_mode', 'spot')
    leverage = getattr(args, 'leverage', 1.0)
    
    data_config = DataConfig(
        tick_data_dir=args.data_dir,
        symbols=[args.symbol],
        timeframe=args.timeframe,
    )
    
    config = BacktestConfig(
        data=data_config,
        initial_capital=args.capital,
        random_seed=42,
        leverage_mode=leverage_mode,
        leverage=int(leverage),
    )
    
    # Define parameter search space
    search_space = SearchSpace()
    search_space.add("lookback", [10, 20, 30, 50])
    search_space.add("position_size", [0.005, 0.01, 0.02])
    
    # Add constraint: lookback must be reasonable
    search_space.add_constraint(lambda p: p["lookback"] >= 5)
    
    # Optionally add leverage constraint
    if leverage_mode == "margin":
        search_space.require_max_leverage(leverage, param_name="position_size")
    
    print(f"\nParameter Space: {search_space.info()}")
    print(f"Leverage Mode: {leverage_mode}" + (f" ({leverage}x)" if leverage_mode == "margin" else ""))
    
    def strategy_factory(params):
        return SimpleMomentumStrategy(**params)
    
    batch = BatchBacktest(config, strategy_factory)
    batch_result = batch.run(search_space)
    
    # ----- Selection Bias Report (AFML discipline) -----
    print("\n" + "="*60)
    print("SELECTION BIAS REPORT (AFML)")
    print("="*60)
    if batch_result.selection_bias_report:
        report = batch_result.selection_bias_report
        print(f"Total Trials:       {report.get('total_trials', 'N/A')}")
        print(f"Failed Trials:      {report.get('failed_trials', 0)}")
        print(f"Best Strategy:      {report.get('best_strategy', 'N/A')}")
        print(f"Best Sharpe (SR*):  {report.get('best_sharpe', float('nan')):.4f}")
        
        dsr = report.get('deflated_sharpe_ratio')
        if dsr is not None:
            print(f"Deflated SR (DSR):  {dsr:.4f}")
            if dsr < 1.0:
                print("\n⚠️  WARNING: DSR < 1.0 indicates potential selection bias!")
                print("    The best strategy's performance may be due to luck rather than skill.")
                print("    Consider: reducing # of trials, more data, out-of-sample validation.")
            else:
                print("\n✓  DSR >= 1.0 suggests robust performance after bias adjustment.")
        
        warning = report.get('selection_warning')
        if warning:
            print(f"\n{warning}")
    else:
        print("No selection bias report available.")
    print("="*60)
    
    # Select top results
    selector = Selector(
        min_sharpe=0.0,  # Relaxed for example
        max_drawdown=0.5,
        min_trades=1,
    )
    
    top_results = selector.select_top_k(batch_result, k=5, apply_filters=True)
    
    print("\n" + "="*60)
    print("PARAMETER SWEEP RESULTS - TOP 5")
    print("="*60)
    
    if not top_results:
        print("No results passed the filter criteria.")
        if batch_result.failed_count > 0:
            print(f"\nNote: {batch_result.failed_count} trials failed. Check logs for details.")
    else:
        for i, (result, score, rank) in enumerate(top_results, 1):
            print(f"\n#{i} (Rank {rank}, Score: {score:.4f})")
            print(f"  Parameters:   {result.params}")
            print(f"  Total Return: {result.total_return:.2%}")
            print(f"  Max Drawdown: {result.max_drawdown:.2%}")
            print(f"  Sharpe:       {result.sharpe_ratio:.3f}")
            print(f"  Trades:       {result.total_trades}")
    
    print("\n" + "="*60)
    
    return batch_result


def run_walk_forward(args):
    """Run walk-forward analysis."""
    log("INFO", f"Running walk-forward analysis on {args.symbol}")
    
    # Validate data exists
    validate_data_exists(args.data_dir, args.symbol, getattr(args, 'synthetic', False))
    
    # This is a simplified demonstration
    # In practice, you'd split your data and run CV
    
    print("\n" + "="*60)
    print("WALK-FORWARD ANALYSIS")
    print("="*60)
    print("This would perform:")
    print("1. Split data into train/test periods")
    print("2. Optimize on train, validate on test")
    print("3. Roll forward and repeat")
    print("4. Aggregate out-of-sample results")
    print("="*60)
    
    # Get walk-forward splits
    total_bars = 1000  # Example
    splits = walk_forward_splits(
        total_samples=total_bars,
        train_size=500,
        test_size=100,
        min_steps=3,
    )
    
    print(f"\nGenerated {len(splits)} walk-forward folds:")
    for i, (train_indices, test_indices) in enumerate(splits):
        print(f"  Fold {i+1}: Train bars {min(train_indices)}-{max(train_indices)}, "
              f"Test bars {min(test_indices)}-{max(test_indices)}")


def main():
    parser = argparse.ArgumentParser(
        description="Backtest runner CLI - Run backtests on historical tick data",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Single backtest (spot mode)
  python -m Backtest.run_backtest --data-dir ./data/ticks --symbol BTCUSDT
  
  # Single backtest with margin/leverage
  python -m Backtest.run_backtest --data-dir ./data/ticks --symbol BTCUSDT \\
      --leverage-mode margin --leverage 10
  
  # Parameter sweep with AFML selection bias report
  python -m Backtest.run_backtest --mode sweep --data-dir ./data/ticks --symbol BTCUSDT
  
  # Walk-forward analysis
  python -m Backtest.run_backtest --mode walkforward --data-dir ./data/ticks --symbol BTCUSDT
  
  # Use synthetic data for testing (NOT for real analysis)
  python -m Backtest.run_backtest --data-dir ./data/ticks --symbol TESTCOIN --synthetic

Notes:
  - Tick data must exist in data_dir/{symbol}/YYYY-MM-DD.csv (partitioned) 
    or data_dir/{symbol}_ticks.csv (legacy)
  - Use tools/fetch_ticks.py to download real Binance tick data
  - Selection bias report uses AFML Deflated Sharpe Ratio methodology
        """
    )
    
    # ----- Mode selection -----
    parser.add_argument(
        "--mode",
        choices=["single", "sweep", "walkforward"],
        default="single",
        help="Backtest mode (default: single)"
    )
    
    # ----- Data arguments -----
    parser.add_argument(
        "--data-dir",
        type=str,
        required=True,
        help="Directory containing tick CSV files"
    )
    
    parser.add_argument(
        "--symbol",
        type=str,
        default="BTCUSDT",
        help="Symbol to backtest (default: BTCUSDT)"
    )
    
    parser.add_argument(
        "--timeframe",
        type=str,
        default="1m",
        help="Bar timeframe (default: 1m)"
    )
    
    parser.add_argument(
        "--synthetic",
        action="store_true",
        help="Allow synthetic/missing data (for testing ONLY, not real analysis)"
    )
    
    # ----- Capital & Leverage -----
    parser.add_argument(
        "--capital",
        type=float,
        default=10000.0,
        help="Initial capital (default: 10000)"
    )
    
    parser.add_argument(
        "--leverage-mode",
        choices=["spot", "margin"],
        default="spot",
        help="Trading mode: spot (full cash) or margin (collateral-based)"
    )
    
    parser.add_argument(
        "--leverage",
        type=float,
        default=1.0,
        help="Leverage multiplier for margin mode (default: 1.0, max: 125)"
    )
    
    # ----- Execution Realism -----
    parser.add_argument(
        "--close-positions-at-end",
        action="store_true",
        default=False,
        help="Close all open positions at end of backtest (AFML-compliant bet accounting)"
    )
    
    parser.add_argument(
        "--partial-fills",
        action="store_true",
        default=False,
        help="Enable partial fill simulation based on market liquidity"
    )
    
    parser.add_argument(
        "--liquidity-scale",
        type=float,
        default=10.0,
        help="Liquidity multiplier for partial fill calculation (default: 10.0)"
    )
    
    parser.add_argument(
        "--latency-ms",
        type=float,
        default=0.0,
        help="Simulated order latency in milliseconds (default: 0 = instant fill)"
    )
    
    parser.add_argument(
        "--latency-jitter-ms",
        type=float,
        default=0.0,
        help="Random jitter added to latency in milliseconds (default: 0)"
    )
    
    parser.add_argument(
        "--spread-bps",
        type=float,
        default=2.0,
        help="Bid-ask spread in basis points for cost calculation (default: 2.0)"
    )
    
    parser.add_argument(
        "--slippage-bps",
        type=float,
        default=1.0,
        help="Slippage in basis points for cost calculation (default: 1.0)"
    )
    
    args = parser.parse_args()
    
    # ----- Validation -----
    
    # Validate data directory exists
    if not os.path.isdir(args.data_dir):
        print(f"Error: Data directory not found: {args.data_dir}")
        print(f"       Please create the directory or use a valid path.")
        sys.exit(1)
    
    # Validate leverage
    if args.leverage_mode == "margin":
        if args.leverage < 1.0 or args.leverage > 125:
            print(f"Error: Leverage must be between 1 and 125 (got {args.leverage})")
            sys.exit(1)
        log("INFO", f"Margin mode enabled with {args.leverage}x leverage")
    elif args.leverage != 1.0:
        log("WARNING", f"--leverage is ignored in spot mode (only applies to margin)")
    
    # Warn about synthetic data
    if args.synthetic:
        log("WARNING", "Synthetic data mode enabled - results are NOT valid for real analysis")
    
    # ----- Run appropriate mode -----
    if args.mode == "single":
        run_single_backtest(args)
    elif args.mode == "sweep":
        run_parameter_sweep(args)
    elif args.mode == "walkforward":
        run_walk_forward(args)


if __name__ == "__main__":
    main()
