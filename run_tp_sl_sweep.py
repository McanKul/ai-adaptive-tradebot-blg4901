#!/usr/bin/env python3
"""
run_tp_sl_sweep.py
==================
Parameter sweep for optimal TP/SL + ATR multiplier values.

Tests combinations of:
- atr_mult: ATR trailing stop multiplier (strategy-level)
- tp_pct: Take profit % (engine-level ExitManager, margin-return based)
- sl_pct: Stop loss % (engine-level ExitManager, margin-return based)

Uses BatchBacktest with ParameterGrid for systematic evaluation.
Results ranked by composite score (Sharpe, drawdown, win rate).

Usage:
    python run_tp_sl_sweep.py
    python run_tp_sl_sweep.py --symbol DOGEUSDT --timeframe 15m
"""
import sys
import os
import logging
import time

sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from Backtest.engine import EngineConfig
from Backtest.runner import BacktestRunner, BacktestConfig, DataConfig
from Backtest.scoring.search_space import ParameterGrid
from Backtest.scoring.scorer import Scorer, ScorerWeights
from Interfaces.metrics_interface import BacktestResult
from Interfaces.strategy_adapter import SizingConfig, SizingMode

from Strategy.RSIThreshold import Strategy as RSIThresholdStrategy

logging.basicConfig(
    level=logging.WARNING,
    format="%(asctime)s | %(levelname)s | %(message)s",
)
log = logging.getLogger(__name__)
log.setLevel(logging.INFO)


# ── Configuration ─────────────────────────────────────────────────────
SYMBOL = "AVAXUSDT"
TIMEFRAME = "5m"
DATA_DIR = "./data/ticks"
INITIAL_CAPITAL = 10_000.0
LEVERAGE = 10
MARGIN_USD = 10.0

# ── Parameter Grid ────────────────────────────────────────────────────
# rsi_period: RSI lookback period
# rsi_overbought / rsi_oversold: RSI threshold levels
# tp_pct / sl_pct: engine-level ExitManager (margin-return %)
# None = disabled
PARAM_GRID = {
    "rsi_period": [7, 14, 21],
    "rsi_overbought": [70, 75, 80],
    "rsi_oversold": [20,25,30],
    "tp_pct": [ 0.02, 0.04, 0.06],
    "sl_pct": [ 0.01, 0.02, 0.03],
}


def run_single(params: dict) -> BacktestResult:
    """Run a single backtest with given params."""
    rsi_period = params["rsi_period"]
    rsi_ob = params["rsi_overbought"]
    rsi_os = params["rsi_oversold"]
    tp_pct = params.get("tp_pct")
    sl_pct = params.get("sl_pct")

    data_cfg = DataConfig(
        tick_data_dir=DATA_DIR,
        symbols=[SYMBOL],
        bar_type="time",
        timeframe=TIMEFRAME,
    )

    bt_cfg = BacktestConfig(
        data=data_cfg,
        initial_capital=INITIAL_CAPITAL,
        leverage_mode="margin",
        leverage=LEVERAGE,
        maintenance_margin_ratio=0.5,
        taker_fee_bps=4.0,
        maker_fee_bps=2.0,
        slippage_bps=1.0,
        spread_bps=2.0,
        max_position_size=1_000_000.0,
        max_position_notional=50_000.0,
        max_daily_loss=INITIAL_CAPITAL * 0.1,
        max_drawdown=0.5,
        enable_tick_exit=True,
        bar_store_maxlen=600,
        # TP/SL from sweep params (None = disabled)
        tp_pct=tp_pct,
        sl_pct=sl_pct,
    )

    engine_cfg = bt_cfg.to_engine_config()
    from Backtest.engine import BacktestEngine
    engine = BacktestEngine(engine_cfg)

    # Sizing config matching live_config.yaml
    sizing = SizingConfig(
        mode=SizingMode.MARGIN_USD,
        margin_usd=MARGIN_USD,
        leverage=float(LEVERAGE),
        leverage_mode="margin",
    )

    strategy = RSIThresholdStrategy(
        rsi_period=rsi_period,
        rsi_overbought=rsi_ob,
        rsi_oversold=rsi_os,
    )

    result = engine.run(strategy, sizing_config=sizing)
    result.params = params
    return result


def main():
    grid = ParameterGrid(PARAM_GRID)
    all_combos = list(grid)
    total = len(all_combos)

    log.info("=" * 70)
    log.info("RSI THRESHOLD + TP/SL PARAMETER SWEEP")
    log.info("=" * 70)
    log.info(f"Symbol: {SYMBOL} | TF: {TIMEFRAME} | Leverage: {LEVERAGE}x")
    log.info(f"Total combinations: {total}")
    log.info(f"Grid: rsi_period={PARAM_GRID['rsi_period']}")
    log.info(f"      rsi_overbought={PARAM_GRID['rsi_overbought']}")
    log.info(f"      rsi_oversold={PARAM_GRID['rsi_oversold']}")
    log.info(f"      tp_pct={PARAM_GRID['tp_pct']}")
    log.info(f"      sl_pct={PARAM_GRID['sl_pct']}")
    log.info("=" * 70)

    scorer = Scorer()
    results = []
    t0 = time.time()

    for i, params in enumerate(all_combos):
        try:
            result = run_single(params)
            score = scorer.score(result)
            results.append((params, result, score))

            tp_str = f"{params['tp_pct']*100:.0f}%" if params['tp_pct'] else "OFF"
            sl_str = f"{params['sl_pct']*100:.0f}%" if params['sl_pct'] else "OFF"

            log.info(
                "[%3d/%d] rsi(%d,%d/%d) tp=%-4s sl=%-4s | ret=%+6.1f%% sharpe=%.2f dd=%.1f%% trades=%d wr=%.0f%% | score=%.3f",
                i + 1, total,
                params["rsi_period"], params["rsi_oversold"], params["rsi_overbought"],
                tp_str, sl_str,
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

    # Print top 10
    print("\n" + "=" * 90)
    print("TOP 10 PARAMETER COMBINATIONS")
    print("=" * 90)
    print(f"{'Rank':<5} {'RSI':>4} {'OB':>4} {'OS':>4} {'TP':>6} {'SL':>6} | {'Return':>8} {'Sharpe':>7} {'MaxDD':>7} {'Trades':>7} {'WinR':>6} {'PF':>6} | {'Score':>7}")
    print("-" * 100)

    for rank, (params, result, score) in enumerate(results[:10], 1):
        tp_str = f"{params['tp_pct']*100:.0f}%" if params['tp_pct'] else "OFF"
        sl_str = f"{params['sl_pct']*100:.0f}%" if params['sl_pct'] else "OFF"
        print(
            f"{rank:<5} {params['rsi_period']:>4} {params['rsi_overbought']:>4} {params['rsi_oversold']:>4} {tp_str:>6} {sl_str:>6} | "
            f"{result.total_return_pct:>+7.1f}% {result.sharpe_ratio:>7.2f} "
            f"{result.max_drawdown*100:>6.1f}% {result.total_trades:>7} "
            f"{result.win_rate*100:>5.0f}% {result.profit_factor:>6.2f} | "
            f"{score:>7.3f}"
        )

    # Best result details
    if results:
        best_params, best_result, best_score = results[0]
        print("\n" + "=" * 90)
        print("BEST PARAMETERS")
        print("=" * 90)
        print(f"  rsi_period:     {best_params['rsi_period']}")
        print(f"  rsi_overbought: {best_params['rsi_overbought']}")
        print(f"  rsi_oversold:   {best_params['rsi_oversold']}")
        print(f"  tp_pct:         {best_params['tp_pct'] or 'None (disabled)'}")
        print(f"  sl_pct:         {best_params['sl_pct'] or 'None (disabled)'}")
        print(f"  ---")
        print(f"  Return:         {best_result.total_return_pct:+.2f}%")
        print(f"  Sharpe:         {best_result.sharpe_ratio:.3f}")
        print(f"  Max Drawdown:   {best_result.max_drawdown*100:.2f}%")
        print(f"  Total Trades:   {best_result.total_trades}")
        print(f"  Win Rate:       {best_result.win_rate*100:.1f}%")
        print(f"  Profit Factor:  {best_result.profit_factor:.3f}")
        print(f"  Score:          {best_score:.4f}")

        # Live config recommendation
        print("\n" + "-" * 100)
        print("LIVE CONFIG RECOMMENDATION (live_config.yaml)")
        print("-" * 100)
        print(f"  strategy.params.rsi_period: {best_params['rsi_period']}")
        print(f"  strategy.params.rsi_overbought: {best_params['rsi_overbought']}")
        print(f"  strategy.params.rsi_oversold: {best_params['rsi_oversold']}")
        if best_params['tp_pct']:
            print(f"  exit.take_profit_pct: {best_params['tp_pct']}")
        else:
            print(f"  exit.take_profit_pct: null")
        if best_params['sl_pct']:
            print(f"  exit.stop_loss_pct: {best_params['sl_pct']}")
        else:
            print(f"  exit.stop_loss_pct: null")

    print(f"\nTotal time: {elapsed:.1f}s ({elapsed/total:.1f}s per run)")
    print("=" * 100)


if __name__ == "__main__":
    main()
