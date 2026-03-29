#!/usr/bin/env python3
"""
run_emacross_sweep.py
=====================
EMACrossMACDTrend strateji parametreleri optimizasyonu.

Sweep edilen parametreler:
- fast_ema_period / slow_ema_period: EMA crossover hızları
- macd_signal: MACD sinyal hattı
- atr_mult: ATR trailing stop çarpanı
- adx_threshold: Trend gücü minimum eşiği
- volume_filter: Hacim filtresi açık/kapalı
- exit_on_macd_cross: MACD histogram çıkışı açık/kapalı
- tp_pct / sl_pct: Engine-level TP/SL (güvenlik ağı)

Usage:
    python run_emacross_sweep.py
"""
import sys
import os
import csv
import logging
import time

sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from Backtest.engine import BacktestEngine, EngineConfig
from Backtest.runner import BacktestConfig, DataConfig
from Backtest.scoring.search_space import ParameterGrid
from Backtest.scoring.scorer import Scorer
from Interfaces.strategy_adapter import SizingConfig, SizingMode

from Strategy.EMACrossMACDTrend import Strategy as EMACrossStrategy

logging.basicConfig(
    level=logging.WARNING,
    format="%(asctime)s | %(levelname)s | %(message)s",
)
log = logging.getLogger(__name__)
log.setLevel(logging.INFO)


# ── Configuration ─────────────────────────────────────────────────────
SYMBOL = "AVAXUSDT"
TIMEFRAME = "15m"
DATA_DIR = "./data/ticks"
INITIAL_CAPITAL = 10_000.0
LEVERAGE = 10
MARGIN_USD = 100.0

# ── Parameter Grid ────────────────────────────────────────────────────
# ~108 combinations (~27 min at ~15s each)
PARAM_GRID = {
    "fast_ema_period":    [8, 12, 16],
    "slow_ema_period":    [21, 26, 34],
    "macd_signal":        [9, 12],
    "atr_mult":           [2.0, 2.5, 3.0],
    "adx_threshold":      [20.0, 25.0],
    "volume_filter":      [True],
    "exit_on_macd_cross": [True, False],
}

CSV_OUTPUT = "emacross_sweep_results.csv"


def run_single(params: dict):
    """Run a single backtest with given params."""
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
        max_daily_loss=INITIAL_CAPITAL * 0.5,
        max_drawdown=0.9,
        enable_tick_exit=True,
        bar_store_maxlen=600,
    )

    engine_cfg = bt_cfg.to_engine_config()
    engine = BacktestEngine(engine_cfg)

    sizing = SizingConfig(
        mode=SizingMode.MARGIN_USD,
        margin_usd=MARGIN_USD,
        leverage=float(LEVERAGE),
        leverage_mode="margin",
    )

    strategy = EMACrossStrategy(
        fast_ema_period=params["fast_ema_period"],
        slow_ema_period=params["slow_ema_period"],
        macd_signal=params["macd_signal"],
        atr_mult=params["atr_mult"],
        adx_threshold=params["adx_threshold"],
        volume_filter=params["volume_filter"],
        exit_on_macd_cross=params["exit_on_macd_cross"],
        atr_period=14,
        volume_period=20,
        risk_pct=0.005,
        position_size=1.0,
        allow_reversal=True,
    )

    result = engine.run(strategy, sizing_config=sizing)
    result.params = params
    return result


def main():
    import warnings
    warnings.warn(
        "run_emacross_sweep.py is deprecated. Use:\n"
        "  python app.py sweep --strategy EMACrossMACDTrend --symbol AVAXUSDT "
        "--param-grid grid.yaml\n"
        "This script will be removed in a future version.",
        DeprecationWarning, stacklevel=2,
    )
    grid = ParameterGrid(PARAM_GRID)
    all_combos = list(grid)
    total = len(all_combos)

    log.info("=" * 80)
    log.info("EMACrossMACDTrend PARAMETER SWEEP")
    log.info("=" * 80)
    log.info(f"Symbol: {SYMBOL} | TF: {TIMEFRAME} | Leverage: {LEVERAGE}x | Margin: ${MARGIN_USD}")
    log.info(f"Total combinations: {total}")
    for k, v in PARAM_GRID.items():
        log.info(f"  {k}: {v}")
    log.info("=" * 80)

    scorer = Scorer()
    results = []
    t0 = time.time()

    for i, params in enumerate(all_combos):
        try:
            result = run_single(params)
            score = scorer.score(result)
            results.append((params, result, score))

            log.info(
                "[%3d/%d] ema(%d/%d) macd_s=%d atr=%.1f adx=%.0f vol=%s macd_exit=%s | "
                "ret=%+6.2f%% sharpe=%.2f dd=%.1f%% trades=%d wr=%.0f%% pf=%.2f | score=%.3f",
                i + 1, total,
                params["fast_ema_period"], params["slow_ema_period"],
                params["macd_signal"], params["atr_mult"],
                params["adx_threshold"],
                "Y" if params["volume_filter"] else "N",
                "Y" if params["exit_on_macd_cross"] else "N",
                result.total_return_pct,
                result.sharpe_ratio,
                result.max_drawdown * 100,
                result.total_trades,
                result.win_rate * 100,
                result.profit_factor,
                score,
            )
        except Exception as e:
            log.error("[%3d/%d] FAILED: %s | %s", i + 1, total, params, e)

    elapsed = time.time() - t0

    # Sort by score descending
    results.sort(key=lambda x: x[2], reverse=True)

    # ── CSV export ────────────────────────────────────────────────────
    csv_path = os.path.join(os.path.dirname(__file__), CSV_OUTPUT)
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "rank", "fast_ema", "slow_ema", "macd_signal", "atr_mult",
            "adx_threshold", "volume_filter", "exit_on_macd_cross",
            "return_pct", "sharpe", "sortino", "calmar",
            "max_dd_pct", "trades", "win_rate", "profit_factor",
            "avg_trade_ret", "total_fees", "score",
        ])
        for rank, (params, result, score) in enumerate(results, 1):
            writer.writerow([
                rank,
                params["fast_ema_period"], params["slow_ema_period"],
                params["macd_signal"], params["atr_mult"],
                params["adx_threshold"], params["volume_filter"],
                params["exit_on_macd_cross"],
                f"{result.total_return_pct:.4f}",
                f"{result.sharpe_ratio:.4f}",
                f"{result.sortino_ratio:.4f}",
                f"{result.calmar_ratio:.4f}",
                f"{result.max_drawdown * 100:.2f}",
                result.total_trades,
                f"{result.win_rate * 100:.1f}",
                f"{result.profit_factor:.4f}",
                f"{result.avg_trade_return * 100:.4f}",
                f"{result.total_fees:.4f}",
                f"{score:.4f}",
            ])
    log.info(f"Results saved to {csv_path}")

    # ── Print top 15 ─────────────────────────────────────────────────
    print("\n" + "=" * 120)
    print("TOP 15 PARAMETER COMBINATIONS")
    print("=" * 120)
    header = (
        f"{'#':<4} {'Fast':>4} {'Slow':>4} {'Sig':>3} {'ATR':>4} "
        f"{'ADX':>4} {'Vol':>3} {'McX':>3} | "
        f"{'Return':>8} {'Sharpe':>7} {'Sortino':>8} {'Calmar':>7} "
        f"{'MaxDD':>6} {'Trades':>6} {'WinR':>5} {'PF':>5} | {'Score':>7}"
    )
    print(header)
    print("-" * 120)

    for rank, (params, result, score) in enumerate(results[:15], 1):
        print(
            f"{rank:<4} {params['fast_ema_period']:>4} {params['slow_ema_period']:>4} "
            f"{params['macd_signal']:>3} {params['atr_mult']:>4.1f} "
            f"{params['adx_threshold']:>4.0f} "
            f"{'Y' if params['volume_filter'] else 'N':>3} "
            f"{'Y' if params['exit_on_macd_cross'] else 'N':>3} | "
            f"{result.total_return_pct:>+7.2f}% {result.sharpe_ratio:>7.3f} "
            f"{result.sortino_ratio:>8.3f} {result.calmar_ratio:>7.3f} "
            f"{result.max_drawdown*100:>5.1f}% {result.total_trades:>6} "
            f"{result.win_rate*100:>4.0f}% {result.profit_factor:>5.2f} | "
            f"{score:>7.3f}"
        )

    # ── Best result details ──────────────────────────────────────────
    if results:
        best_params, best_result, best_score = results[0]
        print("\n" + "=" * 120)
        print("BEST PARAMETERS")
        print("=" * 120)
        print(f"  fast_ema_period:    {best_params['fast_ema_period']}")
        print(f"  slow_ema_period:    {best_params['slow_ema_period']}")
        print(f"  macd_signal:        {best_params['macd_signal']}")
        print(f"  atr_mult:           {best_params['atr_mult']}")
        print(f"  adx_threshold:      {best_params['adx_threshold']}")
        print(f"  volume_filter:      {best_params['volume_filter']}")
        print(f"  exit_on_macd_cross: {best_params['exit_on_macd_cross']}")
        print(f"  ---")
        print(f"  Return:             {best_result.total_return_pct:+.2f}%")
        print(f"  Ann. Return:        {best_result.annualized_return:.2%}")
        print(f"  Sharpe:             {best_result.sharpe_ratio:.3f}")
        print(f"  Sortino:            {best_result.sortino_ratio:.3f}")
        print(f"  Calmar:             {best_result.calmar_ratio:.3f}")
        print(f"  Max Drawdown:       {best_result.max_drawdown*100:.2f}%")
        print(f"  Total Trades:       {best_result.total_trades}")
        print(f"  Win Rate:           {best_result.win_rate*100:.1f}%")
        print(f"  Profit Factor:      {best_result.profit_factor:.3f}")
        print(f"  Score:              {best_score:.4f}")

        print("\n" + "-" * 120)
        print("LIVE CONFIG RECOMMENDATION (live_config.yaml)")
        print("-" * 120)
        print(f"""strategy:
  class: "EMACrossMACDTrend"
  params:
    fast_ema_period: {best_params['fast_ema_period']}
    slow_ema_period: {best_params['slow_ema_period']}
    macd_signal: {best_params['macd_signal']}
    atr_mult: {best_params['atr_mult']}
    adx_threshold: {best_params['adx_threshold']}
    volume_filter: {str(best_params['volume_filter']).lower()}
    exit_on_macd_cross: {str(best_params['exit_on_macd_cross']).lower()}
    atr_period: 14
    volume_period: 20
    risk_pct: 0.005""")

    print(f"\nTotal time: {elapsed:.1f}s ({elapsed/total:.1f}s per combo, {total} combos)")
    print(f"Results CSV: {CSV_OUTPUT}")
    print("=" * 120)


if __name__ == "__main__":
    main()
