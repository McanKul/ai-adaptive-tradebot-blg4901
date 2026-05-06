#!/usr/bin/env python3
"""
tools/promote_to_live.py
========================
End-to-end promotion gate.  Refuses to flip the canary to real money
unless every numeric threshold passes.

Steps (each must succeed; first failure aborts the chain):

    1) ``app.py validate --config <yaml> --real-money``
    2) ``app.py backtest --strategy <s> --symbol <sym> ... --export-trades``
    3) ``app.py dry-run`` is the user's job (cannot be automated here);
       this script expects the resulting CSV at ``--live-trades``
    4) ``compare_backtest_live --strict`` over the dry-run window
    5) ``eod_rollup`` over the dry-run window — profit_factor / MDD
       thresholds checked

Use::

    python tools/promote_to_live.py \\
        --config config/profiles/canary.yaml \\
        --strategy EMACrossMACDTrend --symbol BTCUSDT --timeframe 15m \\
        --realism-config example_realism_config.yaml \\
        --live-trades logs/live_trades_canary.csv \\
        --date 2026-05-02 \\
        --window 2026-05-02T00:00,2026-05-03T00:00

Exit codes::

    0  every gate passed — bot is cleared for ``app.py live``
    1  validation failed
    2  backtest produced no trades
    3  divergence gate failed
    4  EOD rollup thresholds breached

Each failure prints the exact reasons and the command that produced
them so the user can fix one thing at a time.
"""
from __future__ import annotations
import argparse
import json
import math
import os
import subprocess
import sys
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional


def write_promotion_stamp(
    out_dir: str,
    run_id: str,
    *,
    config: str,
    strategy: str,
    symbol: str,
    timeframe: str,
    live_trades: str,
    backtest_export: str,
    eod_rollup: str,
    thresholds: Dict[str, Any],
) -> str:
    """Write a machine-readable promotion-gate stamp.

    `app.py live` refuses to start without a stamp at
    ``{out_dir}/promotion_gate_{run_id}.json`` (unless the user passes
    ``--force-live``).  The stamp records which artefacts cleared which
    thresholds so the live launch is auditable after the fact.
    """
    os.makedirs(out_dir, exist_ok=True)
    stamp = {
        "run_id": run_id,
        "config": config,
        "strategy": strategy,
        "symbol": symbol,
        "timeframe": timeframe,
        "passed_at_utc": datetime.now(timezone.utc).isoformat(),
        "live_trades": live_trades,
        "backtest_export": backtest_export,
        "eod_rollup": eod_rollup,
        "thresholds": thresholds,
    }
    path = os.path.join(out_dir, f"promotion_gate_{run_id}.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(stamp, f, indent=2, sort_keys=True)
    return path


def _run(cmd: List[str], step: str) -> int:
    print("=" * 70)
    print(f"STEP: {step}")
    print(f"  $ {' '.join(cmd)}")
    print("-" * 70)
    rc = subprocess.call(cmd)
    print(f"  -> exit code {rc}")
    return rc


def _parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", required=True,
                   help="Live config YAML (e.g. config/profiles/canary.yaml)")
    p.add_argument("--strategy", required=True)
    p.add_argument("--symbol", required=True)
    p.add_argument("--timeframe", default="15m")
    p.add_argument("--strategy-params", default=None,
                   help="JSON string passed through to backtest")
    p.add_argument("--realism-config", default=None)
    p.add_argument("--data-dir", default="./data/ticks")

    # Live-side artefacts (the dry-run is the user's job; we just gate)
    p.add_argument("--live-trades", required=True,
                   help="LiveMetrics CSV from the dry-run")
    p.add_argument("--date", required=True,
                   help="Date prefix for EOD rollup (YYYY-MM-DD)")
    p.add_argument("--run-id", default="canary")
    p.add_argument("--window", default=None,
                   help="ISO 'start,end' for the divergence harness")

    # Gate thresholds
    p.add_argument("--match-rate", type=float, default=0.80)
    p.add_argument("--max-side-mismatch", type=int, default=0)
    p.add_argument("--max-pnl-diff-avg", type=float, default=1.0)
    p.add_argument("--require-min-matched", type=int, default=10)
    p.add_argument("--min-profit-factor", type=float, default=1.10)
    p.add_argument("--max-drawdown-pct", type=float, default=3.0)

    p.add_argument("--out-dir", default="logs",
                   help="Where to drop intermediate JSON / CSV artefacts")
    return p.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> int:
    args = _parse_args(argv)
    os.makedirs(args.out_dir, exist_ok=True)
    py = sys.executable

    # Step 1 — validate config (real-money mode)
    rc = _run(
        [py, "app.py", "validate", "--config", args.config, "--real-money"],
        step="validate config (real-money strict)",
    )
    if rc != 0:
        print("\nGATE FAIL [step 1]: config does not satisfy real-money rules.")
        return 1

    # Step 2 — backtest with --export-trades
    bt_json = os.path.join(args.out_dir, f"backtest_{args.run_id}.json")
    bt_cmd = [
        py, "app.py", "backtest",
        "--strategy", args.strategy,
        "--symbol", args.symbol,
        "--timeframe", args.timeframe,
        "--data-dir", args.data_dir,
        "--export-trades", bt_json,
    ]
    if args.strategy_params:
        bt_cmd.extend(["--strategy-params", args.strategy_params])
    if args.realism_config:
        bt_cmd.extend(["--realism-config", args.realism_config])
    rc = _run(bt_cmd, step="backtest + export trades")
    if rc != 0:
        print("\nGATE FAIL [step 2]: backtest CLI failed.")
        return 2
    if not os.path.exists(bt_json):
        print(f"\nGATE FAIL [step 2]: backtest did not write {bt_json}.")
        return 2
    try:
        with open(bt_json, "r", encoding="utf-8") as f:
            payload = json.load(f)
        n_trades = len(payload.get("trades") or [])
    except Exception as e:
        print(f"\nGATE FAIL [step 2]: cannot read {bt_json}: {e}")
        return 2
    if n_trades == 0:
        print("\nGATE FAIL [step 2]: backtest produced 0 trades — "
              "tune parameters or extend tick data before promotion.")
        return 2
    print(f"\n  backtest exported {n_trades} trades → {bt_json}")

    # Step 3 — divergence harness in strict mode
    div_cmd = [
        py, "tools/compare_backtest_live.py",
        "--backtest-trades", bt_json,
        "--live-trades", args.live_trades,
        "--strict",
        "--require-match-rate", str(args.match_rate),
        "--max-side-mismatch", str(args.max_side_mismatch),
        "--max-pnl-diff-avg", str(args.max_pnl_diff_avg),
        "--require-min-matched", str(args.require_min_matched),
        "--by-symbol",
    ]
    if args.window:
        div_cmd.extend(["--time-window", args.window])
    rc = _run(div_cmd, step="divergence gate (strict)")
    if rc != 0:
        print("\nGATE FAIL [step 3]: backtest-vs-live divergence too "
              "large to promote.")
        return 3

    # Step 4 — EOD rollup + numeric thresholds
    eod_csv = os.path.join(args.out_dir, f"eod_{args.date}_{args.run_id}.csv")
    rc = _run(
        [py, "tools/eod_rollup.py",
         "--trade-csv", args.live_trades,
         "--date", args.date,
         "--run-id", args.run_id,
         "--out-dir", args.out_dir],
        step="EOD rollup",
    )
    if rc != 0 or not os.path.exists(eod_csv):
        print(f"\nGATE FAIL [step 4]: EOD rollup did not produce {eod_csv}.")
        return 4

    pf, mdd = _read_eod_metrics(eod_csv)
    fail = []
    if pf < args.min_profit_factor:
        fail.append(
            f"profit_factor {pf:.3f} < required {args.min_profit_factor:.2f}"
        )
    if mdd > args.max_drawdown_pct:
        fail.append(
            f"max_drawdown_pct {mdd:.2f}% > allowed {args.max_drawdown_pct:.2f}%"
        )
    if fail:
        print("\nGATE FAIL [step 4]: EOD thresholds breached:")
        for f in fail:
            print(f"  - {f}")
        return 4

    stamp_path = write_promotion_stamp(
        args.out_dir,
        args.run_id,
        config=args.config,
        strategy=args.strategy,
        symbol=args.symbol,
        timeframe=args.timeframe,
        live_trades=args.live_trades,
        backtest_export=bt_json,
        eod_rollup=eod_csv,
        thresholds={
            "match_rate": args.match_rate,
            "max_side_mismatch": args.max_side_mismatch,
            "max_pnl_diff_avg": args.max_pnl_diff_avg,
            "min_profit_factor": args.min_profit_factor,
            "max_drawdown_pct": args.max_drawdown_pct,
        },
    )

    print("\n" + "=" * 70)
    print("GATE PASS — every check cleared.  Bot is cleared for live.")
    print("=" * 70)
    print(
        f"  artefacts:\n"
        f"    backtest export   : {bt_json}\n"
        f"    EOD rollup        : {eod_csv}\n"
        f"    promotion stamp   : {stamp_path}\n"
    )
    return 0


def _read_eod_metrics(path: str) -> tuple[float, float]:
    """Pull profit_factor and max_drawdown_pct from the EOD rollup CSV."""
    import csv
    with open(path, "r", encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        return 0.0, math.inf
    row = rows[0]
    raw_pf = (row.get("profit_factor") or "0").strip()
    pf = math.inf if raw_pf == "inf" else float(raw_pf or 0.0)
    mdd = float(row.get("max_drawdown_pct") or 0.0)
    return pf, mdd


if __name__ == "__main__":
    sys.exit(main())
