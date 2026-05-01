#!/usr/bin/env python3
"""
tools/eod_rollup.py
===================
End-of-day rollup of a live-trade CSV.

Reads ``logs/live_trades_<run_id>.csv`` (the schema produced by
``LiveMetrics.record``), filters by date, and emits a single-row
summary CSV ``logs/eod_<date>_<run_id>.csv`` plus a stdout report.

Columns produced:
    date, run_id, total_trades, winning_trades, losing_trades,
    win_rate, profit_factor, total_pnl_gross, total_pnl_net,
    total_fees, total_funding, avg_slippage_bps, max_drawdown_pct,
    consecutive_losses_max, longest_hold_seconds,
    per_symbol_pnl_json

Usage::

    python tools/eod_rollup.py \\
        --trade-csv logs/live_trades_canary.csv \\
        --date 2026-05-02 \\
        --run-id canary

The script is small on purpose — pandas is intentionally avoided so
we keep dependencies thin and the rollup runnable inside the same
venv as the bot.
"""
from __future__ import annotations
import argparse
import csv
import json
import math
import os
import sys
from collections import defaultdict
from typing import Dict, List, Optional


def _parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--trade-csv", required=True,
                   help="Path to logs/live_trades_<run_id>.csv")
    p.add_argument("--date", required=True,
                   help="Date filter (YYYY-MM-DD).  Trades whose timestamp "
                        "begins with this prefix are included.")
    p.add_argument("--run-id", default="default")
    p.add_argument("--out-dir", default="logs",
                   help="Directory for the EOD CSV (default 'logs').")
    return p.parse_args(argv)


def _read_rows(path: str, date_prefix: str) -> List[Dict[str, str]]:
    if not os.path.exists(path):
        return []
    out: List[Dict[str, str]] = []
    with open(path, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            ts = (row.get("timestamp") or "").strip()
            if ts.startswith(date_prefix):
                out.append(row)
    return out


def _to_float(s, default=0.0) -> float:
    try:
        return float(s)
    except (TypeError, ValueError):
        return default


def _consecutive_losses_max(rows: List[Dict[str, str]]) -> int:
    streak = best = 0
    # Trades are in CSV order (chronological by entry exit).
    for row in rows:
        pnl = _to_float(row.get("pnl_usd"))
        if pnl < 0:
            streak += 1
            best = max(best, streak)
        else:
            streak = 0
    return best


def _max_drawdown_pct(rows: List[Dict[str, str]],
                      start_equity: float = 1_000.0) -> float:
    """Drawdown calculation from a running cumulative net P&L.

    We don't have absolute equity timestamps in the CSV, so this is
    an approximation: assume start equity = 1000 USDT and track the
    running peak of cumulative net P&L over trades.  Returns the
    worst peak-to-trough drawdown as a percent of the running peak
    equity (peak_equity = start_equity + peak_pnl).
    """
    cum = 0.0
    peak = start_equity
    worst = 0.0
    for row in rows:
        net = _to_float(row.get("pnl_net_usd")) or _to_float(row.get("pnl_usd"))
        cum += net
        equity = start_equity + cum
        peak = max(peak, equity)
        if peak > 0:
            dd = (peak - equity) / peak * 100.0
            worst = max(worst, dd)
    return worst


def aggregate(rows: List[Dict[str, str]]) -> Dict[str, object]:
    if not rows:
        return {
            "total_trades": 0, "winning_trades": 0, "losing_trades": 0,
            "win_rate": 0.0, "profit_factor": 0.0,
            "total_pnl_gross": 0.0, "total_pnl_net": 0.0,
            "total_fees": 0.0, "total_funding": 0.0,
            "avg_slippage_bps": 0.0, "max_drawdown_pct": 0.0,
            "consecutive_losses_max": 0, "longest_hold_seconds": 0.0,
            "per_symbol_pnl": {},
        }

    wins = [r for r in rows if _to_float(r.get("pnl_usd")) > 0]
    losses = [r for r in rows if _to_float(r.get("pnl_usd")) <= 0]
    win_pnl = sum(_to_float(r.get("pnl_usd")) for r in wins)
    loss_pnl = -sum(_to_float(r.get("pnl_usd")) for r in losses)
    pf = win_pnl / loss_pnl if loss_pnl > 0 else (math.inf if win_pnl > 0 else 0.0)

    total_gross = sum(
        _to_float(r.get("pnl_gross_usd")) or _to_float(r.get("pnl_usd"))
        for r in rows
    )
    total_net = sum(
        _to_float(r.get("pnl_net_usd")) or _to_float(r.get("pnl_usd"))
        for r in rows
    )
    total_fees = sum(_to_float(r.get("entry_fee_usd")) +
                     _to_float(r.get("exit_fee_usd")) for r in rows)
    total_funding = sum(_to_float(r.get("funding_usd")) for r in rows)

    slip_vals = [abs(_to_float(r.get("slippage_bps"))) for r in rows
                 if r.get("slippage_bps") not in (None, "")]
    avg_slip = (sum(slip_vals) / len(slip_vals)) if slip_vals else 0.0

    by_symbol: Dict[str, float] = defaultdict(float)
    for r in rows:
        sym = r.get("symbol") or "?"
        by_symbol[sym] += _to_float(r.get("pnl_net_usd")) or _to_float(r.get("pnl_usd"))

    longest_hold = max(
        (_to_float(r.get("hold_seconds")) for r in rows), default=0.0,
    )

    return {
        "total_trades": len(rows),
        "winning_trades": len(wins),
        "losing_trades": len(losses),
        "win_rate": (len(wins) / len(rows)) if rows else 0.0,
        "profit_factor": pf,
        "total_pnl_gross": total_gross,
        "total_pnl_net": total_net,
        "total_fees": total_fees,
        "total_funding": total_funding,
        "avg_slippage_bps": avg_slip,
        "max_drawdown_pct": _max_drawdown_pct(rows),
        "consecutive_losses_max": _consecutive_losses_max(rows),
        "longest_hold_seconds": longest_hold,
        "per_symbol_pnl": dict(by_symbol),
    }


def write_eod(out_path: str, date: str, run_id: str, summary: Dict) -> None:
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    fields = [
        "date", "run_id", "total_trades", "winning_trades", "losing_trades",
        "win_rate", "profit_factor", "total_pnl_gross", "total_pnl_net",
        "total_fees", "total_funding", "avg_slippage_bps",
        "max_drawdown_pct", "consecutive_losses_max",
        "longest_hold_seconds", "per_symbol_pnl_json",
    ]
    pf = summary["profit_factor"]
    pf_str = "inf" if pf == math.inf else f"{pf:.4f}"
    row = {
        "date": date, "run_id": run_id,
        "total_trades": summary["total_trades"],
        "winning_trades": summary["winning_trades"],
        "losing_trades": summary["losing_trades"],
        "win_rate": f"{summary['win_rate']:.4f}",
        "profit_factor": pf_str,
        "total_pnl_gross": f"{summary['total_pnl_gross']:.4f}",
        "total_pnl_net": f"{summary['total_pnl_net']:.4f}",
        "total_fees": f"{summary['total_fees']:.4f}",
        "total_funding": f"{summary['total_funding']:.4f}",
        "avg_slippage_bps": f"{summary['avg_slippage_bps']:.4f}",
        "max_drawdown_pct": f"{summary['max_drawdown_pct']:.4f}",
        "consecutive_losses_max": summary["consecutive_losses_max"],
        "longest_hold_seconds": f"{summary['longest_hold_seconds']:.1f}",
        "per_symbol_pnl_json": json.dumps(
            summary["per_symbol_pnl"], sort_keys=True,
        ),
    }
    with open(out_path, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerow(row)


def print_report(date: str, run_id: str, summary: Dict) -> None:
    print("=" * 60)
    print(f"EOD ROLLUP — {date} — run {run_id}")
    print("=" * 60)
    print(f"  trades:        {summary['total_trades']:>6d}  "
          f"(W/L {summary['winning_trades']}/{summary['losing_trades']})")
    print(f"  win rate:      {summary['win_rate']:.2%}")
    pf = summary["profit_factor"]
    print(f"  profit factor: {'inf' if pf == math.inf else f'{pf:.3f}'}")
    print(f"  pnl gross:     ${summary['total_pnl_gross']:.2f}")
    print(f"  pnl net:       ${summary['total_pnl_net']:.2f}")
    print(f"  fees:          ${summary['total_fees']:.2f}")
    print(f"  funding:       ${summary['total_funding']:.2f}")
    print(f"  avg slip:      {summary['avg_slippage_bps']:.2f} bps")
    print(f"  max drawdown:  {summary['max_drawdown_pct']:.2f}%")
    print(f"  longest loss streak: {summary['consecutive_losses_max']}")
    print(f"  longest hold:  {summary['longest_hold_seconds']:.0f}s")
    if summary["per_symbol_pnl"]:
        print("  per-symbol P&L:")
        for sym, p in sorted(summary["per_symbol_pnl"].items(),
                              key=lambda kv: kv[1], reverse=True):
            print(f"    {sym:<10} ${p:.2f}")
    print("=" * 60)


def main(argv: Optional[List[str]] = None) -> int:
    args = _parse_args(argv)
    rows = _read_rows(args.trade_csv, args.date)
    summary = aggregate(rows)
    out_path = os.path.join(
        args.out_dir, f"eod_{args.date}_{args.run_id}.csv",
    )
    write_eod(out_path, args.date, args.run_id, summary)
    print_report(args.date, args.run_id, summary)
    print(f"\nWrote: {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
