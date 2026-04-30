#!/usr/bin/env python3
"""
tools/compare_dry_runs.py
=========================
Side-by-side comparison of two dry-run trade logs.

Designed for the thesis A/B demo where one dry-run has sentiment ON
and the other has sentiment OFF, both fed by the same live market.
Reads the two CSVs produced by ``LiveMetrics`` (schema: timestamp,
symbol, side, entry_price, exit_price, qty, pnl_usd, pnl_pct,
bars_held, hold_seconds, exit_type, strategy) and prints aggregate
metrics + a delta column.

Usage:
    python tools/compare_dry_runs.py \\
        --baseline logs/live_trades_sentiment_off.csv \\
        --variant  logs/live_trades_sentiment_on.csv \\
        --baseline-label "no sentiment" \\
        --variant-label  "sentiment on"

If you also want a per-symbol breakdown::

    python tools/compare_dry_runs.py ... --by-symbol
"""
from __future__ import annotations
import argparse
import csv
import math
import os
import sys
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Dict, List, Optional


@dataclass
class RunStats:
    label: str
    csv_path: str
    n_trades: int = 0
    n_wins: int = 0
    n_losses: int = 0
    pnl_total: float = 0.0
    pnl_best: float = -math.inf
    pnl_worst: float = math.inf
    avg_hold_seconds: float = 0.0
    by_exit: Dict[str, int] = field(default_factory=lambda: defaultdict(int))
    by_symbol_pnl: Dict[str, float] = field(default_factory=lambda: defaultdict(float))
    _hold_sum: float = 0.0

    @property
    def win_rate(self) -> float:
        return self.n_wins / self.n_trades if self.n_trades else 0.0

    @property
    def avg_pnl(self) -> float:
        return self.pnl_total / self.n_trades if self.n_trades else 0.0

    @property
    def profit_factor(self) -> float:
        wins = sum(p for p in self._wins_list if p > 0)
        losses = -sum(p for p in self._losses_list if p < 0)
        return wins / losses if losses > 0 else math.inf if wins else 0.0

    _wins_list: List[float] = field(default_factory=list)
    _losses_list: List[float] = field(default_factory=list)


def _read_run(path: str, label: str) -> RunStats:
    s = RunStats(label=label, csv_path=path)
    if not os.path.exists(path):
        return s
    with open(path, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                pnl = float(row["pnl_usd"])
                hold = float(row.get("hold_seconds") or 0.0)
            except (ValueError, KeyError):
                continue
            s.n_trades += 1
            s.pnl_total += pnl
            s._hold_sum += hold
            if pnl > 0:
                s.n_wins += 1
                s._wins_list.append(pnl)
            else:
                s.n_losses += 1
                s._losses_list.append(pnl)
            s.pnl_best = max(s.pnl_best, pnl)
            s.pnl_worst = min(s.pnl_worst, pnl)
            s.by_exit[row.get("exit_type", "UNKNOWN")] += 1
            s.by_symbol_pnl[row.get("symbol", "?")] += pnl
    if s.n_trades:
        s.avg_hold_seconds = s._hold_sum / s.n_trades
    if s.pnl_best == -math.inf:
        s.pnl_best = 0.0
    if s.pnl_worst == math.inf:
        s.pnl_worst = 0.0
    return s


def _delta(a: float, b: float) -> str:
    """Pretty-print delta b - a with sign."""
    d = b - a
    sign = "+" if d >= 0 else ""
    return f"{sign}{d:.4f}"


def _row(label: str, a, b, fmt="{:.4f}", delta_pct: bool = False) -> str:
    sa = fmt.format(a) if isinstance(a, (int, float)) else str(a)
    sb = fmt.format(b) if isinstance(b, (int, float)) else str(b)
    if isinstance(a, (int, float)) and isinstance(b, (int, float)):
        if delta_pct and a:
            d_str = f"{(b - a) / abs(a) * 100:+.2f}%"
        else:
            d_str = _delta(a, b)
    else:
        d_str = "-"
    return f"  {label:<26} {sa:>14} {sb:>14} {d_str:>14}"


def _print_summary(a: RunStats, b: RunStats, by_symbol: bool) -> None:
    width = 70
    print("=" * width)
    print(f"DRY-RUN A/B COMPARISON")
    print("=" * width)
    print(f"  {'metric':<26} {a.label:>14} {b.label:>14} {'delta':>14}")
    print("-" * width)
    print(_row("trades", a.n_trades, b.n_trades, "{:d}"))
    print(_row("wins", a.n_wins, b.n_wins, "{:d}"))
    print(_row("losses", a.n_losses, b.n_losses, "{:d}"))
    print(_row("win_rate", a.win_rate, b.win_rate, "{:.2%}"))
    print(_row("total_pnl_usd", a.pnl_total, b.pnl_total, "${:.2f}"))
    print(_row("avg_pnl_usd", a.avg_pnl, b.avg_pnl, "${:.4f}"))
    print(_row("best_trade", a.pnl_best, b.pnl_best, "${:.2f}"))
    print(_row("worst_trade", a.pnl_worst, b.pnl_worst, "${:.2f}"))
    print(_row("profit_factor", a.profit_factor, b.profit_factor, "{:.3f}"))
    print(_row("avg_hold_seconds", a.avg_hold_seconds, b.avg_hold_seconds, "{:.1f}"))
    print("=" * width)
    # Exit-type breakdown
    exits = sorted(set(list(a.by_exit) + list(b.by_exit)))
    if exits:
        print("\nEXIT TYPE COUNTS")
        for et in exits:
            print(_row(f"  {et}", a.by_exit.get(et, 0), b.by_exit.get(et, 0), "{:d}"))
    if by_symbol:
        symbols = sorted(set(list(a.by_symbol_pnl) + list(b.by_symbol_pnl)))
        if symbols:
            print("\nPER-SYMBOL P&L (USD)")
            for sym in symbols:
                print(_row(f"  {sym}",
                           a.by_symbol_pnl.get(sym, 0.0),
                           b.by_symbol_pnl.get(sym, 0.0),
                           "${:.2f}"))
    print("=" * width)
    # Sentiment-impact note
    n_diff = b.n_trades - a.n_trades
    pnl_diff = b.pnl_total - a.pnl_total
    print(f"\nVariant fired {n_diff:+d} trade(s) and produced "
          f"{pnl_diff:+.2f} USD vs baseline.")
    if a.n_trades and b.n_trades:
        wr_diff_pp = (b.win_rate - a.win_rate) * 100
        print(f"Win-rate delta: {wr_diff_pp:+.2f} pp")
    print()


def main(argv: Optional[List[str]] = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--baseline", required=True,
                   help="Baseline CSV (e.g. sentiment OFF)")
    p.add_argument("--variant", required=True,
                   help="Variant CSV (e.g. sentiment ON)")
    p.add_argument("--baseline-label", default="baseline")
    p.add_argument("--variant-label", default="variant")
    p.add_argument("--by-symbol", action="store_true",
                   help="Include per-symbol P&L breakdown")
    args = p.parse_args(argv)

    a = _read_run(args.baseline, args.baseline_label)
    b = _read_run(args.variant, args.variant_label)

    if a.n_trades == 0 and b.n_trades == 0:
        print("Both files contain no trades. Nothing to compare.",
              file=sys.stderr)
        return 1

    _print_summary(a, b, by_symbol=args.by_symbol)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
