#!/usr/bin/env python3
"""
tools/compare_backtest_live.py
==============================
Backtest-vs-live divergence harness.

The promotion gate before real money: pick a window where the bot ran
in dry-run/live, replay that exact window through the backtest engine
with the same strategy/params/realism YAML, then run this script to
quantify how much the two execution paths agree.

Discrepancy signals are blunt:

* low **match_rate** ⇒ live takes trades the backtest doesn't (or vice
  versa); execution model wrong, news/sentiment toggle different, or
  data feed lagging.
* large **avg_pnl_diff** on matched trades ⇒ slippage / latency model
  mis-tuned in backtest; tighten the realism YAML.
* nonzero **side_mismatch_count** ⇒ catastrophic — same bar, opposite
  decisions.  Strategy state-machine bug or non-determinism.
* skewed **exit_type histogram** ⇒ live tick-exits / liquidation guard
  fire while backtest closes on bar boundaries (or vice versa).

Inputs come from two sources:

* **Backtest**: a JSON file containing ``BacktestResult.trades`` (a
  list of dicts with the round-trip schema produced by
  ``Backtest/metrics.py:_close_bet``) — produced by ``--export-trades``
  on a backtest run, or by hand-dumping the python object.
* **Live**: a CSV file written by ``LiveMetrics`` — typically
  ``logs/live_trades_<run_id>.csv``.

Usage::

    python tools/compare_backtest_live.py \\
        --backtest-trades logs/backtest_2026_04_15.json \\
        --live-trades     logs/live_trades_sentiment_off.csv \\
        --time-window     2026-04-15T00:00,2026-04-15T08:00 \\
        --match-tolerance 60 \\
        --by-symbol \\
        --json-out        logs/divergence.json
"""
from __future__ import annotations
import argparse
import csv
import json
import math
import os
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Dict, List, Optional, Tuple


# ---------------------------------------------------------------------------
# Canonical trade schema
# ---------------------------------------------------------------------------

@dataclass
class CanonicalTrade:
    """Source-agnostic trade record used by the diff engine.

    All timestamps in seconds (float).  ``side`` is ``"LONG"`` /
    ``"SHORT"`` regardless of how the source labelled it.
    """
    source: str          # "backtest" | "live"
    symbol: str
    side: str            # "LONG" | "SHORT"
    entry_ts: float
    exit_ts: float
    entry_price: float
    exit_price: float
    quantity: float
    pnl: float
    exit_type: str = "UNKNOWN"


# ---------------------------------------------------------------------------
# Loaders
# ---------------------------------------------------------------------------

def _normalize_side(raw: str) -> str:
    raw = (raw or "").upper().strip()
    if raw in ("LONG", "BUY", "+1"):
        return "LONG"
    if raw in ("SHORT", "SELL", "-1"):
        return "SHORT"
    return raw or "UNKNOWN"


def load_backtest_trades(path: str) -> List[CanonicalTrade]:
    """Load round-trip trades produced by ``Backtest/metrics.py``.

    Two on-disk shapes are accepted:
    * a JSON array of trade dicts;
    * a JSON object with a ``trades`` field (the shape
      ``BacktestResult.to_dict()`` produces).
    """
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if isinstance(data, dict):
        records = data.get("trades", [])
    elif isinstance(data, list):
        records = data
    else:
        raise ValueError(f"unexpected backtest-trades JSON shape in {path}")

    out: List[CanonicalTrade] = []
    for r in records:
        try:
            entry_ts_ns = int(r["entry_timestamp_ns"])
            exit_ts_ns = int(r["exit_timestamp_ns"])
            out.append(CanonicalTrade(
                source="backtest",
                symbol=r.get("symbol", "?"),
                side=_normalize_side(r.get("entry_side", "")),
                entry_ts=entry_ts_ns / 1e9,
                exit_ts=exit_ts_ns / 1e9,
                entry_price=float(r.get("entry_price", 0.0)),
                exit_price=float(r.get("exit_price", 0.0)),
                quantity=float(r.get("quantity", 0.0)),
                pnl=float(r.get("pnl", 0.0)),
                exit_type=r.get("exit_type", "UNKNOWN"),
            ))
        except (KeyError, TypeError, ValueError):
            continue
    return out


def load_live_trades(path: str) -> List[CanonicalTrade]:
    """Load trades from a ``LiveMetrics`` CSV.

    Columns: timestamp,symbol,side,entry_price,exit_price,qty,pnl_usd,
    pnl_pct,bars_held,hold_seconds,exit_type,strategy.
    The CSV records only the **exit** wall-clock; we recover the entry
    timestamp by subtracting ``hold_seconds`` so matching can use the
    entry side, which is the most robust anchor.
    """
    out: List[CanonicalTrade] = []
    if not os.path.exists(path):
        return out
    with open(path, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                exit_ts = _parse_datetime(row.get("timestamp", ""))
                hold = float(row.get("hold_seconds") or 0.0)
                entry_ts = exit_ts - hold
                out.append(CanonicalTrade(
                    source="live",
                    symbol=row.get("symbol", "?"),
                    side=_normalize_side(row.get("side", "")),
                    entry_ts=entry_ts,
                    exit_ts=exit_ts,
                    entry_price=float(row.get("entry_price") or 0.0),
                    exit_price=float(row.get("exit_price") or 0.0),
                    quantity=float(row.get("qty") or 0.0),
                    pnl=float(row.get("pnl_usd") or 0.0),
                    exit_type=row.get("exit_type", "UNKNOWN"),
                ))
            except (ValueError, TypeError):
                continue
    return out


def _parse_datetime(s: str) -> float:
    """Parse the ``LiveMetrics`` exit timestamp ('YYYY-MM-DD HH:MM:SS').

    Returns POSIX seconds.  Falls back to 0 on parse failure so the row
    is still loaded (it just won't match anything in the time window).
    """
    s = s.strip()
    for fmt in ("%Y-%m-%d %H:%M:%S", "%Y-%m-%dT%H:%M:%S"):
        try:
            return datetime.strptime(s, fmt).replace(
                tzinfo=timezone.utc,
            ).timestamp()
        except ValueError:
            pass
    try:
        return float(s)
    except ValueError:
        return 0.0


# ---------------------------------------------------------------------------
# Diff engine
# ---------------------------------------------------------------------------

@dataclass
class MatchResult:
    matched: List[Tuple[CanonicalTrade, CanonicalTrade]] = field(default_factory=list)
    only_backtest: List[CanonicalTrade] = field(default_factory=list)
    only_live: List[CanonicalTrade] = field(default_factory=list)


def filter_window(
    trades: List[CanonicalTrade],
    start_ts: Optional[float],
    end_ts: Optional[float],
) -> List[CanonicalTrade]:
    if start_ts is None and end_ts is None:
        return trades
    out = []
    for t in trades:
        if start_ts is not None and t.entry_ts < start_ts:
            continue
        if end_ts is not None and t.entry_ts > end_ts:
            continue
        out.append(t)
    return out


def match_trades(
    backtest: List[CanonicalTrade],
    live: List[CanonicalTrade],
    tolerance_seconds: float,
) -> MatchResult:
    """Greedy bipartite matching keyed on (symbol, |entry_ts diff|).

    * Sort both sides by entry_ts.
    * For each backtest trade, find the closest unmatched live trade
      on the same symbol whose entry_ts is within ``tolerance_seconds``.
    * Side mismatches are still treated as matches — they show up in
      the ``side_mismatch_count`` aggregate so they don't get hidden
      among the unmatched.
    """
    res = MatchResult()
    # Pre-bucket live by symbol for cheaper inner loop
    by_symbol: Dict[str, List[CanonicalTrade]] = {}
    for t in live:
        by_symbol.setdefault(t.symbol, []).append(t)
    for v in by_symbol.values():
        v.sort(key=lambda t: t.entry_ts)
    used: set = set()  # ids of consumed live trades

    for bt in sorted(backtest, key=lambda t: t.entry_ts):
        candidates = by_symbol.get(bt.symbol, [])
        best: Optional[CanonicalTrade] = None
        best_diff = math.inf
        for cand in candidates:
            if id(cand) in used:
                continue
            diff = abs(cand.entry_ts - bt.entry_ts)
            if diff > tolerance_seconds:
                continue
            if diff < best_diff:
                best, best_diff = cand, diff
        if best is None:
            res.only_backtest.append(bt)
        else:
            res.matched.append((bt, best))
            used.add(id(best))

    for cand in live:
        if id(cand) not in used:
            res.only_live.append(cand)
    return res


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------

@dataclass
class DivergenceReport:
    backtest_count: int
    live_count: int
    matched_count: int
    only_backtest_count: int
    only_live_count: int
    side_mismatch_count: int
    pnl_total_backtest: float
    pnl_total_live: float
    pnl_diff_total: float
    pnl_diff_avg_matched: float
    pnl_diff_max_abs: float
    exit_hist_backtest: Dict[str, int]
    exit_hist_live: Dict[str, int]
    by_symbol: Dict[str, Dict[str, float]] = field(default_factory=dict)

    @property
    def match_rate(self) -> float:
        denom = max(self.backtest_count, self.live_count)
        return self.matched_count / denom if denom else 0.0


def aggregate(res: MatchResult, by_symbol: bool = False) -> DivergenceReport:
    pnl_bt = sum(t.pnl for pair in res.matched for t in (pair[0],))
    pnl_lv = sum(t.pnl for pair in res.matched for t in (pair[1],))
    diffs = [pair[1].pnl - pair[0].pnl for pair in res.matched]
    side_mm = sum(1 for bt, lv in res.matched if bt.side != lv.side)
    bt_hist: Dict[str, int] = {}
    lv_hist: Dict[str, int] = {}
    for bt, lv in res.matched:
        bt_hist[bt.exit_type] = bt_hist.get(bt.exit_type, 0) + 1
        lv_hist[lv.exit_type] = lv_hist.get(lv.exit_type, 0) + 1
    for t in res.only_backtest:
        bt_hist[t.exit_type] = bt_hist.get(t.exit_type, 0) + 1
    for t in res.only_live:
        lv_hist[t.exit_type] = lv_hist.get(t.exit_type, 0) + 1

    by_sym: Dict[str, Dict[str, float]] = {}
    if by_symbol:
        symbols = {p[0].symbol for p in res.matched} | \
                  {t.symbol for t in res.only_backtest} | \
                  {t.symbol for t in res.only_live}
        for sym in sorted(symbols):
            mp = [(b, l) for b, l in res.matched if b.symbol == sym]
            ob = [t for t in res.only_backtest if t.symbol == sym]
            ol = [t for t in res.only_live if t.symbol == sym]
            by_sym[sym] = {
                "matched": len(mp),
                "only_backtest": len(ob),
                "only_live": len(ol),
                "pnl_diff_total": sum(l.pnl - b.pnl for b, l in mp),
            }

    bt_count = len(res.matched) + len(res.only_backtest)
    lv_count = len(res.matched) + len(res.only_live)
    return DivergenceReport(
        backtest_count=bt_count,
        live_count=lv_count,
        matched_count=len(res.matched),
        only_backtest_count=len(res.only_backtest),
        only_live_count=len(res.only_live),
        side_mismatch_count=side_mm,
        pnl_total_backtest=pnl_bt,
        pnl_total_live=pnl_lv,
        pnl_diff_total=pnl_lv - pnl_bt,
        pnl_diff_avg_matched=(sum(diffs) / len(diffs)) if diffs else 0.0,
        pnl_diff_max_abs=max((abs(d) for d in diffs), default=0.0),
        exit_hist_backtest=bt_hist,
        exit_hist_live=lv_hist,
        by_symbol=by_sym,
    )


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

def _row(label, a, b, fmt="{}"):
    sa = fmt.format(a) if isinstance(a, (int, float)) else str(a)
    sb = fmt.format(b) if isinstance(b, (int, float)) else str(b)
    if isinstance(a, (int, float)) and isinstance(b, (int, float)):
        diff = b - a
        sign = "+" if diff >= 0 else ""
        d = f"{sign}{diff:.4f}" if isinstance(diff, float) else f"{sign}{diff}"
    else:
        d = "-"
    return f"  {label:<28} {sa:>14} {sb:>14} {d:>14}"


def print_report(report: DivergenceReport) -> None:
    print("=" * 72)
    print("BACKTEST vs LIVE DIVERGENCE")
    print("=" * 72)
    print(f"  {'metric':<28} {'backtest':>14} {'live':>14} {'delta':>14}")
    print("-" * 72)
    print(_row("trade_count", report.backtest_count, report.live_count, "{:d}"))
    print(_row("pnl_total_usd",
               report.pnl_total_backtest, report.pnl_total_live, "${:.2f}"))
    print("=" * 72)
    print(f"  matched:                {report.matched_count:>4d}")
    print(f"  only-backtest:          {report.only_backtest_count:>4d}")
    print(f"  only-live:              {report.only_live_count:>4d}")
    print(f"  match_rate:             {report.match_rate:.2%}")
    print(f"  side_mismatch (matched):{report.side_mismatch_count:>4d}")
    print(f"  pnl_diff_avg_matched:   ${report.pnl_diff_avg_matched:+.4f}")
    print(f"  pnl_diff_max_abs:       ${report.pnl_diff_max_abs:.4f}")
    print(f"  pnl_diff_total:         ${report.pnl_diff_total:+.2f}")
    print("=" * 72)
    exit_keys = sorted(set(report.exit_hist_backtest) | set(report.exit_hist_live))
    if exit_keys:
        print("EXIT TYPE COUNTS")
        for k in exit_keys:
            a = report.exit_hist_backtest.get(k, 0)
            b = report.exit_hist_live.get(k, 0)
            print(_row(f"  {k}", a, b, "{:d}"))
    if report.by_symbol:
        print("\nPER-SYMBOL")
        for sym, s in report.by_symbol.items():
            print(f"  {sym:<12} matched={s['matched']:>3d} "
                  f"only_bt={s['only_backtest']:>3d} "
                  f"only_lv={s['only_live']:>3d} "
                  f"pnl_diff=${s['pnl_diff_total']:+.2f}")
    print("=" * 72)
    # Slide-ready verdict
    if report.match_rate < 0.8:
        print("VERDICT: LOW match rate — investigate execution divergence "
              "before trusting backtest projections.")
    elif report.side_mismatch_count > 0:
        print("VERDICT: SIDE mismatches present — non-deterministic strategy "
              "or feed lag.  DO NOT promote to real money.")
    elif abs(report.pnl_diff_avg_matched) > 1.0:
        print("VERDICT: PnL drift on matched trades — tune slippage/latency "
              "in the realism YAML.")
    else:
        print("VERDICT: backtest and live agree within tolerance.")
    print()


def report_to_json(report: DivergenceReport) -> dict:
    return {
        "backtest_count": report.backtest_count,
        "live_count": report.live_count,
        "matched_count": report.matched_count,
        "only_backtest_count": report.only_backtest_count,
        "only_live_count": report.only_live_count,
        "match_rate": report.match_rate,
        "side_mismatch_count": report.side_mismatch_count,
        "pnl_total_backtest": report.pnl_total_backtest,
        "pnl_total_live": report.pnl_total_live,
        "pnl_diff_total": report.pnl_diff_total,
        "pnl_diff_avg_matched": report.pnl_diff_avg_matched,
        "pnl_diff_max_abs": report.pnl_diff_max_abs,
        "exit_hist_backtest": report.exit_hist_backtest,
        "exit_hist_live": report.exit_hist_live,
        "by_symbol": report.by_symbol,
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_window(s: Optional[str]) -> Tuple[Optional[float], Optional[float]]:
    if not s:
        return None, None
    parts = [p.strip() for p in s.split(",")]
    if len(parts) != 2:
        raise SystemExit("--time-window must be 'start,end'")
    return _parse_datetime(parts[0]) or None, _parse_datetime(parts[1]) or None


def main(argv: Optional[List[str]] = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--backtest-trades", required=True,
                   help="JSON file with BacktestResult.trades")
    p.add_argument("--live-trades", required=True,
                   help="LiveMetrics CSV (logs/live_trades_*.csv)")
    p.add_argument("--time-window", default=None,
                   help="ISO 'start,end' e.g. '2026-04-15T00:00,2026-04-15T08:00'")
    p.add_argument("--match-tolerance", type=float, default=60.0,
                   help="Max entry-time diff in seconds for a match (default 60)")
    p.add_argument("--by-symbol", action="store_true")
    p.add_argument("--json-out", default=None,
                   help="Optional path for machine-readable report")
    args = p.parse_args(argv)

    bt = load_backtest_trades(args.backtest_trades)
    lv = load_live_trades(args.live_trades)
    start_ts, end_ts = _parse_window(args.time_window)
    bt = filter_window(bt, start_ts, end_ts)
    lv = filter_window(lv, start_ts, end_ts)

    res = match_trades(bt, lv, tolerance_seconds=args.match_tolerance)
    report = aggregate(res, by_symbol=args.by_symbol)
    print_report(report)

    if args.json_out:
        with open(args.json_out, "w", encoding="utf-8") as f:
            json.dump(report_to_json(report), f, indent=2)
        print(f"JSON report: {args.json_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
