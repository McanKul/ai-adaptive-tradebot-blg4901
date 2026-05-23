"""
live/live_metrics.py
====================
Lightweight live trading metrics tracker.

Responsibilities:
- Record every closed trade to a CSV file (append-only, survives restarts)
- Compute running statistics: win rate, P&L, avg hold time, drawdown
- Log periodic summaries (after each trade + daily)
- Expose a snapshot dict for monitoring / health checks

CSV columns:
    timestamp, symbol, side, entry_price, exit_price, qty, pnl_usd,
    pnl_pct, bars_held, hold_seconds, exit_type, strategy
"""
from __future__ import annotations

import csv
import os
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, TYPE_CHECKING

from utils.logger import setup_logger

if TYPE_CHECKING:
    from live.position_manager import Position

log = setup_logger("LiveMetrics")

_CSV_HEADER = [
    "timestamp", "symbol", "side", "entry_price", "exit_price",
    "qty", "pnl_usd", "pnl_pct", "bars_held", "hold_seconds",
    "exit_type", "strategy",
    # Phase B2 — execution-quality columns.  Empty string when the
    # broker did not report a fill price yet (Phase C1 lands those).
    "intended_price", "fill_price", "slippage_bps",
    # Phase B4 — fee / funding accounting.  Populated whenever the
    # broker has reported commissions / funding payments for this
    # position; otherwise zero.  ``pnl_net_usd = pnl_usd - fees -
    # funding`` so downstream analysis sees the real after-cost number.
    "entry_fee_usd", "exit_fee_usd", "funding_usd",
    "pnl_gross_usd", "pnl_net_usd",
]


@dataclass
class _RunningStats:
    """Mutable running statistics — reset daily if desired."""
    total_trades: int = 0
    winning_trades: int = 0
    losing_trades: int = 0
    total_pnl: float = 0.0
    total_pnl_pct: float = 0.0
    best_trade_pnl: float = 0.0
    worst_trade_pnl: float = 0.0
    total_bars_held: int = 0
    total_hold_seconds: float = 0.0
    peak_equity: float = 0.0
    max_drawdown_pct: float = 0.0
    # per-symbol tracking
    symbol_pnl: Dict[str, float] = field(default_factory=dict)
    # Phase B4 — execution-cost aggregation
    total_fees: float = 0.0
    total_funding: float = 0.0
    total_pnl_net: float = 0.0
    total_slippage_bps_abs: float = 0.0
    slippage_count: int = 0


class LiveMetrics:
    """
    Append-only trade logger + running statistics for live trading.

    Usage:
        metrics = LiveMetrics(csv_path="logs/trades.csv", start_equity=1000.0)
        # After a position closes:
        metrics.record(position)
        # Periodic equity update (from balance cache):
        metrics.update_equity(current_equity)
        # Get snapshot for monitoring:
        snap = metrics.snapshot()
    """

    def __init__(
        self,
        csv_path: str = "logs/live_trades.csv",
        start_equity: float = 0.0,
    ):
        self._csv_path = csv_path
        self._start_equity = start_equity
        self._stats = _RunningStats(peak_equity=start_equity)
        self._session_start = time.time()
        self._daily_key: str = ""

        # Ensure CSV directory exists and write header if file is new
        os.makedirs(os.path.dirname(csv_path) or ".", exist_ok=True)
        if not os.path.exists(csv_path):
            with open(csv_path, "w", newline="", encoding="utf-8") as f:
                csv.writer(f).writerow(_CSV_HEADER)

    # ── Record a closed trade ────────────────────────────────────────

    def record(self, pos: "Position") -> None:
        """
        Record a closed position: append to CSV + update running stats.
        Should be called right after a position is closed.
        """
        if pos.exit is None or pos.entry == 0:
            log.warning("Skipping incomplete position: %s (exit=%s, entry=%s)",
                        pos.symbol, pos.exit, pos.entry)
            return

        # Compute P&L
        if pos.is_long:
            pnl_usd = (pos.exit - pos.entry) * pos.qty
        else:
            pnl_usd = (pos.entry - pos.exit) * pos.qty

        notional = pos.entry * pos.qty
        pnl_pct = (pnl_usd / notional * 100) if notional > 0 else 0.0

        hold_seconds = (pos.exit_ts - pos.open_ts) if pos.exit_ts else 0.0

        # Phase B2: surface execution-quality fields when populated
        intended = getattr(pos, "intended_price", None) or pos.entry
        fill = getattr(pos, "fill_price", None)
        slip_bps = getattr(pos, "slippage_bps", None)

        # Phase B4: fee / funding accounting (broker-supplied)
        entry_fee = float(getattr(pos, "entry_fee_usd", 0.0) or 0.0)
        exit_fee = float(getattr(pos, "exit_fee_usd", 0.0) or 0.0)
        funding = float(getattr(pos, "funding_usd", 0.0) or 0.0)
        pnl_gross = pnl_usd
        pnl_net = pnl_gross - entry_fee - exit_fee - funding

        # Append to CSV
        row = [
            time.strftime("%Y-%m-%d %H:%M:%S"),
            pos.symbol,
            pos.side,
            f"{pos.entry:.8f}",
            f"{pos.exit:.8f}",
            f"{pos.qty:.6f}",
            f"{pnl_usd:.4f}",
            f"{pnl_pct:.4f}",
            pos.bars_held,
            f"{hold_seconds:.1f}",
            pos.exit_type or "UNKNOWN",
            pos.strategy or "",
            f"{intended:.8f}" if intended is not None else "",
            f"{fill:.8f}" if fill is not None else "",
            f"{slip_bps:.4f}" if slip_bps is not None else "",
            f"{entry_fee:.4f}",
            f"{exit_fee:.4f}",
            f"{funding:.4f}",
            f"{pnl_gross:.4f}",
            f"{pnl_net:.4f}",
        ]
        try:
            with open(self._csv_path, "a", newline="", encoding="utf-8") as f:
                csv.writer(f).writerow(row)
        except Exception as e:
            log.warning("Failed to write trade CSV: %s", e)

        # Update running stats
        s = self._stats
        s.total_trades += 1
        s.total_pnl += pnl_usd
        s.total_pnl_pct += pnl_pct
        s.total_bars_held += pos.bars_held
        s.total_hold_seconds += hold_seconds

        if pnl_usd > 0:
            s.winning_trades += 1
        else:
            s.losing_trades += 1

        s.best_trade_pnl = max(s.best_trade_pnl, pnl_usd)
        s.worst_trade_pnl = min(s.worst_trade_pnl, pnl_usd)

        # Phase B4 — execution-cost aggregation
        s.total_fees += entry_fee + exit_fee
        s.total_funding += funding
        s.total_pnl_net += pnl_net
        if slip_bps is not None:
            s.total_slippage_bps_abs += abs(float(slip_bps))
            s.slippage_count += 1

        # Per-symbol P&L
        s.symbol_pnl[pos.symbol] = s.symbol_pnl.get(pos.symbol, 0.0) + pnl_usd

        # Log the trade
        log.info(
            "TRADE CLOSED | %s %s %s | entry=%.8f exit=%.8f qty=%.6f | "
            "P&L=$%.4f (%.2f%%) | held=%d bars (%.0fs) | exit=%s",
            pos.symbol, pos.side, pos.strategy or "",
            pos.entry, pos.exit, pos.qty,
            pnl_usd, pnl_pct, pos.bars_held, hold_seconds,
            pos.exit_type,
        )

        # Log running summary every trade
        self._log_summary()

    # ── Equity tracking (for drawdown) ───────────────────────────────

    def update_equity(self, current_equity: float) -> None:
        """Update peak equity and max drawdown from latest balance."""
        s = self._stats
        if current_equity > s.peak_equity:
            s.peak_equity = current_equity
        if s.peak_equity > 0:
            dd = (s.peak_equity - current_equity) / s.peak_equity * 100
            s.max_drawdown_pct = max(s.max_drawdown_pct, dd)

    def set_start_equity(self, equity: float) -> None:
        """Set starting equity (called once at engine start)."""
        self._start_equity = equity
        self._stats.peak_equity = max(self._stats.peak_equity, equity)

    # ── Summary logging ──────────────────────────────────────────────

    def _log_summary(self) -> None:
        s = self._stats
        win_rate = (s.winning_trades / s.total_trades * 100) if s.total_trades > 0 else 0
        avg_pnl = s.total_pnl / s.total_trades if s.total_trades > 0 else 0
        avg_hold = s.total_hold_seconds / s.total_trades if s.total_trades > 0 else 0
        avg_slip = (s.total_slippage_bps_abs / s.slippage_count
                    if s.slippage_count > 0 else 0.0)

        log.info(
            "SESSION STATS | trades=%d | win_rate=%.1f%% | "
            "gross=$%.2f | net=$%.2f | fees=$%.2f | funding=$%.2f | "
            "avg_pnl=$%.4f | avg_slip=%.1fbps | avg_hold=%.0fs | "
            "best=$%.4f | worst=$%.4f | max_dd=%.2f%%",
            s.total_trades, win_rate,
            s.total_pnl, s.total_pnl_net, s.total_fees, s.total_funding,
            avg_pnl, avg_slip, avg_hold,
            s.best_trade_pnl, s.worst_trade_pnl, s.max_drawdown_pct,
        )

    def log_daily_summary(self) -> None:
        """Force-log a daily summary. Call from engine on day change."""
        today = time.strftime("%Y-%m-%d")
        if today == self._daily_key:
            return
        self._daily_key = today

        s = self._stats
        uptime = time.time() - self._session_start
        hours = uptime / 3600

        log.info(
            "═══ DAILY SUMMARY (%s) ═══ | uptime=%.1fh | trades=%d | "
            "W/L=%d/%d | total_pnl=$%.2f | max_dd=%.2f%%",
            today, hours, s.total_trades,
            s.winning_trades, s.losing_trades,
            s.total_pnl, s.max_drawdown_pct,
        )

        # Per-symbol breakdown
        if s.symbol_pnl:
            for sym, pnl in sorted(s.symbol_pnl.items(), key=lambda x: x[1], reverse=True):
                log.info("  %s: $%.4f", sym, pnl)

    # ── Snapshot for monitoring / health checks ──────────────────────

    def snapshot(self) -> Dict:
        """Return current metrics as a plain dict (JSON-serializable)."""
        s = self._stats
        uptime = time.time() - self._session_start
        return {
            "uptime_seconds": round(uptime, 1),
            "total_trades": s.total_trades,
            "winning_trades": s.winning_trades,
            "losing_trades": s.losing_trades,
            "win_rate_pct": round(s.winning_trades / s.total_trades * 100, 2) if s.total_trades > 0 else 0,
            "total_pnl_usd": round(s.total_pnl, 4),
            "avg_pnl_usd": round(s.total_pnl / s.total_trades, 4) if s.total_trades > 0 else 0,
            "best_trade_usd": round(s.best_trade_pnl, 4),
            "worst_trade_usd": round(s.worst_trade_pnl, 4),
            "avg_hold_seconds": round(s.total_hold_seconds / s.total_trades, 1) if s.total_trades > 0 else 0,
            "avg_bars_held": round(s.total_bars_held / s.total_trades, 1) if s.total_trades > 0 else 0,
            "max_drawdown_pct": round(s.max_drawdown_pct, 2),
            "start_equity": round(self._start_equity, 2),
            "peak_equity": round(s.peak_equity, 2),
            "symbol_pnl": {k: round(v, 4) for k, v in s.symbol_pnl.items()},
        }
