"""
Strategy/arb/funding_arb_strategy.py
====================================
Funding-rate arbitrage between perpetual futures and spot.

Idea
----
Perpetual futures pay/receive funding every funding interval (e.g. 8h on
Binance USDT-M).  When funding is significantly **positive**, longs pay
shorts → shorting the perp + buying the same notional in spot earns the
funding while the basis tracks back to zero.  When funding is negative,
the reverse trade earns it.

This v1 implementation is **backtest-only**.  Live execution across
spot + perp products needs cross-product order routing that is out of
scope for this branch.

Inputs
------
* ``spot_symbol`` and ``perp_symbol`` — the two legs.
* Funding source: pass either ``funding_constant_bps`` (single value
  per interval) or ``funding_series_csv`` (a CSV with columns
  ``timestamp,funding_rate`` — same shape as
  :class:`Backtest.realism_config.FundingConfig.series_csv_path`).
* ``funding_threshold_bps`` — minimum funding magnitude (in bps per
  interval) to take a trade.
* ``funding_interval_hours`` — how often funding is paid (8h default).
* ``max_holding_intervals`` — exit after N funding intervals regardless.
* ``delever_on_basis_pct`` — if perp/spot basis converges below this,
  unwind early.

The strategy buffers per-symbol bars internally with
:class:`BarSyncBuffer` so the engine doesn't need to be aware that the
strategy is multi-symbol.  The engine still calls ``on_bar`` once per
symbol per bar; we only act when both legs are aligned.
"""
from __future__ import annotations
import csv
import logging
import os
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, TYPE_CHECKING

from Interfaces.IStrategy import IStrategy, StrategyDecision
from Interfaces.orders import Order, OrderSide, OrderType
from Backtest.multi_symbol_sync import BarSyncBuffer, MultiBar

if TYPE_CHECKING:
    from Interfaces.market_data import Bar

log = logging.getLogger(__name__)

_HOUR_NS = 60 * 60 * 1_000_000_000


@dataclass
class _LegPosition:
    """Tracks an open arbitrage position across two legs."""
    entry_ts_ns: int
    entry_spot_price: float
    entry_perp_price: float
    entry_funding_bps: float
    direction: int  # +1 = long spot / short perp; -1 = inverse
    notional: float
    intervals_held: int = 0


class FundingRateArbStrategy(IStrategy):
    """Funding-rate carry between spot and perp.

    Args:
        spot_symbol, perp_symbol: symbol identifiers, both must appear in
            the engine's ``symbols`` list.
        funding_threshold_bps: minimum |funding| to enter (bps/interval).
        funding_interval_hours: funding payment interval.
        funding_constant_bps: optional flat funding (when no CSV).
        funding_series_csv: path to ``timestamp,funding_rate`` CSV.
        funding_timestamp_unit: ``"ms"`` | ``"ns"`` for the CSV col.
        notional_per_leg_usd: notional sized per leg in USD.
        max_holding_intervals: hard exit after N intervals.
        delever_on_basis_pct: unwind when ``|perp - spot| / spot`` <
            this threshold (basis converged).
    """

    def __init__(
        self,
        spot_symbol: str,
        perp_symbol: str,
        funding_threshold_bps: float = 5.0,
        funding_interval_hours: int = 8,
        funding_constant_bps: float = 0.0,
        funding_series_csv: Optional[str] = None,
        funding_timestamp_unit: str = "ms",
        notional_per_leg_usd: float = 1_000.0,
        max_holding_intervals: int = 6,
        delever_on_basis_pct: float = 0.0005,
    ):
        if spot_symbol == perp_symbol:
            raise ValueError("spot_symbol and perp_symbol must differ")
        self.spot_symbol = spot_symbol
        self.perp_symbol = perp_symbol
        self.funding_threshold_bps = float(funding_threshold_bps)
        self.funding_interval_hours = int(funding_interval_hours)
        self.funding_constant_bps = float(funding_constant_bps)
        self.notional_per_leg_usd = float(notional_per_leg_usd)
        self.max_holding_intervals = int(max_holding_intervals)
        self.delever_on_basis_pct = float(delever_on_basis_pct)

        # Strict mode: both legs must have a bar at the same ts before we
        # decide.  forward_fill would double-emit during the per-symbol
        # call sequence at the same timestamp.
        self._buffer = BarSyncBuffer(
            symbols=[spot_symbol, perp_symbol],
            mode="strict",
        )
        self._funding_series: List[tuple] = []  # (ts_ns, rate_bps)
        if funding_series_csv:
            self._load_funding_csv(funding_series_csv, funding_timestamp_unit)

        # State
        self._position: Optional[_LegPosition] = None
        self._last_funding_check_ts: int = 0

    # ------------------------------------------------------------------
    # IStrategy
    # ------------------------------------------------------------------

    def reset(self) -> None:
        self._buffer.reset()
        self._position = None
        self._last_funding_check_ts = 0

    def on_bar(self, bar: "Bar", ctx: Any) -> StrategyDecision:
        # Engine calls us once per symbol — buffer and try to emit
        if bar.symbol not in (self.spot_symbol, self.perp_symbol):
            return StrategyDecision.no_action()

        self._buffer.update(bar)
        multi = self._buffer.try_emit()
        if multi is None:
            return StrategyDecision.no_action()

        return self._on_multi_bar(multi, ctx)

    # ------------------------------------------------------------------
    # Core logic
    # ------------------------------------------------------------------

    def _on_multi_bar(self, mb: MultiBar, ctx: Any) -> StrategyDecision:
        spot = mb[self.spot_symbol]
        perp = mb[self.perp_symbol]
        funding_bps = self._funding_at(mb.timestamp_ns)

        # Basis: relative deviation of perp from spot
        if spot.close <= 0:
            return StrategyDecision.no_action()
        basis_pct = (perp.close - spot.close) / spot.close

        orders: List[Order] = []
        meta: Dict[str, Any] = {
            "funding_bps": funding_bps,
            "basis_pct": basis_pct,
            "ts_ns": mb.timestamp_ns,
        }

        if self._position is None:
            # ---- Entry ----
            if abs(funding_bps) >= self.funding_threshold_bps:
                direction = 1 if funding_bps > 0 else -1
                qty_spot = self.notional_per_leg_usd / spot.close
                qty_perp = self.notional_per_leg_usd / perp.close

                if direction == 1:
                    # funding > 0 → longs pay shorts. We short perp, buy spot.
                    spot_side, perp_side = OrderSide.BUY, OrderSide.SELL
                else:
                    spot_side, perp_side = OrderSide.SELL, OrderSide.BUY

                orders.append(self._make_order(
                    self.spot_symbol, spot_side, qty_spot,
                    mb.timestamp_ns, leg="spot",
                ))
                orders.append(self._make_order(
                    self.perp_symbol, perp_side, qty_perp,
                    mb.timestamp_ns, leg="perp",
                ))

                self._position = _LegPosition(
                    entry_ts_ns=mb.timestamp_ns,
                    entry_spot_price=spot.close,
                    entry_perp_price=perp.close,
                    entry_funding_bps=funding_bps,
                    direction=direction,
                    notional=self.notional_per_leg_usd,
                )
                meta["action"] = "enter"
                meta["direction"] = direction
        else:
            # ---- Exit checks ----
            self._maybe_advance_interval(mb.timestamp_ns)
            should_exit, reason = self._should_exit(funding_bps, basis_pct)
            if should_exit:
                orders.extend(self._build_unwind_orders(mb))
                meta["action"] = "exit"
                meta["exit_reason"] = reason
                self._position = None

        return StrategyDecision(orders=orders, metadata=meta)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _make_order(
        self, symbol: str, side: OrderSide, qty: float,
        ts: int, leg: str, reduce_only: bool = False,
    ) -> Order:
        return Order(
            symbol=symbol, side=side, order_type=OrderType.MARKET,
            quantity=qty, timestamp_ns=ts,
            strategy_id="funding_arb",
            reduce_only=reduce_only,
            metadata={"leg": leg, "arb": "funding_rate"},
        )

    def _build_unwind_orders(self, mb: MultiBar) -> List[Order]:
        assert self._position is not None
        spot = mb[self.spot_symbol]
        perp = mb[self.perp_symbol]
        qty_spot = self.notional_per_leg_usd / self._position.entry_spot_price
        qty_perp = self.notional_per_leg_usd / self._position.entry_perp_price

        if self._position.direction == 1:
            # Was long spot / short perp → sell spot / buy perp
            spot_side, perp_side = OrderSide.SELL, OrderSide.BUY
        else:
            spot_side, perp_side = OrderSide.BUY, OrderSide.SELL

        return [
            self._make_order(
                self.spot_symbol, spot_side, qty_spot,
                mb.timestamp_ns, leg="spot", reduce_only=True,
            ),
            self._make_order(
                self.perp_symbol, perp_side, qty_perp,
                mb.timestamp_ns, leg="perp", reduce_only=True,
            ),
        ]

    def _should_exit(self, funding_bps: float, basis_pct: float) -> tuple[bool, str]:
        assert self._position is not None
        # 1. Hard time-out
        if self._position.intervals_held >= self.max_holding_intervals:
            return True, "max_holding"
        # 2. Funding sign flipped — carry reversed
        if (
            (self._position.direction == 1 and funding_bps <= 0)
            or (self._position.direction == -1 and funding_bps >= 0)
        ):
            return True, "funding_flip"
        # 3. Basis converged below threshold
        if abs(basis_pct) <= self.delever_on_basis_pct:
            return True, "basis_converged"
        return False, ""

    def _maybe_advance_interval(self, ts_ns: int) -> None:
        assert self._position is not None
        elapsed = ts_ns - self._position.entry_ts_ns
        intervals = elapsed // (self.funding_interval_hours * _HOUR_NS)
        self._position.intervals_held = max(intervals, self._position.intervals_held)

    def _funding_at(self, ts_ns: int) -> float:
        if self._funding_series:
            # Find most recent rate ≤ ts_ns
            lo, hi = 0, len(self._funding_series) - 1
            best = self.funding_constant_bps
            while lo <= hi:
                mid = (lo + hi) // 2
                m_ts, m_rate = self._funding_series[mid]
                if m_ts <= ts_ns:
                    best = m_rate
                    lo = mid + 1
                else:
                    hi = mid - 1
            return best
        return self.funding_constant_bps

    def _load_funding_csv(self, path: str, ts_unit: str) -> None:
        if not os.path.exists(path):
            log.warning("funding CSV not found: %s", path)
            return
        scale_to_ns = {"ms": 1_000_000, "ns": 1, "s": 1_000_000_000}.get(ts_unit, 1_000_000)
        rows: List[tuple] = []
        with open(path, "r", encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                try:
                    ts = int(row["timestamp"]) * scale_to_ns
                    rate = float(row["funding_rate"])
                    # Funding stored as decimal in CSV; convert to bps for our threshold.
                    rows.append((ts, rate * 10_000.0))
                except (KeyError, ValueError, TypeError):
                    continue
        rows.sort(key=lambda x: x[0])
        self._funding_series = rows
        log.info("loaded %d funding rows from %s", len(rows), path)

    # ------------------------------------------------------------------
    # Introspection (used by tests)
    # ------------------------------------------------------------------

    @property
    def position(self) -> Optional[_LegPosition]:
        return self._position


# Alias so ``StrategyFactory`` (which expects ``Strategy``) resolves us.
Strategy = FundingRateArbStrategy
