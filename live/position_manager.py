"""
live/position_manager.py
========================
Config-driven position management for live trading.

Responsibilities:
- Open positions with sizing computed from SizingConfig
- Place exchange-level SL / TP orders and track their IDs
- Cancel counter-order when one side fills (OCO-like lifecycle)
- Monitor positions for local exit rules (trailing stop, max bars, USD targets)
- Startup reconciliation: detect orphaned exchange positions
- LiveSupervisor: one PositionManager per symbol with per-symbol config
"""
from __future__ import annotations

import math
import time
from typing import Dict, List, Optional, Any

from binance.enums import (
    SIDE_BUY, SIDE_SELL,
    FUTURE_ORDER_TYPE_MARKET,
)
from utils.logger import setup_logger
from Interfaces.IBroker import IBroker
from live.live_config import SizingConfig, ExitConfig
from live.position_store import PositionStore

log = setup_logger("PositionManager")


# ======================================================================
# Position dataclass
# ======================================================================
class Position:
    """Represents a single open position with tracked exchange order IDs."""

    def __init__(
        self,
        symbol: str,
        side: str,
        qty: float,
        entry_price: float,
        sl_price: Optional[float] = None,
        tp_price: Optional[float] = None,
        opened_ts: Optional[float] = None,
        tick: Optional[float] = None,
        strategy: Optional[str] = None,
        timeframe: str = "1h",
        bar_index: int = 0,
        levels: Optional[Any] = None,
        # Kept for backward compat — ignored
        client: Optional[Any] = None,
    ):
        self.symbol = symbol
        self.side = side
        self.qty = qty
        self.entry = entry_price
        self.sl = sl_price
        self.tp = tp_price
        self.open_ts = opened_ts or time.time()
        self.closed = False
        self.exit_ts: Optional[float] = None
        self.exit: Optional[float] = None
        self.exit_type: Optional[str] = None
        self.tick = tick
        self.timeframe = timeframe
        self.strategy = strategy
        self.bar_index = bar_index
        self.bars_held = 0
        self.levels = levels

        # Exchange order IDs for safe lifecycle management
        self.sl_order_id: Optional[int] = None
        self.tp_order_id: Optional[int] = None

        # Trailing stop tracking
        self.peak_price = entry_price
        self.peak_pnl: float = 0.0

        # Liquidation early-warning bookkeeping
        # Set by ``PositionManager.open()`` once leverage / MMR are known.
        self.liq_price: Optional[float] = None

        # Phase B2 — execution-quality bookkeeping.  ``intended_price``
        # is captured at signal time (mark) before any rounding;
        # ``fill_price`` lands once the fill confirmation polling in
        # Phase C1 reports the broker's avg fill.  Until C1 they are
        # equal so ``slippage_bps`` is 0 and the abort branch silent.
        self.intended_price: float = entry_price
        self.fill_price: Optional[float] = None
        self.slippage_bps: Optional[float] = None

        # Phase B4 — fee / funding accounting.  Populated by
        # ``LiveMetrics.record`` from broker fill data and funding
        # payment history (best-effort; broker errors leave them at
        # 0 so the gross PnL still gets logged).
        self.entry_fee_usd: float = 0.0
        self.exit_fee_usd: float = 0.0
        self.funding_usd: float = 0.0

        # Throttle for exchange-side fill checks
        self._last_exchange_check: float = 0.0

    @property
    def is_long(self) -> bool:
        return self.side == SIDE_BUY

    def unrealized_pnl(self, current_price: float) -> float:
        if self.is_long:
            return (current_price - self.entry) * self.qty
        return (self.entry - current_price) * self.qty

    def unrealized_pnl_pct(self, current_price: float) -> float:
        notional = self.entry * self.qty
        if notional == 0:
            return 0.0
        return self.unrealized_pnl(current_price) / notional

    def update_peak(self, current_price: float):
        pnl = self.unrealized_pnl(current_price)
        if self.is_long:
            if current_price > self.peak_price:
                self.peak_price = current_price
                self.peak_pnl = pnl
        else:
            if current_price < self.peak_price:
                self.peak_price = current_price
                self.peak_pnl = pnl


# ======================================================================
# Position Manager
# ======================================================================
class PositionManager:
    """
    Config-driven position manager.

    Uses SizingConfig for position sizing and ExitConfig for TP/SL rules.
    """

    def __init__(
        self,
        broker: IBroker,
        sizing_cfg: SizingConfig,
        exit_cfg: ExitConfig,
        max_concurrent: int = 3,
        symbol: Optional[str] = None,
        liq_guard_cfg: Optional[Any] = None,
        execution_cfg: Optional[Any] = None,
        book_streamer: Optional[Any] = None,
    ):
        self.broker = broker
        self.sizing_cfg = sizing_cfg
        self.exit_cfg = exit_cfg
        # Optional ``LiquidationGuardConfig``; when None the guard is off
        # and the tick checker short-circuits.
        self.liq_guard_cfg = liq_guard_cfg
        # Phase B1 — execution policy (spread filter, slippage budget, ...)
        # and the bookTicker streamer used to read live bid/ask.
        self.execution_cfg = execution_cfg
        self.book_streamer = book_streamer
        self.max_open = max_concurrent
        self.symbol: Optional[str] = symbol

        self.open_positions: Dict[tuple, Position] = {}
        self.history: List[Position] = []

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def position_qty(self, symbol: str) -> float:
        """Return signed qty for a symbol (positive=long, negative=short)."""
        for (sym, _), pos in self.open_positions.items():
            if sym == symbol and not pos.closed:
                return pos.qty if pos.is_long else -pos.qty
        return 0.0

    @staticmethod
    def _round_price(raw: float, tick: float, up: bool = False) -> float:
        if tick <= 0:
            return raw
        factor = 1 / tick
        return (math.ceil if up else math.floor)(raw * factor) / factor

    async def _symbol_filters(
        self, symbol: str, qty_f: float,
    ) -> tuple[float, float, Optional[float]]:
        """Query exchange LOT_SIZE / PRICE_FILTER / MIN_NOTIONAL.

        Returns ``(rounded_qty, tick_size, min_notional_usd)``.
        ``min_notional_usd`` may be ``None`` when the exchange does
        not advertise the filter or for stub brokers (DryBroker).
        Falls back to ``(qty_f, 0.0, None)`` on any error.
        """
        try:
            if hasattr(self.broker, 'exchange_info'):
                info = await self.broker.exchange_info()
            else:
                info = await self.broker.client.futures_exchange_info()
            step = tick = None
            min_notional: Optional[float] = None
            for s in info["symbols"]:
                if s["symbol"] == symbol:
                    for f in s["filters"]:
                        if f["filterType"] == "LOT_SIZE":
                            lot = float(f["stepSize"])
                            factor = 1 / lot
                            step = math.floor(qty_f * factor) / factor
                        elif f["filterType"] == "PRICE_FILTER":
                            tick = float(f["tickSize"])
                        elif f["filterType"] == "MIN_NOTIONAL":
                            # Binance USDT-M futures uses ``notional``;
                            # spot uses ``minNotional``.  Accept either.
                            raw = f.get("notional") or f.get("minNotional")
                            if raw is not None:
                                try:
                                    min_notional = float(raw)
                                except (TypeError, ValueError):
                                    min_notional = None
                    if tick is not None and step is not None:
                        return step, tick, min_notional
            log.debug("%s not found in exchange_info, using raw qty=%.6f", symbol, qty_f)
            return qty_f, 0.0, None
        except Exception as e:
            log.error("Failed to get LOT_SIZE/PRICE_FILTER %s: %s", symbol, e)
            return qty_f, 0.0, None

    # ------------------------------------------------------------------
    # Compute SL / TP prices
    # ------------------------------------------------------------------
    def _compute_sl_tp_prices(
        self,
        side_str: str,
        entry_price: float,
        tick: float,
    ) -> tuple[Optional[float], Optional[float]]:
        """
        Compute SL and TP prices from ExitConfig.

        Supports:
        - Percentage-based (stop_loss_pct, take_profit_pct)
        - USD-based (stop_loss_usd, take_profit_usd) — converted to price delta

        Returns (sl_price, tp_price), either may be None.
        """
        sl_price: Optional[float] = None
        tp_price: Optional[float] = None
        ecfg = self.exit_cfg

        # --- Stop Loss ---
        # Convert margin-return pct to price pct (leverage-aware)
        # In backtest, stop_loss_pct is margin-based: 0.02 = 2% of margin
        # Price move needed = margin_pct / leverage
        leverage = self.sizing_cfg.leverage if self.sizing_cfg.leverage > 0 else 1

        if ecfg.stop_loss_pct is not None:
            sl_pct = ecfg.stop_loss_pct / leverage
            if side_str == SIDE_BUY:
                sl_price = entry_price * (1 - sl_pct)
            else:
                sl_price = entry_price * (1 + sl_pct)

        elif ecfg.stop_loss_usd is not None and self.sizing_cfg.margin_usd > 0:
            # Convert USD loss to price delta:  delta_price = loss_usd / qty
            # qty ≈ margin * leverage / price
            leverage = self.sizing_cfg.leverage
            notional = self.sizing_cfg.margin_usd * leverage
            approx_qty = notional / entry_price if entry_price > 0 else 1.0
            delta = ecfg.stop_loss_usd / approx_qty if approx_qty > 0 else 0.0
            if side_str == SIDE_BUY:
                sl_price = entry_price - delta
            else:
                sl_price = entry_price + delta

        # --- Take Profit ---
        if ecfg.take_profit_pct is not None:
            tp_pct = ecfg.take_profit_pct / leverage
            if side_str == SIDE_BUY:
                tp_price = entry_price * (1 + tp_pct)
            else:
                tp_price = entry_price * (1 - tp_pct)

        elif ecfg.take_profit_usd is not None and self.sizing_cfg.margin_usd > 0:
            leverage = self.sizing_cfg.leverage
            notional = self.sizing_cfg.margin_usd * leverage
            approx_qty = notional / entry_price if entry_price > 0 else 1.0
            delta = ecfg.take_profit_usd / approx_qty if approx_qty > 0 else 0.0
            if side_str == SIDE_BUY:
                tp_price = entry_price + delta
            else:
                tp_price = entry_price - delta

        # Round to tick
        if sl_price is not None and tick > 0:
            sl_price = self._round_price(sl_price, tick, up=(side_str == SIDE_SELL))
        if tp_price is not None and tick > 0:
            tp_price = self._round_price(tp_price, tick, up=(side_str == SIDE_BUY))

        return sl_price, tp_price

    # ------------------------------------------------------------------
    # Open position
    # ------------------------------------------------------------------
    async def open_position(
        self,
        symbol: str,
        side: int,
        strategy_name: str,
        leverage: int,
        timeframe: str,
        levels: Optional[Any] = None,
    ) -> bool:
        """
        Open a new position using SizingConfig & ExitConfig.

        Args:
            symbol: Trading pair
            side: 1 for long, -1 for short
            strategy_name: For logging / tracking
            leverage: Leverage multiplier
            timeframe: Bar timeframe
            levels: Optional SupportResistanceResult
        """
        key = (symbol, strategy_name)
        if key in self.open_positions or len(self.open_positions) >= self.max_open:
            return False

        # ── Phase B1: spread filter (default-deny) ────────────────────
        # Reject when the live bookTicker spread is wider than the
        # configured max OR when no tick has arrived yet (we don't
        # send orders into an opaque market).  Disabled by setting
        # ``max_entry_spread_bps`` to 0.
        max_spread = float(getattr(self.execution_cfg, "max_entry_spread_bps", 0.0) or 0.0)
        if max_spread > 0 and self.book_streamer is not None:
            spread_bps = self.book_streamer.get_spread_bps(symbol)
            if spread_bps is None:
                log.warning("%s [%s] entry blocked — no bookTicker yet (spread filter on)",
                            symbol, strategy_name)
                return False
            if spread_bps > max_spread:
                log.info("%s [%s] entry blocked — spread %.2fbps > %.2fbps",
                         symbol, strategy_name, spread_bps, max_spread)
                return False

        # 1) Get mark price (via broker — rate limited + cached).
        # This is the ``intended price`` we'll compare the actual fill
        # against in Phase B2 once C1 (fill confirmation) lands.
        mark_price = await self.broker.get_mark_price(symbol)
        intended_price = mark_price

        # 2) Compute quantity from SizingConfig
        raw_qty = self.sizing_cfg.compute_qty(mark_price)
        qty, tick, min_notional = await self._symbol_filters(symbol, raw_qty)
        if qty <= 0:
            log.warning("%s qty rounded to 0 — skipping", symbol)
            return False

        # Phase B3 — MIN_NOTIONAL filter.  Binance silently rejects
        # orders below the symbol's min notional; we surface the
        # rejection up front and skip the trade rather than letting
        # the broker raise on submit.
        if min_notional is not None:
            order_notional = qty * mark_price
            if order_notional < min_notional:
                log.warning(
                    "%s [%s] entry blocked — notional %.4f USD < min %.4f USD "
                    "(qty=%.6f price=%.6f)",
                    symbol, strategy_name, order_notional, min_notional,
                    qty, mark_price,
                )
                return False

        side_str = SIDE_BUY if side == 1 else SIDE_SELL
        opp_str = SIDE_SELL if side_str == SIDE_BUY else SIDE_BUY

        # 3) Margin & leverage on exchange
        try:
            await self.broker.ensure_isolated_margin(symbol)
            await self.broker.set_leverage(symbol, leverage)
        except Exception as e:
            log.error("%s margin/leverage error: %s", symbol, e)
            return False

        # 4) Compute SL/TP prices from ExitConfig
        sl_price, tp_price = self._compute_sl_tp_prices(side_str, mark_price, tick)

        # 5) Place orders — capture exchange order IDs.  Phase C1
        # introduces fill-confirmation polling: we now treat the
        # broker response's order_id as a handle and wait until the
        # exchange confirms FILLED (or PARTIAL_FILLED + timeout).
        sl_order_id: Optional[int] = None
        tp_order_id: Optional[int] = None
        fill: Optional[dict] = None
        try:
            mo_resp = await self.broker.market_order(symbol, side_str, qty)
            mo_order_id = (
                mo_resp.get("orderId") or mo_resp.get("order_id")
                if isinstance(mo_resp, dict) else None
            )
            if mo_order_id is not None and hasattr(self.broker, "wait_for_fill"):
                try:
                    fill = await self.broker.wait_for_fill(
                        symbol, int(mo_order_id), timeout=5.0,
                    )
                except Exception as e:  # pragma: no cover — defensive
                    log.warning("%s wait_for_fill error: %s", symbol, e)
                    fill = None

            if self.exit_cfg.use_exchange_orders:
                if sl_price is not None:
                    sl_order_id = await self.broker.place_stop_market(symbol, opp_str, sl_price)
                if tp_price is not None:
                    tp_order_id = await self.broker.place_take_profit(symbol, opp_str, tp_price)
        except Exception as e:
            log.error("Order placement failed %s: %s", symbol, e)
            return False

        # ── Phase B2+C1: slippage measurement + abort ──────────────────
        # Compute realised slippage (signed by side; positive=adverse).
        # Abort if it exceeds the configured budget — we eat the small
        # round-trip fee instead of carrying a position with a price
        # we did not agree to.
        actual_fill_price = (
            float(fill["avg_price"]) if fill and fill.get("avg_price") else mark_price
        )
        executed_qty = (
            float(fill["executed_qty"]) if fill and fill.get("executed_qty") else qty
        )
        if executed_qty <= 0:
            log.error("%s [%s] fill returned executed_qty=0 — aborting entry",
                      symbol, strategy_name)
            try:
                await self.broker.close_position(symbol)
            except Exception:
                pass
            return False
        if actual_fill_price > 0:
            sign = 1.0 if side_str == SIDE_BUY else -1.0
            slippage_bps_val = (
                (actual_fill_price - intended_price) / intended_price * 1e4 * sign
            )
        else:
            slippage_bps_val = 0.0

        max_slip = float(getattr(self.execution_cfg, "max_slippage_bps", 0.0) or 0.0)
        if max_slip > 0 and abs(slippage_bps_val) > max_slip:
            log.error(
                "%s [%s] SLIPPAGE_ABORT — %.2fbps > %.2fbps; closing immediately",
                symbol, strategy_name, slippage_bps_val, max_slip,
            )
            try:
                await self.broker.close_position(symbol)
                if sl_order_id and hasattr(self.broker, "cancel_order"):
                    try:
                        await self.broker.cancel_order(symbol, sl_order_id)
                    except Exception:
                        pass
                if tp_order_id and hasattr(self.broker, "cancel_order"):
                    try:
                        await self.broker.cancel_order(symbol, tp_order_id)
                    except Exception:
                        pass
            except Exception as e:
                log.error("%s slippage abort close failed: %s", symbol, e)
            return False

        # ── Phase A2: missing-stop guard ──────────────────────────────
        # Entry succeeded but the server-side STOP_MARKET did NOT.  We
        # are now naked-long/short with no on-exchange protection and
        # local tick-exits depend on a healthy WS feed.  That is not
        # acceptable for real money — close immediately and leave the
        # ledger flat instead of accepting a position we can't protect.
        if (getattr(self.exit_cfg, "use_exchange_orders", False)
                and sl_price is not None and sl_order_id is None):
            log.error(
                "%s [%s] MISSING-STOP at entry — closing position; "
                "sl_price=%.8f failed to register on exchange",
                symbol, strategy_name, sl_price,
            )
            try:
                await self.broker.close_position(symbol)
                # Cancel any TP that did register, to avoid orphan
                if tp_order_id and hasattr(self.broker, "cancel_order"):
                    try:
                        await self.broker.cancel_order(symbol, tp_order_id)
                    except Exception:
                        pass
            except Exception as e:
                log.error("%s missing-stop force-close failed: %s — manual review",
                          symbol, e)
            return False

        # 6) Track position with the *actual* fill price + qty
        pos = Position(
            symbol=symbol,
            side=side_str,
            qty=executed_qty,
            entry_price=actual_fill_price,
            sl_price=sl_price,
            tp_price=tp_price,
            opened_ts=time.time(),
            tick=tick,
            strategy=strategy_name,
            timeframe=timeframe,
            levels=levels,
        )
        pos.sl_order_id = sl_order_id
        pos.tp_order_id = tp_order_id
        # Phase B2 / C1: persisted execution fields
        pos.intended_price = intended_price
        pos.fill_price = actual_fill_price
        pos.slippage_bps = slippage_bps_val
        # Phase B4: entry-side commission from the fill (USDT-equivalent)
        if fill:
            pos.entry_fee_usd = float(fill.get("commission_usd") or 0.0)
        self.assign_liq_price(pos)
        self.open_positions[key] = pos

        log.info(
            "%s [%s] [%s] OPEN %s qty=%.4f @ %.8f | SL=%.8f TP=%.8f | leverage=%dx",
            symbol,
            strategy_name,
            timeframe,
            side_str,
            qty,
            mark_price,
            sl_price or 0.0,
            tp_price or 0.0,
            leverage,
        )
        return True

    # ------------------------------------------------------------------
    # Update / exit monitoring
    # ------------------------------------------------------------------
    # How often to check exchange position_amt (seconds).
    # Between checks, rely on local exit logic only.
    _EXCHANGE_CHECK_INTERVAL = 10.0

    async def update_all(self):
        """
        Check all open positions for exit conditions.

        Safe lifecycle:
        - Periodically detect exchange-side fills (position_amt == 0)
        - Cancel counter-orders by ID when one side triggers
        - Local checks every bar: TP / SL / Trailing / Max bars
        """
        now = time.time()
        closed_keys: List[tuple] = []

        for key, pos in self.open_positions.items():
            symbol, _ = key

            try:
                mark_price = await self.broker.get_mark_price(symbol)
            except Exception as e:
                log.warning("%s failed to get mark price: %s", symbol, e)
                continue

            # --- Detect exchange-side fill (throttled) ---
            exit_type: Optional[str] = None
            last_check = pos._last_exchange_check
            if now - last_check >= self._EXCHANGE_CHECK_INTERVAL:
                try:
                    exchange_amt = await self.broker.position_amt(symbol)
                    if exchange_amt == 0 and not pos.closed:
                        exit_type = "EXCHANGE_FILL"
                except Exception:
                    pass
                pos._last_exchange_check = now

            # --- Local exit checks (every bar, no API call) ---
            if exit_type is None:
                exit_type = self._check_local_exit(pos, mark_price)

            if exit_type:
                log.info("%s exit triggered: %s @ %.8f", symbol, exit_type, mark_price)
                try:
                    if exit_type != "EXCHANGE_FILL":
                        await self.broker.close_position(symbol)
                    # Cancel specific counter-orders by ID
                    await self._cancel_position_orders(pos)
                except Exception as e:
                    log.error("%s close failed: %s", symbol, e)
                    continue

                pos.closed = True
                pos.exit_ts = now
                pos.exit = mark_price
                pos.exit_type = exit_type
                self.history.append(pos)
                closed_keys.append(key)

            # Increment bars held counter (only for still-open positions)
            if key not in closed_keys:
                pos.bars_held += 1

        for key in closed_keys:
            del self.open_positions[key]

    # ------------------------------------------------------------------
    # Liquidation early-warning
    # ------------------------------------------------------------------

    @staticmethod
    def compute_liq_price(
        side: str,
        entry_price: float,
        leverage: float,
        maintenance_margin_ratio: float,
    ) -> float:
        """Approximate Binance USDT-M cross-margin liquidation price.

        Long:  ``liq = entry * (1 - 1/leverage + mmr)``
        Short: ``liq = entry * (1 + 1/leverage - mmr)``

        This is a *first-order* estimate — the real liquidation engine
        looks at maintenance margin tiers (notional-dependent) and the
        wallet's cross balance.  For the early-warning use-case we
        only need a price within ~1 % of the true level so we can
        bail out before the engine hits us.  The 1% accuracy is well
        inside the ``buffer_pct`` cushion the user picks.
        """
        if leverage <= 0 or entry_price <= 0:
            return 0.0
        if side == SIDE_BUY:  # long
            return entry_price * max(0.0, 1.0 - 1.0 / leverage + maintenance_margin_ratio)
        # short
        return entry_price * (1.0 + 1.0 / leverage - maintenance_margin_ratio)

    def assign_liq_price(self, pos: Position) -> None:
        """Compute and store the liquidation price on a freshly-opened
        position.  Idempotent — reassigning is safe when leverage or MMR
        changes mid-life (rare)."""
        if self.liq_guard_cfg is None:
            return
        leverage = self.sizing_cfg.leverage if self.sizing_cfg.leverage > 0 else 1.0
        mmr = self.liq_guard_cfg.maintenance_margin_ratio
        pos.liq_price = self.compute_liq_price(
            pos.side, pos.entry, leverage, mmr,
        )

    async def check_missing_stop(
        self, symbol: str, tick_price: float,
    ) -> List[str]:
        """Phase A2 tick-time recheck for stop-less open positions.

        The entry path already aborts when a fresh ``STOP_MARKET`` fails
        to register, but a stop can also disappear later (manual cancel
        from the Binance UI, exchange-side glitch, exchange-side fill +
        we missed the close).  Anything still open with
        ``sl_order_id is None`` while ``use_exchange_orders=True`` is
        unprotected — force-close at the current mark.

        Returns the strategy keys that were closed.
        """
        # Defensive: backtest's ExitConfig (Backtest/exit_manager.py) does
        # not carry ``use_exchange_orders`` — assume False there to keep
        # the guard live-only.
        if not getattr(self.exit_cfg, "use_exchange_orders", False):
            return []
        closed: List[str] = []
        now = time.time()
        for key, pos in list(self.open_positions.items()):
            sym, strat = key
            if sym != symbol or pos.closed:
                continue
            if pos.sl_order_id is not None:
                continue
            log.error(
                "%s [%s] MISSING-STOP detected at tick — force closing @ %.8f",
                sym, strat, tick_price,
            )
            try:
                await self.broker.close_position(sym)
                await self._cancel_position_orders(pos)
            except Exception as e:
                log.error("%s missing-stop close failed: %s", sym, e)
                continue
            pos.closed = True
            pos.exit_ts = now
            pos.exit = tick_price
            pos.exit_type = "MISSING_STOP"
            self.history.append(pos)
            del self.open_positions[key]
            closed.append(strat)
        return closed

    async def check_liquidation_warning(
        self, symbol: str, tick_price: float,
    ) -> List[str]:
        """Pre-emptive close if mark price is uncomfortably close to liq.

        Called by ``on_tick``.  Skips if the guard is disabled or no
        position has a recorded ``liq_price``.  Returns the list of
        strategy keys that closed during this call.

        Trigger rule:
            ``distance_to_liq / abs(entry - liq) < buffer_pct``

        i.e. we have less than ``buffer_pct`` of the original safety
        margin remaining.  Default buffer is 0.20 → close once we are
        within the last 20 % of the cushion.
        """
        cfg = self.liq_guard_cfg
        if cfg is None or not cfg.enabled:
            return []

        closed: List[str] = []
        now = time.time()
        for key, pos in list(self.open_positions.items()):
            sym, strat = key
            if sym != symbol or pos.closed or pos.liq_price is None:
                continue

            # Both sides: distance is positive when we're alive.
            if pos.is_long:
                distance = tick_price - pos.liq_price
                cushion = pos.entry - pos.liq_price
            else:
                distance = pos.liq_price - tick_price
                cushion = pos.liq_price - pos.entry

            if cushion <= 0:
                # Should not happen with sensible inputs; bail to be safe.
                continue

            ratio = distance / cushion
            if ratio >= cfg.buffer_pct:
                continue

            log.error(
                "%s [%s] LIQ-WARN price=%.8f liq=%.8f cushion=%.4f "
                "remaining=%.1f%% (buffer=%.1f%%) action=%s",
                sym, strat, tick_price, pos.liq_price, cushion,
                ratio * 100.0, cfg.buffer_pct * 100.0, cfg.action,
            )

            if cfg.action == "alarm":
                continue

            try:
                await self.broker.close_position(sym)
                await self._cancel_position_orders(pos)
            except Exception as e:
                log.error("%s liq-guard close failed: %s", sym, e)
                continue

            pos.closed = True
            pos.exit_ts = now
            pos.exit = tick_price
            pos.exit_type = "LIQ_GUARD"
            self.history.append(pos)
            del self.open_positions[key]
            closed.append(strat)
        return closed

    # ------------------------------------------------------------------
    # Tick-level exits (intra-bar)
    # ------------------------------------------------------------------
    async def check_tick_exits(self, symbol: str, tick_price: float) -> List[str]:
        """Run local-only exit checks at tick granularity.

        Called by ``LiveEngine`` whenever the mark-price tick stream
        produces a new price for *symbol*.  Server-side ``STOP_MARKET``
        / ``TAKE_PROFIT_MARKET`` orders cover plain TP/SL at the
        exchange, so this method **deliberately skips** flat TP/SL
        triggers (they would race the server-side fills and cause
        double-close attempts).  It evaluates only the rules that the
        exchange cannot run for us:

        * **Trailing stop** — peak update is now per-tick instead of
          per-bar, which is a real correctness fix as well as a speed
          fix.
        * **Max-holding (time-based)** — uses wall-clock seconds so
          we don't have to wait for a bar boundary to release a stale
          position.
        * **USD-target exit** — ``take_profit_usd`` / ``stop_loss_usd``
          on the local ``ExitConfig`` if set.

        Returns the list of strategy keys that closed during this call
        (mostly for tests; production callers can ignore it).
        """
        closed: List[str] = []
        now = time.time()
        for key, pos in list(self.open_positions.items()):
            sym, strat = key
            if sym != symbol or pos.closed:
                continue

            exit_type = self._check_local_exit_tick(pos, tick_price, now)
            if exit_type is None:
                continue

            log.info(
                "%s [%s] tick-exit: %s @ %.8f",
                sym, strat, exit_type, tick_price,
            )
            try:
                await self.broker.close_position(sym)
                await self._cancel_position_orders(pos)
            except Exception as e:
                log.error("%s tick-exit close failed: %s", sym, e)
                continue

            pos.closed = True
            pos.exit_ts = now
            pos.exit = tick_price
            pos.exit_type = exit_type
            self.history.append(pos)
            del self.open_positions[key]
            closed.append(strat)
        return closed

    def _check_local_exit_tick(
        self,
        pos: Position,
        tick_price: float,
        now: float,
    ) -> Optional[str]:
        """Local-only exit rules suitable for tick granularity.

        Skips flat TP/SL since the exchange's ``STOP_MARKET`` /
        ``TAKE_PROFIT_MARKET`` orders handle those at higher fidelity.
        """
        ecfg = self.exit_cfg

        # --- Trailing stop (peak updated per-tick) ---
        if ecfg.trailing_stop_pct is not None:
            pos.update_peak(tick_price)
            if pos.peak_pnl > 0:
                current_pnl = pos.unrealized_pnl(tick_price)
                notional = pos.entry * pos.qty
                if notional > 0:
                    peak_pct = pos.peak_pnl / notional
                    curr_pct = current_pnl / notional
                    drawdown = peak_pct - curr_pct
                    leverage = self.sizing_cfg.leverage if self.sizing_cfg.leverage > 0 else 1
                    trail_pct = ecfg.trailing_stop_pct / leverage
                    if drawdown >= trail_pct:
                        return "TRAILING"

        # --- USD-target exits (the exchange has no equivalent) ---
        tp_usd = getattr(ecfg, "take_profit_usd", None)
        sl_usd = getattr(ecfg, "stop_loss_usd", None)
        if tp_usd or sl_usd:
            pnl = pos.unrealized_pnl(tick_price)
            if tp_usd and pnl >= tp_usd:
                return "TP_USD"
            if sl_usd and pnl <= -sl_usd:
                return "SL_USD"

        # --- Wall-clock max-holding ---
        max_hold_s = getattr(ecfg, "max_holding_seconds", None)
        if max_hold_s and (now - pos.open_ts) >= max_hold_s:
            return "MAX_HOLD_S"

        return None

    async def _cancel_position_orders(self, pos: Position):
        """Cancel SL and TP orders by their tracked IDs only."""
        cancelled = 0
        for oid in (pos.sl_order_id, pos.tp_order_id):
            if oid and hasattr(self.broker, 'cancel_order'):
                try:
                    await self.broker.cancel_order(pos.symbol, oid)
                    cancelled += 1
                except Exception as e:
                    log.debug("%s cancel order %s: %s", pos.symbol, oid, e)
        if cancelled:
            log.info("%s cancelled %d tracked orders", pos.symbol, cancelled)

    def _check_local_exit(self, pos: Position, mark_price: float) -> Optional[str]:
        """
        Check local exit rules.  Returns exit reason or None.
        """
        ecfg = self.exit_cfg

        # --- TP ---
        if pos.tp is not None:
            if pos.is_long and mark_price >= pos.tp:
                return "TP"
            if not pos.is_long and mark_price <= pos.tp:
                return "TP"

        # --- SL ---
        if pos.sl is not None:
            if pos.is_long and mark_price <= pos.sl:
                return "SL"
            if not pos.is_long and mark_price >= pos.sl:
                return "SL"

        # --- Trailing Stop ---
        if ecfg.trailing_stop_pct is not None:
            pos.update_peak(mark_price)
            if pos.peak_pnl > 0:
                current_pnl = pos.unrealized_pnl(mark_price)
                notional = pos.entry * pos.qty
                if notional > 0:
                    peak_pct = pos.peak_pnl / notional
                    curr_pct = current_pnl / notional
                    drawdown = peak_pct - curr_pct
                    # Convert margin-based trailing_stop_pct to price-based
                    leverage = self.sizing_cfg.leverage if self.sizing_cfg.leverage > 0 else 1
                    trail_pct = ecfg.trailing_stop_pct / leverage
                    if drawdown >= trail_pct:
                        return "TRAILING"

        # --- Max Holding Bars ---
        if ecfg.max_holding_bars is not None:
            if pos.bars_held >= ecfg.max_holding_bars:
                return "MAX_BARS"

        return None

    # ------------------------------------------------------------------
    # Force close all
    # ------------------------------------------------------------------
    async def close_position_by_strategy(
        self, symbol: str, strategy_name: str, exit_type: str = "STRATEGY_EXIT"
    ) -> bool:
        """Close a position triggered by strategy exit signal (e.g. ATR trailing stop)."""
        key = (symbol, strategy_name)
        pos = self.open_positions.get(key)
        if pos is None or pos.closed:
            return False

        try:
            mark_price = await self.broker.get_mark_price(symbol)
            await self.broker.close_position(symbol)
            await self._cancel_position_orders(pos)
        except Exception as e:
            log.error("%s strategy-exit close failed: %s", symbol, e)
            return False

        pos.closed = True
        pos.exit_ts = time.time()
        pos.exit = mark_price
        pos.exit_type = exit_type
        self.history.append(pos)
        del self.open_positions[key]
        return True

    async def force_close_all(self):
        to_remove: List[tuple] = []

        for key, pos in list(self.open_positions.items()):
            symbol, _ = key
            try:
                await self.broker.close_position(symbol)
                await self._cancel_position_orders(pos)
            except Exception as e:
                log.warning("%s force close failed: %s", symbol, e)

            pos.closed = True
            pos.exit_type = "MANUAL"
            self.history.append(pos)
            to_remove.append(key)

        for key in to_remove:
            del self.open_positions[key]

        log.info("All positions force closed (%d).", len(to_remove))

    # ------------------------------------------------------------------
    # Startup reconciliation
    # ------------------------------------------------------------------
    async def reconcile(self, strategy_name: str = "reconciled"):
        """
        Query exchange for orphaned positions in *self.symbol*.
        Adopt them into open_positions so the exit lifecycle can manage them.
        """
        if not self.symbol:
            return

        try:
            exchange_amt = await self.broker.position_amt(self.symbol)
        except Exception as e:
            log.warning("%s reconcile failed: %s", self.symbol, e)
            return

        if exchange_amt == 0:
            return

        key = (self.symbol, strategy_name)
        if key in self.open_positions:
            return  # already tracked

        side_str = SIDE_BUY if exchange_amt > 0 else SIDE_SELL
        qty = abs(exchange_amt)

        try:
            mark_price = await self.broker.get_mark_price(self.symbol)
        except Exception:
            mark_price = 0.0

        pos = Position(
            symbol=self.symbol,
            side=side_str,
            qty=qty,
            entry_price=mark_price,
            strategy=strategy_name,
        )

        # Adopt existing exchange orders
        try:
            if hasattr(self.broker, 'get_open_orders'):
                orders = await self.broker.get_open_orders(self.symbol)
                for o in orders:
                    otype = o.get("type", "")
                    oid = int(o.get("orderId", 0))
                    if "STOP_MARKET" in otype:
                        pos.sl_order_id = oid
                        pos.sl = float(o.get("stopPrice", 0))
                    elif "TAKE_PROFIT" in otype:
                        pos.tp_order_id = oid
                        pos.tp = float(o.get("stopPrice", 0))
        except Exception:
            pass

        self.open_positions[key] = pos
        log.info(
            "%s RECONCILED orphan: %s qty=%.4f @ %.8f (SL_oid=%s TP_oid=%s)",
            self.symbol, side_str, qty, mark_price,
            pos.sl_order_id, pos.tp_order_id,
        )


# ======================================================================
# Live Supervisor — one PositionManager per symbol
# ======================================================================
class LiveSupervisor:
    """
    Manages a per-symbol PositionManager with per-symbol config overrides.

    Provides a unified interface for the LiveEngine while isolating
    each symbol's position state, sizing, and exit rules.
    """

    def __init__(self, broker: IBroker, persist_path: str = "logs/live_positions.json",
                 max_global_positions: int = 2,
                 liq_guard_cfg: Optional[Any] = None,
                 execution_cfg: Optional[Any] = None,
                 book_streamer: Optional[Any] = None):
        self.broker = broker
        self._managers: Dict[str, PositionManager] = {}
        self.history: List[Position] = []
        self._store = PositionStore(path=persist_path)
        self._max_global_positions = max_global_positions
        # Forwarded to every PositionManager registered below.
        self.liq_guard_cfg = liq_guard_cfg
        # Phase B1 — shared execution policy + bookTicker streamer
        self.execution_cfg = execution_cfg
        self.book_streamer = book_streamer

    def register_symbol(
        self,
        symbol: str,
        sizing_cfg: SizingConfig,
        exit_cfg: ExitConfig,
        max_concurrent: int = 1,
    ):
        """Register a symbol with its own PositionManager."""
        if symbol not in self._managers:
            pm = PositionManager(
                broker=self.broker,
                sizing_cfg=sizing_cfg,
                exit_cfg=exit_cfg,
                max_concurrent=max_concurrent,
                symbol=symbol,
                liq_guard_cfg=self.liq_guard_cfg,
                execution_cfg=self.execution_cfg,
                book_streamer=self.book_streamer,
            )
            self._managers[symbol] = pm

    def get(self, symbol: str) -> PositionManager:
        return self._managers[symbol]

    @property
    def open_positions(self) -> Dict[tuple, Position]:
        """Merged view of all managers' open positions."""
        merged: Dict[tuple, Position] = {}
        for pm in self._managers.values():
            merged.update(pm.open_positions)
        return merged

    def position_qty(self, symbol: str) -> float:
        pm = self._managers.get(symbol)
        if pm is None:
            return 0.0
        return pm.position_qty(symbol)

    # ------------------------------------------------------------------
    # Periodic drift detection (real-money safety net)
    # ------------------------------------------------------------------
    async def detect_drift(self, qty_tolerance: float = 1e-6) -> List[Dict[str, Any]]:
        """Compare local intended-position quantity to the exchange.

        Returns a list of drift records, one per drifted symbol::

            {"symbol": "BTCUSDT",
             "local_qty": 0.5, "exchange_qty": 0.0,
             "abs_diff": 0.5, "kind": "ghost_local"}

        ``kind`` values:
            * ``ghost_local``    — local has position, exchange flat
              (server-side SL/TP fired and we didn't notice; rare with
              the periodic ``_EXCHANGE_CHECK_INTERVAL`` poll, but the
              drift loop closes the loophole during quiet bars).
            * ``orphan_exchange`` — exchange has position, local flat
              (manual order, restart with stale state, etc.).
            * ``size_mismatch``  — both sides hold position but qty
              differs (partial fill, missed leg, ADL).
            * ``side_mismatch``  — both hold but signs disagree
              (catastrophic — manual reversal or broker bug).
        """
        drifts: List[Dict[str, Any]] = []
        for symbol, pm in self._managers.items():
            try:
                exchange_amt = await self.broker.position_amt(symbol)
            except Exception as e:
                log.warning("%s drift check failed: %s", symbol, e)
                continue

            local_qty = self._signed_local_qty(pm, symbol)
            diff = exchange_amt - local_qty
            if abs(diff) <= qty_tolerance:
                continue

            kind = self._classify_drift(local_qty, exchange_amt, qty_tolerance)
            drifts.append({
                "symbol": symbol,
                "local_qty": local_qty,
                "exchange_qty": exchange_amt,
                "abs_diff": abs(diff),
                "kind": kind,
            })
        return drifts

    @staticmethod
    def _signed_local_qty(pm: "PositionManager", symbol: str) -> float:
        """Sum the symbol's open positions as a signed quantity.

        Long → positive; short → negative.  Multiple slot-strategies on
        the same symbol are summed (which matches what the exchange
        sees — a single net position per symbol)."""
        total = 0.0
        for (sym, _strategy), pos in pm.open_positions.items():
            if sym != symbol or pos.closed:
                continue
            sign = 1.0 if pos.is_long else -1.0
            total += sign * pos.qty
        return total

    @staticmethod
    def _classify_drift(
        local_qty: float, exchange_qty: float, tol: float,
    ) -> str:
        local_zero = abs(local_qty) <= tol
        exch_zero = abs(exchange_qty) <= tol
        if local_zero and not exch_zero:
            return "orphan_exchange"
        if exch_zero and not local_zero:
            return "ghost_local"
        # Both nonzero — sign or size mismatch
        if (local_qty > 0) != (exchange_qty > 0):
            return "side_mismatch"
        return "size_mismatch"

    async def on_tick(self, symbol: str, tick_price: float, ts_ms: int = 0) -> None:
        """Forward a mark-price tick to the symbol's PositionManager.

        Wired by ``LiveEngine`` to ``MarkPriceTickStreamer`` so trailing
        stops, USD-target exits, and time-based exits run intra-bar
        instead of waiting for the next bar close.  Server-side
        ``STOP_MARKET``/``TAKE_PROFIT_MARKET`` orders still cover
        plain TP/SL — see ``PositionManager.check_tick_exits``.

        Order of operations on every tick:

        1. **Liquidation guard** runs FIRST — emergency pre-emptive
           close beats every other rule because the alternative is the
           exchange's auto-liquidation engine.
        2. **Missing-stop guard** — any position without a server-side
           SL is forcibly closed before we check anything else.  If
           the stop disappeared (manual cancel, glitch) we are naked
           and must flatten.
        3. Local exit rules (trailing / USD targets / time-based).

        All three run in sequence so a single tick can pre-empt liq,
        prune missing-stop positions, *and* fire normal exits.
        Persistence fires once if anything closed.
        """
        pm = self._managers.get(symbol)
        if pm is None:
            return
        liq_closed = await pm.check_liquidation_warning(symbol, tick_price)
        miss_closed = await pm.check_missing_stop(symbol, tick_price)
        tick_closed = await pm.check_tick_exits(symbol, tick_price)
        if liq_closed or miss_closed or tick_closed:
            self._persist()

    def _persist(self):
        """Save all open positions to disk."""
        self._store.save(self.open_positions)

    def restore_positions(self):
        """
        Load persisted positions into the appropriate PositionManagers.
        Call AFTER register_symbol() for all symbols.
        """
        records = self._store.load()
        restored = 0
        for rec in records:
            sym = rec.get("symbol", "")
            pm = self._managers.get(sym)
            if pm is None:
                log.warning("No manager for persisted position %s — skipping", sym)
                continue

            strategy = rec.get("strategy", "restored")
            key = (sym, strategy)
            if key in pm.open_positions:
                continue  # already tracked

            pos = Position(
                symbol=sym,
                side=rec.get("side", "BUY"),
                qty=rec.get("qty", 0),
                entry_price=rec.get("entry_price", 0),
                sl_price=rec.get("sl_price"),
                tp_price=rec.get("tp_price"),
                opened_ts=rec.get("open_ts"),
                tick=rec.get("tick"),
                strategy=strategy,
                timeframe=rec.get("timeframe", "1m"),
            )
            pos.bars_held = rec.get("bars_held", 0)
            pos.sl_order_id = rec.get("sl_order_id")
            pos.tp_order_id = rec.get("tp_order_id")
            pos.peak_price = rec.get("peak_price", pos.entry)
            pos.peak_pnl = rec.get("peak_pnl", 0.0)

            pm.open_positions[key] = pos
            restored += 1

        if restored:
            log.info("Restored %d positions from disk", restored)

    async def open_position(
        self,
        symbol: str,
        side: int,
        strategy_name: str,
        leverage: int,
        timeframe: str,
        levels=None,
    ) -> bool:
        # Global concurrent position limit — hard gate before any exchange call
        total_open = sum(len(pm.open_positions) for pm in self._managers.values())
        if total_open >= self._max_global_positions:
            log.info("%s blocked: global position limit (%d/%d)",
                     symbol, total_open, self._max_global_positions)
            return False

        pm = self._managers.get(symbol)
        if pm is None:
            log.warning("No manager registered for %s", symbol)
            return False
        ok = await pm.open_position(
            symbol=symbol,
            side=side,
            strategy_name=strategy_name,
            leverage=leverage,
            timeframe=timeframe,
            levels=levels,
        )
        if ok:
            self._persist()
        return ok

    async def close_position(
        self, symbol: str, strategy_name: str, exit_type: str = "STRATEGY_EXIT"
    ) -> bool:
        """Close a position triggered by strategy exit signal."""
        pm = self._managers.get(symbol)
        if pm is None:
            return False
        ok = await pm.close_position_by_strategy(symbol, strategy_name, exit_type)
        if ok:
            self.history.extend(pm.history)
            pm.history.clear()
            self._persist()
        return ok

    async def update_symbol(self, symbol: str):
        """Update only the position manager for a specific symbol."""
        pm = self._managers.get(symbol)
        if pm is None:
            return
        had = len(pm.open_positions)
        await pm.update_all()
        self.history.extend(pm.history)
        pm.history.clear()
        if len(pm.open_positions) != had:
            self._persist()

    async def update_all(self):
        prev_count = sum(len(pm.open_positions) for pm in self._managers.values())
        for pm in self._managers.values():
            await pm.update_all()
            self.history.extend(pm.history)
            pm.history.clear()
        new_count = sum(len(pm.open_positions) for pm in self._managers.values())
        if new_count != prev_count:
            self._persist()

    async def force_close_all(self):
        for pm in self._managers.values():
            await pm.force_close_all()
            self.history.extend(pm.history)
            pm.history.clear()
        self._persist()
        log.info("Supervisor: all symbols force-closed.")

    async def reconcile_all(self, strategy_name: str = "reconciled"):
        """Reconcile orphaned exchange positions for all registered symbols."""
        for sym, pm in self._managers.items():
            await pm.reconcile(strategy_name)
        log.info("Supervisor: reconciliation complete for %d symbols.", len(self._managers))
