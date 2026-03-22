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
    ):
        self.broker = broker
        self.sizing_cfg = sizing_cfg
        self.exit_cfg = exit_cfg
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

    async def _symbol_filters(self, symbol: str, qty_f: float) -> tuple[float, float]:
        """Query exchange LOT_SIZE & PRICE_FILTER for *symbol* (cached).
        Returns (rounded_qty, tick_size). Falls back to (qty_f, 0.0) if
        filters are unavailable (e.g. DryBroker or exchange_info error).
        """
        try:
            if hasattr(self.broker, 'exchange_info'):
                info = await self.broker.exchange_info()
            else:
                info = await self.broker.client.futures_exchange_info()
            step = tick = None
            for s in info["symbols"]:
                if s["symbol"] == symbol:
                    for f in s["filters"]:
                        if f["filterType"] == "LOT_SIZE":
                            lot = float(f["stepSize"])
                            factor = 1 / lot
                            step = math.floor(qty_f * factor) / factor
                        if f["filterType"] == "PRICE_FILTER":
                            tick = float(f["tickSize"])
                    if tick is not None and step is not None:
                        return step, tick
            # Symbol not found in exchange info — use raw qty (e.g. DryBroker)
            log.debug("%s not found in exchange_info, using raw qty=%.6f", symbol, qty_f)
            return qty_f, 0.0
        except Exception as e:
            log.error("Failed to get LOT_SIZE/PRICE_FILTER %s: %s", symbol, e)
            return qty_f, 0.0

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

        # 1) Get mark price (via broker — rate limited + cached)
        mark_price = await self.broker.get_mark_price(symbol)

        # 2) Compute quantity from SizingConfig
        raw_qty = self.sizing_cfg.compute_qty(mark_price)
        qty, tick = await self._symbol_filters(symbol, raw_qty)
        if qty <= 0:
            log.warning("%s qty rounded to 0 — skipping", symbol)
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

        # 5) Place orders — capture exchange order IDs
        sl_order_id: Optional[int] = None
        tp_order_id: Optional[int] = None
        try:
            await self.broker.market_order(symbol, side_str, qty)

            if self.exit_cfg.use_exchange_orders:
                if sl_price is not None:
                    sl_order_id = await self.broker.place_stop_market(symbol, opp_str, sl_price)
                if tp_price is not None:
                    tp_order_id = await self.broker.place_take_profit(symbol, opp_str, tp_price)
        except Exception as e:
            log.error("Order placement failed %s: %s", symbol, e)
            return False

        # 6) Track position
        pos = Position(
            symbol=symbol,
            side=side_str,
            qty=qty,
            entry_price=mark_price,
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
                 max_global_positions: int = 2):
        self.broker = broker
        self._managers: Dict[str, PositionManager] = {}
        self.history: List[Position] = []
        self._store = PositionStore(path=persist_path)
        self._max_global_positions = max_global_positions

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
