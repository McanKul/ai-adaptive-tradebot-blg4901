"""
Strategy/composite_strategy.py
==============================
Multi-strategy composition behind a single ``IStrategy`` facade.

The engine still calls ``on_bar(bar, ctx)`` on a single strategy.
``CompositeStrategy`` fans the call out to N child strategies (slots),
applies per-slot sizing / entry coefficient / exit policy / regime
gating, tags every order with ``slot.id``, and aggregates the result.

Design notes
------------
* When a composite is in use, the **caller must pass
  ``sizing_config=None`` to the engine** — composite owns sizing per
  slot.  Slots without an explicit ``slot.sizing`` fall back to the
  composite's ``default_sizing``.
* Per-slot ``ExitManager`` instances are isolated — slot A's TP doesn't
  touch slot B's position.
* Slot positions are tracked **eagerly** from the orders the composite
  emits.  This is an approximation when the engine rejects an order or
  partial-fills it; it matches the existing single-strategy
  ``ExitManager`` behaviour, which also derives ``position`` from the
  caller rather than from filled qty.
"""
from __future__ import annotations
import logging
from typing import List, Optional, Dict, Any, TYPE_CHECKING

from Interfaces.IStrategy import IStrategy, StrategyDecision
from Interfaces.strategy_adapter import (
    StrategyContext,
    SizingConfig,
    apply_sizing_to_orders,
    adapt_strategy_output,
)
from Interfaces.strategy_slot import StrategySlot
from Interfaces.orders import Order, OrderSide
from Backtest.exit_manager import ExitManager

if TYPE_CHECKING:
    from Interfaces.market_data import Bar

log = logging.getLogger(__name__)


class CompositeStrategy(IStrategy):
    """Aggregates ``StrategySlot`` instances behind a single strategy.

    Args:
        slots: Ordered list of ``StrategySlot``.
        default_sizing: Fallback ``SizingConfig`` for slots that did
            not declare their own.  When ``None`` and slot has no sizing,
            the slot's child orders keep the placeholder qty (=1.0)
            from ``adapt_strategy_output`` — usually wrong, so set this.
        regime_detector: Optional detector with ``detect(bar, ctx) ->
            RegimeState``.  Phase-2 plug-in.  When ``None`` no gating.
        policy: ``"fixed"`` | ``"regime_gate"`` | ``"regime_weight"``.
            * ``"fixed"`` runs every enabled slot, weights ignored.
            * ``"regime_gate"`` only runs slots whose ``regimes`` tag
              intersects current regime tags.
            * ``"regime_weight"`` runs all enabled slots but multiplies
              ``entry_coefficient`` by the slot's regime weight (1.0 if
              regime matches, 0.0 if not — extension point).
        name: Display name.
    """

    def __init__(
        self,
        slots: List[StrategySlot],
        default_sizing: Optional[SizingConfig] = None,
        regime_detector: Any = None,
        policy: str = "fixed",
        name: str = "composite",
    ):
        if not slots:
            raise ValueError("CompositeStrategy requires at least one slot")
        ids = [s.id for s in slots]
        if len(set(ids)) != len(ids):
            raise ValueError(f"duplicate slot ids: {ids}")
        if policy not in ("fixed", "regime_gate", "regime_weight"):
            raise ValueError(f"unknown composite policy: {policy}")

        self.slots = slots
        self.default_sizing = default_sizing
        self.regime_detector = regime_detector
        self.policy = policy
        self.name = name

        self._slot_exits: Dict[str, ExitManager] = {
            s.id: ExitManager(config=s.exit) for s in slots if s.exit is not None
        }
        self._slot_positions: Dict[str, float] = {s.id: 0.0 for s in slots}
        self._slot_entry_prices: Dict[str, float] = {s.id: 0.0 for s in slots}
        self._slot_entry_bars: Dict[str, int] = {s.id: 0 for s in slots}
        self._bar_index = 0

    # ------------------------------------------------------------------
    # IStrategy
    # ------------------------------------------------------------------

    def reset(self) -> None:
        for slot in self.slots:
            try:
                slot.strategy.reset()
            except Exception:  # pragma: no cover — defensive
                pass
        for em in self._slot_exits.values():
            em.reset()
        for sid in list(self._slot_positions):
            self._slot_positions[sid] = 0.0
            self._slot_entry_prices[sid] = 0.0
            self._slot_entry_bars[sid] = 0
        self._bar_index = 0

    def on_bar(self, bar: "Bar", ctx: Any) -> StrategyDecision:
        self._bar_index += 1

        regime_tags: List[str] = []
        if self.regime_detector is not None:
            try:
                state = self.regime_detector.detect(bar, ctx)
                if state and getattr(state, "tags", None):
                    regime_tags = list(state.tags)
            except Exception as e:  # pragma: no cover — defensive
                log.debug("regime_detector.detect raised: %s", e)

        all_orders: List[Order] = []
        slot_meta: Dict[str, Dict[str, Any]] = {}

        for slot in self.slots:
            if not slot.enabled:
                continue

            ec = slot.entry_coefficient
            active = True
            if self.policy == "regime_gate" and slot.regimes:
                if not (set(slot.regimes) & set(regime_tags)):
                    active = False
            elif self.policy == "regime_weight" and slot.regimes:
                if not (set(slot.regimes) & set(regime_tags)):
                    ec = 0.0  # zero out entries; existing exits still run

            slot_position = self._slot_positions[slot.id]

            slot_orders: List[Order] = []

            if active and ec > 0.0:
                slot_ctx = self._build_slot_ctx(ctx, slot, slot_position)
                adapted = adapt_strategy_output(
                    slot.strategy, bar, slot_ctx,
                    position_size=1.0, strategy_id=slot.id,
                )
                child_orders = list(adapted.orders)

                # Tag (defensive — adapter already does it but signal-only path needs it)
                for o in child_orders:
                    o.strategy_id = slot.id

                # Per-slot sizing (placeholder qty → real qty)
                sizing = slot.sizing or self.default_sizing
                if sizing is not None:
                    child_orders = apply_sizing_to_orders(
                        child_orders, sizing, bar.close
                    )

                if ec != 1.0:
                    child_orders = [self._scale_qty(o, ec) for o in child_orders]

                slot_orders.extend(child_orders)

            # Per-slot ExitManager (always runs while position open, even if
            # the slot is regime-gated off — exits should not be blocked)
            if slot.id in self._slot_exits and abs(slot_position) > 1e-10:
                em = self._slot_exits[slot.id]
                exit_result = em.check_exit(
                    bar=bar,
                    position=slot_position,
                    avg_entry_price=self._slot_entry_prices[slot.id],
                    bar_index=self._bar_index,
                    entry_bar_index=self._slot_entry_bars[slot.id],
                    strategy_id=slot.id,
                )
                if exit_result is not None:
                    exit_order, _reason = exit_result
                    exit_order.reduce_only = True  # keep field consistent
                    slot_orders.append(exit_order)

            # Update slot intended position from emitted orders
            for o in slot_orders:
                self._update_slot_state(slot.id, o, bar)

            all_orders.extend(slot_orders)
            slot_meta[slot.id] = {
                "active": active,
                "entry_coefficient": ec,
                "weight": slot.weight,
                "n_orders": len(slot_orders),
                "position": self._slot_positions[slot.id],
            }

        return StrategyDecision(
            orders=all_orders,
            metadata={"slots": slot_meta, "regime_tags": regime_tags},
            regime_tags=regime_tags,
        )

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _build_slot_ctx(
        self,
        ctx: Any,
        slot: StrategySlot,
        slot_position: float,
    ) -> StrategyContext:
        """Build a child context that mirrors the engine's ctx but
        replaces ``position`` with the slot-level position so child
        strategies see *their* book, not the aggregate."""
        return StrategyContext(
            symbol=getattr(ctx, "symbol", ""),
            timeframe=getattr(ctx, "timeframe", ""),
            bar_store=getattr(ctx, "bar_store", None),
            portfolio=getattr(ctx, "portfolio", None),
            position=slot_position,
            equity=getattr(ctx, "equity", 0.0),
            cash=getattr(ctx, "cash", 0.0),
            timestamp_ns=getattr(ctx, "timestamp_ns", 0),
            params=dict(getattr(ctx, "params", {}) or {}),
            metadata={
                **(getattr(ctx, "metadata", {}) or {}),
                "_slot_id": slot.id,
            },
        )

    @staticmethod
    def _scale_qty(o: Order, factor: float) -> Order:
        """Return a new Order with quantity scaled by ``factor``.
        Reduce-only orders are not scaled (they close existing position)."""
        if o.reduce_only or factor == 1.0:
            return o
        return Order(
            symbol=o.symbol,
            side=o.side,
            order_type=o.order_type,
            quantity=o.quantity * factor,
            price=o.price,
            stop_price=o.stop_price,
            timestamp_ns=o.timestamp_ns,
            strategy_id=o.strategy_id,
            reduce_only=o.reduce_only,
            metadata=dict(o.metadata or {}),
        )

    def _update_slot_state(self, slot_id: str, order: Order, bar: "Bar") -> None:
        """Eagerly update slot's intended position from an emitted order."""
        signed = order.quantity if order.side == OrderSide.BUY else -order.quantity
        prev = self._slot_positions[slot_id]
        new = prev + signed

        if order.reduce_only:
            self._slot_positions[slot_id] = new
            if abs(new) < 1e-10:
                self._slot_entry_prices[slot_id] = 0.0
                if slot_id in self._slot_exits:
                    self._slot_exits[slot_id].close_position(order.symbol)
            return

        # Entry / scale-in / reversal
        if abs(prev) < 1e-10 or (prev > 0) != (new > 0):
            # New entry or sign flip
            self._slot_entry_prices[slot_id] = bar.close
            self._slot_entry_bars[slot_id] = self._bar_index
            if slot_id in self._slot_exits:
                self._slot_exits[slot_id].register_entry(
                    symbol=order.symbol,
                    quantity=new,
                    entry_price=bar.close,
                    bar_index=self._bar_index,
                    timestamp_ns=order.timestamp_ns,
                )
        else:
            # Same-side scale-in: weighted average
            old_notional = abs(prev) * self._slot_entry_prices[slot_id]
            add_notional = abs(signed) * bar.close
            denom = abs(new)
            self._slot_entry_prices[slot_id] = (
                (old_notional + add_notional) / denom if denom > 1e-10 else bar.close
            )
            if slot_id in self._slot_exits:
                self._slot_exits[slot_id].update_position(
                    symbol=order.symbol,
                    quantity=new,
                    avg_entry_price=self._slot_entry_prices[slot_id],
                )
        self._slot_positions[slot_id] = new

    # ------------------------------------------------------------------
    # Introspection helpers
    # ------------------------------------------------------------------

    def get_slot_position(self, slot_id: str) -> float:
        return self._slot_positions.get(slot_id, 0.0)

    def get_slot_entry_price(self, slot_id: str) -> float:
        return self._slot_entry_prices.get(slot_id, 0.0)
