"""
Backtest/execution_models.py
============================
Execution models for simulating order fills.

Design decisions:
- SimpleExecutionModel: MARKET orders fill at BAR CLOSE price (documented choice)
  This is conservative: assumes orders submitted during bar execute at close.
- Applies slippage and fees from cost model
- Returns deterministic fills (using seeded RNG for any randomness)
- LimitExecutionModel skeleton for future extension
- PartialFillExecutionModel: Supports partial fills based on bar liquidity

Why bar close?
- More conservative than bar open
- Represents "you react to information, then trade"
- Common in academic backtests

PARTIAL FILL MODEL (NEW):
=========================
For realistic execution simulation, partial fills based on bar liquidity:
    fill_ratio = min(1.0, bar.volume / (order.qty * liquidity_scale))
If fill_ratio < 1.0 => only partial fill, remainder stays unfilled.
Unfilled portions do NOT auto-fill unless explicitly modeled.

LATENCY MODEL (NEW):
====================
Deterministic latency affects fill timestamp:
    fill_timestamp = bar.timestamp_ns + latency_ns
Latency is computed from LatencyModel using seeded RNG for determinism.
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import List, Optional, Protocol, Tuple, Dict, Any
from random import Random
import logging
import re as _re

from Interfaces.market_data import Bar, Tick
from Interfaces.orders import Order, Fill, OrderType, OrderSide, OrderStatus
from Interfaces.cost_interface import ICostModel

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_BAR_DUR_CACHE: Dict[str, int] = {}


def _parse_bar_duration_ns(timeframe: str) -> int:
    """Convert a timeframe string (e.g. '1m', '15m', '1h') to nanoseconds."""
    cached = _BAR_DUR_CACHE.get(timeframe)
    if cached is not None:
        return cached
    m = _re.match(r"(\d+)([smhd])", timeframe)
    if not m:
        return 0
    val, unit = int(m.group(1)), m.group(2)
    ns = {"s": 1_000_000_000, "m": 60_000_000_000,
          "h": 3_600_000_000_000, "d": 86_400_000_000_000}.get(unit, 0) * val
    _BAR_DUR_CACHE[timeframe] = ns
    return ns


class ReservoirSampler:
    """Fixed-size reservoir sampling (Vitter's Algorithm R).

    Keeps at most *max_size* representative samples **and** an exact
    running sum so that ``mean`` is always precise.  Percentile
    estimates are approximate when the total count exceeds *max_size*.

    Memory: O(max_size) regardless of stream length.
    """
    __slots__ = ("_max", "_buf", "_n", "_sum", "_rng")

    def __init__(self, max_size: int = 10_000):
        self._max = max_size
        self._buf: List[float] = []
        self._n: int = 0
        self._sum: float = 0.0
        self._rng = Random(0)

    def add(self, value: float) -> None:
        self._n += 1
        self._sum += value
        if len(self._buf) < self._max:
            self._buf.append(value)
        else:
            j = self._rng.randint(0, self._n - 1)
            if j < self._max:
                self._buf[j] = value

    @property
    def mean(self) -> float:
        return self._sum / self._n if self._n > 0 else 0.0

    def percentile(self, pct: float) -> float:
        if not self._buf:
            return 0.0
        s = sorted(self._buf)
        idx = min(int(len(s) * pct / 100), len(s) - 1)
        return float(s[idx])

    def __len__(self) -> int:
        return self._n

    def __bool__(self) -> bool:
        return self._n > 0


@dataclass
class PartialFillConfig:
    """Configuration for partial fill behavior."""
    # Enable partial fills based on liquidity
    enable_partial_fills: bool = False
    # Scale factor: fill_ratio = bar.volume / (order.qty * liquidity_scale)
    # Higher values = more fills, lower = stricter liquidity requirements
    liquidity_scale: float = 10.0
    # Minimum fill ratio (below this, order is rejected)
    min_fill_ratio: float = 0.1


@dataclass
class LatencyConfig:
    """Configuration for latency simulation."""
    # Enable latency simulation
    enable_latency: bool = False
    # Base latency in nanoseconds
    base_latency_ns: int = 0
    # Maximum additional random latency (using seeded RNG)
    max_jitter_ns: int = 0


@dataclass
class ExecutionStats:
    """Statistics from execution model.

    Latency and slippage samples are stored in **ReservoirSampler**
    instances (bounded to 10 000 items) so memory stays O(1) even for
    high-trade strategies.  Exact mean is still computed via running sum.
    """
    total_fills: int = 0
    partial_fills: int = 0
    rejected_orders: int = 0
    total_latency_ns: int = 0
    max_latency_ns: int = 0
    latency_samples: int = 0
    total_unfilled_qty: float = 0.0
    total_fill_ratio: float = 0.0  # Sum of fill ratios for averaging
    fills_bar_shifted: int = 0     # fills whose price changed due to latency
    # Bounded reservoir samplers (PART 2 — memory safety)
    _latency_sampler: ReservoirSampler = field(default_factory=lambda: ReservoirSampler(10_000))
    _slippage_sampler: ReservoirSampler = field(default_factory=lambda: ReservoirSampler(10_000))
    _price_shift_sampler: ReservoirSampler = field(default_factory=lambda: ReservoirSampler(10_000))

    @property
    def avg_latency_ns(self) -> float:
        """Average latency in nanoseconds."""
        return self.total_latency_ns / self.latency_samples if self.latency_samples > 0 else 0.0

    @property
    def avg_fill_ratio(self) -> float:
        """Average fill ratio across all fills."""
        return self.total_fill_ratio / self.total_fills if self.total_fills > 0 else 0.0

    @property
    def p95_latency_ns(self) -> float:
        return self._latency_sampler.percentile(95)

    @property
    def p99_latency_ns(self) -> float:
        return self._latency_sampler.percentile(99)

    @property
    def p95_slippage_bps(self) -> float:
        return self._slippage_sampler.percentile(95)

    @property
    def p99_slippage_bps(self) -> float:
        return self._slippage_sampler.percentile(99)

    @property
    def avg_slippage_bps(self) -> float:
        return self._slippage_sampler.mean

    @property
    def avg_price_shift(self) -> float:
        return self._price_shift_sampler.mean

    @property
    def p95_price_shift(self) -> float:
        return self._price_shift_sampler.percentile(95)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for metrics."""
        return {
            "total_fills": self.total_fills,
            "partial_fills": self.partial_fills,
            "rejected_orders": self.rejected_orders,
            "avg_latency_ns": self.avg_latency_ns,
            "max_latency_ns": self.max_latency_ns,
            "p95_latency_ns": self.p95_latency_ns,
            "p99_latency_ns": self.p99_latency_ns,
            "latency_samples": self.latency_samples,
            "total_unfilled_qty": self.total_unfilled_qty,
            "avg_fill_ratio": self.avg_fill_ratio,
            "avg_slippage_bps": self.avg_slippage_bps,
            "p95_slippage_bps": self.p95_slippage_bps,
            "p99_slippage_bps": self.p99_slippage_bps,
            "fills_bar_shifted": self.fills_bar_shifted,
            "avg_price_shift": self.avg_price_shift,
            "p95_price_shift": self.p95_price_shift,
        }


class SimpleExecutionModel:
    """
    Simple execution model: MARKET orders fill at bar close.
    
    Execution logic:
    1. MARKET orders fill immediately at bar close price
    2. Apply slippage from cost model (unfavorable direction)
    3. Apply spread cost (half-spread added to price)
    4. Calculate fee
    5. Return fill
    
    Note: This model fills ALL market orders - no rejection based on liquidity.
    For more realistic simulation, use PartialFillExecutionModel.
    """
    
    def __init__(
        self,
        use_bar_close: bool = True,
        partial_fill_config: Optional[PartialFillConfig] = None,
        latency_config: Optional[LatencyConfig] = None,
        realism_config: Optional[Any] = None,
    ):
        """
        Initialize SimpleExecutionModel.
        
        Args:
            use_bar_close: If True, use bar close price. If False, use bar open.
            partial_fill_config: Configuration for partial fills (optional)
            latency_config: Configuration for latency simulation (optional)
            realism_config: RealismConfig for advanced features (optional)
        """
        self.use_bar_close = use_bar_close
        self.partial_fill_config = partial_fill_config or PartialFillConfig()
        self.latency_config = latency_config or LatencyConfig()
        self._realism = realism_config  # RealismConfig or None
        self._stats = ExecutionStats()
        # Track unfilled portions (order_id -> remaining_qty)
        self._unfilled: Dict[str, float] = {}
        # Bar buffer for price-aware latency (A2)
        self._bar_history: List[Bar] = []
        self._max_bar_history: int = 20

    # ---- helpers for realism config access ----
    @property
    def _tc_config(self):
        """Return the TransactionCostConfig or None."""
        if self._realism is None:
            return None
        return getattr(self._realism, "transaction_costs", None)

    def _resolve_base_price(
        self,
        bar: Bar,
        fill_timestamp_ns: int,
        rng: Optional[Random] = None,
    ) -> float:
        """
        Determine the base execution price.

        Legacy (``timestamp_only``): bar.close or bar.open as configured.
        Price-aware (``price_aware``): first check bar history for a bar
        whose timestamp >= fill_time.  If none found, behaviour depends on
        ``intrabar_price_model``:
          * ``"none"`` (default) — return the bar field price unchanged.
            No bias / lookahead.  Latency affects fill *timestamp* only.
          * ``"gaussian_clamped"`` — apply a latency-proportional random
            shift from the bar's OHLC range.  **Approximate** — may
            introduce clamping bias on small bars.
        """
        tc = self._tc_config
        if tc is None or tc.price_latency_mode != "price_aware":
            return bar.close if self.use_bar_close else bar.open

        if fill_timestamp_ns <= bar.timestamp_ns:
            return bar.close if self.use_bar_close else bar.open

        bar_field = tc.price_latency_bar_field  # "open" | "close"

        # Check bar history for a bar at/after fill time
        for b in self._bar_history:
            if b.timestamp_ns >= fill_timestamp_ns and b.symbol == bar.symbol:
                return getattr(b, bar_field, b.open)

        # --- intra-bar price model ---
        base = bar.close if self.use_bar_close else bar.open
        intrabar_model = getattr(tc, "intrabar_price_model", "none")
        if intrabar_model == "none" or rng is None:
            return base

        # "gaussian_clamped" model
        bar_dur_ns = _parse_bar_duration_ns(bar.timeframe)
        if bar_dur_ns <= 0:
            return base

        latency_frac = min(2.0, (fill_timestamp_ns - bar.timestamp_ns) / bar_dur_ns)
        if latency_frac < 0.005:          # <0.5 % of bar → negligible
            return base

        bar_range = bar.high - bar.low
        if bar_range <= 0:
            return base

        # Gaussian shift: σ = latency_fraction × half-range
        sigma = latency_frac * bar_range * 0.5
        shift = rng.gauss(0, sigma)
        adjusted = base + shift

        # Clamp to bar range (fills can't beat the bar's extremes)
        adjusted = max(bar.low, min(bar.high, adjusted))

        # Track the shift for reporting
        abs_shift = abs(adjusted - base)
        self._stats._price_shift_sampler.add(abs_shift)
        if abs_shift > 1e-12:
            self._stats.fills_bar_shifted += 1

        return adjusted
    
    def process_orders(
        self,
        orders: List[Order],
        bar: Bar,
        portfolio: "Any",
        cost_model: ICostModel,
        rng: Random
    ) -> List[Fill]:
        """
        Process orders and return fills.
        """
        fills = []
        
        # Store bar in history for price-aware latency
        self._bar_history.append(bar)
        if len(self._bar_history) > self._max_bar_history:
            self._bar_history = self._bar_history[-self._max_bar_history:]
        
        for order in orders:
            if order.symbol != bar.symbol:
                log.warning(
                    f"Order symbol {order.symbol} doesn't match bar {bar.symbol}"
                )
                continue
            
            fill = self._execute_order(order, bar, cost_model, rng)
            if fill is not None:
                fills.append(fill)
                self._stats.total_fills += 1
        
        return fills
    
    def _compute_latency(self, rng: Random, cost_model=None) -> int:
        """Compute latency.

        When *cost_model* is supplied and its latency model is enabled the
        distributional sampler in ``LatencyModel`` is used (supports gamma,
        lognormal, weibull, mixture, Poisson spikes).  Otherwise the
        legacy simple jitter path is used.
        """
        # Prefer cost_model's distributional latency when available
        if cost_model is not None and hasattr(cost_model, "latency_model") and cost_model.latency_model.is_enabled:
            latency = cost_model.get_latency_ns(rng)
        elif not self.latency_config.enable_latency:
            return 0
        else:
            latency = self.latency_config.base_latency_ns
            if self.latency_config.max_jitter_ns > 0:
                latency += rng.randint(0, self.latency_config.max_jitter_ns)

        # Track stats
        self._stats.total_latency_ns += latency
        self._stats.max_latency_ns = max(self._stats.max_latency_ns, latency)
        self._stats.latency_samples += 1
        self._stats._latency_sampler.add(float(latency))

        return latency
    
    def _compute_fill_ratio(self, order: Order, bar: Bar, rng: Random) -> float:
        """
        Compute fill ratio based on bar liquidity (dollar-volume proxy).
        
        DETERMINISTIC FORMULA:
            notional = abs(order.qty) * bar.close
            liquidity = max(bar.volume * bar.close, 1e-12)  # Dollar-volume proxy
            fill_ratio = min(1.0, liquidity / (notional * liquidity_scale))
        
        This ensures that:
        - Small orders in liquid bars fill completely
        - Large orders in illiquid bars fill partially
        - No randomness: same inputs => same fill ratio
        
        Args:
            order: Order to fill
            bar: Current bar (provides volume and close price)
            rng: Seeded RNG (NOT used - deterministic formula)
            
        Returns:
            Fill ratio in [0, 1]. 1.0 means full fill.
        """
        if not self.partial_fill_config.enable_partial_fills:
            return 1.0
        
        if bar.volume <= 0:
            log.warning(f"Bar has no volume, rejecting order")
            return 0.0
        
        # DETERMINISTIC dollar-volume-based fill ratio
        # notional = abs(order.qty) * bar.close
        # liquidity = bar.volume * bar.close (dollar volume)
        # fill_ratio = liquidity / (notional * liquidity_scale)
        order_notional = order.quantity * bar.close  # Dollar notional
        bar_liquidity = max(bar.volume * bar.close, 1e-12)  # Dollar-volume proxy
        
        fill_ratio = min(1.0, bar_liquidity / (order_notional * self.partial_fill_config.liquidity_scale)) if order_notional > 0 else 1.0
        
        # Check minimum threshold
        if fill_ratio < self.partial_fill_config.min_fill_ratio:
            log.debug(f"Fill ratio {fill_ratio:.2f} below minimum {self.partial_fill_config.min_fill_ratio}")
            return 0.0
        
        return fill_ratio
    
    def _execute_order(
        self,
        order: Order,
        bar: Bar,
        cost_model: ICostModel,
        rng: Random
    ) -> Optional[Fill]:
        """Execute a single order."""
        
        if order.order_type == OrderType.MARKET:
            return self._execute_market_order(order, bar, cost_model, rng)
        elif order.order_type == OrderType.LIMIT:
            # Simple limit: fill if price touched
            return self._execute_limit_order(order, bar, cost_model, rng)
        else:
            log.warning(f"Unsupported order type: {order.order_type}")
            return None
    
    def _execute_market_order(
        self,
        order: Order,
        bar: Bar,
        cost_model: ICostModel,
        rng: Random
    ) -> Optional[Fill]:
        """Execute a market order at bar close (or open) with full realism support."""
        
        # Compute fill ratio
        fill_ratio = self._compute_fill_ratio(order, bar, rng)
        
        if fill_ratio <= 0:
            self._stats.rejected_orders += 1
            self._stats.total_unfilled_qty += order.quantity
            log.debug(f"Order rejected due to insufficient liquidity: {order.order_id}")
            return None
        
        # Compute fill quantity
        fill_quantity = order.quantity * fill_ratio
        
        if fill_ratio < 1.0:
            self._stats.partial_fills += 1
            unfilled = order.quantity - fill_quantity
            self._unfilled[order.order_id] = unfilled
            self._stats.total_unfilled_qty += unfilled
            log.debug(f"Partial fill: {fill_quantity:.6f}/{order.quantity:.6f} ({fill_ratio:.1%})")
        
        # Track fill ratio for averaging
        self._stats.total_fill_ratio += fill_ratio
        
        # Compute latency
        latency = self._compute_latency(rng, cost_model)
        fill_timestamp = bar.timestamp_ns + latency
        
        # --- Price-aware latency (A2) ---
        base_price = self._resolve_base_price(bar, fill_timestamp, rng)

        # Calculate costs (with decomposition for A3)
        slippage = cost_model.calculate_slippage(
            base_price, fill_quantity, order.side, bar, rng
        )
        spread_cost = cost_model.calculate_spread_cost(base_price, order.side)

        # Track slippage in bps for distribution reporting
        if base_price > 0:
            self._stats._slippage_sampler.add(slippage / base_price * 10_000)
        
        # Adjust price: buy pays more, sell receives less
        if order.is_buy:
            fill_price = base_price + slippage + spread_cost
        else:
            fill_price = base_price - slippage - spread_cost
        
        # Calculate fee
        notional = fill_price * fill_quantity
        fee = cost_model.calculate_fee(notional, order.side, is_maker=False)
        
        # --- Build cost breakdown (A3) ---
        cost_bd = {
            "fee_cost_quote": fee,
            "spread_cost_quote": spread_cost * fill_quantity,
            "slippage_cost_quote": slippage * fill_quantity,
            "total_cost_quote": fee + (spread_cost + slippage) * fill_quantity,
        }
        
        return Fill(
            order_id=order.order_id,
            symbol=order.symbol,
            side=order.side,
            fill_price=fill_price,
            fill_quantity=fill_quantity,
            fee=fee,
            slippage=slippage + spread_cost,
            timestamp_ns=fill_timestamp,
            metadata={
                "execution_type": "market",
                "base_price": base_price,
                "fill_ratio": fill_ratio,
                "latency_ns": latency,
                "partial": fill_ratio < 1.0,
                "cost_breakdown": cost_bd,
            }
        )
    
    def _execute_limit_order(
        self,
        order: Order,
        bar: Bar,
        cost_model: ICostModel,
        rng: Random
    ) -> Optional[Fill]:
        """
        Execute a limit order if price was touched.
        
        A5: If marketable_limit_is_taker is enabled, aggressive LIMITs that
        would immediately cross the spread are treated as taker orders.
        """
        if order.price is None:
            log.warning("Limit order without price")
            return None
        
        limit_price = order.price
        tc = self._tc_config  # shortcut (may be None)
        marketable_taker = tc is not None and tc.marketable_limit_is_taker
        
        # --- A5: marketable-limit detection ---
        is_marketable = False
        if marketable_taker:
            spread_bps = tc.spread_bps if tc else 2.0
            mid = bar.close  # proxy for midprice
            half_spread = spread_bps / 20_000
            est_ask = mid * (1 + half_spread)
            est_bid = mid * (1 - half_spread)
            if order.is_buy and limit_price >= est_ask:
                is_marketable = True
            elif order.is_sell and limit_price <= est_bid:
                is_marketable = True
        
        if is_marketable:
            # Treat as market order (taker fee + spread + slippage)
            fill = self._execute_market_order(order, bar, cost_model, rng)
            if fill is not None:
                fill.metadata["execution_type"] = "limit_marketable"
                fill.metadata["original_limit_price"] = limit_price
            return fill
        
        # --- Standard passive limit logic ---
        # Check if limit was touched
        if order.is_buy:
            if bar.low > limit_price:
                return None
            fill_price = limit_price
        else:
            if bar.high < limit_price:
                return None
            fill_price = limit_price
        
        # Compute fill ratio
        fill_ratio = self._compute_fill_ratio(order, bar, rng)
        if fill_ratio <= 0:
            self._stats.rejected_orders += 1
            return None
        
        fill_quantity = order.quantity * fill_ratio
        if fill_ratio < 1.0:
            self._stats.partial_fills += 1
            self._unfilled[order.order_id] = order.quantity - fill_quantity
        
        # Limit orders are maker orders
        notional = fill_price * fill_quantity
        fee = cost_model.calculate_fee(notional, order.side, is_maker=True)
        
        # Compute latency
        latency = self._compute_latency(rng, cost_model)
        fill_timestamp = bar.timestamp_ns + latency
        
        # Cost breakdown (A3) - passive limit: no slippage/spread
        cost_bd = {
            "fee_cost_quote": fee,
            "spread_cost_quote": 0.0,
            "slippage_cost_quote": 0.0,
            "total_cost_quote": fee,
        }
        
        return Fill(
            order_id=order.order_id,
            symbol=order.symbol,
            side=order.side,
            fill_price=fill_price,
            fill_quantity=fill_quantity,
            fee=fee,
            slippage=0.0,
            timestamp_ns=fill_timestamp,
            metadata={
                "execution_type": "limit",
                "limit_price": limit_price,
                "fill_ratio": fill_ratio,
                "latency_ns": latency,
                "partial": fill_ratio < 1.0,
                "cost_breakdown": cost_bd,
            }
        )
    
    def get_unfilled_quantity(self, order_id: str) -> float:
        """Get unfilled quantity for an order."""
        return self._unfilled.get(order_id, 0.0)
    
    def get_stats(self) -> ExecutionStats:
        """Get execution statistics."""
        return self._stats
    
    def reset(self) -> None:
        """Reset execution model state."""
        self._stats = ExecutionStats()
        self._unfilled.clear()
        self._bar_history.clear()
    
    @property
    def fill_count(self) -> int:
        """Number of fills executed."""
        return self._stats.total_fills


class PartialFillExecutionModel(SimpleExecutionModel):
    """
    Execution model with partial fills enabled by default.
    
    This is a convenience class that enables partial fills out of the box.
    """
    
    def __init__(
        self,
        use_bar_close: bool = True,
        liquidity_scale: float = 10.0,
        min_fill_ratio: float = 0.1,
        enable_latency: bool = False,
        base_latency_ns: int = 0,
        max_jitter_ns: int = 0,
    ):
        super().__init__(
            use_bar_close=use_bar_close,
            partial_fill_config=PartialFillConfig(
                enable_partial_fills=True,
                liquidity_scale=liquidity_scale,
                min_fill_ratio=min_fill_ratio,
            ),
            latency_config=LatencyConfig(
                enable_latency=enable_latency,
                base_latency_ns=base_latency_ns,
                max_jitter_ns=max_jitter_ns,
            ),
        )


class LimitExecutionModel(SimpleExecutionModel):
    """
    Extended execution model with better limit order handling.
    
    Features:
    - Pending limit order queue
    - Partial fills
    - Latency simulation
    """
    
    def __init__(
        self,
        use_bar_close: bool = True,
        partial_fill_config: Optional[PartialFillConfig] = None,
        latency_config: Optional[LatencyConfig] = None,
    ):
        super().__init__(use_bar_close, partial_fill_config, latency_config)
        self._pending_limits: List[Order] = []
    
    def process_orders(
        self,
        orders: List[Order],
        bar: Bar,
        portfolio: "Any",
        cost_model: ICostModel,
        rng: Random
    ) -> List[Fill]:
        """Process orders including any pending limit orders."""
        fills = []
        
        # First check pending limit orders
        remaining_pending = []
        for order in self._pending_limits:
            if order.symbol != bar.symbol:
                remaining_pending.append(order)
                continue
            
            fill = self._execute_limit_order(order, bar, cost_model, rng)
            if fill is not None:
                fills.append(fill)
                self._stats.total_fills += 1
            else:
                remaining_pending.append(order)
        
        self._pending_limits = remaining_pending
        
        # Then process new orders
        for order in orders:
            if order.symbol != bar.symbol:
                continue
            
            if order.order_type == OrderType.LIMIT:
                # Try to fill immediately, otherwise queue
                fill = self._execute_limit_order(order, bar, cost_model, rng)
                if fill is not None:
                    fills.append(fill)
                    self._stats.total_fills += 1
                else:
                    self._pending_limits.append(order)
            else:
                fill = self._execute_order(order, bar, cost_model, rng)
                if fill is not None:
                    fills.append(fill)
                    self._stats.total_fills += 1
        
        return fills
    
    def reset(self) -> None:
        """Reset execution model state."""
        super().reset()
        self._pending_limits.clear()
    
    @property
    def pending_order_count(self) -> int:
        """Number of pending limit orders."""
        return len(self._pending_limits)
