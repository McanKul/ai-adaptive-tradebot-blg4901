"""
Backtest/engine.py
==================
Core backtest engine that orchestrates all components.

DEPENDENCY MAP (who depends on whom):
=====================================
BacktestEngine
├── Interfaces/market_data.py      (Tick, Bar)
├── Interfaces/orders.py           (Order, Fill)
├── Interfaces/IStrategy.py        (IStrategy, StrategyDecision)
├── Interfaces/strategy_adapter.py (StrategyContext, IBacktestStrategy)
├── Backtest/tick_store.py         (TickStore)
├── Backtest/disk_streamer.py      (DiskTickStreamer)
├── Backtest/bar_builder.py        (TimeBarBuilder, etc.)
├── Backtest/portfolio.py          (Portfolio)
├── Backtest/execution_models.py   (SimpleExecutionModel)
├── Backtest/cost_models.py        (CompositeCostModel)
├── Backtest/risk.py               (BasicRiskManager)
├── Backtest/metrics.py            (MetricsSink)
└── utils/bar_store.py             (BarStore - existing)

Data flow:
==========
DiskTickStreamer (yields Tick)
    │
    v
BarBuilder (on_tick -> optional Bar)
    │
    v
BarStore (maintains history)
    │
    v
Strategy.on_bar(bar, ctx) -> list[Order]
    │
    v
RiskManager.pre_trade_check(order)
    │
    v
ExecutionModel.process_orders() -> list[Fill]
    │
    v
Portfolio.apply_fill()
    │
    v
MetricsSink (records everything)

CRITICAL:
- Strategy ONLY sees Bars, NEVER Ticks
- Backtest uses disk replay ONLY (no network)
- Deterministic: same seed + same data = same results
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional, Type, Callable
from random import Random
import logging

from Interfaces.market_data import Tick, Bar
from Interfaces.orders import Order
from Interfaces.metrics_interface import BacktestResult
from Interfaces.strategy_adapter import StrategyContext, IBacktestStrategy

from Backtest.tick_store import TickStore, TickStoreConfig
from Backtest.disk_streamer import DiskTickStreamer, DiskStreamerConfig
from Backtest.bar_builder import (
    BarBuilder, TimeBarBuilder, TickBarBuilder, 
    VolumeBarBuilder, DollarBarBuilder, create_bar_builder
)
from Backtest.portfolio import Portfolio, LeverageMode, MarginConfig
from Backtest.execution_models import SimpleExecutionModel, PartialFillConfig, LatencyConfig, ExecutionStats
from Backtest.cost_models import CompositeCostModel, create_cost_model, create_cost_model_from_config
from Backtest.risk import BasicRiskManager, RiskLimits
from Backtest.metrics import MetricsSink
from Backtest.exit_manager import ExitManager
from Backtest.realism_config import RealismConfig

from utils.bar_store import BarStore
from utils.levels import SupportResistanceTracker, SupportResistanceResult


# ---------------------------------------------------------------------------
# CSV rate-series loader (for funding / borrow series mode)
# ---------------------------------------------------------------------------

def _load_rate_series(csv_path: str, timestamp_unit: str = "ms") -> List[tuple]:
    """Load a CSV with (timestamp, rate) rows sorted by time.

    Returns list of (timestamp_ns, rate) tuples sorted ascending.
    CSV must have columns: timestamp, rate  (or funding_rate / annual_rate).
    """
    import csv as _csv
    rows: List[tuple] = []
    if not csv_path:
        return rows
    multiplier = 1_000_000 if timestamp_unit == "ms" else 1  # ms → ns
    with open(csv_path, "r", newline="") as f:
        reader = _csv.DictReader(f)
        for row in reader:
            ts_raw = row.get("timestamp", "")
            rate_raw = row.get("rate") or row.get("funding_rate") or row.get("annual_rate") or "0"
            ts_ns = int(float(ts_raw) * multiplier) if ts_raw else 0
            rate = float(rate_raw)
            rows.append((ts_ns, rate))
    rows.sort(key=lambda x: x[0])
    return rows


def _lookup_rate(series: List[tuple], ts_ns: int, default: float) -> float:
    """Binary-search for the most-recent rate at or before *ts_ns*."""
    if not series:
        return default
    import bisect
    idx = bisect.bisect_right(series, (ts_ns, float("inf"))) - 1
    if idx < 0:
        return default
    return series[idx][1]


def _stable_symbol_hash(symbol: str) -> int:
    """Deterministic hash for a symbol string, stable across Python sessions.

    Uses a simple polynomial rolling hash (same result on every run,
    unlike Python's built-in ``hash()`` which is randomised).
    """
    h = 0
    for c in symbol:
        h = (h * 31 + ord(c)) & 0x7FFFFFFF
    return h

log = logging.getLogger(__name__)


@dataclass
class EngineConfig:
    """Configuration for BacktestEngine."""
    # Data
    tick_data_dir: str
    symbols: List[str]
    start_ts_ns: Optional[int] = None
    end_ts_ns: Optional[int] = None
    
    # Bar building
    bar_type: str = "time"  # 'time', 'tick', 'volume', 'dollar'
    timeframe: str = "1m"
    tick_threshold: int = 100
    volume_threshold: float = 1000.0
    dollar_threshold: float = 100000.0
    
    # Portfolio
    initial_capital: float = 10000.0
    margin_requirement: float = 0.1
    
    # NEW: Leverage mode
    leverage_mode: str = "spot"  # "spot" or "margin"
    leverage: float = 10.0  # Max leverage for margin mode
    maintenance_margin_ratio: float = 0.5  # Maintenance = initial * this ratio
    
    # Costs (in basis points)
    taker_fee_bps: float = 4.0
    maker_fee_bps: float = 2.0
    slippage_bps: float = 1.0
    spread_bps: float = 2.0
    
    # NEW: Latency simulation
    latency_ns: int = 0  # Base latency in nanoseconds
    latency_jitter_ns: int = 0  # Max random jitter
    
    # NEW: Partial fill settings
    enable_partial_fills: bool = False
    liquidity_scale: float = 10.0  # Higher = easier fills
    min_fill_ratio: float = 0.1  # Minimum fill to accept
    
    # Risk
    max_position_size: float = 10.0
    max_position_notional: float = 100000.0
    max_daily_loss: float = 1000.0
    max_drawdown: float = 0.2
    
    # Execution
    use_bar_close: bool = True  # If False, use bar open
    
    # NEW: Close positions at end (AFML-compliant bet accounting)
    close_positions_at_end: bool = False
    
    # NEW: Tick-level TP/SL exit checking
    # When enabled, TP/SL are checked on every tick, not just at bar close
    # This enables true intrabar exit simulation
    enable_tick_exit: bool = True  # Default ON for realistic exits
    
    # Determinism
    random_seed: int = 42
    
    # BarStore
    bar_store_maxlen: int = 600
    
    # Exit rules (strategy-independent TP/SL/trailing safety net)
    # These are margin-return percentages when leverage > 1
    # e.g. tp_pct=0.04 with leverage=10 → exit when price moves 0.4% (4% margin return)
    tp_pct: Optional[float] = None
    sl_pct: Optional[float] = None
    trailing_stop_pct: Optional[float] = None
    max_holding_bars: Optional[int] = None

    # NEW: Support/Resistance tracking
    enable_sr_tracking: bool = False
    sr_window: int = 50
    sr_swing_window: int = 5
    sr_num_levels: int = 3

    # NEW: Realism configuration (Part A refactor)
    realism: RealismConfig = field(default_factory=RealismConfig)


class BacktestEngine:
    """
    Core backtest engine.
    
    Orchestrates the entire backtest pipeline:
    1. Load tick data from disk
    2. Build bars from ticks
    3. Feed bars to strategy (strategy never sees ticks!)
    4. Execute orders through risk and execution models
    5. Track portfolio and metrics
    
    TICK-LEVEL TP/SL EXIT:
    ======================
    When enable_tick_exit=True (default), the engine checks TP/SL conditions
    on EVERY tick, not just at bar close. This enables true intrabar exit
    simulation:
    
    Tick arrives -> update price -> check TP/SL -> if triggered, execute exit
    
    This means exits happen at the actual tick price where TP/SL was hit,
    not at bar close, which is more realistic.
    
    Usage:
        config = EngineConfig(...)
        engine = BacktestEngine(config)
        result = engine.run(strategy)
    """
    
    def __init__(self, config: EngineConfig):
        """
        Initialize BacktestEngine.
        
        Args:
            config: EngineConfig with all settings
        """
        self.config = config
        self._initialized = False
        
        # Components (created in _initialize)
        self.tick_store: Optional[TickStore] = None
        self.streamer: Optional[DiskTickStreamer] = None
        self.bar_builders: Dict[str, BarBuilder] = {}
        self.bar_store: Optional[BarStore] = None
        self.portfolio: Optional[Portfolio] = None
        self.execution_model: Optional[SimpleExecutionModel] = None
        self.cost_model: Optional[CompositeCostModel] = None  # default / legacy
        self.cost_models: Dict[str, CompositeCostModel] = {}   # per-symbol (Part A)
        self.risk_manager: Optional[BasicRiskManager] = None
        self.metrics: Optional[MetricsSink] = None
        self.rng: Optional[Random] = None
        self._symbol_rngs: Dict[str, Random] = {}   # per-symbol RNG streams
        
        # NEW: Exit manager for tick-level TP/SL
        self.exit_manager: Optional[ExitManager] = None
        
        # NEW: Support/Resistance tracker (optional)
        self.sr_trackers: Dict[str, SupportResistanceTracker] = {}
        
        # State
        self._current_bar_ts: Dict[str, int] = {}  # symbol -> last bar timestamp
        self._tick_count = 0
        self._bar_count = 0
        self._order_count = 0  # Total orders submitted
        self._risk_rejected_orders = 0  # Orders rejected by risk manager
        self._tick_exit_count = 0  # Exits triggered at tick level
        
        # NEW: Liquidation tracking
        self._liquidation_triggered = False
        
        # NEW: Active strategy reference (for exit manager)
        self._active_strategy = None
    
    def _initialize(self) -> None:
        """Initialize all components."""
        if self._initialized:
            return
        
        log.info(f"Initializing BacktestEngine with seed {self.config.random_seed}")
        
        # RNG for determinism
        self.rng = Random(self.config.random_seed)

        # Per-symbol seeded RNG streams for multi-symbol stability.
        # seed_sym = base_seed + deterministic_hash(symbol)
        for sym in self.config.symbols:
            sym_seed = self.config.random_seed + _stable_symbol_hash(sym)
            self._symbol_rngs[sym] = Random(sym_seed)
        
        # Tick store
        self.tick_store = TickStore(TickStoreConfig(
            data_dir=self.config.tick_data_dir,
            validate_order=True,
            deduplicate=False,
        ))
        
        # Disk streamer
        self.streamer = DiskTickStreamer(
            self.tick_store,
            DiskStreamerConfig(
                symbols=self.config.symbols,
                start_ts_ns=self.config.start_ts_ns,
                end_ts_ns=self.config.end_ts_ns,
            )
        )
        
        # Bar builders (one per symbol)
        for symbol in self.config.symbols:
            self.bar_builders[symbol] = create_bar_builder(
                symbol=symbol,
                bar_type=self.config.bar_type,
                timeframe=self.config.timeframe,
                tick_threshold=self.config.tick_threshold,
                volume_threshold=self.config.volume_threshold,
                dollar_threshold=self.config.dollar_threshold,
            )
        
        # Bar store (existing utility)
        self.bar_store = BarStore(maxlen=self.config.bar_store_maxlen)
        
        # Portfolio with leverage mode support
        leverage_mode = LeverageMode.MARGIN if self.config.leverage_mode == "margin" else LeverageMode.SPOT
        margin_config = MarginConfig(
            leverage=self.config.leverage,
            maintenance_margin_ratio=self.config.maintenance_margin_ratio,
        )
        
        self.portfolio = Portfolio(
            initial_cash=self.config.initial_capital,
            margin_requirement=self.config.margin_requirement,
            leverage_mode=leverage_mode,
            margin_config=margin_config,
        )
        
        # Cost model – prefer realism config if present, else legacy
        rc = self.config.realism
        tc = rc.transaction_costs
        use_new_factory = (
            tc.slippage_model != "fixed"
            or tc.price_latency_mode != "timestamp_only"
            or tc.marketable_limit_is_taker
            or tc.per_symbol_overrides
        )
        if use_new_factory:
            # Build per-symbol cost models using for_symbol() overrides
            for sym in self.config.symbols:
                sym_tc = tc.for_symbol(sym)
                self.cost_models[sym] = create_cost_model_from_config(sym_tc)
            self.cost_model = self.cost_models.get(
                self.config.symbols[0],
                create_cost_model_from_config(tc),
            )
        else:
            self.cost_model = create_cost_model(
                taker_fee_bps=self.config.taker_fee_bps,
                maker_fee_bps=self.config.maker_fee_bps,
                slippage_bps=self.config.slippage_bps,
                spread_bps=self.config.spread_bps,
                latency_ns=self.config.latency_ns,
                latency_jitter_ns=self.config.latency_jitter_ns,
            )
        
        # Execution model with partial fill and latency support
        partial_fill_config = PartialFillConfig(
            enable_partial_fills=self.config.enable_partial_fills,
            liquidity_scale=self.config.liquidity_scale,
            min_fill_ratio=self.config.min_fill_ratio,
        )
        latency_config = LatencyConfig(
            enable_latency=self.config.latency_ns > 0 or self.config.latency_jitter_ns > 0,
            base_latency_ns=self.config.latency_ns,
            max_jitter_ns=self.config.latency_jitter_ns,
        )
        self.execution_model = SimpleExecutionModel(
            use_bar_close=self.config.use_bar_close,
            partial_fill_config=partial_fill_config,
            latency_config=latency_config,
            realism_config=rc,
        )
        
        # Risk manager
        self.risk_manager = BasicRiskManager(RiskLimits(
            max_position_size=self.config.max_position_size,
            max_position_notional=self.config.max_position_notional,
            max_daily_loss=self.config.max_daily_loss,
            max_drawdown=self.config.max_drawdown,
            max_leverage=self.config.leverage,
            enforce_margin_check=(leverage_mode == LeverageMode.MARGIN),
        ))
        
        # Metrics
        self.metrics = MetricsSink()
        
        # NEW: Support/Resistance trackers (one per symbol)
        if self.config.enable_sr_tracking:
            for symbol in self.config.symbols:
                self.sr_trackers[symbol] = SupportResistanceTracker(
                    window=self.config.sr_window,
                    swing_window=self.config.sr_swing_window,
                    num_levels=self.config.sr_num_levels,
                )
        
        self._initialized = True
        log.info(f"BacktestEngine initialized (leverage_mode={leverage_mode.value})")

    def _get_cost_model(self, symbol: str) -> CompositeCostModel:
        """Return the per-symbol cost model, falling back to the default."""
        return self.cost_models.get(symbol, self.cost_model)

    def _get_rng(self, symbol: str) -> Random:
        """Return the per-symbol RNG, falling back to the default."""
        return self._symbol_rngs.get(symbol, self.rng)

    def run(self, strategy: IBacktestStrategy, sizing_config=None) -> BacktestResult:
        """
        Run backtest with given strategy.
        
        Args:
            strategy: Strategy implementing IBacktestStrategy
            sizing_config: Optional SizingConfig for position sizing.
                          If None, uses strategy's position_size attribute.
            
        Returns:
            BacktestResult with all metrics
        """
        self._initialize()
        self._reset_state()
        
        # Store sizing config for use in _process_bar
        self._sizing_config = sizing_config
        
        # Reset sizing log counter
        from Interfaces.strategy_adapter import reset_sizing_log_counter
        reset_sizing_log_counter()
        
        # Reset strategy
        strategy.reset()
        
        # Set daily start equity for risk manager
        self.risk_manager.set_daily_start_equity(self.config.initial_capital)
        
        start_ts: Optional[int] = None
        end_ts: Optional[int] = None
        last_day: Optional[int] = None
        last_bar: Optional[Bar] = None  # Track last bar for close_positions_at_end
        
        log.info(f"Starting backtest for symbols: {self.config.symbols}")
        
        # Store active strategy for exit manager access
        self._active_strategy = strategy
        
        # Initialize exit manager from strategy if available
        if self.config.enable_tick_exit:
            self._setup_exit_manager(strategy)
        
        # Main loop: process ticks
        for tick in self.streamer.iter_ticks():
            self._tick_count += 1
            
            # Track time range
            if start_ts is None:
                start_ts = tick.timestamp_ns
            end_ts = tick.timestamp_ns
            
            # Day boundary check for daily reset
            current_day = tick.timestamp_ns // (86400 * 1_000_000_000)
            if last_day is not None and current_day != last_day:
                self.risk_manager.reset_daily()
                self.risk_manager.set_daily_start_equity(self.portfolio.equity())
            last_day = current_day
            
            # TICK-LEVEL TP/SL CHECK (before building bar)
            # This enables intrabar exit: exit happens at tick price, not bar close
            if self.config.enable_tick_exit and self.exit_manager is not None:
                self._check_tick_exit(tick)
            
            # Build bar from tick
            builder = self.bar_builders.get(tick.symbol)
            if builder is None:
                continue
            
            bar = builder.on_tick(tick)
            
            # If bar completed, process it
            if bar is not None:
                self._process_bar(bar, strategy)
                last_bar = bar
        
        # Flush any remaining partial bars
        for symbol, builder in self.bar_builders.items():
            bar = builder.flush()
            if bar is not None:
                self._process_bar(bar, strategy)
                last_bar = bar
        
        # NEW: Close positions at end if configured (AFML-compliant bet accounting)
        forced_close_orders = 0
        forced_close_fills = 0
        if self.config.close_positions_at_end and last_bar is not None:
            forced_close_orders, forced_close_fills = self._close_all_positions(last_bar)
        
        # Finalize metrics
        if start_ts is None:
            start_ts = 0
        if end_ts is None:
            end_ts = 0
        
        # CRITICAL: Pass portfolio's max_drawdown to ensure consistency between
        # kill-switch checks and reported metrics
        # Also pass leverage stats for AFML metrics
        leverage_stats = self.portfolio.get_leverage_stats()
        
        # Get execution stats for latency reporting
        exec_stats = self.execution_model.get_stats()
        
        result = self.metrics.finalize(
            initial_capital=self.config.initial_capital,
            final_equity=self.portfolio.equity(),
            start_ts=start_ts,
            end_ts=end_ts,
            portfolio_max_drawdown=self.portfolio.max_drawdown,
            leverage_stats=leverage_stats,
            leverage_mode=self.config.leverage_mode,
            liquidation_count=len(self.portfolio.liquidations),
        )
        
        # Add execution stats to metadata
        result.metadata["execution_stats"] = exec_stats.to_dict()
        result.metadata["forced_close_orders"] = forced_close_orders
        result.metadata["forced_close_fills"] = forced_close_fills
        result.metadata["close_positions_at_end"] = self.config.close_positions_at_end
        
        # Add order and fill counts (normal orders only - forced close tracked separately)
        result.metadata["order_count"] = self._order_count
        result.metadata["fill_count"] = exec_stats.total_fills
        result.metadata["total_traded_notional"] = self.portfolio.total_traded_notional
        result.metadata["tick_exit_count"] = self._tick_exit_count
        result.metadata["enable_tick_exit"] = self.config.enable_tick_exit
        result.metadata["risk_rejected_orders"] = self._risk_rejected_orders
        
        # Add fee model config for clarity in reporting
        result.metadata["fee_config"] = {
            "taker_fee_bps": self.config.taker_fee_bps,
            "maker_fee_bps": self.config.maker_fee_bps,
            "slippage_bps": self.config.slippage_bps,
            "spread_bps": self.config.spread_bps,
        }
        
        # Add partial fill feature flag for truthful reporting
        result.metadata["partial_fills_enabled"] = self.config.enable_partial_fills
        
        # Add latency stats if latency was enabled
        if exec_stats.latency_samples > 0:
            result.metadata["avg_latency_ns"] = exec_stats.avg_latency_ns
            result.metadata["max_latency_ns"] = exec_stats.max_latency_ns
            result.metadata["p95_latency_ns"] = exec_stats.p95_latency_ns
            result.metadata["p99_latency_ns"] = exec_stats.p99_latency_ns

        # Slippage distribution stats
        if exec_stats._slippage_sampler:
            result.metadata["avg_slippage_bps"] = exec_stats.avg_slippage_bps
            result.metadata["p95_slippage_bps"] = exec_stats.p95_slippage_bps
            result.metadata["p99_slippage_bps"] = exec_stats.p99_slippage_bps

        # Price-shift stats from latency-proportional fill adjustment
        result.metadata["fills_bar_shifted"] = exec_stats.fills_bar_shifted
        result.metadata["avg_price_shift"] = exec_stats.avg_price_shift
        result.metadata["p95_price_shift"] = exec_stats.p95_price_shift
        
        # Add partial fill stats if enabled
        if self.config.enable_partial_fills:
            result.metadata["partial_fills"] = exec_stats.partial_fills
            result.metadata["rejected_orders"] = exec_stats.rejected_orders
            result.metadata["total_unfilled_qty"] = exec_stats.total_unfilled_qty
            result.metadata["avg_fill_ratio"] = exec_stats.avg_fill_ratio
        
        log.info(
            f"Backtest complete: {self._tick_count} ticks, {self._bar_count} bars, "
            f"return={result.total_return_pct:.2%}, sharpe={result.sharpe_ratio:.2f}"
        )
        
        return result
    
    def _close_all_positions(self, last_bar: Bar) -> tuple:
        """
        Close all open positions at end of backtest (AFML-compliant).
        
        Args:
            last_bar: Last bar for pricing and timestamp
            
        Returns:
            Tuple of (forced_close_orders, forced_close_fills)
        """
        from Interfaces.orders import Order, OrderSide, OrderType
        
        forced_order_count = 0
        forced_fill_count = 0
        
        for symbol in self.config.symbols:
            pos = self.portfolio.position_quantity(symbol)
            if abs(pos) < 1e-10:
                continue
            
            log.info(f"Forced close at end: {symbol} position={pos:.6f}")
            
            # Create close order
            if pos > 0:
                side = OrderSide.SELL
            else:
                side = OrderSide.BUY
            
            close_order = Order(
                symbol=symbol,
                side=side,
                order_type=OrderType.MARKET,
                quantity=abs(pos),
                timestamp_ns=last_bar.timestamp_ns,
                strategy_id="FORCED_CLOSE",
                reduce_only=True,
                metadata={"forced_close": True},
            )
            
            # Count the order submission
            forced_order_count += 1
            
            # Create a synthetic bar for execution (use last bar's close)
            exec_bar = Bar(
                symbol=symbol,
                timeframe=last_bar.timeframe,
                timestamp_ns=last_bar.timestamp_ns,
                open=last_bar.close,
                high=last_bar.close,
                low=last_bar.close,
                close=last_bar.close,
                volume=last_bar.volume,
            )
            
            # Execute (bypass risk manager for forced close)
            fills = self.execution_model.process_orders(
                [close_order],
                exec_bar,
                self.portfolio,
                self._get_cost_model(symbol),
                self._get_rng(symbol),
            )
            
            for fill in fills:
                pnl = self.portfolio.apply_fill(fill)
                self.metrics.on_fill(fill, self.portfolio.equity())
                forced_fill_count += 1
        
        return (forced_order_count, forced_fill_count)
    
    def _process_bar(self, bar: Bar, strategy: IBacktestStrategy) -> None:
        """
        Process a completed bar.
        
        This is where the magic happens:
        1. Add bar to BarStore
        2. Update portfolio prices
        3. Check liquidation (margin mode)
        4. Call strategy (bar only, never ticks!)
        5. Risk check and execute orders
        6. Record metrics
        """
        self._bar_count += 1
        
        # Add to bar store (compatible format)
        self.bar_store.add_bar(bar.symbol, bar.timeframe, bar.to_dict())
        
        # Update portfolio price
        self.portfolio.update_price(bar.symbol, bar.close)
        
        # --- Funding & borrow cost charging (A6) ---
        self._charge_funding_and_borrow(bar)
        
        # Sample leverage stats for AFML metrics
        self.portfolio.sample_leverage_stats()
        
        # Update S/R tracker if enabled
        sr_result = None
        if bar.symbol in self.sr_trackers:
            sr_result = self.sr_trackers[bar.symbol].on_bar(bar.high, bar.low, bar.close)
        
        # Update drawdown in portfolio (for internal tracking)
        current_drawdown = self.portfolio.update_drawdown()
        
        # CRITICAL: Use the SAME drawdown for kill-switch check
        # The metrics will also compute drawdown from equity_curve in on_bar
        # We check kill-switch BEFORE on_bar so we use portfolio.max_drawdown
        self.risk_manager.check_drawdown(self.portfolio)
        
        # NEW: Check liquidation in margin mode
        if self.portfolio.is_margin_mode and not self._liquidation_triggered:
            if self.portfolio.is_liquidation_triggered():
                self._handle_liquidation(bar)
        
        # Reset per-bar counters
        self.risk_manager.on_new_bar()
        
        # Create strategy context with S/R data if available
        ctx = StrategyContext(
            symbol=bar.symbol,
            timeframe=bar.timeframe,
            bar_store=self.bar_store,
            portfolio=self.portfolio.get_snapshot(),
            position=self.portfolio.position_quantity(bar.symbol),
            equity=self.portfolio.equity(),
            cash=self.portfolio.cash,
            timestamp_ns=bar.timestamp_ns,
        )
        
        # Add S/R levels to metadata if available
        if sr_result is not None:
            ctx.metadata["support_resistance"] = {
                "levels": sr_result.levels,
                "supports": sr_result.supports,
                "resistances": sr_result.resistances,
            }
        
        # Add position info for exit management
        ctx.metadata["avg_entry_price"] = self.portfolio.position_avg_entry(bar.symbol)
        ctx.metadata["bar_index"] = self._bar_count
        
        # CRITICAL: Strategy receives BAR, never Tick
        # Use strategy adapter to handle both legacy and new strategy interfaces
        from Interfaces.strategy_adapter import adapt_strategy_output, apply_sizing_to_orders
        from Interfaces.IStrategy import StrategyDecision
        
        result = strategy.on_bar(bar, ctx)
        
        # Handle different return types
        if isinstance(result, StrategyDecision):
            orders = result.orders
        elif isinstance(result, list):
            orders = result
        else:
            orders = []
        
        # Apply sizing configuration to orders (compute qty at current price)
        if orders and self._sizing_config is not None:
            orders = apply_sizing_to_orders(
                orders=orders,
                sizing_config=self._sizing_config,
                current_price=bar.close,
                log_first_n=3,  # Log first 3 trades for visibility
            )
        
        # Count all orders submitted (before risk filtering)
        self._order_count += len(orders)
        
        # Filter orders through risk manager
        approved_orders: List[Order] = []
        risk_rejected = 0
        for order in orders:
            order.timestamp_ns = bar.timestamp_ns
            self.metrics.on_order(order)
            
            if self.risk_manager.pre_trade_check(order, self.portfolio, bar):
                approved_orders.append(order)
            else:
                risk_rejected += 1
        
        self._risk_rejected_orders += risk_rejected
        
        # Execute approved orders
        if approved_orders:
            fills = self.execution_model.process_orders(
                approved_orders,
                bar,
                self.portfolio,
                self._get_cost_model(bar.symbol),
                self._get_rng(bar.symbol),
            )
            
            # Apply fills to portfolio
            for fill in fills:
                # Track position before fill for entry detection
                pos_before = self.portfolio.position_quantity(fill.symbol)
                
                pnl = self.portfolio.apply_fill(fill)
                self.risk_manager.update_daily_pnl(pnl)
                self.metrics.on_fill(fill, self.portfolio.equity())
                
                # Register entry with exit manager if we opened/increased a position
                if self.exit_manager is not None and self.config.enable_tick_exit:
                    pos_after = self.portfolio.position_quantity(fill.symbol)
                    # Opened new position or increased existing
                    if abs(pos_before) < 1e-10 and abs(pos_after) > 1e-10:
                        # New position opened
                        self.exit_manager.register_entry(
                            symbol=fill.symbol,
                            quantity=pos_after,
                            entry_price=fill.fill_price,
                            bar_index=self._bar_count,
                            timestamp_ns=fill.timestamp_ns,
                        )
                    elif abs(pos_after) > abs(pos_before):
                        # Position increased - update exit manager
                        avg_entry = self.portfolio.position_avg_entry(fill.symbol)
                        self.exit_manager.update_position(
                            symbol=fill.symbol,
                            quantity=pos_after,
                            avg_entry_price=avg_entry,
                        )
        
        # Record metrics
        self.metrics.on_bar(
            bar,
            self.portfolio.equity(),
            self.portfolio.position_quantity(bar.symbol),
        )
    
    def _charge_funding_and_borrow(self, bar: Bar) -> None:
        """
        Apply funding-rate and borrow-interest costs (A6).

        Both features default OFF (backward compatible).
        Costs are subtracted from portfolio cash and tracked in MetricsSink.

        Funding supports ``funding_rate_mode="series"`` — loads rates from a
        CSV and looks up the most-recent rate at each charge.
        """
        rc = self.config.realism
        HOUR_NS = 3_600_000_000_000

        # --- Funding ---
        fc = rc.funding
        if fc.enabled:
            interval_ns = fc.funding_interval_hours * HOUR_NS
            # Simple heuristic: charge once per interval based on bar timestamps
            if not hasattr(self, '_last_funding_ts'):
                self._last_funding_ts = 0
            # Lazy-load funding series (once)
            if not hasattr(self, '_funding_series'):
                if fc.funding_rate_mode == "series" and fc.series_csv_path:
                    self._funding_series = _load_rate_series(
                        fc.series_csv_path, getattr(fc, "timestamp_unit", "ms"))
                else:
                    self._funding_series = []

            if bar.timestamp_ns - self._last_funding_ts >= interval_ns:
                self._last_funding_ts = bar.timestamp_ns
                # Determine rate for this interval
                if self._funding_series:
                    rate = _lookup_rate(self._funding_series, bar.timestamp_ns, fc.funding_rate)
                else:
                    rate = fc.funding_rate
                for sym in self.config.symbols:
                    pos = self.portfolio.position_quantity(sym)
                    if abs(pos) < 1e-10:
                        continue
                    price = self.portfolio._last_prices.get(sym, bar.close)
                    notional = pos * price  # signed
                    payment = notional * rate
                    # Long pays if rate > 0, short receives (and vice versa)
                    self.portfolio.cash -= payment
                    self.metrics._total_funding_cost += abs(payment)

        # --- Borrow ---
        bc = rc.borrow
        if bc.enabled and bc.annual_borrow_rate > 0:
            # Determine dt
            if not hasattr(self, '_last_borrow_ts'):
                self._last_borrow_ts = 0
            # Lazy-load borrow series (once)
            if not hasattr(self, '_borrow_series'):
                if getattr(bc, "series_csv_path", "") and bc.series_csv_path:
                    self._borrow_series = _load_rate_series(bc.series_csv_path, "ms")
                else:
                    self._borrow_series = []

            if bc.charge_interval == "daily":
                threshold_ns = 24 * HOUR_NS
            elif bc.charge_interval == "hourly":
                threshold_ns = HOUR_NS
            else:  # "bar" — charge every bar
                threshold_ns = 0

            elapsed_ns = bar.timestamp_ns - self._last_borrow_ts if self._last_borrow_ts > 0 else 0
            if elapsed_ns >= threshold_ns:
                dt_years = elapsed_ns / (365.25 * 24 * 3600 * 1e9) if elapsed_ns > 0 else 0
                self._last_borrow_ts = bar.timestamp_ns
                for sym in self.config.symbols:
                    pos = self.portfolio.position_quantity(sym)
                    if abs(pos) < 1e-10:
                        continue
                    price = self.portfolio._last_prices.get(sym, bar.close)
                    abs_notional = abs(pos) * price
                    # Use series rate if available
                    annual_rate = (_lookup_rate(self._borrow_series, bar.timestamp_ns,
                                               bc.annual_borrow_rate)
                                  if self._borrow_series else bc.annual_borrow_rate)
                    cost = abs_notional * annual_rate * dt_years
                    self.portfolio.cash -= cost
                    self.metrics._total_borrow_cost += cost

    def _handle_liquidation(self, bar: Bar) -> None:
        """
        Handle liquidation when margin requirements are breached.
        
        Generates reduce-only orders to close all positions.
        """
        log.warning(
            f"LIQUIDATION TRIGGERED: equity={self.portfolio.equity():.2f}, "
            f"maintenance_margin={self.portfolio.maintenance_margin_required():.2f}"
        )
        
        # Get all symbols to liquidate
        symbols_to_liquidate = self.portfolio.get_liquidation_symbols()
        
        for symbol in symbols_to_liquidate:
            pos = self.portfolio.position_quantity(symbol)
            if abs(pos) < 1e-10:
                continue
            
            # Record liquidation
            price = self.portfolio._last_prices.get(symbol, bar.close)
            self.portfolio.record_liquidation(symbol, price, bar.timestamp_ns)
            
            # Create liquidation order
            from Interfaces.orders import Order, OrderSide, OrderType
            
            if pos > 0:
                side = OrderSide.SELL
            else:
                side = OrderSide.BUY
            
            liq_order = Order(
                symbol=symbol,
                side=side,
                order_type=OrderType.MARKET,
                quantity=abs(pos),
                timestamp_ns=bar.timestamp_ns,
                strategy_id="LIQUIDATION",
                reduce_only=True,
                metadata={"liquidation": True},
            )
            
            # Execute liquidation order (bypass risk manager)
            fills = self.execution_model.process_orders(
                [liq_order],
                bar,
                self.portfolio,
                self._get_cost_model(symbol),
                self._get_rng(symbol),
            )
            
            for fill in fills:
                pnl = self.portfolio.apply_fill(fill)
                self.risk_manager.update_daily_pnl(pnl)
                self.metrics.on_fill(fill, self.portfolio.equity())
        
        # Activate kill switch after liquidation
        self._liquidation_triggered = True
        self.risk_manager._activate_kill_switch("liquidation triggered")
    
    def _reset_state(self) -> None:
        """Reset all state for a new run."""
        self.streamer.reset()
        
        for builder in self.bar_builders.values():
            builder.reset()
        
        self.bar_store = BarStore(maxlen=self.config.bar_store_maxlen)
        self.portfolio.reset()
        self.execution_model.reset()
        self.risk_manager.reset()
        # Pass initial_capital so metrics peak_equity matches portfolio
        self.metrics.reset(self.config.initial_capital)
        self.rng = Random(self.config.random_seed)
        
        # Reset S/R trackers
        for tracker in self.sr_trackers.values():
            tracker.reset()
        
        # Reset exit manager
        if self.exit_manager is not None:
            self.exit_manager.reset()
        
        self._current_bar_ts.clear()
        self._tick_count = 0
        self._bar_count = 0
        self._order_count = 0
        self._risk_rejected_orders = 0
        self._tick_exit_count = 0
        self._liquidation_triggered = False
        self._active_strategy = None
        self._sizing_config = None  # Reset sizing config
        self._last_funding_ts = 0
        self._last_borrow_ts = 0
    
    def _setup_exit_manager(self, strategy) -> None:
        """
        Setup exit manager from strategy configuration.
        
        Looks for exit parameters in:
        1. Strategy.exit_manager (if strategy has its own)
        2. Strategy.get_exit_params() method
        3. Falls back to no exit manager
        """
        from Backtest.exit_manager import ExitManager, ExitConfig
        
        # Try to get strategy's exit manager
        strat_exit_mgr = getattr(strategy, 'exit_manager', None)
        if strat_exit_mgr is not None:
            self.exit_manager = strat_exit_mgr
            log.info("Using strategy's exit manager for tick-level TP/SL")
            return
        
        # Try to get exit params from strategy
        if hasattr(strategy, 'get_exit_params') and callable(strategy.get_exit_params):
            params = strategy.get_exit_params()
            if params:
                config = ExitConfig(
                    take_profit_pct=params.get('tp_pct'),
                    stop_loss_pct=params.get('sl_pct'),
                    take_profit_usd=params.get('tp_usd'),
                    stop_loss_usd=params.get('sl_usd'),
                    trailing_stop_pct=params.get('trailing_stop_pct'),
                    leverage=self.config.leverage,
                )
                if config.has_any_rule():
                    self.exit_manager = ExitManager(config)
                    log.info(f"Created exit manager from strategy params: {params}")
                    return
        
        # Try engine config TP/SL (strategy-independent)
        if any([self.config.tp_pct, self.config.sl_pct,
                self.config.trailing_stop_pct, self.config.max_holding_bars]):
            config = ExitConfig(
                take_profit_pct=self.config.tp_pct,
                stop_loss_pct=self.config.sl_pct,
                trailing_stop_pct=self.config.trailing_stop_pct,
                max_holding_bars=self.config.max_holding_bars,
                leverage=self.config.leverage,
            )
            self.exit_manager = ExitManager(config)
            log.info("Created exit manager from engine config: tp=%s sl=%s trail=%s",
                     self.config.tp_pct, self.config.sl_pct, self.config.trailing_stop_pct)
            return

        # No exit manager
        self.exit_manager = None
        log.debug("No exit manager configured - tick-level TP/SL disabled")
    
    def _check_tick_exit(self, tick: Tick) -> None:
        """
        Check if TP/SL should trigger at current tick.
        
        This is the core of intrabar exit simulation:
        1. Get current position for the tick's symbol
        2. Check if TP/SL is triggered at tick price
        3. If triggered, execute exit order immediately
        
        Args:
            tick: Current tick
        """
        if self.exit_manager is None:
            return
        
        symbol = tick.symbol
        pos = self.portfolio.position_quantity(symbol)
        
        # No position, nothing to check
        if abs(pos) < 1e-10:
            return
        
        # Get entry price
        avg_entry = self.portfolio.position_avg_entry(symbol)
        if avg_entry <= 0:
            return
        
        # Check for exit
        result = self.exit_manager.check_exit_tick(
            symbol=symbol,
            tick_price=tick.price,
            tick_timestamp_ns=tick.timestamp_ns,
            position=pos,
            avg_entry_price=avg_entry,
            strategy_id="TICK_EXIT"
        )
        
        if result is None:
            return
        
        exit_order, exit_reason = result
        
        # Create synthetic bar at tick price for execution
        synthetic_bar = Bar(
            symbol=symbol,
            timeframe="tick",  # Mark as tick-generated
            timestamp_ns=tick.timestamp_ns,
            open=tick.price,
            high=tick.price,
            low=tick.price,
            close=tick.price,
            volume=tick.volume,
        )
        
        # Execute exit order (bypass risk manager for reduce-only exits)
        fills = self.execution_model.process_orders(
            [exit_order],
            synthetic_bar,
            self.portfolio,
            self._get_cost_model(symbol),
            self._get_rng(symbol),
        )
        
        for fill in fills:
            pnl = self.portfolio.apply_fill(fill)
            self.risk_manager.update_daily_pnl(pnl)
            self.metrics.on_fill(fill, self.portfolio.equity())
            self._tick_exit_count += 1
            
            # Improved logging with full precision
            # log.info(
            #     f"TICK EXIT: {symbol} {exit_reason.value} | "
            #     f"qty={fill.fill_quantity:.6f} | "
            #     f"entry={avg_entry:.6f} | "
            #     f"exit={tick.price:.6f} | "
            #     f"pnl=${pnl:.4f}"
            # )
        
        # Clear position state in exit manager
        self.exit_manager.close_position(symbol)
    
    @property
    def tick_count(self) -> int:
        """Number of ticks processed."""
        return self._tick_count
    
    @property
    def bar_count(self) -> int:
        """Number of bars processed."""
        return self._bar_count
