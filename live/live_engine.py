"""
live/live_engine.py
===================
Live trading engine using the unified strategy interface.

Multi-coin ready with:
- LiveSupervisor: per-symbol PositionManager with isolated config
- Startup reconciliation of orphaned exchange positions
- Global risk manager with JSON persistence
- Per-symbol S/R levels, leverage, margin type overrides
- Rate-limited broker for Binance weight safety
"""
from __future__ import annotations

import asyncio
import time
from typing import Optional, Dict, Any

import numpy as np

from utils.bar_store import BarStore
from utils.logger import setup_logger
from Interfaces.IBroker import IBroker
from Interfaces.market_data import Bar
from Interfaces.orders import OrderSide
from Interfaces.IStrategy import StrategyDecision
from Interfaces.strategy_adapter import adapt_strategy_output, StrategyContext

from live.live_config import LiveConfig
from live.position_manager import PositionManager, LiveSupervisor
from live.global_risk import LiveGlobalRisk
from live.streamer import Streamer

log = setup_logger("LiveEngine")


def _resolve_strategy_cls(class_name: str):
    """Dynamically import a strategy class by name."""
    modules = [
        "Strategy.RSIThreshold",
        "Strategy.DonchianATRVolTarget",
        "Strategy.binary_base_strategy",
    ]
    import importlib
    for mod_path in modules:
        try:
            mod = importlib.import_module(mod_path)
            cls = getattr(mod, class_name, None)
            if cls is not None:
                return cls
            # Also check for a generic "Strategy" export
            cls = getattr(mod, "Strategy", None)
            if cls is not None and cls.__name__ == class_name:
                return cls
        except ImportError:
            continue
    raise ImportError(f"Strategy class '{class_name}' not found")


# ======================================================================
# Live Risk Checker
# ======================================================================
class LiveRiskChecker:
    """
    Lightweight pre-trade risk checker for live trading.

    Mirrors the checks in Backtest/risk.py (RiskLimits / BasicRiskManager)
    but operates against live state (exchange balance, position manager).
    """

    def __init__(self, cfg: LiveConfig):
        self.risk = cfg.risk
        self._daily_pnl: float = 0.0
        self._start_equity: float = 0.0
        self._kill_switch: bool = False
        self._kill_reason: str = ""
        self._day_marker: str = ""

    def set_start_equity(self, equity: float):
        today = time.strftime("%Y-%m-%d")
        if today != self._day_marker:
            self._day_marker = today
            self._daily_pnl = 0.0
            self._start_equity = equity
            self._kill_switch = False

    def record_pnl(self, pnl: float):
        self._daily_pnl += pnl

    @property
    def is_kill_switch_active(self) -> bool:
        return self._kill_switch

    def pre_trade_check(
        self,
        symbol: str,
        qty: float,
        price: float,
        leverage: float,
        open_position_count: int,
    ) -> tuple[bool, str]:
        """Return (ok, reason).  ok=True means order is allowed."""
        if self._kill_switch:
            return False, f"kill switch: {self._kill_reason}"

        if open_position_count >= self.risk.max_concurrent_positions:
            return False, "max concurrent positions reached"

        if abs(qty) > self.risk.max_position_size:
            return False, f"qty {abs(qty):.4f} > limit {self.risk.max_position_size}"

        notional = abs(qty) * price
        if notional > self.risk.max_position_notional:
            return False, f"notional {notional:.2f} > limit {self.risk.max_position_notional}"

        if leverage > self.risk.max_leverage:
            return False, f"leverage {leverage} > limit {self.risk.max_leverage}"

        if self._daily_pnl < -self.risk.max_daily_loss:
            self._kill_switch = True
            self._kill_reason = "daily loss limit exceeded"
            return False, self._kill_reason

        if self._start_equity > 0:
            loss_pct = -self._daily_pnl / self._start_equity
            if loss_pct > self.risk.max_daily_loss_pct:
                self._kill_switch = True
                self._kill_reason = "daily loss % limit exceeded"
                return False, self._kill_reason

        return True, ""


# ======================================================================
# Live Engine
# ======================================================================
class LiveEngine:
    """
    Live trading engine with full multi-coin support.

    Uses the same strategy classes as BacktestEngine.
    All parameters driven by LiveConfig (loaded from YAML).
    """

    def __init__(self, cfg: LiveConfig, broker: IBroker, strategy_cls=None):
        self.cfg = cfg
        self.broker = broker
        self.bar_store = BarStore()

        # Resolve strategy class
        if strategy_cls is None:
            strategy_cls = _resolve_strategy_cls(cfg.strategy_class)

        # Instantiate strategy
        params = dict(cfg.strategy_params)
        self.strategy = strategy_cls(bar_store=self.bar_store, **params)

        # Per-symbol supervisor (replaces old single PositionManager)
        self.supervisor = LiveSupervisor(broker=self.broker)

        # Backward-compat alias used by live_runner.py force_close_all
        self.pos_mgr = self.supervisor

        # Pre-trade risk checker (per-trade limits)
        self.risk_checker = LiveRiskChecker(cfg)

        # Global risk (account-level, persistent)
        self.global_risk = LiveGlobalRisk(cfg.global_risk)

        # S/R levels cache: symbol -> SupportResistanceResult
        self._levels_cache: Dict[str, Any] = {}
        self._bar_count: Dict[str, int] = {}

        # Streamer
        self.timeframes = [cfg.timeframe]
        self.symbols = list(cfg.symbols)
        self.streamer: Optional[Streamer] = None

    # ------------------------------------------------------------------
    # S/R level helpers
    # ------------------------------------------------------------------
    def _update_levels(self, symbol: str, force: bool = False):
        """Recompute S/R levels from bar history when enabled."""
        lcfg = self.cfg.levels
        if not lcfg.enabled:
            return

        count = self._bar_count.get(symbol, 0)
        if not force and count % lcfg.update_interval_bars != 0:
            return

        bars = self.bar_store.get_recent(symbol, self.cfg.timeframe, limit=500)
        if not bars or len(bars) < 2 * lcfg.swing_window + 1:
            return

        highs = np.array([b.get("high", b.get("h", 0.0)) for b in bars], dtype=float)
        lows = np.array([b.get("low", b.get("l", 0.0)) for b in bars], dtype=float)
        closes = np.array([b.get("close", b.get("c", 0.0)) for b in bars], dtype=float)

        from utils.levels import detect_swing_levels, compute_pivot_levels

        sr = detect_swing_levels(
            highs, lows, closes,
            window=lcfg.swing_window,
            num_levels=lcfg.num_levels,
            min_touches=lcfg.min_touches,
            tolerance_bps=lcfg.tolerance_bps,
        )

        # Pivot levels from the last completed bar
        if len(bars) >= 2:
            prev = bars[-2]
            pivot_result = compute_pivot_levels(
                high=float(prev.get("high", prev.get("h", 0))),
                low=float(prev.get("low", prev.get("l", 0))),
                close=float(prev.get("close", prev.get("c", 0))),
                method=lcfg.method,
            )
            sr.metadata["pivot_levels"] = pivot_result.levels

        self._levels_cache[symbol] = sr
        log.info(
            "%s S/R updated: %d levels (method=%s)",
            symbol, len(sr.levels), sr.method,
        )

    # ------------------------------------------------------------------
    # Main loop
    # ------------------------------------------------------------------
    async def run(self):
        """Main event loop for live trading."""
        # 1) Resolve symbols
        self.symbols = await Streamer.resolve_symbols(
            self.broker.client, list(set(self.cfg.symbols))
        )

        # 2) Register per-symbol managers in supervisor
        for sym in self.symbols:
            self.supervisor.register_symbol(
                symbol=sym,
                sizing_cfg=self.cfg.sizing_for(sym),
                exit_cfg=self.cfg.exit_for(sym),
                max_concurrent=1,  # 1 position per symbol
            )

        # 3) Create streamer
        self.streamer = Streamer(
            self.broker.client,
            self.symbols,
            self.timeframes,
            bar_store=self.bar_store,
        )

        # 4) Preload history
        await self.streamer.preload_history(
            self.symbols,
            self.timeframes,
            limit=self.cfg.execution.preload_bars,
            batch=self.cfg.execution.preload_batch,
        )

        # 5) Initial equity for risk tracking
        try:
            balance = await self.broker.balance()
            self.risk_checker.set_start_equity(balance)
            self.global_risk.set_start_equity(balance)
            log.info("Initial balance: %.2f USDT", balance)
        except Exception as e:
            log.warning("Could not fetch initial balance: %s", e)

        # 6) Compute initial S/R levels
        for sym in self.symbols:
            self._bar_count[sym] = 0
            self._update_levels(sym, force=True)

        # 7) Setup exchange margin/leverage per symbol (using per-symbol config)
        for sym in self.symbols:
            try:
                mt = self.cfg.margin_type_for(sym)
                if mt == "ISOLATED":
                    await self.broker.ensure_isolated_margin(sym)
                elif mt == "CROSSED":
                    log.warning("%s using CROSSED margin — risk is shared across symbols", sym)
                await self.broker.set_leverage(sym, self.cfg.leverage_for(sym))
            except Exception as e:
                log.warning("%s margin/leverage setup: %s", sym, e)

        # 8) Startup reconciliation — adopt orphaned exchange positions
        await self.supervisor.reconcile_all(strategy_name=self.cfg.name)

        # 9) Start live stream
        await self.streamer.start()
        log.info(
            "Live Engine Started: %d symbols | tf=%s | sizing=%s",
            len(self.symbols),
            self.cfg.timeframe,
            self.cfg.sizing.mode,
        )

        try:
            while True:
                bar_data = await self.streamer.get()
                sym = bar_data["s"]
                k = bar_data.get("k", {})
                tf = k.get("i", self.cfg.timeframe)

                if tf != self.cfg.timeframe:
                    continue

                self._bar_count[sym] = self._bar_count.get(sym, 0) + 1
                self._update_levels(sym)

                await self._process_bar(sym, tf, k)
        finally:
            await self.streamer.stop()

    # ------------------------------------------------------------------
    # Bar processing
    # ------------------------------------------------------------------
    async def _process_bar(self, symbol: str, timeframe: str, k: dict):
        """Process a completed bar through the unified strategy adapter."""

        bar = Bar(
            symbol=symbol,
            timeframe=timeframe,
            timestamp_ns=int(k.get("t", k.get("start", 0)) * 1_000_000),
            open=float(k.get("o", 0)),
            high=float(k.get("h", 0)),
            low=float(k.get("l", 0)),
            close=float(k.get("c", 0)),
            volume=float(k.get("v", 0)),
        )

        # Current position from supervisor
        pos_qty = self.supervisor.position_qty(symbol)

        # Build context (identical structure to backtest)
        ctx = StrategyContext(
            symbol=symbol,
            timeframe=timeframe,
            bar_store=self.bar_store,
            portfolio=None,
            position=pos_qty,
            equity=0.0,
            cash=0.0,
            timestamp_ns=bar.timestamp_ns,
            metadata={
                "levels": self._levels_cache.get(symbol),
            },
        )

        # Unified strategy adapter — same function used in backtest engine
        adapted = adapt_strategy_output(
            strategy=self.strategy,
            bar=bar,
            ctx=ctx,
            position_size=1.0,
            strategy_id=self.cfg.name,
        )

        # Extract signal / direction
        signal = None
        if adapted.decision and adapted.decision.has_signal:
            signal = adapted.decision.signal
        elif adapted.decision and adapted.decision.has_orders:
            for order in adapted.decision.orders:
                if not order.reduce_only:
                    signal = "+1" if order.side == OrderSide.BUY else "-1"
                    break

        if signal:
            direction = 1 if signal in ("+1", "1") else -1
            mark_price = bar.close

            # Per-symbol sizing config
            sym_sizing = self.cfg.sizing_for(symbol)
            qty = sym_sizing.compute_qty(mark_price)
            leverage = self.cfg.leverage_for(symbol)

            # Pre-trade risk check
            ok, reason = self.risk_checker.pre_trade_check(
                symbol=symbol,
                qty=qty,
                price=mark_price,
                leverage=leverage,
                open_position_count=len(self.supervisor.open_positions),
            )
            if not ok:
                log.warning("Order REJECTED for %s: %s", symbol, reason)
                return

            # Global account-level risk check
            try:
                balance = await self.broker.balance()
                total_exposure = sum(
                    p.entry * p.qty
                    for p in self.supervisor.open_positions.values()
                )
                ok, reason = self.global_risk.check_account_risk(
                    current_equity=balance,
                    total_exposure_usd=total_exposure + qty * mark_price,
                    open_position_count=len(self.supervisor.open_positions),
                )
                if not ok:
                    log.warning("GLOBAL RISK REJECTED for %s: %s", symbol, reason)
                    return
            except Exception as e:
                log.warning("Balance fetch for global risk failed: %s", e)

            # Open via supervisor (per-symbol manager)
            await self.supervisor.open_position(
                symbol=symbol,
                side=direction,
                strategy_name=self.cfg.name,
                leverage=leverage,
                timeframe=timeframe,
                levels=self._levels_cache.get(symbol),
            )
        else:
            # No signal — monitor existing positions & record closed P&L
            prev_history_len = len(self.supervisor.history)
            await self.supervisor.update_all()
            # Record realized PnL from newly closed positions
            for pos in self.supervisor.history[prev_history_len:]:
                if pos.exit is not None:
                    pnl = pos.unrealized_pnl(pos.exit)
                    self.risk_checker.record_pnl(pnl)
                    self.global_risk.record_pnl(pnl)
