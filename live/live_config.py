"""
live/live_config.py
===================
Configuration dataclasses and YAML loader for live trading.

Mirrors the structure of Backtest/realism_config.py but focused on
live-specific settings: sizing, exit rules, risk limits, S/R levels.

Multi-coin support:
    symbol_routes allows per-symbol overrides for leverage, sizing, exit.
    global_risk adds account-level risk with JSON state persistence.

Usage:
    cfg = LiveConfig.from_yaml("example_live_config.yaml")
    engine = LiveEngine(cfg, broker, strategy_cls)
"""
from __future__ import annotations

import copy
from dataclasses import dataclass, field
from typing import Dict, Any, Optional, List
import logging

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Sizing
# ---------------------------------------------------------------------------
@dataclass
class SizingConfig:
    """Position sizing configuration for live trading."""
    mode: str = "margin_usd"        # "fixed_qty" | "notional_usd" | "margin_usd"
    fixed_qty: Optional[float] = None
    notional_usd: Optional[float] = None
    margin_usd: float = 10.0
    leverage: int = 5
    leverage_mode: str = "margin"    # "spot" | "margin"

    def compute_qty(self, price: float) -> float:
        """Compute order quantity from current price."""
        if price <= 0:
            return 0.0
        if self.mode == "fixed_qty":
            return self.fixed_qty or 1.0
        elif self.mode == "notional_usd":
            return (self.notional_usd or 100.0) / price
        elif self.mode == "margin_usd":
            if self.leverage_mode == "spot":
                notional = self.margin_usd
            else:
                notional = self.margin_usd * self.leverage
            return notional / price
        return 1.0


# ---------------------------------------------------------------------------
# Exit rules
# ---------------------------------------------------------------------------
@dataclass
class ExitConfig:
    """TP / SL / Trailing / Time-based exit rules."""
    take_profit_pct: Optional[float] = None
    take_profit_usd: Optional[float] = None
    stop_loss_pct: Optional[float] = None
    stop_loss_usd: Optional[float] = None
    trailing_stop_pct: Optional[float] = None
    max_holding_bars: Optional[int] = None
    use_exchange_orders: bool = True

    def has_any_rule(self) -> bool:
        return any([
            self.take_profit_pct is not None,
            self.take_profit_usd is not None,
            self.stop_loss_pct is not None,
            self.stop_loss_usd is not None,
            self.trailing_stop_pct is not None,
            self.max_holding_bars is not None,
        ])


# ---------------------------------------------------------------------------
# Risk limits
# ---------------------------------------------------------------------------
@dataclass
class RiskConfig:
    """Pre-trade risk check limits."""
    max_position_size: float = 100_000.0
    max_position_notional: float = 5000.0
    max_daily_loss: float = 50.0
    max_daily_loss_pct: float = 0.10
    max_drawdown: float = 0.20
    max_leverage: float = 20.0
    max_concurrent_positions: int = 3


# ---------------------------------------------------------------------------
# Support / Resistance levels
# ---------------------------------------------------------------------------
@dataclass
class LevelsConfig:
    """Support and resistance level settings."""
    enabled: bool = False
    method: str = "classic"         # "classic" | "fibonacci" | "woodie" | "camarilla"
    swing_window: int = 5
    num_levels: int = 3
    min_touches: int = 1
    tolerance_bps: float = 10.0
    update_interval_bars: int = 10


# ---------------------------------------------------------------------------
# Execution / exchange
# ---------------------------------------------------------------------------
@dataclass
class ExecutionConfig:
    """Exchange-specific execution settings."""
    margin_type: str = "ISOLATED"   # "ISOLATED" | "CROSSED"
    preload_bars: int = 100
    preload_batch: int = 10


# ---------------------------------------------------------------------------
# Per-symbol route (multi-coin overrides)
# ---------------------------------------------------------------------------
@dataclass
class SymbolRoute:
    """Per-symbol overrides.  Unset fields fall back to global defaults."""
    symbol: str = ""
    leverage: Optional[int] = None
    margin_usd: Optional[float] = None
    margin_type: Optional[str] = None          # "ISOLATED" | "CROSSED"
    take_profit_pct: Optional[float] = None
    stop_loss_pct: Optional[float] = None
    trailing_stop_pct: Optional[float] = None
    max_holding_bars: Optional[int] = None
    enabled: bool = True


# ---------------------------------------------------------------------------
# Global account-level risk  (persisted to JSON)
# ---------------------------------------------------------------------------
@dataclass
class GlobalRiskConfig:
    """Account-wide risk limits beyond per-trade checks."""
    max_account_drawdown_pct: float = 0.20
    max_total_exposure_usd: float = 50_000.0
    max_correlated_positions: int = 5
    persist_path: str = "logs/live_risk_state.json"


# ---------------------------------------------------------------------------
# Liquidation early-warning guard
# ---------------------------------------------------------------------------
@dataclass
class LiquidationGuardConfig:
    """Pre-emptive close before the exchange auto-liquidates.

    Real-money safety net.  When the mark-price tick stream brings the
    price within ``buffer_pct`` of the position's liquidation price we
    close the position ourselves, paying market-taker slippage instead
    of the much larger ADL/liquidation penalty (Binance closes at
    bankruptcy price plus a clearing fee — typically 1–3 % vs the
    fraction of a percent we eat with a market exit).

    Attributes:
        enabled: Master switch.  Off by default; turn on when going
            real-money on margin (above 1×).
        buffer_pct: Trigger threshold expressed as a fraction of the
            initial entry-to-liquidation distance.  ``0.20`` ⇒ close
            once we are within the last 20 % of the safety margin.
            Lower = closer to liquidation, higher = earlier exit.
        maintenance_margin_ratio: Override exchange MMR.  Binance
            USDT-M tier-0 is ~0.5 % (0.005); we default to a slightly
            conservative 0.01 to absorb tier shifts.  Set per real
            account from the exchange's risk-parameters page if
            trading large notional.
        action: ``"close"`` (default) closes via market reduce-only
            and trips the kill switch.  ``"alarm"`` only logs +
            Telegram (debugging mode — DO NOT use with real money).
    """
    enabled: bool = False
    buffer_pct: float = 0.20
    maintenance_margin_ratio: float = 0.01
    action: str = "close"


# ---------------------------------------------------------------------------
# Position reconciliation (drift detection)
# ---------------------------------------------------------------------------
@dataclass
class ReconciliationConfig:
    """Periodic local-vs-exchange position drift detection.

    Real-money safety net.  The exchange is the source of truth: if our
    local position quantity diverges from what the exchange reports
    (manual intervention, missed fill, broker bug, partial liquidation),
    we want to know within seconds — not at the next bar close.

    Attributes:
        enabled: Master switch.  Off by default to keep dry-run cheap.
        interval_seconds: How often to poll the exchange.  30s is a
            healthy balance — Binance position endpoint is rate-limit
            cheap (weight 5).  Lower than 10s wastes calls.
        qty_tolerance: Max absolute qty diff treated as "in sync".
            Set above zero to absorb rounding noise (e.g. 1e-6 for
            most pairs; bigger for satoshi-precision symbols).
        action: What to do when drift is detected:
            * ``"alarm"``      — log + Telegram only.
            * ``"halt"``       — trip the global kill-switch (default,
              recommended for real money).  Existing positions keep
              their server-side TP/SL; new entries are blocked.
            * ``"force_flat"`` — close every drifted position via
              market order.  Aggressive; use only with full trust in
              the kill-switch persistence story.
    """
    enabled: bool = False
    interval_seconds: float = 30.0
    qty_tolerance: float = 1e-6
    action: str = "halt"


# ---------------------------------------------------------------------------
# News sentiment settings
# ---------------------------------------------------------------------------
@dataclass
class NewsConfig:
    """News sentiment analysis configuration."""
    enabled: bool = False
    sentiment_provider: str = "gemini"      # "gemini" | "openai"
    api_key: Optional[str] = None           # Override env var (GOOGLE_API_KEY / OPENAI_API_KEY)
    refresh_interval: int = 300             # Seconds between sentiment refreshes
    news_limit: int = 5                     # Max articles per symbol per fetch
    buy_threshold: float = 0.6             # Sentiment > this + BUY signal → LONG
    sell_threshold: float = 0.4            # Sentiment < this + SELL signal → SHORT


# ---------------------------------------------------------------------------
# Rate-limit settings
# ---------------------------------------------------------------------------
@dataclass
class RateLimitConfig:
    """API rate-limit guard."""
    requests_per_minute: int = 1000    # conservative vs Binance 1200
    exchange_info_ttl_sec: int = 300   # cache exchange_info for 5 min


# ---------------------------------------------------------------------------
# Top-level live config
# ---------------------------------------------------------------------------
@dataclass
class LiveConfig:
    """
    Top-level container for all live trading settings.

    All sections default to safe values.
    """
    # Strategy
    strategy_class: str = "RSIThreshold"
    strategy_params: Dict[str, Any] = field(default_factory=dict)
    # Optional path to a composite spec (YAML/JSON).  When set, the live
    # engine runs a CompositeStrategy and ignores strategy_class/params.
    composite_spec: Optional[str] = None

    # Symbols & timeframe
    symbols: List[str] = field(default_factory=lambda: ["DOGEUSDT"])
    timeframe: str = "1m"
    name: str = "LiveStrategy"

    # Run tag — used to namespace log/state files so two parallel
    # dry-runs (e.g. sentiment ON vs OFF) do not stomp on each other.
    # CLI ``--run-id`` overrides this; CLI > YAML > falls back to ``name``.
    run_id: Optional[str] = None

    # Sub-configs
    sizing: SizingConfig = field(default_factory=SizingConfig)
    exit: ExitConfig = field(default_factory=ExitConfig)
    risk: RiskConfig = field(default_factory=RiskConfig)
    levels: LevelsConfig = field(default_factory=LevelsConfig)
    execution: ExecutionConfig = field(default_factory=ExecutionConfig)

    # News sentiment
    news: NewsConfig = field(default_factory=NewsConfig)

    # Multi-coin & global risk
    symbol_routes: Dict[str, SymbolRoute] = field(default_factory=dict)
    global_risk: GlobalRiskConfig = field(default_factory=GlobalRiskConfig)
    rate_limit: RateLimitConfig = field(default_factory=RateLimitConfig)
    reconciliation: ReconciliationConfig = field(default_factory=ReconciliationConfig)
    liquidation_guard: LiquidationGuardConfig = field(default_factory=LiquidationGuardConfig)

    # API
    testnet: bool = False

    # ----- per-symbol helpers -----
    def sizing_for(self, symbol: str) -> SizingConfig:
        """Return SizingConfig with per-symbol overrides applied."""
        route = self.symbol_routes.get(symbol)
        if route is None:
            return self.sizing
        s = copy.copy(self.sizing)
        if route.leverage is not None:
            s.leverage = route.leverage
        if route.margin_usd is not None:
            s.margin_usd = route.margin_usd
        return s

    def exit_for(self, symbol: str) -> ExitConfig:
        """Return ExitConfig with per-symbol overrides applied."""
        route = self.symbol_routes.get(symbol)
        if route is None:
            return self.exit
        e = copy.copy(self.exit)
        if route.take_profit_pct is not None:
            e.take_profit_pct = route.take_profit_pct
        if route.stop_loss_pct is not None:
            e.stop_loss_pct = route.stop_loss_pct
        if route.trailing_stop_pct is not None:
            e.trailing_stop_pct = route.trailing_stop_pct
        if route.max_holding_bars is not None:
            e.max_holding_bars = route.max_holding_bars
        return e

    def leverage_for(self, symbol: str) -> int:
        route = self.symbol_routes.get(symbol)
        if route and route.leverage is not None:
            return route.leverage
        return self.sizing.leverage

    def margin_type_for(self, symbol: str) -> str:
        route = self.symbol_routes.get(symbol)
        if route and route.margin_type is not None:
            return route.margin_type
        return self.execution.margin_type

    # ----- run tagging helpers -----
    def effective_run_id(self) -> str:
        """``run_id`` if set, else ``name``.  Used as a path suffix."""
        return self.run_id or self.name or "live"

    def trade_log_path(self) -> str:
        """CSV path for live trade metrics, namespaced by run_id."""
        return f"logs/live_trades_{self.effective_run_id()}.csv"

    def positions_state_path(self) -> str:
        """JSON path for the position store, namespaced by run_id."""
        return f"logs/live_positions_{self.effective_run_id()}.json"

    def risk_state_path(self) -> str:
        """JSON path for the global-risk persist file, namespaced by run_id.
        Falls back to the explicit ``global_risk.persist_path`` if it has
        been set to a non-default value."""
        default = "logs/live_risk_state.json"
        if self.global_risk.persist_path != default:
            return self.global_risk.persist_path
        return f"logs/live_risk_state_{self.effective_run_id()}.json"

    # ----- serialisation helpers -----
    @staticmethod
    def from_dict(d: Dict[str, Any]) -> "LiveConfig":
        """Build from a plain dict (parsed YAML / JSON)."""
        strat = d.get("strategy", {})
        sizing_d = d.get("sizing", {})
        exit_d = d.get("exit", {})
        risk_d = d.get("risk", {})
        levels_d = d.get("levels", {})
        exec_d = d.get("execution", {})
        news_d = d.get("news", {})
        api_d = d.get("api", {})
        gr_d = d.get("global_risk", {})
        rl_d = d.get("rate_limit", {})
        recon_d = d.get("reconciliation", {})
        liq_d = d.get("liquidation_guard", {})

        def _pick(cls, src):
            return cls(**{k: v for k, v in src.items()
                          if k in cls.__dataclass_fields__})

        # Parse symbol_routes
        routes: Dict[str, SymbolRoute] = {}
        for rd in d.get("symbol_routes", []):
            sym = rd.get("symbol", "")
            if sym:
                routes[sym] = _pick(SymbolRoute, rd)
                routes[sym].symbol = sym

        return LiveConfig(
            strategy_class=strat.get("class", "RSIThreshold"),
            strategy_params=strat.get("params", {}),
            composite_spec=strat.get("composite_spec"),
            symbols=d.get("symbols", ["DOGEUSDT"]),
            timeframe=d.get("timeframe", "1m"),
            name=d.get("name", "LiveStrategy"),
            run_id=d.get("run_id"),
            sizing=_pick(SizingConfig, sizing_d),
            exit=_pick(ExitConfig, exit_d),
            risk=_pick(RiskConfig, risk_d),
            levels=_pick(LevelsConfig, levels_d),
            execution=_pick(ExecutionConfig, exec_d),
            news=_pick(NewsConfig, news_d),
            symbol_routes=routes,
            global_risk=_pick(GlobalRiskConfig, gr_d),
            rate_limit=_pick(RateLimitConfig, rl_d),
            reconciliation=_pick(ReconciliationConfig, recon_d),
            liquidation_guard=_pick(LiquidationGuardConfig, liq_d),
            testnet=api_d.get("testnet", False),
        )

    @staticmethod
    def from_yaml(path: str) -> "LiveConfig":
        """Load from a YAML file."""
        try:
            import yaml
        except ImportError:
            raise ImportError("PyYAML required: pip install pyyaml")
        with open(path, "r", encoding="utf-8") as f:
            d = yaml.safe_load(f) or {}
        return LiveConfig.from_dict(d)

    @staticmethod
    def from_json(path: str) -> "LiveConfig":
        """Load from a JSON file."""
        import json
        with open(path, "r", encoding="utf-8") as f:
            d = json.load(f)
        return LiveConfig.from_dict(d)
