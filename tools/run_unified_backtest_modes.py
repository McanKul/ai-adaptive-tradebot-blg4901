#!/usr/bin/env python3
"""
tools/run_unified_backtest_modes.py
====================================
Experiment harness: runs N realism modes × M strategy configs and outputs
a Markdown comparison table plus an optional CSV.

Modes
-----
MODE_0 : Legacy (all realism OFF, low fixed costs)
MODE_1 : Latency + price-aware fills
MODE_2 : Partial fills enabled
MODE_3 : Full realism (latency + partial + higher costs + funding + borrow)

Usage
-----
    python tools/run_unified_backtest_modes.py
    python tools/run_unified_backtest_modes.py --data-dir ./data/ticks --symbol DOGEUSDT
    python tools/run_unified_backtest_modes.py --csv results.csv
    python tools/run_unified_backtest_modes.py --modes-csv tools/example_modes.csv --modes CSV_FULL
    python tools/run_unified_backtest_modes.py --dump-effective-config --modes MODE_0 FULL

--modes-csv schema
------------------
Each row in the CSV defines one mode.  The CSV is read with
``csv.DictReader``, so the first row must be a header.

Required columns:
    mode_name          Unique identifier used with ``--modes`` (e.g. MY_MODE).

Optional meta columns (string, default = mode_name / empty):
    label              Human-readable label shown in tables.
    description        Freeform description for documentation.

Optional top-level columns:
    seed               int   -- random seed (default 42)
    close_positions_at_end  bool -- force-close at data end (default false)

Engine-override columns (float/int/bool, defaults shown):
    taker_fee_bps      4.0     maker_fee_bps       2.0
    slippage_bps       1.0     spread_bps          2.0
    latency_ns         0       latency_jitter_ns   0
    enable_partial_fills  false
    liquidity_scale    1.0     min_fill_ratio      (engine default)

TransactionCostConfig columns (routed to ``realism.transaction_costs``):
    fee_model           "fixed" | "tiered"
    slippage_model      "fixed" | "volume_sqrt" | "volatility_atr" | "composite"
    impact_factor       float (0.1)
    max_slippage_bps    float (50.0)
    spread_model        "fixed" | "ohlc_estimator"
    latency_distribution  "uniform"|"gamma"|"lognormal"|"weibull"|"mixture"
    latency_gamma_shape   float (2.0)
    latency_gamma_scale_ns  float (10_000_000)
    latency_lognormal_mu    float (16.0)
    latency_lognormal_sigma float (0.5)
    price_latency_mode  "timestamp_only" | "price_aware"
    intrabar_price_model  "none" | "gaussian_clamped"   (default "none")
    slippage_noise      "none"|"multiplicative_lognormal"|"additive_halfnormal"|"mixture"
    slippage_noise_sigma  float (0.2)
    (... any other TransactionCostConfig field -- see realism_config.py)

FundingConfig columns (routed to ``realism.funding``):
    enabled             bool (false)
    funding_interval_hours  int (8)
    funding_rate_mode   "constant" | "series"
    funding_rate        float (0.0001)
    series_csv_path     str -- path to rate-series CSV
    timestamp_unit      "ms" | "ns"
    interpolation       "prev"

BorrowConfig columns (routed to ``realism.borrow``):
    enabled             bool (false)
    annual_borrow_rate  float (0.0)
    charge_interval     "bar" | "daily" | "hourly"
    series_csv_path     str -- path to rate-series CSV

Bool coercion: true/1/yes/on -> True;  false/0/no/off/"" -> False.
Unrecognised columns produce a warning and are ignored.
Missing columns fall back to dataclass defaults.

Example (tools/example_modes.csv)::

    mode_name,label,taker_fee_bps,latency_distribution,price_latency_mode
    MY_FAST,Fast baseline,4.0,uniform,timestamp_only
    MY_REAL,Realistic,4.0,gamma,price_aware
"""
from __future__ import annotations

import argparse
import csv
import sys
import os
import time
import logging
import math
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional

# Ensure project root is on path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from Backtest.engine import BacktestEngine, EngineConfig
from Backtest.realism_config import (
    RealismConfig,
    TransactionCostConfig,
    FundingConfig,
    BorrowConfig,
)
from Backtest.scoring.scorer import Scorer, ScorerWeights
from Interfaces.strategy_adapter import SizingConfig, SizingMode
from Interfaces.metrics_interface import BacktestResult

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Mode definitions
# ---------------------------------------------------------------------------

MODES: Dict[str, Dict[str, Any]] = {
    # ── A) Baseline ──────────────────────────────────────────────────────
    "MODE_0": {
        "label": "Legacy (baseline)",
        "description": "All realism OFF, fixed low costs.",
        "engine_overrides": {
            "taker_fee_bps": 4.0,
            "maker_fee_bps": 2.0,
            "slippage_bps": 1.0,
            "spread_bps": 2.0,
            "latency_ns": 0,
            "latency_jitter_ns": 0,
            "enable_partial_fills": False,
        },
        "realism": RealismConfig(),  # all defaults = legacy
    },
    # ── B) Latency distribution comparison ───────────────────────────────
    "LAT_UNIFORM": {
        "label": "Latency: uniform",
        "description": "50ms base + 20ms uniform jitter, price-aware.",
        "engine_overrides": {
            "taker_fee_bps": 4.0, "maker_fee_bps": 2.0,
            "slippage_bps": 1.0, "spread_bps": 2.0,
            "latency_ns": 50_000_000, "latency_jitter_ns": 20_000_000,
            "enable_partial_fills": False,
        },
        "realism": RealismConfig(
            transaction_costs=TransactionCostConfig(
                latency_ns=50_000_000,
                latency_jitter_ns=20_000_000,
                latency_distribution="uniform",
                price_latency_mode="price_aware",
            ),
        ),
    },
    "LAT_GAMMA": {
        "label": "Latency: gamma(2,10ms)",
        "description": "Gamma(k=2, θ=10ms) — peaked with right tail.",
        "engine_overrides": {
            "taker_fee_bps": 4.0, "maker_fee_bps": 2.0,
            "slippage_bps": 1.0, "spread_bps": 2.0,
            "latency_ns": 0, "latency_jitter_ns": 0,
            "enable_partial_fills": False,
        },
        "realism": RealismConfig(
            transaction_costs=TransactionCostConfig(
                latency_ns=0,
                latency_distribution="gamma",
                latency_gamma_shape=2.0,
                latency_gamma_scale_ns=10_000_000,
                price_latency_mode="price_aware",
            ),
        ),
    },
    "LAT_LOGNORM": {
        "label": "Latency: lognormal",
        "description": "LogN(μ=16, σ=0.5) — median ~9ms, heavy right tail.",
        "engine_overrides": {
            "taker_fee_bps": 4.0, "maker_fee_bps": 2.0,
            "slippage_bps": 1.0, "spread_bps": 2.0,
            "latency_ns": 0, "latency_jitter_ns": 0,
            "enable_partial_fills": False,
        },
        "realism": RealismConfig(
            transaction_costs=TransactionCostConfig(
                latency_ns=0,
                latency_distribution="lognormal",
                latency_lognormal_mu=16.0,
                latency_lognormal_sigma=0.5,
                price_latency_mode="price_aware",
            ),
        ),
    },
    "LAT_MIXTURE": {
        "label": "Latency: mixture(5% spike)",
        "description": "95% gamma(2,10ms) + 5% lognormal spike.",
        "engine_overrides": {
            "taker_fee_bps": 4.0, "maker_fee_bps": 2.0,
            "slippage_bps": 1.0, "spread_bps": 2.0,
            "latency_ns": 0, "latency_jitter_ns": 0,
            "enable_partial_fills": False,
        },
        "realism": RealismConfig(
            transaction_costs=TransactionCostConfig(
                latency_ns=0,
                latency_distribution="mixture",
                latency_mixture_p_spike=0.05,
                latency_mixture_base_dist="gamma",
                latency_mixture_spike_dist="lognormal",
                latency_gamma_shape=2.0,
                latency_gamma_scale_ns=10_000_000,
                latency_lognormal_mu=18.5,      # spike median ~108ms
                latency_lognormal_sigma=0.7,
                price_latency_mode="price_aware",
            ),
        ),
    },
    # ── C) Slippage noise comparison ─────────────────────────────────────
    "SLIP_NONE": {
        "label": "Slippage: deterministic",
        "description": "Volume-sqrt slippage, no residual noise.",
        "engine_overrides": {
            "taker_fee_bps": 4.0, "maker_fee_bps": 2.0,
            "slippage_bps": 1.0, "spread_bps": 2.0,
            "latency_ns": 0, "latency_jitter_ns": 0,
            "enable_partial_fills": False,
        },
        "realism": RealismConfig(
            transaction_costs=TransactionCostConfig(
                slippage_model="volume_sqrt",
                impact_factor=0.1,
                max_slippage_bps=50.0,
                slippage_noise="none",
            ),
        ),
    },
    "SLIP_LOGNORM": {
        "label": "Slippage: mult-lognorm(σ=0.3)",
        "description": "Volume-sqrt + multiplicative lognormal noise (E[mult]=1).",
        "engine_overrides": {
            "taker_fee_bps": 4.0, "maker_fee_bps": 2.0,
            "slippage_bps": 1.0, "spread_bps": 2.0,
            "latency_ns": 0, "latency_jitter_ns": 0,
            "enable_partial_fills": False,
        },
        "realism": RealismConfig(
            transaction_costs=TransactionCostConfig(
                slippage_model="volume_sqrt",
                impact_factor=0.1,
                max_slippage_bps=50.0,
                slippage_noise="multiplicative_lognormal",
                slippage_noise_sigma=0.3,
            ),
        ),
    },
    "SLIP_MIXTURE": {
        "label": "Slippage: mixture(10% fat)",
        "description": "Volume-sqrt + mixture noise (σ=0.2, 10% fat σ₂=0.6).",
        "engine_overrides": {
            "taker_fee_bps": 4.0, "maker_fee_bps": 2.0,
            "slippage_bps": 1.0, "spread_bps": 2.0,
            "latency_ns": 0, "latency_jitter_ns": 0,
            "enable_partial_fills": False,
        },
        "realism": RealismConfig(
            transaction_costs=TransactionCostConfig(
                slippage_model="volume_sqrt",
                impact_factor=0.1,
                max_slippage_bps=50.0,
                slippage_noise="mixture",
                slippage_noise_sigma=0.2,
                slippage_noise_mixture_p=0.10,
                slippage_noise_mixture_sigma2=0.6,
            ),
        ),
    },
    # ── D) Full realism (all features) ───────────────────────────────────
    "FULL": {
        "label": "Full realism",
        "description": "Gamma latency + lognorm slippage noise + partial + funding + borrow.",
        "engine_overrides": {
            "taker_fee_bps": 4.0, "maker_fee_bps": 2.0,
            "slippage_bps": 1.0, "spread_bps": 3.0,
            "latency_ns": 0, "latency_jitter_ns": 0,
            "enable_partial_fills": True,
            "liquidity_scale": 5.0, "min_fill_ratio": 0.1,
        },
        "realism": RealismConfig(
            transaction_costs=TransactionCostConfig(
                slippage_model="volume_sqrt",
                impact_factor=0.1,
                max_slippage_bps=50.0,
                spread_bps=3.0,
                latency_distribution="gamma",
                latency_gamma_shape=2.0,
                latency_gamma_scale_ns=10_000_000,
                price_latency_mode="price_aware",
                marketable_limit_is_taker=True,
                slippage_noise="multiplicative_lognormal",
                slippage_noise_sigma=0.3,
                per_symbol_overrides={
                    "DOGEUSDT": {"spread_bps": 5.0, "impact_factor": 0.15},
                    "BTCUSDT": {"spread_bps": 1.0, "impact_factor": 0.05},
                },
            ),
            funding=FundingConfig(enabled=True, funding_rate=0.0001),
            borrow=BorrowConfig(enabled=True, annual_borrow_rate=0.05),
        ),
    },
    # ── E) Large latency demo (PART 1 — observable price impact) ─────────
    "LAT_BIG_DEMO": {
        "label": "Latency: BIG seconds-range",
        "description": "Gamma(2, 60s) latency → fills shift within bar OHLC range.",
        "engine_overrides": {
            "taker_fee_bps": 4.0, "maker_fee_bps": 2.0,
            "slippage_bps": 1.0, "spread_bps": 2.0,
            "latency_ns": 0, "latency_jitter_ns": 0,
            "enable_partial_fills": False,
        },
        "realism": RealismConfig(
            transaction_costs=TransactionCostConfig(
                latency_ns=0,
                latency_distribution="gamma",
                latency_gamma_shape=2.0,
                latency_gamma_scale_ns=60_000_000_000,   # 60 seconds (!)
                price_latency_mode="price_aware",
            ),
        ),
    },
    "LAT_BIG_LOGNORM": {
        "label": "Latency: BIG lognormal",
        "description": "LogN(μ=24, σ=0.8) ≈ median 27s — heavy right tail.",
        "engine_overrides": {
            "taker_fee_bps": 4.0, "maker_fee_bps": 2.0,
            "slippage_bps": 1.0, "spread_bps": 2.0,
            "latency_ns": 0, "latency_jitter_ns": 0,
            "enable_partial_fills": False,
        },
        "realism": RealismConfig(
            transaction_costs=TransactionCostConfig(
                latency_ns=0,
                latency_distribution="lognormal",
                latency_lognormal_mu=24.0,               # exp(24) ns ≈ 27 seconds
                latency_lognormal_sigma=0.8,
                price_latency_mode="price_aware",
            ),
        ),
    },
    # ── F) Funding/borrow trigger (PART 3) ───────────────────────────────
    "FUNDING_TRIGGER": {
        "label": "Funding+Borrow trigger",
        "description": "1h funding interval, 10% borrow, wide TP/SL so positions last.",
        "engine_overrides": {
            "taker_fee_bps": 4.0, "maker_fee_bps": 2.0,
            "slippage_bps": 1.0, "spread_bps": 2.0,
            "latency_ns": 0, "latency_jitter_ns": 0,
            "enable_partial_fills": False,
        },
        "realism": RealismConfig(
            transaction_costs=TransactionCostConfig(
                slippage_model="fixed",
                slippage_bps=1.0,
            ),
            funding=FundingConfig(
                enabled=True,
                funding_interval_hours=1,     # charge every hour (not 8h)
                funding_rate=0.0003,          # 0.03% per interval — aggressive
            ),
            borrow=BorrowConfig(
                enabled=True,
                annual_borrow_rate=0.10,      # 10% annualised
                charge_interval="bar",
            ),
        ),
    },
    # ── G) Realistic funding (8h interval, small rate) ───────────────────
    "REALISTIC_FUNDING": {
        "label": "Realistic 8h funding",
        "description": "Production-like 8h funding interval with 0.01% rate.",
        "engine_overrides": {
            "taker_fee_bps": 4.0, "maker_fee_bps": 2.0,
            "slippage_bps": 1.0, "spread_bps": 2.0,
            "latency_ns": 0, "latency_jitter_ns": 0,
            "enable_partial_fills": False,
        },
        "realism": RealismConfig(
            transaction_costs=TransactionCostConfig(
                slippage_model="fixed",
                slippage_bps=1.0,
            ),
            funding=FundingConfig(
                enabled=True,
                funding_interval_hours=8,
                funding_rate=0.0001,          # 0.01% per interval
            ),
            borrow=BorrowConfig(
                enabled=True,
                annual_borrow_rate=0.05,      # 5% annualised
                charge_interval="bar",
            ),
        ),
    },
}

@dataclass
class StrategySpec:
    """Describes a strategy + sizing to instantiate."""
    name: str
    factory: str          # "rsi" or "donchian"
    params: Dict[str, Any]
    sizing: SizingConfig


def _build_strategies(leverage: float, leverage_mode: str) -> List[StrategySpec]:
    """Return the default experiment matrix of strategy configs."""
    lev = leverage if leverage_mode == "margin" else 1.0
    sizing = SizingConfig(
        mode=SizingMode.MARGIN_USD,
        margin_usd=100.0,
        leverage=lev,
        leverage_mode=leverage_mode,
    )
    return [
        StrategySpec(
            name="RSI_default",
            factory="rsi",
            params=dict(
                rsi_period=14, rsi_overbought=70, rsi_oversold=30,
                take_profit_pct=0.02, stop_loss_pct=0.01,
                leverage=lev,
            ),
            sizing=sizing,
        ),
        StrategySpec(
            name="RSI_tight",
            factory="rsi",
            params=dict(
                rsi_period=7, rsi_overbought=65, rsi_oversold=35,
                take_profit_pct=0.01, stop_loss_pct=0.005,
                leverage=lev,
            ),
            sizing=sizing,
        ),
        # Wide TP/SL → longer holds → funding/borrow actually trigger
        StrategySpec(
            name="RSI_longhold",
            factory="rsi",
            params=dict(
                rsi_period=14, rsi_overbought=75, rsi_oversold=25,
                take_profit_pct=0.10, stop_loss_pct=0.05,
                leverage=lev,
            ),
            sizing=sizing,
        ),
        StrategySpec(
            name="Donchian_short_ema",
            factory="donchian",
            params=dict(
                dc_period=20, atr_period=14, atr_mult=3.0,
                risk_pct=0.005, filter_type="ema", ema_period=50,
            ),
            sizing=sizing,
        ),
        StrategySpec(
            name="Donchian_ema200",
            factory="donchian",
            params=dict(
                dc_period=20, atr_period=14, atr_mult=3.0,
                risk_pct=0.005, filter_type="ema", ema_period=200,
            ),
            sizing=sizing,
        ),
        # PART 4 — filter disabled, shorter dc_period for easier breakout
        StrategySpec(
            name="Donchian_no_filter",
            factory="donchian",
            params=dict(
                dc_period=10, atr_period=14, atr_mult=2.0,
                risk_pct=0.005, filter_type="none", ema_period=1,
            ),
            sizing=sizing,
        ),
        # PART 3 — extreme long-hold for funding/borrow demonstration
        StrategySpec(
            name="RSI_hold_forever",
            factory="rsi",
            params=dict(
                rsi_period=14, rsi_overbought=80, rsi_oversold=20,
                take_profit_pct=0.50, stop_loss_pct=0.25,
                leverage=lev,
            ),
            sizing=sizing,
        ),
    ]


def _create_strategy(spec: StrategySpec):
    """Instantiate a strategy from its spec."""
    if spec.factory == "rsi":
        from Strategy.RSIThreshold import Strategy as RSIStrategy
        return RSIStrategy(
            position_size=1.0,
            sizing_config=spec.sizing,
            **spec.params,
        )
    elif spec.factory == "donchian":
        from Strategy.DonchianATRVolTarget import Strategy as DonchianStrategy
        return DonchianStrategy(**spec.params)
    else:
        raise ValueError(f"Unknown strategy factory: {spec.factory}")


# ---------------------------------------------------------------------------
# Run one (mode, strategy) pair
# ---------------------------------------------------------------------------

@dataclass
class RunResult:
    mode: str
    mode_label: str
    strategy_name: str
    # Classic metrics (restored)
    total_return_pct: float
    annualized_return: float
    sharpe: float
    sortino: float
    calmar: float
    max_drawdown: float
    total_trades: int
    win_rate: float
    profit_factor: float
    avg_trade_return: float
    turnover: float
    # Costs
    total_fees: float
    total_slippage: float
    total_costs: float
    # Scoring
    score: float
    score_breakdown: Dict[str, float]
    # Latency distribution stats
    avg_latency_ms: float
    p95_latency_ms: float
    p99_latency_ms: float
    # Slippage distribution stats
    avg_slippage_bps: float
    p95_slippage_bps: float
    p99_slippage_bps: float
    # Latency price-impact stats (PART 1)
    fills_bar_shifted: int = 0
    avg_price_shift: float = 0.0
    p95_price_shift: float = 0.0
    # Mode config labels
    price_latency_mode: str = "timestamp_only"
    intrabar_price_model: str = "none"
    # Timing
    elapsed_s: float = 0.0
    # Extended metadata
    metadata: Dict[str, Any] = field(default_factory=dict)


# Shared scorer instance
_scorer = Scorer(
    weights=ScorerWeights(sharpe=1.0, max_drawdown=0.5, turnover=0.1, costs=0.2),
    min_trades=5,
)


def _run_one(
    mode_name: str,
    mode_cfg: Dict[str, Any],
    spec: StrategySpec,
    base_config: Dict[str, Any],
) -> RunResult:
    """Run a single backtest and return structured results."""
    # Merge base config + mode overrides
    cfg_dict = dict(base_config)
    cfg_dict.update(mode_cfg["engine_overrides"])
    cfg_dict["realism"] = mode_cfg["realism"]
    cfg_dict["random_seed"] = mode_cfg.get("seed", 42)
    # Per-mode close_positions_at_end override
    if "close_positions_at_end" in mode_cfg:
        cfg_dict["close_positions_at_end"] = mode_cfg["close_positions_at_end"]

    config = EngineConfig(**cfg_dict)
    engine = BacktestEngine(config)
    strategy = _create_strategy(spec)

    t0 = time.perf_counter()
    result: BacktestResult = engine.run(strategy, sizing_config=spec.sizing)
    elapsed = time.perf_counter() - t0

    # Compute score + breakdown
    sc = _scorer.score(result)
    sb = _scorer.score_breakdown(result)

    meta = result.metadata or {}
    pf = result.profit_factor
    if pf == float("inf"):
        pf = 999.0

    # Latency distribution stats (ns → ms for readability)
    avg_lat_ms = meta.get("avg_latency_ns", 0) / 1e6
    p95_lat_ms = meta.get("p95_latency_ns", 0) / 1e6
    p99_lat_ms = meta.get("p99_latency_ns", 0) / 1e6

    return RunResult(
        mode=mode_name,
        mode_label=mode_cfg["label"],
        strategy_name=spec.name,
        total_return_pct=result.total_return_pct,
        annualized_return=result.annualized_return,
        sharpe=result.sharpe_ratio,
        sortino=result.sortino_ratio,
        calmar=result.calmar_ratio,
        max_drawdown=result.max_drawdown,
        total_trades=result.total_trades,
        win_rate=result.win_rate,
        profit_factor=pf,
        avg_trade_return=result.avg_trade_return,
        turnover=result.turnover,
        total_fees=result.total_fees,
        total_slippage=result.total_slippage,
        total_costs=result.total_costs,
        score=sc,
        score_breakdown=sb,
        avg_latency_ms=avg_lat_ms,
        p95_latency_ms=p95_lat_ms,
        p99_latency_ms=p99_lat_ms,
        avg_slippage_bps=meta.get("avg_slippage_bps", 0),
        p95_slippage_bps=meta.get("p95_slippage_bps", 0),
        p99_slippage_bps=meta.get("p99_slippage_bps", 0),
        fills_bar_shifted=meta.get("fills_bar_shifted", 0),
        avg_price_shift=meta.get("avg_price_shift", 0),
        p95_price_shift=meta.get("p95_price_shift", 0),
        price_latency_mode=mode_cfg["realism"].transaction_costs.price_latency_mode,
        intrabar_price_model=getattr(mode_cfg["realism"].transaction_costs,
                                     "intrabar_price_model", "none"),
        elapsed_s=elapsed,
        metadata={
            "fill_count": meta.get("fill_count", 0),
            "partial_fills": meta.get("partial_fills", 0),
            "avg_latency_ns": meta.get("avg_latency_ns", 0),
            "total_fee_cost": meta.get("total_fee_cost", 0),
            "total_spread_cost": meta.get("total_spread_cost", 0),
            "total_slippage_cost": meta.get("total_slippage_cost", 0),
            "total_funding_cost": meta.get("total_funding_cost", 0),
            "total_borrow_cost": meta.get("total_borrow_cost", 0),
            "total_traded_notional": meta.get("total_traded_notional", 0),
        },
    )


# ---------------------------------------------------------------------------
# Donchian 0-trade diagnostic (PART 4)
# ---------------------------------------------------------------------------

def _donchian_debug(results: List[RunResult], base_config: dict) -> None:
    """Print diagnostic info for any Donchian runs that produced 0 trades."""
    zero_trade_runs = [
        r for r in results if "Donchian" in r.strategy_name and r.total_trades == 0
    ]
    if not zero_trade_runs:
        return

    print("\n" + "=" * 80)
    print("DONCHIAN 0-TRADE DIAGNOSTIC")
    print("=" * 80)

    # Count available bars once
    bar_count: Optional[int] = None
    try:
        from Backtest.tick_store import TickStore, TickStoreConfig
        from Backtest.bar_builder import TimeBarBuilder
        sym = base_config["symbols"][0]
        tf = base_config["timeframe"]
        data_dir = base_config.get("tick_data_dir", "data/ticks")
        store = TickStore(TickStoreConfig(data_dir=data_dir))
        builder = TimeBarBuilder(symbol=sym, timeframe=tf)
        bar_count = 0
        for tick in store.iter_ticks(sym):
            completed = builder.on_tick(tick)
            if completed is not None:
                bar_count += 1
        print(f"\n  Total {tf} bars from data: {bar_count}")
    except Exception as exc:
        print(f"\n  [could not count bars: {exc}]")

    for r in zero_trade_runs:
        print(f"\n--- {r.mode} × {r.strategy_name} ---")
        print(f"  Return: {r.total_return_pct:.6f}% | Trades: {r.total_trades}")

        # Find the matching strategy spec
        spec = None
        for s in _build_strategies(base_config.get("leverage", 1.0),
                                   base_config.get("leverage_mode", "notional")):
            if s.name == r.strategy_name:
                spec = s
                break
        if spec is None:
            print("  [skip: strategy spec not found]")
            continue

        params = spec.params
        print(f"  Strategy params: {params}")
        dc_period = params.get("dc_period", 20)
        filter_type = params.get("filter_type", "ema")
        ema_period = params.get("ema_period", 100)
        print(f"  Warmup: dc_period={dc_period}, filter_type={filter_type}, ema_period={ema_period}")

        if bar_count is not None:
            warmup_needed = max(dc_period, ema_period) if filter_type != "none" else dc_period
            if bar_count <= warmup_needed:
                print(f"  >> ROOT CAUSE: Only {bar_count} bars vs {warmup_needed}+ needed for warmup!")
            else:
                print(f"  Bars after warmup: ~{bar_count - warmup_needed}")
                print("  >> Strategy has enough bars. Possible causes:")
                print("     - Donchian channel never breached (tight range data)")
                if filter_type != "none":
                    print(f"     - {filter_type} filter (period={ema_period}) blocks entries")
                print("     - Check _live_signal() logic for signal generation conditions")

        # Compare with non-zero-trade Donchian runs in results
        non_zero = [x for x in results if "Donchian" in x.strategy_name and x.total_trades > 0]
        if non_zero:
            print(f"  FYI: {len(non_zero)} other Donchian runs DO produce trades:")
            for x in non_zero:
                print(f"    {x.mode} × {x.strategy_name}: {x.total_trades} trades")

    print("=" * 80 + "\n")


# ---------------------------------------------------------------------------
# Output formatters
# ---------------------------------------------------------------------------

def _render_markdown(results: List[RunResult]) -> str:
    """Build a Markdown table from run results."""
    lines: List[str] = []
    lines.append("# Experiment Results\n")

    # Main performance table
    lines.append("## Performance Summary\n")
    lines.append(
        "| Mode | Strategy | Return% | Sharpe | Sortino | MaxDD "
        "| Trades | WinRate | PF | Costs | Score | Time |")
    lines.append(
        "|------|----------|---------|--------|---------|------"
        "|--------|---------|-----|-------|-------|------|")
    for r in results:
        pf_str = f"{r.profit_factor:.2f}" if r.profit_factor < 900 else "inf"
        score_str = f"{r.score:+.3f}" if math.isfinite(r.score) else "N/A"
        lines.append(
            f"| {r.mode} "
            f"| {r.strategy_name} "
            f"| {r.total_return_pct:+.3f} "
            f"| {r.sharpe:+.3f} "
            f"| {r.sortino:+.3f} "
            f"| {r.max_drawdown:.4f} "
            f"| {r.total_trades} "
            f"| {r.win_rate:.1%} "
            f"| {pf_str} "
            f"| {r.total_costs:.2f} "
            f"| {score_str} "
            f"| {r.elapsed_s:.1f}s |")

    # Latency distribution table (only modes with latency)
    lat_runs = [r for r in results if r.avg_latency_ms > 0]
    if lat_runs:
        lines.append("\n## Latency Distribution (ms)\n")
        lines.append("| Mode | Strategy | Avg | P95 | P99 |")
        lines.append("|------|----------|-----|-----|-----|")
        for r in lat_runs:
            lines.append(
                f"| {r.mode} "
                f"| {r.strategy_name} "
                f"| {r.avg_latency_ms:.2f} "
                f"| {r.p95_latency_ms:.2f} "
                f"| {r.p99_latency_ms:.2f} |")

    # Slippage distribution table (only modes with slippage noise)
    slip_runs = [r for r in results if r.avg_slippage_bps > 0]
    if slip_runs:
        lines.append("\n## Slippage Distribution (bps)\n")
        lines.append("| Mode | Strategy | Avg | P95 | P99 | TotalSlipCost |")
        lines.append("|------|----------|-----|-----|-----|---------------|")
        for r in slip_runs:
            lines.append(
                f"| {r.mode} "
                f"| {r.strategy_name} "
                f"| {r.avg_slippage_bps:.3f} "
                f"| {r.p95_slippage_bps:.3f} "
                f"| {r.p99_slippage_bps:.3f} "
                f"| {r.metadata['total_slippage_cost']:.4f} |")

    # Fill price-shift summary (PART 1 — observable latency price impact)
    shift_runs = [r for r in results if r.fills_bar_shifted > 0]
    if shift_runs:
        lines.append("\n## Fill Price-Shift Summary (latency impact)\n")
        lines.append("| Mode | Strategy | PriceLatMode | IntrabarModel | Shifted | AvgShift | P95Shift |")
        lines.append("|------|----------|-------------|---------------|---------|----------|----------|")
        for r in shift_runs:
            lines.append(
                f"| {r.mode} "
                f"| {r.strategy_name} "
                f"| {r.price_latency_mode} "
                f"| {r.intrabar_price_model} "
                f"| {r.fills_bar_shifted} "
                f"| {r.avg_price_shift:.8f} "
                f"| {r.p95_price_shift:.8f} |")
    # Note for latency-only runs (no intrabar model)
    lat_noshifts = [r for r in results
                    if r.price_latency_mode == "price_aware"
                    and r.intrabar_price_model == "none"
                    and r.fills_bar_shifted == 0
                    and r.avg_latency_ms > 0]
    if lat_noshifts:
        lines.append("")
        lines.append("> **Note:** No bar boundary crossed → latency distribution doesn't "
                     "affect fill price at this timeframe.  Use shorter timeframes (1m) "
                     "or longer latencies (seconds) for observable price differences, "
                     'or set `intrabar_price_model="gaussian_clamped"` for an approximation.')

    # Funding/borrow highlight (PART 3 — non-zero costs)
    fb_runs = [r for r in results
                if r.metadata["total_funding_cost"] > 0 or r.metadata["total_borrow_cost"] > 0]
    if fb_runs:
        lines.append("\n## Funding & Borrow Costs (non-zero only)\n")
        lines.append("| Mode | Strategy | Funding | Borrow | Trades | Return% |")
        lines.append("|------|----------|---------|--------|--------|---------|")
        for r in fb_runs:
            m = r.metadata
            lines.append(
                f"| {r.mode} "
                f"| {r.strategy_name} "
                f"| {m['total_funding_cost']:.6f} "
                f"| {m['total_borrow_cost']:.6f} "
                f"| {r.total_trades} "
                f"| {r.total_return_pct:+.3f} |")

    # Decomposed cost table
    lines.append("\n## Decomposed Costs\n")
    lines.append("| Mode | Strategy | Fee | Spread | Slippage | Funding | Borrow | Notional |")
    lines.append("|------|----------|-----|--------|----------|---------|--------|----------|")
    for r in results:
        m = r.metadata
        lines.append(
            f"| {r.mode} "
            f"| {r.strategy_name} "
            f"| {m['total_fee_cost']:.4f} "
            f"| {m['total_spread_cost']:.4f} "
            f"| {m['total_slippage_cost']:.4f} "
            f"| {m['total_funding_cost']:.6f} "
            f"| {m['total_borrow_cost']:.6f} "
            f"| {m.get('total_traded_notional', 0):.2f} |")

    # Score breakdown table (only for runs with enough trades)
    scored = [r for r in results if r.score_breakdown.get("min_trades_met", False)]
    if scored:
        lines.append("\n## Score Breakdown (min-trades met)\n")
        lines.append(
            "| Mode | Strategy | Sharpe | DD-pen | Turn-pen | Cost-pen "
            "| WR-bonus | PF-bonus | Calmar-bonus | Total |")
        lines.append(
            "|------|----------|--------|--------|----------|---------"
            "|----------|----------|--------------|-------|")
        for r in scored:
            sb = r.score_breakdown
            lines.append(
                f"| {r.mode} "
                f"| {r.strategy_name} "
                f"| {sb.get('sharpe_component', 0):+.3f} "
                f"| {sb.get('drawdown_component', 0):+.3f} "
                f"| {sb.get('turnover_component', 0):+.3f} "
                f"| {sb.get('costs_component', 0):+.3f} "
                f"| {sb.get('win_rate_component', 0):+.3f} "
                f"| {sb.get('profit_factor_component', 0):+.3f} "
                f"| {sb.get('calmar_component', 0):+.3f} "
                f"| {sb.get('total_score', 0):+.3f} |")

    return "\n".join(lines) + "\n"


def _write_csv(results: List[RunResult], path: str) -> None:
    """Write results to a CSV file."""
    fieldnames = [
        "mode", "mode_label", "strategy", "return_pct", "annualized_return",
        "sharpe", "sortino", "calmar", "max_drawdown", "trades", "win_rate",
        "profit_factor", "avg_trade_return", "turnover",
        "fees", "slippage", "total_costs", "score",
        "avg_latency_ms", "p95_latency_ms", "p99_latency_ms",
        "avg_slippage_bps", "p95_slippage_bps", "p99_slippage_bps",
        "fills_bar_shifted", "avg_price_shift", "p95_price_shift",
        "price_latency_mode", "intrabar_price_model",
        "fee_cost", "spread_cost", "slippage_cost", "funding_cost", "borrow_cost",
        "total_traded_notional",
        "fill_count", "partial_fills", "elapsed_s",
    ]
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in results:
            m = r.metadata
            writer.writerow({
                "mode": r.mode,
                "mode_label": r.mode_label,
                "strategy": r.strategy_name,
                "return_pct": f"{r.total_return_pct:.4f}",
                "annualized_return": f"{r.annualized_return:.6f}",
                "sharpe": f"{r.sharpe:.4f}",
                "sortino": f"{r.sortino:.4f}",
                "calmar": f"{r.calmar:.4f}",
                "max_drawdown": f"{r.max_drawdown:.6f}",
                "trades": r.total_trades,
                "win_rate": f"{r.win_rate:.4f}",
                "profit_factor": f"{r.profit_factor:.4f}",
                "avg_trade_return": f"{r.avg_trade_return:.6f}",
                "turnover": f"{r.turnover:.6f}",
                "fees": f"{r.total_fees:.6f}",
                "slippage": f"{r.total_slippage:.6f}",
                "total_costs": f"{r.total_costs:.6f}",
                "score": f"{r.score:.4f}",
                "avg_latency_ms": f"{r.avg_latency_ms:.4f}",
                "p95_latency_ms": f"{r.p95_latency_ms:.4f}",
                "p99_latency_ms": f"{r.p99_latency_ms:.4f}",
                "avg_slippage_bps": f"{r.avg_slippage_bps:.4f}",
                "p95_slippage_bps": f"{r.p95_slippage_bps:.4f}",
                "p99_slippage_bps": f"{r.p99_slippage_bps:.4f}",
                "fills_bar_shifted": r.fills_bar_shifted,
                "avg_price_shift": f"{r.avg_price_shift:.8f}",
                "p95_price_shift": f"{r.p95_price_shift:.8f}",
                "price_latency_mode": r.price_latency_mode,
                "intrabar_price_model": r.intrabar_price_model,
                "fee_cost": f"{m['total_fee_cost']:.6f}",
                "spread_cost": f"{m['total_spread_cost']:.6f}",
                "slippage_cost": f"{m['total_slippage_cost']:.6f}",
                "funding_cost": f"{m['total_funding_cost']:.8f}",
                "borrow_cost": f"{m['total_borrow_cost']:.8f}",
                "total_traded_notional": f"{m.get('total_traded_notional', 0):.2f}",
                "fill_count": m["fill_count"],
                "partial_fills": m["partial_fills"],
                "elapsed_s": f"{r.elapsed_s:.2f}",
            })
    print(f"\nCSV written to: {path}")


# ---------------------------------------------------------------------------
# CSV mode loader (PART 4 — --modes-csv)
# ---------------------------------------------------------------------------

# Type coercion helpers for CSV string values
_BOOL_TRUE = {"true", "1", "yes", "on"}
_BOOL_FALSE = {"false", "0", "no", "off", ""}


def _coerce(value: str) -> Any:
    """Best-effort coerce a CSV cell string to int, float, or bool."""
    v = value.strip()
    if v.lower() in _BOOL_TRUE:
        return True
    if v.lower() in _BOOL_FALSE:
        return False
    try:
        return int(v)
    except ValueError:
        pass
    try:
        return float(v)
    except ValueError:
        pass
    return v


# TransactionCostConfig field names for CSV → realism mapping
_TC_FIELDS = set(TransactionCostConfig.__dataclass_fields__)
_FUND_FIELDS = set(FundingConfig.__dataclass_fields__)
_BORR_FIELDS = set(BorrowConfig.__dataclass_fields__)

# Engine-level overrides that can be set from CSV
_ENGINE_KEYS = {
    "taker_fee_bps", "maker_fee_bps", "slippage_bps", "spread_bps",
    "latency_ns", "latency_jitter_ns", "enable_partial_fills",
    "liquidity_scale", "min_fill_ratio",
}


def _load_modes_from_csv(csv_path: str) -> Dict[str, Dict[str, Any]]:
    """
    Parse a CSV file where each row defines a mode.

    Required column: ``mode_name``
    Optional column: ``label`` (defaults to mode_name), ``description``

    Any column whose name matches a TransactionCostConfig / FundingConfig /
    BorrowConfig field will be routed to the appropriate config.  Columns
    matching engine-level keys (taker_fee_bps, spread_bps, …) go into
    ``engine_overrides``.  Columns named ``seed``, ``close_positions_at_end``
    are set at the mode top-level.

    Unrecognised columns are silently ignored.
    """
    modes: Dict[str, Dict[str, Any]] = {}
    with open(csv_path, "r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row_num, row in enumerate(reader, start=2):
            name = row.get("mode_name", "").strip()
            if not name:
                print(f"  [modes-csv] skipping row {row_num}: missing mode_name")
                continue

            label = row.get("label", "").strip() or name
            desc = row.get("description", "").strip()

            tc_kw: Dict[str, Any] = {}
            fund_kw: Dict[str, Any] = {}
            borr_kw: Dict[str, Any] = {}
            eng: Dict[str, Any] = {}
            top: Dict[str, Any] = {}

            for col, raw in row.items():
                if col in ("mode_name", "label", "description") or raw is None:
                    continue
                val = _coerce(raw)
                if col in _TC_FIELDS:
                    tc_kw[col] = val
                elif col in _FUND_FIELDS:
                    fund_kw[col] = val
                elif col in _BORR_FIELDS:
                    borr_kw[col] = val
                elif col in _ENGINE_KEYS:
                    eng[col] = val
                elif col in ("seed", "close_positions_at_end"):
                    top[col] = val
                else:
                    log.warning("[modes-csv] row %d: unknown column '%s' — ignored", row_num, col)

            # Build RealismConfig from collected kwargs
            realism = RealismConfig(
                transaction_costs=TransactionCostConfig(**tc_kw),
                funding=FundingConfig(**fund_kw),
                borrow=BorrowConfig(**borr_kw),
            )

            # Fill in engine defaults that weren't specified
            eng.setdefault("taker_fee_bps", 4.0)
            eng.setdefault("maker_fee_bps", 2.0)
            eng.setdefault("slippage_bps", 1.0)
            eng.setdefault("spread_bps", 2.0)
            eng.setdefault("latency_ns", 0)
            eng.setdefault("latency_jitter_ns", 0)
            eng.setdefault("enable_partial_fills", False)

            mode_entry: Dict[str, Any] = {
                "label": label,
                "description": desc,
                "engine_overrides": eng,
                "realism": realism,
            }
            mode_entry.update(top)
            modes[name] = mode_entry

    return modes


# ---------------------------------------------------------------------------
# --dump-effective-config helper
# ---------------------------------------------------------------------------

def _dump_effective_configs(
    mode_names: List[str], all_modes: Dict[str, Dict[str, Any]]
) -> None:
    """Print the resolved RealismConfig + engine overrides per mode as
    indented YAML-pasteable text so users can copy into config files."""
    import dataclasses

    def _fmt(obj: Any, indent: int = 0) -> str:
        """Recursively format a value as YAML-ish text."""
        pad = "  " * indent
        if isinstance(obj, dict):
            if not obj:
                return "{}"
            lines = []
            for k, v in obj.items():
                formatted = _fmt(v, indent + 1)
                if isinstance(v, (dict,)) and v:
                    lines.append(f"{pad}{k}:")
                    lines.append(formatted)
                else:
                    lines.append(f"{pad}{k}: {formatted}")
            return "\n".join(lines)
        if isinstance(obj, bool):
            return "true" if obj else "false"
        if isinstance(obj, float):
            return f"{obj:g}"
        return str(obj)

    print("=" * 72)
    print("EFFECTIVE CONFIG DUMP  (--dump-effective-config)")
    print("=" * 72)
    for name in mode_names:
        cfg = all_modes[name]
        realism: RealismConfig = cfg["realism"]
        d = dataclasses.asdict(realism)
        print(f"\n--- {name} ({cfg.get('label', '')}) ---")
        print(f"# engine_overrides:")
        print(_fmt(cfg["engine_overrides"], indent=1))
        print(f"# realism:")
        print(_fmt(d, indent=1))
    print()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Experiment harness: 4 realism modes × multiple strategies"
    )
    parser.add_argument("--data-dir", default="./data/ticks", help="Tick data directory")
    parser.add_argument("--symbol", default="DOGEUSDT", help="Trading symbol")
    parser.add_argument("--timeframe", default="15m", help="Bar timeframe")
    parser.add_argument("--capital", type=float, default=10000.0, help="Initial capital")
    parser.add_argument("--leverage-mode", default="margin", choices=["spot", "margin"])
    parser.add_argument("--leverage", type=float, default=10.0, help="Leverage multiplier")
    parser.add_argument("--csv", default=None, help="Optional CSV output path")
    parser.add_argument("--modes", nargs="*", default=None,
                        help="Run only specific modes, e.g. MODE_0 MODE_3")
    parser.add_argument("--close-at-end", action="store_true", default=False,
                        help="Force-close open positions at end of data for all modes")
    parser.add_argument("--modes-csv", default=None,
                        help="Path to CSV file defining additional modes (see example_modes.csv)")
    parser.add_argument("--dump-effective-config", action="store_true", default=False,
                        help="Print resolved RealismConfig (YAML-pasteable) per mode and exit")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.WARNING,
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
    )

    # Base engine config shared across all runs
    target_notional = 100.0 * (args.leverage if args.leverage_mode == "margin" else 1.0)
    base_config: Dict[str, Any] = {
        "tick_data_dir": args.data_dir,
        "symbols": [args.symbol],
        "bar_type": "time",
        "timeframe": args.timeframe,
        "initial_capital": args.capital,
        "leverage_mode": args.leverage_mode,
        "leverage": args.leverage,
        "maintenance_margin_ratio": 0.5,
        "max_position_size": 1_000_000.0,
        "max_position_notional": max(target_notional * 2, 10_000.0),
        "max_daily_loss": args.capital * 0.1,
        "max_drawdown": 0.5,
        "close_positions_at_end": False,
        "enable_tick_exit": True,
        "bar_store_maxlen": 600,
    }

    if args.close_at_end:
        base_config["close_positions_at_end"] = True

    # Strategy matrix
    strategies = _build_strategies(args.leverage, args.leverage_mode)

    # Load CSV-defined modes (if provided) and merge into MODES
    all_modes = dict(MODES)
    if args.modes_csv:
        csv_modes = _load_modes_from_csv(args.modes_csv)
        all_modes.update(csv_modes)
        print(f"Loaded {len(csv_modes)} mode(s) from {args.modes_csv}: "
              f"{list(csv_modes.keys())}")

    # Filter modes
    mode_names = args.modes or list(all_modes.keys())
    for m in mode_names:
        if m not in all_modes:
            print(f"ERROR: unknown mode '{m}'. Available: {list(all_modes.keys())}")
            sys.exit(1)

    # --dump-effective-config: print resolved realism and exit
    if args.dump_effective_config:
        _dump_effective_configs(mode_names, all_modes)
        sys.exit(0)

    total_runs = len(mode_names) * len(strategies)
    print(f"Running {len(mode_names)} modes × {len(strategies)} strategies = {total_runs} experiments\n")

    results: List[RunResult] = []
    run_idx = 0
    for mode_name in mode_names:
        mode_cfg = all_modes[mode_name]
        for spec in strategies:
            run_idx += 1
            tag = f"[{run_idx}/{total_runs}] {mode_name} × {spec.name}"
            print(f"  {tag} ...", end=" ", flush=True)
            try:
                r = _run_one(mode_name, mode_cfg, spec, base_config)
                results.append(r)
                extra = ""
                if r.fills_bar_shifted > 0:
                    extra = f", shifted={r.fills_bar_shifted}"
                if r.metadata["total_funding_cost"] > 0 or r.metadata["total_borrow_cost"] > 0:
                    extra += f", fund={r.metadata['total_funding_cost']:.4f}/borr={r.metadata['total_borrow_cost']:.4f}"
                print(f"done ({r.total_trades} trades, {r.elapsed_s:.1f}s{extra})")
            except Exception as exc:
                print(f"FAILED: {exc}")
                log.exception("Run failed: %s", tag)

    # Donchian 0-trade diagnostic (PART 4)
    _donchian_debug(results, base_config)

    # Output
    print("\n" + "=" * 80)
    md = _render_markdown(results)
    print(md)

    if args.csv:
        _write_csv(results, args.csv)

    print("=" * 80)
    print(f"Completed {len(results)}/{total_runs} runs.")


if __name__ == "__main__":
    main()
