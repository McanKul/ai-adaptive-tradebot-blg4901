"""
Backtest/realism_config.py
==========================
Configuration dataclasses for execution realism features.

All new realism features are **behind flags** and default OFF (or to legacy
behaviour) so that existing strategies produce identical results.

Hierarchy:
    TransactionCostConfig  – fees, slippage, spread, latency models
    FundingConfig           – perpetual-futures funding cost
    BorrowConfig            – margin/borrow interest cost
    RealismConfig           – top-level container forwarded through runner

YAML / JSON loading:
    RealismConfig.from_dict(d)  loads from a plain dict.
    RealismConfig.from_yaml(path)  loads from a YAML file.
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, Any, Optional
import json
import logging

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Transaction costs
# ---------------------------------------------------------------------------
@dataclass
class TransactionCostConfig:
    """Controls fee / slippage / spread / latency models."""

    # --- fee model ---
    fee_model: str = "fixed"  # "fixed" | "tiered"
    maker_fee_bps: float = 2.0
    taker_fee_bps: float = 4.0

    # --- slippage model ---
    slippage_model: str = "fixed"  # "fixed" | "volume_sqrt" | "volatility_atr" | "composite"
    slippage_bps: float = 1.0
    impact_factor: float = 0.1  # scale for sqrt / ATR models
    max_slippage_bps: float = 50.0
    atr_period_for_slippage: int = 14
    vola_lookback: int = 20
    volume_is_quote: bool = False  # True → bar.volume is already in quote currency
    liquidity_scale: float = 1.0

    # --- spread model ---
    spread_model: str = "fixed"  # "fixed" | "ohlc_estimator"
    spread_bps: float = 2.0

    # --- latency model ---
    latency_model: str = "fixed"  # "fixed" | "jitter" | "distribution"
    latency_ns: int = 0
    latency_jitter_ns: int = 0
    # Distribution: "none"|"uniform"|"exponential"|"gamma"|"lognormal"|"weibull"|"mixture"
    # Poisson is for spike EVENT arrivals only — NOT for latency magnitude.
    # Heavy-tailed dists (gamma/lognormal/weibull) model latency magnitude.
    latency_distribution: str = "uniform"
    # Gamma(k, θ): mean = k*θ ns
    latency_gamma_shape: float = 2.0
    latency_gamma_scale_ns: float = 10_000_000       # 10 ms
    # LogNormal(μ, σ) in log-ns space: median ≈ exp(μ) ns
    latency_lognormal_mu: float = 16.0               # ln(~9 ms) ≈ 16.0
    latency_lognormal_sigma: float = 0.5
    # Weibull(shape, scale): shape>1 → peaked, shape<1 → heavy tail
    latency_weibull_shape: float = 1.5
    latency_weibull_scale_ns: float = 10_000_000     # 10 ms
    # Mixture: p_spike chance of spike component, else base component
    latency_mixture_p_spike: float = 0.05
    latency_mixture_base_dist: str = "gamma"          # base component dist
    latency_mixture_spike_dist: str = "lognormal"     # spike component dist
    # Poisson spike overlay (additive, on top of base distribution)
    # Poisson models spike ARRIVALS (count process); magnitude uses a
    # separate heavy-tailed dist.  spike_lambda is per-order probability.
    spike_event_mode: str = "off"                     # "off" | "poisson_spikes"
    spike_lambda: float = 0.05                        # P(spike) per order
    spike_magnitude_dist: str = "exponential"          # "exponential" | "lognormal"
    spike_magnitude_ns: int = 100_000_000             # scale param (100 ms)

    # --- price-aware latency (A2) ---
    price_latency_mode: str = "timestamp_only"  # "timestamp_only" | "price_aware"
    price_latency_bar_field: str = "open"  # which bar field for fallback price

    # --- intra-bar price model ---
    # Controls how fill price is perturbed when latency keeps the fill
    # inside the same bar (no bar-boundary crossing).
    #   "none"             – no intra-bar perturbation (default, avoids bias)
    #   "gaussian_clamped" – Gaussian shift proportional to latency/bar_dur,
    #                        clamped to [low, high].  Approximate; read docs.
    intrabar_price_model: str = "none"  # "none" | "gaussian_clamped"

    # --- marketable-limit detection (A5) ---
    marketable_limit_is_taker: bool = False

    # --- slippage residual noise ---
    # Adds calibrated noise ON TOP of the deterministic slippage model.
    # "none" = legacy (no noise).  E[multiplier]=1 so mean slippage is preserved.
    slippage_noise: str = "none"        # "none"|"multiplicative_lognormal"|"additive_halfnormal"|"mixture"
    slippage_noise_sigma: float = 0.2   # primary noise width
    slippage_noise_mixture_p: float = 0.1   # fraction of "fat-tail" component
    slippage_noise_mixture_sigma2: float = 0.5  # fat-tail sigma

    # --- determinism ---
    seed: int = 42

    # --- per-symbol overrides ---
    per_symbol_overrides: Dict[str, Dict[str, Any]] = field(default_factory=dict)

    # helpers -----------------------------------------------------------------
    def for_symbol(self, symbol: str) -> "TransactionCostConfig":
        """Return a copy with per-symbol overrides applied (if any)."""
        overrides = self.per_symbol_overrides.get(symbol)
        if not overrides:
            return self
        import dataclasses
        d = dataclasses.asdict(self)
        d.pop("per_symbol_overrides")
        d.update(overrides)
        return TransactionCostConfig(**d, per_symbol_overrides={})


# ---------------------------------------------------------------------------
# Funding (perpetual futures)
# ---------------------------------------------------------------------------
@dataclass
class FundingConfig:
    """Perpetual-futures funding rate cost."""
    enabled: bool = False
    funding_interval_hours: int = 8
    funding_rate_mode: str = "constant"  # "constant" | "series"
    funding_rate: float = 0.0001  # per interval, e.g. 0.01 %
    # --- series mode (optional CSV) ---
    series_csv_path: str = ""           # CSV with columns: timestamp, funding_rate
    timestamp_unit: str = "ms"          # "ms" | "ns" for the CSV timestamp column
    interpolation: str = "prev"         # "prev" – use most-recent rate at each charge


# ---------------------------------------------------------------------------
# Borrow / margin interest
# ---------------------------------------------------------------------------
@dataclass
class BorrowConfig:
    """Margin borrow / interest cost."""
    enabled: bool = False
    annual_borrow_rate: float = 0.0
    charge_interval: str = "bar"  # "bar" | "daily" | "hourly"
    # --- series mode (optional CSV) ---
    series_csv_path: str = ""           # CSV: timestamp, annual_rate


# ---------------------------------------------------------------------------
# Top-level realism container
# ---------------------------------------------------------------------------
@dataclass
class RealismConfig:
    """
    Top-level container grouping all realism knobs.

    Defaults reproduce legacy behaviour (everything off or fixed models).
    """
    transaction_costs: TransactionCostConfig = field(default_factory=TransactionCostConfig)
    funding: FundingConfig = field(default_factory=FundingConfig)
    borrow: BorrowConfig = field(default_factory=BorrowConfig)

    # --- CV config (Part C) ---
    cv_enabled: bool = False
    cv_method: str = "purged_kfold"  # "purged_kfold" | "walk_forward" | "combinatorial_purged"
    cv_n_splits: int = 5
    cv_embargo_pct: float = 0.01
    cv_purge_pct: float = 0.0
    cv_train_duration_ns: int = 0
    cv_test_duration_ns: int = 0
    cv_expanding: bool = False

    # ---- serialisation helpers ---
    @staticmethod
    def from_dict(d: Dict[str, Any]) -> "RealismConfig":
        """Build from a plain dict (e.g. parsed YAML/JSON)."""
        tc_d = d.get("transaction_costs", {})
        funding_d = d.get("funding", {})
        borrow_d = d.get("borrow", {})
        tc = TransactionCostConfig(**{k: v for k, v in tc_d.items()
                                      if k in TransactionCostConfig.__dataclass_fields__})
        funding = FundingConfig(**{k: v for k, v in funding_d.items()
                                   if k in FundingConfig.__dataclass_fields__})
        borrow = BorrowConfig(**{k: v for k, v in borrow_d.items()
                                  if k in BorrowConfig.__dataclass_fields__})
        remaining = {k: v for k, v in d.items()
                     if k not in ("transaction_costs", "funding", "borrow")
                     and k in RealismConfig.__dataclass_fields__}
        return RealismConfig(transaction_costs=tc, funding=funding, borrow=borrow, **remaining)

    @staticmethod
    def from_yaml(path: str) -> "RealismConfig":
        """Load from a YAML file (requires PyYAML)."""
        try:
            import yaml  # type: ignore
        except ImportError:
            raise ImportError("PyYAML is required for YAML config loading. pip install pyyaml")
        with open(path, "r") as f:
            d = yaml.safe_load(f) or {}
        return RealismConfig.from_dict(d)

    @staticmethod
    def from_json(path: str) -> "RealismConfig":
        """Load from a JSON file."""
        with open(path, "r") as f:
            d = json.load(f)
        return RealismConfig.from_dict(d)

    def to_dict(self) -> Dict[str, Any]:
        """Serialise to plain dict."""
        import dataclasses
        return dataclasses.asdict(self)
