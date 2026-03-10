"""
Backtest/cost_models.py
=======================
Cost models for realistic transaction cost simulation.

Components:
- FixedFeeModel: Flat percentage fee
- FixedSlippageModel: Fixed basis point slippage
- SpreadCostModel: Bid-ask spread modeling
- LatencyModel: Simulated latency (no-op in backtest, but pluggable)
- CompositeCostModel: Combines all cost components

Design decisions:
- All models support deterministic behavior via seeded RNG
- Costs are always returned as positive values
- Direction handling (buy/sell) is in the caller's responsibility
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional, Dict, Any, List
from random import Random
import math
import logging

from Interfaces.orders import Order, OrderSide
from Interfaces.market_data import Bar
from Interfaces.cost_interface import ICostModel, IFeeModel, ISlippageModel, ISpreadModel, ILatencyModel

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Cost breakdown (A3)
# ---------------------------------------------------------------------------
@dataclass
class CostBreakdown:
    """Per-fill cost decomposition for reporting & debugging."""
    fee_cost_quote: float = 0.0
    spread_cost_quote: float = 0.0
    slippage_cost_quote: float = 0.0
    funding_cost_quote: float = 0.0
    borrow_cost_quote: float = 0.0

    @property
    def total_cost_quote(self) -> float:
        return (self.fee_cost_quote + self.spread_cost_quote +
                self.slippage_cost_quote + self.funding_cost_quote +
                self.borrow_cost_quote)

    def to_dict(self) -> Dict[str, float]:
        return {
            "fee_cost_quote": self.fee_cost_quote,
            "spread_cost_quote": self.spread_cost_quote,
            "slippage_cost_quote": self.slippage_cost_quote,
            "funding_cost_quote": self.funding_cost_quote,
            "borrow_cost_quote": self.borrow_cost_quote,
            "total_cost_quote": self.total_cost_quote,
        }


class FixedFeeModel:
    """
    Fixed percentage fee model.
    
    Fee = notional * fee_rate
    
    Typical values:
    - Binance taker: 0.0004 (0.04%)
    - Binance maker: 0.0002 (0.02%)
    """
    
    def __init__(
        self,
        taker_fee_rate: float = 0.0004,
        maker_fee_rate: float = 0.0002
    ):
        """
        Initialize FixedFeeModel.
        
        Args:
            taker_fee_rate: Fee rate for market orders (default 0.04%)
            maker_fee_rate: Fee rate for limit orders (default 0.02%)
        """
        self.taker_fee_rate = taker_fee_rate
        self.maker_fee_rate = maker_fee_rate
    
    def calculate(
        self,
        notional: float,
        order_side: OrderSide,
        is_maker: bool = False
    ) -> float:
        """Calculate fee for given notional value."""
        rate = self.maker_fee_rate if is_maker else self.taker_fee_rate
        return notional * rate


class FixedSlippageModel:
    """
    Fixed slippage model.
    
    Slippage = price * slippage_bps / 10000
    
    Optionally adds random component bounded by max_random_bps.
    """
    
    def __init__(
        self,
        slippage_bps: float = 1.0,
        max_random_bps: float = 0.0
    ):
        """
        Initialize FixedSlippageModel.
        
        Args:
            slippage_bps: Base slippage in basis points (default 1 bp = 0.01%)
            max_random_bps: Maximum additional random slippage in bps
        """
        self.slippage_bps = slippage_bps
        self.max_random_bps = max_random_bps
    
    def calculate(
        self,
        price: float,
        quantity: float,
        order_side: OrderSide,
        bar: Bar,
        rng: Random
    ) -> float:
        """
        Calculate slippage.
        
        Returns positive value (caller handles direction).
        """
        base_slippage = price * (self.slippage_bps / 10000)
        
        if self.max_random_bps > 0:
            random_component = rng.uniform(0, self.max_random_bps / 10000) * price
            return base_slippage + random_component
        
        return base_slippage


class VolumeSlippageModel:
    """
    Volume-based slippage model.
    
    Larger orders relative to bar volume have more slippage.
    Slippage = price * base_bps/10000 * (1 + volume_impact * (order_vol / bar_vol))
    """
    
    def __init__(
        self,
        base_bps: float = 0.5,
        volume_impact: float = 1.0
    ):
        """
        Initialize VolumeSlippageModel.
        
        Args:
            base_bps: Base slippage in basis points
            volume_impact: Multiplier for volume-dependent component
        """
        self.base_bps = base_bps
        self.volume_impact = volume_impact
    
    def calculate(
        self,
        price: float,
        quantity: float,
        order_side: OrderSide,
        bar: Bar,
        rng: Random
    ) -> float:
        """Calculate volume-adjusted slippage."""
        base = price * (self.base_bps / 10000)
        
        if bar.volume > 0:
            volume_ratio = quantity / bar.volume
            volume_factor = 1 + self.volume_impact * volume_ratio
        else:
            volume_factor = 2.0  # High slippage for no volume
        
        return base * volume_factor


class VolumeSqrtSlippageModel:
    """
    Square-root market impact model.

    impact_bps = min(max_slippage_bps,
                     impact_factor * sqrt(notional / (bar_notional_vol * liquidity_scale)) * 10000)
    """

    def __init__(
        self,
        impact_factor: float = 0.1,
        max_slippage_bps: float = 50.0,
        volume_is_quote: bool = False,
        liquidity_scale: float = 1.0,
    ):
        self.impact_factor = impact_factor
        self.max_slippage_bps = max_slippage_bps
        self.volume_is_quote = volume_is_quote
        self.liquidity_scale = max(liquidity_scale, 1e-12)

    def calculate(
        self,
        price: float,
        quantity: float,
        order_side: OrderSide,
        bar: Bar,
        rng: Random,
    ) -> float:
        notional = abs(quantity) * price
        if self.volume_is_quote:
            bar_notional = max(bar.volume, 1e-12)
        else:
            bar_notional = max(bar.volume * bar.close, 1e-12)
        ratio = notional / (bar_notional * self.liquidity_scale)
        impact_bps = min(
            self.max_slippage_bps,
            self.impact_factor * math.sqrt(ratio) * 10_000,
        )
        return price * impact_bps / 10_000


class VolatilityATRSlippageModel:
    """
    ATR-based volatility slippage: impact ∝ ATR / close.

    Requires an externally-managed ATR value injected via `set_atr`.
    """

    def __init__(
        self,
        impact_factor: float = 0.1,
        max_slippage_bps: float = 50.0,
    ):
        self.impact_factor = impact_factor
        self.max_slippage_bps = max_slippage_bps
        self._atr: float = 0.0

    def set_atr(self, atr: float) -> None:
        """Inject the current ATR value (computed externally from bars)."""
        self._atr = atr

    def calculate(
        self,
        price: float,
        quantity: float,
        order_side: OrderSide,
        bar: Bar,
        rng: Random,
    ) -> float:
        if price <= 0 or self._atr <= 0:
            return 0.0
        atr_pct = self._atr / price
        impact_bps = min(self.max_slippage_bps, self.impact_factor * atr_pct * 10_000)
        return price * impact_bps / 10_000


class SpreadCostModel:
    """
    Bid-ask spread cost model.
    
    Models the half-spread: cost of crossing from mid to bid/ask.
    """
    
    def __init__(self, spread_bps: float = 2.0):
        """
        Initialize SpreadCostModel.
        
        Args:
            spread_bps: Full spread in basis points (half-spread applied)
        """
        self.spread_bps = spread_bps
        self.half_spread_bps = spread_bps / 2
    
    def calculate(self, price: float, order_side: OrderSide) -> float:
        """
        Calculate half-spread cost.
        
        Returns positive value representing cost to cross spread.
        """
        return price * (self.half_spread_bps / 10000)


# ---------------------------------------------------------------------------
# Slippage noise wrapper (Part 1-B)
# ---------------------------------------------------------------------------

class SlippageNoiseWrapper:
    """
    Decorates any ISlippageModel with calibrated residual noise.

    Noise types
    -----------
    - ``none``                    – pass-through (legacy default).
    - ``multiplicative_lognormal``– multiply base slippage by
      exp(σZ − σ²/2) so E[multiplier]=1.0 and mean slippage is preserved.
    - ``additive_halfnormal``     – add |N(0,σ)| * price/10000 bps.
    - ``mixture``                 – with prob p use fat-tail σ₂; else σ₁.

    After noise is applied the result is clamped to [0, max_slippage_bps].
    """

    def __init__(
        self,
        base_model,
        noise_type: str = "none",
        sigma: float = 0.2,
        max_slippage_bps: float = 50.0,
        mixture_p: float = 0.1,
        mixture_sigma2: float = 0.5,
    ):
        self.base = base_model
        self.noise_type = noise_type
        self.sigma = sigma
        self.max_slippage_bps = max_slippage_bps
        self.mixture_p = mixture_p
        self.mixture_sigma2 = mixture_sigma2

    # --- delegate helpers for models that need external state ---
    def set_atr(self, atr: float) -> None:
        if hasattr(self.base, "set_atr"):
            self.base.set_atr(atr)

    def calculate(
        self,
        price: float,
        quantity: float,
        order_side: OrderSide,
        bar: Bar,
        rng: Random,
    ) -> float:
        base_slip = self.base.calculate(price, quantity, order_side, bar, rng)
        if self.noise_type == "none" or price <= 0:
            return base_slip

        max_slip = price * self.max_slippage_bps / 10_000

        if self.noise_type == "multiplicative_lognormal":
            z = rng.gauss(0, 1)
            # E[exp(σZ − σ²/2)] = 1.0  → mean-preserving multiplier
            mult = math.exp(self.sigma * z - 0.5 * self.sigma ** 2)
            noisy = base_slip * mult

        elif self.noise_type == "additive_halfnormal":
            z = abs(rng.gauss(0, self.sigma))
            noisy = base_slip + z * price / 10_000

        elif self.noise_type == "mixture":
            if rng.random() < self.mixture_p:
                sig = self.mixture_sigma2
            else:
                sig = self.sigma
            z = rng.gauss(0, 1)
            mult = math.exp(sig * z - 0.5 * sig ** 2)
            noisy = base_slip * mult

        else:
            noisy = base_slip

        return max(0.0, min(noisy, max_slip))


class LatencyModel:
    """
    Latency simulation model with configurable distributions.

    Distributions
    -------------
    - ``none``        – deterministic, returns ``base_latency_ns`` only.
    - ``uniform``     – base + U(0, max_random_latency_ns).
    - ``exponential`` – base + Exp(mean = max/2), capped at 3×max.
    - ``gamma``       – base + Gamma(k, θ_ns).   Heavy-tailed; good default.
    - ``lognormal``   – base + LogN(μ, σ) in **log-ns** space.
    - ``weibull``     – base + Weibull(shape, scale_ns).
    - ``mixture``     – with P(spike), pick spike_dist; else pick base_dist.

    Poisson spike overlay
    ---------------------
    When ``spike_event_mode="poisson_spikes"``, **each sample** has an
    independent P(spike)=``spike_lambda`` chance of an additive spike whose
    magnitude is drawn from ``spike_magnitude_dist``.  This is ON TOP of
    the chosen base distribution.

    **Design note:** Poisson models event *arrivals* (count process).
    Latency *magnitude* should use heavy-tailed dists (gamma / lognormal /
    weibull).  We use Poisson only to decide *whether* a spike event fires.
    """

    def __init__(
        self,
        base_latency_ns: int = 0,
        max_random_latency_ns: int = 0,
        latency_distribution: str = "uniform",
        # Gamma
        gamma_shape: float = 2.0,
        gamma_scale_ns: float = 10_000_000,
        # LogNormal (log-ns space)
        lognormal_mu: float = 16.0,
        lognormal_sigma: float = 0.5,
        # Weibull
        weibull_shape: float = 1.5,
        weibull_scale_ns: float = 10_000_000,
        # Mixture
        mixture_p_spike: float = 0.05,
        mixture_base_dist: str = "gamma",
        mixture_spike_dist: str = "lognormal",
        # Poisson spike overlay
        spike_event_mode: str = "off",
        spike_lambda: float = 0.05,
        spike_magnitude_dist: str = "exponential",
        spike_magnitude_ns: int = 100_000_000,
    ):
        self.base_latency_ns = base_latency_ns
        self.max_random_latency_ns = max_random_latency_ns
        self.latency_distribution = latency_distribution
        # dist params
        self.gamma_shape = gamma_shape
        self.gamma_scale_ns = gamma_scale_ns
        self.lognormal_mu = lognormal_mu
        self.lognormal_sigma = lognormal_sigma
        self.weibull_shape = weibull_shape
        self.weibull_scale_ns = weibull_scale_ns
        self.mixture_p_spike = mixture_p_spike
        self.mixture_base_dist = mixture_base_dist
        self.mixture_spike_dist = mixture_spike_dist
        # spike overlay
        self.spike_event_mode = spike_event_mode
        self.spike_lambda = spike_lambda
        self.spike_magnitude_dist = spike_magnitude_dist
        self.spike_magnitude_ns = spike_magnitude_ns

    # ---- internal samplers --------------------------------------------------

    def _sample_component(self, dist_name: str, rng: Random) -> int:
        """Sample a single latency component from the named distribution."""
        if dist_name == "gamma":
            return max(0, int(rng.gammavariate(self.gamma_shape, self.gamma_scale_ns)))
        if dist_name == "lognormal":
            return max(0, int(rng.lognormvariate(self.lognormal_mu, self.lognormal_sigma)))
        if dist_name == "weibull":
            return max(0, int(rng.weibullvariate(self.weibull_shape, 1.0) * self.weibull_scale_ns))
        if dist_name == "exponential":
            return max(0, int(rng.expovariate(1.0 / max(self.spike_magnitude_ns, 1))))
        # fallback: uniform
        if self.max_random_latency_ns > 0:
            return rng.randint(0, self.max_random_latency_ns)
        return 0

    def _sample_spike(self, rng: Random) -> int:
        """Sample spike magnitude (used by Poisson spike overlay)."""
        if self.spike_magnitude_dist == "lognormal":
            mu_spike = math.log(max(self.spike_magnitude_ns, 1))
            return max(0, int(rng.lognormvariate(mu_spike, 0.5)))
        # default: exponential with mean = spike_magnitude_ns
        return max(0, int(rng.expovariate(1.0 / max(self.spike_magnitude_ns, 1))))

    # ---- public API ---------------------------------------------------------

    def get_latency_ns(self, rng: Random) -> int:
        """
        Sample a latency value in nanoseconds (deterministic given RNG state).

        Returns
        -------
        int
            Non-negative latency in nanoseconds.
        """
        dist = self.latency_distribution

        if dist == "none":
            latency = self.base_latency_ns

        elif dist == "uniform":
            jitter = rng.randint(0, self.max_random_latency_ns) if self.max_random_latency_ns > 0 else 0
            latency = self.base_latency_ns + jitter

        elif dist == "exponential":
            if self.max_random_latency_ns > 0:
                lam = 2.0 / self.max_random_latency_ns
                sample = int(-math.log(max(rng.random(), 1e-10)) / lam)
                sample = min(sample, self.max_random_latency_ns * 3)
            else:
                sample = 0
            latency = self.base_latency_ns + sample

        elif dist == "gamma":
            latency = self.base_latency_ns + self._sample_component("gamma", rng)

        elif dist == "lognormal":
            latency = self.base_latency_ns + self._sample_component("lognormal", rng)

        elif dist == "weibull":
            latency = self.base_latency_ns + self._sample_component("weibull", rng)

        elif dist == "mixture":
            if rng.random() < self.mixture_p_spike:
                latency = self.base_latency_ns + self._sample_component(self.mixture_spike_dist, rng)
            else:
                latency = self.base_latency_ns + self._sample_component(self.mixture_base_dist, rng)

        else:
            latency = self.base_latency_ns

        # Poisson spike overlay (additive, independent of base dist choice)
        if self.spike_event_mode == "poisson_spikes" and rng.random() < self.spike_lambda:
            latency += self._sample_spike(rng)

        return max(0, latency)

    @property
    def is_enabled(self) -> bool:
        """Check if latency simulation is enabled."""
        return (
            self.base_latency_ns > 0
            or self.max_random_latency_ns > 0
            or self.latency_distribution not in ("none", "uniform")
        )


class CompositeCostModel:
    """
    Composite cost model combining all cost components.
    
    Implements the full ICostModel interface.
    """
    
    def __init__(
        self,
        fee_model: Optional[FixedFeeModel] = None,
        slippage_model: Optional[FixedSlippageModel] = None,
        spread_model: Optional[SpreadCostModel] = None,
        latency_model: Optional[LatencyModel] = None
    ):
        """
        Initialize CompositeCostModel.
        
        Args:
            fee_model: Fee model (defaults to FixedFeeModel)
            slippage_model: Slippage model (defaults to FixedSlippageModel)
            spread_model: Spread model (defaults to SpreadCostModel)
            latency_model: Latency model (defaults to LatencyModel no-op)
        """
        self.fee_model = fee_model or FixedFeeModel()
        self.slippage_model = slippage_model or FixedSlippageModel()
        self.spread_model = spread_model or SpreadCostModel()
        self.latency_model = latency_model or LatencyModel()
    
    def calculate_fee(
        self,
        notional: float,
        order_side: OrderSide,
        is_maker: bool = False
    ) -> float:
        """Calculate transaction fee."""
        return self.fee_model.calculate(notional, order_side, is_maker)
    
    def calculate_slippage(
        self,
        price: float,
        quantity: float,
        order_side: OrderSide,
        bar: Bar,
        rng: Random
    ) -> float:
        """Calculate slippage."""
        return self.slippage_model.calculate(price, quantity, order_side, bar, rng)
    
    def calculate_spread_cost(
        self,
        price: float,
        order_side: OrderSide
    ) -> float:
        """Calculate half-spread cost."""
        return self.spread_model.calculate(price, order_side)
    
    def get_latency_ns(self, rng: Random) -> int:
        """Get simulated latency."""
        return self.latency_model.get_latency_ns(rng)
    
    def total_cost(
        self,
        order: Order,
        fill_price: float,
        bar: Bar,
        rng: Random,
        is_maker: bool = False
    ) -> tuple[float, float, float]:
        """
        Calculate total transaction costs.
        
        Returns:
            Tuple of (fee, slippage, spread_cost)
        """
        notional = fill_price * order.quantity
        fee = self.calculate_fee(notional, order.side, is_maker)
        slippage = self.calculate_slippage(
            fill_price, order.quantity, order.side, bar, rng
        )
        spread = self.calculate_spread_cost(fill_price, order.side)
        
        return (fee, slippage, spread)

    def cost_breakdown(
        self,
        order: Order,
        fill_price: float,
        fill_qty: float,
        bar: Bar,
        rng: Random,
        is_maker: bool = False,
    ) -> CostBreakdown:
        """Return a detailed CostBreakdown (A3)."""
        notional = fill_price * fill_qty
        fee = self.calculate_fee(notional, order.side, is_maker)
        slippage = self.calculate_slippage(fill_price, fill_qty, order.side, bar, rng)
        spread = self.calculate_spread_cost(fill_price, order.side)
        return CostBreakdown(
            fee_cost_quote=fee,
            spread_cost_quote=spread * fill_qty,
            slippage_cost_quote=slippage * fill_qty,
        )


# Convenience factory
def create_cost_model(
    taker_fee_bps: float = 4.0,
    maker_fee_bps: float = 2.0,
    slippage_bps: float = 1.0,
    spread_bps: float = 2.0,
    latency_ns: int = 0,
    latency_jitter_ns: int = 0,
    latency_distribution: str = "uniform",
) -> CompositeCostModel:
    """
    Create a cost model with specified parameters.
    
    All inputs in basis points (bps) for consistency.
    1 bp = 0.01% = 0.0001
    
    Args:
        taker_fee_bps: Taker fee in basis points (default 4 bp = 0.04%)
        maker_fee_bps: Maker fee in basis points (default 2 bp = 0.02%)
        slippage_bps: Slippage in basis points (default 1 bp)
        spread_bps: Full spread in basis points (default 2 bp)
        latency_ns: Base latency in nanoseconds (default 0 = no latency)
        latency_jitter_ns: Max random latency jitter in ns (default 0)
        latency_distribution: "uniform" or "exponential"
        
    Returns:
        Configured CompositeCostModel
    """
    return CompositeCostModel(
        fee_model=FixedFeeModel(
            taker_fee_rate=taker_fee_bps / 10000,
            maker_fee_rate=maker_fee_bps / 10000
        ),
        slippage_model=FixedSlippageModel(slippage_bps=slippage_bps),
        spread_model=SpreadCostModel(spread_bps=spread_bps),
        latency_model=LatencyModel(
            base_latency_ns=latency_ns,
            max_random_latency_ns=latency_jitter_ns,
            latency_distribution=latency_distribution,
        )
    )


def create_cost_model_from_config(tc: "TransactionCostConfig") -> CompositeCostModel:
    """
    Build a CompositeCostModel from a TransactionCostConfig (A1/A4).

    Selects the appropriate slippage model based on ``tc.slippage_model``
    and wraps it with SlippageNoiseWrapper when ``tc.slippage_noise != "none"``.
    Latency model now supports gamma / lognormal / weibull / mixture / Poisson spikes.
    """
    from Backtest.realism_config import TransactionCostConfig  # noqa: F811

    # Fee model (always fixed for now; 'tiered' can be added later)
    fee_model = FixedFeeModel(
        taker_fee_rate=tc.taker_fee_bps / 10_000,
        maker_fee_rate=tc.maker_fee_bps / 10_000,
    )

    # Slippage model selection (A4)
    if tc.slippage_model == "volume_sqrt":
        raw_slip = VolumeSqrtSlippageModel(
            impact_factor=tc.impact_factor,
            max_slippage_bps=tc.max_slippage_bps,
            volume_is_quote=tc.volume_is_quote,
            liquidity_scale=tc.liquidity_scale,
        )
    elif tc.slippage_model == "volatility_atr":
        raw_slip = VolatilityATRSlippageModel(
            impact_factor=tc.impact_factor,
            max_slippage_bps=tc.max_slippage_bps,
        )
    else:  # "fixed" (legacy default)
        raw_slip = FixedSlippageModel(slippage_bps=tc.slippage_bps)

    # Wrap with residual noise if configured
    if tc.slippage_noise != "none":
        slippage_model = SlippageNoiseWrapper(
            base_model=raw_slip,
            noise_type=tc.slippage_noise,
            sigma=tc.slippage_noise_sigma,
            max_slippage_bps=tc.max_slippage_bps,
            mixture_p=tc.slippage_noise_mixture_p,
            mixture_sigma2=tc.slippage_noise_mixture_sigma2,
        )
    else:
        slippage_model = raw_slip

    spread_model = SpreadCostModel(spread_bps=tc.spread_bps)

    latency_model = LatencyModel(
        base_latency_ns=tc.latency_ns,
        max_random_latency_ns=tc.latency_jitter_ns,
        latency_distribution=tc.latency_distribution,
        # Gamma
        gamma_shape=tc.latency_gamma_shape,
        gamma_scale_ns=tc.latency_gamma_scale_ns,
        # LogNormal
        lognormal_mu=tc.latency_lognormal_mu,
        lognormal_sigma=tc.latency_lognormal_sigma,
        # Weibull
        weibull_shape=tc.latency_weibull_shape,
        weibull_scale_ns=tc.latency_weibull_scale_ns,
        # Mixture
        mixture_p_spike=tc.latency_mixture_p_spike,
        mixture_base_dist=tc.latency_mixture_base_dist,
        mixture_spike_dist=tc.latency_mixture_spike_dist,
        # Poisson spike overlay
        spike_event_mode=tc.spike_event_mode,
        spike_lambda=tc.spike_lambda,
        spike_magnitude_dist=tc.spike_magnitude_dist,
        spike_magnitude_ns=tc.spike_magnitude_ns,
    )

    return CompositeCostModel(
        fee_model=fee_model,
        slippage_model=slippage_model,
        spread_model=spread_model,
        latency_model=latency_model,
    )
