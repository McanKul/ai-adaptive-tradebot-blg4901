"""
utils/leverage_utils.py
=======================
Reusable leverage-aware utility functions for strategies and backtest components.

These utilities provide consistent leverage calculations that can be used by:
- Strategy wrappers for sizing decisions
- Exit managers for TP/SL calculations
- Risk managers for exposure checks

DESIGN PRINCIPLES:
- All functions are pure (no side effects)
- All functions are deterministic
- Default behavior (leverage=1.0) matches spot trading
- Leverage-aware behavior is opt-in

SEMANTIC CLARITY:
- quantity: Number of units/contracts (e.g., 1000 DOGE)
- notional: Value in quote currency (quantity * price, e.g., $80 USD)
- margin: Collateral required (notional / leverage, e.g., $8 USD at 10x)
- exposure: Market exposure (same as notional for linear instruments)
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Optional


@dataclass
class LeverageConfig:
    """Configuration for leverage-aware calculations."""
    # Leverage multiplier (1.0 = spot, >1.0 = leveraged)
    leverage: float = 1.0
    
    # Fee rates in basis points
    taker_fee_bps: float = 10.0  # 0.1%
    maker_fee_bps: float = 10.0  # 0.1%
    
    # Margin settings
    maintenance_margin_ratio: float = 0.5  # Maintenance = initial * this
    
    @property
    def initial_margin_rate(self) -> float:
        """Fraction of notional required as initial margin."""
        return 1.0 / self.leverage if self.leverage > 0 else 1.0
    
    @property
    def maintenance_margin_rate(self) -> float:
        """Fraction of notional required as maintenance margin."""
        return self.initial_margin_rate * self.maintenance_margin_ratio


def calculate_notional(
    quantity: float,
    price: float,
) -> float:
    """
    Calculate position notional value.
    
    Args:
        quantity: Position size (can be negative for short)
        price: Current price
        
    Returns:
        Absolute notional value in quote currency
        
    SEMANTIC: Notional is always the market value of the position.
    """
    return abs(quantity) * price


def calculate_margin_required(
    quantity: float,
    price: float,
    leverage: float = 1.0,
) -> float:
    """
    Calculate initial margin required for a position.
    
    Args:
        quantity: Position size
        price: Entry/current price
        leverage: Leverage multiplier (1.0 = full notional, 10.0 = 10% of notional)
        
    Returns:
        Margin required in quote currency
        
    FORMULA: margin = notional / leverage
    
    Example:
        - 1000 DOGE at $0.08 = $80 notional
        - At 10x leverage: margin = $80 / 10 = $8
    """
    notional = calculate_notional(quantity, price)
    if leverage <= 0:
        leverage = 1.0
    return notional / leverage


def calculate_unrealized_pnl(
    quantity: float,
    entry_price: float,
    current_price: float,
) -> float:
    """
    Calculate unrealized P&L for a position.
    
    Args:
        quantity: Position size (positive = long, negative = short)
        entry_price: Average entry price
        current_price: Current market price
        
    Returns:
        Unrealized P&L in quote currency
        
    IMPORTANT: P&L is the SAME regardless of leverage.
    Leverage affects margin, not P&L magnitude.
    """
    return (current_price - entry_price) * quantity


def calculate_pnl_percentage(
    quantity: float,
    entry_price: float,
    current_price: float,
    leverage: float = 1.0,
) -> float:
    """
    Calculate P&L as percentage of capital used.
    
    Args:
        quantity: Position size
        entry_price: Average entry price
        current_price: Current market price
        leverage: Leverage multiplier (affects denominator)
        
    Returns:
        P&L as fraction (0.10 = 10%)
        
    FORMULA: pnl% = unrealized_pnl / margin_used
             margin_used = notional / leverage
    
    Example:
        - 1000 DOGE at $0.08, now $0.088 = $8 PnL
        - Spot (1x): $8 / $80 = 10%
        - 10x leverage: $8 / $8 = 100%
        
    This explains why leverage amplifies returns (and losses).
    """
    pnl = calculate_unrealized_pnl(quantity, entry_price, current_price)
    margin = calculate_margin_required(quantity, entry_price, leverage)
    if margin == 0:
        return 0.0
    return pnl / margin


def calculate_liquidation_price(
    quantity: float,
    entry_price: float,
    margin: float,
    maintenance_margin_ratio: float = 0.5,
) -> Optional[float]:
    """
    Calculate price at which position would be liquidated.
    
    Args:
        quantity: Position size (positive = long, negative = short)
        entry_price: Average entry price
        margin: Margin/collateral provided
        maintenance_margin_ratio: Maintenance / Initial margin ratio
        
    Returns:
        Liquidation price, or None if position cannot be liquidated
        
    FORMULA:
        For longs: liq_price = entry - (margin * (1 - mm_ratio)) / quantity
        For shorts: liq_price = entry + (margin * (1 - mm_ratio)) / quantity
    """
    if abs(quantity) < 1e-10:
        return None
    
    # Buffer before liquidation (margin minus maintenance)
    buffer = margin * (1 - maintenance_margin_ratio)
    
    if quantity > 0:  # Long position
        liq_price = entry_price - (buffer / abs(quantity))
        return max(0.0, liq_price)  # Can't go below 0
    else:  # Short position
        liq_price = entry_price + (buffer / abs(quantity))
        return liq_price


def is_liquidation_triggered(
    quantity: float,
    entry_price: float,
    current_price: float,
    margin: float,
    maintenance_margin_ratio: float = 0.5,
) -> bool:
    """
    Check if position should be liquidated at current price.
    
    Args:
        quantity: Position size
        entry_price: Average entry price
        current_price: Current market price
        margin: Margin/collateral provided
        maintenance_margin_ratio: Maintenance / Initial margin ratio
        
    Returns:
        True if liquidation should occur
    """
    liq_price = calculate_liquidation_price(
        quantity, entry_price, margin, maintenance_margin_ratio
    )
    
    if liq_price is None:
        return False
    
    if quantity > 0:  # Long
        return current_price <= liq_price
    else:  # Short
        return current_price >= liq_price


def calculate_position_size_from_notional(
    target_notional: float,
    price: float,
) -> float:
    """
    Convert target notional to position quantity.
    
    Args:
        target_notional: Desired notional exposure
        price: Current price
        
    Returns:
        Position quantity
        
    Useful when strategy specifies exposure in USD rather than quantity.
    """
    if price <= 0:
        return 0.0
    return target_notional / price


def calculate_position_size_from_margin(
    available_margin: float,
    price: float,
    leverage: float = 1.0,
) -> float:
    """
    Calculate maximum position size given available margin.
    
    Args:
        available_margin: Available margin/collateral
        price: Current price
        leverage: Leverage multiplier
        
    Returns:
        Maximum position quantity
        
    FORMULA: quantity = (margin * leverage) / price
    """
    if price <= 0:
        return 0.0
    return (available_margin * leverage) / price


def calculate_fee(
    quantity: float,
    price: float,
    fee_bps: float = 10.0,
) -> float:
    """
    Calculate trading fee for an order.
    
    Args:
        quantity: Order size
        price: Execution price
        fee_bps: Fee in basis points (10 = 0.1%)
        
    Returns:
        Fee in quote currency
    """
    notional = calculate_notional(quantity, price)
    return notional * fee_bps / 10000.0


# ============================================================================
# CONVENIENCE FUNCTIONS FOR STRATEGY USE
# ============================================================================

def get_effective_exposure(
    quantity: float,
    price: float,
    leverage: float = 1.0,
) -> dict:
    """
    Get complete exposure breakdown for a position.
    
    Returns dict with:
    - quantity: Position size
    - notional: Market value
    - margin: Collateral required
    - leverage: Effective leverage
    """
    notional = calculate_notional(quantity, price)
    margin = calculate_margin_required(quantity, price, leverage)
    
    return {
        "quantity": quantity,
        "notional": notional,
        "margin": margin,
        "leverage": leverage,
    }


def validate_order_against_limits(
    order_quantity: float,
    order_price: float,
    current_position: float,
    max_position_size: float,
    max_notional: float,
    available_margin: float,
    leverage: float = 1.0,
    is_buy: bool = True,
) -> tuple[bool, str]:
    """
    Validate an order against risk limits.
    
    Returns:
        Tuple of (is_valid, rejection_reason)
    """
    # Calculate resulting position
    if is_buy:
        new_position = current_position + order_quantity
    else:
        new_position = current_position - order_quantity
    
    # Check position size limit
    if abs(new_position) > max_position_size:
        return False, f"Position size {abs(new_position):.4f} exceeds limit {max_position_size}"
    
    # Check notional limit
    new_notional = calculate_notional(new_position, order_price)
    if new_notional > max_notional:
        return False, f"Notional {new_notional:.2f} exceeds limit {max_notional}"
    
    # Check margin requirement
    margin_needed = calculate_margin_required(order_quantity, order_price, leverage)
    if margin_needed > available_margin:
        return False, f"Insufficient margin: need {margin_needed:.2f}, have {available_margin:.2f}"
    
    return True, ""
