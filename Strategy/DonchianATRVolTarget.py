"""
Strategy/DonchianATRVolTarget.py
================================
Donchian Channel Breakout  +  ATR Trailing Stop  +  Volatility-Targeted Sizing
+  Regime Filter (EMA/ADX).

Parameters
----------
dc_period : int      Donchian channel look-back (default 20).
atr_period : int     ATR period (default 14).
atr_mult : float     Multiplier for ATR trailing stop / Chandelier exit.
risk_pct : float     Equity fraction risked per trade (default 0.005 = 0.5 %).
filter_type : str    "ema" | "adx" | "none" (default "ema").
ema_period : int     EMA period for regime filter (default 200).
adx_period : int     ADX period (default 14).
adx_threshold : float  Minimum ADX for trend classification (default 25).
allow_reversal : bool  If True, entry in opposite direction triggers exit+reverse.
max_holding_bars : int  Optional time stop.

Outputs
-------
StrategyDecision with MARKET orders, stop_price metadata, and debug features.

Integration
-----------
Implements ``IStrategy.on_bar(bar, ctx)`` → ``StrategyDecision``.
Uses ``ctx.get_ohlcv()`` for indicator computation and ``ctx.position``
for state.  Sizing is done internally via volatility-targeting; the engine's
``sizing_config`` may override (``qty_method`` in metadata signals intent).
"""
from __future__ import annotations

import math
from typing import Optional, List, Dict, Any
from collections import deque

import numpy as np

from Interfaces.IStrategy import IStrategy, StrategyDecision
from Interfaces.orders import Order, OrderType, OrderSide
from Interfaces.market_data import Bar


class Strategy(IStrategy):
    """Donchian Channel Breakout + ATR Trailing Stop + Vol-Target sizing."""

    def __init__(
        self,
        # Donchian
        dc_period: int = 20,
        # ATR / exit
        atr_period: int = 14,
        atr_mult: float = 3.0,
        # Sizing
        risk_pct: float = 0.005,
        position_size: float = 1.0,   # fallback if sizing cannot be computed
        # Regime filter
        filter_type: str = "ema",     # "ema" | "adx" | "none"
        ema_period: int = 200,
        adx_period: int = 14,
        adx_threshold: float = 25.0,
        # Behaviour
        allow_reversal: bool = True,
        max_holding_bars: Optional[int] = None,
        # misc
        **kw,
    ):
        self.dc_period = dc_period
        self.atr_period = atr_period
        self.atr_mult = atr_mult
        self.risk_pct = risk_pct
        self.position_size = position_size
        self.filter_type = filter_type
        self.ema_period = ema_period
        self.adx_period = adx_period
        self.adx_threshold = adx_threshold
        self.allow_reversal = allow_reversal
        self.max_holding_bars = max_holding_bars

        # Internal state
        self._entry_bar_index: int = 0
        self._entry_price: float = 0.0
        self._highest_since_entry: float = -math.inf
        self._lowest_since_entry: float = math.inf
        self._bar_count: int = 0

    # ------------------------------------------------------------------
    # IStrategy interface
    # ------------------------------------------------------------------
    def on_bar(self, bar: Bar, ctx: Any) -> StrategyDecision:  # noqa: C901
        self._bar_count += 1
        ohlcv = ctx.get_ohlcv()
        if ohlcv is None:
            return StrategyDecision.no_action()

        highs = np.array(ohlcv["high"], dtype=float)
        lows = np.array(ohlcv["low"], dtype=float)
        closes = np.array(ohlcv["close"], dtype=float)

        min_len = max(self.dc_period, self.atr_period, 2) + 1  # +1: channel excludes current bar
        if len(closes) < min_len:
            return StrategyDecision.no_action()

        # ---- indicators ----
        # Donchian channel is computed on the PREVIOUS dc_period bars
        # (excluding the current bar) so that a breakout is detectable.
        upper = float(np.max(highs[-self.dc_period - 1:-1]))
        lower = float(np.min(lows[-self.dc_period - 1:-1]))
        atr = self._compute_atr(highs, lows, closes, self.atr_period)

        ema_val: Optional[float] = None
        adx_val: Optional[float] = None
        filter_pass_long = True
        filter_pass_short = True

        if self.filter_type == "ema" and len(closes) >= self.ema_period:
            ema_val = float(self._ema(closes, self.ema_period))
            filter_pass_long = closes[-1] > ema_val
            filter_pass_short = closes[-1] < ema_val
        elif self.filter_type == "adx":
            adx_val = self._compute_adx(highs, lows, closes, self.adx_period)
            filter_pass_long = adx_val is not None and adx_val > self.adx_threshold
            filter_pass_short = filter_pass_long  # ADX doesn't have direction

        # ---- sizing (vol target) ----
        stop_distance = self.atr_mult * atr if atr > 0 else 1e-9
        equity = getattr(ctx, "equity", 10000.0) or 10000.0
        risk_amount = equity * self.risk_pct
        qty = risk_amount / stop_distance if stop_distance > 0 else self.position_size

        # ---- position info ----
        position = getattr(ctx, "position", 0.0) or 0.0
        close = float(closes[-1])

        orders: List[Order] = []
        features: Dict[str, Any] = {
            "upper": upper,
            "lower": lower,
            "atr": atr,
            "ema": ema_val,
            "adx": adx_val,
            "stop_distance": stop_distance,
            "vol_target_qty": qty,
        }

        # ---- trailing stop for open position ----
        if position > 1e-10:
            # Long
            self._highest_since_entry = max(self._highest_since_entry, float(bar.high))
            stop_price = self._highest_since_entry - self.atr_mult * atr
            features["stop_price"] = stop_price
            if bar.low <= stop_price:
                orders.append(self._exit_order(bar, position))
                return self._decision(orders, features, {"exit_reason": "trailing_stop_long"})
        elif position < -1e-10:
            # Short
            self._lowest_since_entry = min(self._lowest_since_entry, float(bar.low))
            stop_price = self._lowest_since_entry + self.atr_mult * atr
            features["stop_price"] = stop_price
            if bar.high >= stop_price:
                orders.append(self._exit_order(bar, position))
                return self._decision(orders, features, {"exit_reason": "trailing_stop_short"})

        # ---- time stop ----
        if self.max_holding_bars is not None and abs(position) > 1e-10:
            bars_held = self._bar_count - self._entry_bar_index
            if bars_held >= self.max_holding_bars:
                orders.append(self._exit_order(bar, position))
                return self._decision(orders, features, {"exit_reason": "time_stop"})

        # ---- entry signals ----
        if close > upper and filter_pass_long and position <= 1e-10:
            # Breakout long
            if position < -1e-10 and self.allow_reversal:
                orders.append(self._exit_order(bar, position))
            orders.append(Order(
                symbol=bar.symbol,
                side=OrderSide.BUY,
                order_type=OrderType.MARKET,
                quantity=qty,
                timestamp_ns=bar.timestamp_ns,
                strategy_id="DONCHIAN_ENTRY",
                metadata={"stop_price": close - self.atr_mult * atr,
                          "atr": atr, "donchian_upper": upper,
                          "donchian_lower": lower, "qty_method": "vol_target"},
            ))
            self._register_entry(close)
            return self._decision(orders, features, {"entry_side": "long"})

        if close < lower and filter_pass_short and position >= -1e-10:
            # Breakout short
            if position > 1e-10 and self.allow_reversal:
                orders.append(self._exit_order(bar, position))
            orders.append(Order(
                symbol=bar.symbol,
                side=OrderSide.SELL,
                order_type=OrderType.MARKET,
                quantity=qty,
                timestamp_ns=bar.timestamp_ns,
                strategy_id="DONCHIAN_ENTRY",
                metadata={"stop_price": close + self.atr_mult * atr,
                          "atr": atr, "donchian_upper": upper,
                          "donchian_lower": lower, "qty_method": "vol_target"},
            ))
            self._register_entry(close)
            return self._decision(orders, features, {"entry_side": "short"})

        return self._decision([], features, {})

    def reset(self) -> None:
        self._entry_bar_index = 0
        self._entry_price = 0.0
        self._highest_since_entry = -math.inf
        self._lowest_since_entry = math.inf
        self._bar_count = 0

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------
    def _register_entry(self, price: float) -> None:
        self._entry_bar_index = self._bar_count
        self._entry_price = price
        self._highest_since_entry = price
        self._lowest_since_entry = price

    @staticmethod
    def _exit_order(bar: Bar, signed_position: float) -> Order:
        """Create exit order. Pass SIGNED position (+ for long, - for short)."""
        # Long position → SELL to close; Short position → BUY to close
        side = OrderSide.SELL if signed_position > 0 else OrderSide.BUY
        return Order(
            symbol=bar.symbol,
            side=side,
            order_type=OrderType.MARKET,
            quantity=abs(signed_position),
            timestamp_ns=bar.timestamp_ns,
            strategy_id="DONCHIAN_EXIT",
            reduce_only=True,
        )

    @staticmethod
    def _decision(orders, features, metadata) -> StrategyDecision:
        return StrategyDecision(orders=orders, features=features, metadata=metadata)

    # ------------------------------------------------------------------
    # Indicator calculations (pure numpy, no talib dependency)
    # ------------------------------------------------------------------
    @staticmethod
    def _compute_atr(highs: np.ndarray, lows: np.ndarray, closes: np.ndarray,
                     period: int) -> float:
        """Simple ATR (Wilder-smoothed true range)."""
        if len(closes) < 2:
            return 0.0
        tr = np.maximum(
            highs[1:] - lows[1:],
            np.maximum(
                np.abs(highs[1:] - closes[:-1]),
                np.abs(lows[1:] - closes[:-1]),
            ),
        )
        if len(tr) < period:
            return float(np.mean(tr)) if len(tr) > 0 else 0.0
        # Wilder smoothing
        atr = float(np.mean(tr[:period]))
        for i in range(period, len(tr)):
            atr = (atr * (period - 1) + float(tr[i])) / period
        return atr

    @staticmethod
    def _ema(arr: np.ndarray, period: int) -> float:
        """Exponential moving average (last value)."""
        if len(arr) < period:
            return float(np.mean(arr))
        alpha = 2.0 / (period + 1)
        ema = float(arr[0])
        for v in arr[1:]:
            ema = alpha * float(v) + (1 - alpha) * ema
        return ema

    @staticmethod
    def _compute_adx(highs: np.ndarray, lows: np.ndarray, closes: np.ndarray,
                     period: int) -> Optional[float]:
        """Average Directional Index (simplified)."""
        n = len(closes)
        if n < period + 1:
            return None
        up = highs[1:] - highs[:-1]
        down = lows[:-1] - lows[1:]
        plus_dm = np.where((up > down) & (up > 0), up, 0.0)
        minus_dm = np.where((down > up) & (down > 0), down, 0.0)

        tr = np.maximum(
            highs[1:] - lows[1:],
            np.maximum(np.abs(highs[1:] - closes[:-1]), np.abs(lows[1:] - closes[:-1])),
        )
        # Wilder smoothing
        atr_sum = float(np.sum(tr[:period]))
        plus_sum = float(np.sum(plus_dm[:period]))
        minus_sum = float(np.sum(minus_dm[:period]))

        dx_vals = []
        for i in range(period, len(tr)):
            atr_sum = atr_sum - atr_sum / period + float(tr[i])
            plus_sum = plus_sum - plus_sum / period + float(plus_dm[i])
            minus_sum = minus_sum - minus_sum / period + float(minus_dm[i])
            if atr_sum > 0:
                plus_di = 100 * plus_sum / atr_sum
                minus_di = 100 * minus_sum / atr_sum
                denom = plus_di + minus_di
                dx = 100 * abs(plus_di - minus_di) / denom if denom > 0 else 0
                dx_vals.append(dx)

        if not dx_vals:
            return None
        # ADX = smoothed mean of DX
        adx = sum(dx_vals[:period]) / min(len(dx_vals), period)
        for v in dx_vals[period:]:
            adx = (adx * (period - 1) + v) / period
        return adx
