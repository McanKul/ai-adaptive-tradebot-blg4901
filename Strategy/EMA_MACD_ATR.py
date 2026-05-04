"""
Strategy/EMA_MACD_ATR.py
=========================
Trend-Following Strategy — EMA (50/200) + MACD + ATR Trailing Stop.

Combines three indicators:
- **EMA 50/200 crossover** for trend direction (filter)
- **MACD crossover** for entry timing
- **ATR trailing stop** for dynamic exit management

Entry Logic
-----------
- LONG:  EMA_fast > EMA_slow (uptrend) AND MACD line crosses above signal line
- SHORT: EMA_fast < EMA_slow (downtrend) AND MACD line crosses below signal line

Exit Logic
----------
- ATR trailing stop: tracks highest (long) / lowest (short) since entry.
  Stop = best_price ∓ atr_mult × ATR.
- Optional time stop (max_holding_bars).

Sizing
------
Volatility-targeted: qty = (risk_pct × equity) / (atr_mult × ATR).
The engine's sizing_config may override this.

Parameters
----------
ema_fast : int       Fast EMA period (default 50).
ema_slow : int       Slow EMA period (default 200).
macd_fast : int      MACD fast EMA period (default 12).
macd_slow : int      MACD slow EMA period (default 26).
macd_signal : int    MACD signal EMA period (default 9).
atr_period : int     ATR calculation period (default 14).
atr_mult : float     ATR multiplier for trailing stop (default 2.0).
risk_pct : float     Equity fraction risked per trade (default 0.005 = 0.5%).
position_size : float  Fallback position size if sizing cannot be computed.
allow_reversal : bool  If True, opposite signal triggers exit+reverse.
max_holding_bars : int  Optional time stop.

Integration
-----------
Implements ``IStrategy.on_bar(bar, ctx)`` → ``StrategyDecision``.
Uses ``ctx.get_ohlcv()`` for indicator computation and ``ctx.position``
for state tracking.
"""
from __future__ import annotations

import math
from typing import Optional, List, Dict, Any

import numpy as np

from Interfaces.IStrategy import IStrategy, StrategyDecision
from Interfaces.orders import Order, OrderType, OrderSide
from Interfaces.market_data import Bar


class Strategy(IStrategy):
    """EMA (50/200) + MACD + ATR Trailing Stop trend-following strategy."""

    def __init__(
        self,
        # EMA trend filter
        ema_fast: int = 50,
        ema_slow: int = 200,
        # MACD entry timing
        macd_fast: int = 12,
        macd_slow: int = 26,
        macd_signal: int = 9,
        # ATR exit
        atr_period: int = 14,
        atr_mult: float = 2.0,
        # Sizing
        risk_pct: float = 0.005,
        position_size: float = 1.0,
        sizing_config=None,
        # Exit params (for ExitManager / tick-level TP/SL)
        take_profit_pct: Optional[float] = None,
        stop_loss_pct: Optional[float] = None,
        # Behaviour
        allow_reversal: bool = True,
        max_holding_bars: Optional[int] = None,
        leverage: float = 1.0,
        **kw,
    ):
        self.ema_fast = ema_fast
        self.ema_slow = ema_slow
        self.macd_fast = macd_fast
        self.macd_slow = macd_slow
        self.macd_signal = macd_signal
        self.atr_period = atr_period
        self.atr_mult = atr_mult
        self.risk_pct = risk_pct
        self.position_size = position_size
        self.sizing_config = sizing_config
        self.allow_reversal = allow_reversal
        self.max_holding_bars = max_holding_bars
        self.leverage = leverage

        # Exit params for get_exit_params()
        self._exit_params: Dict[str, Any] = {}
        if take_profit_pct is not None:
            self._exit_params["tp_pct"] = take_profit_pct
        if stop_loss_pct is not None:
            self._exit_params["sl_pct"] = stop_loss_pct

        # Setup exit manager if TP/SL provided
        self.exit_manager = None
        if take_profit_pct or stop_loss_pct:
            try:
                from Backtest.exit_manager import ExitManager, ExitConfig
                exit_config = ExitConfig(
                    take_profit_pct=take_profit_pct,
                    stop_loss_pct=stop_loss_pct,
                    leverage=leverage,
                )
                self.exit_manager = ExitManager(exit_config)
            except ImportError:
                pass

        # Internal state
        self._entry_bar_index: int = 0
        self._entry_price: float = 0.0
        self._highest_since_entry: float = -math.inf
        self._lowest_since_entry: float = math.inf
        self._bar_count: int = 0
        self._prev_macd_hist: Optional[float] = None  # for crossover detection

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

        # Need enough data for the slowest indicator (EMA slow period)
        min_len = max(self.ema_slow, self.macd_slow + self.macd_signal, self.atr_period) + 2
        if len(closes) < min_len:
            return StrategyDecision.no_action()

        # ---- Compute indicators ----
        ema_fast_val = self._ema(closes, self.ema_fast)
        ema_slow_val = self._ema(closes, self.ema_slow)
        macd_line, signal_line, histogram = self._compute_macd(
            closes, self.macd_fast, self.macd_slow, self.macd_signal
        )
        atr = self._compute_atr(highs, lows, closes, self.atr_period)

        # ---- Trend filter ----
        uptrend = ema_fast_val > ema_slow_val
        downtrend = ema_fast_val < ema_slow_val

        # ---- MACD crossover detection ----
        # Crossover = histogram sign change from previous bar
        macd_cross_up = False
        macd_cross_down = False
        if self._prev_macd_hist is not None:
            macd_cross_up = self._prev_macd_hist <= 0 and histogram > 0
            macd_cross_down = self._prev_macd_hist >= 0 and histogram < 0
        self._prev_macd_hist = histogram

        # ---- Sizing (vol target) ----
        stop_distance = self.atr_mult * atr if atr > 0 else 1e-9
        equity = getattr(ctx, "equity", 10000.0) or 10000.0
        risk_amount = equity * self.risk_pct
        qty = risk_amount / stop_distance if stop_distance > 0 else self.position_size

        # ---- Position info ----
        position = getattr(ctx, "position", 0.0) or 0.0
        close = float(closes[-1])

        orders: List[Order] = []
        features: Dict[str, Any] = {
            "ema_fast": ema_fast_val,
            "ema_slow": ema_slow_val,
            "macd_line": macd_line,
            "signal_line": signal_line,
            "macd_histogram": histogram,
            "atr": atr,
            "uptrend": uptrend,
            "downtrend": downtrend,
            "macd_cross_up": macd_cross_up,
            "macd_cross_down": macd_cross_down,
            "stop_distance": stop_distance,
            "vol_target_qty": qty,
        }

        # ---- Trailing stop for open position ----
        if position > 1e-10:
            # Long position — trail from highest
            self._highest_since_entry = max(self._highest_since_entry, float(bar.high))
            stop_price = self._highest_since_entry - self.atr_mult * atr
            features["stop_price"] = stop_price
            if bar.low <= stop_price:
                orders.append(self._exit_order(bar, abs(position)))
                return self._decision(orders, features, {"exit_reason": "trailing_stop_long"})
        elif position < -1e-10:
            # Short position — trail from lowest
            self._lowest_since_entry = min(self._lowest_since_entry, float(bar.low))
            stop_price = self._lowest_since_entry + self.atr_mult * atr
            features["stop_price"] = stop_price
            if bar.high >= stop_price:
                orders.append(self._exit_order(bar, abs(position)))
                return self._decision(orders, features, {"exit_reason": "trailing_stop_short"})

        # ---- Time stop ----
        if self.max_holding_bars is not None and abs(position) > 1e-10:
            bars_held = self._bar_count - self._entry_bar_index
            if bars_held >= self.max_holding_bars:
                orders.append(self._exit_order(bar, abs(position)))
                return self._decision(orders, features, {"exit_reason": "time_stop"})

        # ---- Entry signals ----
        # LONG: uptrend (EMA50 > EMA200) + MACD crosses above signal
        if uptrend and macd_cross_up and position <= 1e-10:
            if position < -1e-10 and self.allow_reversal:
                orders.append(self._exit_order(bar, abs(position)))
            orders.append(Order(
                symbol=bar.symbol,
                side=OrderSide.BUY,
                order_type=OrderType.MARKET,
                quantity=qty,
                timestamp_ns=bar.timestamp_ns,
                strategy_id="EMA_MACD_ENTRY",
                metadata={
                    "stop_price": close - self.atr_mult * atr,
                    "atr": atr,
                    "ema_fast": ema_fast_val,
                    "ema_slow": ema_slow_val,
                    "macd_histogram": histogram,
                    "qty_method": "vol_target",
                    **self._exit_params,
                },
            ))
            self._register_entry(close)
            return self._decision(orders, features, {"entry_side": "long"})

        # SHORT: downtrend (EMA50 < EMA200) + MACD crosses below signal
        if downtrend and macd_cross_down and position >= -1e-10:
            if position > 1e-10 and self.allow_reversal:
                orders.append(self._exit_order(bar, abs(position)))
            orders.append(Order(
                symbol=bar.symbol,
                side=OrderSide.SELL,
                order_type=OrderType.MARKET,
                quantity=qty,
                timestamp_ns=bar.timestamp_ns,
                strategy_id="EMA_MACD_ENTRY",
                metadata={
                    "stop_price": close + self.atr_mult * atr,
                    "atr": atr,
                    "ema_fast": ema_fast_val,
                    "ema_slow": ema_slow_val,
                    "macd_histogram": histogram,
                    "qty_method": "vol_target",
                    **self._exit_params,
                },
            ))
            self._register_entry(close)
            return self._decision(orders, features, {"entry_side": "short"})

        return self._decision([], features, {})

    def reset(self) -> None:
        """Reset strategy state for a new run."""
        self._entry_bar_index = 0
        self._entry_price = 0.0
        self._highest_since_entry = -math.inf
        self._lowest_since_entry = math.inf
        self._bar_count = 0
        self._prev_macd_hist = None
        if self.exit_manager:
            self.exit_manager.reset()

    def get_exit_params(self) -> Dict[str, Any]:
        """Return exit parameters for tick-level TP/SL checking."""
        return {k: v for k, v in self._exit_params.items() if v is not None}

    def get_exit_manager(self):
        """Return the exit manager if configured."""
        return self.exit_manager

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------
    def _register_entry(self, price: float) -> None:
        self._entry_bar_index = self._bar_count
        self._entry_price = price
        self._highest_since_entry = price
        self._lowest_since_entry = price

    @staticmethod
    def _exit_order(bar: Bar, qty: float) -> Order:
        side = OrderSide.SELL if qty > 0 else OrderSide.BUY
        return Order(
            symbol=bar.symbol,
            side=side,
            order_type=OrderType.MARKET,
            quantity=abs(qty),
            timestamp_ns=bar.timestamp_ns,
            strategy_id="EMA_MACD_EXIT",
            reduce_only=True,
        )

    @staticmethod
    def _decision(orders, features, metadata) -> StrategyDecision:
        return StrategyDecision(orders=orders, features=features, metadata=metadata)

    # ------------------------------------------------------------------
    # Indicator calculations (pure numpy, no TA-Lib dependency)
    # ------------------------------------------------------------------
    @staticmethod
    def _ema(arr: np.ndarray, period: int) -> float:
        """Exponential moving average — returns the last value."""
        if len(arr) < period:
            return float(np.mean(arr))
        alpha = 2.0 / (period + 1)
        ema = float(arr[0])
        for v in arr[1:]:
            ema = alpha * float(v) + (1 - alpha) * ema
        return ema

    @staticmethod
    def _ema_series(arr: np.ndarray, period: int) -> np.ndarray:
        """Exponential moving average — returns full series."""
        out = np.empty_like(arr, dtype=float)
        alpha = 2.0 / (period + 1)
        out[0] = float(arr[0])
        for i in range(1, len(arr)):
            out[i] = alpha * float(arr[i]) + (1 - alpha) * out[i - 1]
        return out

    @staticmethod
    def _compute_macd(
        closes: np.ndarray,
        fast_period: int = 12,
        slow_period: int = 26,
        signal_period: int = 9,
    ) -> tuple[float, float, float]:
        """
        Compute MACD line, signal line, and histogram (last values).

        MACD line   = EMA(fast) - EMA(slow)
        Signal line = EMA(MACD line, signal_period)
        Histogram   = MACD line - Signal line
        """
        ema_fast = Strategy._ema_series(closes, fast_period)
        ema_slow = Strategy._ema_series(closes, slow_period)
        macd_line = ema_fast - ema_slow

        # Signal line is EMA of the MACD line
        signal_line = Strategy._ema_series(macd_line, signal_period)
        histogram = macd_line - signal_line

        return float(macd_line[-1]), float(signal_line[-1]), float(histogram[-1])

    @staticmethod
    def _compute_atr(
        highs: np.ndarray,
        lows: np.ndarray,
        closes: np.ndarray,
        period: int,
    ) -> float:
        """Wilder-smoothed Average True Range."""
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
