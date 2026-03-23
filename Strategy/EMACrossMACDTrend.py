"""
Strategy/EMACrossMACDTrend.py
==============================
EMA Crossover  +  MACD Histogram Momentum  +  ADX Trend Filter
+  ATR Trailing Stop  +  Volume Confirmation.

Tarihsel olarak en çok kanıtlanmış trend-following bileşenlerini
tek bir tutarlı sistemde birleştirir.

Giriş Kuralları
----------------
LONG:
  1. Fast EMA > Slow EMA  (trend yönü)
  2. MACD histogram > 0 ve yükseliyor  (momentum teyidi)
  3. ADX > adx_threshold  (trend gücü filtresi)
  4. Volume > volume SMA  (hacim teyidi, opsiyonel)
  5. Pozisyon flat veya short ise → long aç

SHORT:
  Yukarının tersi.

Çıkış Kuralları
-----------------
  - ATR-tabanlı trailing stop  (Chandelier Exit)
  - MACD histogram ters yöne dönerse → pozisyon kapat
  - Opsiyonel zaman stopu  (max_holding_bars)

Sizing
------
Volatilite-hedefli: qty = (equity × risk_pct) / (atr_mult × ATR).
Engine seviyesinde sizing_config ile override edilebilir.

Parameters
----------
fast_ema_period : int      Hızlı EMA periyodu (default 12).
slow_ema_period : int      Yavaş EMA periyodu (default 26).
macd_signal     : int      MACD sinyal hattı periyodu (default 9).
atr_period      : int      ATR periyodu (default 14).
atr_mult        : float    ATR çarpanı trailing stop için (default 2.5).
adx_period      : int      ADX periyodu (default 14).
adx_threshold   : float    Minimum ADX trend gücü (default 20).
volume_filter   : bool     Hacim filtresi aktif mi (default True).
volume_period   : int      Hacim SMA periyodu (default 20).
risk_pct        : float    Equity oranı risk (default 0.005 = %0.5).
position_size   : float    Fallback pozisyon büyüklüğü (default 1.0).
allow_reversal  : bool     Ters pozisyona geçişe izin ver (default True).
exit_on_macd_cross : bool  MACD histogram ters dönüşte çık (default True).
max_holding_bars : int     Opsiyonel zaman stopu.
"""
from __future__ import annotations

import math
from typing import Optional, List, Dict, Any

import numpy as np

from Interfaces.IStrategy import IStrategy, StrategyDecision
from Interfaces.orders import Order, OrderType, OrderSide
from Interfaces.market_data import Bar


class Strategy(IStrategy):
    """EMA Cross + MACD Momentum + ADX Filter + ATR Trailing Stop."""

    def __init__(
        self,
        # EMA crossover
        fast_ema_period: int = 12,
        slow_ema_period: int = 26,
        # MACD
        macd_signal: int = 9,
        # ATR / exit
        atr_period: int = 14,
        atr_mult: float = 2.5,
        # ADX trend filter
        adx_period: int = 14,
        adx_threshold: float = 20.0,
        # Volume filter
        volume_filter: bool = True,
        volume_period: int = 20,
        # Sizing
        risk_pct: float = 0.005,
        position_size: float = 1.0,
        # Behaviour
        allow_reversal: bool = True,
        exit_on_macd_cross: bool = True,
        max_holding_bars: Optional[int] = None,
        # RSI filter
        rsi_period: int = 14,
        rsi_overbought: float = 75.0,
        rsi_oversold: float = 25.0,
        use_rsi_filter: bool = True,
        # EMA slope confirmation
        use_ema_slope: bool = True,
        # Minimum histogram threshold (ATR yüzdesi)
        min_histogram_pct: float = 0.0,
        # misc
        **kw,
    ):
        self.fast_ema_period = fast_ema_period
        self.slow_ema_period = slow_ema_period
        self.macd_signal = macd_signal
        self.atr_period = atr_period
        self.atr_mult = atr_mult
        self.adx_period = adx_period
        self.adx_threshold = adx_threshold
        self.volume_filter = volume_filter
        self.volume_period = volume_period
        self.risk_pct = risk_pct
        self.position_size = position_size
        self.allow_reversal = allow_reversal
        self.exit_on_macd_cross = exit_on_macd_cross
        self.max_holding_bars = max_holding_bars
        self.rsi_period = rsi_period
        self.rsi_overbought = rsi_overbought
        self.rsi_oversold = rsi_oversold
        self.use_rsi_filter = use_rsi_filter
        self.use_ema_slope = use_ema_slope
        self.min_histogram_pct = min_histogram_pct

        # Internal state — per-symbol to support multi-coin live trading
        self._state: Dict[str, Dict[str, Any]] = {}

    def _get_state(self, symbol: str) -> Dict[str, Any]:
        """Get or create per-symbol state."""
        if symbol not in self._state:
            self._state[symbol] = {
                "entry_bar_index": 0,
                "entry_price": 0.0,
                "highest_since_entry": -math.inf,
                "lowest_since_entry": math.inf,
                "bar_count": 0,
                "prev_histogram": 0.0,
            }
        return self._state[symbol]

    # ------------------------------------------------------------------
    # IStrategy interface
    # ------------------------------------------------------------------
    def on_bar(self, bar: Bar, ctx: Any) -> StrategyDecision:  # noqa: C901
        sym = bar.symbol
        st = self._get_state(sym)
        st["bar_count"] += 1
        ohlcv = ctx.get_ohlcv()
        if ohlcv is None:
            return StrategyDecision.no_action()

        highs = np.array(ohlcv["high"], dtype=float)
        lows = np.array(ohlcv["low"], dtype=float)
        closes = np.array(ohlcv["close"], dtype=float)
        volumes = np.array(ohlcv["volume"], dtype=float)

        min_len = max(self.slow_ema_period + self.macd_signal,
                      self.atr_period, self.adx_period) + 2
        if len(closes) < min_len:
            return StrategyDecision.no_action()

        # ---- indicators ----
        fast_ema = self._ema_series(closes, self.fast_ema_period)
        slow_ema = self._ema_series(closes, self.slow_ema_period)

        # MACD line = fast EMA - slow EMA
        macd_line = fast_ema - slow_ema
        signal_line = self._ema_series(macd_line, self.macd_signal)
        histogram = macd_line - signal_line

        curr_histogram = float(histogram[-1])
        prev_histogram = float(histogram[-2]) if len(histogram) >= 2 else 0.0

        curr_fast_ema = float(fast_ema[-1])
        curr_slow_ema = float(slow_ema[-1])

        # ATR
        atr = self._compute_atr(highs, lows, closes, self.atr_period)

        # ADX
        adx_val = self._compute_adx(highs, lows, closes, self.adx_period)
        adx_pass = adx_val is not None and adx_val > self.adx_threshold

        # Volume filter
        vol_pass = True
        vol_sma: Optional[float] = None
        if self.volume_filter and len(volumes) >= self.volume_period:
            vol_sma = float(np.mean(volumes[-self.volume_period:]))
            vol_pass = float(volumes[-1]) > vol_sma

        # RSI filter — overbought'ta long açma, oversold'da short açma
        rsi_val: Optional[float] = None
        rsi_pass_long = True
        rsi_pass_short = True
        if self.use_rsi_filter and len(closes) >= self.rsi_period + 1:
            rsi_val = self._compute_rsi(closes, self.rsi_period)
            rsi_pass_long = rsi_val < self.rsi_overbought
            rsi_pass_short = rsi_val > self.rsi_oversold

        # EMA slope confirmation — EMA'lar aktif olarak ayrışıyor mu?
        ema_slope_long = True
        ema_slope_short = True
        if self.use_ema_slope and len(fast_ema) >= 4:
            fast_slope = float(fast_ema[-1] - fast_ema[-4])
            slow_slope = float(slow_ema[-1] - slow_ema[-4])
            ema_slope_long = fast_slope > 0 and slow_slope > 0
            ema_slope_short = fast_slope < 0 and slow_slope < 0

        # ---- sizing ----
        # Placeholder qty — engine overrides via SizingConfig.
        # Vol-target qty stored in features for reference only.
        stop_distance = self.atr_mult * atr if atr > 0 else 1e-9
        equity = getattr(ctx, "equity", 10000.0) or 10000.0
        risk_amount = equity * self.risk_pct
        vol_target_qty = risk_amount / stop_distance if stop_distance > 0 else self.position_size
        qty = self.position_size  # placeholder, engine applies actual sizing

        # ---- position info ----
        position = getattr(ctx, "position", 0.0) or 0.0
        close = float(closes[-1])

        orders: List[Order] = []
        features: Dict[str, Any] = {
            "fast_ema": curr_fast_ema,
            "slow_ema": curr_slow_ema,
            "macd_line": float(macd_line[-1]),
            "signal_line": float(signal_line[-1]),
            "histogram": curr_histogram,
            "atr": atr,
            "adx": adx_val,
            "vol_sma": vol_sma,
            "rsi": rsi_val,
            "stop_distance": stop_distance,
            "vol_target_qty": vol_target_qty,
        }

        # ---- trailing stop for open position ----
        if position > 1e-10:
            st["highest_since_entry"] = max(st["highest_since_entry"], float(bar.high))
            stop_price = st["highest_since_entry"] - self.atr_mult * atr
            features["stop_price"] = stop_price
            if bar.low <= stop_price:
                orders.append(self._exit_order(bar, position))
                st["prev_histogram"] = curr_histogram
                return self._decision(orders, features,
                                      {"exit_reason": "trailing_stop_long"})
        elif position < -1e-10:
            st["lowest_since_entry"] = min(st["lowest_since_entry"], float(bar.low))
            stop_price = st["lowest_since_entry"] + self.atr_mult * atr
            features["stop_price"] = stop_price
            if bar.high >= stop_price:
                orders.append(self._exit_order(bar, position))
                st["prev_histogram"] = curr_histogram
                return self._decision(orders, features,
                                      {"exit_reason": "trailing_stop_short"})

        # ---- MACD histogram reversal exit ----
        if self.exit_on_macd_cross and abs(position) > 1e-10:
            if position > 1e-10 and curr_histogram < 0 and prev_histogram >= 0:
                orders.append(self._exit_order(bar, position))
                st["prev_histogram"] = curr_histogram
                return self._decision(orders, features,
                                      {"exit_reason": "macd_cross_exit_long"})
            if position < -1e-10 and curr_histogram > 0 and prev_histogram <= 0:
                orders.append(self._exit_order(bar, position))
                st["prev_histogram"] = curr_histogram
                return self._decision(orders, features,
                                      {"exit_reason": "macd_cross_exit_short"})

        # ---- time stop ----
        if self.max_holding_bars is not None and abs(position) > 1e-10:
            bars_held = st["bar_count"] - st["entry_bar_index"]
            if bars_held >= self.max_holding_bars:
                orders.append(self._exit_order(bar, position))
                st["prev_histogram"] = curr_histogram
                return self._decision(orders, features,
                                      {"exit_reason": "time_stop"})

        # ---- entry signals ----
        # Minimum histogram eşiği (ATR yüzdesi — whipsaw koruması)
        hist_threshold = self.min_histogram_pct * atr if atr > 0 else 0.0

        # LONG: fast EMA > slow EMA + MACD histogram > threshold & rising
        #       + ADX + volume + RSI not overbought + EMA slope up
        long_ema = curr_fast_ema > curr_slow_ema
        long_macd = curr_histogram > hist_threshold and curr_histogram > prev_histogram
        long_ok = (long_ema and long_macd and adx_pass and vol_pass
                   and rsi_pass_long and ema_slope_long)

        # SHORT: fast EMA < slow EMA + MACD histogram < -threshold & falling
        #        + ADX + volume + RSI not oversold + EMA slope down
        short_ema = curr_fast_ema < curr_slow_ema
        short_macd = curr_histogram < -hist_threshold and curr_histogram < prev_histogram
        short_ok = (short_ema and short_macd and adx_pass and vol_pass
                    and rsi_pass_short and ema_slope_short)

        # Debug: log entry condition breakdown periodically
        if st["bar_count"] % 50 == 0 or long_ok or short_ok:
            import logging
            _log = logging.getLogger("EMACross")
            _log.info(
                "[%s] ENTRY CHECK: fast_ema=%.4f slow_ema=%.4f | "
                "long_ema=%s long_macd=%s(hist=%.6f prev=%.6f) adx=%s(%.1f) vol=%s | "
                "short_ema=%s short_macd=%s | LONG=%s SHORT=%s | pos=%.4f",
                bar.symbol, curr_fast_ema, curr_slow_ema,
                long_ema, long_macd, curr_histogram, prev_histogram,
                adx_pass, adx_val or 0, vol_pass,
                short_ema, short_macd,
                long_ok, short_ok, position,
            )

        if long_ok and position <= 1e-10:
            if position < -1e-10 and self.allow_reversal:
                orders.append(self._exit_order(bar, position))
            orders.append(Order(
                symbol=bar.symbol,
                side=OrderSide.BUY,
                order_type=OrderType.MARKET,
                quantity=qty,
                timestamp_ns=bar.timestamp_ns,
                strategy_id="EMACROSS_ENTRY",
                metadata={
                    "stop_price": close - self.atr_mult * atr,
                    "atr": atr,
                    "fast_ema": curr_fast_ema,
                    "slow_ema": curr_slow_ema,
                    "histogram": curr_histogram,
                    "adx": adx_val,
                    "qty_method": "vol_target",
                },
            ))
            self._register_entry(st, close)
            st["prev_histogram"] = curr_histogram
            return self._decision(orders, features, {"entry_side": "long"})

        if short_ok and position >= -1e-10:
            if position > 1e-10 and self.allow_reversal:
                orders.append(self._exit_order(bar, position))
            orders.append(Order(
                symbol=bar.symbol,
                side=OrderSide.SELL,
                order_type=OrderType.MARKET,
                quantity=qty,
                timestamp_ns=bar.timestamp_ns,
                strategy_id="EMACROSS_ENTRY",
                metadata={
                    "stop_price": close + self.atr_mult * atr,
                    "atr": atr,
                    "fast_ema": curr_fast_ema,
                    "slow_ema": curr_slow_ema,
                    "histogram": curr_histogram,
                    "adx": adx_val,
                    "qty_method": "vol_target",
                },
            ))
            self._register_entry(st, close)
            st["prev_histogram"] = curr_histogram
            return self._decision(orders, features, {"entry_side": "short"})

        st["prev_histogram"] = curr_histogram
        return self._decision([], features, {})

    def reset(self) -> None:
        self._state.clear()

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------
    def _register_entry(self, st: Dict[str, Any], price: float) -> None:
        st["entry_bar_index"] = st["bar_count"]
        st["entry_price"] = price
        st["highest_since_entry"] = price
        st["lowest_since_entry"] = price

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
            strategy_id="EMACROSS_EXIT",
            reduce_only=True,
        )

    @staticmethod
    def _decision(orders, features, metadata) -> StrategyDecision:
        return StrategyDecision(orders=orders, features=features, metadata=metadata)

    # ------------------------------------------------------------------
    # Indicator calculations (pure numpy, no talib dependency)
    # ------------------------------------------------------------------
    @staticmethod
    def _ema_series(arr: np.ndarray, period: int) -> np.ndarray:
        """Full EMA series (returns array same length as input)."""
        out = np.empty_like(arr, dtype=float)
        alpha = 2.0 / (period + 1)
        out[0] = float(arr[0])
        for i in range(1, len(arr)):
            out[i] = alpha * float(arr[i]) + (1.0 - alpha) * out[i - 1]
        return out

    @staticmethod
    def _compute_rsi(closes: np.ndarray, period: int) -> float:
        """RSI (Relative Strength Index)."""
        deltas = np.diff(closes[-(period + 1):])
        gains = np.where(deltas > 0, deltas, 0.0)
        losses = np.where(deltas < 0, -deltas, 0.0)
        avg_gain = float(np.mean(gains))
        avg_loss = float(np.mean(losses))
        if avg_loss == 0:
            return 100.0
        rs = avg_gain / avg_loss
        return 100.0 - (100.0 / (1.0 + rs))

    @staticmethod
    def _compute_atr(highs: np.ndarray, lows: np.ndarray, closes: np.ndarray,
                     period: int) -> float:
        """ATR (Wilder-smoothed true range)."""
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
        atr = float(np.mean(tr[:period]))
        for i in range(period, len(tr)):
            atr = (atr * (period - 1) + float(tr[i])) / period
        return atr

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
            np.maximum(np.abs(highs[1:] - closes[:-1]),
                       np.abs(lows[1:] - closes[:-1])),
        )
        atr_sum = float(np.sum(tr[:period]))
        plus_sum = float(np.sum(plus_dm[:period]))
        minus_sum = float(np.sum(minus_dm[:period]))

        dx_vals = []
        for i in range(period, len(tr)):
            atr_sum = atr_sum - atr_sum / period + float(tr[i])
            plus_sum = plus_sum - plus_sum / period + float(plus_dm[i])
            minus_sum = minus_sum - minus_sum / period + float(minus_dm[i])
            if atr_sum > 0:
                plus_di = 100.0 * plus_sum / atr_sum
                minus_di = 100.0 * minus_sum / atr_sum
                denom = plus_di + minus_di
                dx = 100.0 * abs(plus_di - minus_di) / denom if denom > 0 else 0.0
                dx_vals.append(dx)

        if not dx_vals:
            return None
        adx = sum(dx_vals[:period]) / min(len(dx_vals), period)
        for v in dx_vals[period:]:
            adx = (adx * (period - 1) + v) / period
        return adx
