"""
Strategy/regime/atr_vol_regime.py
=================================
ATR-percentile volatility regime detector.

Tags the current bar's ATR against its trailing percentile rank
inside ``window`` bars:

* ``vol_high``  if ATR rank ≥ ``hi_pct`` (e.g. top 20%)
* ``vol_low``   if ATR rank ≤ ``lo_pct`` (e.g. bottom 20%)
* ``vol_mid``   otherwise

Empty tags during warmup (history < ``window``).
"""
from __future__ import annotations
from typing import Any, List, TYPE_CHECKING

import numpy as np

try:
    import talib  # type: ignore
    _TALIB_OK = True
except ImportError:  # pragma: no cover
    _TALIB_OK = False

from Interfaces.IRegimeDetector import IRegimeDetector, RegimeState

if TYPE_CHECKING:
    from Interfaces.market_data import Bar


class ATRPercentileRegime(IRegimeDetector):
    """Volatility regime via ATR percentile rank.

    Args:
        atr_period: ATR lookback (default 14).
        window: Trailing window for percentile rank (default 200).
        hi_pct: Upper boundary as fraction in [0, 1] (default 0.8).
        lo_pct: Lower boundary as fraction in [0, 1] (default 0.2).
    """

    def __init__(
        self,
        atr_period: int = 14,
        window: int = 200,
        hi_pct: float = 0.8,
        lo_pct: float = 0.2,
    ):
        if not 0.0 < lo_pct < hi_pct < 1.0:
            raise ValueError(
                f"need 0 < lo_pct < hi_pct < 1, got lo={lo_pct} hi={hi_pct}"
            )
        if atr_period < 2:
            raise ValueError("atr_period must be >= 2")
        if window < atr_period * 2:
            raise ValueError("window must be >= 2 * atr_period")
        self.atr_period = atr_period
        self.window = window
        self.hi_pct = float(hi_pct)
        self.lo_pct = float(lo_pct)

    # ------------------------------------------------------------------
    def detect(self, bar: "Bar", ctx: Any) -> RegimeState:
        if not (hasattr(ctx, "get_ohlcv") and callable(ctx.get_ohlcv)):
            return RegimeState()
        ohlcv = ctx.get_ohlcv(limit=self.window + self.atr_period * 2)
        if not ohlcv:
            return RegimeState()
        try:
            h = np.asarray(ohlcv["high"], dtype=np.float64)
            l = np.asarray(ohlcv["low"], dtype=np.float64)
            c = np.asarray(ohlcv["close"], dtype=np.float64)
        except (KeyError, TypeError):
            return RegimeState()

        if len(c) < self.window:
            return RegimeState()

        atr = self._compute_atr(h, l, c, self.atr_period)
        if atr is None:
            return RegimeState()
        # Drop warmup NaNs and take the last `window`
        atr_clean = atr[~np.isnan(atr)]
        if len(atr_clean) < self.window:
            return RegimeState()
        recent = atr_clean[-self.window:]
        current = float(recent[-1])
        if current <= 0:
            return RegimeState()

        # Percentile rank of `current` within `recent`
        rank = float((recent <= current).mean())

        if rank >= self.hi_pct:
            tags = ["vol_high"]
        elif rank <= self.lo_pct:
            tags = ["vol_low"]
        else:
            tags = ["vol_mid"]

        return RegimeState(
            tags=tags,
            score={"atr": current, "atr_rank": rank},
        )

    # ------------------------------------------------------------------
    @staticmethod
    def _compute_atr(h, l, c, period):
        if _TALIB_OK:
            try:
                return talib.ATR(h, l, c, timeperiod=period)
            except Exception:  # pragma: no cover
                pass
        # Fallback: Wilder's ATR
        if len(c) < period + 1:
            return None
        prev_close = np.roll(c, 1)
        prev_close[0] = c[0]
        tr = np.maximum.reduce([h - l, np.abs(h - prev_close), np.abs(l - prev_close)])
        atr = np.full_like(tr, np.nan, dtype=np.float64)
        atr[period - 1] = float(np.mean(tr[:period]))
        for i in range(period, len(tr)):
            atr[i] = (atr[i - 1] * (period - 1) + tr[i]) / period
        return atr
