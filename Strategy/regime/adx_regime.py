"""
Strategy/regime/adx_regime.py
=============================
ADX-based regime detector.

ADX measures trend strength (regardless of direction); +DI / −DI
tell the side.  We tag:

* ``trend_up``    when ADX ≥ threshold AND +DI − −DI ≥ di_gap
* ``trend_down``  when ADX ≥ threshold AND −DI − +DI ≥ di_gap
* ``range``       when ADX < threshold

Returns empty tags during warmup (history < period * 2 — TA-Lib
requires the longer warmup).

Reuses TA-Lib's ADX / PLUS_DI / MINUS_DI which are already a project
dependency.
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


class ADXRegimeDetector(IRegimeDetector):
    """Trending-vs-ranging regime detector.

    Args:
        period: ADX lookback (default 14).
        trend_threshold: ADX value above which we consider the market
            "trending" (default 25).  Wilder's classic threshold.
        di_gap: Minimum ``|+DI − −DI|`` to award a directional tag.
            Set to 0 to drop the gap requirement.
    """

    def __init__(
        self,
        period: int = 14,
        trend_threshold: float = 25.0,
        di_gap: float = 5.0,
    ):
        if period < 2:
            raise ValueError(f"ADX period must be >= 2, got {period}")
        self.period = period
        self.trend_threshold = float(trend_threshold)
        self.di_gap = float(di_gap)
        # TA-Lib ADX needs ~2*period bars of warmup.
        self._min_warmup = period * 2 + 1

    # ------------------------------------------------------------------
    def detect(self, bar: "Bar", ctx: Any) -> RegimeState:
        ohlcv = self._extract_ohlcv(ctx)
        if ohlcv is None:
            return RegimeState()
        h, l, c = ohlcv["high"], ohlcv["low"], ohlcv["close"]
        if len(c) < self._min_warmup:
            return RegimeState()

        try:
            if _TALIB_OK:
                adx = talib.ADX(h, l, c, timeperiod=self.period)
                plus_di = talib.PLUS_DI(h, l, c, timeperiod=self.period)
                minus_di = talib.MINUS_DI(h, l, c, timeperiod=self.period)
            else:
                adx, plus_di, minus_di = self._fallback_adx(h, l, c)
        except Exception:  # pragma: no cover — defensive
            return RegimeState()

        adx_v = float(adx[-1]) if not np.isnan(adx[-1]) else float("nan")
        pdi_v = float(plus_di[-1]) if not np.isnan(plus_di[-1]) else float("nan")
        ndi_v = float(minus_di[-1]) if not np.isnan(minus_di[-1]) else float("nan")

        if any(np.isnan(x) for x in (adx_v, pdi_v, ndi_v)):
            return RegimeState()

        tags: List[str] = []
        score = {"adx": adx_v, "plus_di": pdi_v, "minus_di": ndi_v}

        if adx_v >= self.trend_threshold:
            gap = pdi_v - ndi_v
            if gap >= self.di_gap:
                tags.append("trend_up")
            elif -gap >= self.di_gap:
                tags.append("trend_down")
            else:
                tags.append("trend")  # strong but undirected
        else:
            tags.append("range")

        return RegimeState(tags=tags, score=score)

    # ------------------------------------------------------------------
    @staticmethod
    def _extract_ohlcv(ctx: Any):
        if hasattr(ctx, "get_ohlcv") and callable(ctx.get_ohlcv):
            ohlcv = ctx.get_ohlcv(limit=500)
            if not ohlcv:
                return None
            try:
                return {
                    "high": np.asarray(ohlcv["high"], dtype=np.float64),
                    "low": np.asarray(ohlcv["low"], dtype=np.float64),
                    "close": np.asarray(ohlcv["close"], dtype=np.float64),
                }
            except (KeyError, TypeError):
                return None
        return None

    @staticmethod
    def _fallback_adx(high, low, close):  # pragma: no cover — exercised only when TA-Lib missing
        """Pure-numpy ADX (Wilder).  Slower but no external dep."""
        n = 14
        h, l, c = high, low, close
        up = np.diff(h, prepend=h[0])
        dn = -np.diff(l, prepend=l[0])
        plus_dm = np.where((up > dn) & (up > 0), up, 0.0)
        minus_dm = np.where((dn > up) & (dn > 0), dn, 0.0)
        tr = np.maximum.reduce([
            h - l,
            np.abs(h - np.roll(c, 1)),
            np.abs(l - np.roll(c, 1)),
        ])
        tr[0] = h[0] - l[0]

        def _wilder_smooth(x, p):
            out = np.full_like(x, np.nan, dtype=np.float64)
            if len(x) < p:
                return out
            out[p - 1] = np.sum(x[:p])
            for i in range(p, len(x)):
                out[i] = out[i - 1] - out[i - 1] / p + x[i]
            return out

        atr = _wilder_smooth(tr, n)
        pdm_s = _wilder_smooth(plus_dm, n)
        ndm_s = _wilder_smooth(minus_dm, n)
        with np.errstate(divide="ignore", invalid="ignore"):
            plus_di = 100.0 * pdm_s / atr
            minus_di = 100.0 * ndm_s / atr
            dx = 100.0 * np.abs(plus_di - minus_di) / (plus_di + minus_di)
        adx = _wilder_smooth(np.nan_to_num(dx), n) / n
        return adx, plus_di, minus_di
