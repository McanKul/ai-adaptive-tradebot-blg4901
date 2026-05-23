"""
Strategy/ICTLiquiditySweepFVG.py
================================
ICT (Inner Circle Trader) Liquidity Sweep + FVG (OTE) strategy.

Pipeline (top-down, strict):
  1. Killzone filter — London Open + NY AM only (default ON).
  2. HTF bias — resample LTF bars into HTF (default 12 × 5m = 1h),
     run swing detection, require HH/HL + discount zone for bullish
     bias or LL/LH + premium zone for bearish.
  3. Liquidity sweep — current bar wicks above the prior N-bar high
     (or below the prior N-bar low) with a rejection close.
  4. Market Structure Shift (MSS / CHoCH) — close breaks the latest
     intervening swing in the bias direction.
  5. Fair Value Gap (FVG) — find an unfilled 3-bar imbalance after
     the MSS, in the bias direction, sized above
     ``fvg_min_size_atr × ATR``.
  6. OTE confluence — FVG mid must sit inside the Fibonacci 0.62–0.79
     retracement of the sweep→MSS leg.
  7. Emit entry — MARKET (default) or LIMIT at FVG mid.  Stop loss
     sits just outside the sweep wick; take profit at
     ``rr_target × R`` from entry.

The strategy publishes ``decision.metadata["strategy_stop_price"]``
every bar so the engine tick-exit handler can fire the SL intra-bar
instead of waiting for bar close.

NOTE on ``entry_mode="limit_fvg"``:  the default ``SimpleExecutionModel``
only fills LIMIT orders within the same bar.  For limits to persist
across bars (true ICT-style FVG fill behaviour) the engine must be
configured with ``LimitExecutionModel``.  Until that's exposed at the
engine layer, ``"market"`` is the practical default.
"""
from __future__ import annotations

import math
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from Interfaces.IStrategy import IStrategy, StrategyDecision
from Interfaces.orders import Order, OrderSide, OrderType
from Interfaces.market_data import Bar


class Strategy(IStrategy):
    """ICT liquidity-sweep + FVG entry with HTF bias gate."""

    def __init__(
        self,
        # ----- HTF bias -----
        htf_bars_per_ltf: int = 12,        # 12 × 5m = 1h
        htf_swing_window: int = 5,
        fib_discount_max: float = 0.5,     # bullish must be <=50% retracement
        fib_premium_min: float = 0.5,      # bearish must be >=50% retracement
        min_htf_bars: int = 20,
        # ----- Liquidity sweep (LTF) -----
        sweep_lookback: int = 24,
        sweep_wick_ratio: float = 0.5,
        sweep_max_age_bars: int = 8,
        # ----- MSS / CHoCH -----
        mss_swing_window: int = 3,
        mss_max_age_bars: int = 6,
        # ----- FVG -----
        fvg_max_age_bars: int = 10,
        fvg_min_size_atr: float = 0.3,
        # ----- OTE confluence -----
        ote_lower: float = 0.62,
        ote_upper: float = 0.79,
        # ----- Entry -----
        entry_mode: str = "market",        # "market" | "limit_fvg"
        risk_pct: float = 0.005,
        position_size: float = 1.0,        # placeholder; engine applies sizing
        # ----- Killzone (UTC hours) -----
        kill_zone_enabled: bool = True,
        london_start_utc: int = 7,
        london_end_utc: int = 10,
        ny_start_utc: int = 13,
        ny_end_utc: int = 16,
        # ----- Risk / exits -----
        atr_period: int = 14,
        rr_target: float = 3.0,
        max_holding_bars: Optional[int] = 48,   # 4h on 5m
        allow_reversal: bool = False,
        # ----- misc -----
        **kw,
    ):
        if entry_mode not in ("market", "limit_fvg"):
            raise ValueError(
                f"entry_mode must be 'market' or 'limit_fvg', got {entry_mode!r}"
            )

        # HTF bias
        self.htf_bars_per_ltf = max(2, int(htf_bars_per_ltf))
        self.htf_swing_window = max(2, int(htf_swing_window))
        self.fib_discount_max = float(fib_discount_max)
        self.fib_premium_min = float(fib_premium_min)
        self.min_htf_bars = max(self.htf_swing_window * 2 + 2, int(min_htf_bars))

        # Sweep
        self.sweep_lookback = max(3, int(sweep_lookback))
        self.sweep_wick_ratio = float(sweep_wick_ratio)
        self.sweep_max_age_bars = max(1, int(sweep_max_age_bars))

        # MSS
        self.mss_swing_window = max(1, int(mss_swing_window))
        self.mss_max_age_bars = max(1, int(mss_max_age_bars))

        # FVG
        self.fvg_max_age_bars = max(1, int(fvg_max_age_bars))
        self.fvg_min_size_atr = float(fvg_min_size_atr)

        # OTE
        self.ote_lower = float(ote_lower)
        self.ote_upper = float(ote_upper)

        # Entry
        self.entry_mode = entry_mode
        self.risk_pct = float(risk_pct)
        self.position_size = float(position_size)

        # Killzone
        self.kill_zone_enabled = bool(kill_zone_enabled)
        self.london_start_utc = int(london_start_utc)
        self.london_end_utc = int(london_end_utc)
        self.ny_start_utc = int(ny_start_utc)
        self.ny_end_utc = int(ny_end_utc)

        # Risk / exits
        self.atr_period = max(2, int(atr_period))
        self.rr_target = float(rr_target)
        self.max_holding_bars = int(max_holding_bars) if max_holding_bars else None
        self.allow_reversal = bool(allow_reversal)

        # Per-symbol state
        self._state: Dict[str, Dict[str, Any]] = {}

        # Diagnostic counters (env-flagged in __init__; printed at GC)
        import os
        self._debug = bool(os.environ.get("ICT_DEBUG"))
        self._counters: Dict[str, int] = {
            "bars_seen": 0,
            "killzone_blocked": 0,
            "insufficient_ltf": 0,
            "insufficient_htf": 0,
            "bias_none": 0,
            "atr_zero": 0,
            "no_sweep": 0,
            "sweep_dir_mismatch": 0,
            "no_mss": 0,
            "no_fvg": 0,
            "ote_fail": 0,
            "ote_pass": 0,
            "position_guard": 0,
            "entries": 0,
            "exits": 0,
        }

    # ------------------------------------------------------------------
    # State helpers
    # ------------------------------------------------------------------
    def _get_state(self, symbol: str) -> Dict[str, Any]:
        if symbol not in self._state:
            self._state[symbol] = {
                "bar_count": 0,
                "entry_price": 0.0,
                "entry_bar_index": 0,
                "entry_sl": 0.0,
                "entry_tp": 0.0,
                "entry_direction": None,        # "bullish" | "bearish" | None
                "last_sweep": None,             # dict
                "armed_setup": None,            # dict: MSS+FVG+OTE confirmed,
                                                # waiting for pullback into FVG
                "peak_since_entry": -math.inf,  # for longs
                "trough_since_entry": math.inf, # for shorts
            }
        return self._state[symbol]

    def reset(self) -> None:
        self._state.clear()

    def __del__(self):
        # Dump counters to file for diagnostic visibility
        try:
            if hasattr(self, "_counters"):
                with open("logs/ict_counters.txt", "a") as fh:
                    fh.write(f"{self._counters}\n")
        except Exception:
            pass

    # ------------------------------------------------------------------
    # IStrategy interface
    # ------------------------------------------------------------------
    def on_bar(self, bar: Bar, ctx: Any) -> StrategyDecision:  # noqa: C901
        st = self._get_state(bar.symbol)
        st["bar_count"] += 1
        self._counters["bars_seen"] += 1
        position = getattr(ctx, "position", 0.0) or 0.0

        features: Dict[str, Any] = {}
        active_stop: Optional[float] = None

        # 1) Manage existing position
        if abs(position) > 1e-10:
            exit_result = self._manage_position(bar, st, position, features)
            if exit_result is not None:
                return self._decision([exit_result], features,
                                      {"strategy_stop_price": None,
                                       "exit_reason": st.get("_last_exit_reason", "")})
            # Position still open: keep current entry SL as active_stop
            active_stop = st["entry_sl"] if st["entry_sl"] else None

        # 1b) If a setup is armed and we're flat, check if this bar reaches
        # the FVG zone for a market entry.  Pullback-style.
        if abs(position) < 1e-10 and st.get("armed_setup") is not None:
            armed = st["armed_setup"]
            # Staleness
            if st["bar_count"] - armed["armed_bar_index"] > self.fvg_max_age_bars:
                st["armed_setup"] = None
            else:
                in_fvg = self._bar_in_fvg(bar, armed)
                if in_fvg:
                    side = OrderSide.BUY if armed["direction"] == "bullish" \
                                          else OrderSide.SELL
                    entry_order = Order(
                        symbol=bar.symbol,
                        side=side,
                        order_type=OrderType.MARKET,
                        quantity=self.position_size,
                        timestamp_ns=bar.timestamp_ns,
                        strategy_id="ICT_PULLBACK_ENTRY",
                        metadata={
                            "stop_price": armed["sl"],
                            "tp_price": armed["tp"],
                            "atr": armed["atr"],
                            "fvg_mid": armed["fvg"]["mid"],
                            "fvg_upper": armed["fvg"]["upper"],
                            "fvg_lower": armed["fvg"]["lower"],
                            "bias": armed["direction"],
                            "sweep_side": armed["sweep_side"],
                            "qty_method": "vol_target",
                        },
                    )
                    st["entry_price"] = float(armed["fvg"]["mid"])
                    st["entry_bar_index"] = st["bar_count"]
                    st["entry_sl"] = armed["sl"]
                    st["entry_tp"] = armed["tp"]
                    st["entry_direction"] = armed["direction"]
                    st["peak_since_entry"] = float(bar.close)
                    st["trough_since_entry"] = float(bar.close)
                    st["armed_setup"] = None  # consumed
                    self._counters["entries"] += 1
                    return self._decision(
                        [entry_order], features,
                        {"entry_side": armed["direction"],
                         "strategy_stop_price": armed["sl"],
                         "tp_price": armed["tp"],
                         "bias": armed["direction"],
                         "fvg_mid": armed["fvg"]["mid"]},
                    )

        # 2) Killzone filter (applies to NEW entries only)
        if self.kill_zone_enabled and not self._in_killzone(bar.timestamp_ns):
            features["filter"] = "killzone"
            self._counters["killzone_blocked"] += 1
            return self._decision([], features,
                                  {"strategy_stop_price": active_stop})

        # 3) Pull LTF history
        ohlcv = ctx.get_ohlcv()
        if not ohlcv:
            return self._decision([], features,
                                  {"strategy_stop_price": active_stop})

        closes = np.asarray(ohlcv["close"], dtype=float)
        highs = np.asarray(ohlcv["high"], dtype=float)
        lows = np.asarray(ohlcv["low"], dtype=float)
        opens = np.asarray(ohlcv["open"], dtype=float)

        # Need enough LTF history for sweep lookback + ATR + a couple HTF bars
        min_ltf = max(self.sweep_lookback + 4,
                      self.atr_period + 2,
                      self.htf_bars_per_ltf * self.min_htf_bars // self.htf_bars_per_ltf
                      + self.htf_swing_window * 2 * self.htf_bars_per_ltf)
        if len(closes) < min_ltf:
            self._counters["insufficient_ltf"] += 1
            return self._decision([], features,
                                  {"strategy_stop_price": active_stop})

        # 4) HTF bias via internal resampling
        htf = self._resample_to_htf(opens, highs, lows, closes,
                                    self.htf_bars_per_ltf)
        if htf is None or len(htf["close"]) < self.min_htf_bars:
            self._counters["insufficient_htf"] += 1
            return self._decision([], features,
                                  {"strategy_stop_price": active_stop})

        bias_info = self._htf_bias(htf)
        if bias_info["bias"] == "none":
            features["bias"] = "none"
            self._counters["bias_none"] += 1
            return self._decision([], features,
                                  {"strategy_stop_price": active_stop})

        features["bias"] = bias_info["bias"]
        bias = bias_info["bias"]

        # 5) ATR for sizing-size filters
        atr = self._compute_atr(highs, lows, closes, self.atr_period)
        if atr <= 0:
            self._counters["atr_zero"] += 1
            return self._decision([], features,
                                  {"strategy_stop_price": active_stop})

        # 6) Liquidity sweep detection (this-bar) — when found, persist to
        # state with post-sweep extreme tracker.  MSS happens on a LATER
        # bar that closes through that extreme.
        fresh_sweep = self._detect_liquidity_sweep(
            highs, lows, closes,
            self.sweep_lookback, self.sweep_wick_ratio,
        )
        if fresh_sweep is not None and (
            (fresh_sweep["side"] == "high" and bias == "bearish") or
            (fresh_sweep["side"] == "low" and bias == "bullish")
        ):
            # Initialise the sweep state for MSS tracking.
            st["last_sweep"] = {
                **fresh_sweep,
                "bar_index": st["bar_count"],
                # For bullish (low sweep) we track the running HIGH after
                # the sweep — when a future close breaks it that's MSS.
                # For bearish (high sweep) we track the running LOW.
                "post_extreme": float(highs[-1]) if fresh_sweep["side"] == "low"
                                                else float(lows[-1]),
            }
        elif fresh_sweep is not None:
            features["sweep_dir_mismatch"] = fresh_sweep["side"]
            self._counters["sweep_dir_mismatch"] += 1

        sweep_state = st.get("last_sweep")
        if sweep_state is None:
            self._counters["no_sweep"] += 1
            return self._decision([], features,
                                  {"strategy_stop_price": active_stop})

        # Staleness check — abandon old sweeps so MSS doesn't trigger weeks later
        sweep_age = st["bar_count"] - sweep_state["bar_index"]
        if sweep_age > self.sweep_max_age_bars:
            st["last_sweep"] = None
            self._counters["no_sweep"] += 1
            return self._decision([], features,
                                  {"strategy_stop_price": active_stop})

        # Sweep side must still match bias (it always does at insert time,
        # but bias can flip between bars).
        if (sweep_state["side"] == "high" and bias != "bearish") or \
           (sweep_state["side"] == "low" and bias != "bullish"):
            st["last_sweep"] = None
            self._counters["sweep_dir_mismatch"] += 1
            return self._decision([], features,
                                  {"strategy_stop_price": active_stop})

        features["sweep"] = sweep_state

        # 7) MSS / CHoCH — update post-sweep extreme tracker, then test
        # whether the current close breaks it.  Must be at least 1 bar
        # past the sweep so there's a real post-sweep structure to break.
        if sweep_age < 1:
            self._counters["no_mss"] += 1
            return self._decision([], features,
                                  {"strategy_stop_price": active_stop})

        if sweep_state["side"] == "low":  # bullish setup
            # Update peak of counter-move (using the prior bar, not current)
            sweep_state["post_extreme"] = max(sweep_state["post_extreme"],
                                              float(highs[-2]))
            mss_triggered = float(closes[-1]) > sweep_state["post_extreme"]
            mss_level = sweep_state["post_extreme"]
        else:  # high sweep, bearish setup
            sweep_state["post_extreme"] = min(sweep_state["post_extreme"],
                                              float(lows[-2]))
            mss_triggered = float(closes[-1]) < sweep_state["post_extreme"]
            mss_level = sweep_state["post_extreme"]

        if not mss_triggered:
            self._counters["no_mss"] += 1
            return self._decision([], features,
                                  {"strategy_stop_price": active_stop})

        mss = {"mss_level": float(mss_level),
               "mss_bar_offset": len(closes) - 1,
               "direction": bias}

        features["mss"] = mss

        # 8) FVG in bias direction after the MSS
        direction = "bullish" if bias == "bullish" else "bearish"
        fvg = self._detect_fvg(highs, lows, closes,
                               direction, mss["mss_bar_offset"],
                               self.fvg_max_age_bars,
                               self.fvg_min_size_atr, atr)
        if fvg is None:
            self._counters["no_fvg"] += 1
            return self._decision([], features,
                                  {"strategy_stop_price": active_stop})

        features["fvg"] = fvg

        # 9) OTE check using the sweep→MSS leg
        swing_low, swing_high = self._sweep_to_mss_leg(sweep_state, mss, direction,
                                                       highs, lows)
        if not self._check_ote(swing_low, swing_high, fvg["mid"],
                               direction,
                               self.ote_lower, self.ote_upper):
            features["ote_fail"] = True
            self._counters["ote_fail"] += 1
            return self._decision([], features,
                                  {"strategy_stop_price": active_stop})

        self._counters["ote_pass"] += 1

        # 10) Setup confirmed → ARM (don't enter at MSS bar; wait for
        # pullback into the FVG zone, which is the actual ICT entry).
        # Skip if already in a position (no pyramiding/reversal here).
        if abs(position) > 1e-10:
            self._counters["position_guard"] += 1
            return self._decision([], features,
                                  {"strategy_stop_price": active_stop})

        sl, tp, entry_price = self._compute_levels(direction, sweep_state, fvg,
                                                   atr, float(closes[-1]))
        st["armed_setup"] = {
            "direction": direction,
            "fvg": fvg,
            "sl": sl,
            "tp": tp,
            "atr": atr,
            "sweep_side": sweep_state["side"],
            "armed_bar_index": st["bar_count"],
        }

        if self._debug:
            import logging
            logging.getLogger("ICT").info(
                "ARMED %s %s bar=%d sl=%.6f tp=%.6f fvg=[%.6f,%.6f]",
                bar.symbol, direction, st["bar_count"], sl, tp,
                fvg["lower"], fvg["upper"],
            )

        # Many setups will arm AND the current bar's range already overlaps
        # the FVG (e.g. the MSS bar reached down into the gap during the
        # break-and-retrace).  Try a same-bar pullback entry here.
        in_fvg = self._bar_in_fvg(bar, st["armed_setup"])
        if in_fvg:
            side = OrderSide.BUY if direction == "bullish" else OrderSide.SELL
            entry_order = Order(
                symbol=bar.symbol,
                side=side,
                order_type=OrderType.MARKET,
                quantity=self.position_size,
                timestamp_ns=bar.timestamp_ns,
                strategy_id="ICT_MSS_BAR_ENTRY",
                metadata={
                    "stop_price": sl,
                    "tp_price": tp,
                    "atr": atr,
                    "fvg_mid": fvg["mid"],
                    "fvg_upper": fvg["upper"],
                    "fvg_lower": fvg["lower"],
                    "bias": bias,
                    "sweep_side": sweep_state["side"],
                    "qty_method": "vol_target",
                },
            )
            st["entry_price"] = float(fvg["mid"])
            st["entry_bar_index"] = st["bar_count"]
            st["entry_sl"] = sl
            st["entry_tp"] = tp
            st["entry_direction"] = direction
            st["peak_since_entry"] = float(bar.close)
            st["trough_since_entry"] = float(bar.close)
            st["armed_setup"] = None  # consumed
            self._counters["entries"] += 1
            return self._decision(
                [entry_order], features,
                {"entry_side": direction,
                 "strategy_stop_price": sl,
                 "tp_price": tp,
                 "bias": bias,
                 "fvg_mid": fvg["mid"]},
            )

        # Otherwise just stay armed and wait for a subsequent bar.
        return self._decision([], features,
                              {"strategy_stop_price": active_stop,
                               "armed": True,
                               "direction": direction})

    # ------------------------------------------------------------------
    # Position management
    # ------------------------------------------------------------------
    def _manage_position(self, bar: Bar, st: Dict[str, Any],
                         position: float, features: Dict[str, Any]
                         ) -> Optional[Order]:
        """Check SL/TP/time-stop on the open position.  Returns an exit
        order if one triggers, else None.
        """
        direction = st.get("entry_direction")
        sl = st.get("entry_sl", 0.0)
        tp = st.get("entry_tp", 0.0)

        # Track peak / trough for diagnostics (strategy stop is fixed at sweep wick)
        if direction == "bullish":
            st["peak_since_entry"] = max(st["peak_since_entry"], float(bar.high))
        elif direction == "bearish":
            st["trough_since_entry"] = min(st["trough_since_entry"], float(bar.low))

        # Bar-close fallback SL/TP (the engine tick-exit fires intra-bar via
        # strategy_stop_price, but we keep this as a safety net).
        if direction == "bullish":
            if bar.low <= sl:
                st["_last_exit_reason"] = "sl_long"
                return self._exit_order(bar, position)
            if bar.high >= tp:
                st["_last_exit_reason"] = "tp_long"
                return self._exit_order(bar, position)
        elif direction == "bearish":
            if bar.high >= sl:
                st["_last_exit_reason"] = "sl_short"
                return self._exit_order(bar, position)
            if bar.low <= tp:
                st["_last_exit_reason"] = "tp_short"
                return self._exit_order(bar, position)

        # Time stop
        if self.max_holding_bars is not None:
            bars_held = st["bar_count"] - st["entry_bar_index"]
            if bars_held >= self.max_holding_bars:
                st["_last_exit_reason"] = "time_stop"
                return self._exit_order(bar, position)

        return None

    # ------------------------------------------------------------------
    # Killzone
    # ------------------------------------------------------------------
    def _in_killzone(self, ts_ns: int) -> bool:
        try:
            hour = datetime.fromtimestamp(ts_ns / 1e9,
                                          tz=timezone.utc).hour
        except (OSError, ValueError, OverflowError):
            return False
        if self.london_start_utc <= hour < self.london_end_utc:
            return True
        if self.ny_start_utc <= hour < self.ny_end_utc:
            return True
        return False

    # ------------------------------------------------------------------
    # HTF resampling
    # ------------------------------------------------------------------
    @staticmethod
    def _resample_to_htf(opens: np.ndarray, highs: np.ndarray,
                         lows: np.ndarray, closes: np.ndarray,
                         ratio: int) -> Optional[Dict[str, np.ndarray]]:
        """Aggregate every ``ratio`` LTF bars into one HTF bar.  Only
        returns COMPLETED HTF bars (drops any trailing partial)."""
        n = len(closes)
        if n < ratio:
            return None
        # Drop the trailing partial group
        n_full = (n // ratio) * ratio
        if n_full < ratio:
            return None
        chunks = n_full // ratio
        o = opens[:n_full].reshape(chunks, ratio)
        h = highs[:n_full].reshape(chunks, ratio)
        l = lows[:n_full].reshape(chunks, ratio)
        c = closes[:n_full].reshape(chunks, ratio)
        return {
            "open": o[:, 0],
            "high": h.max(axis=1),
            "low": l.min(axis=1),
            "close": c[:, -1],
        }

    # ------------------------------------------------------------------
    # Swing detection
    # ------------------------------------------------------------------
    @staticmethod
    def _find_swings(highs: np.ndarray, lows: np.ndarray, window: int
                     ) -> Tuple[List[Tuple[int, float]],
                                List[Tuple[int, float]]]:
        """Pivot swing detection.  A bar at index i is a swing high if
        highs[i] is strictly greater than highs in ``window`` bars on
        each side.  Returns (swing_highs, swing_lows) as lists of
        (index, price), oldest first.
        """
        swing_highs: List[Tuple[int, float]] = []
        swing_lows: List[Tuple[int, float]] = []
        n = len(highs)
        for i in range(window, n - window):
            seg_h = highs[i - window:i + window + 1]
            seg_l = lows[i - window:i + window + 1]
            if highs[i] == seg_h.max() and (seg_h == highs[i]).sum() == 1:
                swing_highs.append((i, float(highs[i])))
            if lows[i] == seg_l.min() and (seg_l == lows[i]).sum() == 1:
                swing_lows.append((i, float(lows[i])))
        return swing_highs, swing_lows

    # ------------------------------------------------------------------
    # HTF bias
    # ------------------------------------------------------------------
    def _htf_bias(self, htf: Dict[str, np.ndarray]) -> Dict[str, Any]:
        """Determine HTF bias: bullish if HH+HL and price in discount,
        bearish if LL+LH and price in premium, otherwise none."""
        highs = htf["high"]
        lows = htf["low"]
        closes = htf["close"]
        swing_highs, swing_lows = self._find_swings(
            highs, lows, self.htf_swing_window
        )
        if len(swing_highs) < 2 or len(swing_lows) < 2:
            return {"bias": "none"}

        h1 = swing_highs[-2][1]
        h2 = swing_highs[-1][1]
        l1 = swing_lows[-2][1]
        l2 = swing_lows[-1][1]
        current = float(closes[-1])

        hh_hl = h2 > h1 and l2 > l1
        ll_lh = h2 < h1 and l2 < l1

        # Reference swing range = the most recent leg.  Use the most
        # recent swing high and swing low regardless of order.
        range_low = min(l2, l1, swing_lows[-1][1])
        range_high = max(h2, h1, swing_highs[-1][1])
        if range_high - range_low <= 0:
            return {"bias": "none"}

        # Retracement: 0.0 = at range_low, 1.0 = at range_high
        retrace = (current - range_low) / (range_high - range_low)
        in_discount = retrace <= self.fib_discount_max
        in_premium = retrace >= self.fib_premium_min

        if hh_hl and in_discount:
            return {"bias": "bullish", "retrace": float(retrace),
                    "range_low": float(range_low),
                    "range_high": float(range_high)}
        if ll_lh and in_premium:
            return {"bias": "bearish", "retrace": float(retrace),
                    "range_low": float(range_low),
                    "range_high": float(range_high)}
        return {"bias": "none"}

    # ------------------------------------------------------------------
    # Liquidity sweep
    # ------------------------------------------------------------------
    def _detect_liquidity_sweep(self, highs: np.ndarray, lows: np.ndarray,
                                closes: np.ndarray,
                                lookback: int, wick_ratio: float
                                ) -> Optional[Dict[str, Any]]:
        """Detect if the most recent CLOSED bar swept a prior level.

        High-side sweep: bar.high > prior lookback-bar high, with a
        rejection wick ≥ ``wick_ratio × bar range`` and bar closes back
        below the prior high.  Low-side sweep is symmetric.
        """
        if len(highs) < lookback + 2:
            return None
        # Prior window EXCLUDES the current bar
        prior_high = float(np.max(highs[-lookback - 1:-1]))
        prior_low = float(np.min(lows[-lookback - 1:-1]))

        h = float(highs[-1])
        l = float(lows[-1])
        c = float(closes[-1])
        rng = h - l
        if rng <= 0:
            return None

        # High-side sweep
        if h > prior_high and c < prior_high:
            upper_wick = h - max(c, l)
            if upper_wick >= wick_ratio * rng:
                return {"side": "high", "level": prior_high,
                        "bar_high": h, "bar_close": c}
        # Low-side sweep
        if l < prior_low and c > prior_low:
            lower_wick = min(c, h) - l
            if lower_wick >= wick_ratio * rng:
                return {"side": "low", "level": prior_low,
                        "bar_low": l, "bar_close": c}
        return None

    # ------------------------------------------------------------------
    # MSS / CHoCH
    # ------------------------------------------------------------------
    def _detect_mss(self, highs: np.ndarray, lows: np.ndarray,
                    closes: np.ndarray,
                    sweep: Dict[str, Any], bias: str,
                    swing_window: int, max_age: int
                    ) -> Optional[Dict[str, Any]]:
        """After the sweep bar, look for a structural break in the bias
        direction within ``max_age`` bars.

        Bullish MSS: close breaks ABOVE the most recent intervening
        swing high (between sweep bar and current bar).
        Bearish MSS: close breaks BELOW the most recent intervening
        swing low.
        """
        n = len(closes)
        # The sweep was on the latest bar (index n-1) when detected, so
        # look for the swing pivot in the window [n-1-max_age, n-1].
        start = max(0, n - 1 - max_age)
        end = n  # exclusive
        if end - start < 2 * swing_window + 1:
            return None

        seg_high = highs[start:end]
        seg_low = lows[start:end]

        swing_highs, swing_lows = self._find_swings(
            seg_high, seg_low, swing_window
        )

        current = float(closes[-1])
        if bias == "bullish":
            # need a recent swing high inside the window that current close breaks
            for idx, price in reversed(swing_highs):
                if current > price:
                    return {"mss_level": float(price),
                            "mss_bar_offset": start + idx,
                            "direction": "bullish"}
            return None
        else:  # bearish
            for idx, price in reversed(swing_lows):
                if current < price:
                    return {"mss_level": float(price),
                            "mss_bar_offset": start + idx,
                            "direction": "bearish"}
            return None

    # ------------------------------------------------------------------
    # FVG
    # ------------------------------------------------------------------
    @staticmethod
    def _detect_fvg(highs: np.ndarray, lows: np.ndarray,
                    closes: np.ndarray,
                    direction: str, mss_bar_offset: int,
                    max_age: int, min_size_atr: float, atr: float
                    ) -> Optional[Dict[str, Any]]:
        """Find the latest unfilled FVG in ``direction`` whose middle bar
        index falls in [mss_bar_offset, n-1] and is within ``max_age``
        of the current bar.

        Bullish FVG: highs[i-2] < lows[i] (gap below current price);
        the strategy treats the gap as a potential pullback zone.  An
        FVG is "filled" if any bar after i has a low (bullish) or high
        (bearish) that closes the gap.
        """
        n = len(closes)
        # Need at least 3 bars to form an FVG plus 1 to check fills
        first_idx = max(2, mss_bar_offset)
        last_idx = n - 1
        oldest = max(first_idx, last_idx - max_age)

        best: Optional[Dict[str, Any]] = None
        # Iterate newest → oldest, return first valid (newest) FVG
        for i in range(last_idx, oldest - 1, -1):
            if i < 2:
                break
            if direction == "bullish":
                upper = float(lows[i])
                lower = float(highs[i - 2])
                if upper <= lower:
                    continue
                gap = upper - lower
                if gap < min_size_atr * atr:
                    continue
                # Filled if any subsequent bar trades through lower edge
                fill_seg = lows[i + 1:]
                if fill_seg.size > 0 and fill_seg.min() <= lower:
                    continue
                best = {"upper": upper, "lower": lower,
                        "mid": 0.5 * (upper + lower),
                        "bar_offset": i, "direction": "bullish"}
                break
            else:  # bearish
                upper = float(lows[i - 2])
                lower = float(highs[i])
                if upper <= lower:
                    continue
                gap = upper - lower
                if gap < min_size_atr * atr:
                    continue
                fill_seg = highs[i + 1:]
                if fill_seg.size > 0 and fill_seg.max() >= upper:
                    continue
                best = {"upper": upper, "lower": lower,
                        "mid": 0.5 * (upper + lower),
                        "bar_offset": i, "direction": "bearish"}
                break
        return best

    # ------------------------------------------------------------------
    # OTE confluence
    # ------------------------------------------------------------------
    @staticmethod
    def _check_ote(swing_low: float, swing_high: float, fvg_mid: float,
                   direction: str,
                   ote_lower: float, ote_upper: float) -> bool:
        """Check if FVG mid sits inside the Fibonacci OTE retracement of
        the sweep→MSS leg."""
        rng = swing_high - swing_low
        if rng <= 0:
            return False
        if direction == "bullish":
            # Leg moved low→high; retracement of FVG mid is measured
            # from the high.
            retrace = (swing_high - fvg_mid) / rng
        else:
            retrace = (fvg_mid - swing_low) / rng
        return ote_lower <= retrace <= ote_upper

    @staticmethod
    def _sweep_to_mss_leg(sweep: Dict[str, Any], mss: Dict[str, Any],
                          direction: str,
                          highs: np.ndarray, lows: np.ndarray
                          ) -> Tuple[float, float]:
        """Return (swing_low, swing_high) for the sweep→MSS leg.

        For a bullish setup, the sweep was on the low side; leg endpoints
        are sweep['level'] (the swept low) and the most recent high in
        the segment between the sweep and the MSS confirmation.
        """
        if direction == "bullish":
            swing_low = float(sweep["level"])
            # High of the segment after the sweep (use last bars)
            seg = highs[-min(20, len(highs)):]
            swing_high = float(seg.max())
        else:
            swing_high = float(sweep["level"])
            seg = lows[-min(20, len(lows)):]
            swing_low = float(seg.min())
        return swing_low, swing_high

    # ------------------------------------------------------------------
    # Entry levels
    # ------------------------------------------------------------------
    def _compute_levels(self, direction: str, sweep: Dict[str, Any],
                        fvg: Dict[str, Any], atr: float,
                        last_close: float
                        ) -> Tuple[float, float, float]:
        """Return (sl, tp, entry_price)."""
        # Sweep wick = bar_high (for high sweep) or bar_low (for low sweep)
        if direction == "bullish":
            wick = float(sweep.get("bar_low", sweep["level"]))
            sl = wick - 0.1 * atr
            entry_price = float(fvg["mid"])
            risk = entry_price - sl
            tp = entry_price + self.rr_target * risk
        else:
            wick = float(sweep.get("bar_high", sweep["level"]))
            sl = wick + 0.1 * atr
            entry_price = float(fvg["mid"])
            risk = sl - entry_price
            tp = entry_price - self.rr_target * risk
        return sl, tp, entry_price

    # ------------------------------------------------------------------
    # Helpers shared with sister strategies
    # ------------------------------------------------------------------
    @staticmethod
    def _bar_in_fvg(bar: Bar, armed: Dict[str, Any]) -> bool:
        """Return True if the bar's range overlaps the armed FVG zone."""
        fvg = armed["fvg"]
        lo = float(fvg["lower"])
        hi = float(fvg["upper"])
        bar_lo = float(bar.low)
        bar_hi = float(bar.high)
        # Overlap check: ranges intersect
        return bar_lo <= hi and bar_hi >= lo

    @staticmethod
    def _exit_order(bar: Bar, signed_position: float) -> Order:
        side = OrderSide.SELL if signed_position > 0 else OrderSide.BUY
        return Order(
            symbol=bar.symbol,
            side=side,
            order_type=OrderType.MARKET,
            quantity=abs(signed_position),
            timestamp_ns=bar.timestamp_ns,
            strategy_id="ICT_EXIT",
            reduce_only=True,
        )

    @staticmethod
    def _decision(orders: List[Order], features: Dict[str, Any],
                  metadata: Dict[str, Any]) -> StrategyDecision:
        return StrategyDecision(orders=orders, features=features,
                                metadata=metadata)

    @staticmethod
    def _compute_atr(highs: np.ndarray, lows: np.ndarray,
                     closes: np.ndarray, period: int) -> float:
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
