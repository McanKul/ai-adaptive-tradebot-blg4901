"""
live/live_engine.py
===================
Live trading engine that coordinates:
- Strategy signals (via IStrategy)
- News sentiment analysis (via NewsEngine + SentimentMarginAdjuster)
- Position management (via LiveSupervisor — per-symbol PositionManager)
- Account-level risk (via LiveGlobalRisk)

Sentiment-driven margin adjustment:
- Strategy decides entries independently (LONG / SHORT)
- News sentiment adjusts position margin when it strongly confirms
  the direction: strongly bullish + LONG → bigger position,
  strongly bearish + SHORT → bigger position.
- No news engine configured → default margin
"""
from __future__ import annotations

import asyncio
import time
from typing import Optional, Type

from utils.bar_store import BarStore
from Interfaces.IBroker import IBroker
from Interfaces.INewsSource import INewsSource
from Interfaces.ISentimentAnalyzer import ISentimentAnalyzer
from Interfaces.ISignalCombiner import IMarginAdjuster
from Interfaces.market_data import Bar
from Interfaces.orders import OrderSide
from Interfaces.strategy_adapter import StrategyContext
from live.live_config import LiveConfig
from live.position_manager import LiveSupervisor
from live.global_risk import LiveGlobalRisk
from live.live_metrics import LiveMetrics
from live.notifier import TelegramNotifier
from live.streamer import Streamer
from utils.logger import setup_logger

try:
    from news.news_engine import NewsEngine
except ImportError:
    NewsEngine = None

log = setup_logger("LiveEngine")


class LiveEngine:
    """
    Config-driven live trading engine.

    Uses LiveConfig for all settings, LiveSupervisor for per-symbol
    position management, and optional NewsEngine for sentiment signals.
    """

    def __init__(
        self,
        cfg: LiveConfig,
        broker: IBroker,
        strategy_cls: Optional[Type] = None,
        global_risk: LiveGlobalRisk = None,
        news_source: Optional[INewsSource] = None,
        sentiment_analyzer: Optional[ISentimentAnalyzer] = None,
        margin_adjuster: Optional[IMarginAdjuster] = None,
        market_client=None,  # Real client for WebSocket/preload in dry-run mode
        strategy=None,  # Pre-built IStrategy (e.g. CompositeStrategy); takes precedence
    ):
        self.cfg = cfg
        self.broker = broker
        self.global_risk = global_risk

        # Phase C3 — rejection circuit-breaker.  Wires every
        # BinanceAPIException raised by the broker into a sliding
        # 5-minute window; a 5-strike storm trips the kill switch.
        # DryBroker exceptions don't reach here (no API), so this is
        # effectively live-only safety.
        try:
            from live.rejection_counter import RejectionCounter
            self._rejection_counter = RejectionCounter(
                max_count=5, window_seconds=300.0,
                on_trip=self.global_risk.trip_kill_switch,
            )
            if hasattr(broker, "attach_rejection_counter"):
                broker.attach_rejection_counter(self._rejection_counter)
        except Exception as e:  # pragma: no cover — defensive
            log.warning("rejection counter disabled: %s", e)
            self._rejection_counter = None
        self.bar_store = BarStore()
        # Use dedicated market client if provided (e.g. dry-run needs real WS)
        self._market_client = market_client or broker.client

        # ── Strategy Instance ──────────────────────────────────────────
        if strategy is not None:
            self.strategy = strategy
        elif strategy_cls is not None:
            self.strategy = strategy_cls(
                bar_store=self.bar_store,
                **cfg.strategy_params,
            )
        else:
            raise ValueError(
                "LiveEngine requires either strategy_cls or a pre-built strategy"
            )

        # ── LiveSupervisor (per-symbol PositionManager) ────────────────
        # ``persist_path`` is run_id-namespaced so two parallel dry-runs
        # (sentiment ON vs OFF demo) keep separate position stores.
        self.supervisor = LiveSupervisor(
            broker,
            max_global_positions=cfg.risk.max_concurrent_positions,
            persist_path=cfg.positions_state_path(),
            liq_guard_cfg=getattr(cfg, "liquidation_guard", None),
            execution_cfg=cfg.execution,
        )
        # ``book_streamer`` is wired in run() once the streamer is built —
        # supervisor.book_streamer is initially None.

        # ── Streamer ───────────────────────────────────────────────────
        self.timeframes = [cfg.timeframe]
        self.streamer: Optional[Streamer] = None
        self.tick_streamer = None  # markPrice MarkPriceTickStreamer
        self.book_streamer = None  # bookTicker MarkPriceTickStreamer
        self.symbols: list[str] = []

        # ── Equity Cache (avoid API call every bar) ────────────────────
        self._last_equity: float = 10_000.0
        self._equity_ts: float = 0.0
        self._EQUITY_REFRESH_SEC: float = 30.0

        # ── Live Metrics ─────────────────────────────────────────────
        # CSV path is run_id-namespaced so parallel dry-runs don't share rows.
        self.metrics = LiveMetrics(csv_path=cfg.trade_log_path())

        # ── Notifications ────────────────────────────────────────────
        self.notifier = TelegramNotifier()

        # ── News Sentiment (margin adjustment) ──────────────────────────
        self.news_engine = None
        self.margin_adjuster = margin_adjuster

        if news_source and sentiment_analyzer and NewsEngine:
            self.news_engine = NewsEngine(
                news_source=news_source,
                sentiment_analyzer=sentiment_analyzer,
                refresh_interval=cfg.news.refresh_interval,
                news_limit=cfg.news.news_limit,
            )
            log.info("News sentiment enabled (refresh=%ds)", cfg.news.refresh_interval)

        if self.news_engine and not self.margin_adjuster:
            log.warning(
                "News engine enabled but no margin adjuster — "
                "sentiment will be logged but NOT used for margin adjustment"
            )

    # ── Realised cost accounting (Phase B4 follow-up) ─────────────────

    async def _apply_realised_costs(self, pos) -> None:
        """Populate ``pos.exit_fee_usd`` and ``pos.funding_usd`` from
        the broker before the CSV row is written.

        Entry-side commission already lands at fill time
        (``open_position`` Phase C1 path).  This hook covers the
        exit leg + the funding payments accrued while the position
        was open.  Best-effort: any broker error logs and leaves the
        attribute at its current value so gross-only logging still
        works.
        """
        try:
            since_ms = int((getattr(pos, "open_ts", time.time())) * 1000)
            until_ms = int((getattr(pos, "exit_ts", None) or time.time()) * 1000) + 1000

            # Total commission across BOTH legs, then subtract whatever
            # the entry leg already recorded so we don't double-count.
            try:
                total_fee = await self.broker.get_realised_commission(
                    pos.symbol, since_ms=since_ms, until_ms=until_ms,
                )
            except Exception as e:
                log.debug("realised commission fetch failed for %s: %s",
                          pos.symbol, e)
                total_fee = 0.0

            entry_fee = float(getattr(pos, "entry_fee_usd", 0.0) or 0.0)
            exit_fee = max(0.0, total_fee - entry_fee)
            pos.exit_fee_usd = exit_fee

            try:
                pos.funding_usd = await self.broker.get_funding_paid(
                    pos.symbol, since_ms=since_ms, until_ms=until_ms,
                )
            except Exception as e:
                log.debug("funding fetch failed for %s: %s", pos.symbol, e)
                # Leave existing value (default 0) — don't overwrite
                # any value the strategy may have set manually.
        except Exception as e:  # pragma: no cover — top-level guard
            log.warning("apply_realised_costs unexpected: %s", e)

    # ── Reconciliation (drift detection) ──────────────────────────────

    async def _reconciliation_loop(self) -> None:
        """Periodically compare local positions vs the exchange.

        Runs as a background task while the engine is alive.  On every
        non-empty drift report calls :meth:`_handle_drift` which either
        logs, trips the kill-switch, or force-closes (per
        ``cfg.reconciliation.action``).
        """
        rcfg = self.cfg.reconciliation
        interval = max(5.0, float(rcfg.interval_seconds))
        try:
            while True:
                await asyncio.sleep(interval)
                try:
                    drifts = await self.supervisor.detect_drift(
                        qty_tolerance=rcfg.qty_tolerance,
                    )
                except Exception as e:  # pragma: no cover — defensive
                    log.warning("drift detection raised: %s", e)
                    continue
                if drifts:
                    await self._handle_drift(drifts, rcfg.action)
        except asyncio.CancelledError:
            log.info("Reconciliation loop cancelled")
            raise

    async def _handle_drift(
        self, drifts: list[dict], action: str,
    ) -> None:
        """Apply the configured response to a non-empty drift list."""
        for d in drifts:
            log.warning(
                "DRIFT %s | local=%.6f exchange=%.6f abs_diff=%.6f kind=%s",
                d["symbol"], d["local_qty"], d["exchange_qty"],
                d["abs_diff"], d["kind"],
            )

        # Telegram (best-effort, failures are non-fatal)
        try:
            summary = ", ".join(
                f"{d['symbol']}({d['kind']}: Δ{d['abs_diff']:.4f})" for d in drifts
            )
            await self.notifier.kill_switch(f"position drift detected — {summary}")
        except Exception:  # pragma: no cover
            pass

        if action == "alarm":
            return

        if action == "halt":
            reason = (
                "position drift: "
                + ", ".join(f"{d['symbol']}/{d['kind']}" for d in drifts)
            )
            self.global_risk.trip_kill_switch(reason)
            return

        if action == "force_flat":
            for d in drifts:
                try:
                    await self.broker.close_position(d["symbol"])
                except Exception as e:
                    log.error("force_flat close failed for %s: %s",
                              d["symbol"], e)
            # Trip kill-switch anyway so we don't keep entering blind
            self.global_risk.trip_kill_switch(
                "position drift: force_flat applied"
            )
            return

        log.warning("Unknown reconciliation action %r — falling back to halt", action)
        self.global_risk.trip_kill_switch(f"unknown action {action!r}")

    # ── Equity Cache ───────────────────────────────────────────────────

    async def _refresh_equity(self) -> float:
        """Return cached equity; only hits the API if stale (>30s)."""
        now = time.monotonic()
        if now - self._equity_ts >= self._EQUITY_REFRESH_SEC:
            try:
                self._last_equity = await self.broker.balance("USDT")
                self._equity_ts = now
            except Exception as e:
                log.warning("Balance refresh failed, using cached: %s", e)
        return self._last_equity

    # ── Sentiment-Driven Margin Adjustment ─────────────────────────────

    async def _get_sentiment_margin_multiplier(
        self, symbol: str, direction: int,
    ) -> float:
        """
        Compute a margin multiplier from news sentiment.

        Args:
            symbol: Trading symbol (e.g. "BTCUSDT")
            direction: +1 for LONG, -1 for SHORT

        Returns:
            Float >= 1.0.  1.0 = default margin; >1.0 = boosted margin.
        """
        # No news engine → default margin
        if not self.news_engine:
            return 1.0

        # Fetch sentiment score (cached or fresh)
        sentiment_score = await self.news_engine.get_sentiment(symbol)
        log.info(
            "[%s] direction=%+d, sentiment=%.2f",
            symbol, direction, sentiment_score,
        )

        # No adjuster → log only, default margin
        if not self.margin_adjuster:
            return 1.0

        # Compute margin multiplier
        multiplier = self.margin_adjuster.compute_margin_multiplier(
            direction, sentiment_score,
        )

        if multiplier > 1.0:
            log.info(
                "[%s] Sentiment margin boost: multiplier=%.2f",
                symbol, multiplier,
            )

        return multiplier

    # ── Main Trading Loop ─────────────────────────────────────────────

    async def run(self):
        """Main event loop for live trading."""

        # 1) Resolve symbols
        self.symbols = await Streamer.resolve_symbols(
            self._market_client, self.cfg.symbols,
        )
        log.info("Resolved %d symbols: %s", len(self.symbols), self.symbols)

        # 1b) Phase D — drop symbols below the 24h-volume gate so we
        # never trade thin alts on real money.  Threshold of 0 disables
        # the gate (test-net / new listings).
        min_vol = float(getattr(self.cfg, "min_24h_volume_usd", 0.0) or 0.0)
        if min_vol > 0:
            kept: list[str] = []
            for sym in self.symbols:
                try:
                    vol = await self.broker.get_24h_volume(sym)
                except Exception as e:
                    log.warning("[volume_gate] %s probe failed (%s) — keeping",
                                sym, e)
                    kept.append(sym)
                    continue
                if vol >= min_vol:
                    kept.append(sym)
                else:
                    log.warning(
                        "[low_volume] %s dropped (24h=$%.0f < min=$%.0f)",
                        sym, vol, min_vol,
                    )
                    try:
                        await self.notifier.kill_switch(
                            f"low volume: {sym} 24h=${vol:,.0f}"
                        )
                    except Exception:
                        pass
            self.symbols = kept

        # 2) Register per-symbol managers in supervisor (config-driven)
        for sym in self.symbols:
            self.supervisor.register_symbol(
                symbol=sym,
                sizing_cfg=self.cfg.sizing_for(sym),
                exit_cfg=self.cfg.exit_for(sym),
                max_concurrent=self.cfg.risk.max_concurrent_positions,
            )

        # 3) Set margin type & leverage per symbol
        for sym in self.symbols:
            try:
                margin_type = self.cfg.margin_type_for(sym)
                if margin_type == "ISOLATED":
                    await self.broker.ensure_isolated_margin(sym)
                await self.broker.set_leverage(sym, self.cfg.leverage_for(sym))
            except Exception as e:
                log.error("[%s] margin/leverage setup failed: %s", sym, e)

        # 4) Create streamer (use market_client if set, e.g. in dry-run mode)
        self.streamer = Streamer(
            self._market_client,
            self.symbols,
            self.timeframes,
            bar_store=self.bar_store,
        )

        # 5) Start WebSocket FIRST in buffer mode (no data gap)
        await self.streamer.start_buffering()

        # 6) Preload historical bars (skips current open bar)
        await self.streamer.preload_history(
            self.symbols,
            self.timeframes,
            limit=self.cfg.execution.preload_bars,
            batch=self.cfg.execution.preload_batch,
        )

        # 7) Initialize global risk + metrics with current equity
        try:
            self._last_equity = await self.broker.balance("USDT")
            self._equity_ts = time.monotonic()
            self.global_risk.set_start_equity(self._last_equity)
            self.metrics.set_start_equity(self._last_equity)
            log.info("Starting equity: %.2f USDT", self._last_equity)
        except Exception as e:
            log.warning("Could not fetch starting equity: %s", e)

        # 8) Restore persisted positions + reconcile orphans from exchange
        self.supervisor.restore_positions()
        await self.supervisor.reconcile_all(strategy_name=self.cfg.name)

        # 9) Prefetch news sentiment (if enabled)
        if self.news_engine:
            await self.news_engine.prefetch_symbols(self.symbols)

        # 10a) Tick streamer for intra-bar exits (trailing/time/USD-target).
        # Server-side STOP_MARKET / TAKE_PROFIT_MARKET handle plain TP/SL;
        # this stream is dedicated to local-only exit rules.  Failure to
        # start does not abort the engine — strategy entries still run.
        try:
            from live.tick_stream import MarkPriceTickStreamer
            self.tick_streamer = MarkPriceTickStreamer(
                client=self._market_client,
                symbols=self.symbols,
                on_tick=self.supervisor.on_tick,
                source="mark",
            )
            await self.tick_streamer.start()
        except Exception as e:
            log.warning("tick streamer disabled: %s", e)
            self.tick_streamer = None

        # 10a-2) Book-ticker streamer powers the entry spread filter
        # (Phase B1).  Lightweight — no callback work, just populates
        # the bid/ask cache the supervisor reads from.
        try:
            from live.tick_stream import MarkPriceTickStreamer as _MPTS

            async def _noop(symbol, price, ts_ms):  # noqa: ARG001
                return None

            self.book_streamer = _MPTS(
                client=self._market_client,
                symbols=self.symbols,
                on_tick=_noop,
                source="book",
            )
            await self.book_streamer.start()
            self.supervisor.book_streamer = self.book_streamer
        except Exception as e:
            log.warning("book streamer disabled: %s — spread filter will allow all", e)
            self.book_streamer = None
            self.supervisor.book_streamer = None

        # 10b) Periodic position reconciliation (drift detection).
        # Real-money safety net.  Compares local intended-qty vs the
        # exchange's truth every N seconds; on drift the configured
        # action (alarm / halt / force_flat) is taken.  Off by default
        # so dry-run stays cheap on rate limits.
        self._reconciliation_task: Optional[asyncio.Task] = None
        if getattr(self.cfg, "reconciliation", None) and self.cfg.reconciliation.enabled:
            self._reconciliation_task = asyncio.create_task(self._reconciliation_loop())
            log.info(
                "Reconciliation loop started (interval=%.1fs, action=%s, tol=%g)",
                self.cfg.reconciliation.interval_seconds,
                self.cfg.reconciliation.action,
                self.cfg.reconciliation.qty_tolerance,
            )

        # 10) Flush buffered WS bars → bar_store + queue (dedup handles overlaps)
        self.streamer.flush_buffer()
        await self.notifier.engine_started()
        log.info(
            "LiveEngine started: %d symbols | tf=%s | sentiment=%s | risk=%s",
            len(self.symbols),
            self.timeframes,
            "ON" if self.news_engine else "OFF",
            "ON",
        )

        try:
            while True:
                bar_data = await self.streamer.get()
                sym = bar_data["s"]
                tf = bar_data["k"]["i"]

                # Only process our configured timeframe
                if tf != self.cfg.timeframe:
                    continue

                # Only process registered symbols
                if sym not in self.symbols:
                    continue

                # ── Global Risk Check ──────────────────────────────────
                equity = await self._refresh_equity()
                self.metrics.update_equity(equity)
                self.metrics.log_daily_summary()
                # Default-deny on entries: if the risk check itself blows up
                # we must NOT silently allow new entries (which is what the
                # previous code did by leaving this name unbound).
                risk_block_entries = True
                try:
                    open_count = len(self.supervisor.open_positions)
                    total_exposure = sum(
                        pos.entry * pos.qty
                        for pos in self.supervisor.open_positions.values()
                    )
                    risk_ok, risk_reason = self.global_risk.check_account_risk(
                        current_equity=equity,
                        total_exposure_usd=total_exposure,
                        open_position_count=open_count,
                    )
                    # If risk blocked AND this symbol has no open position → skip entirely
                    # If risk blocked BUT this symbol HAS a position → let strategy run for exits only
                    has_open_pos = self.supervisor.position_qty(sym) != 0.0
                    if not risk_ok and not has_open_pos:
                        if not getattr(self, '_risk_block_logged', False):
                            log.warning("Global risk block: %s — skipping new entries", risk_reason)
                            self._risk_block_logged = True
                        if "correlated" not in risk_reason:
                            await self.notifier.kill_switch(risk_reason)
                        await self.supervisor.update_symbol(sym)
                        continue
                    # Flag: block new entries but allow exits
                    risk_block_entries = not risk_ok
                    if risk_ok:
                        self._risk_block_logged = False
                except Exception as e:
                    log.warning("Global risk check error: %s — blocking new entries", e)
                    # risk_block_entries already True by default (line above try)

                # ── Build Bar + Context for on_bar strategies ─────────
                bar_dict = bar_data["k"]
                bar = Bar(
                    symbol=sym,
                    timeframe=tf,
                    timestamp_ns=int(bar_dict.get("t", 0)) * 1_000_000,
                    open=float(bar_dict.get("o", 0)),
                    high=float(bar_dict.get("h", 0)),
                    low=float(bar_dict.get("l", 0)),
                    close=float(bar_dict.get("c", 0)),
                    volume=float(bar_dict.get("v", 0)),
                )

                # Feed latest close price to DryBroker so sizing works correctly
                close_price = float(bar_dict.get("c", 0))
                if hasattr(self.broker, "set_price") and close_price > 0:
                    self.broker.set_price(sym, close_price)

                # Current position for this symbol
                current_pos = self.supervisor.position_qty(sym)

                ctx = StrategyContext(
                    symbol=sym,
                    timeframe=tf,
                    bar_store=self.bar_store,
                    position=current_pos,
                    equity=equity,
                    cash=equity,
                    timestamp_ns=bar.timestamp_ns,
                )

                # ── Generate Signal (supports both modes) ─────────────
                decision = self.strategy.on_bar(bar, ctx)

                # Debug: log strategy decision summary
                if decision.has_orders:
                    order_sides = [f"{o.side.value}{'(exit)' if getattr(o,'reduce_only',False) else ''}" for o in decision.orders]
                    log.info("[%s] Strategy decision: %d orders [%s] | features=%s",
                             sym, len(decision.orders), ", ".join(order_sides),
                             {k: f"{v:.4f}" if isinstance(v, float) else v
                              for k, v in (decision.features or {}).items()
                              if k in ("fast_ema", "slow_ema", "histogram", "adx", "atr")})

                # Extract signal: from orders or from legacy signal field
                strategy_sig = None
                has_exit_signal = False
                exit_reason = None

                if decision.has_orders:
                    for order in decision.orders:
                        if getattr(order, 'reduce_only', False):
                            has_exit_signal = True
                            exit_reason = (decision.metadata or {}).get(
                                "exit_reason", "STRATEGY_EXIT"
                            )
                        else:
                            strategy_sig = "+1" if order.side == OrderSide.BUY else "-1"
                elif decision.has_signal:
                    strategy_sig = decision.signal

                # ── Handle strategy exit signals (reduce_only) ────────
                prev_history_len = len(self.supervisor.history)

                if has_exit_signal and self.supervisor.position_qty(sym) != 0.0:
                    exit_type_str = exit_reason or "STRATEGY_EXIT"
                    log.info("[%s] Strategy exit signal: %s", sym, exit_type_str)
                    await self.supervisor.close_position(
                        symbol=sym,
                        strategy_name=self.cfg.name,
                        exit_type=exit_type_str,
                    )

                # ── Use strategy signal directly; sentiment adjusts margin ──
                final_sig = int(strategy_sig) if strategy_sig is not None else None

                if final_sig and not risk_block_entries:
                    # Phase E1 — stale-feed gate.  When the WS hasn't
                    # spoken in a while we don't trust our view of the
                    # market enough to open a fresh position.  Existing
                    # positions stay protected by their server-side
                    # SL/TP; only entries are blocked.
                    max_age = float(getattr(self.cfg.execution,
                                             "max_tick_age_seconds", 0.0) or 0.0)
                    if max_age > 0 and self.streamer is not None:
                        stale = self.streamer.seconds_since_last_message
                        if stale > max_age:
                            log.warning(
                                "[%s] entry blocked — feed stale %.1fs > %.1fs",
                                sym, stale, max_age,
                            )
                            try:
                                await self.notifier.kill_switch(
                                    f"feed stale {stale:.0f}s — blocking entries"
                                )
                            except Exception:
                                pass
                            continue

                    leverage = self.cfg.leverage_for(sym)

                    # Sentiment-driven margin boost
                    margin_multiplier = await self._get_sentiment_margin_multiplier(
                        sym, final_sig,
                    )

                    log.info(
                        "[%s] Opening position: direction=%+d | margin_mult=%.2f",
                        sym, final_sig, margin_multiplier,
                    )

                    opened = await self.supervisor.open_position(
                        symbol=sym,
                        side=final_sig,
                        strategy_name=self.cfg.name,
                        leverage=leverage,
                        timeframe=tf,
                        margin_multiplier=margin_multiplier,
                    )
                    if opened:
                        actual_qty = abs(self.supervisor.position_qty(sym))
                        await self.notifier.position_opened(
                            sym, "LONG" if final_sig > 0 else "SHORT",
                            actual_qty, float(bar_dict.get("c", 0)), leverage,
                        )

                # ── Update existing positions (exit checks) ────────────
                # Only update the current symbol's position manager
                # to prevent bars_held from incrementing once per every
                # symbol's bar event (25 coins = 25x faster expiry).
                await self.supervisor.update_symbol(sym)

                # Record newly closed positions to metrics + notify +
                # feed P&L into global_risk so the daily-loss circuit
                # breaker (Phase A1) and the consecutive-loss cooldown
                # (Phase A3) actually see closed-trade outcomes.  Until
                # this commit ``LiveGlobalRisk.record_pnl`` was never
                # called and the daily counter stayed at 0 forever.
                for pos in self.supervisor.history[prev_history_len:]:
                    # Realised cost accounting: pull exit-leg commission
                    # and funding payments from the broker before the
                    # CSV row is written.  Best-effort — broker errors
                    # leave the values at 0 so we still log gross PnL.
                    await self._apply_realised_costs(pos)

                    self.metrics.record(pos)
                    if pos.exit is not None:
                        pnl = pos.unrealized_pnl(pos.exit)
                        pnl_pct = pos.unrealized_pnl_pct(pos.exit) * 100
                        try:
                            self.global_risk.record_pnl(pnl)
                        except Exception as e:
                            log.warning("global_risk.record_pnl failed: %s", e)
                        await self.notifier.position_closed(
                            pos.symbol, pos.side, pnl, pnl_pct,
                            pos.exit_type or "UNKNOWN",
                        )

        finally:
            recon_task = getattr(self, "_reconciliation_task", None)
            if recon_task:
                recon_task.cancel()
                try:
                    await recon_task
                except (asyncio.CancelledError, Exception):
                    pass
            if self.tick_streamer:
                try:
                    await self.tick_streamer.stop()
                except Exception:
                    pass
            if self.book_streamer:
                try:
                    await self.book_streamer.stop()
                except Exception:
                    pass
            await self.streamer.stop()
            log.info("LiveEngine stopped")

    # ── Shutdown ──────────────────────────────────────────────────────

    async def shutdown(self):
        """Graceful shutdown: close all positions, log final metrics, stop streamer."""
        log.info("Shutting down...")
        try:
            prev_len = len(self.supervisor.history)
            await self.supervisor.force_close_all()
            # Record force-closed positions
            for pos in self.supervisor.history[prev_len:]:
                self.metrics.record(pos)
        except Exception as e:
            log.error("Error during force close: %s", e)

        # Final metrics summary
        log.info("═══ FINAL SESSION METRICS ═══")
        self.metrics._log_summary()
        log.info("Trade log saved to: %s", self.metrics._csv_path)

        if self.tick_streamer:
            try:
                await self.tick_streamer.stop()
            except Exception:
                pass

        if self.book_streamer:
            try:
                await self.book_streamer.stop()
            except Exception:
                pass

        if self.streamer:
            try:
                await self.streamer.stop()
            except Exception:
                pass
