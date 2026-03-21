"""
live/live_engine.py
===================
Live trading engine that coordinates:
- Strategy signals (via IStrategy)
- News sentiment analysis (via NewsEngine + SignalCombiner)
- Position management (via LiveSupervisor — per-symbol PositionManager)
- Account-level risk (via LiveGlobalRisk)

Signal combination logic:
- Both strategy + sentiment agree BUY  → LONG
- Both strategy + sentiment agree SELL → SHORT
- Mixed or neutral signals             → NO ACTION
- No news engine configured            → strategy signal only
"""
from __future__ import annotations

import time
from typing import Optional, Type

from utils.bar_store import BarStore
from Interfaces.IBroker import IBroker
from Interfaces.INewsSource import INewsSource
from Interfaces.ISentimentAnalyzer import ISentimentAnalyzer
from Interfaces.ISignalCombiner import ISignalCombiner
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
        strategy_cls: Type,
        global_risk: LiveGlobalRisk,
        news_source: Optional[INewsSource] = None,
        sentiment_analyzer: Optional[ISentimentAnalyzer] = None,
        signal_combiner: Optional[ISignalCombiner] = None,
        market_client=None,  # Real client for WebSocket/preload in dry-run mode
    ):
        self.cfg = cfg
        self.broker = broker
        self.global_risk = global_risk
        self.bar_store = BarStore()
        # Use dedicated market client if provided (e.g. dry-run needs real WS)
        self._market_client = market_client or broker.client

        # ── Strategy Instance ──────────────────────────────────────────
        self.strategy = strategy_cls(
            bar_store=self.bar_store,
            **cfg.strategy_params,
        )

        # ── LiveSupervisor (per-symbol PositionManager) ────────────────
        self.supervisor = LiveSupervisor(broker)

        # ── Streamer ───────────────────────────────────────────────────
        self.timeframes = [cfg.timeframe]
        self.streamer: Optional[Streamer] = None
        self.symbols: list[str] = []

        # ── Equity Cache (avoid API call every bar) ────────────────────
        self._last_equity: float = 10_000.0
        self._equity_ts: float = 0.0
        self._EQUITY_REFRESH_SEC: float = 30.0

        # ── Live Metrics ─────────────────────────────────────────────
        self.metrics = LiveMetrics(csv_path="logs/live_trades.csv")

        # ── Notifications ────────────────────────────────────────────
        self.notifier = TelegramNotifier()

        # ── News Sentiment ─────────────────────────────────────────────
        self.news_engine = None
        self.signal_combiner = signal_combiner

        if news_source and sentiment_analyzer and NewsEngine:
            self.news_engine = NewsEngine(
                news_source=news_source,
                sentiment_analyzer=sentiment_analyzer,
                refresh_interval=cfg.news.refresh_interval,
                news_limit=cfg.news.news_limit,
            )
            log.info("News sentiment enabled (refresh=%ds)", cfg.news.refresh_interval)

        if self.news_engine and not self.signal_combiner:
            log.warning(
                "News engine enabled but no signal combiner — "
                "sentiment will be logged but NOT used for trading"
            )

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

    # ── Signal Combination ────────────────────────────────────────────

    async def _get_combined_signal(
        self, symbol: str, strategy_signal: Optional[str]
    ) -> Optional[int]:
        """
        Combine strategy signal with news sentiment.

        Returns:
            +1 for long, -1 for short, None for no action.
        """
        strat_int = int(strategy_signal) if strategy_signal is not None else None

        # No news engine → pass through raw strategy signal
        if not self.news_engine:
            return strat_int

        # Fetch sentiment score (cached or fresh)
        sentiment_score = await self.news_engine.get_sentiment(symbol)
        log.info(
            "[%s] strategy_signal=%s, sentiment=%.2f",
            symbol, strategy_signal, sentiment_score,
        )

        # No combiner → log only, pass through raw strategy signal
        if not self.signal_combiner:
            return strat_int

        # Combine strategy + sentiment
        combined = self.signal_combiner.combine(strat_int, sentiment_score)

        if combined != strat_int:
            log.info("[%s] Signal modified by sentiment: %s → %s", symbol, strat_int, combined)

        return combined

    # ── Main Trading Loop ─────────────────────────────────────────────

    async def run(self):
        """Main event loop for live trading."""

        # 1) Resolve symbols
        self.symbols = await Streamer.resolve_symbols(
            self._market_client, self.cfg.symbols,
        )
        log.info("Resolved %d symbols: %s", len(self.symbols), self.symbols)

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
                    if not risk_ok:
                        log.warning("Global risk block: %s — skipping signals", risk_reason)
                        await self.notifier.kill_switch(risk_reason)
                        await self.supervisor.update_all()
                        continue
                except Exception as e:
                    log.warning("Global risk check error: %s", e)

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

                # ── Combine with Sentiment ─────────────────────────────
                final_sig = await self._get_combined_signal(sym, strategy_sig)

                if final_sig:
                    leverage = self.cfg.leverage_for(sym)

                    log.info(
                        "[%s] Opening position: direction=%+d (strategy=%s, final=%s)",
                        sym, final_sig, strategy_sig, final_sig,
                    )

                    opened = await self.supervisor.open_position(
                        symbol=sym,
                        side=final_sig,
                        strategy_name=self.cfg.name,
                        leverage=leverage,
                        timeframe=tf,
                    )
                    if opened:
                        actual_qty = abs(self.supervisor.position_qty(sym))
                        await self.notifier.position_opened(
                            sym, "LONG" if final_sig > 0 else "SHORT",
                            actual_qty, float(bar_dict.get("c", 0)), leverage,
                        )

                # ── Update existing positions (exit checks) ────────────
                await self.supervisor.update_all()

                # Record newly closed positions to metrics + notify
                for pos in self.supervisor.history[prev_history_len:]:
                    self.metrics.record(pos)
                    if pos.exit is not None:
                        pnl = pos.unrealized_pnl(pos.exit)
                        pnl_pct = pos.unrealized_pnl_pct(pos.exit) * 100
                        await self.notifier.position_closed(
                            pos.symbol, pos.side, pnl, pnl_pct,
                            pos.exit_type or "UNKNOWN",
                        )

        finally:
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

        if self.streamer:
            try:
                await self.streamer.stop()
            except Exception:
                pass
