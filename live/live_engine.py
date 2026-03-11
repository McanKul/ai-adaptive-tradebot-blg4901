"""
live/live_engine.py
===================
Live trading engine that coordinates strategy signals with optional news sentiment.

Signal combination logic:
- Both strategy + sentiment agree BUY  → LONG
- Both strategy + sentiment agree SELL → SHORT
- Mixed or neutral signals             → NO ACTION
- No news engine configured            → strategy signal only
"""
from __future__ import annotations

import asyncio
from typing import Optional

from utils.bar_store import BarStore
from Interfaces.IBroker import IBroker
from Interfaces.INewsSource import INewsSource
from Interfaces.ISentimentAnalyzer import ISentimentAnalyzer
from Interfaces.ISignalCombiner import ISignalCombiner
from live.position_manager import PositionManager
from live.streamer import Streamer
from utils.logger import setup_logger

try:
    from news.news_engine import NewsEngine
except ImportError:
    NewsEngine = None

log = setup_logger("LiveEngine")


def load_strategy_instance(strategy_cls, config, bar_store):
    """Load strategy instance with configuration."""
    return strategy_cls(bar_store=bar_store, **config.get("params", {}))


class LiveEngine:
    """
    Live trading engine that coordinates strategy signals with optional news sentiment.

    When news components are provided, signals are combined:
    - Both BUY (strategy + sentiment) → LONG
    - Both SELL (strategy + sentiment) → SHORT
    - Mixed signals → NO ACTION

    When news components are NOT provided, raw strategy signals drive positions.
    """

    def __init__(
        self,
        config,
        broker: IBroker,
        strategy_cls,
        news_source: Optional[INewsSource] = None,
        sentiment_analyzer: Optional[ISentimentAnalyzer] = None,
        signal_combiner: Optional[ISignalCombiner] = None,
        news_refresh_interval: int = 300,
    ):
        """
        Initialize the live engine.

        Args:
            config: Strategy configuration dict or list
            broker: Broker implementation for trading
            strategy_cls: Strategy class to instantiate
            news_source: Optional news source for sentiment analysis
            sentiment_analyzer: Optional sentiment analyzer (requires news_source)
            signal_combiner: Optional signal combiner (requires news_source + analyzer)
            news_refresh_interval: Seconds between news sentiment refreshes
        """
        self.cfg = config
        self.broker = broker
        self.bar_store = BarStore()

        # ── Configure Strategies ──────────────────────────────────────────
        self.strategies = []
        strategy_configs = [self.cfg] if isinstance(self.cfg, dict) else self.cfg

        for scfg in strategy_configs:
            instance = load_strategy_instance(strategy_cls, scfg, self.bar_store)
            self.strategies.append({**scfg, "instance": instance})

        self.pos_mgr = PositionManager(
            self.broker,
            base_capital=config.get("base_capital", 10.0) if isinstance(config, dict) else 10.0,
            max_concurrent=config.get("max_concurrent", 1) if isinstance(config, dict) else 1,
        )

        # ── Timeframes & Streamer ─────────────────────────────────────────
        self.timeframes = list(set(s.get("timeframe", "1m") for s in self.strategies))
        self.streamer = None
        self.symbols = []

        # ── News Sentiment Integration ────────────────────────────────────
        self.news_engine = None
        self.signal_combiner = signal_combiner

        if news_source and sentiment_analyzer:
            if NewsEngine:
                self.news_engine = NewsEngine(
                    news_source=news_source,
                    sentiment_analyzer=sentiment_analyzer,
                    refresh_interval=news_refresh_interval,
                )
                log.info("News sentiment integration enabled (refresh=%ds)", news_refresh_interval)
            else:
                log.warning("NewsEngine module not available — running without sentiment")

        if self.news_engine and not self.signal_combiner:
            log.warning(
                "News engine enabled but no signal combiner provided — "
                "sentiment will be logged but NOT used for trading decisions"
            )

    # ── Signal Combination ────────────────────────────────────────────────

    async def _get_combined_signal(
        self, symbol: str, strategy_signal: Optional[str]
    ) -> Optional[int]:
        """
        Combine strategy signal with news sentiment.

        Returns:
            +1 for long, -1 for short, None for no action.
        """
        # Convert string signal to int for combiner
        if strategy_signal is not None:
            strat_int = int(strategy_signal)
        else:
            strat_int = None

        # No news engine → pass through raw strategy signal
        if not self.news_engine:
            return strat_int

        # Fetch sentiment score (cached or fresh)
        sentiment_score = await self.news_engine.get_sentiment(symbol)
        log.info(
            "Symbol %s: strategy_signal=%s, sentiment=%.2f",
            symbol, strategy_signal, sentiment_score,
        )

        # No combiner → log only, pass through raw strategy signal
        if not self.signal_combiner:
            return strat_int

        # Combine strategy + sentiment
        combined = self.signal_combiner.combine(strat_int, sentiment_score)

        if combined != strat_int:
            log.info("Signal modified by sentiment: %s → %s", strat_int, combined)

        return combined

    # ── Main Trading Loop ─────────────────────────────────────────────────

    async def run(self):
        """Main event loop for live trading."""

        # 1) Resolve Symbols
        all_coins = []
        for s in self.strategies:
            all_coins.extend(s.get("coins", []))

        self.symbols = await Streamer.resolve_symbols(
            self.broker.client, list(set(all_coins))
        )

        # 2) Register per-symbol managers in supervisor
        for sym in self.symbols:
            self.supervisor.register_symbol(
                symbol=sym,
                sizing_cfg=self.cfg.sizing_for(sym),
                exit_cfg=self.cfg.exit_for(sym),
                max_concurrent=1,  # 1 position per symbol
            )

        # 3) Create streamer
        self.streamer = Streamer(
            self.broker.client,
            self.symbols,
            self.timeframes,
            bar_store=self.bar_store,
        )

        # 4) Preload history
        await self.streamer.preload_history(
            self.symbols, self.timeframes,
            limit=100, batch=10,
        )

        # 4) Prefetch news sentiment (if enabled)
        if self.news_engine:
            await self.news_engine.prefetch_symbols(self.symbols)

        # 5) Start Live Stream
        await self.streamer.start()
        log.info(
            "Live Engine Started: %d symbols | tf=%s | sentiment=%s",
            len(self.symbols),
            self.timeframes,
            "enabled" if self.news_engine else "disabled",
        )

        try:
            while True:
                bar_data = await self.streamer.get()
                sym = bar_data["s"]
                tf = bar_data["k"]["i"]

                for s in self.strategies:
                    # Check if this strategy cares about this symbol + timeframe
                    if sym not in s.get("coins", []) or tf != s.get("timeframe", "1m"):
                        continue

                    inst = s["instance"]
                    strategy_sig = inst.generate_signal(sym)

                    # Combine with news sentiment (pass-through if no news engine)
                    final_sig = await self._get_combined_signal(sym, strategy_sig)

                    if final_sig:
                        direction = 1 if final_sig == 1 else -1
                        exec_params = s.get("execution", {})

                        log.info(
                            "Opening position: %s direction=%d (strategy=%s, combined=%s)",
                            sym, direction, strategy_sig, final_sig,
                        )

                        await self.pos_mgr.open_position(
                            sym,
                            direction,
                            s.get("name", "UnknownStrategy"),
                            leverage=exec_params.get("leverage", 1),
                            sl_pct=exec_params.get("sl_pct", 1.0),
                            tp_pct=exec_params.get("tp_pct", 1.0),
                            expire_sec=exec_params.get("expire_sec", 3600),
                            timeframes=tf,
                        )
                    else:
                        await self.pos_mgr.update_all()
        finally:
            await self.streamer.stop()
            log.info("Live Engine stopped")
