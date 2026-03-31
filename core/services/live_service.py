"""
core/services/live_service.py
==============================
Thin orchestration wrapper for live / dry-run trading.

Composes BrokerFactory + StrategyFactory + NewsFactory + LiveEngine.
Composes BrokerFactory + StrategyFactory + NewsFactory + LiveEngine.
"""
from __future__ import annotations

import asyncio
import logging
import signal
import sys

from core.factories.strategy_factory import StrategyFactory
from core.factories.broker_factory import BrokerFactory
from core.factories.news_factory import NewsFactory
from live.live_config import LiveConfig
from live.live_engine import LiveEngine
from live.global_risk import LiveGlobalRisk
from utils.logger import setup_logger

log = setup_logger("LiveService")


class LiveService:
    """Run live or dry-run trading using factory-wired components."""

    async def run(self, config_path: str, dry_run: bool = False) -> None:
        """
        Load config, create components via factories, and run the engine.

        Args:
            config_path: Path to YAML or JSON config file.
            dry_run: If ``True``, use ``DryBroker`` (paper trading).
        """
        # 1) Load config
        if config_path.endswith(".json"):
            cfg = LiveConfig.from_json(config_path)
        else:
            cfg = LiveConfig.from_yaml(config_path)

        mode_label = "DRY-RUN" if dry_run else "LIVE"
        log.info(
            "[%s] Config loaded: %s | symbols=%s | strategy=%s",
            mode_label, cfg.name, cfg.symbols, cfg.strategy_class,
        )

        # 2) Resolve strategy class
        strategy_cls = StrategyFactory.resolve_class(cfg.strategy_class)
        log.info("Strategy class resolved: %s", strategy_cls.__name__)

        # 3) Create broker via factory
        broker_mode = "dry" if dry_run else "live"
        broker, client = await BrokerFactory.create(broker_mode, cfg)
        # For dry-run the client is also used as market_client for WS
        market_client = client if dry_run else None

        # 4) Global risk
        global_risk = LiveGlobalRisk(cfg.global_risk)

        # 5) News sentiment
        news = NewsFactory.create(cfg)

        # 6) Create engine
        engine = LiveEngine(
            cfg=cfg,
            broker=broker,
            strategy_cls=strategy_cls,
            global_risk=global_risk,
            news_source=news.news_source,
            sentiment_analyzer=news.sentiment_analyzer,
            signal_combiner=news.signal_combiner,
            market_client=market_client,
        )

        # 7) Graceful shutdown handling
        loop = asyncio.get_running_loop()
        shutdown_event = asyncio.Event()

        def _signal_handler():
            log.info("Shutdown signal received")
            shutdown_event.set()

        for sig in (signal.SIGINT, signal.SIGTERM):
            try:
                loop.add_signal_handler(sig, _signal_handler)
            except NotImplementedError:
                # Windows doesn't support add_signal_handler
                pass

        try:
            engine_task = asyncio.create_task(engine.run())
            shutdown_wait = asyncio.create_task(shutdown_event.wait())

            done, pending = await asyncio.wait(
                [engine_task, shutdown_wait],
                return_when=asyncio.FIRST_COMPLETED,
            )
            for task in pending:
                task.cancel()
                try:
                    await task
                except (asyncio.CancelledError, Exception):
                    pass

        except KeyboardInterrupt:
            log.info("Stopping (KeyboardInterrupt)...")
        except Exception as e:
            log.error("Fatal error: %s", e)
            import traceback
            traceback.print_exc()
        finally:
            await engine.shutdown()
            if client:
                await client.close_connection()
