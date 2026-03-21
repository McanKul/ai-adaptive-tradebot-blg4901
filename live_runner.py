"""
live_runner.py
==============
Entry-point for live trading.

Loads LiveConfig from YAML, resolves the strategy class dynamically,
wires up news sentiment, global risk, and starts the LiveEngine.

Usage:
    python live_runner.py                              # uses example_live_config.yaml
    python live_runner.py --config my_config.yaml      # custom config
"""
import asyncio
import argparse
import importlib
import os
import signal
import sys

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

from binance import AsyncClient

from live.live_config import LiveConfig
from live.live_engine import LiveEngine
from live.broker_binance import BinanceBroker
from live.binance_client import BinanceClient
from live.rate_limiter import AsyncRateLimiter
from live.global_risk import LiveGlobalRisk
from utils.logger import setup_logger

log = setup_logger("LiveRunner")

# ── Strategy Registry ─────────────────────────────────────────────────
STRATEGY_MAP = {
    "RSIThreshold": "Strategy.RSIThreshold",
    "DonchianATRVolTarget": "Strategy.DonchianATRVolTarget",
}


def resolve_strategy_class(class_name: str):
    """Resolve strategy class by name from STRATEGY_MAP or dotted path."""
    module_path = STRATEGY_MAP.get(class_name, class_name)

    # If it's a known name, import module and get the Strategy class
    if class_name in STRATEGY_MAP:
        mod = importlib.import_module(module_path)
        return getattr(mod, "Strategy")

    # Otherwise treat as dotted path: "some.module.ClassName"
    parts = module_path.rsplit(".", 1)
    if len(parts) == 2:
        mod = importlib.import_module(parts[0])
        return getattr(mod, parts[1])

    raise ValueError(f"Cannot resolve strategy class: {class_name}")


# ── News Sentiment Setup ──────────────────────────────────────────────
def create_news_components(cfg: LiveConfig):
    """Create news source, sentiment analyzer, and signal combiner if configured."""
    news_source = None
    sentiment_analyzer = None
    signal_combiner = None

    news_cfg = cfg.news

    if not news_cfg.enabled:
        log.info("News sentiment disabled by config")
        return news_source, sentiment_analyzer, signal_combiner

    # Resolve sentiment provider
    provider = news_cfg.sentiment_provider.lower()

    if provider == "gemini":
        api_key = news_cfg.api_key or os.getenv("GOOGLE_API_KEY", "")
        if not api_key:
            log.warning("Gemini sentiment enabled but no API key found (config or GOOGLE_API_KEY env)")
            return news_source, sentiment_analyzer, signal_combiner
        from news.ddg_news_source import DDGNewsSource
        from news.gemini_sentiment import GeminiSentimentAnalyzer
        news_source = DDGNewsSource()
        sentiment_analyzer = GeminiSentimentAnalyzer(api_key=api_key)

    elif provider == "openai":
        api_key = news_cfg.api_key or os.getenv("OPENAI_API_KEY", "")
        if not api_key:
            log.warning("OpenAI sentiment enabled but no API key found (config or OPENAI_API_KEY env)")
            return news_source, sentiment_analyzer, signal_combiner
        from news.ddg_news_source import DDGNewsSource
        from news.openai_sentiment import OpenAISentimentAnalyzer
        news_source = DDGNewsSource()
        sentiment_analyzer = OpenAISentimentAnalyzer(api_key=api_key)

    else:
        log.warning("Unknown sentiment provider: %s", provider)
        return news_source, sentiment_analyzer, signal_combiner

    # Signal combiner
    from news.signal_combiner import BinarySignalCombiner
    signal_combiner = BinarySignalCombiner(
        buy_threshold=news_cfg.buy_threshold,
        sell_threshold=news_cfg.sell_threshold,
    )

    log.info(
        "News sentiment enabled: provider=%s, refresh=%ds, thresholds=(buy=%.2f, sell=%.2f)",
        provider, news_cfg.refresh_interval,
        news_cfg.buy_threshold, news_cfg.sell_threshold,
    )

    return news_source, sentiment_analyzer, signal_combiner


# ── Main ──────────────────────────────────────────────────────────────
async def main(config_path: str, dry_run: bool = False):
    # 1) Load config
    if config_path.endswith(".json"):
        cfg = LiveConfig.from_json(config_path)
    else:
        cfg = LiveConfig.from_yaml(config_path)

    mode = "DRY-RUN" if dry_run else "LIVE"
    log.info("[%s] Config loaded: %s | symbols=%s | strategy=%s",
             mode, cfg.name, cfg.symbols, cfg.strategy_class)

    # 2) Resolve strategy class
    strategy_cls = resolve_strategy_class(cfg.strategy_class)
    log.info("Strategy class resolved: %s", strategy_cls.__name__)

    # 3) Create broker (real or paper)
    client = None
    if dry_run:
        from live.dry_broker import DryBroker
        broker = DryBroker(initial_balance=10_000.0)
    else:
        api_key = os.getenv("BINANCE_API_KEY", "")
        api_secret = os.getenv("BINANCE_API_SECRET", "")
        if not api_key:
            log.warning("BINANCE_API_KEY not set — client may fail or be read-only")
        raw_client = await AsyncClient.create(api_key, api_secret, testnet=cfg.testnet)
        client = BinanceClient(raw_client)
        rate_limiter = AsyncRateLimiter(
            max_per_minute=cfg.rate_limit.requests_per_minute,
        )
        broker = BinanceBroker(
            client,
            rate_limiter=rate_limiter,
            exchange_info_ttl=cfg.rate_limit.exchange_info_ttl_sec,
        )

    # 4) Global risk manager
    global_risk = LiveGlobalRisk(cfg.global_risk)

    # 5) News sentiment components
    news_source, sentiment_analyzer, signal_combiner = create_news_components(cfg)

    # 6) Create & run engine
    engine = LiveEngine(
        cfg=cfg,
        broker=broker,
        strategy_cls=strategy_cls,
        global_risk=global_risk,
        news_source=news_source,
        sentiment_analyzer=sentiment_analyzer,
        signal_combiner=signal_combiner,
    )

    # 7) Register SIGTERM/SIGINT for graceful shutdown (Docker/systemd)
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

        # Wait for either engine to finish or shutdown signal
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


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Live trading runner")
    parser.add_argument(
        "--config",
        type=str,
        default="example_live_config.yaml",
        help="Path to YAML/JSON config file",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Paper trading mode — no real orders sent",
    )
    args = parser.parse_args()

    if not os.path.exists(args.config):
        print(f"Config file not found: {args.config}")
        sys.exit(1)

    asyncio.run(main(args.config, dry_run=args.dry_run))
