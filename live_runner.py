"""
live_runner.py
==============
Entry-point for live trading.

Usage:
    python live_runner.py                              # uses example_live_config.yaml
    python live_runner.py --config my_config.yaml      # custom config
"""
import asyncio
import argparse
import os
import sys

from binance.client import AsyncClient

from live.live_config import LiveConfig
from live.live_engine import LiveEngine
from live.broker_binance import BinanceBroker
from live.binance_client import BinanceClient
from live.rate_limiter import AsyncRateLimiter


async def main(config_path: str):
    # 1) Load config
    cfg = LiveConfig.from_yaml(config_path)
    print(f"Config loaded: {config_path}")
    print(f"  strategy : {cfg.strategy_class}")
    print(f"  symbols  : {cfg.symbols}")
    print(f"  timeframe: {cfg.timeframe}")
    print(f"  leverage : {cfg.sizing.leverage}x")
    print(f"  sizing   : {cfg.sizing.mode} (margin={cfg.sizing.margin_usd} USD)")
    print(f"  TP       : {cfg.exit.take_profit_pct}  SL: {cfg.exit.stop_loss_pct}")
    print(f"  risk     : max_pos={cfg.risk.max_concurrent_positions}, "
          f"max_daily_loss={cfg.risk.max_daily_loss}")

    # 2) API keys
    api_key = os.getenv("BINANCE_API_KEY", "")
    api_secret = os.getenv("BINANCE_API_SECRET", "")

    if not api_key:
        print("WARNING: BINANCE_API_KEY not set. Client may fail or be read-only.")

    testnet = cfg.testnet
    raw_client = await AsyncClient.create(api_key, api_secret, testnet=testnet)
    client = BinanceClient(raw_client)
    rl = AsyncRateLimiter(max_per_minute=cfg.rate_limit.requests_per_minute)
    broker = BinanceBroker(client, rate_limiter=rl,
                           exchange_info_ttl=cfg.rate_limit.exchange_info_ttl_sec)

    # 3) Start engine
    engine = LiveEngine(cfg, broker)

    try:
        await engine.run()
    except KeyboardInterrupt:
        print("\nStopping (KeyboardInterrupt)...")
    except Exception as e:
        print(f"Fatal error: {e}")
        import traceback
        traceback.print_exc()
    finally:
        # Force close open positions on shutdown
        try:
            await engine.pos_mgr.force_close_all()
        except Exception:
            pass
        await client.close_connection()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Live trading runner")
    parser.add_argument(
        "--config",
        type=str,
        default="example_live_config.yaml",
        help="Path to YAML config file",
    )
    args = parser.parse_args()

    if not os.path.exists(args.config):
        print(f"Config file not found: {args.config}")
        sys.exit(1)

    asyncio.run(main(args.config))
