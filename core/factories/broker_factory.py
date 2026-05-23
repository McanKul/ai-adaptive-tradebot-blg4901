"""
core/factories/broker_factory.py
=================================
Factory for creating broker instances (live or dry-run).

Replaces the former if/else broker creation logic.
"""
from __future__ import annotations

import logging
import os
from typing import Optional, Tuple

from Interfaces.IBroker import IBroker

log = logging.getLogger(__name__)


class BrokerFactory:
    """
    Creates broker + client pair based on the requested mode.

    Returns ``(broker, client)`` where *client* is the ``BinanceClient``
    wrapper that must be closed on shutdown (via ``client.close_connection()``).
    """

    @staticmethod
    async def create(
        mode: str,
        cfg,  # LiveConfig
    ) -> Tuple[IBroker, Optional[object]]:
        """
        Create a broker for the given mode.

        Args:
            mode: ``"live"`` or ``"dry"``.
            cfg:  A ``LiveConfig`` instance.

        Returns:
            ``(broker, binance_client_wrapper)``

        Raises:
            ValueError: If *mode* is not ``"live"`` or ``"dry"``.
        """
        from binance import AsyncClient
        from live.binance_client import BinanceClient

        api_key = os.getenv("BINANCE_API_KEY", "")
        api_secret = os.getenv("BINANCE_API_SECRET", "")

        if mode == "dry":
            from live.dry_broker import DryBroker

            raw_client = await AsyncClient.create(
                api_key, api_secret, testnet=cfg.testnet,
            )
            client = BinanceClient(raw_client)
            broker = DryBroker(initial_balance=10_000.0, market_client=client)
            log.info("[DRY-RUN] Real client for market data, DryBroker for orders")
            return broker, client

        if mode == "live":
            from live.broker_binance import BinanceBroker
            from live.rate_limiter import AsyncRateLimiter

            if not api_key:
                log.warning("BINANCE_API_KEY not set — client may fail or be read-only")
            raw_client = await AsyncClient.create(
                api_key, api_secret, testnet=cfg.testnet,
            )
            client = BinanceClient(raw_client)
            rate_limiter = AsyncRateLimiter(
                max_per_minute=cfg.rate_limit.requests_per_minute,
            )
            broker = BinanceBroker(
                client,
                rate_limiter=rate_limiter,
                exchange_info_ttl=cfg.rate_limit.exchange_info_ttl_sec,
            )
            return broker, client

        raise ValueError(
            f"Unknown broker mode: '{mode}'. Must be 'live' or 'dry'."
        )
