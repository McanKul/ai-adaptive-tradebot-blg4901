"""
tests/test_broker_factory.py
=============================
Tests for BrokerFactory: live vs dry mode, invalid mode error.

NOTE: test_broker_binance.py replaces sys.modules["binance"] with a
SimpleNamespace at module scope.  To survive that, we avoid importing
``binance`` at the top level and do all patching inside each test.
"""
import os
import sys
import unittest
from unittest.mock import AsyncMock, patch, MagicMock

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from core.factories.broker_factory import BrokerFactory


def _make_mock_cfg():
    """Create a minimal mock LiveConfig for BrokerFactory."""
    cfg = MagicMock()
    cfg.testnet = True
    cfg.rate_limit.requests_per_minute = 1000
    cfg.rate_limit.exchange_info_ttl_sec = 300
    return cfg


def _patch_async_client_create():
    """Return a context-manager that patches AsyncClient.create regardless of module state."""
    import binance as _binance_mod

    # test_broker_binance.py may have replaced binance with a SimpleNamespace.
    # If AsyncClient doesn't exist there, create a placeholder for the patch.
    if not hasattr(_binance_mod, "AsyncClient"):
        _binance_mod.AsyncClient = type("AsyncClient", (), {"create": None})

    return patch.object(_binance_mod.AsyncClient, "create", new_callable=AsyncMock)


class TestBrokerFactory(unittest.IsolatedAsyncioTestCase):

    async def test_create_dry_returns_dry_broker(self):
        with _patch_async_client_create() as mock_create:
            mock_create.return_value = AsyncMock()

            cfg = _make_mock_cfg()
            broker, client = await BrokerFactory.create("dry", cfg)

            from live.dry_broker import DryBroker
            self.assertIsInstance(broker, DryBroker)
            self.assertIsNotNone(client)
            mock_create.assert_awaited_once()

    async def test_create_live_returns_binance_broker(self):
        with _patch_async_client_create() as mock_create:
            mock_create.return_value = AsyncMock()

            cfg = _make_mock_cfg()
            broker, client = await BrokerFactory.create("live", cfg)

            from live.broker_binance import BinanceBroker
            self.assertIsInstance(broker, BinanceBroker)
            self.assertIsNotNone(client)
            mock_create.assert_awaited_once()

    async def test_create_invalid_mode_raises(self):
        cfg = _make_mock_cfg()
        with self.assertRaises(ValueError) as ctx:
            await BrokerFactory.create("invalid_mode", cfg)
        self.assertIn("Unknown broker mode", str(ctx.exception))


if __name__ == "__main__":
    unittest.main()
