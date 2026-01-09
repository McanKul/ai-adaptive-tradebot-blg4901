import sys, os, types
from unittest.mock import AsyncMock, MagicMock
import asyncio

# Mock binance for Streamer init
binance_mod = types.SimpleNamespace()
binance_mod.BinanceSocketManager = MagicMock(return_value=MagicMock())
sys.modules["binance"] = binance_mod

# Ensure project root on path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from live.streamer import Streamer
from utils.bar_store import BarStore


def test_resolve_symbols_all_usdt():
    client = AsyncMock()
    client.futures_exchange_info.return_value = {
        "symbols": [
            {"symbol": "BTCUSDT", "quoteAsset": "USDT", "status": "TRADING"},
            {"symbol": "ETHBUSD", "quoteAsset": "BUSD", "status": "TRADING"},
        ]
    }
    syms = asyncio.run(Streamer.resolve_symbols(client, "ALL_USDT"))
    assert syms == ["BTCUSDT"]


def test_resolve_symbols_list_normalizes():
    client = AsyncMock()
    syms = asyncio.run(Streamer.resolve_symbols(client, ["eth/usdt", "BTCUSDT"]))
    assert syms == ["ETHUSDT", "BTCUSDT"]


def test_preload_history_adds_bars():
    # Setup client with klines
    client = AsyncMock()
    client.futures_klines = AsyncMock(return_value=[
        [1600000000000, "100", "110", "90", "105", "1000", 1600000059999, "0", 0, "0", "0", "0"],
        [1600000060000, "105", "115", "95", "110", "900", 1600000119999, "0", 0, "0", "0", "0"],
    ])
    # Provide raw_client for BSM in Streamer init
    client.raw_client = MagicMock()

    bs = BarStore()
    st = Streamer(client, ["BTCUSDT"], ["1m"], bs)
    asyncio.run(st.preload_history(["BTCUSDT"], ["1m"], limit=2, batch=10))
    data = bs.get_ohlcv("BTCUSDT", "1m")
    assert data["close"][-2:] == [105.0, 110.0]
