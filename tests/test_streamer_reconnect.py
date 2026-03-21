"""Tests for Streamer auto-reconnect, heartbeat timeout, and gap recovery."""
import sys, os, asyncio, types
from unittest.mock import AsyncMock, MagicMock, patch

# Mock binance — must set before importing streamer
binance_mod = types.SimpleNamespace()
binance_mod.BinanceSocketManager = MagicMock(return_value=MagicMock())
sys.modules["binance"] = binance_mod

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from live.streamer import Streamer
from utils.bar_store import BarStore


def test_heartbeat_timeout_triggers_reconnect():
    """If no WS message for HEARTBEAT_TIMEOUT, ConnectionError is raised."""
    client = AsyncMock()
    client.raw_client = MagicMock()
    bs = BarStore()
    st = Streamer(client, ["BTCUSDT"], ["1m"], bs)

    # Set very short timeout for testing
    st._HEARTBEAT_TIMEOUT = 0.1
    st._stopped = False

    # Mock socket that never yields data (simulates silent disconnect)
    mock_stream = AsyncMock()
    mock_stream.recv = AsyncMock(side_effect=asyncio.TimeoutError())

    mock_sock = MagicMock()
    mock_sock.__aenter__ = AsyncMock(return_value=mock_stream)
    mock_sock.__aexit__ = AsyncMock(return_value=False)
    st.bsm.futures_multiplex_socket = MagicMock(return_value=mock_sock)

    # _stream_klines should raise ConnectionError on heartbeat timeout
    with __import__('pytest').raises(ConnectionError, match="Heartbeat timeout"):
        asyncio.run(st._stream_klines())


def test_gap_recovery_calls_preload():
    """After reconnect, _recover_gap fetches recent bars via REST."""
    client = AsyncMock()
    client.raw_client = MagicMock()
    client.futures_klines = AsyncMock(return_value=[])
    bs = BarStore()
    st = Streamer(client, ["BTCUSDT"], ["1m"], bs)

    asyncio.run(st._recover_gap())
    # Should have called futures_klines for gap recovery
    client.futures_klines.assert_awaited()


def test_reconnect_count_increments():
    """Verify reconnect counter increments after each retry."""
    client = AsyncMock()
    client.raw_client = MagicMock()
    client.futures_klines = AsyncMock(return_value=[])
    bs = BarStore()
    st = Streamer(client, ["BTCUSDT"], ["1m"], bs)

    assert st.reconnect_count == 0

    # Simulate: _stream_klines fails twice, then we stop
    call_count = 0

    async def fake_stream():
        nonlocal call_count
        call_count += 1
        if call_count <= 2:
            raise ConnectionError("test disconnect")
        st._stopped = True  # stop after 2 reconnects

    st._stream_klines = fake_stream
    st._INITIAL_BACKOFF = 0.01  # fast for tests
    st._stopped = False

    asyncio.run(st._run_with_reconnect())
    assert st.reconnect_count == 2


def test_buffer_mode_collects_events():
    """Events during buffering should accumulate, not go to queue."""
    client = AsyncMock()
    client.raw_client = MagicMock()
    bs = BarStore()
    st = Streamer(client, ["BTCUSDT"], ["1m"], bs)

    st._buffering = True
    event = {"s": "BTCUSDT", "k": {"i": "1m", "t": 1000, "o": "1", "h": "2",
             "l": "0.5", "c": "1.5", "v": "100", "x": True}}
    st._buffer.append(event)

    assert st.queue.empty()
    assert len(st._buffer) == 1

    # Flush should move to bar_store + queue
    st.flush_buffer()
    assert len(st._buffer) == 0
    assert not st.queue.empty()
    assert st._buffering is False
