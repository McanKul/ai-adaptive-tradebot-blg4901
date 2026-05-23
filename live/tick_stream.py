"""
live/tick_stream.py
===================
Mark-price tick streamer for intra-bar exit checks.

The kline streamer (``live/streamer.py``) only emits **closed** bars
(``x: True``).  That is enough for strategy entry decisions but kills
intra-bar exit logic that the backtest engine supports — trailing
stops, time-based exits, USD-target exits, and strategy-driven exits
all wait for the next bar close before firing in live mode, which on
a 15m timeframe is up to a 15-minute reaction lag.

Server-side ``STOP_MARKET`` / ``TAKE_PROFIT_MARKET`` orders cover the
flat TP/SL case (Binance triggers on mark price, network-independent).
This streamer covers the rest: every mark-price update (≈1 per
second) is forwarded to a callback so ``PositionManager`` can run its
trailing/max-bars/strategy exits at tick granularity.

Subscribes to ``<symbol>@markPrice@1s`` and ``<symbol>@bookTicker``
streams; mark price is the default (matches what the exchange uses
for liquidation and stop triggers — keeps live and server-side
consistent).
"""
from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass
from typing import Awaitable, Callable, List, Optional

from binance import BinanceSocketManager

from Interfaces.IClient import IClient
from utils.logger import setup_logger

log = setup_logger("TickStream")


TickCallback = Callable[[str, float, int], Awaitable[None]]
"""Coroutine callback ``(symbol, price, ts_ms) -> None`` invoked per tick."""


@dataclass
class _TickStats:
    received: int = 0
    forwarded: int = 0
    last_msg_ts: float = 0.0
    reconnects: int = 0


class MarkPriceTickStreamer:
    """Subscribe to ``markPrice@1s`` for N symbols and forward to a callback.

    Args:
        client: Async Binance client wrapper (same shape as Streamer's).
        symbols: Symbol list (case-insensitive).
        on_tick: Async callback invoked with ``(symbol, price, ts_ms)``.
            Each tick fires the callback once; the callback should be
            non-blocking — long work belongs in a queue worker.
        source: ``"mark"`` (default, matches exchange liquidation
            triggers) or ``"book"`` (best-bid/ask mid; lower latency
            but liquidates earlier than a real flush).
    """

    _INITIAL_BACKOFF = 1.0
    _MAX_BACKOFF = 60.0
    _BACKOFF_FACTOR = 2.0
    _HEARTBEAT_TIMEOUT = 90.0  # seconds without messages → reconnect

    def __init__(
        self,
        client: IClient,
        symbols: List[str],
        on_tick: TickCallback,
        source: str = "mark",
    ):
        if source not in ("mark", "book"):
            raise ValueError(f"unknown source '{source}', use 'mark' or 'book'")
        self.client = client
        self.raw_client = getattr(client, "raw_client", client)
        self.symbols = [s.upper().replace("/", "") for s in symbols]
        self.on_tick = on_tick
        self.source = source
        self.bsm = BinanceSocketManager(self.raw_client)
        self._task: Optional[asyncio.Task] = None
        self._stopped = False
        self.stats = _TickStats()
        # Phase B1 — bookTicker cache (best bid/ask + ts_ms) populated
        # only when ``source="book"``.  Used by ``open_position`` to
        # reject entries when the spread is wider than allowed.
        self._book_cache: dict[str, tuple[float, float, int]] = {}

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    async def start(self) -> None:
        if self._task is not None:
            return
        self._stopped = False
        self._task = asyncio.create_task(self._run_with_reconnect())
        log.info("Tick stream started (source=%s, %d symbols)",
                 self.source, len(self.symbols))

    async def stop(self) -> None:
        self._stopped = True
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
        log.info(
            "Tick stream stopped (received=%d forwarded=%d reconnects=%d)",
            self.stats.received, self.stats.forwarded, self.stats.reconnects,
        )

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    async def _run_with_reconnect(self) -> None:
        backoff = self._INITIAL_BACKOFF
        while not self._stopped:
            try:
                await self._stream()
                # Clean exit (rare); attempt reconnect
                if self._stopped:
                    break
                log.warning("Tick stream ended unexpectedly; reconnecting...")
            except asyncio.CancelledError:
                raise
            except Exception as e:
                log.warning("Tick stream error: %s — reconnecting in %.1fs",
                            e, backoff)
                self.stats.reconnects += 1

            await asyncio.sleep(backoff)
            backoff = min(backoff * self._BACKOFF_FACTOR, self._MAX_BACKOFF)
            self.bsm = BinanceSocketManager(self.raw_client)

    def _build_streams(self) -> List[str]:
        if self.source == "mark":
            return [f"{s.lower()}@markPrice@1s" for s in self.symbols]
        return [f"{s.lower()}@bookTicker" for s in self.symbols]

    async def _stream(self) -> None:
        streams = self._build_streams()
        self.stats.last_msg_ts = time.monotonic()
        sock = self.bsm.futures_multiplex_socket(streams=streams)

        async with sock as stream:
            while not self._stopped:
                try:
                    msg = await asyncio.wait_for(
                        stream.recv(),
                        timeout=self._HEARTBEAT_TIMEOUT,
                    )
                except asyncio.TimeoutError:
                    elapsed = time.monotonic() - self.stats.last_msg_ts
                    raise ConnectionError(f"tick heartbeat timeout ({elapsed:.0f}s)")

                self.stats.last_msg_ts = time.monotonic()
                self.stats.received += 1

                data = msg.get("data", msg) if isinstance(msg, dict) else None
                if not isinstance(data, dict):
                    continue

                symbol, price, ts_ms = self._extract(data)
                if symbol is None or price is None or symbol not in self.symbols:
                    continue

                self.stats.forwarded += 1
                try:
                    await self.on_tick(symbol, price, ts_ms)
                except asyncio.CancelledError:
                    raise
                except Exception as e:  # pragma: no cover — defensive
                    log.warning("on_tick callback raised for %s: %s", symbol, e)

    def _extract(self, data: dict):
        """Pull (symbol, price, ts_ms) out of a Binance multiplex payload.

        Mark-price payload::
            {"e":"markPriceUpdate","E":1626265555000,"s":"BTCUSDT","p":"33500.00",...}

        Book-ticker payload::
            {"u":..., "s":"BTCUSDT", "b":"<bid>", "a":"<ask>", ...}

        Side-effect: when ``source == "book"`` the bid/ask is cached
        for the spread filter (Phase B1).
        """
        try:
            symbol = data.get("s", "").upper()
            ts_ms = int(data.get("E") or data.get("T") or time.time() * 1000)
            if self.source == "mark":
                return symbol, float(data["p"]), ts_ms
            # bookTicker — cache bid/ask and forward the mid
            bid = float(data["b"])
            ask = float(data["a"])
            self._book_cache[symbol] = (bid, ask, ts_ms)
            return symbol, (bid + ask) / 2.0, ts_ms
        except (KeyError, ValueError, TypeError):
            return None, None, 0

    # ------------------------------------------------------------------
    # Public spread accessors (Phase B1)
    # ------------------------------------------------------------------

    def get_book(self, symbol: str) -> Optional[tuple[float, float, int]]:
        """Return ``(best_bid, best_ask, ts_ms)`` or ``None`` if no
        bookTicker tick has arrived yet for *symbol*.

        Only meaningful when the streamer was constructed with
        ``source="book"``.  Returns ``None`` for mark-price streamers.
        """
        if self.source != "book":
            return None
        return self._book_cache.get(symbol.upper())

    def get_spread_bps(self, symbol: str) -> Optional[float]:
        """Return the current spread in basis points or ``None``.

        ``None`` when (a) the streamer is not in book mode or (b) no
        tick has arrived for the symbol yet.  Callers should treat
        ``None`` as a default-deny signal.
        """
        book = self.get_book(symbol)
        if book is None:
            return None
        bid, ask, _ = book
        if bid <= 0 or ask <= 0 or ask <= bid:
            return None
        mid = (bid + ask) / 2.0
        return (ask - bid) / mid * 10_000.0
