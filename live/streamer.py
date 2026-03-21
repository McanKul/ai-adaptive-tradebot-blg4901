import asyncio
import time
from utils.bar_store import BarStore
from utils.logger import setup_logger
from Interfaces.IStreamer import IStreamer
from Interfaces.IClient import IClient
from binance import BinanceSocketManager

log = setup_logger("Streamer")


class Streamer(IStreamer):
    """
    Kline WebSocket streamer with:
    - Gap-free preload→live transition (buffer mode)
    - Auto-reconnect with exponential backoff
    - Heartbeat timeout detection (silent disconnect protection)
    - Gap recovery via REST after reconnect

    Flow (driven by LiveEngine):
        1. start_buffering()   → WS connects, closed bars buffered in memory
        2. preload_history()   → REST klines fetched (skips current open bar)
        3. flush_buffer()      → buffered WS bars deduped into bar_store & queue
        4. main loop consumes  queue.get()
    """

    # ── Reconnect settings ───────────────────────────────────────────
    _INITIAL_BACKOFF = 1.0       # first retry after 1s
    _MAX_BACKOFF = 60.0          # cap at 60s
    _BACKOFF_FACTOR = 2.0        # double each retry
    _MAX_RECONNECTS = 0          # 0 = unlimited
    _HEARTBEAT_TIMEOUT = 90.0    # no message for 90s → force reconnect

    def __init__(self, client: IClient, symbols, intervals, bar_store: BarStore):
        self.client: IClient = client
        self.raw_client = getattr(client, "raw_client", client)

        self.symbols = [s.upper().replace("/", "") for s in symbols]
        self.intervals = intervals
        self.bar_store = bar_store
        self.queue = asyncio.Queue()
        self.bsm = BinanceSocketManager(self.raw_client)
        self.task = None
        self._stopped = False

        # Buffer mode: WS events collected here until flush_buffer()
        self._buffering = False
        self._buffer: list[dict] = []

        # Reconnect stats
        self._reconnect_count = 0
        self._last_msg_time: float = 0.0

    # ── Preload historical bars ──────────────────────────────────────

    async def _fetch_kline(self, client: IClient, sym, tf, limit):
        try:
            kl = await client.futures_klines(symbol=sym, interval=tf, limit=limit)
            return sym, tf, kl
        except Exception as e:
            log.warning("%s | %s preload error: %s", sym, tf, e)
            return sym, tf, None

    async def preload_history(self, symbols, intervals, limit=250, batch=50):
        """
        Fetch historical klines via REST.
        Skips the current (still-open) bar to avoid adding incomplete data.
        """
        now_ms = int(time.time() * 1000)

        tasks = []
        for tf in intervals:
            for sym in symbols:
                tasks.append(self._fetch_kline(self.client, sym, tf, limit))

        for i in range(0, len(tasks), batch):
            chunk = tasks[i : i + batch]
            results = await asyncio.gather(*chunk)
            for sym, tf, klines in results:
                if not klines:
                    continue
                added = 0
                for k in klines:
                    close_time_ms = int(k[6])
                    # Skip current open bar (close time in the future)
                    if close_time_ms > now_ms:
                        continue
                    self.bar_store.add_bar(sym, tf, {
                        "t": int(k[0]), "T": int(k[6]),
                        "o": k[1], "h": k[2], "l": k[3],
                        "c": k[4], "v": k[5],
                        "x": True, "i": tf,
                    })
                    added += 1
                log.info("Preloaded %s x %d closed bars (%s)", sym, added, tf)

            await asyncio.sleep(1)

    # ── Kline WebSocket stream with auto-reconnect ───────────────────

    async def start_buffering(self):
        """
        Start the kline WS stream immediately and buffer closed bars.
        Call this BEFORE preload_history() to avoid a data gap.
        """
        self._buffering = True
        self._buffer.clear()
        self._stopped = False
        self.task = asyncio.create_task(self._run_with_reconnect())
        log.info(
            "Kline stream started (buffer mode) — %d symbols | tf=%s",
            len(self.symbols), self.intervals,
        )

    def flush_buffer(self):
        """
        Stop buffering: push all buffered closed bars into bar_store
        and queue, then switch to live mode.
        bar_store's timestamp dedup prevents duplicates with preloaded data.
        """
        count = 0
        for event in self._buffer:
            k = event["k"]
            sym = event["s"]
            tf = k["i"]
            self.bar_store.add_bar(sym, tf, k)
            self.queue.put_nowait(event)
            count += 1

        self._buffer.clear()
        self._buffering = False
        log.info("Flushed %d buffered bars to live queue", count)

    async def _run_with_reconnect(self):
        """
        Outer loop: keeps _stream_klines alive with auto-reconnect.

        On disconnect:
        1. Log the error
        2. Wait (exponential backoff)
        3. Recover missed bars via REST (gap recovery)
        4. Reconnect WS
        """
        backoff = self._INITIAL_BACKOFF

        while not self._stopped:
            try:
                await self._stream_klines()
            except asyncio.CancelledError:
                raise
            except Exception as e:
                if self._stopped:
                    break

                self._reconnect_count += 1
                log.warning(
                    "WS disconnected (attempt #%d): %s — reconnecting in %.1fs",
                    self._reconnect_count, e, backoff,
                )

                if self._MAX_RECONNECTS > 0 and self._reconnect_count > self._MAX_RECONNECTS:
                    log.error("Max reconnect attempts (%d) reached, giving up", self._MAX_RECONNECTS)
                    break

                await asyncio.sleep(backoff)
                backoff = min(backoff * self._BACKOFF_FACTOR, self._MAX_BACKOFF)

                # Gap recovery: fetch recent bars via REST to fill any gap
                await self._recover_gap()

                # Reset BSM for fresh connection
                self.bsm = BinanceSocketManager(self.raw_client)
                continue

            # Clean exit from _stream_klines (shouldn't normally happen)
            if not self._stopped:
                log.warning("WS stream ended unexpectedly, reconnecting...")
                await asyncio.sleep(backoff)
                await self._recover_gap()
                self.bsm = BinanceSocketManager(self.raw_client)

        log.info("Reconnect loop exited (total reconnects: %d)", self._reconnect_count)

    async def _recover_gap(self):
        """
        After a reconnect, fetch the last few bars via REST to fill
        any gap caused by the disconnect. BarStore dedup handles overlaps.
        """
        log.info("Recovering gap: fetching recent bars via REST...")
        try:
            await self.preload_history(
                self.symbols,
                self.intervals,
                limit=5,    # last 5 bars is enough to cover most gaps
                batch=50,
            )
            log.info("Gap recovery complete")
        except Exception as e:
            log.warning("Gap recovery failed: %s", e)

    async def _stream_klines(self):
        """
        Connect to Binance futures kline multiplex WebSocket.
        Each closed bar (x=True) is either buffered or sent to bar_store+queue.

        Raises on disconnect so _run_with_reconnect can handle retry.
        Uses heartbeat timeout to detect silent disconnects.
        """
        streams = [
            f"{sym.lower()}@kline_{tf}"
            for sym in self.symbols
            for tf in self.intervals
        ]

        self._last_msg_time = time.monotonic()

        sock = self.bsm.futures_multiplex_socket(streams=streams)
        async with sock as stream:
            while not self._stopped:
                try:
                    msg = await asyncio.wait_for(
                        stream.recv(),
                        timeout=self._HEARTBEAT_TIMEOUT,
                    )
                except asyncio.TimeoutError:
                    elapsed = time.monotonic() - self._last_msg_time
                    log.warning(
                        "No WS message for %.0fs (timeout=%.0fs) — forcing reconnect",
                        elapsed, self._HEARTBEAT_TIMEOUT,
                    )
                    raise ConnectionError(f"Heartbeat timeout ({elapsed:.0f}s)")

                self._last_msg_time = time.monotonic()

                # Multiplex wraps in {"stream": ..., "data": {...}}
                data = msg.get("data", msg)
                k = data.get("k", {})
                sym = data.get("s", k.get("s", ""))

                if sym not in self.symbols:
                    continue

                # Only process closed bars
                if not k.get("x", False):
                    continue

                event = {"s": sym, "k": k}

                if self._buffering:
                    self._buffer.append(event)
                else:
                    tf = k["i"]
                    self.bar_store.add_bar(sym, tf, k)
                    self.queue.put_nowait(event)

    # ── IStreamer interface ───────────────────────────────────────────

    async def start(self):
        """Start streaming (if not already started via start_buffering)."""
        if self.task is None:
            self._stopped = False
            self.task = asyncio.create_task(self._run_with_reconnect())
            log.info(
                "Kline stream started — %d symbols | tf=%s",
                len(self.symbols), self.intervals,
            )

    async def stop(self):
        self._stopped = True
        if self.task:
            self.task.cancel()
            try:
                await self.task
            except asyncio.CancelledError:
                pass
        try:
            await self.client.close_connection()
        except Exception:
            pass
        log.info(
            "Streamer stopped (total reconnects: %d).",
            self._reconnect_count,
        )

    def get_queue(self):
        return self.queue

    async def get(self):
        return await self.queue.get()

    @property
    def reconnect_count(self) -> int:
        return self._reconnect_count

    # ── Symbol resolution ────────────────────────────────────────────

    @staticmethod
    async def resolve_symbols(client: IClient, coins_spec):
        """
        coins_spec  ->  ["BTCUSDT", ...]   or   "ALL_USDT"
        """
        if isinstance(coins_spec, (list, tuple)) and len(coins_spec) == 1 \
                and coins_spec[0].upper() == "ALL_USDT":
            coins_spec = "ALL_USDT"

        if coins_spec == "ALL_USDT":
            try:
                info = await client.futures_exchange_info()
                return [
                    s["symbol"] for s in info["symbols"]
                    if s["quoteAsset"] == "USDT" and s["status"] == "TRADING"
                ]
            except Exception as e:
                log.error("Exchange info failed: %s", e)
                return []

        return [
            sym.upper().replace("/", "")
            for sym in coins_spec if isinstance(sym, str)
        ]
