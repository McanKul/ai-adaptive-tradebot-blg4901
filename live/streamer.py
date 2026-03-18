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
    Kline WebSocket streamer with gap-free preload→live transition.

    Flow (driven by LiveEngine):
        1. start_buffering()   → WS connects, closed bars buffered in memory
        2. preload_history()   → REST klines fetched (skips current open bar)
        3. flush_buffer()      → buffered WS bars deduped into bar_store & queue
        4. main loop consumes  queue.get()

    This eliminates the gap between preload and WebSocket start.
    """

    def __init__(self, client: IClient, symbols, intervals, bar_store: BarStore):
        self.client: IClient = client
        self.raw_client = getattr(client, "raw_client", client)

        self.symbols = [s.upper().replace("/", "") for s in symbols]
        self.intervals = intervals
        self.bar_store = bar_store
        self.queue = asyncio.Queue()
        self.bsm = BinanceSocketManager(self.raw_client)
        self.task = None

        # Buffer mode: WS events collected here until flush_buffer()
        self._buffering = False
        self._buffer: list[dict] = []

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

    # ── Kline WebSocket stream ───────────────────────────────────────

    async def start_buffering(self):
        """
        Start the kline WS stream immediately and buffer closed bars.
        Call this BEFORE preload_history() to avoid a data gap.
        """
        self._buffering = True
        self._buffer.clear()
        self.task = asyncio.create_task(self._stream_klines())
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

    async def _stream_klines(self):
        """
        Connect to Binance futures kline multiplex WebSocket.
        Each closed bar (x=True) is either buffered or sent to bar_store+queue.
        """
        streams = [
            f"{sym.lower()}@kline_{tf}"
            for sym in self.symbols
            for tf in self.intervals
        ]

        sock = self.bsm.futures_multiplex_socket(streams=streams)
        async with sock as stream:
            async for msg in stream:
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
            self.task = asyncio.create_task(self._stream_klines())
            log.info(
                "Kline stream started — %d symbols | tf=%s",
                len(self.symbols), self.intervals,
            )

    async def stop(self):
        if self.task:
            self.task.cancel()
            try:
                await self.task
            except asyncio.CancelledError:
                pass
        await self.client.close_connection()
        log.info("Streamer stopped.")

    def get_queue(self):
        return self.queue

    async def get(self):
        return await self.queue.get()

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
