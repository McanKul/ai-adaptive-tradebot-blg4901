import asyncio, time
from collections import defaultdict
from utils.bar_store import BarStore
from utils.logger import setup_logger
from Interfaces.IStreamer import IStreamer
from binance import BinanceSocketManager

log = setup_logger("Streamer")

TF_SEC = {"1m":60, "5m":300, "15m":900, "30m":1800,
          "1h":3600, "2h":7200, "4h":14400,
          "6h":21600, "8h":28800, "12h":43200}

class Streamer(IStreamer):

    def __init__(self, client, symbols, intervals, bar_store: BarStore):
        # Unwrap IClient wrapper if present to get raw AsyncClient for BSM
        if hasattr(client, "raw_client"):
            self.client = client.raw_client
        else:
            self.client   = client
            
        self.symbols  = [s.upper().replace("/","") for s in symbols]
        self.intervals= intervals
        self.bar_store= bar_store
        self.queue    = asyncio.Queue()
        self.bsm      = BinanceSocketManager(self.client)
        self.task     = None

        # partial bar buffer
        self.partial = defaultdict(
            lambda: {"o":None,"h":0,"l":1e18,"c":None,
                     "v":0,"start":None,"i":None,"x":False}
        )

    # -----------------------------------------------------------------
    async def _fetch_kline(self, client, sym, tf, limit):
        try:
            kl = await client.futures_klines(symbol=sym, interval=tf, limit=limit)
            return sym, tf, kl
        except Exception as e:
            log.warning("%s | %s preload error: %s", sym, tf, e)
            return sym, tf, None

    async def preload_history(self, symbols, intervals, limit=250, batch=50):
        tasks = []
        for tf in intervals:
            for sym in symbols:
                tasks.append(self._fetch_kline(self.client, sym, tf, limit))

        # batch processing
        for i in range(0, len(tasks), batch):
            chunk = tasks[i:i + batch]
            results = await asyncio.gather(*chunk)
            for sym, tf, klines in results:
                if not klines:
                    continue
                for k in klines:
                    self.bar_store.add_bar(sym, tf, {
                        "t":k[0],"T":k[6],"o":k[1],"h":k[2],
                        "l":k[3],"c":k[4],"v":k[5],
                        "x":True,"i":tf})
                log.info("Preloaded %s x %s bars (%s)",
                        sym, len(klines), tf)

            await asyncio.sleep(1)

    # -----------------------------------------------------------------
    async def _stream_aggregate(self):
        sock = self.bsm.futures_socket(path="!miniTicker@arr")
        async with sock as stream:
            async for arr in stream:
                # miniTicker timestamp is in ms
                ts = int(arr[0]["E"]//1000) 
                for t in arr:
                    sym = t["s"]
                    if sym not in self.symbols: continue
                    self._update_partial(sym, float(t["c"]),
                                         float(t["q"]), ts)

    def _update_partial(self, sym, price, vol, ts):
        for tf in self.intervals:
            bucket = ts - ts % TF_SEC[tf]
            buf = self.partial[(sym, tf)]
            if buf["start"] != bucket:          # bar closing/new bar
                if buf["start"] is not None:    # commit old bar
                    buf["x"] = True
                    self.bar_store.add_bar(sym, tf, buf.copy())
                    self.queue.put_nowait({"s":sym, "k":buf.copy()}) # notify consumers
                buf.update(o=price,h=price,l=price,c=price,
                           v=vol,start=bucket,i=tf,x=False)
            else:                               # bar continues
                buf["c"] = price
                buf["h"] = max(buf["h"], price)
                buf["l"] = min(buf["l"], price)
                buf["v"] += vol

    # -----------------------------------------------------------------
    async def start(self):
        self.task = asyncio.create_task(self._stream_aggregate())
        log.info("Aggregate miniTicker stream started - %s symbols | tf=%s",
                 len(self.symbols), self.intervals)

    async def stop(self):
        if self.task:
            self.task.cancel()
            try:
                await self.task
            except asyncio.CancelledError:
                pass
        await self.client.close_connection()
        log.info("Streamer stopped.")

    # IStreamer interface
    def get_queue(self): return self.queue
    async def get(self):  return await self.queue.get()

    # -----------------------------------------------------------------
    @staticmethod
    async def resolve_symbols(client, coins_spec):
        """
        coins_spec  ->  ["BTCUSDT", ...]   or   "ALL_USDT"
        """
        # 1) Handle single element list for ALL_USDT
        if isinstance(coins_spec, (list, tuple)) and len(coins_spec) == 1 \
        and coins_spec[0].upper() == "ALL_USDT":
            coins_spec = "ALL_USDT"

        if coins_spec == "ALL_USDT":
            try:
                info = await client.futures_exchange_info()
                return [s["symbol"] for s in info["symbols"]
                        if s["quoteAsset"] == "USDT" and s["status"] == "TRADING"]
            except Exception as e:
                log.error("Exchange info failed: %s", e)
                return []

        # 2) Normal list
        return [sym.upper().replace("/", "")
                for sym in coins_spec if isinstance(sym, str)]
