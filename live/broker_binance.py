from binance.exceptions import BinanceAPIException
from binance.enums import *
from utils.logger import setup_logger
from Interfaces.IBroker import IBroker
from Interfaces.IClient import IClient
from live.rate_limiter import AsyncRateLimiter, ExchangeInfoCache
import math

class BinanceBroker(IBroker):
    """Binance API wrapper implementing IBroker with rate-limiting."""

    def __init__(self, client: IClient, rate_limiter: AsyncRateLimiter | None = None,
                 exchange_info_ttl: int = 300):
        self._client = client
        self.log = setup_logger("BinanceBroker")
        self._rl = rate_limiter or AsyncRateLimiter()
        self._ei_cache = ExchangeInfoCache(ttl_sec=exchange_info_ttl)
        # idempotent state caches
        self._margin_set: set[str] = set()      # symbols already set to ISOLATED
        self._leverage_set: dict[str, int] = {}  # symbol -> last leverage sent

    @property
    def client(self) -> IClient:
        return self._client

    async def get_mark_price(self, symbol: str) -> float:
        await self._rl.acquire()
        data = await self._client.futures_mark_price(symbol=symbol)
        return float(data["markPrice"])

    # ——————————————————— IBroker API ————————————————————
    async def market_order(self, symbol: str, side: str, qty: float):
        await self._rl.acquire(weight=1)
        return await self._client.futures_create_order(
            symbol=symbol,
            side=side,
            type=FUTURE_ORDER_TYPE_MARKET,
            quantity=qty,
        )

    async def close_position(self, symbol: str):
        amt = await self.position_amt(symbol)
        if amt == 0:
            return
        side = SIDE_SELL if amt > 0 else SIDE_BUY
        await self.market_order(symbol, side, abs(amt))

    async def position_amt(self, symbol: str) -> float:
        await self._rl.acquire()
        info = await self._client.futures_position_information(symbol=symbol)
        p = next((x for x in info if float(x["positionAmt"]) != 0), None)
        return float(p["positionAmt"]) if p else 0.0

    # ───── Margin & Leverage (idempotent) ─────
    async def ensure_isolated_margin(self, symbol: str):
        if symbol in self._margin_set:
            return
        try:
            await self._rl.acquire()
            await self._client.futures_change_margin_type(symbol=symbol, marginType="ISOLATED")
        except BinanceAPIException as e:
            if e.code != -4046:  # ignore if already isolated
                raise
        self._margin_set.add(symbol)

    async def set_leverage(self, symbol: str, leverage: int):
        if self._leverage_set.get(symbol) == leverage:
            return
        await self._rl.acquire()
        await self._client.futures_change_leverage(symbol=symbol, leverage=leverage)
        self._leverage_set[symbol] = leverage

    # ───── SL / TP Orders (return order ID) ─────
    async def place_stop_market(self, symbol: str, side: str, stop_price: float) -> int:
        tick = await self._tick_size(symbol)
        fmt = f"{stop_price:.{abs(int(math.log10(tick)))}f}"
        await self._rl.acquire()
        resp = await self._client.futures_create_order(
            symbol=symbol, side=side,
            type=FUTURE_ORDER_TYPE_STOP_MARKET,
            stopPrice=fmt, closePosition=True,
        )
        return int(resp.get("orderId", 0))

    async def place_take_profit(self, symbol: str, side: str, stop_price: float) -> int:
        tick = await self._tick_size(symbol)
        fmt = f"{stop_price:.{abs(int(math.log10(tick)))}f}"
        await self._rl.acquire()
        resp = await self._client.futures_create_order(
            symbol=symbol, side=side,
            type=FUTURE_ORDER_TYPE_TAKE_PROFIT_MARKET,
            stopPrice=fmt, closePosition=True,
        )
        return int(resp.get("orderId", 0))

    async def cancel_order(self, symbol: str, order_id: int):
        """Cancel a specific order by ID."""
        await self._rl.acquire()
        try:
            await self._client.futures_cancel_order(symbol=symbol, orderId=order_id)
        except BinanceAPIException as e:
            if e.code != -2011:  # UNKNOWN_ORDER — already filled or cancelled
                raise

    async def get_open_orders(self, symbol: str) -> list[dict]:
        """Return open orders for *symbol*."""
        await self._rl.acquire()
        return await self._client.futures_get_open_orders(symbol=symbol)

    # ───── Helpers ─────
    async def _tick_size(self, symbol: str) -> float:
        info = await self._ei_cache.get(self._fetch_exchange_info)
        f = next(x for x in info["symbols"] if x["symbol"] == symbol)
        return float(next(fl["tickSize"] for fl in f["filters"] if fl["filterType"] == "PRICE_FILTER"))

    async def exchange_info(self) -> dict:
        """Cached exchange info."""
        return await self._ei_cache.get(self._fetch_exchange_info)

    async def _fetch_exchange_info(self) -> dict:
        await self._rl.acquire(weight=10)
        return await self._client.futures_exchange_info()

    async def balance(self, asset: str = "USDT") -> float:
        await self._rl.acquire()
        for bal in await self._client.futures_account_balance():
            if bal["asset"] == asset:
                return float(bal["balance"])
        return 0.0
