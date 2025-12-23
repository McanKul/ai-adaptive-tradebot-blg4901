from binance.exceptions import BinanceAPIException
from binance.enums import *
from utils.logger import setup_logger
from utils.interfaces import IBroker
import math

class BinanceBroker(IBroker):
    """Binance API wrapper implementing IBroker"""

    def __init__(self, client):
        self.client = client
        self.log = setup_logger("BinanceBroker")

    async def get_mark_price(self, symbol: str) -> float:
        data = await self.client.futures_mark_price(symbol=symbol)
        return float(data["markPrice"])

    # ——————————————————— IBroker API ————————————————————
    async def market_order(self, symbol: str, side: str, qty: float):
        return await self.client.futures_create_order(
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
        info = await self.client.futures_position_information(symbol=symbol)
        p = next((x for x in info if float(x["positionAmt"]) != 0), None)
        return float(p["positionAmt"]) if p else 0.0

    # ───── Margin & Leverage ─────
    async def ensure_isolated_margin(self, symbol: str):
        try:
            await self.client.futures_change_margin_type(symbol=symbol, marginType="ISOLATED")
        except BinanceAPIException as e:
            if e.code != -4046:  # ignore if already isolated
                raise

    async def set_leverage(self, symbol: str, leverage: int):
        await self.client.futures_change_leverage(symbol=symbol, leverage=leverage)

    # ───── SL / TP Orders ─────
    async def place_stop_market(self, symbol: str, side: str, stop_price: float):
        tick = await self._tick_size(symbol)
        # format price to correct tick precision
        fmt = f"{stop_price:.{abs(int(math.log10(tick)))}f}"
        await self.client.futures_create_order(symbol=symbol, side=side,
                                               type=FUTURE_ORDER_TYPE_STOP_MARKET,
                                               stopPrice=fmt, closePosition=True)

    async def place_take_profit(self, symbol: str, side: str, stop_price: float):
        tick = await self._tick_size(symbol)
        fmt = f"{stop_price:.{abs(int(math.log10(tick)))}f}"
        await self.client.futures_create_order(symbol=symbol, side=side,
                                               type=FUTURE_ORDER_TYPE_TAKE_PROFIT_MARKET,
                                               stopPrice=fmt, closePosition=True)

    # ───── Helpers ─────
    async def _tick_size(self, symbol: str) -> float:
        info = await self.client.futures_exchange_info()
        f = next(x for x in info["symbols"] if x["symbol"] == symbol)
        return float(next(fl["tickSize"] for fl in f["filters"] if fl["filterType"] == "PRICE_FILTER"))

    async def balance(self, asset: str = "USDT") -> float:
        for bal in await self.client.futures_account_balance():
            if bal["asset"] == asset:
                return float(bal["balance"])
        return 0.0
