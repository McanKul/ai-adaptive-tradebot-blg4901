import math
import time

from binance.exceptions import BinanceAPIException
from binance.enums import *
from binance import AsyncClient
from utils.logger import setup_logger
log = setup_logger("PositionManager")
from utils.interfaces import IBroker

class Position:

    def __init__(self, client: AsyncClient, symbol: str, side: str,
                 qty: float, entry_price: float,
                 sl_price: float = None, tp_price: float = None,
                 opened_ts: float = None,
                 tick: int = None, strategy: str = None,
                 expire_sec: int = 3600,
                 timeframes: str = "1h"):
        self.client = client
        self.symbol = symbol
        self.side = side
        self.qty = qty
        self.entry = entry_price
        self.sl = sl_price
        self.tp = tp_price
        self.open_ts = opened_ts or time.time()
        self.closed = False
        self.exit_ts = None
        self.exit = None
        self.exit_type = None
        self.expire_sec = expire_sec
        self.tick = tick
        self.timeframes = timeframes
        self.strategy = strategy
        self.expire = self.open_ts + expire_sec


class PositionManager:

    def __init__(self, broker: IBroker, base_capital: float = 10.0, max_concurrent: int = 1):
        self.broker = broker
        self.client = broker.client 
        self.base_cap = base_capital

        self.max_open = max_concurrent
        self.open_positions = {}
        self.history = []

    def round_price(self, raw, tick, up=False):
        factor = 1 / tick
        return (math.ceil if up else math.floor)(raw * factor) / factor


    async def _symbol_filters(self, symbol: str, qty_f: float) -> tuple[float, float]:
        try:
            info = await self.client.futures_exchange_info()
            step = tick = None
            for s in info['symbols']:
                if s['symbol'] == symbol:
                    for f in s['filters']:
                        if f['filterType'] == 'LOT_SIZE':
                            lot = float(f['stepSize'])
                            factor = 1 / lot
                            step = math.floor(qty_f * factor) / factor
                        if f['filterType'] == 'PRICE_FILTER':
                            tick = float(f['tickSize'])
                    if tick and step:
                        return step, tick
            return 0.0, 0.0

        except Exception as e:
            log.error("Failed to get LOT_SIZE/PRICE_FILTER %s: %s", symbol, e)
            return 0.0, 0.0

    async def open_position(self, symbol: str, side: int, strategy_name: str, leverage: int, sl_pct: float, tp_pct: float, expire_sec: int, timeframes: str):
        key = (symbol, strategy_name)
        if key in self.open_positions or len(self.open_positions) >= self.max_open:
            return False

        mark_price = float((await self.client.futures_mark_price(symbol=symbol))["markPrice"])
        notional = self.base_cap * leverage
        raw_qty = notional / mark_price
        qty, tick = await self._symbol_filters(symbol, raw_qty)
        if qty <= 0:
            return False

        side_str = SIDE_BUY if side == 1 else SIDE_SELL
        opp_str = SIDE_SELL if side_str == SIDE_BUY else SIDE_BUY


        try:
            await self.broker.ensure_isolated_margin(symbol)
            await self.broker.set_leverage(symbol, leverage)
        except Exception as e:
            log.error("%s margin/leverage error: %s", symbol, e)
            return False

        raw_sl = mark_price * (1 - sl_pct/leverage / 100) if side_str == SIDE_BUY else mark_price * (1 + sl_pct/leverage / 100)
        raw_tp = mark_price * (1 + tp_pct/leverage / 100) if side_str == SIDE_BUY else mark_price * (1 - tp_pct/leverage / 100)

        price_sl = self.round_price(raw_sl, tick, up=(side_str == SIDE_SELL))
        price_tp = self.round_price(raw_tp, tick, up=(side_str == SIDE_BUY))

        try:
            await self.broker.market_order(symbol, side_str, qty)
            await self.broker.place_stop_market(symbol, opp_str, price_sl)
            await self.broker.place_take_profit(symbol, opp_str, price_tp)
        except Exception as e:
             log.error("Order placement failed %s: %s", symbol, e)
             return False

        pos = Position(self.client, symbol, side_str, qty, mark_price, price_sl, price_tp, time.time(), tick, strategy=strategy_name, expire_sec=expire_sec, timeframes=timeframes)
        self.open_positions[key] = pos
        log.info("%s [%s] [%s] Open pos: qty=%.4f, SL=%.8f, TP=%.8f",
                 symbol, strategy_name,timeframes, qty, price_sl, price_tp)
        return True


    async def update_all(self):
        now = time.time()
        closed = []

        for key, pos in self.open_positions.items():
            symbol, strategy_name = key

            try:
                mark_price = await self.broker.get_mark_price(symbol)
            except Exception as e:
                log.warning("%s failed to get mark price: %s", symbol, e)
                continue

            hit_tp = pos.side == SIDE_BUY and mark_price >= pos.tp or pos.side == SIDE_SELL and mark_price <= pos.tp
            hit_sl = pos.side == SIDE_BUY and mark_price <= pos.sl or pos.side == SIDE_SELL and mark_price >= pos.sl
            expired = now >= pos.expire

            if hit_tp:
                log.info("%s TP triggered (%.2f)", symbol, mark_price)
            elif hit_sl:
                log.info("%s SL triggered (%.2f)", symbol, mark_price)
            elif expired:
                log.info("%s position expired", symbol)
            else:
                continue

            await self.broker.close_position(symbol)
            closed.append(key)
            
        for key in closed:
            del self.open_positions[key]


    async def force_close_all(self):
        to_remove = []

        for key, pos in list(self.open_positions.items()):
            symbol, _ = key
            try:
                await self.broker.close_position(symbol)
            except Exception as e:
                log.warning("%s force close failed: %s", symbol, e)
            else:
                log.info("%s force closed.", symbol)

            pos.closed = True
            pos.exit_type = "MANUAL"
            self.history.append(pos)
            to_remove.append(key)

        for key in to_remove:
            del self.open_positions[key]

        log.info("All positions force closed.")
