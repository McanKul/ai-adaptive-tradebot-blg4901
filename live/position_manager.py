import math
import time

# from binance.exceptions import BinanceAPIException # Exceptions should ideally be handled or wrapped, but keeping for now as they bubble up
# from binance.enums import * # Enums are used
from binance.enums import *
from utils.logger import setup_logger
log = setup_logger("PositionManager")
from Interfaces.IBroker import IBroker
from Interfaces.IClient import IClient

class Position:

    def __init__(self, client: IClient, symbol: str, side: str,
                 qty: float, entry_price: float,
                 sl_price: float = None, tp_price: float = None,
                 opened_ts: float = None,
                 tick: int = None, strategy: str = None,
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
        self.tick = tick
        self.timeframes = timeframes
        self.strategy = strategy
        # Expiration removed

    async def _current_price(self) -> float:
        res = await self.client.futures_mark_price(symbol=self.symbol)
        return float(res["markPrice"])

    async def _close_market(self):
        opp = SIDE_SELL if self.side == SIDE_BUY else SIDE_BUY
        try:
            await self.client.futures_create_order(
                symbol=self.symbol,
                side=opp,
                type=FUTURE_ORDER_TYPE_MARKET,
                quantity=f"{self.qty:.{abs(int(math.log10(self.tick)))}f}"
            )
        except Exception as e:
            log.error("Market close error %s: %s", self.symbol, e)
            raise

    async def check_exit(self, now: float) -> bool:
        if self.closed:
            return True

        price = await self._current_price()

        if self.tp and (
            (self.side == SIDE_BUY and price >= self.tp) or
            (self.side == SIDE_SELL and price <= self.tp)
        ):
            await self._close_market()
            self.closed = True
            self.exit_type = "TP"

        elif self.sl and (
            (self.side == SIDE_BUY and price <= self.sl) or
            (self.side == SIDE_SELL and price >= self.sl)
        ):
            await self._close_market()
            self.closed = True
            self.exit_type = "SL"

        if self.closed:
            self.exit_ts = now
            self.exit = price
            log.info("%s [%s] closed @ %.8f (%s)",
                     self.symbol, self.strategy or self.side, price, self.exit_type)
            try:
                await self.client.futures_cancel_all_open_orders(symbol=self.symbol)
            except Exception:
                log.info("%s error cancelling orders", self.symbol)

        return self.closed


class PositionManager:

    def __init__(self, broker: IBroker, base_capital: float = 10.0, max_concurrent: int = 1):
        self.broker = broker
        # self.client = broker.client # REMOVED: Use broker.client directly
        self.base_cap = base_capital

        self.max_open = max_concurrent
        self.open_positions = {}
        self.history = []

    def __round_price(self, raw, tick, up=False):
        factor = 1 / tick
        return (math.ceil if up else math.floor)(raw * factor) / factor


    async def __symbol_filters(self, symbol: str, qty_f: float) -> tuple[float, float]:
        try:
            # Use broker.client accessor
            info = await self.broker.client.futures_exchange_info()
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

    async def open_position(self, symbol: str, side: int, strategy_name: str, leverage: int, sl_pct: float, tp_pct: float, timeframes: str):
        # expire_sec removed
        key = (symbol, strategy_name)
        if key in self.open_positions or len(self.open_positions) >= self.max_open:
            return False

        mark_price = float((await self.broker.client.futures_mark_price(symbol=symbol))["markPrice"])
        notional = self.base_cap * leverage
        raw_qty = notional / mark_price
        qty, tick = await self.__symbol_filters(symbol, raw_qty)
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

        price_sl = self.__round_price(raw_sl, tick, up=(side_str == SIDE_SELL))
        price_tp = self.__round_price(raw_tp, tick, up=(side_str == SIDE_BUY))

        try:
            await self.broker.market_order(symbol, side_str, qty)
            await self.broker.place_stop_market(symbol, opp_str, price_sl)
            await self.broker.place_take_profit(symbol, opp_str, price_tp)
        except Exception as e:
             log.error("Order placement failed %s: %s", symbol, e)
             return False

        # Pass broker.client to Position
        pos = Position(self.broker.client, symbol, side_str, qty, mark_price, price_sl, price_tp, time.time(), tick, strategy=strategy_name, timeframes=timeframes)
        self.open_positions[key] = pos
        log.info("%s [%s] [%s] Open pos: qty=%.4f, SL=%.8f, TP=%.8f",
                 symbol, strategy_name,timeframes, qty, price_sl, price_tp)
        return True


    async def update_all(self):
        # Expire logic removed
        now = time.time()
        closed = []

        for key, pos in self.open_positions.items():
            symbol, strategy_name = key

            # Check logic: use Position.check_exit or just monitor price here?
            # Original code monitored here. Position class has `check_exit` which was unused in original PositionManager but useful.
            # Original PositionManager did manual checks.
            # Let's keep manual checks here but clean them up, or use pos.check_exit?
            # User didn't specify to switch to pos.check_exit, but Position class in original file had it.
            # I will stick to original logic structure for safety, just removing expiry.
            
            try:
                mark_price = await self.broker.get_mark_price(symbol)
            except Exception as e:
                log.warning("%s failed to get mark price: %s", symbol, e)
                continue

            hit_tp = pos.side == SIDE_BUY and mark_price >= pos.tp or pos.side == SIDE_SELL and mark_price <= pos.tp
            hit_sl = pos.side == SIDE_BUY and mark_price <= pos.sl or pos.side == SIDE_SELL and mark_price >= pos.sl
            
            if hit_tp:
                log.info("%s TP triggered (%.2f)", symbol, mark_price)
            elif hit_sl:
                log.info("%s SL triggered (%.2f)", symbol, mark_price)
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
