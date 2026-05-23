import time

from binance.exceptions import BinanceAPIException
from binance.enums import (
    SIDE_BUY, SIDE_SELL,
    FUTURE_ORDER_TYPE_MARKET,
    FUTURE_ORDER_TYPE_STOP_MARKET,
    FUTURE_ORDER_TYPE_TAKE_PROFIT_MARKET,
)
from utils.logger import setup_logger
from Interfaces.IBroker import IBroker
from Interfaces.IClient import IClient
from live.rate_limiter import AsyncRateLimiter, ExchangeInfoCache
import math

class BinanceBroker(IBroker):
    """Binance API wrapper implementing IBroker with rate-limiting."""

    _MARK_PRICE_TTL = 2.0  # seconds — cache mark price briefly to reduce API calls

    def __init__(self, client: IClient, rate_limiter: AsyncRateLimiter | None = None,
                 exchange_info_ttl: int = 300, rejection_counter=None):
        self._client = client
        self.log = setup_logger("BinanceBroker")
        self._rl = rate_limiter or AsyncRateLimiter()
        self._ei_cache = ExchangeInfoCache(ttl_sec=exchange_info_ttl)
        # idempotent state caches
        self._margin_set: set[str] = set()      # symbols already set to ISOLATED
        self._leverage_set: dict[str, int] = {}  # symbol -> last leverage sent
        # mark price short-lived cache: symbol -> (price, timestamp)
        self._mark_cache: dict[str, tuple[float, float]] = {}
        # Phase C3 — rejection counter (optional).  When set, every
        # BinanceAPIException raised by ``market_order``,
        # ``place_stop_market`` or ``place_take_profit`` is recorded;
        # the configured callback (e.g. global_risk.trip_kill_switch)
        # is fired once the threshold is crossed.
        self._rejection_counter = rejection_counter

    @property
    def client(self) -> IClient:
        return self._client

    async def get_mark_price(self, symbol: str) -> float:
        """Return mark price with short-lived cache to reduce API calls."""
        now = time.monotonic()
        cached = self._mark_cache.get(symbol)
        if cached and (now - cached[1]) < self._MARK_PRICE_TTL:
            return cached[0]
        await self._rl.acquire()
        data = await self._client.futures_mark_price(symbol=symbol)
        price = float(data["markPrice"])
        self._mark_cache[symbol] = (price, now)
        return price

    # ——————————————————— IBroker API ————————————————————
    async def market_order(self, symbol: str, side: str, qty: float):
        await self._rl.acquire(weight=1)
        try:
            return await self._client.futures_create_order(
                symbol=symbol,
                side=side,
                type=FUTURE_ORDER_TYPE_MARKET,
                quantity=qty,
            )
        except BinanceAPIException as e:
            self._record_rejection(f"market_order {symbol} {side}: {e}")
            raise

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
    # Phase C2: ``workingType="MARK_PRICE"`` matches the trigger price
    # the exchange uses for liquidation, so our SL/TP cannot be sniped
    # by a wick on the contract feed.  ``closePosition=True`` already
    # implies reduce-only on Binance USDT-M; we still log the intent.
    async def place_stop_market(self, symbol: str, side: str, stop_price: float) -> int:
        tick = await self._tick_size(symbol)
        fmt = f"{stop_price:.{abs(int(math.log10(tick)))}f}"
        await self._rl.acquire()
        try:
            resp = await self._client.futures_create_order(
                symbol=symbol, side=side,
                type=FUTURE_ORDER_TYPE_STOP_MARKET,
                stopPrice=fmt, closePosition=True,
                workingType="MARK_PRICE",
            )
        except BinanceAPIException as e:
            self._record_rejection(f"place_stop_market {symbol} {side}: {e}")
            raise
        return int(resp.get("orderId", 0))

    async def place_take_profit(self, symbol: str, side: str, stop_price: float) -> int:
        tick = await self._tick_size(symbol)
        fmt = f"{stop_price:.{abs(int(math.log10(tick)))}f}"
        await self._rl.acquire()
        try:
            resp = await self._client.futures_create_order(
                symbol=symbol, side=side,
                type=FUTURE_ORDER_TYPE_TAKE_PROFIT_MARKET,
                stopPrice=fmt, closePosition=True,
                workingType="MARK_PRICE",
            )
        except BinanceAPIException as e:
            self._record_rejection(f"place_take_profit {symbol} {side}: {e}")
            raise
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

    # ───── Phase C1 — fill confirmation polling ─────
    async def wait_for_fill(
        self, symbol: str, order_id: int, timeout: float = 5.0,
    ) -> dict:
        """Poll ``futures_get_order`` until the order finalises.

        Returns a dict with the canonicalised fields documented on
        ``IBroker.wait_for_fill``.  On timeout returns ``status="TIMEOUT"``
        with whatever partial data has accumulated; the caller decides
        whether to cancel the remaining qty.
        """
        import asyncio
        deadline = time.monotonic() + max(0.0, float(timeout))
        last: dict = {}
        while True:
            await self._rl.acquire()
            try:
                last = await self._client.futures_get_order(
                    symbol=symbol, orderId=order_id,
                )
            except BinanceAPIException as e:
                self.log.warning("wait_for_fill api error %s: %s", order_id, e)
                last = {"status": "ERROR", "code": e.code, "msg": str(e)}
                break
            status = last.get("status", "")
            if status in ("FILLED", "CANCELED", "EXPIRED", "REJECTED"):
                break
            if status == "PARTIALLY_FILLED" and time.monotonic() >= deadline:
                # Don't keep polling forever on partials; let the caller
                # decide whether to cancel the remainder.
                break
            if time.monotonic() >= deadline:
                last = {**last, "status": last.get("status") or "TIMEOUT"}
                if last["status"] not in ("FILLED", "PARTIALLY_FILLED",
                                          "CANCELED", "EXPIRED", "REJECTED"):
                    last["status"] = "TIMEOUT"
                break
            await asyncio.sleep(0.2)

        executed = float(last.get("executedQty") or 0.0)
        avg_price = float(last.get("avgPrice") or 0.0) or float(last.get("price") or 0.0)
        # Sum commission across fills if present (USDT-M reports it on
        # the order itself when small; on get_order's "fills" otherwise).
        commission_usd = 0.0
        for f in (last.get("fills") or []):
            try:
                commission_usd += float(f.get("commission") or 0.0)
            except (TypeError, ValueError):
                pass
        return {
            "status": last.get("status", "TIMEOUT"),
            "executed_qty": executed,
            "avg_price": avg_price,
            "commission_usd": commission_usd,
            "raw": last,
        }

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

    # ───── Phase D — 24h volume liquidity gate ─────
    async def get_24h_volume(self, symbol: str) -> float:
        """Quote-asset (USDT) 24h volume from ``futures_ticker_24hr``."""
        await self._rl.acquire()
        try:
            data = await self._client.futures_ticker(symbol=symbol)
            # API returns ``quoteVolume`` (in USDT) for USDT-M pairs.
            return float(data.get("quoteVolume") or 0.0)
        except Exception as e:
            self.log.warning("get_24h_volume %s failed: %s", symbol, e)
            return 0.0

    # ───── Realised cost accounting hooks ─────
    async def get_realised_commission(
        self, symbol: str, since_ms: int, until_ms: int = 0,
    ) -> float:
        """Sum commissions on user trades for *symbol* in the window.

        Uses ``futures_account_trades`` (also called ``GET /fapi/v1/userTrades``).
        Returns the absolute USDT-equivalent commission summed over
        every fill the broker has logged for the position lifetime.
        """
        params = {"symbol": symbol, "startTime": int(since_ms)}
        if until_ms:
            params["endTime"] = int(until_ms)
        await self._rl.acquire(weight=5)
        try:
            trades = await self._client.futures_account_trades(**params)
        except BinanceAPIException as e:
            self.log.warning("get_realised_commission %s: %s", symbol, e)
            return 0.0
        total = 0.0
        for t in trades or []:
            try:
                total += float(t.get("commission") or 0.0)
            except (TypeError, ValueError):
                continue
        return total

    async def get_funding_paid(
        self, symbol: str, since_ms: int, until_ms: int = 0,
    ) -> float:
        """Net funding paid in USDT over the window.

        Reads ``futures_income_history`` filtered by
        ``incomeType="FUNDING_FEE"``.  Sign convention: positive value
        means we PAID funding; negative means we received it (because
        Binance reports the income side; we negate so PnL math reads
        cleanly as ``pnl_net = pnl_gross - fees - funding``).
        """
        params = {
            "symbol": symbol,
            "incomeType": "FUNDING_FEE",
            "startTime": int(since_ms),
        }
        if until_ms:
            params["endTime"] = int(until_ms)
        await self._rl.acquire(weight=10)
        try:
            rows = await self._client.futures_income_history(**params)
        except BinanceAPIException as e:
            self.log.warning("get_funding_paid %s: %s", symbol, e)
            return 0.0
        net = 0.0
        for r in rows or []:
            try:
                net += float(r.get("income") or 0.0)
            except (TypeError, ValueError):
                continue
        # Binance's "income" is positive when we received funding; we
        # want the COST side, so negate.
        return -net

    # Phase C3 — internal hook for the rejection circuit-breaker.
    def _record_rejection(self, reason: str) -> None:
        if self._rejection_counter is None:
            return
        try:
            self._rejection_counter.record(reason)
        except Exception as e:  # pragma: no cover — defensive
            self.log.warning("rejection_counter.record raised: %s", e)

    def attach_rejection_counter(self, counter) -> None:
        """Wire the engine's RejectionCounter into the broker so every
        BinanceAPIException is recorded.  Called by LiveEngine after
        global_risk is built (the counter's on_trip is bound to
        ``global_risk.trip_kill_switch``)."""
        self._rejection_counter = counter
