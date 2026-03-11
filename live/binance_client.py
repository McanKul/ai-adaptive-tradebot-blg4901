from Interfaces.IClient import IClient
from binance.client import AsyncClient
from typing import Dict, Any, List

class BinanceClient(IClient):
    """Wrapper for binance.AsyncClient implementing IClient"""
    
    def __init__(self, client: AsyncClient):
        self._client = client

    async def futures_exchange_info(self) -> Dict[str, Any]:
        return await self._client.futures_exchange_info()

    async def futures_mark_price(self, symbol: str) -> Dict[str, Any]:
        return await self._client.futures_mark_price(symbol=symbol)

    async def futures_create_order(self, **kwargs) -> Dict[str, Any]:
        return await self._client.futures_create_order(**kwargs)
        
    async def futures_cancel_all_open_orders(self, symbol: str) -> Any:
        return await self._client.futures_cancel_all_open_orders(symbol=symbol)

    async def futures_cancel_order(self, symbol: str, orderId: int) -> Any:
        return await self._client.futures_cancel_order(symbol=symbol, orderId=orderId)

    async def futures_get_open_orders(self, symbol: str) -> List[Dict[str, Any]]:
        return await self._client.futures_get_open_orders(symbol=symbol)

    async def futures_position_information(self, symbol: str) -> List[Dict[str, Any]]:
        return await self._client.futures_position_information(symbol=symbol)

    async def futures_account_balance(self) -> List[Dict[str, Any]]:
        return await self._client.futures_account_balance()

    async def futures_change_margin_type(self, symbol: str, marginType: str) -> Any:
        return await self._client.futures_change_margin_type(symbol=symbol, marginType=marginType)

    async def futures_change_leverage(self, symbol: str, leverage: int) -> Any:
        return await self._client.futures_change_leverage(symbol=symbol, leverage=leverage)

    async def futures_klines(self, symbol: str, interval: str, limit: int) -> List[Any]:
        return await self._client.futures_klines(symbol=symbol, interval=interval, limit=limit)
        
    async def close_connection(self):
        await self._client.close_connection()

    @property
    def raw_client(self):
        return self._client
