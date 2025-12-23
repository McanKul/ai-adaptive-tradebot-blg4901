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
        
    async def futures_klines(self, symbol: str, interval: str, limit: int) -> List[Any]:
        return await self._client.futures_klines(symbol=symbol, interval=interval, limit=limit)
        
    async def close_connection(self):
        await self._client.close_connection()
        
    # Accessor for raw client if needed (e.g. for SocketManager)
    # The user said "Client will be implemented with an interface and an implementation".
    # SocketManager takes the raw client. Ideally SocketManager logic should also be wrapped or IStreamer should handle it.
    # streamer.py uses BinanceSocketManager(self.client). If self.client is BinanceClientWrapper, BSM won't work unless we expose raw.
    @property
    def raw_client(self):
        return self._client
