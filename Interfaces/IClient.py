from abc import ABC, abstractmethod
from typing import Dict, Any, List

class IClient(ABC):
    @abstractmethod
    async def futures_exchange_info(self) -> Dict[str, Any]: ...

    @abstractmethod
    async def futures_mark_price(self, symbol: str) -> Dict[str, Any]: ...

    @abstractmethod
    async def futures_create_order(self, **kwargs) -> Dict[str, Any]: ...
    
    @abstractmethod
    async def futures_cancel_all_open_orders(self, symbol: str) -> Any: ...
    
    # Add other methods as needed by PositionManager/Streamer usage
    # Streamer uses: futures_klines, futures_socket (via BSM, but BSM takes client)
    # If BSM is used, client must be compatible or we need IStreamer to handle connection details internally without exposing BSM.
    # live/streamer.py uses BinanceSocketManager(self.client). This implies self.client is a real Binance Client or a Mock that BSM accepts.
    # User said "Client will be implemented... contains implementation for only Binance for now".
    # We might need to keep the underlying client accessible for BSM or wrap BSM too.
    # For now, let's include methods used directly.
    
    @abstractmethod
    async def futures_klines(self, symbol: str, interval: str, limit: int) -> List[Any]: ...
    
    @abstractmethod
    async def close_connection(self): ...
