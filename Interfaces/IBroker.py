from abc import ABC, abstractmethod
from Interfaces.IClient import IClient

class IBroker(ABC):
    """Exchange-agnostic broker interface."""
    
    @property
    @abstractmethod
    def client(self) -> IClient: ...

    @abstractmethod
    async def get_mark_price(self, symbol: str) -> float: ...
    # ---- order & position ----
    @abstractmethod
    async def market_order(self, symbol: str, side: str, qty: float): ...

    @abstractmethod
    async def close_position(self, symbol: str): ...

    @abstractmethod
    async def position_amt(self, symbol: str) -> float: ...

    # ---- settings ----
    @abstractmethod
    async def ensure_isolated_margin(self, symbol: str): ...

    @abstractmethod
    async def set_leverage(self, symbol: str, leverage: int): ...

    # ---- SL / TP ----
    @abstractmethod
    async def place_stop_market(self, symbol: str, side: str, stop_price: float): ...

    @abstractmethod
    async def place_take_profit(self, symbol: str, side: str, stop_price: float): ...
