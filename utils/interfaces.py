from abc import ABC, abstractmethod
import asyncio
from typing import Optional, Dict
import pandas as pd

class IStrategy(ABC):
    @abstractmethod
    def update_bar(self, symbol: str, bar: dict) -> None: pass

    @abstractmethod
    def generate_signal(self, symbol: str) -> Optional[str]: pass

    # Optional extended interface 
    # @staticmethod
    # @abstractmethod
    # def generate_signals(df: pd.DataFrame) -> pd.Series: pass
    # @abstractmethod
    # def sl_pct(self) -> float: pass


class IBroker(ABC):
    """Exchange-agnostic broker interface."""
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

class IStreamer(ABC):
    @abstractmethod
    async def start(self): ...

    @abstractmethod
    async def stop(self): ...

    @abstractmethod
    def get_queue(self) -> "asyncio.Queue": ...
