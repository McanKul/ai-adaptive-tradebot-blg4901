from abc import ABC, abstractmethod
import asyncio
from typing import List, Any

class IStreamer(ABC):
    @abstractmethod
    async def start(self): ...

    @abstractmethod
    async def stop(self): ...

    @abstractmethod
    def get_queue(self) -> "asyncio.Queue": ...

    @abstractmethod
    async def preload_history(self, symbols: List[str], intervals: List[str], limit: int, batch: int): ...
    
    # Static method in implementation, but good to have in protocol if we want interchangeable factories
    # For now, let's keep it abstract if we expect instances to handle it, or just rely on concrete implementation for static utility
    # User said: "implement the IStreamer so that live/streamer.py file's definition will meet the declarations"
    # live/streamer has resolve_symbols as static.
    # Python abstract static methods exist.
    # @staticmethod
    # @abstractmethod
    # async def resolve_symbols(client, coins_spec): ...
