from Interfaces.IStreamer import IStreamer
import asyncio
from typing import List

class Streamer(IStreamer):
    """
    Mock Streamer meeting IStreamer interface requirements.
    """
    def __init__(self):
        self.queue = asyncio.Queue()

    @staticmethod
    def fetch_kline(client, sym, tf, limit):
        # Implementation to satisfy old static usage if needed, or pass
        return sym, tf, []

    async def start(self):
        pass

    async def stop(self):
        pass

    def get_queue(self) -> asyncio.Queue:
        return self.queue
        
    async def preload_history(self, symbols: List[str], intervals: List[str], limit: int, batch: int):
        pass
        
    # If we want to support the static resolve_symbols from interface (if we added it there)
    # logic here needs to match IStreamer definition if strictly enforced by abc
    # But currently IStreamer.py defined preload_history etc as instance methods.
