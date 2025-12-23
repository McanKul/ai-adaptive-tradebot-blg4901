from abc import abstractmethod
from typing import Optional
import numpy as np
import pandas as pd
from strategy.IStrategy import IStrategy
# import time

class BinaryBaseStrategy(IStrategy):
    def __init__(
        self,
        bar_store = None, 
        bars: pd.DataFrame = None,
        **params,
    ):
        self.bar_store = bar_store
        self.buf       = bars
        self.params    = params
        
    @abstractmethod
    def _live_signal(
        self,
        o: np.ndarray,
        h: np.ndarray,
        l: np.ndarray,
        c: np.ndarray,
        v: np.ndarray,
    ) -> Optional[str]:
        """+1 | -1 | None"""

    def generate_signal(self, symbol: str = None) -> Optional[str]:
        # Priority: BarStore > Buffer
        if self.bar_store and symbol:
            # Fetch from bar_store (BarStore returns list[float])
            data = self.bar_store.get_ohlcv(symbol, self.params.get("timeframe", "1m")) # Default or config timeframe
            if not data or len(data["close"]) < 2: 
                return None
            
            o = np.array(data["open"], dtype=float)
            h = np.array(data["high"], dtype=float)
            l = np.array(data["low"], dtype=float)
            c = np.array(data["close"], dtype=float)
            v = np.array(data["volume"], dtype=float)
            
            return self._live_signal(o, h, l, c, v)

        elif self.buf is not None:
            buf = self.buf
            o = np.asarray(buf["open"],   dtype=float)
            h = np.asarray(buf["high"],   dtype=float)
            l = np.asarray(buf["low"],    dtype=float)
            c = np.asarray(buf["close"],  dtype=float)
            v = np.asarray(buf["volume"], dtype=float)
            return self._live_signal(o, h, l, c, v)
        
        return None
