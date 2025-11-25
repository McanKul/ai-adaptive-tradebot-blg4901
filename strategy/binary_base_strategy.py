from abc import abstractmethod
from typing import Optional
import numpy as np
import pandas as pd
from Strategy.IStrategy import IStrategy

class BinaryBaseStrategy(IStrategy):
    def __init__(
        self,
        #bar_store: BarStore, #todo strategyler tek tek barlar üzerinde çalışmaması için bare storedan referans verilecek
        bars: pd,
        **params,
    ):
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
        """"+1" | "-1" | None"""

    def generate_signal(self) -> Optional[str]:
        buf = self.buf

        o = np.asarray(buf["open"],   dtype=float)
        h = np.asarray(buf["high"],   dtype=float)
        l = np.asarray(buf["low"],    dtype=float)
        c = np.asarray(buf["close"],  dtype=float)
        v = np.asarray(buf["volume"], dtype=float)
        return self._live_signal(o, h, l, c, v)
    