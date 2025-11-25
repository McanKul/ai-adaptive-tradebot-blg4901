from Strategy.binary_base_strategy import BinaryBaseStrategy
import talib
import numpy as np

class Strategy(BinaryBaseStrategy):
    def __init__(self, bars, rsi_period: int = 14, rsi_overbought: int = 45, rsi_oversold: int = 40, **kw):
      super().__init__(bars=bars,rsi_period=rsi_period, overbought=rsi_overbought, oversold=rsi_oversold, **kw)
      self.rsi_period = rsi_period
      self.ob = rsi_overbought
      self.os = rsi_oversold

    def _live_signal(self, o, h, l, c, v):
      if c.size < self.rsi_period:
         return None
      rsi = talib.RSI(c, timeperiod=self.rsi_period)[-1]
      print(f"debug rsi: {rsi}")
      if np.isnan(rsi):
         return None
      if rsi > self.ob:
         return -1
      if rsi < self.os:
         return +1
      return None
