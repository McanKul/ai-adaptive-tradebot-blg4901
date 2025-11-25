from abc import ABC, abstractmethod
import pandas as pd
from typing import Optional

#abstract class for strategies
class IStrategy(ABC):
   #signal generation
   @abstractmethod
   def generate_signal(self, symbol: str) -> Optional[str]: pass
   #multi signal generations
   # @staticmethod
   # @abstractmethod
   # def generate_signals(df: pd.DataFrame) -> pd.Series: pass

   #for future devolopment
   # @abstractmethod
   # def sl_pct(self) -> float: pass
   
   # @abstractmethod
   # def tp_pct(self) -> float: pass
