from abc import ABC, abstractmethod
from typing import Optional

class IStrategy(ABC):
    @abstractmethod
    def generate_signal(self, symbol: str) -> Optional[str]: pass
    
    # Add other methods if needed
