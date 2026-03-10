"""
Strategy/binary_base_strategy.py
=================================
Base strategy class for binary signal strategies.

This is the foundation for strategies that generate simple +1/-1 signals.
Extends IStrategy to provide the unified interface for both live and backtest.

UNIFIED INTERFACE:
==================
- generate_signal(symbol) -> Optional[str]: Returns "+1", "-1", or None
- on_bar(bar, ctx) -> StrategyDecision: Returns full decision with orders

Subclasses should implement:
- _live_signal(o, h, l, c, v) -> Optional[str]: Core signal logic
"""
from abc import abstractmethod
from typing import Optional, Dict, Any
import numpy as np
import pandas as pd
from Interfaces.IStrategy import IStrategy, StrategyDecision
# import time


class BinaryBaseStrategy(IStrategy):
    """
    Base class for binary (+1/-1) signal strategies.
    
    Provides data access from either:
    - BarStore (live trading with streaming data)
    - DataFrame buffer (backtest with preloaded data)
    
    Subclasses implement _live_signal() which receives OHLCV arrays
    and returns a signal string.
    """
    
    def __init__(
        self,
        bar_store=None, 
        bars: pd.DataFrame = None,
        **params,
    ):
        self.bar_store = bar_store
        self.buf = bars
        self.params = params
        
    @abstractmethod
    def _live_signal(
        self,
        o: np.ndarray,
        h: np.ndarray,
        l: np.ndarray,
        c: np.ndarray,
        v: np.ndarray,
    ) -> Optional[str]:
        """
        Generate a trading signal from OHLCV data.
        
        Args:
            o: Open prices array
            h: High prices array
            l: Low prices array
            c: Close prices array
            v: Volume array
            
        Returns:
            "+1" for buy, "-1" for sell, None for no action
        """
        pass

    def generate_signal(self, symbol: str = None) -> Optional[str]:
        """
        Generate a signal for the given symbol.
        
        This is the legacy interface used by live trading.
        It fetches data from BarStore or buffer and calls _live_signal.
        
        Args:
            symbol: Trading symbol (optional if using buffer)
            
        Returns:
            "+1" for buy, "-1" for sell, None for no action
        """
        # Priority: BarStore > Buffer
        if self.bar_store and symbol:
            # Fetch from bar_store (BarStore returns list[float])
            data = self.bar_store.get_ohlcv(symbol, self.params.get("timeframe", "1m"))
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
    
    def on_bar(self, bar, ctx) -> StrategyDecision:
        """
        Process a bar and return a decision.
        
        Default implementation uses generate_signal() and wraps in StrategyDecision.
        Subclasses can override for more complex behavior.
        
        Args:
            bar: The completed bar
            ctx: Strategy context
            
        Returns:
            StrategyDecision with signal
        """
        # Use context's bar_store if we don't have one
        if self.bar_store is None and ctx.bar_store is not None:
            self.bar_store = ctx.bar_store
        
        signal = self.generate_signal(bar.symbol if hasattr(bar, 'symbol') else ctx.symbol)
        return StrategyDecision.from_signal(signal)
    
    def reset(self) -> None:
        """Reset strategy state for a new run."""
        pass
    
    def get_exit_params(self) -> Dict[str, Any]:
        """Get exit parameters. Override in subclass if needed."""
        return {}
    
    def get_exit_manager(self):
        """Get exit manager. Override in subclass if needed."""
        return getattr(self, 'exit_manager', None)
