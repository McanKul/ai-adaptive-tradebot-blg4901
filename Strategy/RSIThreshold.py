"""
Strategy/RSIThreshold.py
========================
RSI Threshold Strategy - Unified interface for both Live and Backtest.

This strategy uses the UNIFIED IStrategy interface, meaning:
- Works with both live trading and backtesting
- Can use generate_signal() for simple signal generation
- Can use on_bar() for advanced order control
- Supports TP/SL via ExitManager for intrabar exits
"""
from Strategy.binary_base_strategy import BinaryBaseStrategy
from Backtest.exit_manager import ExitManager, ExitConfig
from Interfaces.IStrategy import StrategyDecision
from Interfaces.orders import Order, OrderType, OrderSide
import talib
import numpy as np
from typing import Optional, List, Dict, Any


class Strategy(BinaryBaseStrategy):
    """
    RSI Threshold Strategy with unified Live/Backtest support.
    
    This strategy implements both interfaces:
    - generate_signal(): Returns "+1"/"-1"/None for simple usage
    - on_bar(): Returns StrategyDecision for advanced usage
    
    SIZING:
    =======
    Strategy outputs "intent" with placeholder qty (position_size).
    Actual qty is computed at runtime by the engine using sizing_config:
    - FIXED_QTY: Use explicit quantity
    - NOTIONAL_USD: qty = notional / price
    - MARGIN_USD: qty = (margin * leverage) / price
    
    Parameters:
    -----------
    bars : pd.DataFrame or None
        OHLCV data (for backtest buffer mode)
    bar_store : BarStore or None
        BarStore reference (for live mode)
    rsi_period : int
        RSI calculation period (default: 14)
    rsi_overbought : int
        RSI level above which to go short (default: 45)
    rsi_oversold : int
        RSI level below which to go long (default: 40)
    position_size : float
        Default position size for orders (placeholder, engine may override)
    sizing_config : SizingConfig or None
        Optional sizing config (engine uses this to compute actual qty)
    
    Exit Parameters (optional):
    ---------------------------
    take_profit_pct : float
        Take profit as percentage of entry price (e.g., 0.02 = 2%)
    stop_loss_pct : float
        Stop loss as percentage of entry price (e.g., 0.01 = 1%)
    take_profit_usd : float
        Take profit in USD
    stop_loss_usd : float
        Stop loss in USD
    trailing_stop_pct : float
        Trailing stop as percentage (e.g., 0.015 = 1.5%)
    max_holding_bars : int
        Maximum bars to hold a position
    leverage : float
        Leverage for P&L calculations (default: 1.0)
    """
    
    def __init__(
        self, 
        bars=None,
        bar_store=None,
        rsi_period: int = 14, 
        rsi_overbought: int = 45, 
        rsi_oversold: int = 40,
        position_size: float = 1.0,
        sizing_config=None,  # Optional SizingConfig for engine-side sizing
        # Exit manager params
        take_profit_pct: Optional[float] = None,
        stop_loss_pct: Optional[float] = None,
        take_profit_usd: Optional[float] = None,
        stop_loss_usd: Optional[float] = None,
        trailing_stop_pct: Optional[float] = None,
        max_holding_bars: Optional[int] = None,
        leverage: float = 1.0,
        **kw
    ):
        super().__init__(
            bars=bars,
            bar_store=bar_store,
            rsi_period=rsi_period, 
            overbought=rsi_overbought, 
            oversold=rsi_oversold, 
            **kw
        )
        self.rsi_period = rsi_period
        self.ob = rsi_overbought
        self.os = rsi_oversold
        self.position_size = position_size
        self.sizing_config = sizing_config  # Stored for reference, engine uses this
        self.leverage = leverage
        
        # Store exit params for get_exit_params()
        self._exit_params = {
            'tp_pct': take_profit_pct,
            'sl_pct': stop_loss_pct,
            'tp_usd': take_profit_usd,
            'sl_usd': stop_loss_usd,
            'trailing_stop_pct': trailing_stop_pct,
        }
        
        # Setup exit manager if any exit params provided
        self.exit_manager: Optional[ExitManager] = None
        if any([take_profit_pct, stop_loss_pct, take_profit_usd, 
                stop_loss_usd, trailing_stop_pct, max_holding_bars]):
            exit_config = ExitConfig(
                take_profit_pct=take_profit_pct,
                stop_loss_pct=stop_loss_pct,
                take_profit_usd=take_profit_usd,
                stop_loss_usd=stop_loss_usd,
                trailing_stop_pct=trailing_stop_pct,
                max_holding_bars=max_holding_bars,
                leverage=leverage,
            )
            self.exit_manager = ExitManager(exit_config)
        
        # Track current position for on_bar logic
        self._current_position = 0.0

    def _live_signal(self, o, h, l, c, v) -> Optional[str]:
        """Generate raw RSI signal: "+1" (long), "-1" (short), None (no signal)"""
        if c.size < self.rsi_period:
            return None
        rsi = talib.RSI(c, timeperiod=self.rsi_period)[-1]
        if np.isnan(rsi):
            return None
        if rsi > self.ob:
            return "-1"
        if rsi < self.os:
            return "+1"
        return None
    
    def on_bar(self, bar, ctx) -> StrategyDecision:
        """
        Process a bar and return a trading decision.
        
        This is the advanced interface for backtest compatibility.
        Uses the context to access historical data and portfolio state.
        
        Args:
            bar: The completed bar
            ctx: Strategy context with market state
            
        Returns:
            StrategyDecision with orders and metadata
        """
        orders: List[Order] = []
        features: Dict[str, Any] = {}
        
        # Get OHLCV from context
        ohlcv = ctx.get_ohlcv()
        closes = ohlcv.get("close", [])
        
        if len(closes) < self.rsi_period:
            return StrategyDecision.no_action()
        
        # Calculate RSI
        c = np.array(closes, dtype=float)
        rsi = talib.RSI(c, timeperiod=self.rsi_period)[-1]
        
        if np.isnan(rsi):
            return StrategyDecision.no_action()
        
        # Store features for debugging
        features["rsi"] = float(rsi)
        features["rsi_period"] = self.rsi_period
        features["overbought"] = self.ob
        features["oversold"] = self.os
        
        # Get current position from context
        current_pos = ctx.position
        
        # Generate signal
        signal = None
        if rsi > self.ob and current_pos >= 0:
            # Overbought - go short (or close long)
            signal = "-1"
            if current_pos > 0:
                # Close long first
                orders.append(Order(
                    symbol=bar.symbol,
                    side=OrderSide.SELL,
                    order_type=OrderType.MARKET,
                    quantity=abs(current_pos),
                    timestamp_ns=bar.timestamp_ns,
                    strategy_id="RSI_EXIT",
                    reduce_only=True,
                ))
            # Open short
            orders.append(Order(
                symbol=bar.symbol,
                side=OrderSide.SELL,
                order_type=OrderType.MARKET,
                quantity=self.position_size,
                timestamp_ns=bar.timestamp_ns,
                strategy_id="RSI_ENTRY",
                metadata=self._exit_params.copy(),
            ))
            
        elif rsi < self.os and current_pos <= 0:
            # Oversold - go long (or close short)
            signal = "+1"
            if current_pos < 0:
                # Close short first
                orders.append(Order(
                    symbol=bar.symbol,
                    side=OrderSide.BUY,
                    order_type=OrderType.MARKET,
                    quantity=abs(current_pos),
                    timestamp_ns=bar.timestamp_ns,
                    strategy_id="RSI_EXIT",
                    reduce_only=True,
                ))
            # Open long
            orders.append(Order(
                symbol=bar.symbol,
                side=OrderSide.BUY,
                order_type=OrderType.MARKET,
                quantity=self.position_size,
                timestamp_ns=bar.timestamp_ns,
                strategy_id="RSI_ENTRY",
                metadata=self._exit_params.copy(),
            ))
        
        return StrategyDecision(
            orders=orders,
            signal=signal,
            features=features,
            metadata=self._exit_params.copy(),
        )
    
    def get_exit_params(self) -> Dict[str, Any]:
        """Return exit parameters for tick-level TP/SL checking."""
        return {k: v for k, v in self._exit_params.items() if v is not None}
    
    def get_exit_manager(self) -> Optional[ExitManager]:
        """Return the exit manager if configured."""
        return self.exit_manager
    
    def reset(self) -> None:
        """Reset strategy state for a new run."""
        self._current_position = 0.0
        if self.exit_manager:
            self.exit_manager.reset()
