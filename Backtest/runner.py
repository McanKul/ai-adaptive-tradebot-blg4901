"""
Backtest/runner.py
==================
Backtest runner for executing single and batch backtests.

Design decisions:
- run_once(): Single backtest run with structured result
- Accepts strategy class and params, instantiates internally
- Returns BacktestResult with params included for tracking
- Designed for use by batch/scoring system
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional, Type, Callable
import logging
import copy

from Interfaces.strategy_adapter import IBacktestStrategy
from Interfaces.metrics_interface import BacktestResult

from Backtest.engine import BacktestEngine, EngineConfig
from Backtest.realism_config import RealismConfig

log = logging.getLogger(__name__)


@dataclass
class DataConfig:
    """Data configuration for backtest."""
    tick_data_dir: str
    symbols: List[str]
    start_ts_ns: Optional[int] = None
    end_ts_ns: Optional[int] = None
    
    # Bar configuration
    bar_type: str = "time"
    timeframe: str = "1m"
    tick_threshold: int = 100
    volume_threshold: float = 1000.0
    dollar_threshold: float = 100000.0


@dataclass
class BacktestConfig:
    """Full backtest configuration."""
    # Data
    data: DataConfig
    
    # Portfolio
    initial_capital: float = 10000.0
    margin_requirement: float = 0.1
    
    # Costs (basis points)
    taker_fee_bps: float = 4.0
    maker_fee_bps: float = 2.0
    slippage_bps: float = 1.0
    spread_bps: float = 2.0
    
    # NEW: Latency simulation
    latency_ns: int = 0  # Base latency
    latency_jitter_ns: int = 0  # Max random jitter
    
    # NEW: Partial fill settings
    enable_partial_fills: bool = False
    liquidity_scale: float = 10.0
    min_fill_ratio: float = 0.1
    
    # Risk
    max_position_size: float = 10.0
    max_position_notional: float = 100000.0
    max_daily_loss: float = 1000.0
    max_drawdown: float = 0.2
    
    # Execution
    use_bar_close: bool = True
    
    # NEW: Close positions at end (AFML bet accounting)
    close_positions_at_end: bool = False
    
    # Determinism
    random_seed: int = 42
    
    # BarStore
    bar_store_maxlen: int = 600
    
    # NEW: Leverage mode settings
    leverage_mode: str = "spot"  # "spot" or "margin"
    leverage: int = 1  # Leverage multiplier (only for margin mode)
    maintenance_margin_ratio: float = 0.5  # Maintenance margin as fraction of initial
    
    # NEW: Tick-level exit checking
    enable_tick_exit: bool = True  # Enable tick-level TP/SL checking

    # Exit rules (strategy-independent TP/SL safety net)
    tp_pct: Optional[float] = None
    sl_pct: Optional[float] = None
    trailing_stop_pct: Optional[float] = None
    max_holding_bars: Optional[int] = None

    # NEW: Realism configuration (Part A refactor)
    realism: RealismConfig = field(default_factory=RealismConfig)

    def to_engine_config(self) -> EngineConfig:
        """Convert to EngineConfig for BacktestEngine."""
        return EngineConfig(
            tick_data_dir=self.data.tick_data_dir,
            symbols=self.data.symbols,
            start_ts_ns=self.data.start_ts_ns,
            end_ts_ns=self.data.end_ts_ns,
            bar_type=self.data.bar_type,
            timeframe=self.data.timeframe,
            tick_threshold=self.data.tick_threshold,
            volume_threshold=self.data.volume_threshold,
            dollar_threshold=self.data.dollar_threshold,
            initial_capital=self.initial_capital,
            margin_requirement=self.margin_requirement,
            taker_fee_bps=self.taker_fee_bps,
            maker_fee_bps=self.maker_fee_bps,
            slippage_bps=self.slippage_bps,
            spread_bps=self.spread_bps,
            latency_ns=self.latency_ns,
            latency_jitter_ns=self.latency_jitter_ns,
            enable_partial_fills=self.enable_partial_fills,
            liquidity_scale=self.liquidity_scale,
            min_fill_ratio=self.min_fill_ratio,
            max_position_size=self.max_position_size,
            max_position_notional=self.max_position_notional,
            max_daily_loss=self.max_daily_loss,
            max_drawdown=self.max_drawdown,
            use_bar_close=self.use_bar_close,
            close_positions_at_end=self.close_positions_at_end,
            random_seed=self.random_seed,
            bar_store_maxlen=self.bar_store_maxlen,
            # NEW: Leverage settings
            leverage_mode=self.leverage_mode,
            leverage=self.leverage,
            maintenance_margin_ratio=self.maintenance_margin_ratio,
            # NEW: Tick exit
            enable_tick_exit=self.enable_tick_exit,
            # Exit rules
            tp_pct=self.tp_pct,
            sl_pct=self.sl_pct,
            trailing_stop_pct=self.trailing_stop_pct,
            max_holding_bars=self.max_holding_bars,
            # NEW: Realism config passthrough
            realism=self.realism,
        )


# Type alias for strategy factory
StrategyFactory = Callable[[Dict[str, Any]], IBacktestStrategy]


class BacktestRunner:
    """
    Runner for executing backtests.
    
    Supports:
    - Single runs with run_once()
    - Strategy factories for parameter instantiation
    - Result tracking with params included
    """
    
    def __init__(self, config: BacktestConfig):
        """
        Initialize BacktestRunner.
        
        Args:
            config: BacktestConfig with all settings
        """
        self.config = config
        self._run_count = 0
    
    def run_once(
        self,
        strategy_factory: StrategyFactory,
        params: Dict[str, Any],
        strategy_name: str = ""
    ) -> BacktestResult:
        """
        Run a single backtest.
        
        Args:
            strategy_factory: Callable that creates strategy from params
            params: Strategy parameters
            strategy_name: Optional name for identification
            
        Returns:
            BacktestResult with params and metrics
        """
        self._run_count += 1
        
        log.info(f"Starting backtest run #{self._run_count} with params: {params}")
        
        # Create engine
        engine_config = self.config.to_engine_config()
        engine = BacktestEngine(engine_config)
        
        # Create strategy
        strategy = strategy_factory(params)
        
        # Configure metrics
        engine._initialize()
        engine.metrics.strategy_name = strategy_name or strategy.__class__.__name__
        engine.metrics.params = copy.deepcopy(params)
        engine.metrics.data_config = {
            "tick_data_dir": self.config.data.tick_data_dir,
            "symbols": self.config.data.symbols,
            "start_ts_ns": self.config.data.start_ts_ns,
            "end_ts_ns": self.config.data.end_ts_ns,
            "bar_type": self.config.data.bar_type,
            "timeframe": self.config.data.timeframe,
        }
        
        # Run backtest
        result = engine.run(strategy)
        
        log.info(
            f"Run #{self._run_count} complete: "
            f"return={result.total_return_pct:.2%}, "
            f"sharpe={result.sharpe_ratio:.2f}, "
            f"drawdown={result.max_drawdown_pct:.1f}%"
        )
        
        return result
    
    def run_with_class(
        self,
        strategy_class: Type[IBacktestStrategy],
        params: Dict[str, Any],
        strategy_name: str = ""
    ) -> BacktestResult:
        """
        Run backtest with a strategy class.
        
        Convenience method that creates a factory from the class.
        
        Args:
            strategy_class: Strategy class to instantiate
            params: Constructor parameters
            strategy_name: Optional name
            
        Returns:
            BacktestResult
        """
        def factory(p: Dict[str, Any]) -> IBacktestStrategy:
            return strategy_class(**p)
        
        return self.run_once(
            factory,
            params,
            strategy_name or strategy_class.__name__
        )
    
    @property
    def run_count(self) -> int:
        """Number of backtests run."""
        return self._run_count


def create_runner(
    tick_data_dir: str,
    symbols: List[str],
    initial_capital: float = 10000.0,
    timeframe: str = "1m",
    **kwargs
) -> BacktestRunner:
    """
    Convenience factory for creating a BacktestRunner.
    
    Args:
        tick_data_dir: Directory with tick CSV files
        symbols: List of symbols to backtest
        initial_capital: Starting capital
        timeframe: Bar timeframe
        **kwargs: Additional config options
        
    Returns:
        Configured BacktestRunner
    """
    data_config = DataConfig(
        tick_data_dir=tick_data_dir,
        symbols=symbols,
        timeframe=timeframe,
        start_ts_ns=kwargs.pop("start_ts_ns", None),
        end_ts_ns=kwargs.pop("end_ts_ns", None),
        bar_type=kwargs.pop("bar_type", "time"),
    )
    
    config = BacktestConfig(
        data=data_config,
        initial_capital=initial_capital,
        **kwargs
    )
    
    return BacktestRunner(config)
