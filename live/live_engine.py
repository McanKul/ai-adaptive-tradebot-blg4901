"""
live/live_engine.py
===================
Live trading engine using the unified strategy interface.

This engine runs strategies in real-time with live market data.
It uses the SAME strategy classes as backtesting, providing
a consistent development experience.

KEY DIFFERENCES FROM BACKTEST:
- Uses async streaming for real-time data
- Connects to real broker for order execution
- No slippage/cost simulation (real execution)
- Position management via PositionManager

UNIFIED STRATEGY USAGE:
======================
The live engine uses the same IStrategy interface as backtest:
- Strategies implement generate_signal() or on_bar()
- The adapter converts outputs to orders
- TP/SL is handled by the broker, not simulated
"""
import asyncio
from typing import Optional
from utils.bar_store import BarStore
from Interfaces.IBroker import IBroker
from Interfaces.INewsSource import INewsSource
from Interfaces.ISentimentAnalyzer import ISentimentAnalyzer
from Interfaces.ISignalCombiner import ISignalCombiner
from Interfaces.strategy_adapter import adapt_strategy_output, AdaptedOutput, StrategyContext
from live.position_manager import PositionManager
from live.broker_binance import BinanceBroker
from live.streamer import Streamer
from utils.logger import setup_logger
from Strategy.binary_base_strategy import BinaryBaseStrategy

# Optional: Import news components (will be None if not used)
try:
    from news.news_engine import NewsEngine
except ImportError:
    NewsEngine = None
# Mock Strategy Loader for now - in real implementation this might be dynamic
def load_strategy_instance(strategy_cls, config, bar_store):
    return strategy_cls(bar_store=bar_store, **config.get("params", {}))

log = setup_logger("LiveEngine")


def load_strategy_instance(strategy_cls, config, bar_store):
    """Load strategy instance with configuration."""
    return strategy_cls(bar_store=bar_store, **config["params"])


class LiveEngine:
    """
    Live trading engine that coordinates strategy signals with optional news sentiment.
    
    When news components are provided, signals are combined:
    - Both BUY (strategy + sentiment) → LONG
    - Both SELL (strategy + sentiment) → SHORT
    - Mixed signals → NO ACTION
    """
    
    def __init__(
        self,
        config,
        broker: IBroker,
        strategy_cls,
        news_source: Optional[INewsSource] = None,
        sentiment_analyzer: Optional[ISentimentAnalyzer] = None,
        signal_combiner: Optional[ISignalCombiner] = None,
        news_refresh_interval: int = 300
    ):
        """
        Initialize the live engine.
        
        Args:
            config: Strategy configuration dict or list
            broker: Broker implementation for trading
            strategy_cls: Strategy class to instantiate
            news_source: Optional news source for sentiment analysis
            sentiment_analyzer: Optional sentiment analyzer (requires news_source)
            signal_combiner: Optional signal combiner (requires sentiment_analyzer)
            news_refresh_interval: Seconds between news sentiment refreshes
    Live trading engine with unified strategy support.
    
    Uses the same strategy classes as BacktestEngine for consistency.
    Strategies can implement either:
    - generate_signal(symbol) -> "+1"/"-1"/None
    - on_bar(bar, ctx) -> StrategyDecision/List[Order]
    
    The engine handles both interfaces via the strategy adapter.
    """
    
    def __init__(self, config, broker: IBroker, strategy_cls):
        """
        Initialize LiveEngine.
        
        Args:
            config: Configuration dict with strategy params
            broker: IBroker implementation for order execution
            strategy_cls: Strategy class to instantiate
        """
        self.cfg = config
        self.broker = broker
        self.bar_store = BarStore()

        # Configure Strategies
        self.strategies = []
        strategy_configs = [self.cfg] if isinstance(self.cfg, dict) else self.cfg

        for scfg in strategy_configs:
            instance = load_strategy_instance(strategy_cls, scfg, self.bar_store)
            self.strategies.append({**scfg, "instance": instance})

        self.pos_mgr = PositionManager(self.broker, base_capital=10.0, max_concurrent=1)

        # — Extract timeframes —
        self.timeframes = list(set(s["timeframe"] for s in self.strategies))
        self.streamer = None
        self.symbols = []

        # — News Sentiment Integration —
        self.news_engine = None
        self.signal_combiner = signal_combiner
        
        if news_source and sentiment_analyzer:
            if NewsEngine:
                self.news_engine = NewsEngine(
                    news_source=news_source,
                    sentiment_analyzer=sentiment_analyzer,
                    refresh_interval=news_refresh_interval
                )
                log.info("News sentiment integration enabled")
            else:
                log.warning("NewsEngine not available, running without sentiment")
        
        if self.news_engine and not self.signal_combiner:
            log.warning("News engine provided but no signal combiner - sentiment will be logged but not used for trading")

    async def _get_combined_signal(self, symbol: str, strategy_signal: Optional[int]) -> Optional[int]:
        """
        Combine strategy signal with news sentiment if available.
        
        Args:
            symbol: Trading symbol
            strategy_signal: Signal from strategy (+1, -1, or None)
            
        Returns:
            Combined signal or original strategy signal if no sentiment
        """
        if not self.news_engine:
            return strategy_signal
        
        # Get sentiment score
        sentiment_score = await self.news_engine.get_sentiment(symbol)
        log.info("Symbol %s: strategy_signal=%s, sentiment=%.2f", 
                symbol, strategy_signal, sentiment_score)
        
        if not self.signal_combiner:
            # Just log sentiment, don't combine
            return strategy_signal
        
        # Combine signals
        combined = self.signal_combiner.combine(strategy_signal, sentiment_score)
        
        if combined != strategy_signal:
            log.info("Signal modified by sentiment: %s → %s", strategy_signal, combined)
        
        return combined

    async def run(self):
        """Main trading loop."""
        # 1) Resolve Symbols
        all_coins = []
        for s in self.strategies:
            all_coins.extend(s["coins"])

        
        # Support single strategy config or list
        strategy_configs = [self.cfg] if isinstance(self.cfg, dict) else self.cfg

        for scfg in strategy_configs:
            # Instantiate strategy
            instance = load_strategy_instance(strategy_cls, scfg, self.bar_store)
            self.strategies.append({**scfg, "instance": instance})
            
        self.pos_mgr = PositionManager(
            self.broker, 
            base_capital=config.get("base_capital", 10.0), 
            max_concurrent=config.get("max_concurrent", 1)
        )
        
        # Extract timeframes
        self.timeframes = list(set(s.get("timeframe", "1m") for s in self.strategies))
        self.streamer = None
        self.symbols = []

    async def run(self):
        """
        Main event loop for live trading.
        
        Processes bars as they stream in and executes strategy decisions.
        """
        # 1) Resolve Symbols
        all_coins = []
        for s in self.strategies:
            all_coins.extend(s.get("coins", []))
        
        self.symbols = await Streamer.resolve_symbols(
            self.broker.client, list(set(all_coins)))

        # 2) Create Streamer
        self.streamer = Streamer(
            self.broker.client,
            self.symbols,
            self.timeframes,
            bar_store=self.bar_store
        )

        # 3) Preload History
        await self.streamer.preload_history(
            self.symbols, self.timeframes,
            self.symbols, 
            self.timeframes,
            limit=100,
            batch=10
        )

        # 4) Prefetch news sentiment if enabled
        if self.news_engine:
            await self.news_engine.prefetch_symbols(self.symbols)

        # 5) Start Live Stream
        await self.streamer.start()
        log.info("Live Engine Started: %s symbols | tf=%s | sentiment=%s",
                 len(self.symbols), self.timeframes, 
                 "enabled" if self.news_engine else "disabled")
        log.info(
            "Live Engine Started: %s symbols | tf=%s",
            len(self.symbols), 
            self.timeframes
        )

        try:
            while True:
                bar_data = await self.streamer.get()
                sym = bar_data["s"]
                tf = bar_data["k"]["i"]

                for s in self.strategies:
                    if sym not in s["coins"] or tf != s["timeframe"]:
                        continue

                    inst = s["instance"]
                    strategy_sig = inst.generate_signal(sym)

                    # Combine with news sentiment if available
                    final_sig = await self._get_combined_signal(sym, strategy_sig)

                    if final_sig:
                        direction = 1 if final_sig == +1 or final_sig == "+1" else -1
                        exec_params = s.get("execution", {})

                        log.info(
                            "Opening position: %s direction=%d (strategy=%s, combined=%s)",
                            sym, direction, strategy_sig, final_sig
                        )

                        await self.pos_mgr.open_position(
                            sym, direction,
                            s.get("name", "UnknownStrategy"),
                            leverage=exec_params.get("leverage", 1),
                            sl_pct=exec_params.get("sl_pct", 1.0),
                            tp_pct=exec_params.get("tp_pct", 1.0),
                            expire_sec=exec_params.get("expire_sec", 3600),
                            timeframes=tf
                        )
                    else:
                        await self.pos_mgr.update_all()
        finally:
            await self.streamer.stop()
            log.info("Live Engine stopped")
                    # Check if this strategy cares about this symbol/tf
                    if sym not in s.get("coins", []) or tf != s.get("timeframe", "1m"):
                        continue
                    
                    await self._process_strategy_bar(s, sym, tf, bar_data)
                    
        finally:
            await self.streamer.stop()
    
    async def _process_strategy_bar(self, strategy_config: dict, symbol: str, timeframe: str, bar_data: dict):
        """
        Process a bar for a specific strategy.
        
        Uses the unified strategy adapter to handle both legacy
        generate_signal() and modern on_bar() interfaces.
        
        Args:
            strategy_config: Strategy configuration dict
            symbol: Trading symbol
            timeframe: Bar timeframe
            bar_data: Raw bar data from streamer
        """
        inst = strategy_config["instance"]
        exec_params = strategy_config.get("execution", {})
        
        # Create a minimal context for on_bar strategies
        ctx = StrategyContext(
            symbol=symbol,
            timeframe=timeframe,
            bar_store=self.bar_store,
            portfolio=None,  # Live doesn't have portfolio snapshot
            position=0.0,  # TODO: Get from position manager
            equity=0.0,
            cash=0.0,
            timestamp_ns=int(bar_data["k"]["t"] * 1_000_000) if "t" in bar_data.get("k", {}) else 0,
        )
        
        # Try modern on_bar first, fall back to generate_signal
        signal = None
        
        if hasattr(inst, 'on_bar') and callable(inst.on_bar):
            try:
                # Create a mock bar for on_bar
                from Interfaces.market_data import Bar
                k = bar_data.get("k", {})
                bar = Bar(
                    symbol=symbol,
                    timeframe=timeframe,
                    timestamp_ns=int(k.get("t", 0) * 1_000_000),
                    open=float(k.get("o", 0)),
                    high=float(k.get("h", 0)),
                    low=float(k.get("l", 0)),
                    close=float(k.get("c", 0)),
                    volume=float(k.get("v", 0)),
                )
                
                result = inst.on_bar(bar, ctx)
                
                # Extract signal from result
                from Interfaces.IStrategy import StrategyDecision
                if isinstance(result, StrategyDecision):
                    if result.has_signal:
                        signal = result.signal
                    elif result.has_orders:
                        # Determine signal from orders
                        for order in result.orders:
                            if not order.metadata.get("reduce_only", False):
                                from Interfaces.orders import OrderSide
                                signal = "+1" if order.side == OrderSide.BUY else "-1"
                                break
            except Exception as e:
                log.warning(f"on_bar failed, falling back to generate_signal: {e}")
                signal = inst.generate_signal(symbol)
        else:
            # Use legacy generate_signal
            signal = inst.generate_signal(symbol)

        if signal:
            # Map signal to direction
            direction = 1 if signal in ("+1", "1") else -1
            
            await self.pos_mgr.open_position(
                symbol,
                direction,
                strategy_config.get("name", "UnknownStrategy"),
                leverage=exec_params.get("leverage", 1),
                sl_pct=exec_params.get("sl_pct", 1.0),
                tp_pct=exec_params.get("tp_pct", 1.0),
                expire_sec=exec_params.get("expire_sec", 3600),
                timeframes=timeframe
            )
        else:
            await self.pos_mgr.update_all()
