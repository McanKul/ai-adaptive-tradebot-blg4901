import asyncio
from utils.bar_store import BarStore
from Interfaces.IBroker import IBroker
# from strategies import load_strategy # Removed, we load locally
from live.position_manager import PositionManager
from live.broker_binance import BinanceBroker
from live.streamer import Streamer
from utils.logger import setup_logger
from Strategy.binary_base_strategy import BinaryBaseStrategy # Import local strategy base

# Mock Strategy Loader for now - in real implementation this might be dynamic
# Assumes strategy classes are available or passed in
def load_strategy_instance(strategy_cls, config, bar_store):
    return strategy_cls(bar_store=bar_store, **config["params"])

log = setup_logger("LiveEngine")


class LiveEngine:
    def __init__(self, config, broker:IBroker, strategy_cls):
        self.cfg     = config
        self.broker  = broker          
        self.bar_store = BarStore()    

        # — Configure Strategies —
        self.strategies = []
        # Support single strategy config for now based on user request context, 
        # or list if config provides it. Assuming config is a dict with strategy params.
        
        # Example config structure expected:
        # {
        #   "coins": ["DOGEUSDT"],
        #   "timeframe": "1m",
        #   "name": "RSI_Strategy",
        #   "params": { ... strategy params ... },
        #   "execution": { "leverage": 10, "sl_pct": 1.0, "tp_pct": 2.0, "expire_sec": 3600 }
        # }
        
        # We wrap it in list to support multiple if needed later
        strategy_configs = [self.cfg] if isinstance(self.cfg, dict) else self.cfg

        for scfg in strategy_configs:
             # Instantiate strategy
            instance = load_strategy_instance(strategy_cls, scfg, self.bar_store)
            self.strategies.append({**scfg, "instance": instance})
            
        self.pos_mgr = PositionManager(self.broker, base_capital=10.0, max_concurrent=1) # Hardcoded defaults for now, can come from config
        
        # — Extract timeframes —
        self.timeframes = list(set(s["timeframe"] for s in self.strategies))
        self.streamer = None
        self.symbols  = []

    # -------------------------------------------------------------
    async def run(self):
        # 1) Resolve Symbols
        # Collect all coins from all strategies
        all_coins = []
        for s in self.strategies:
            all_coins.extend(s["coins"])
        
        self.symbols = await Streamer.resolve_symbols(
            self.broker.client, list(set(all_coins)))

        # 2) Create Streamer
        self.streamer = Streamer(self.broker.client,
                                 self.symbols,
                                 self.timeframes,
                                 bar_store=self.bar_store)

        # 3) Preload History
        await self.streamer.preload_history(
            self.symbols, self.timeframes,
            limit=100, # reduced limit for safety/speed
            batch=10)

        # 4) Start Live Stream
        await self.streamer.start()
        log.info("Live Engine Started: %s symbols | tf=%s",
                 len(self.symbols), self.timeframes)

        try:
            while True:
                bar = await self.streamer.get()
                sym, tf = bar["s"], bar["k"]["i"]

                for s in self.strategies:
                    # Check if this strategy cares about this symbol/tf
                    if sym not in s["coins"] or tf != s["timeframe"]:
                        continue
                        
                    inst = s["instance"]
                    # Generate signal using the bar store (implicitly inside instance)
                    # We pass sym because strategy might handle multiple, but usually 1 instance per config
                    sig  = inst.generate_signal(sym) 

                    if sig:
                        # Map +1/-1 to Buy/Sell
                        direction = 1 if sig == "+1" else -1
                        exec_params = s.get("execution", {})
                        
                        await self.pos_mgr.open_position(
                            sym, direction,
                            s.get("name", "UnknownStrategy"),
                            leverage   = exec_params.get("leverage", 1),
                            sl_pct     = exec_params.get("sl_pct", 1.0),
                            tp_pct     = exec_params.get("tp_pct", 1.0),
                            expire_sec = exec_params.get("expire_sec", 3600),
                            timeframes = tf)
                    else:
                        await self.pos_mgr.update_all()
        finally:
            await self.streamer.stop()
