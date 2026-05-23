from abc import ABC, abstractmethod
from Interfaces.IClient import IClient

class IBroker(ABC):
    """Exchange-agnostic broker interface."""
    
    @property
    @abstractmethod
    def client(self) -> IClient: ...

    @abstractmethod
    async def get_mark_price(self, symbol: str) -> float: ...
    # ---- order & position ----
    @abstractmethod
    async def market_order(self, symbol: str, side: str, qty: float): ...

    @abstractmethod
    async def close_position(self, symbol: str): ...

    @abstractmethod
    async def position_amt(self, symbol: str) -> float: ...

    # ---- settings ----
    @abstractmethod
    async def ensure_isolated_margin(self, symbol: str): ...

    @abstractmethod
    async def set_leverage(self, symbol: str, leverage: int): ...

    # ---- SL / TP ----
    @abstractmethod
    async def place_stop_market(self, symbol: str, side: str, stop_price: float) -> int: ...

    @abstractmethod
    async def place_take_profit(self, symbol: str, side: str, stop_price: float) -> int: ...

    @abstractmethod
    async def cancel_order(self, symbol: str, order_id: int): ...

    @abstractmethod
    async def get_open_orders(self, symbol: str) -> list: ...

    # ---- balance ----
    @abstractmethod
    async def balance(self, asset: str = "USDT") -> float: ...

    @abstractmethod
    async def exchange_info(self) -> dict: ...

    # ---- liquidity gate (Phase D) ----
    async def get_24h_volume(self, symbol: str) -> float:
        """Return 24-hour quote-asset volume in USDT.

        Used by ``LiveEngine`` startup to drop symbols below the
        whitelist's ``min_24h_volume_usd`` threshold.  Defaults to
        ``+inf`` so brokers that don't override this method (e.g.
        ``DryBroker``) admit every symbol.
        """
        return float("inf")

    # ---- realised cost accounting hooks (Phase B4 follow-up) ----
    # These are intentionally OPTIONAL with safe defaults so brokers
    # that don't expose the data (DryBroker, test stubs) keep working.
    # Live ``BinanceBroker`` overrides both to surface the real numbers.
    async def get_realised_commission(
        self, symbol: str, since_ms: int, until_ms: int = 0,
    ) -> float:
        """Sum of realised USDT commissions over the [since, until]
        window for *symbol*.  ``until_ms=0`` means "now".  Defaults to
        0 so summary CSVs degrade to gross-only rather than crashing."""
        return 0.0

    async def get_funding_paid(
        self, symbol: str, since_ms: int, until_ms: int = 0,
    ) -> float:
        """Net funding USDT paid (positive) or received (negative)
        over the window.  Default 0."""
        return 0.0

    # ---- fill confirmation (Phase C1) ----
    # Default implementation lives on the dataclass via duck-typing —
    # both BinanceBroker and DryBroker override.  Returns a dict with
    # at minimum:
    #   ``status``         : "FILLED" | "PARTIALLY_FILLED" | "CANCELED" |
    #                        "EXPIRED" | "TIMEOUT"
    #   ``executed_qty``   : float (in base asset)
    #   ``avg_price``      : float
    #   ``commission_usd`` : float (USDT-equivalent total commission)
    async def wait_for_fill(
        self, symbol: str, order_id: int, timeout: float = 5.0,
    ) -> dict:
        """Poll the exchange until the order finalises or *timeout* hits."""
        raise NotImplementedError
