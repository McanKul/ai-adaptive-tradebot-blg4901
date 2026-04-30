"""
Interfaces/IRegimeDetector.py
=============================
Regime detector interface used by ``CompositeStrategy`` for dynamic
slot gating / weighting.

A detector inspects the latest bar (and the OHLCV history available
via ``ctx.get_ohlcv()``) and returns a ``RegimeState`` with one or
more tags such as ``"trend_up"``, ``"vol_high"``, ``"range"``.
``CompositeStrategy`` uses these tags to gate or weight slots.

Detectors must be **deterministic** and side-effect free.  Warmup is
the detector's responsibility — return an empty tag set if there is
not enough history.
"""
from __future__ import annotations
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, TYPE_CHECKING

if TYPE_CHECKING:
    from Interfaces.market_data import Bar


@dataclass
class RegimeState:
    """Output of a regime detector for one bar.

    Attributes:
        tags: Tags describing the regime (used for gating).  Examples:
            ``"trend_up"``, ``"trend_down"``, ``"range"``, ``"vol_high"``,
            ``"vol_mid"``, ``"vol_low"``.  Order is not significant.
        score: Optional named numeric scores (e.g. ``{"adx": 28.4}``)
            for downstream weighting and debugging.
    """
    tags: List[str] = field(default_factory=list)
    score: Dict[str, float] = field(default_factory=dict)


class IRegimeDetector(ABC):
    """Detects market regime from current bar + history.

    Implementations should:
    * be deterministic (same input ⇒ same output);
    * return empty tags during warmup;
    * be cheap to call every bar.
    """

    @abstractmethod
    def detect(self, bar: "Bar", ctx: Any) -> RegimeState:
        """Return regime tags for the current bar.

        Args:
            bar: Latest closed bar.
            ctx: ``StrategyContext`` (or a duck-typed equivalent) with
                ``get_ohlcv(limit)`` exposing recent OHLCV history.
        """

    def reset(self) -> None:  # pragma: no cover — default is a no-op
        """Reset internal state (warmup buffers, EWMs).  Optional."""
