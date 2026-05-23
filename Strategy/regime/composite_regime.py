"""
Strategy/regime/composite_regime.py
===================================
Aggregates multiple ``IRegimeDetector`` outputs into a single tag set.

Tags from all sub-detectors are merged (deduped union); scores are
namespaced by detector index.  This lets ``CompositeStrategy`` gate on
combinations like ``regimes: [trend_up, vol_low]`` (slot fires only if
BOTH tags are present in the merged state — that is the standard set
intersection semantics inside the composite).
"""
from __future__ import annotations
from typing import Any, List, TYPE_CHECKING

from Interfaces.IRegimeDetector import IRegimeDetector, RegimeState

if TYPE_CHECKING:
    from Interfaces.market_data import Bar


class CompositeRegimeDetector(IRegimeDetector):
    """Merge tags from multiple detectors.

    Args:
        detectors: Ordered list of sub-detectors.
    """

    def __init__(self, detectors: List[IRegimeDetector]):
        if not detectors:
            raise ValueError("CompositeRegimeDetector requires at least one detector")
        self.detectors = detectors

    def detect(self, bar: "Bar", ctx: Any) -> RegimeState:
        seen: List[str] = []
        score: dict = {}
        for i, det in enumerate(self.detectors):
            state = det.detect(bar, ctx)
            for t in state.tags:
                if t not in seen:
                    seen.append(t)
            for k, v in state.score.items():
                score[f"d{i}.{k}"] = v
        return RegimeState(tags=seen, score=score)

    def reset(self) -> None:
        for d in self.detectors:
            try:
                d.reset()
            except Exception:  # pragma: no cover
                pass
