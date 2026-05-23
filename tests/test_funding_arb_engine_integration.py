"""
tests/test_funding_arb_engine_integration.py
=============================================
Visible debt marker for the funding-rate arbitrage strategy.

`Strategy/arb/funding_arb_strategy.py` declares itself backtest-only:
its on_bar can emit two legs (spot + perp) but BacktestEngine does not
yet guarantee atomic, cross-product execution of both legs in the same
tick.  The `_validate_live_strategy_safety` guard
(core/services/live_service.py) blocks this strategy from going live;
this test marks the *engine-side* gap so that any future PR that wires
atomic multi-symbol fills can flip the xfail to xpass and remove the
guard at the same time.

Until then the test is intentionally failing — we want CI to *show* the
debt rather than burying it in a TODO comment.
"""
import os
import sys

import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))


@pytest.mark.xfail(
    reason="Funding arb needs atomic multi-symbol execution support",
    strict=False,
)
def test_funding_arb_two_legs_require_atomic_engine_execution():
    """When BacktestEngine acquires atomic spot+perp fills, swap this
    assert for a real round-trip and remove the live-mode guard in
    `core/services/live_service.py:_validate_live_strategy_safety`.
    """
    atomic_multi_symbol_supported = False
    assert atomic_multi_symbol_supported, (
        "BacktestEngine does not yet execute funding-arb spot+perp legs "
        "atomically; live promotion stays blocked until this lands."
    )
