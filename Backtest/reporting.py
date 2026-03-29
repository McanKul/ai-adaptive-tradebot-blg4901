"""
Backtest/reporting.py
=====================
Lightweight post-backtest summary with AFML selection-bias metrics.

Prints a human-readable report and optionally writes a JSON artifact
to ``logs/backtest_report_<timestamp>.json``.

Usage (after a sweep):
    from Backtest.reporting import print_afml_summary, save_afml_report

    print_afml_summary(best_result, n_trials=216)
    save_afml_report(best_result, n_trials=216)

Usage (single backtest — n_trials=1 skips bias warning):
    print_afml_summary(result, n_trials=1)
"""
from __future__ import annotations

import json
import logging
import math
import os
from datetime import datetime, timezone
from typing import Optional

from Interfaces.metrics_interface import BacktestResult
from Backtest.scoring.scorer import compute_deflated_sharpe, selection_bias_warning

log = logging.getLogger(__name__)


def _bias_state(n_trials: int, dsr: float, sharpe: float) -> str:
    """Classify selection-bias severity."""
    if n_trials <= 1:
        return "N/A (single trial)"
    if dsr > 0.5:
        return "LOW — DSR still positive and healthy"
    if dsr > 0:
        return "MODERATE — DSR positive but deflated"
    if dsr > -0.5:
        return "HIGH — DSR near zero, results may be noise"
    return "VERY HIGH — DSR negative, likely overfitted"


def print_afml_summary(
    result: BacktestResult,
    n_trials: int = 1,
    *,
    t_years: float = 1.0,
) -> None:
    """Print a concise AFML-aware post-backtest summary to stdout."""

    sharpe = result.sharpe_ratio if not math.isnan(result.sharpe_ratio) else 0.0
    dsr = compute_deflated_sharpe(sharpe, n_trials, t_years=t_years)
    bias = _bias_state(n_trials, dsr, sharpe)

    lines = [
        "",
        "=" * 60,
        "  AFML POST-BACKTEST SUMMARY",
        "=" * 60,
        f"  Strategy:          {result.strategy_name or 'N/A'}",
        f"  Return:            {result.total_return_pct:+.2f}%",
        f"  Sharpe:            {sharpe:.3f}",
        f"  Max Drawdown:      {result.max_drawdown:.2%}",
        f"  Win Rate:          {result.win_rate:.1%}",
        f"  Total Trades:      {result.total_trades}",
        "-" * 60,
        f"  Trials Tested:     {n_trials}",
        f"  Deflated Sharpe:   {dsr:.3f}",
        f"  Sharpe Inflation:  {sharpe - dsr:+.3f}",
        f"  Bias State:        {bias}",
        "=" * 60,
    ]

    if n_trials > 5:
        lines.insert(-1, "")
        lines.insert(-1, "  " + selection_bias_warning(n_trials, sharpe).replace("\n", "\n  "))

    print("\n".join(lines))


def save_afml_report(
    result: BacktestResult,
    n_trials: int = 1,
    *,
    t_years: float = 1.0,
    output_dir: str = "logs",
) -> str:
    """
    Save AFML summary as a JSON artifact.

    Returns the file path written.
    """
    sharpe = result.sharpe_ratio if not math.isnan(result.sharpe_ratio) else 0.0
    dsr = compute_deflated_sharpe(sharpe, n_trials, t_years=t_years)

    report = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "strategy_name": result.strategy_name or "N/A",
        "params": result.params,
        "metrics": {
            "total_return_pct": round(result.total_return_pct, 4),
            "sharpe_ratio": round(sharpe, 4),
            "max_drawdown": round(result.max_drawdown, 4),
            "win_rate": round(result.win_rate, 4),
            "total_trades": result.total_trades,
            "profit_factor": round(result.profit_factor, 4),
            "calmar_ratio": round(result.calmar_ratio, 4),
        },
        "afml": {
            "n_trials": n_trials,
            "deflated_sharpe": round(dsr, 4),
            "sharpe_inflation": round(sharpe - dsr, 4),
            "bias_state": _bias_state(n_trials, dsr, sharpe),
        },
    }

    os.makedirs(output_dir, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = os.path.join(output_dir, f"backtest_report_{ts}.json")

    with open(path, "w") as f:
        json.dump(report, f, indent=2, default=str)

    log.info("AFML report saved: %s", path)
    return path
