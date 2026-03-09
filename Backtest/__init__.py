"""
Backtest/__init__.py
====================
Backtest package initialization.

This package provides a complete backtesting subsystem that:
- Reads tick-level data from disk
- Builds bars from ticks
- Feeds bars to strategies (strategies never see ticks)
- Executes orders with realistic cost modeling
- Tracks portfolio and computes metrics

Key components:
- TickStore: Loads tick data from CSV/Parquet
- DiskTickStreamer: Iterates ticks deterministically
- BarBuilder: Converts ticks to bars
- BacktestEngine: Orchestrates the backtest
- BacktestRunner: Runs single backtests
- Scoring: Parameter sweep and selection
"""

from Backtest.tick_store import TickStore
from Backtest.disk_streamer import DiskTickStreamer
from Backtest.bar_builder import BarBuilder, TimeBarBuilder, TickBarBuilder, VolumeBarBuilder, DollarBarBuilder
from Backtest.portfolio import Portfolio
from Backtest.execution_models import SimpleExecutionModel, LimitExecutionModel
from Backtest.cost_models import FixedFeeModel, FixedSlippageModel, SpreadCostModel, LatencyModel, CompositeCostModel
from Backtest.risk import BasicRiskManager
from Backtest.metrics import MetricsSink
from Backtest.engine import BacktestEngine
from Backtest.runner import BacktestRunner

__all__ = [
    "TickStore",
    "DiskTickStreamer",
    "BarBuilder",
    "TimeBarBuilder",
    "TickBarBuilder",
    "VolumeBarBuilder",
    "DollarBarBuilder",
    "Portfolio",
    "SimpleExecutionModel",
    "LimitExecutionModel",
    "FixedFeeModel",
    "FixedSlippageModel",
    "SpreadCostModel",
    "LatencyModel",
    "CompositeCostModel",
    "BasicRiskManager",
    "MetricsSink",
    "BacktestEngine",
    "BacktestRunner",
]
