# Backtest Subsystem Upgrade Summary

## Overview

This document summarizes all fixes and improvements made to the backtest subsystem to align with AFML (Advances in Financial Machine Learning) methodology and ensure deterministic, robust backtesting.

## Core Invariants Maintained

1. **Disk replay only** - never live websocket in backtest
2. **Tick-level input** - strategies operate only on Bars (never see ticks)
3. **Deterministic** - same seed → same results
4. **Scoring layer above engine** - no ML/AI in backtest itself
5. **AFML methodology** - PurgedKFold, Embargo, Selection Bias reporting

---

## Changes Made

### A. Data Layer (TickStore)

**File:** `Backtest/tick_store.py`

**Issue:** Deduplication was using `(timestamp, price, volume)` which could incorrectly merge different trades.

**Fix:** 
- Changed dedup key to `(timestamp_ns, trade_id)` - the correct unique identifier per trade
- Added `TickStoreConfig` fields:
  - `deduplicate: bool = True`
  - `sort_if_needed: bool = True` 
  - `max_sort_size: int = 10_000_000`
- Rewrote `_iter_csv()` to:
  - Collect all rows, deduplicate using set of `(timestamp_ns, trade_id)`
  - Sort by timestamp if needed (with size limit warning)
  - Handle both string and int trade_ids

### B. Order Interface

**File:** `Interfaces/orders.py`

**Fix:** Added `reduce_only: bool = False` field to `Order` dataclass for proper exit management.

### C. Scoring Layer - Trial-Aware Reporting (AFML)

**File:** `Backtest/scoring/scorer.py`

**Added:**
- `compute_deflated_sharpe(observed_sharpe, std_sharpe, num_trials, num_days)` - DSR calculation
- `selection_bias_warning(num_trials, best_sharpe, deflated_sharpe)` - Warning generator
- `TrialAwareScorer` class that:
  - Tracks trial count and best Sharpe across all scored results
  - Generates selection bias report with DSR
  - Issues warnings when DSR < 1.0 (likely selection bias)

### D. Batch Backtest Error Handling

**File:** `Backtest/scoring/batch.py`

**Added:**
- `create_dummy_result(params, error_message)` - Creates placeholder result for failed runs
- Updated `BatchResult` dataclass with:
  - `trial_count: int = 0`
  - `failed_count: int = 0`
  - `selection_bias_report: Optional[Dict] = None`
- Updated `run()` method to:
  - Catch exceptions per-trial without crashing batch
  - Record error in dummy result
  - Generate selection bias report at end

### E. SearchSpace Constraints

**File:** `Backtest/scoring/search_space.py`

**Added predefined constraint functions:**
- `less_than_constraint(param1, param2)`
- `less_equal_constraint(param1, param2)`
- `range_constraint(param, min_val, max_val)`
- `leverage_constraint(leverage_param, max_leverage)`

**Added convenience methods to SearchSpace:**
- `require_less_than(param1, param2)`
- `require_less_equal(param1, param2)`
- `require_range(param, min_val, max_val)`
- `require_max_leverage(max_leverage, param_name)`

### F. CLI Runner

**File:** `Backtest/run_backtest.py`

**Added:**
- `validate_data_exists()` function with clear error messages and instructions
- New CLI arguments:
  - `--leverage-mode {spot,margin}` - Trading mode selection
  - `--leverage` - Leverage multiplier (1-125)
  - `--synthetic` - Allow missing data (testing only)
- Selection bias report printing in sweep mode
- Better error messages for missing data

### G. Module Exports

**File:** `Backtest/scoring/__init__.py`

**Updated exports to include:**
- `TrialAwareScorer`
- `compute_deflated_sharpe`
- `selection_bias_warning`
- `BatchResult`
- `create_dummy_result`

---

## New Test Files

### 1. `tests/test_tick_store_dedup.py`

Tests for correct deduplication behavior:
- Exact duplicates removed (same timestamp + trade_id)
- Same timestamp, different trade_id kept (multiple trades at same moment)
- Sort behavior with `sort_if_needed`
- Combined dedup and sort
- Real-world scenario with aggregated trades

### 2. `tests/test_scoring_afml.py`

Tests for AFML scoring components:
- DSR decreases with more trials
- DSR increases with higher Sharpe
- DSR increases with more data
- Selection bias warning generation
- TrialAwareScorer tracking
- BatchResult failure tracking
- SearchSpace constraints (less_than, range, max_leverage)

### 3. `tests/test_portfolio_margin.py`

Tests for margin/leverage mode:
- Spot mode requires full cash
- Margin mode allows leveraged positions
- Equity calculation in both modes
- Leverage amplifies returns
- Current leverage calculation
- Position closing realizes PnL
- Fee handling
- Drawdown tracking

---

## Usage Examples

### Running a single backtest with margin mode:
```bash
python -m Backtest.run_backtest \
    --data-dir ./data/ticks \
    --symbol BTCUSDT \
    --leverage-mode margin \
    --leverage 10
```

### Running parameter sweep with selection bias report:
```bash
python -m Backtest.run_backtest \
    --mode sweep \
    --data-dir ./data/ticks \
    --symbol BTCUSDT
```

### Using SearchSpace with constraints:
```python
space = SearchSpace()
space.add("fast_period", [5, 10, 20])
space.add("slow_period", [20, 50, 100])
space.add("position_size", [0.1, 0.5, 1.0, 2.0])

# Ensure fast < slow
space.require_less_than("fast_period", "slow_period")

# Limit position size (leverage)
space.require_max_leverage(max_leverage=3.0, param_name="position_size")
```

### Using TrialAwareScorer:
```python
from Backtest.scoring import TrialAwareScorer

scorer = TrialAwareScorer()
for result in all_results:
    score = scorer.score(result)

# Get selection bias report
report = scorer.get_selection_bias_report(num_days=252)
print(f"DSR: {report['deflated_sharpe_ratio']:.4f}")
if report['deflated_sharpe_ratio'] < 1.0:
    print("⚠️ Selection bias warning!")
```

---

## AFML Alignment Notes

1. **Deflated Sharpe Ratio (DSR)**: Accounts for number of trials when evaluating Sharpe. A high Sharpe from 1000 trials is less meaningful than from 10 trials.

2. **Selection Bias**: When DSR < 1.0, the best strategy's performance is likely due to luck (trying many combinations) rather than genuine skill.

3. **Recommendation**: When DSR is low:
   - Reduce number of parameter combinations tested
   - Use more historical data
   - Apply walk-forward validation
   - Use PurgedKFold with embargo for proper cross-validation

---

## Files Modified (pre-existing)

| File | Type of Change |
|------|---------------|
| `Backtest/tick_store.py` | Major rewrite of dedup logic |
| `Interfaces/orders.py` | Added reduce_only field |
| `Backtest/scoring/scorer.py` | Added DSR + TrialAwareScorer |
| `Backtest/scoring/batch.py` | Added error handling + bias report |
| `Backtest/scoring/search_space.py` | Added constraint functions |
| `Backtest/scoring/__init__.py` | Updated exports |
| `Backtest/run_backtest.py` | Added CLI args + validation |

## Files Created (pre-existing)

| File | Purpose |
|------|---------|
| `tests/test_tick_store_dedup.py` | Test deduplication |
| `tests/test_scoring_afml.py` | Test AFML scoring |
| `tests/test_portfolio_margin.py` | Test margin mode |
| `docs/UPGRADE_SUMMARY.md` | This document |

---

## Part A — Execution / Cost / Latency Realism Refactor

### A1  RealismConfig (central configuration)

**New file:** `Backtest/realism_config.py`

Added three nested dataclasses behind flags (all default OFF / legacy):

| Dataclass | Purpose |
|-----------|---------|
| `TransactionCostConfig` | fee / slippage / spread / latency model knobs |
| `FundingConfig` | perpetual-futures funding cost |
| `BorrowConfig` | margin borrow / interest cost |
| `RealismConfig` | top-level container + CV knobs + serialisation helpers |

`RealismConfig` is attached to `BacktestConfig.realism` and flows through
`to_engine_config() → EngineConfig.realism`.

### A2  Price-Aware Latency

**File:** `Backtest/execution_models.py`

When `price_latency_mode = "price_aware"`, `_resolve_base_price()` looks up
the bar whose timestamp is closest to the calculated fill-time instead of
always using the order-bar's close. This makes latency modelling more realistic
for larger latency values.

### A3  Decomposed Cost Reporting

**File:** `Backtest/cost_models.py`  
Added `CostBreakdown` dataclass (fee / spread / slippage / funding / borrow
fields). `CompositeCostModel.cost_breakdown()` returns the decomposition.
Fill metadata receives `cost_breakdown` dict.

**File:** `Backtest/metrics.py`  
`MetricsSink` now tracks per-component cost accumulators
(`_total_spread_cost`, `_total_slippage_cost`, `_total_funding_cost`,
`_total_borrow_cost`) and writes them into `result.metadata` on `finalize()`.

### A4  Selectable Slippage Models

**File:** `Backtest/cost_models.py`

| Model | Formula |
|-------|---------|
| `VolumeSqrtSlippageModel` | slippage ∝ √(qty / bar_volume) × impact_factor |
| `VolatilityATRSlippageModel` | slippage ∝ ATR × impact_factor |

`create_cost_model_from_config(tc: TransactionCostConfig)` factory builds the
composite model from config, selecting the right slippage model by name.

### A5  Marketable-Limit Detection

**File:** `Backtest/execution_models.py`

When `marketable_limit_is_taker = True`, limit orders whose price crosses the
estimated bid/ask midpoint are routed through the market-order path and charged
taker fees, matching real exchange semantics.

### A6  Funding & Borrow Costs

**File:** `Backtest/engine.py`

`_charge_funding_and_borrow()` is called from `_process_bar()`.
- **Funding:** for every `funding_interval_hours` elapsed, charges
  `position_notional × funding_rate`.
- **Borrow:** for every interval, charges
  `|position_notional| × annual_borrow_rate × Δt/year`.

Both costs are deducted from portfolio cash and recorded in MetricsSink.

### A7  Tests

**New file:** `tests/test_realism_refactor.py` — 25 tests covering A1 – A6.

### A8  Example Config

**New file:** `example_realism_config.yaml` — commented YAML showing every knob
with explanations.  Load via `RealismConfig.from_yaml(path)`.

---

## Part B — DonchianATRVolTarget Strategy

**New file:** `Strategy/DonchianATRVolTarget.py`

Donchian Channel Breakout + ATR trailing stop + volatility-target sizing +
regime filter.

| Feature | Detail |
|---------|--------|
| Entry | Price breaks above/below `dc_period` high/low |
| Stop | ATR-based trailing stop (`atr_mult × ATR`) |
| Sizing | Volatility-target: `risk_pct × capital / ATR` |
| Filter | Optional EMA trend filter or ADX regime filter |
| Time stop | `max_holding_bars` auto-exit |
| Reversal | `allow_reversal` flag for immediate direction flip |

All indicators computed with pure NumPy (no talib dependency).

**New file:** `tests/test_donchian_strategy.py` — 8 tests.

---

## Part C — CV Splits Integrated into Scoring Pipeline

### C1-C2  Config-driven CV

`BatchBacktest.run_cv_from_config(param_space)` reads `RealismConfig.cv_*`
fields and delegates to `run_with_cv()`.  If `cv_enabled = False` it falls
back to a plain `run()`.

Supported `cv_method` values: `purged_kfold`, `walk_forward`,
`combinatorial_purged` / `cpcv`.

### C3  Fold config propagation

`_create_fold_config()` now forwards `self.config.realism` to each fold's
`BacktestConfig`, so realism knobs apply inside every CV fold.

### C4  CV-aware Selector

`SelectionCriteria` gains two new fields:

| Field | Default | Effect |
|-------|---------|--------|
| `cv_stability_weight` | 0.0 | Penalty multiplier: `eff_score = score − w × cv_score_std` |
| `max_cv_std` | ∞ | Hard filter: reject candidates with `cv_score_std >` threshold |

Setting `cv_stability_weight > 0` promotes parameter sets that perform
consistently across folds, not just on the luckiest fold.

### C5  Fold breakdown metadata

`BacktestResult.metadata` now carries:

```
cv_scores          List[float]   – per-fold scores
cv_score_mean      float
cv_score_std       float
cv_score_min       float
cv_score_max       float
cv_n_folds         int
cv_fold_details    List[dict]    – per-fold breakdown (sharpe, return, dd, …)
```

`BatchResult` also stores `cv_fold_details` (list parallel to `results`) and
`cv_method`.

### C6  Tests

**New file:** `tests/test_cv_pipeline.py` — 14 tests covering C1 – C6 +
no-leakage verification + end-to-end selector integration.

---

## Full File Inventory (this refactor)

### Files Created

| File | Purpose |
|------|---------|
| `Backtest/realism_config.py` | A1 – central realism configuration |
| `Strategy/DonchianATRVolTarget.py` | Part B – new strategy |
| `example_realism_config.yaml` | A8 – annotated example config |
| `tests/test_realism_refactor.py` | A7 – 25 realism tests |
| `tests/test_donchian_strategy.py` | Part B – 8 strategy tests |
| `tests/test_cv_pipeline.py` | C6 – 14 CV pipeline tests |

### Files Modified

| File | Changes |
|------|---------|
| `Backtest/engine.py` | RealismConfig import, config-aware cost model, funding/borrow |
| `Backtest/runner.py` | RealismConfig field on BacktestConfig + passthrough |
| `Backtest/execution_models.py` | Price-aware latency, marketable limits, cost breakdown |
| `Backtest/cost_models.py` | CostBreakdown, 2 new slippage models, factory |
| `Backtest/metrics.py` | Per-component cost accumulators |
| `Backtest/scoring/batch.py` | run_cv_from_config, fold detail tracking, realism propagation |
| `Backtest/scoring/selector.py` | cv_stability_weight, max_cv_std |

### Test Status

```
271 passed, 2 failed (pre-existing RSI string-vs-int bug, unrelated)
```
