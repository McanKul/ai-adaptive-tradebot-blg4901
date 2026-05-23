# Thesis Demo — Sentiment ON / OFF A/B (Dry-Run)

This is the recipe for the thesis defense run: two parallel paper-trading
sessions on the same live market, identical in everything except whether
news sentiment is allowed to gate the strategy signal. The marginal
delta is the value-add of the sentiment module.

## Why dry-run, not backtest

Backtest does not consume sentiment — sentiment is wired into
`live/live_engine.py` only, and historical timestamped news for a
backtest period is not freely available without a paid feed and a heavy
LLM-scoring pass. Two parallel dry-runs see **the same live tape** at
the same moment, so any decision difference comes only from the
sentiment combiner. That is a controlled A/B for thesis purposes.

## One-time setup

1. **API keys (.env)**
   ```env
   BINANCE_API_KEY=...        # READ-only is fine for dry-run market data
   BINANCE_API_SECRET=...
   GOOGLE_API_KEY=...         # only needed by the SENTIMENT-ON run
   ```
2. **Pick a config.** Either an existing profile or a copy you tweak.
   Recommended starting point: `live_config.yaml` (multi-coin, 15m).
   Make sure `news.enabled: true` is the *file default* — the CLI flag
   below overrides for the OFF run.
3. **Pick the run window.** 30 minutes minimum to get a reasonable trade
   count; 2–4 hours is solid for a defense slide. Same wall-clock window
   for both runs.

## Run the A/B (two terminals)

```bash
# Terminal 1 — sentiment OFF (baseline)
python app.py dry-run \
  --config live_config.yaml \
  --run-id sentiment_off \
  --sentiment off

# Terminal 2 — sentiment ON (variant)
python app.py dry-run \
  --config live_config.yaml \
  --run-id sentiment_on \
  --sentiment on
```

Each run namespaces its log/state files via `--run-id`:

| File | Baseline (`--run-id sentiment_off`) | Variant (`--run-id sentiment_on`) |
|---|---|---|
| Trade CSV | `logs/live_trades_sentiment_off.csv` | `logs/live_trades_sentiment_on.csv` |
| Position store | `logs/live_positions_sentiment_off.json` | `logs/live_positions_sentiment_on.json` |
| Risk state | `logs/live_risk_state_sentiment_off.json` | `logs/live_risk_state_sentiment_on.json` |

`--sentiment off|on` overrides whatever `news.enabled` says in the YAML
so you can keep one config file for both runs.

## Stop both runs

`Ctrl+C` in each terminal — both shut down gracefully and flush the
final trade CSV row.

## Compare

```bash
python tools/compare_dry_runs.py \
  --baseline logs/live_trades_sentiment_off.csv \
  --variant  logs/live_trades_sentiment_on.csv \
  --baseline-label "no sentiment" \
  --variant-label  "sentiment on" \
  --by-symbol
```

Output: side-by-side table with trade count, win rate, total / avg /
best / worst P&L, profit factor, average hold time, exit-type counts,
and (with `--by-symbol`) per-symbol P&L. The footer prints the
sentiment delta in trades, dollars, and win-rate percentage points —
that's the slide-ready number.

## Reading the result

The expected pattern:

* **Variant fires fewer trades.** `BinarySignalCombiner` requires
  strategy intent **and** aligned sentiment; neutral-or-conflicting
  sentiment vetoes the entry. So fewer trades is normal and is itself
  a result.
* **Win rate change.** If sentiment is informative on this tape, win
  rate should go up. If the news flow was thin or off-topic, it can
  drop slightly — that's also a finding worth reporting.
* **PnL change.** The dollars number combines both effects.

If both runs end with zero trades, extend the window or pick a more
volatile symbol set; mean-reverting consolidation kills both runs
equally.

## Files modified by this demo flow

* [app.py](../app.py) — `--run-id` and `--sentiment {on,off}` flags on
  `live` and `dry-run` subcommands.
* [live/live_config.py](../live/live_config.py) — `LiveConfig.run_id`,
  `effective_run_id()`, `trade_log_path()`, `positions_state_path()`,
  `risk_state_path()`.
* [core/services/live_service.py](../core/services/live_service.py) —
  applies `run_id` and `sentiment_override`; namespaces global-risk
  persist path.
* [live/live_engine.py](../live/live_engine.py) — feeds run-id-aware
  paths to `LiveSupervisor` and `LiveMetrics`.
* [tools/compare_dry_runs.py](../tools/compare_dry_runs.py) — reads
  both CSVs, prints comparison table.
