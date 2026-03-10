# ai-adaptive-tradebot-blg4901
AI-powered adaptive trading bot with tick-level backtesting, regime detection, sentiment analysis, and real-time dashboard.

## Execution Realism Notes

### Intra-bar price model (`intrabar_price_model`)

When order latency keeps a fill inside the same bar (no bar-boundary crossing),
the `intrabar_price_model` setting controls whether the fill price is perturbed:

| Value | Behaviour |
|-------|-----------|
| `"none"` **(default)** | Fill stays at the bar's open (or the field set by `price_latency_bar_field`). No intra-bar perturbation — deterministic and bias-free. |
| `"gaussian_clamped"` | A Gaussian shift proportional to `latency / bar_duration` is applied, clamped to `[low, high]`. **This is an approximation** — it uses OHLC bounds that are only known after the bar closes, so it introduces a mild look-ahead. Use for stress-testing, not for unbiased PnL estimation. |

Recommendation: leave the default (`"none"`) for all research runs.
Switch to `"gaussian_clamped"` only when you want to study sensitivity to
intra-bar price uncertainty — and document that the results carry a small
optimistic bias from the OHLC clamp.
