---
name: "drawdown_ladder"
universe: "mixed"
timeframe: "swing"
n_trials: 48
gate: "REJECTED"
failed_checks: ["min_sharpe", "max_pbo", "max_is_oos_decay", "beats_simple_benchmark", "significance_t"]
---

# drawdown_ladder

## The idea

A three-rung ladder that adds hedge depth as the drawdown deepens.

## The failure mode it targets

a slow bleed that shallow overlays never catch

## The conditions

- Hold a target volatility of 0.281 (the `target_vol` parameter)
- Scale the hedge by how far realised volatility sits above that target, with a sensitivity of 0.870 (the `vol_gamma` parameter)
- Only add protection once the portfolio is actually down 3.5% (the `dd_trigger` drawdown trigger), ramping to full by 13.4% (the `dd_full` level)
- Act only when at least 0.572 of the volatility scales agree (the `consensus_required` agreement threshold)
- Cap the hedge at 0.330 of the portfolio (the `max_hedge_fraction` cap), and change it by no more than 0.337 a day (the `max_daily_turnover` cap)
- Hold no hedge at all while the market is expanding and the trend is intact (the `veto_on_expansion` rule)
- Pay 1.0 basis points of commission and 2.0 of spread on every trade, plus 10.0 a year to borrow (the cost model)

## Where it came from

The candidate bank in `hedging_agent/strategies.py`, and the thresholds and their primary sources in [`docs/LITERATURE.md`](../docs/LITERATURE.md).

## What would have falsified it

Before running it, this strategy had to beat a constant hedge carrying the same realised risk, hold up out of sample, and work in more than one market regime. If it could not, it was not a strategy -- it was a lucky configuration.

## What we measured

Every figure below comes from one sweep of **48 trials** over the mixed universe on the swing timeframe, measured over **2021-10-24 to 2026-10-24 (1827 bars, 5.0 years)**. The Sharpe is the winner's, and it is only meaningful next to the number of things tried to find it.

- **Sharpe (annualised):** 0.3838 hedged, against 0.4136 unhedged
- **Deflated Sharpe:** 1.0000 -- the probability that the observed Sharpe beats the best of 48 random trials
- **Probability of backtest overfitting:** 0.9571 over 70 combinations of 48 configurations
- **Newey-West HAC t-statistic:** -1.0210 against a hurdle of 3.0
- **In-sample to out-of-sample decay:** 11.3565 (the worst fold)
- **Positive folds:** 0.7500 over 8 windows
- **Action ratio:** 0.0571 -- the share of days on which the hedge actually changed
- **Veto ratio:** 0.0896 -- the share of days the risk rule said no
- **Zero-hedge share:** 0.0896 -- the share of days on which the correct answer was to hold no hedge at all
- **Max drawdown:** -0.5049 hedged, against -0.5183 unhedged
- **Hedge instrument:** SH (LONG (instrument already inverse to portfolio)), correlated -0.6391 with the portfolio
- **Costs charged:** 28.34x annualised turnover, at 3.0 basis points of commission and spread plus 10.0 a year to borrow

Against a constant hedge carrying the same realised risk: the hedge ratio was 0.1993, and the timing was worth 0.0000 of Sharpe. That is the number that isolates *when* to hedge from *how much*.

## Where it works

| Regime | Sharpe | Return | Max drawdown | Days |
|---|---|---|---|---|
| crash | -4.0550 | -0.4082 | -0.4031 | 122 |
| distribution | -3.0695 | -0.5843 | -0.5836 | 392 |
| expansion | 5.7705 | 0.6085 | -0.0568 | 167 |
| high_vol_chop | 1.8966 | 1.1198 | -0.1019 | 602 |
| trend_up | 5.8658 | 1.2078 | -0.0397 | 215 |

## Where it does not

| Regime | Days | Why there is no number |
|---|---|---|
| crash_rebound | 9 | too few bars to estimate the Sharpe |

The promotion gate failed these checks:

- `min_sharpe`
- `max_pbo`
- `max_is_oos_decay`
- `beats_simple_benchmark`
- `significance_t`

## The verdict

**Rejected.** It failed 5 of the 10 checks in the promotion gate: `min_sharpe`, `max_pbo`, `max_is_oos_decay`, `beats_simple_benchmark`, `significance_t`. The gate is fail-closed, so one failure is enough, and a rejected candidate is a normal, publishable result.

## What we got wrong

**This universe once contained no crypto at all.** Its declared assets were dropped by a filter that tested the wrong character, so the sweep ran on seven equity columns while the documentation described a mixed portfolio. Adding the crypto back moved the Sharpe from 1.1283 to 0.3838 and the zero-hedge share from 0.7966 to 0.0896. See trap 14.
