---
name: "vol_scaled_beta_hedge"
universe: "us_equities"
timeframe: "swing"
n_trials: 48
gate: "REJECTED"
failed_checks: ["max_pbo", "max_is_oos_decay", "min_regimes_positive", "significance_t"]
---

# vol_scaled_beta_hedge

## The idea

Scale a beta hedge inversely to realised volatility across all scales.

## The failure mode it targets

a volatility regime that is expanding faster than the portfolio can absorb

## The conditions

- Hold a target volatility of 0.235 (the `target_vol` parameter)
- Scale the hedge by how far realised volatility sits above that target, with a sensitivity of 0.322 (the `vol_gamma` parameter)
- Only add protection once the portfolio is actually down 8.3% (the `dd_trigger` drawdown trigger), ramping to full by 24.4% (the `dd_full` level)
- Act only when at least 0.513 of the volatility scales agree (the `consensus_required` agreement threshold)
- Cap the hedge at 0.581 of the portfolio (the `max_hedge_fraction` cap), and change it by no more than 0.370 a day (the `max_daily_turnover` cap)
- Hold no hedge at all while the market is expanding and the trend is intact (the `veto_on_expansion` rule)
- Pay 1.0 basis points of commission and 2.0 of spread on every trade, plus 10.0 a year to borrow (the cost model)

## Where it came from

The candidate bank in `hedging_agent/strategies.py`, and the thresholds and their primary sources in [`docs/LITERATURE.md`](../docs/LITERATURE.md).

## What would have falsified it

Before running it, this strategy had to beat a constant hedge carrying the same realised risk, hold up out of sample, and work in more than one market regime. If it could not, it was not a strategy -- it was a lucky configuration.

## What we measured

Every figure below comes from one sweep of **48 trials** over the us_equities universe on the swing timeframe, measured over **2021-10-25 to 2026-10-23 (1255 bars, 4.99 years)**. The Sharpe is the winner's, and it is only meaningful next to the number of things tried to find it.

- **Sharpe (annualised):** 1.2760 hedged, against 0.7635 unhedged
- **Deflated Sharpe:** 1.0000 -- the probability that the observed Sharpe beats the best of 48 random trials
- **Probability of backtest overfitting:** 0.9857 over 70 combinations of 48 configurations
- **Newey-West HAC t-statistic:** 2.7488 against a hurdle of 3.0
- **In-sample to out-of-sample decay:** 0.5284 (the worst fold)
- **Positive folds:** 0.8750 over 8 windows
- **Action ratio:** 0.1273 -- the share of days on which the hedge actually changed
- **Veto ratio:** 0.8299 -- the share of days the risk rule said no
- **Zero-hedge share:** 0.8032 -- the share of days on which the correct answer was to hold no hedge at all
- **Max drawdown:** -0.2245 hedged, against -0.2245 unhedged
- **Hedge instrument:** SH (LONG (instrument already inverse to portfolio)), correlated -0.9476 with the portfolio
- **Costs charged:** 36.84x annualised turnover, at 3.0 basis points of commission and spread plus 10.0 a year to borrow

Against a constant hedge carrying the same realised risk: the hedge ratio was 0.1042, and the timing was worth 0.4584 of Sharpe. That is the number that isolates *when* to hedge from *how much*.

## Where it works

| Regime | Sharpe | Return | Max drawdown | Days |
|---|---|---|---|---|
| expansion | 2.9317 | 1.4484 | -0.0379 | 824 |
| high_vol_chop | -2.7987 | -0.0566 | -0.0519 | 82 |

## Where it does not

| Regime | Days | Why there is no number |
|---|---|---|
| distribution | 26 | too few bars to estimate the Sharpe |
| trend_up | 3 | too few bars to estimate the Sharpe |

The promotion gate failed these checks:

- `max_pbo`
- `max_is_oos_decay`
- `min_regimes_positive`
- `significance_t`

## The verdict

**Rejected.** It failed 4 of the 10 checks in the promotion gate: `max_pbo`, `max_is_oos_decay`, `min_regimes_positive`, `significance_t`. The gate is fail-closed, so one failure is enough, and a rejected candidate is a normal, publishable result.

## What we got wrong

**Our own PBO was biased low.** The out-of-sample rank was scaled by (rank - 1) / N instead of rank / (N + 1), which flagged one rank too few on even N and flattered every candidate by 1/N (0.25 at N=4). Found by cross-checking against an independent implementation. The published values here are post-fix. See trap 1.
