---
name: "asymmetric_fast_on_slow_off"
universe: "crypto"
timeframe: "swing"
n_trials: 48
gate: "REJECTED"
failed_checks: ["min_sharpe", "positive_fold_ratio", "max_is_oos_decay", "min_regimes_positive", "significance_t"]
---

# asymmetric_fast_on_slow_off

## The idea

Add hedge quickly on the vol breach, release it slowly so the hedge does not churn.

## The failure mode it targets

turnover cost from re-hedging on every vol oscillation

## The conditions

- Hold a target volatility of 0.286 (the `target_vol` parameter)
- Scale the hedge by how far realised volatility sits above that target, with a sensitivity of 0.547 (the `vol_gamma` parameter)
- Only add protection once the portfolio is actually down 7.1% (the `dd_trigger` drawdown trigger), ramping to full by 22.4% (the `dd_full` level)
- Act only when at least 0.383 of the volatility scales agree (the `consensus_required` agreement threshold)
- Cap the hedge at 0.698 of the portfolio (the `max_hedge_fraction` cap), and change it by no more than 0.339 a day (the `max_daily_turnover` cap)
- Hold no hedge at all while the market is expanding and the trend is intact (the `veto_on_expansion` rule)
- Pay 1.0 basis points of commission and 2.0 of spread on every trade, plus 10.0 a year to borrow (the cost model)

## Where it came from

The candidate bank in `hedging_agent/strategies.py`, and the thresholds and their primary sources in [`docs/LITERATURE.md`](../docs/LITERATURE.md).

## What would have falsified it

Before running it, this strategy had to beat a constant hedge carrying the same realised risk, hold up out of sample, and work in more than one market regime. If it could not, it was not a strategy -- it was a lucky configuration.

## What we measured

Every figure below comes from one sweep of **48 trials** over the crypto universe on the swing timeframe, measured over **2024-01-29 to 2026-10-24 (1000 bars, 2.74 years)**. The Sharpe is the winner's, and it is only meaningful next to the number of things tried to find it.

- **Sharpe (annualised):** 0.2124 hedged, against 0.0841 unhedged
- **Deflated Sharpe:** 1.0000 -- the probability that the observed Sharpe beats the best of 48 random trials
- **Probability of backtest overfitting:** 0.2857 over 70 combinations of 48 configurations
- **Newey-West HAC t-statistic:** 0.4692 against a hurdle of 3.0
- **In-sample to out-of-sample decay:** 1.8545 (the worst fold)
- **Positive folds:** 0.5000 over 8 windows
- **Action ratio:** 0.1147 -- the share of days on which the hedge actually changed
- **Veto ratio:** 0.1103 -- the share of days the risk rule said no
- **Zero-hedge share:** 0.0647 -- the share of days on which the correct answer was to hold no hedge at all
- **Max drawdown:** -0.6363 hedged, against -0.8024 unhedged
- **Hedge instrument:** BTC/USDT (SHORT (instrument correlated with portfolio)), correlated 0.8166 with the portfolio
- **Costs charged:** 21.66x annualised turnover, at 3.0 basis points of commission and spread plus 10.0 a year to borrow

Against a constant hedge carrying the same realised risk: the hedge ratio was 0.0000, and the timing was worth 0.1283 of Sharpe. That is the number that isolates *when* to hedge from *how much*.

## Where it works

| Regime | Sharpe | Return | Max drawdown | Days |
|---|---|---|---|---|
| distribution | -2.4332 | -0.6307 | -0.6555 | 253 |
| high_vol_chop | -1.7714 | -0.4708 | -0.5238 | 228 |

## Where it does not

| Regime | Days | Why there is no number |
|---|---|---|
| crash | 36 | too few bars to estimate the Sharpe |
| crash_rebound | 2 | too few bars to estimate the Sharpe |
| expansion | 90 | too few bars to estimate the Sharpe |
| trend_up | 71 | too few bars to estimate the Sharpe |

The promotion gate failed these checks:

- `min_sharpe`
- `positive_fold_ratio`
- `max_is_oos_decay`
- `min_regimes_positive`
- `significance_t`

## The verdict

**Rejected.** It failed 5 of the 10 checks in the promotion gate: `min_sharpe`, `positive_fold_ratio`, `max_is_oos_decay`, `min_regimes_positive`, `significance_t`. The gate is fail-closed, so one failure is enough, and a rejected candidate is a normal, publishable result.

## What we got wrong

**This universe's numbers were wrong twice before they were right.** The Sharpe and the borrow charge were both annualised with the **equity** factor (252 bars a year) instead of crypto's 365 -- a 16.9% understatement of the Sharpe and a 45% overstatement of the financing cost. Both were found by noticing that one artefact reported two different Sharpes for the same return series. See traps 3 and 13.
