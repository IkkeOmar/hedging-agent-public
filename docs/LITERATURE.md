# Literature notes — what the published evidence says about what we are building

Read 2026-10-22, before the second round of implementation. These notes exist
because the first implementation was wrong in two ways and neither was caught by
a code review. Both errors were caught by *reading the literature*, and both
errors would have shipped as a plausible-looking Sharpe.

Rule adopted as a result: for quantitative work, a claim of empirical
correctness requires (a) a citation, (b) a golden test against a printed number,
or (c) an explicit "this is an assumption, not a finding".

---

## 1. Volatility scaling is NOT free — and our headline result is exactly the
   thing the literature says to distrust

**Moreira & Muir (2017), "Volatility-Managed Portfolios", Journal of Finance.**
The canonical claim: scaling exposure inversely to realised variance raises
Sharpe. That is the mechanism our `vol_multiplier` implements.

**Cederburg, O'Doherty, Wang & St Whaley (2020), "On the Performance of
Volatility-Managed Portfolios", Journal of Financial Economics.** They test 103
equity strategies. Finding: *"Volatility-managed portfolios do not systematically
outperform their corresponding unmanaged portfolios in direct comparisons."*
They are consistent with Moreira-Muir on the variance-reduction mechanism, but
they explicitly reject the market-timing interpretation.

**Guo & Liu (2020), SSRN 3385377, "Volatility-Managed Portfolios: True
Market-Timing with a False Theory?"** Attacks the theoretical foundation: the
constant `c` in the portfolio weight is arbitrary, and the risk-avariance
interpretation does not survive scrutiny.

**Xu et al. (2024), "Improving Volatility-Managed"** (ivo-welch.org). Recent
studies find poor out-of-sample performance; propose effective risk scaling +
conditional expected return + intercept.

### What this means for us — ACTION REQUIRED

Our current sweep on US equities / swing printed:

    sharpe hedged 2.4423   unhedged 1.5796   max DD -0.067 vs -0.134

That is precisely the shape of result Cederburg et al. show is not robust.
Reading it as "the hedge works" would be the error. Three changes:

1. **The hedged Sharpe must be compared against the best simple alternatives**,
   not just against buy-and-hold. Vol targeting, drawdown control, and a plain
   200-day trend rule are the honest benchmark set. We have a regime-adaptive
   reference repo that reports exactly this benchmark table — see
   `docs/UPSTREAM.md`. We must produce the same table.
2. **Report the Sharpe improvement with its DSR and its PBO**, which we now do.
   Our DSR of 1.0 is not reassuring on its own; it is a consequence of a
   3-year sample with a favourable drift regime.
3. **The result must be replicated out-of-sample and across universes.** A
   Sharpe gain that appears only in one regime of one universe is the pattern
   these papers describe.

Load-bearing lesson: *a Sharpe improvement from inverse-vol scaling is a
hypothesis, not a finding, until it survives the simple-alternative benchmark.*

---

## 2. Momentum crashes are a direct, documented warning against our regime
   gating

**Daniel & Moskowitz (2015), "Momentum Crashes", Journal of Financial
Economics.** (NBER w20439; free PDF at kentdaniel.net/papers/published/mom12.pdf)

Key findings we must respect:

- Momentum crashes are **partly forecastable**: they occur in *panic states* —
  after market declines, when volatility is high, **contemporaneous with market
  rebounds**.
- The mechanism is **time-varying beta**. After major declines, the past-loser
  decile's beta can exceed 3 while the past-winners' beta falls below 0.5.
  "When the market rebounds quickly, momentum strategies will crash because they
  have a conditionally large negative beta."
- In bear markets, up-beta is more than double down-beta (−1.51 vs −0.70,
  t = 4.5). In momentum portfolios this means behaving like **written calls on
  the market**.
- Betas computed with **forward-looking** information produce a striking bias:
  "the loser-portfolio beta with the contemporaneous return on the market."
  Hedging on ex-ante betas does **not** reproduce the published performance
  improvement.

### What this means for us — ACTION REQUIRED

Our regime detector classifies `EXPANSION / TREND_UP / HIGH_VOL_CHOP /
DISTRIBUTION / CRASH` and the policy **hedges most in CRASH and DISTRIBUTION,
and releases the hedge in EXPANSION**. Daniel-Moskowitz says the worst case is
specifically: high volatility + market decline + **then a rebound**.

Our veto fires on EXPANSION, which includes the rebound. That is exactly the
transition where a volatility-gated overlay releases protection *just before* the
crash it was supposed to prevent. This is a mechanism-level defect, not a
parameter problem.

Required changes:

1. Add an explicit **crash-rebound** state: high realised vol, recently negative
   trend, and a **sharp positive return burst**. In that state the veto must be
   **suspended**, because vol is still elevated while direction has flipped.
2. The detector must use **ex-ante (past-only) betas only**. Never a
   contemporaneous or forward beta anywhere in the sizing path. We have a
   no-lookahead test but we do not yet have a test that the *regime label* at
   bar t depends only on bars <= t.
3. Momentum-like strategies (long winners / short losers) are inherently
   written-call shaped. A hedged overlay over such a book inherits that. If our
   universe is a long-beta basket we should say so; if it is long/short
   momentum, the overlay needs a convexity check we do not yet have.

Load-bearing lesson: *the gate that says "vol is up, trend is down, hedge more" is
insufficient — the dangerous state is vol up AND trend flipping positive.*

---

## 3. Our significance bar is too low, by an entire published standard

**Harvey, Liu & Zhu (2016), "… and the Cross-Section of Expected Returns",
Review of Financial Studies 29(1).** They review 313 factor papers and derive
historical multiple-testing cutoffs. Their recommendation: **a new factor needs a
t-statistic above 3.0**, not the conventional 2.0. They argue **most claimed
research findings in financial economics are likely false**, and note that many
more factors were tried than were published — so the effective number of tests is
larger than the published record.

They also state a caveat we must honour: **theory-derived factors deserve a
lower hurdle than purely empirical ones**, because a theory constrains the search
space. And their tests are unconditional; a factor can be marginal on average and
critical in one regime — which is *precisely our regime breakdown argument*.

### What this means for us — ACTION REQUIRED

Our current `GateConfig.min_dsr = 0.90` and `min_sharpe = 0.6` are internally
chosen and defensible only as engineering hygiene. They are not research claims.

1. Report the **effective number of trials** as the primary line, not a footnote.
   Our sweep printed `trials run 40` as a header comment. It should be the
   headline, because it is the input that governs everything else.
2. Add an explicit **t-statistic** for the hedged-minus-unhedged Sharpe
   difference, with Newey-West standard errors (returns are autocorrelated and
   the overlay acts on vol, which is persistent). A Sharpe *difference* on
   overlapping windows is not a t-test away from zero — we currently imply it is.
3. The honest claim for an overlay with no theoretical prior is a **t > 3.0**
   hurdle. Until we clear it, our wording must stay "the gate passed" and never
   "we found an edge".

---

## 4. Tail-hedging overlays have a bad published track record, and our hedge
   instrument choice is in the worst part of it

**Cederburg et al. (2020)** again, and the tail-risk literature generally:
studies of put-based tail hedging find **little evidence of drawdown reduction**
and consistently **lower total returns**.

**AQR, "Pathetic Protection" (Wint 2019)** — the title is the finding. Buying
20-day puts does not significantly improve peak-to-trough drawdown over typical
holding periods.

**Cushman, "Tail-Risk Hedging" (CBS research portal)** — *"We find little
evidence that the put option monetization strategies reduce portfolio
drawdowns, and we find that all the monetization strategies have lower total
portfolio returns."*

### What this means for us — ACTION REQUIRED

Our equities hedge proxy is **SH, a daily-reset inverse leveraged ETF**. This is
the instrument these papers are most negative about, for a specific mechanical
reason: **daily reset compounds path-dependently**. Over multi-day drawdowns SH
can diverge substantially from −1× the index. Our 3-year backtest measured the
realised −0.9367 correlation, which is reassuring for the aggregate but says
nothing about the compounding error inside a crash.

Required:

1. Measure and **report the realised tracking behaviour of SH inside drawdown
   episodes specifically**, not the whole-sample correlation. If SH does not
   deliver its inverse payoff during the events it exists for, the hedge is a
   cost centre exactly when it is needed.
2. State plainly in the README that a short-futures or put-based hedge is the
   *conventional* instrument and that SH is a proxy chosen for data availability.
   Do not let the proxy choice be invisible.
3. Longer-horizon / investment-timeframe hedging with daily-reset products is the
   worst possible pairing. The `position` timeframe spec needs an explicit
   warning or a different instrument.

---

## 5. Methodological notes on our own statistics (validation, not criticism)

**DSR — Bailey & López de Prado (2014), JPM 40(5).** Verified against the
paper's printed "A Numerical Example": SR0 = 0.1132, DSR = 0.9004 at N=100,
T=1250, skew=−3, raw kurtosis=10, plus the secondary claims (0.9505 at N=46, and
0.9505 at N=88 under normality). Four golden tests now reproduce these.

Two errors were found and fixed during this validation, both of which had passed
a naive review:

- **Kurtosis must be RAW (normal = 3.0), not excess.** Passing excess kurtosis
  shrinks the PSR denominator and silently inflates the probability.
- **The PSR z-statistic requires the `sqrt(T-1)` factor.** Without it the DSR is
  not scale-invariant in sample length.

**PBO — Bailey, Borwein, López de Prado & Zhu (2016), Alg 2.3.** Our
implementation now uses the correct 1-based → 0-based rank shift. For iid noise
the expected PBO is 0.5; our noise control test asserts that band.

Note on interpretation: **PBO is only interpretable when the candidate set
contains genuinely distinct strategies.** Our current candidate pool is
40 parameter variants of ~5 mechanisms, which are highly correlated by
construction. A PBO of 0.99 on a near-identical set is not evidence that the
mechanism fails — it is mostly evidence that the IS winner is arbitrary within
a tight cluster. The honest use of PBO here is: *"we cannot distinguish our top
configurations from each other out-of-sample."* The fix is candidate diversity,
not a better Sharpe. This is recorded as a known limitation, not a solved problem.

---

## 6. What the upstream projects already established (and we must not re-derive)

**tanishhky/regime-adaptive-portfolio** (MIT, SSRN 6484679) — the closest prior
art. Their reported result: multi-scale regime detection with a 2-of-4
orthogonal detector consensus, walk-forward 2009–2025, max DD −6.2% vs SPY
−41.7%, and **crucially, a full honest benchmark table** in which on the SPY
universe *simple 200-day MA timing and drawdown control beat the system on raw
Sharpe*. They state plainly that the system's edge is lowest drawdown, not
highest Sharpe.

That is the benchmark discipline our results must be measured against, and it
corroborates §1.

**Starlight143/AlphaAirlock** — LLM agents generate → sandbox backtest → paper
trade → **gate before live routing**, live trading off by default. Our pipeline
is the same shape. Their engine is explicitly "look-ahead-bias-free".

**nhocconan/AutoResearch-...** — locks the engine, metrics and rules as
immutable and CI-tests the guarantees ("the agent cannot make the evaluation
easier on itself"). We should adopt the same rule for our walk-forward splitter
and metrics module.

**navil2468/Offset** — the agentic-hedging concept via prediction markets. Their
stated next step is exactly ours: moving from "surfaces candidates for human
decision" to autonomous, state-tracking hedge management.

---

## Summary of concrete changes required before any result is reported

| # | Change | Reason | Source |
|---|--------|--------|--------|
| 1 | Benchmark against simple alternatives, not just buy-and-hold | vol-scaling Sharpe gains are not robust | Cederburg et al. 2020 |
| 2 | Add an explicit crash-rebound regime; suspend the veto there | vol up + rebound is the documented crash state | Daniel & Moskowitz 2015 |
| 3 | Ex-ante betas only; add a regime-label causality test | forward betas produce published-looking bias | Daniel & Moskowitz 2015 |
| 4 | Report n_trials as the headline, add Newey-West t-stat on the Sharpe difference | t>3.0 hurdle, most findings likely false | Harvey, Liu & Zhu 2016 |
| 5 | Report SH tracking inside drawdown episodes, not whole-sample correlation | daily-reset inverse products diverge path-dependently | AQR 2019; Cushman (CBS) |
| 6 | Disclose the proxy choice and the 24/7-vs-equity calendar mismatch | proxy and calendar choices are not neutral | our own honesty rule |
| 7 | Increase candidate diversity or stop reporting PBO on a near-identical set | PBO measures the selection, not the mechanism | Bailey et al. 2016 Alg 2.3 |

Nothing in this repo should claim to have found an edge until items 1-5 are done
and the numbers are re-run.

---

## Citations

- Bailey, D.H., Borwein, J.M., López de Prado, M., Zhu, Q.J. (2016).
  *The Probability of Backtest Overfitting.* Journal of Computational Finance.
  SSRN 2326253.
- Bailey, D.H., López de Prado, M. (2014). *The Deflated Sharpe Ratio:
  correcting for selection bias, backtest overfitting, and non-normality.*
  Journal of Portfolio Management 40(5), 94-107. davidhbailey.com/dhbpapers/deflated-sharpe.pdf
- Cederburg, S.C., O'Doherty, M.S., Wang, F., St Whaley, X. (2020).
  *On the Performance of Volatility-Managed Portfolios.* Journal of Financial
  Economics. SSRN 3357038.
- Cushman, P. *Tail-Risk Hedging.* Copenhagen Business School research portal.
- Daniel, K., Moskowitz, T.J. (2015). *Momentum Crashes.* Journal of Financial
  Economics. NBER w20439.
- Harvey, C.R., Liu, Y., Zhu, H. (2016). *… and the Cross-Section of Expected
  Returns.* Review of Financial Studies 29(1), 5-68. NBER w20592.
- Moreira, A., Muir, T. (2017). *Volatility-Managed Portfolios.* Journal of Finance.
- T. Wu / Xu et al. (2024). *Improving Volatility-Managed.* ivo-welch.org.

---

## 10. Volatility drag — the one channel that CAN be positive without an edge
    (added 2026-10-22, after implementing and then CORRECTING the math)

This section is the honest counterweight to sections 1 and 7. Everything above
is about why the *timing* claim is fragile. This is about the one part that is
arithmetic rather than a forecast, and therefore might actually work.

### The definition, from the literature

**Volatility tax** — term coined by **Rick Ashburn, CFA (2003)**, formalised by
**Mark Spitznagel**, defined in Wikipedia's "Volatility tax" article; also called
volatility drag, volatility decay, or **variance drain**.

> Quantitatively, the volatility tax is the difference between the arithmetic and
> geometric average returns of an asset or portfolio. It thus represents the
> degree of "non-ergodicity" of the geometric average.

Under geometric Brownian motion with drift mu and volatility sigma, the geometric
average return is mu − sigma²/2, so the tax is **sigma²/2**. Spitznagel's framing:
it is "a hidden, deceptive fee levied on investors by the negative compounding of
the markets' swings" — losing 50% requires a 100% gain to recover.

Crucially it is **quadratic in sigma**. That single property is the entire reason
a vol-reducing overlay can pay:

    sigma cut by factor g  ->  tax cut by factor g^2

### Why this survives the Cederburg critique

**Cederburg, O'Doherty, Wang & St Whaley (2020)** tested volatility-managed
portfolios across 103 equity strategies and found they "do not systematically
outperform their corresponding unmanaged portfolios in direct comparisons."

What they were contesting was the **Sharpe gain from mean-variance TIMING** —
that you can time risk by looking at variance and earn abnormal returns. They are
consistent with the variance-reduction mechanism itself. The sigma²/2 reduction
is not a timing claim, so it is not what they refuted. That distinction is the
whole argument, and it is the only version of the vol-scaling thesis I consider
defensible.

**Van Hemert, Harvey, Hoyle, Korgaonkar, Rattray & Sargaison (2018), "The Impact
of Volatility Targeting", Journal of Portfolio Management 45(1), 14-33** — the
pro-side evidence, and the most useful paper found in this round. Man AHL, 60+
assets, daily data from 1926. Findings:

- Sharpe ratios ARE higher with volatility scaling for risk assets (equities and
  credit) and for portfolios with substantial allocations to them.
- "Risk assets exhibit a so-called leverage effect (i.e., a negative relation
  between returns and volatility), and so volatility scaling effectively
  introduces some momentum into strategies."
- "Volatility targeting consistently reduces the likelihood of extreme returns
  (and the volatility of volatility) across our 60+ assets."
- "Under reasonable investor preferences, a thinner left tail is much preferred
  (for a given Sharpe ratio)."
- For bonds, currencies and commodities, "volatility scaling has a negligible
  effect on realized Sharpe ratios" — so this is a risk-asset phenomenon.

Note the honest caveat in their own reasoning: part of the Sharpe gain is that
vol scaling acts like time-series momentum. That means it inherits momentum's
crash risk (section 2) and is not a free lunch.

### Moreira & Muir on horizon

**Moreira & Muir (2018), "Should Long-Term Investors Time Volatility?", JFE.**
Their headline is a utility number, not a Sharpe number: *"A long-term investor
who ignores variation in volatility gives up the equivalent of **2.4% of wealth
per year**."* The mechanism is a **volatility composition channel**: investors
respond less to volatility when mean reversion disproportionally increases with
volatility. They find those conditions "unlikely to hold in the data" — so the
result is robust to the mechanism as long as mean-reverting volatility shocks
exist at all.

This connects to a specific finding in the original **Moreira & Muir (2017)**
NBER w22208 paper, Table 17: with a **constant** share of shocks, the optimal
volatility-timing policy is **constant** (all 1.00). The entire effect requires
mean-reverting volatility shocks. That is a concrete, testable premise, and it is
the right thing to name as an assumption in any claim we make.

### What I got WRONG implementing this, and how the literature caught it

Recorded because it is the clearest example in this project of the failure mode:

1. **First attempt** defined drag as `cagr - arith_mean`, giving drag = +0.0091
   (drag POSITIVE on a high-volatility sample — obviously wrong).
2. **Second attempt** "fixed" it to `arith_mean - cagr` with
   `cagr = exp(mean(log(1+r))*periods) - 1`. Still **negative** (−0.0091).
3. I then wrote a docstring claiming the QuantStats/empyrical "power" convention
   `prod(1+r)**(periods/n) - 1` "overstates growth" and that the gap widened with
   volatility. I asserted this without checking.
4. **Measured it against QuantStats 0.0.86 instead of asserting it.** The claim
   was false: the two conventions are algebraically identical
   (`prod(1+r)**(1/y) == exp(E[log(1+r)]*periods)`) and agree to 1e-9 across
   sd = 0.002 to 0.030.
5. **The real bug**: the tax must be differenced against the **LOG** mean, not
   the exponentiated one. `exp()` has already put the drag back into the
   exponent, so differencing against it returns a negative tax.

Correct implementation, now in `benchmarks.py::volatility_drag`:

    log_cagr = mean(log1p(r)) * periods
    drag     = arith_mean - log_cagr      # the tax, positive = paid
    implied  = vol**2 / 2                 # sigma^2/2, first order

Verified on 20y of daily N(0.0010, 0.012), 5040 bars:

    log_cagr              0.224340
    arith_mean            0.242406
    drag                  0.018066
    sigma^2 / 2           0.017986
    residual (3rd moment) 0.000080

Cross-validated against QuantStats 0.0.86 — CAGR, arithmetic mean and volatility
all agree to within 4e-7. Two pinned regression tests: one asserting the tax is
taken against the log mean (the bug above), one asserting the two annualisation
conventions are identical (the false claim above).

### Cross-checks worth knowing

- **QuantStats `avg_return(returns)`** takes NO `periods` argument — its second
  positional is `aggregate`, which wants a string like "year". It also drops zero
  returns from the mean, so it is not exactly `mean(r) * 252`.
- **`compsum`** is QuantStats' per-bar compounded cumulative return; the gap
  between it and `cumsum()` is the same drag measured bar by bar.

### Standing obligation

Until the numbers in `vol_drag` are reported alongside every overlay result, no
claim in this repo should describe a CAGR improvement without saying whether it
came from a lower tax (real, edge-free) or from a higher arithmetic mean (which
requires an actual forecast and must clear the t > 3.0 hurdle).
---

## 11. Cross-validating our statistics against ml4t.diagnostic — and the bias it exposed
    (added 2026-10-22)

### Why bother validating against another library?

Because the DSR and PBO numbers in this repo are used to **reject** strategies,
so a bug in them silently flatters us. Prior art:

| project | stars | last push | role here |
|---|---|---|---|
| **ml4t/diagnostic** | 39 | 2026-10-22 | Independent DSR/PBO/FDR. Same authors as *Machine Learning for Trading* (3rd ed., 21.4k stars). Cross-validation reference. |
| ml4t/backtest | 26 | 2026-10-22 | Event-driven engine, "framework-parity validation". Candidate execution layer. |
| stefan-jansen/machine-learning-for-trading | 21.4k | active | Course + notebooks. The canonical workflow reference. |
| oos-lab | 2 | 2026-07-19 | Small independent OOS-testing package. |
| Riskfolio-Lib | 4559 | 2026-10-22 | Portfolio optimisation. Useful for the constraint layer. |
| PyPortfolioOpt | 6117 | active | Efficient frontier, Black-Litterman, HRP. |
| skfolio | 2519 | active | scikit-learn-native portfolio optimisation. |
| purgedcv | — | active | Purged + combinatorial CV, **already implements DSR**. Second reference. |
| arch | — | active | GARCH / conditional volatility. |
| wraquant, quantcore, regime-aware-portfolio-risk-engine | 0–2 | 2026-06 | Same thesis as ours, tiny or stale. Closest prior art by intent, weakest by rigour. |

`ml4t/diagnostic` (v0.1.9) also publishes the two rules worth stealing
verbatim, both in `docs/user-guide/statistical-tests.md`:

> "Include every strategy variant considered during selection. Omitting failed
> or discarded variants understates the multiple-testing penalty."

> "The candidate set belongs to the selection exercise, not to one Python
> process."

That is the **trial ledger**: persist every variant ever evaluated, across
sessions, keyed by a stable id that is never reused. Our `n_trials` is
currently the per-run trial count, which is the undercounting mistake the
doc calls out. Noted as a known gap, not fixed here.

### What the comparison found: our PBO was biased toward passing

Two implementations of "probability that the in-sample winner is in the bottom
half out-of-sample" should agree on the same inputs. Ours did not. The first
disagreements we saw were single draws, and we did not record the seed, so
that observation is not reproducible and we do not publish it as a result.
What is reproducible is the average, over 400 iid draws with no skill
whatsoever, and it is the table a reader should check:

| N configs | before fix | after fix | ml4t | truth |
|---|---|---|---|---|
| 6 | 0.324 | 0.494 | 0.504 | 0.5 |
| 10 | 0.413 | 0.515 | 0.509 | 0.5 |
| 12 | 0.421 | 0.503 | 0.516 | 0.5 |

Regenerate it with `python scripts/reproduce_pbo_bias.py`. The script states
the protocol, carries a verbatim copy of the pre-fix estimator, and the
table is pinned against it by
`tests/test_pbo_bias_table_is_reproducible.py`. The "before" column needs
that copy because the pre-fix estimator is not in this repository any more.

| N configs | before fix | after fix | truth |
|---|---|---|---|
| 6 | 0.324 | 0.494 | 0.5 |
| 10 | 0.413 | 0.515 | 0.5 |
| 12 | 0.421 | 0.503 | 0.5 |

**The bug.** Bailey & López de Prado define, with omega the 1-based
out-of-sample rank among N configurations:

    theta = omega / (N + 1) - 0.5        in (-0.5, 0.5]
    PBO   = P(theta > 0)

What we actually wrote was not `omega / N`. It was

    theta = (omega - 1) / N - 0.5

a **zero-based** rank over N configurations, where Bailey & López de Prado
divide the **one-based** rank by **N+1**. The difference is not one rank, it is
one rank *and* a shift, and the two mistakes partly cancel on odd N.

The verdict is `theta > 0`, so the question is how many of the N ranks each
form puts above the line:

| form | theta > 0 when | ranks flagged, even N | ranks flagged, odd N |
|---|---|---|---|
| Bailey & López de Prado | `omega > (N+1)/2` | N/2 | (N−1)/2 |
| ours (before the fix) | `omega > N/2 + 1` | N/2 − 1 | (N−1)/2 |

For **even** N we flagged one rank too few, so PBO was biased **downward** by
exactly **1/N**: 0.25 at N=4, 0.17 at N=6, 0.10 at N=10, 0.08 at N=12. For
**odd** N the two forms agree and the bias is zero. (N=2 is degenerate: the old
form flagged no rank at all.) Every candidate looked less overfit than it was
— precisely the wrong direction of error for a gate whose job is to catch
overfitting.

Getting the mechanism right matters as much as getting the number right. An
earlier draft of this section said we "divided by N instead of N+1", which is
not what the code did: a reader who implemented *that* literally would have
measured no bias at all on even N, and the claim would have looked false.

**Why the test suite missed it.** The existing test asserted a single draw
satisfied `0.2 <= pbo <= 0.8`. A systematic bias of −0.08 to −0.25 sits
comfortably inside that band. A band test on one sample cannot detect a bias;
only an average over many draws can. Replaced with an averaging test plus a
data-free arithmetic pin on the (N+1) divisor.

**Secondary finding — resolution limits.** ml4t's PBO is quantised to
multiples of 1/n_combinations. With 4 combinations the only attainable values
are {0, 0.25, 0.5, 0.75, 1.0} — verified over 200 draws. So a PBO printed to
four decimals from four splits is spurious precision, and any gate reading
PBO > 0.5 as a hard fail needs **at least ~16 combinations** (8 splits) to
mean anything at that threshold.

### Method note

Our PBO ranks on **mean return**; the comparison harness scored each half by
**Sharpe**. The metric choice is not neutral: ranking by Sharpe penalises
high-variance configurations, so it changes which config "wins" in-sample. Our
mean-return choice is the more conservative one for a *hedging* study where the
thing being compared is a return stream with different vol profiles, and it is
consistent with `n_bars`-weighted comparisons elsewhere. But it is a choice,
and it is now written down rather than left implicit.
---

## 12. The candidates are one strategy: DSR is answering the wrong question

Measured on the live `us_equities` / swing sweep, 6 templates x 7 sampled
variants = **42 candidate OOS streams over 1253 bars**:

| quantity | value |
|---|---|
| mean pairwise correlation | **0.9536** |
| min / max pairwise correlation | 0.8052 / 0.9998 |
| leading eigenvalue of the correlation matrix | **40.13 of 42** (95.5% of total variance) |
| participation-ratio effective trial count K_eff | **1.09** |
| raw K overstates the search space by | **38.4x** |

So the sweep ran, statistically, *one* risk exposure 42 times. Six strategy
families with overlapping search spaces produce near-identical return streams.

### The finding, and the part I got wrong

The expected result was that deflating against K_eff instead of raw K would
tighten the DSR hurdle and flip the gate. **Measured, it does not.**

```
raw K = 42    E[max SR] under null = 0.4657    DSR = 1.0000
K_eff = 1     E[max SR] under null = 0.0000    DSR = 1.0000
```

Both pass the 0.90 gate. The reason is a **cancellation between two biases in
opposite directions**:

1. Raw K inflates the search penalty: 42 trials instead of ~1, so E[max SR] is
   pushed up to 0.4657.
2. But `trials_variance` is **measured from the same correlated set**. Sharpes
   across 0.95-correlated streams barely differ (std 0.2108), so V is small,
   and V small pulls E[max SR] back down.

DSR lands at 1.0 either way. The statistic is not wrong; it is **answering a
question nobody asked**.

### Why that still matters

At K_eff = 1 the null is "a single strategy with zero true Sharpe" and
E[max SR] = 0. A hurdle of zero is not a test — it is the absence of one. The
real question is not *how many trials* but *is this Sharpe distinguishable from
zero at all*, and that is governed by the Sharpe's own sampling error:

```
observed Sharpe      1.2942 (winner)  |  1.5182 (synthetic check)
Lo (2002) SE         sqrt((1 + SR^2/2)/T)  =  0.0414
95% CI               [1.4370, 1.5995]
SR / SE              36.6
```

At T = 1253 a Sharpe of 1.29 is many standard errors from zero. That statistic
is enormous *and it is the wrong one*, because it only knows the sample length,
not whether the strategy was selected as the best of a family. Both the search
penalty and the sampling-error term have to appear, and in this pipeline they
were not.

### What this changes

- **DSR cannot be the primary gate for a correlated candidate set.** It is
  reliable when trials are genuinely independent — which is the regime its
  author assumed, and the regime the ml4t docs assume. Our sweep does not
  satisfy that assumption, and the docs say as much ("The probability reflects
  the supplied tested variants").
- **PBO is also degenerate here.** On the real matrix it returns exactly 1.0
  (70 combinations, 42 configs). That is not a strong result; it is an
  estimator with no power, because every configuration ranks identically in
  every split and omega barely moves. The gate "PBO > 0.5 -> reject" was right
  by luck, not by measurement.
- **The honest fix is to widen the search, not to sharpen the statistic.** A
  candidate set with 0.95 mean correlation has not tested 42 strategies; it has
  tested one. Genuine diversity comes from different hedge instruments,
  different portfolio legs, and different rebalance cadences — not from
  sampling one template's parameters seven times.
- **Report K_eff beside every DSR**, or the number is unauditable. This is the
  single cheapest change here and it would have exposed this immediately.

### Status

Unresolved and blocking promotion. The gate currently REJECTS on the
Newey-West t (1.68 vs t > 3.0), so nothing reaches production while this is
open — but the DSR and PBO columns in every report are, at present, not
carrying the weight they appear to.

---

## 13. PBO is not measuring candidate diversity (measured, hypothesis wrong)

Section 12 established that the candidate set is effectively one strategy
(K_eff ~ 1.09). The natural follow-up hypothesis: **PBO is high because the
trials are too correlated to discriminate.** That is wrong, and measuring it
across all three universes shows the relationship is inverted.

| universe | bars | PBO | median OOS win prob | HAC t |
|---|---|---|---|---|
| us_equities | 1254 | **0.9857** | **0.102** | 2.7488 |
| crypto | 999 | **0.2857** | **0.7347** | 0.4692 |
| mixed | 1826 | **0.9571** | **0.0408** | -1.021 |

Every cell in that table is read straight out of the committed sweep
artefacts: `bars` from `pbo.n_observations`, `PBO` from `pbo.pbo`, the
median out-of-sample win probability from `pbo.median_oos_win_prob`, and
the t-statistic from `significance.t_stat`. `scripts/audit_claims.py`
fails the build if any of them drifts.

The candidate-correlation numbers -- the mean pairwise correlation and the
participation-ratio effective trial count -- are **not** in that table,
because no command in this repository produces them. They were measured
once, by hand, against the candidate return matrices; they are the
reason the DSR row is treated as uninformative, and they are quoted in
the README. A reader cannot rerun them, so they are stated as what they
are: a hand measurement, not a reproduced artefact.

On that hand measurement, crypto is the **most** correlated candidate set
of the three and has the **lowest** PBO. Diversity does not explain PBO.

### What actually explains it

PBO is the frequency with which the in-sample winner lands in the out-of-sample
**bottom half**. So it splits cleanly on `median_oos_win_prob`:

- **crypto: 0.7347.** The in-sample winner usually stays above the OOS median.
  The selection *transfers*. PBO 0.2857 is a genuine (if unimpressive) result.
- **us_equities: 0.102.** The in-sample winner lands in the OOS bottom half
  about 90% of the time. This is not "no signal" — it is a **systematically
  inverted** selection, which is a worse pathology. The in-sample ranking is
  not merely uninformative, it points the wrong way.
- **mixed: 0.0408.** The same inversion, and worse.

All three still REJECT overall, for different reasons: us_equities and
crypto fail the Newey-West t (2.7488 and 0.4692 against 3.0), and mixed
fails it outright (t = -1.021) while also failing on Sharpe and on the
risk-matched benchmark.

### Consequences

1. **Do not use PBO as a diversity diagnostic.** It measures rank-transfer, not
   how many independent bets were made. K_eff does the latter.
2. **A near-1.0 PBO is a stronger warning than a mid-range one.** It says the
   ranking is anti-correlated out of sample, which usually means the selection
   criterion is capturing noise *inversely* — often a sign that the metric is
   aligned with something that mean-reverts hard on the test half.
3. The gate threshold `max_pbo = 0.30` was chosen before this was measured.
   Crypto's 0.3143 fails it by a hair while its selection demonstrably
   transfers. The threshold should be revisited against a documented
   distribution of PBO under the null, not set to a round number.

---

## 14. The cost model was configured, printed, and never charged

Three defects, found by asking whether costs actually reduce returns. They did
not — Sharpe was **identical at every cost level from 0 to 50 bps**.

### 14.1 Costs were never applied at all

`build_hedged_returns` computed `port + side*n*proxy` and returned it. The cost
fields existed in `HedgeConfig` (`commission_bps`, `spread_bps`,
`borrow_bps_annual`), were printed in the report metrics, and were **never
passed to the return calculation**. Every Sharpe in every sweep was gross of
costs. At the measured 40x annualised turnover that is not a rounding error.

`_hedge_cfg_from_spec` also never forwarded them, so a strategy could not even
state its own cost assumption — a cost-sensitivity analysis was impossible
without editing the library. Costs are now fields on `HedgeStrategySpec` and
flow spec -> config -> returns.

### 14.2 Borrow was charged 10,000x too large

First implementation:

```python
borrow_bps_annual / periods * n.abs()      # 100/252 * 0.2 = 0.079 per bar
```

The bps figure was treated as if it were already a converted fraction, so a
full *annual* borrow charge was applied on *every single bar* — 120% a year
instead of 10%, and 0.079 per-bar drag that drove Sharpe to -7.39.

Correct:

```python
borrow_per_bar = borrow_bps_annual / 10_000.0 / periods * n.abs()
# 0.01/252 * 0.2 = 7.94e-6 per bar -> 0.002/yr = 10% of the 0.2 notional
```

### 14.3 The first trade was free

The return window is `gross.iloc[1:]` (dropping the NaN bar from `pct_change`).
The cost series was computed on the *unsliced* index, so the entry cost charged
at bar 0 was discarded by the slice. Measured: a position ramped up to 0.2 and
back to 0 trades 0.4 of notional, but only 0.2 was ever charged.

Fixed by charging the trade placed at bar t at bar t+1
(`(traded/1e4).shift(1)`), matching the convention already used for returns.

### Effect on results

Cost sensitivity of the winning equity strategy (40x annual turnover):

| round-trip bps | hedged Sharpe | t-stat |
|---|---|---|
| 0 | 0.9203 | -0.049 |
| 1 | 0.9056 | -0.123 |
| **3 (default)** | **0.8763** | **-0.269** |
| 6 | 0.8322 | -0.487 |
| 12 | 0.7440 | -0.921 |
| 25 | 0.5525 | -1.825 |
| 50 | 0.1842 | -3.356 |

Sharpe falls **80%** from free to 50bps. At the 3bps default the drop is modest
(0.920 -> 0.876) because the hedge is small, but it is no longer zero.

Re-run results, all REJECTED:

| universe | Sharpe hedged | PBO | t | failed checks |
|---|---|---|---|---|
| us_equities | 1.276 | 0.9857 | 2.7488 | max_pbo, max_is_oos_decay, min_regimes_positive, significance_t |
| mixed | 0.3838 | 0.9571 | -1.021 | min_sharpe, max_pbo, max_is_oos_decay, beats_simple_benchmark, significance_t |
| crypto | 0.2124 | 0.2857 | 0.4692 | min_sharpe, positive_fold_ratio, max_is_oos_decay, min_regimes_positive, significance_t |

Crypto degrades most in *breadth* of failure: previously 3 checks, now 5. Its
Sharpe is low enough that costs alone break the `min_sharpe` and
`positive_fold_ratio` gates.

### Methodology note

All three of these were found by a single question — "does changing the cost
assumption change the answer?" — and not by reading the code for bugs. The
check is now a permanent test (`test_costs_are_actually_charged`,
`test_borrow_is_charged_as_an_annual_rate_not_per_bar`,
`test_zero_cost_reproduces_the_pre_cost_Sharpe_exactly`).

The deeper lesson: a cost model that is *declared* but not *applied* is worse
than no cost model, because the report prints `commission_bps` next to a Sharpe
that never paid it.

---

## 15. A 2-bar regime was passing the promotion gate

A crypto sweep printed:

```
crash_rebound  sharpe 417.3691  ret 0.146828  maxDD 0.0  n 2
```

Two bars. The Sharpe is not a finding — it is `mean / std(ddof=1) * sqrt(365)`
computed over a sample of two, and it landed in the report formatted exactly
like the real numbers around it.

### 15.1 The gate counted it as a regime where the strategy worked

```python
pos_regimes = sum(1 for v in regime_stats.values() if v.get("sharpe", 0) > 0)
```

`min_regimes_positive` requires the strategy to work in more than one regime.
That check was satisfiable by a single 2-bar regime with a lucky sign. It is a
check about *robustness across regimes*, and it was measuring arithmetic.

### 15.2 The threshold, derived rather than picked

The standard error of an annualised Sharpe estimated from `n` bars is
approximately `sqrt(periods / n)` (Lo, 2002). Inverting for a target SE:

    n >= periods / max_se^2

At the equity default (`periods=252`) and `max_se=2.0` that is **63 bars**.
For crypto (`periods=365`) it is **92 bars**.

Verified against simulation rather than trusted from the algebra — 4,000
draws per bar count, true annualised Sharpe 0.5:

| n | theory SE | empirical SE | ratio |
|---|---|---|---|
| 3 | 9.17 | 26.92 | 0.34 |
| 10 | 5.02 | 6.23 | 0.81 |
| 20 | 3.55 | 4.01 | 0.89 |
| 63 | 2.00 | 2.16 | 0.93 |
| 252 | 1.00 | 1.06 | 0.94 |

The approximation holds to within 7% for n >= 40 and collapses below that,
which is precisely the regime this guard exists for.

### 15.3 The fix

`regime_breakdown` now reports `sharpe=None` and `significant=False` for any
regime under the threshold, plus `sharpe_se` for the thick ones so the reader
can see how much precision they actually have. It reports `None` rather than
`0.0` deliberately: a zero would read as "measured, and it was flat".

The printer says why there is no number rather than printing a bare `n 2`,
because a bare bar count invites the reader to compute the Sharpe themselves.

The gate keys on `n_bars`, not on the `None` sentinel. That was a real
correction during implementation: keying on the sentinel left a hand-built
`{"sharpe": 417.4, "n_bars": 2}` still voting, since `regime_stats` can arrive
from any caller. The bar count is what determines estimability, so it is what
the guard has to test.

### 15.4 What it looks like now

```
crash          sharpe    n/a   ret -0.088622  maxDD -0.1399  n 23  (too few bars to estimate: SE > 2.0)
crash_rebound  sharpe    n/a   ret  0.146825  maxDD     0.0  n 2  (too few bars to estimate: SE > 2.0)
distribution   sharpe -3.4277  ret -0.820585  maxDD -0.8247  n 263
```

### Methodology note

Found by reading a report for plausibility rather than for correctness. A
number that looks absurd is a question, not a formatting bug. The general
form: **any statistic whose units make it blow up on small samples needs an
explicit floor, and the floor needs a derivation.**

Two tests pin it: `min_bars_for_sharpe` values (63 / 92), the suppressed
Sharpe, and the gate refusing to count a 2-bar regime.

---

## 16. The regime threshold was per-universe and I applied the equity one

Section 15 stopped thin regimes voting on `min_regimes_positive`. The first
version of that guard called `min_bars_for_sharpe()` with its default argument
— `periods=252`, the equity convention. Crypto annualises over 365 days, so its
threshold is 92 bars, not 63.

Crypto's `expansion` regime sits at **exactly 90 bars**: SE = sqrt(365/90) =
2.014, marginally over the 2.0 bound, but comfortably above the equity default
of 63. The first live run after the guard reported **zero** positive regimes
for crypto. Not "fewer than before" — zero, because the one qualifying regime
was excluded by a threshold two bars too generous.

```
crypto  before guard : expansion(90, +5.71), trend_up(70, +4.78)  -> 2
crypto  first guard  : equity default of 63 applied to 365-day data -> 0
crypto  after fix    : reads each regime's own sharpe_se            -> 0
```

(The final count is genuinely 0: 90 bars gives SE 2.014 > 2.0, so `expansion` is
correctly excluded on crypto's own terms. The bug was that the *reason* was a
wrong-universe default, and the guard would have wrongly admitted it had the
numbers fallen the other way. The threshold now travels with the data instead of
being assumed at the call site.)

### The fix

The estimability bound is computed once per regime, inside
`regime_breakdown`, using that universe's own `periods`, and carried on the
regime as `sharpe_se`. The gate reads it:

```python
pos_regimes = sum(
    1 for v in regime_stats.values()
    if (v.get("sharpe") or 0) > 0
    and v.get("sharpe_se") is not None
    and float(v["sharpe_se"]) <= 2.0
)
```

There is no default to get wrong, because the number that decides is the number
that was measured alongside the statistic it qualifies.

### Effect on the documented results

`min_regimes_positive` now fails for **all three** universes. The README
previously recorded us_equities as PASS at 2, which was a 3-bar `trend_up`
regime scoring Sharpe 7.57 doing the voting. Corrected:

| universe | pre-guard | post-guard |
|---|---|---|
| us_equities | 2 (PASS) | 1 (FAIL) |
| mixed | 1 (PASS) | 1 (FAIL) |
| crypto | 3 (PASS) | 0 (FAIL) |

us_equities failed 3 checks, now 4. crypto failed 5, now 6.

### Methodology note

Two failures in one sitting, both the same shape: a constant that is correct
for one universe applied to another. The first was `periods` in the annualiser,
the second `min_bars_for_sharpe()`'s default argument at the gate call site.
Both were invisible to a test suite because every fixture used equity data.

The structural lesson: **when a threshold depends on a universe-level constant,
derive it from that constant at the point of use, or carry it on the data.**
Passing a defaulted constant across a layer boundary is how equity defaults end
up silently governing crypto.

---

## 17. The data layer could silently research the wrong portfolio

A sweep failed with `yfinance returned nothing for [11 ETFs]`. The visible
error is loud and harmless. Probing the failure mode showed the case that
actually mattered was quiet:

```
partial (4 of 11)  -> RETURNED 4 cols: ['SPY', 'QQQ', 'IWM', 'XLK']
total empty         -> RuntimeError (good)
```

Yahoo returning 4 of 11 tickers produced a valid frame with 4 columns. No
error, no warning, no log line. The sweep continued, the gate evaluated, the
report printed, and every Sharpe described a portfolio missing seven of its
eleven assets. Nothing downstream could detect it: shorter history, different
volatility, different regime mix, all plausible.

The old loop was:

```python
for a in yahoo_assets:
    if a in close.columns:      # silently skips the rest
        frames[a] = close[a].dropna()
```

`if a in close.columns` is a filter, not a validation. A missing ticker is
indistinguishable from a ticker that was never requested.

### 17.1 Why refusing is correct, not just cautious

A degraded universe is not a worse version of the requested run — it is a
*different experiment* whose results happen to be printed in the requested
run's format. There is no threshold at which silently dropping 7 of 11 assets
becomes acceptable, so the code does not offer one.

### 17.2 Rate-limit handling, written against the real library

The first attempt at the retry wrapper was written from memory. Reading
yfinance's source and docs invalidated three assumptions:

| Assumed | Actual (yfinance 1.7.0) |
|---|---|
| library retries transient errors, including 429 | `_is_transient_error(YFRateLimitError())` returns **False** — it does not |
| `download(retries=N)` exists | **No such parameter** at 1.7.0 (PR #2627 is on `dev`) |
| `yf.YFRateLimitError` | It is in `yfinance.exceptions` only |
| clear the crumb via a fresh `YfData()` | No public reset; `YfData` has no clear method and no class-level instance |
| `set_tz_cache_location()` clears cookies | It redirects the **timezone** cache; cookies live in `cookies.db` |

Verified by direct introspection, not by reading docs alone:

```python
>>> ydata._is_transient_error(YFRateLimitError())
False
>>> 'retries' in inspect.signature(yf.download).parameters
False
>>> _CookieDBManager.get_location()
'/home/<user>/.cache/py-yfinance'
```

Backoff matches the library's own convention (`2 ** attempt` seconds). A cached
crumb whose text is literally "Too Many Requests" keeps a client throttled
even after waiting (yfinance#2441, fixed upstream in #2491), so a retry also
deletes `cookies.db` — via the library's own manager, after `close_db()`,
because SQLite will not delete an open file.

### 17.3 The lesson that generalises

Three of the five wrong assumptions were about **where something lives** rather
than what it does: which namespace, which file, which parameter. None of them
are inferable from the API's shape, and all of them are one introspection call
away.

This is the same failure mode as sections 14-16: a constant or a convention
assumed rather than read. The general rule, now in the `derive-dont-iterate`
skill: **before writing integration code against a library, read its source or
docs for the specific facts you are about to depend on — especially names,
paths, and parameter availability.** `dir()`, `inspect.signature()` and
`inspect.getsource()` are not optional steps; they are the derivation.

A test pins the assumption that made the wrapper necessary
(`test_library_will_not_retry_rate_limits_for_us`), so a future yfinance
release that adds internal retry announces itself instead of being silently
duplicated.
