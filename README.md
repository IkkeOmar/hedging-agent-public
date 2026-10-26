# hedging-agent

**The useful hedge is usually zero. An agent that cannot return zero has not been tested.**

A research harness for dynamic hedging, built on one refusal: that a hedging
system which always holds protection is finished. Constant hedging caps upside
and taxes every day it is carried, so this project treats `hedge = 0` as a
first-class output and spends most of its effort proving that the gate which
emits zero is calibrated.

An LLM proposes hedge strategies. They are not trusted: each one is
backtested, deflated for the number of variants tried, checked for backtest
overfitting, and compared against controls that carry the same risk. Only what
survives is handed to a decision model that may veto, scale, escalate, or
abstain.

Current status: **the gate rejects.** See [Results](#results).

---

## Why this exists

The failure mode being built against is specific, not hypothetical. A decision
model given random or unvalidated inputs will still produce confident-looking
decisions, and those decisions look identical to good ones at the interface.
The only defence is upstream: make the inputs earn their way to production.

So the pipeline is deliberately arranged as a funnel, and each stage can only
narrow it:

```
LLM proposes a typed strategy spec
        |
        v
walk-forward + purge/embargo evaluation
        |
        v
Deflated Sharpe Ratio   (Bailey & Lopez de Prado 2014)
Probability of Backtest Overfitting  (Bailey & Lopez de Prado 2014)
Newey-West HAC t > 3.0  (Harvey/Liu/Zhu 2016)
risk-matched benchmark  (Maguire 2026)
        |
        v  only survivors
The decision layer adjudicates live state: veto / scale / escalate / abstain
```

Everything above the decision layer is deliberately boring, inspectable, and
free of model calls. The LLM is confined to generating *specifications*, never
to choosing parameters, never to sizing a position, and never to deciding
whether its own idea worked.

## What "hedging agent" means here

Hedge sizing responds to three independent signals, and the zero state is
reachable from any of them:

- **Realised volatility** against a target. A volatility-managed overlay
  reduces the variance tax (`mu - log CAGR ≈ sigma^2/2`), which is the one
  channel that pays without requiring a forecast.
- **Drawdown state.** A governor ramps protection only after the portfolio is
  actually hurt, rather than hedging in anticipation of loss.
- **Multi-scale regime consensus.** Trend and volatility detectors vote across
  several lookbacks at once. Disagreement is itself a signal, and disagreement
  tends to mean a smaller hedge, not a larger one.

The regime detector is fractal by construction: intraday, swing, and position
horizons run the same detector stack at different scales, with no privileged
timeframe.

## Results

Live US-equities / swing sweep, 6 candidates x 8 trials, 5y. The gate is
**fail-closed** and it says no:

All figures below are **net of transaction costs** (3bps round trip + 10bps
annual borrow). Every Sharpe reported before the cost fix was gross. The
cost model was configured and printed but never applied. See
`docs/LITERATURE.md` section 14.

<!-- BEGIN GENERATED: gate-table -->
| check | value | threshold | verdict |
|---|---|---|---|
| min Sharpe | 1.276 | >= 0.60 | PASS |
| min DSR | 1 | >= 0.90 | PASS (see below; do not trust it) |
| max PBO | 0.9857 | <= 0.30 | **FAIL** |
| positive fold ratio | 0.875 | >= 0.60 | PASS |
| IS->OOS Sharpe decay | 0.5284 | <= 0.50 | **FAIL** |
| nonzero action ratio | 0.12727 | >= 0.02 | PASS |
| regimes positive | 1 | >= 2 | **FAIL** |
| beats risk-matched benchmark | 0.4584 | > 0 | PASS |
| Newey-West HAC t | 2.7488 | > 3.00 | **FAIL** |
<!-- END GENERATED: gate-table -->

Four failures. `regimes positive` was previously reported as 2/PASS because a
3-bar `trend_up` regime scoring Sharpe 7.57 counted as a regime where the
strategy worked. It is now suppressed: a Sharpe needs >= 63 bars (92 for
crypto) before its sign means anything, and the check correctly fails. All
three universes fail it; crypto has no qualifying regime at all. See
`docs/LITERATURE.md` section 15.

The remaining failures carry literature thresholds behind them. Harvey, Liu & Zhu (2016, RFS) set t > 3.0 for a factor
that was *selected*; at 2.75 the hedged-versus-unhedged mean-return difference
does not clear it, even though the Sharpe gap of 0.5125 survives a HAC
correction. PBO of 0.9857 over 70 splits x 48 configs is the second.

**The DSR column is not carrying its weight.** Measured on these same
candidates, mean pairwise correlation is 0.9536 and the participation-ratio
effective trial count K_eff is 1.09. The 48 configs are statistically one
strategy. Raw K overstates the search 38x, and deflating against K_eff does
not fix it, because the measured `trials_variance` shrinks in compensation and
DSR reads 1.0 either way. See `docs/LITERATURE.md` section 12. Until the
candidate set is genuinely diverse, treat that row as uninformative rather
than reassuring.

Other universes, same settings. The **window** differs, and it is in the
table because a Sharpe without the window it was measured in is not a
result: the crypto provider caps at 1000 daily bars, so a 5y request
covers 2.74 years there and five years elsewhere.

<!-- BEGIN GENERATED: universe-table -->
| universe | window | Sharpe hedged / unhedged | PBO | t | verdict |
|---|---|---|---|---|---|
| us_equities | 2021-10-25 to 2026-10-23 (4.99y) | 1.276 / 0.7635 | 0.9857 | 2.7488 | REJECTED (4 failed) |
| mixed | 2021-10-24 to 2026-10-24 (5.0y) | 0.3838 / 0.4136 | 0.9571 | -1.021 | REJECTED (5 failed) |
| crypto | 2024-01-29 to 2026-10-24 (2.74y) | 0.2124 / 0.0841 | 0.2857 | 0.4692 | REJECTED (5 failed) |
<!-- END GENERATED: universe-table -->

Crypto is the interesting one, and not for the reason I first assumed. I
expected its unsaturated PBO (0.2857 vs 0.9857) to mean its candidates were more
diverse. Measured, the opposite: crypto is the **most** correlated of the three
(mean pairwise 0.9932 vs 0.9536 for equities, K_eff 1.01). PBO is measuring
rank-transfer, not diversity.

The split is in the median out-of-sample win probability: crypto **0.7347**,
equities **0.102**. Crypto's in-sample winner usually stays above the OOS
median, so its selection transfers. On the equity runs the in-sample winner
lands in the OOS *bottom half* about 90% of the time, a systematically
inverted selection, which is worse than no signal. It still REJECTs, and now fails five
checks rather than three: costs push it below min_sharpe and
positive_fold_ratio. See `docs/LITERATURE.md` section 13.

Two findings worth stating plainly, because they are unflattering:

1. **A 200-day moving-average rule on the hedge leg beat the entire regime
   system** on Sharpe. That is the pattern Cederburg et al. (2020) and Maguire
   describe, and it is the reason the benchmark table is mandatory rather than
   optional.
2. **A bug in our own PBO was flattering every candidate.** It scaled the
   out-of-sample rank by `(rank - 1) / N` instead of `rank / (N + 1)`, which
   flagged one rank too few on even N and biased PBO *downward* by exactly
   1/N -- 0.25 at N=4. Found by cross-validation against `ml4t/diagnostic`.
   Details and the pinned regression test are in `docs/LITERATURE.md` section 11.

## Layout

```
hedging_agent/
  config.py       timeframes, hedge policy knobs, gate thresholds
  regime.py       multi-scale fractal regime detector
  data.py         yfinance / ccxt providers, hedge-side inference
  strategies.py   typed strategy specs; LLM providers + deterministic fallback
  policy.py       dynamic hedge sizing, zero state, veto logic
  decision.py     the DecisionBackend protocol and the keyless LocalBackend
  stats.py        Sharpe/Sortino/drawdown, DSR, PBO, walk-forward, HAC
  benchmarks.py   risk-matched controls, volatility-tax decomposition
  pipeline.py     sweep -> gate -> JSON evidence
docs/
  LITERATURE.md   every threshold with its primary source, and what we got wrong
tests/
  test_engine.py           51 unit + integration tests
  test_benchmarks.py       42, validated against statsmodels and QuantStats
  test_gate.py             13 fail-closed promotion tests
  test_decision_layer.py   13, the decision-layer seam and its worked example
  test_json_output_contract.py  6, stdout is pure JSON and stderr is for you
  test_optional_dependency_messages.py  4, a missing extra says what to do
  test_data_robustness.py  6, a partial universe is never silently accepted
  test_crossvalidation.py  4, vs ml4t.diagnostic (needs Python 3.12+)
  test_tracked_bytes.py    the published artefacts, compared by content
  test_entry_points.py     the console scripts resolve to something
  test_path_punctuation.py 5, a sentence's full stop is not part of a path
  test_sweep_artefact_precision.py 13, the artefact stores no raw statistic
  test_claim_audit_sees_tables.py 6, the audit can see the numbers in a results table
  test_path_regex.py 5, a JSON key is not a file path
  test_agents_contract.py 5, the contract agrees with the README
  test_sweep_records_its_window.py 13, the artefact records its window
  test_readme_tables_are_generated.py 10, the README's tables are generated
  test_pbo_divisor_mechanism.py 20, the PBO divisor bug, pinned as arithmetic
  test_pbo_bias_table_is_reproducible.py 4, the published table matches the script
scripts/
  run_research.py   sweep CLI (--offline by default, --live to download)
  run_agent.py      forward simulation CLI
```

## Reproducing

```bash
pip install -e ".[dev]"

# the suite: fast, and it needs no network
python3 -m pytest tests/ -q

# cross-validation against ml4t/diagnostic (optional, needs that package)
python3 -m pytest tests/test_crossvalidation.py -q

# the research sweep -- reads the COMMITTED price fixtures, no network
python3 scripts/run_research.py --universe us_equities --timeframe swing

# every universe
python3 scripts/run_research.py --all --timeframes swing

# regenerate the research cards from the sweep results
python3 scripts/make_cards.py

# check every documented number against the artefacts
python3 scripts/audit_claims.py
```

**On Python 3.10, 3.11 and 3.12.** The install above works on all three.
The
cross-validation tests need `ml4t-diagnostic`, which requires Python 3.12
or newer, so on 3.10 and 3.11 they **skip** with a reason rather than fail. The
collected test count is the same either way, so the number this document
quotes does not depend on the interpreter. On 3.12 the cross-validation
tests run and pass: they check our Sharpe, Deflated Sharpe and PBO against
an independent implementation, which is how we found that our own PBO was
biased by scaling the rank by `(rank - 1) / N` instead of `rank / (N + 1)`.

If you want the cross-validation on 3.10 or 3.11, install
`ml4t-diagnostic` from a source distribution yourself. It is not on PyPI
for those versions.

**The default path needs no network, and no API key.** The prices are
committed under `hedging_agent/fixtures/`, and the sweeps are the ones
those prices produced. Two runs of the same command produce byte-identical
files.

That is not a convenience. A live download **cannot** reproduce the
published numbers: the crypto provider returns the most recent 1000 daily
bars, so the window ends at today's date and the same command produces a
different experiment tomorrow. Use `--live` to download instead -- it
needs the `market-data` extra, and the results will not match.

`--no-llm` (the default) uses deterministic template candidates, so a run
is reproducible without an API key.

## Statistics, and how they were checked

Every load-bearing statistic is validated against an independent
implementation, because these numbers decide what gets rejected:

| statistic | validated against | agreement |
|---|---|---|
| DSR | Bailey & Lopez de Prado printed example | `SR0 = 0.1132`, `DSR = 0.9004` exactly |
| PBO | `ml4t/diagnostic` `compute_pbo` | unbiased under noise to 0.494/0.515/0.503 vs true 0.5 |
| Newey-West HAC | `statsmodels` `OLS(cov_type="HAC")` | exact to 1e-9 across lag choices |
| Sharpe | `ml4t/diagnostic` `compute_sharpe` | exact to 1e-7 after annualisation |
| Volatility tax | `quantstats` cagr / avg_return / volatility | ~4e-7 |

`tests/test_crossvalidation.py` skips cleanly when `ml4t` is absent.

## Known limitations

Stated here rather than discovered later:

- **No trial ledger.** `n_trials` is the count from the current run. The
  `ml4t` guidance is to persist every variant ever evaluated across sessions;
  omitting failed variants understates the multiple-testing penalty. We are
  currently undercounting.
- **PBO resolution.** PBO is quantised to 1/n_combinations. Our own 8-split
  runs use 70 combinations, so they resolve 1/70. The coarser 0.25 step
  belongs to `ml4t.diagnostic`, whose API takes precomputed matrices and so
  gets 4 combinations; that is a limit on the *comparison*, not on us. A gate
  reading PBO > 0.5 as a hard fail needs the combinations to resolve near
  0.5 at all.
- **Intrabar path unknown.** Bar data cannot tell you the order of events
  within a bar, so intraday results are optimistic about execution.
- **Costs are modelled, not filled.** Commission and spread are charged as a
  fraction of notional; borrow, market impact, and financing are not.
- **No walk-forward live forward-test.** Everything reported is historical.

## Known errors in this repository

Found by an audit of our own published claims, and fixed here rather than
left for a reader to trip over. The full account of each is in
[`notes/KNOWN-TRAPS.md`](notes/KNOWN-TRAPS.md).

- **The PBO bug was described wrongly.** The published explanation said we
  divided the rank by `N` instead of `N+1`. The code used
  `(rank - 1) / N`, a zero-based rank. Every published NUMBER was right,
  so the number checks could not see it; only implementing the prose
  literally would have, and it would have shown no bias at all. Corrected
  in the README, `docs/LITERATURE.md` section 11, `notes/KNOWN-TRAPS.md`
  entry 1, the code comment and the research cards, and pinned by
  `tests/test_pbo_divisor_mechanism.py`. Trap 28.
- **The PBO resolution limitation named the wrong number.** It said an
  8-split run cannot resolve finer than 0.25. Our 8-split runs use 70
  combinations and resolve 1/70; the 0.25 step belongs to
  `ml4t.diagnostic`, whose API gets 4 combinations. Corrected above.
- **The declared Python floor was not checked by anything.** The project
  said `requires-python = ">=3.10"` and CI only ran 3.11 and 3.12, so
  the promise was unverified. A Python 3.12-only f-string (PEP 701 allows
  a backslash inside an f-string's expression part) compiles on 3.12 and
  is a SyntaxError on 3.10 and 3.11, and it kills the whole suite at
  collection rather than one test. `ast.parse(feature_version=...)` does
  NOT catch it, because CPython does not version its f-string handling.
  The tree is now compiled with a real 3.10 by
  `tests/test_supported_python_parses.py`, CI runs 3.10, and the floor is
  read from `pyproject.toml` so the test cannot drift from the promise.
  Exercising the floor then exposed a second instance: two tests imported
  `tomllib`, which is 3.11+, so the suite could not even collect on 3.10.
  They now use the `tomli` backport there, and `tomli` is a declared test
  dependency. Trap 29.
- **One published table was not reproducible.** The single-draw comparison
  in `docs/LITERATURE.md` section 11 had no seed recorded and the
  pre-fix estimator is not in this repository, so it could not be
  regenerated. It is now the average over 400 draws, which
  `python scripts/reproduce_pbo_bias.py` regenerates and
  `tests/test_pbo_bias_table_is_reproducible.py` pins.

Two things we did **not** change, so they are not fixed and are not
quietly presented as fixed:

- **The single-draw observation is gone, not reproduced.** We know what
  we saw, we do not know the seed that produced it, so we dropped the
  numbers rather than invent a seed. The average carries the same claim
  and can be checked.
- **The published sweep values are post-fix.** The PBO in
  `outputs/sweep_*.json` was computed by the fixed estimator. The
  historical one is only reachable through the reference copy in
  `scripts/reproduce_pbo_bias.py`.

## Where to take this next

This is a **starting point**, not a finished product. The harness is
finished; the research programme is not. The candidate bank is deliberately
narrow and deterministic, and the seams below are real places to change the
code:

| Seam | Where | What you would do |
|---|---|---|
| **Candidate generator** | `hedging_agent/strategies.py` -> `template_candidates()` | Write your own generator. The shipped one is a deterministic template bank so the default path needs no key and no network. |
| **Decision layer** | `hedging_agent/decision.py` -> `DecisionBackend` | Implement the protocol and swap the backend. The keyless `LocalBackend` is a fallback and is **not calibrated**. See `docs/DECISION-LAYER.md`. |
| **Universes** | `hedging_agent/data.py` -> `UNIVERSES` | Define your own universe: the assets, the provider, and the hedge instrument. |
| **Promotion thresholds** | `hedging_agent/config.py` -> `GateConfig` | Change what counts as a promotable strategy. Every threshold is a measured quantity, not a round number -- if you change one, say where yours came from. |

What you may **not** do with it:

- **Do not turn it into a signal generator.** The output is a research
  result, not a position. Do not present any number here as advice to
  hold anything.
- **Do not weaken a gate threshold to make something pass.** If a
  candidate fails, the honest response is to publish the failure and the
  reason.
- **Do not remove the failing checks from the report.** The gate's
  negative results are the credibility.
- **Do not report a number this repository cannot produce.** Every number
  in the documentation comes from a command here, and
  `scripts/audit_claims.py` enforces it.

The other half of this project is
[`portfolio-hedging-desk`](https://github.com/IkkeOmar/portfolio-hedging-desk):
the **risk side** -- is there anything to hedge, does the timing signal
exist, did the reduction pay for itself, does it hold up out of sample.

## Licence and prior art

MIT. See LICENSE.

Built by the [TextToQuant](https://texttoquant.com) team.

Prior art consulted and cross-validated against:

- [ml4t/diagnostic](https://github.com/ml4t/diagnostic) (MIT): DSR, PBO, FDR,
  HAC. The reference implementation for our multiple-testing statistics.
- [stefan-jansen/machine-learning-for-trading](https://github.com/stefan-jansen/machine-learning-for-trading) (MIT): the canonical ML-for-trading workflow.
- [Riskfolio-Lib](https://github.com/dcajasn/Riskfolio-Lib), [PyPortfolioOpt](https://github.com/pyportfolio/PyPortfolioOpt), [skfolio](https://github.com/skfolio/skfolio): portfolio construction and constraint layers.
- [arch](https://github.com/bashtage/arch): conditional volatility models.
- [purgedcv](https://github.com/eslazarev/purged-cross-validation): purged and combinatorial CV.

`docs/LITERATURE.md` carries the primary sources behind every threshold,
including the places where this implementation disagreed with them and was
wrong.