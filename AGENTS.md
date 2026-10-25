# AGENTS.md: the contract for driving this repository

---

## 1. What this repository is

**A research lab for hedging strategies that rejects almost everything.**

An LLM drafts typed candidate hedge strategies. Each candidate is
walk-forward tested with purge and embargo, then put through a
**fail-closed promotion gate** with ten checks: Deflated Sharpe,
Probability of Backtest Overfitting, Newey-West HAC t-statistics,
walk-forward decay, regime stability, significance, and a risk-matched
benchmark. **Only a candidate that clears every check reaches the
decision layer.** The rest are rejected and the rejection is published.

The thesis: **the useful hedge is usually zero, and a hedge engine that
has never returned zero has not been tested.**

### The honest headline result

**The gate currently rejects every universe.** All three -- `us_equities`,
`mixed`, `crypto` -- are REJECTED, with 4, 5 and 5 failed checks
respectively. This is the point, not a failure of the harness. A tool
whose headline is "we ran it and it said no" cannot be mistaken for the
thing it warns about.

### What this repository is NOT

- **Not a trading signal.** It never says what to hold. It produces the
  *method* and the *findings*. A reader's risk allocation is not known
  here, so no position is responsibly derivable.
- **Not live.** Research on historical bars. No loop, no broker, no orders.
- **Not finished.** It is a well-audited **starting point**: the seams
  below are named so the next person can extend it.

---

## 2. How to run it

```bash
pip install -e ".[dev]"
```

That is the whole setup. **No API key. No account. No network.** The
default path reads the committed price fixtures.

| Job | Command | Writes | Exit |
|---|---|---|---|
| The documented run | `python scripts/run_research.py --all` | `outputs/sweep_<universe>_<timeframe>.json` | **1** -- the gate rejected |
| One universe | `python scripts/run_research.py --universe crypto` | `outputs/sweep_crypto_swing.json` | **1** |
| Machine-readable | `python scripts/run_research.py --all --json` | same, plus JSON on **stdout** | **1** |
| Regenerate the cards | `python scripts/make_cards.py` | `cards/card_<universe>_<timeframe>.md` | 0 |
| Check the cards | `python scripts/make_cards.py --check` | nothing | 0 if current, 1 if stale |
| Audit the claims | `python scripts/audit_claims.py` | nothing | 0 if clean, 1 if not |
| The test suite | `python -m pytest tests/ -q` | nothing | 0 if green |
| A forward simulation | `python scripts/run_agent.py --universe crypto` | `outputs/agent_<universe>.json` | 0 |

### Exit codes

**0** = ran and the gate accepted. **1** = the gate **rejected**, which
is a RESULT and not an error (every universe currently returns 1).
**2** = usage error.

### The `--json` contract

**stdout is the JSON payload and nothing else. Every human-readable
message goes to stderr.** So:

```bash
python scripts/run_research.py --universe crypto --json 2>/dev/null | jq .gate.verdict
```

returns `REJECTED`, with no banner to strip. This is enforced by
`tests/test_json_output_contract.py`.

### Optional, and never required

| Extra | Adds | Without it |
|---|---|---|
| `market-data` | live prices | the offline default, which is what the published numbers use |
| `llm` | LLM candidate generators | `template_candidates()`, the deterministic default |
| `plots` | figures | none of the statistics |

**A missing extra says exactly what to do**, in the error message.
**A missing API key is not an error at all** -- the keyless path is the
documented path.

---

## 3. What the output files contain

| Path | What it is |
|---|---|
| `outputs/sweep_<universe>_<timeframe>.json` | the evidence. One sweep per universe/timeframe. |
| `cards/card_<universe>_<timeframe>.md` | **generated** from the sweep JSON. Never hand-written. |
| `hedging_agent/fixtures/*.csv` | the committed price data. The published numbers come from these. |
| `docs/LITERATURE.md` | every threshold and its primary source |
| `notes/KNOWN-TRAPS.md` | the defects that were in this code, and how each was caught |

### The sweep JSON, top level

`universe`, `timeframe`, `n_trials`, `best_spec`, `best_metrics`, `dsr`,
`pbo`, `walk_forward`, `regime_stats`, `gate`, `action_ratio`,
`veto_ratio`, `zero_hedge_ratio`, `hedge`, `benchmarks`, `significance`.

The one that matters most is `gate`. Read it verbatim; do not recompute
it.

### `gate` -- the authoritative verdict

```json
"gate": {
  "passed": false,
  "verdict": "REJECTED",
  "checks": [ ... 10 entries ... ],
  "failed_checks": ["max_pbo", "max_is_oos_decay", "min_regimes_positive", "significance_t"]
}
```

`verdict` is **uppercase**: `PROMOTED` or `REJECTED`. Read
`gate.passed`, `gate.verdict` and `gate.failed_checks` **verbatim**. Do
not recompute the verdict from the individual checks: the gate is
fail-closed, and the checks array carries the threshold that decided
each one.

### The card

Its frontmatter is generated from the sweep JSON:

```
name, universe, timeframe, n_trials, gate, failed_checks
```

`gate` is `result["gate"]["verdict"]` verbatim; `failed_checks` is
`result["gate"]["failed_checks"]` verbatim. The body has ten sections in
a fixed order: the idea, the failure mode, the conditions, where it came
from, what would have falsified it, what we measured, where it works,
where it does not, the verdict, and what we got wrong.

---

## 4. What you may and may not do

### The rule that matters most

**The decision layer may veto and may scale the quantitative read. It may
NOT invent a hedge where the quantitative read says zero**, unless it is
genuinely confident. An LLM that can raise a hedge from nothing is an
LLM that can sell you an overlay.

### Never

- **Never present any output as advice to hold.** It is a method and a
  set of findings. No exception.
- **Never weaken a gate to make it pass.** If a check fails, it fails.
- **Never report a number that is not in an artefact.** Every number
  comes from a sweep JSON, and `scripts/audit_claims.py` checks every
  documented number against one.
- **Never claim a result the repository does not contain.** The gate
  rejects everything. That is the honest headline.
- **Never add personal attribution** to commits or files.

### Always

- **`hedge = 0` is a first-class output.** If the best hedge is zero,
  say zero.
- **Fail closed.** A missing input is an error, never a default that
  lets the run continue with a plausible-looking wrong answer.
- **Keep the published artefacts reproducible.** They are regenerated
  from the committed fixtures and compared byte for byte.
  `tests/test_tracked_bytes.py` compares by CONTENT, because `git
  status` cannot see a line-ending difference in a normalised file.

### The seams, named on purpose

This is a **starting point**. These are the places to extend it:

| Seam | Where | What it is |
|---|---|---|
| `template_candidates()` | `hedging_agent/strategies.py` | the deterministic candidate generator. Add a strategy family here. |
| `DecisionBackend` | `hedging_agent/decision.py` | the Protocol for the decision layer. Implement your own -- see `docs/DECISION-LAYER.md`. |
| `UNIVERSES` | `hedging_agent/data.py` | the universe registry. Add an asset set here. |
| `GateConfig` | `hedging_agent/config.py` | the gate's thresholds. **Change these only if you are re-running the research, and say so.** |

The decision layer is an **adapter, not a model**. This copy ships the
Protocol, a keyless `LocalBackend` (marked `uncalibrated=True`), and a
guide for building your own. The local fallback is **not calibrated**:
any threshold taken from it must be marked as such.

### If you change X, update Y

| You changed | You must also |
|---|---|
| a threshold, a strategy, or the decision layer | re-run the sweep, regenerate the cards, and let `audit_claims.py` check the numbers |
| the universe registry | re-capture the fixture, and regenerate every sweep |
| a published artefact | keep it byte-identical to what the code produces |

If you are unsure whether a number is still true, run
`python scripts/audit_claims.py`. It fails loudly, and that is the point.

---

## 5. Where to read more

- `README.md` -- the thesis, the honest result, and the commands.
- `docs/DECISION-LAYER.md` -- the Protocol verbatim, the hard rule, and
  a worked example that is a real module
  (`docs/table_backend_example.py`), checked against the guide by a test.
- `docs/LITERATURE.md` -- every threshold with its primary source.
- `notes/KNOWN-TRAPS.md` -- the defect ledger. **Read it before you
  trust a number.**
- `cards/` -- the human-readable view of each sweep.
- The sibling desk, for the execution side:
  [`portfolio-hedging-desk`](https://github.com/IkkeOmar/portfolio-hedging-desk).
