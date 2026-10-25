# Known traps

Every entry below is a defect that was in this code at some point, was
caught, and was fixed. They are here because **the mistakes are the
credibility**: a reader who knows which traps are real can judge the ones
that are still in the code, and a reader who does not know about them will
rediscover them the expensive way.

Each entry gives the mechanism, why the wrong answer looked plausible, and
how it was caught. Several of these produced numbers that looked entirely
reasonable, which is what made them dangerous.

---

## 1. The out-of-sample rank scaled by (rank - 1) / N instead of rank / (N + 1)

**The worst one, because it flattered every candidate.**

The Probability of Backtest Overfitting takes the out-of-sample rank of
the in-sample winner and scales it into (-0.5, 0.5]. Bailey & López de
Prado define that scale as

    theta = omega / (N + 1) - 0.5

with omega the **one-based** rank among N configurations. We wrote

    theta = (omega - 1) / N - 0.5

a **zero-based** rank over N. The verdict is `theta > 0`, and that difference
flags one rank too few on **even** N: N/2 − 1 instead of N/2. So PBO was
biased **downward** by exactly **1/N** -- **0.25 at N=4**, 0.17 at N=6, 0.10 at
N=10, 0.08 at N=12. On **odd** N the two forms agree and there is no bias.

Small N is exactly the case where the bias bites hardest, because 1/N is
largest there.

A PBO of 0.30 reads like "this strategy has a 30% chance of being
overfit", which is the sort of number a gate can be argued about. A PBO
of 0.05 reads like "this is fine". The bug made the second number
possible when the first was the truth.

**Why it was plausible:** nothing about the formula looks wrong. A rank
out of N configurations divided by N configurations is the obvious thing
to write. The bias is invisible in the output, because a *flattering*
number looks exactly like a good result.

**How it was caught:** by cross-checking against an **independent
implementation**. `ml4t/diagnostic` computes the same statistic; the two
disagreed, and the disagreement was systematic in the wrong direction and
largest at small N. Our own tests had blessed the number, which is the
part worth remembering: a test that only checks internal self-consistency
cannot catch a shared misconception.

**The rule:** a statistic that can REJECT a strategy must be checked
against a library we did not write. Cross-validation in the other
direction -- trusting our own tests -- is not cross-validation.

**The published sweep values are post-fix.**

---

## 2. A Sharpe ratio that was not a Sharpe ratio

An earlier version of our harness computed `CAGR / volatility` and
labelled the result a **Sharpe ratio**. It reported **0.596**. The true
Sharpe of the same return series -- `mean(excess) / std(excess)`,
annualised -- is **0.420**.

(This one was found in the *other* repository's predecessor, not here --
but it is recorded here because this repository computes a Sharpe too, and
the mistake is the most common one in retail backtesting. The ledger for
the desk carries the full entry.)

The two numbers differ by 42%, and both look plausible on a report.

**Why it was plausible:** `CAGR / volatility` is a common enough
performance ratio that it reads like a Sharpe to anyone who has not
written one. And it produces a *nicer* number.

**How it was caught:** by computing the Sharpe the definition's way and
comparing. The difference was far too large to be a rounding or an
annualisation convention.

**The rule:** a Sharpe is `mean(excess) / std(excess)`, annualised. Never
`CAGR / volatility`. If you report one, report the formula next to it.

---

## 3. The annualisation factor that was right for equities and wrong for crypto

The sweep reported a **crypto** Sharpe of **0.2177** while three other
numbers *in the same JSON file* -- the Deflated Sharpe's
`observed_sharpe`, the benchmark table's `candidate_sharpe`, and the
significance test's `sharpe_hedged` -- all said **0.262**.

The ratio is **1.203491**. The annualisation factor for crypto over
equities is `sqrt(365 / 252) = 1.203500`. The two agree to five decimal
places.

**The mechanism:** `pipeline.py` called `sharpe(returns)` for the
per-candidate *selection* step without passing `periods`. The default is
**252** -- the equity factor -- because that is the right number for
equities and the wrong number for crypto, which trades every calendar
day. The rest of the pipeline resolved the factor correctly, through
`annualisation(universe, timeframe)`, so the file ended up reporting two
different Sharpes for the same return series.

**Why it was plausible:** the error was a *missing argument*, not a wrong
one. The function's default was correct for two of the three universes,
so every equities run was right and the suite stayed green. And the
selection step was the one place that did not use the universe's factor,
so it was the one place that could differ.

**What it did and did not change.** The **winner** did not change: the
factor is a positive constant across all candidates in one universe, so it
cannot reorder them. The **gate verdict** did not change either (0.262 is
still below the 0.6 threshold). But the **published number** was wrong by
16.9%, and one artefact carried two contradictory Sharpes -- which is
worse than either number being wrong on its own, because a reader who
checks one field and finds a second field disagreeing has no way to tell
which one to believe.

**How it was caught:** by a subagent's module map, which flagged a
"consistency caveat" on the selection Sharpe. The flag was then verified
independently -- the ratio against `sqrt(365/252)`, the two orderings, the
gate threshold -- before anything was changed. A subagent's summary is a
claim; the ratio matching to five decimal places is the evidence.

**The rule:** when one artefact reports the same quantity twice, the two
numbers must agree, and the check for that is cheap. Resolve the
annualisation factor **once**, at the top of the function, and pass it
everywhere. A default argument that is right for one universe and wrong
for another is a defect waiting for a data source to change.

---

## 4. Transaction costs that were configured and never applied

The cost model (`commission_bps`, `spread_bps`, `borrow_bps_annual`) was
in the configuration and reported in the metrics, but **never applied to
the returns**. Every Sharpe in every sweep was gross of costs.

At 36x annualised turnover, that is not a rounding error.

**Why it was plausible:** the configuration was visible, the metrics
reported the cost numbers, and the report read as though costs were in
the model. Everything *looked* accounted for.

**How it was caught:** by reading the code that applies the cost, not the
code that configures it.

**The rule:** a knob that is configured and reported but not applied is
worse than a missing knob, because it makes the report look complete.
Trace a number from the configuration to the place it changes the
result.

---

## 5. A borrow cost charged every bar

The borrow rate is an **annual** rate, so it must be divided down to a
per-period charge before it is applied to each bar. The code divided the
bps figure by `periods` and then multiplied by the notional *fraction*
(~0.12), which silently double-counted: it charged 10bps/252 x 0.12
**every day**, i.e. a full annual 10% on the notional every bar -> 120% a
year.

**Why it was plausible:** the arithmetic looks like unit conversion, and
unit conversion is what it looks like. The error is in *which quantity*
the conversion applies to.

**How it was caught:** by writing down the intended annual charge and
comparing it to the realised one.

**The rule:** a rate with a period in its name ("annual") must be
converted exactly once, and the conversion must be checked against the
quantity it is applied to.

---

## 6. The entry cost that vanished because of a slice

`pct_change()` is defined on the *original* bar grid, so the return window
`.iloc[1:]` drops bar 0. A position's entry cost was charged at bar 0 and
then the bar-0 return was dropped -- so **every position was entered for
free**.

**Why it was plausible:** the code looked correct. The cost was charged,
and the returns were sliced. The two facts were in different places.

**How it was caught:** by writing down the closed form (the entry cost
must appear exactly once in the return series) and asserting it inside
the function that computes the returns.

**The rule:** when a series is sliced, ask what the slice drops. The
silent-zero defect is the same shape: `.iloc[1:]` looked like it removed
a `NaN`, and it removed a cost instead.

---

## 7. A partial download, silently accepted

Yahoo will happily return 4 of 11 tickers under rate limiting. The old
loop skipped the missing ones and carried on, so the run continued on a
**different portfolio** than the one the documentation describes. Every
statistic would describe a strategy nobody asked about.

**Why it was plausible:** the run produced plausible numbers, and nothing
in the output said "3 assets are missing".

**How it was caught:** by counting the tickers that came back against the
tickers that were asked for.

**The rule:** a partial universe is not a universe. Fail loudly rather
than research the wrong thing.

---

## 8. A rate limit that no amount of waiting would clear

A cached Yahoo crumb whose text is literally "Too Many Requests" keeps a
client throttled. Waiting alone does not clear it; deleting the cached
cookie database does. The retry logic now clears the crumb cache between
attempts, using the library's own manager to find it, because the cache
location is read from the installed library rather than guessed.

**Why it was plausible:** "it is rate limited, let me try again" is the
obvious response, and it does not work.

**The rule:** when a retry does not help, the state being retried against
is stale. Find it and clear it -- and read where it lives from the
installed library, not from memory.

---

## 9. A threshold set to a round number

The decision layer's action and veto thresholds are **0.70** and **0.30**.
They are not round by accident: they are where the probabilities were
calibrated against labelled data. A threshold picked for looking tidy is a
threshold with no evidence behind it, and it is the kind of thing that
survives review because it is unremarkable.

**The rule:** a threshold in a decision layer is a measured quantity. If
you cannot say where it came from, it is a guess wearing a number.

---

## 10. A chart that rendered empty

The dashboard's dial chart drew a **line** through a series that is
monthly, plotted on a daily axis. The line broke on every `NaN` gap, so
it drew nothing at all: an empty grid that looked like a rendering bug.

**Why it was plausible:** the chart code was correct, and the data was
correct. The two were just in different units.

**How it was caught:** by looking at the rendered chart. No assertion
failed, because nothing was wrong.

**The rule:** every chart gets a test that asserts it plotted **points**.
A chart that renders empty is not a failed test; it is an untested
failure.

---

## 11. Two functions with the same name, and the good one was dead

A module defined a robust test counter and then defined it **again**
later in the same file. The second definition won, silently, and the
first -- the careful one -- was dead code that still looked alive.

**Why it was plausible:** the two definitions were far apart in the file,
and the second one was shorter.

**How it was caught:** by listing every `def` in the module and looking
for duplicates. This is a script, not a review.

**The rule:** a duplicate definition is invisible to the type checker, to
the linter, and to the tests, because the tests exercise the surviving
one. Search for duplicates before trusting a function.

---

## 12. A verdict that said REJECTED and looked like a failure

A promotion gate that says no is a **result**, not an error. The first
version of the CLI reported a rejected gate as a failure, so a
successful run that correctly rejected everything looked like a broken
tool.

**Why it was plausible:** "the gate said no" and "the program failed" are
the same shape on a terminal.

**The rule:** exit codes that mean something. **0** = ran, **1** = the
gate rejected (this is a result), **2** = usage error. An agent driving
this repository has to be able to tell those apart.

---

## 13. A borrow charge annualised at the wrong rate

The same root cause as trap 3, in a second place, and it was found by a
different route: a structural map of the modules flagged it as a
"consistency caveat" alongside the Sharpe one.

`simulate_hedge` took `periods: int = 252` as a default and was called
without it. `periods` is the universe's bars-per-year, and the borrow
rate is an **annual** rate that must be divided by the bars in a year
before it is charged to each bar. So the crypto universe -- 365 days a
year -- was charged 365/252 = **1.45x** the correct borrow: a **45%
overstatement** of the financing cost on every crypto hedge.

**Why it was plausible:** the default was *correct* for equities. Two of
the three universes were fine, the suite stayed green, and the cost is a
second-order effect on a Sharpe rather than a crash.

**How it was caught:** not by a test, and not by reading the numbers.
A structural map of the module listed every call that omitted `periods`,
and the omission was checked against the formula that consumes it.

**The rule:** a default argument that is right for one case and wrong for
another is a defect waiting for a data source to change. Resolve the
value once, at the top, and pass it everywhere -- including into the
functions whose signatures nobody looks at.

---

## 14. A universe that silently lost half its assets

The `mixed` universe declares `BTC-USD`, `ETH-USD` and `SOL-USD`, and its
own note says crypto is forward-filled onto the equity calendar. It
contained **no crypto at all**. The captured price file had the columns
`Date, SPY, QQQ, IWM, EFA, TLT, GLD, SH` -- seven equity and ETF columns,
and not one crypto asset.

**The mechanism.** Two filters, each excluding the crypto tickers:

- the Yahoo branch selected assets where `"-" not in a`, so `BTC-USD` was
  dropped -- Yahoo *does* serve `BTC-USD`;
- the ccxt branch skipped assets where `"-" in a`, so the same tickers
  were dropped there too.

A comment in the second filter said `already handled as a yahoo ticker`,
which was the exact opposite of what the first filter did. The two
filters were consistent with each other and jointly removed every crypto
asset.

**Why it was plausible:** the "/" form (`BTC/USDT`) is the ccxt marker and
the "-" form (`BTC-USD`) is the Yahoo marker, so testing for the dash
looks like the natural discriminator. But the `crypto` universe already
uses the slash form and sets `provider="ccxt"`, so it never entered the
Yahoo branch at all -- the dash test was redundant *and* wrong.

**How it was caught:** by reading the captured price file's column names.
The sweep ran, produced a plausible result, and reported 6 of 7 columns
of the equity sleeve.

**The rule:** a filter that is right for one provider and wrong for the
other removes data without saying so. Test the provider's own marker
(`/` for ccxt), not a character that happens to appear in one ticker's
name. And when a universe declares what it contains, check that the
artefact contains it -- the declaration and the data are two different
claims, and only one of them was true.

---

## 15. A documented command that could not do what it said

The README documented
`python3 scripts/run_research.py --all --timeframes intraday swing position`.
There was no `--timeframes` flag: the script would have exited with an
unrecognised-argument error. And `--all` ran a hard-coded list of two
universes and two timeframes, so it never reached the `mixed` universe at
all -- the one documented way to run everything did not run everything.

Separately, the argparse defaults were `--candidates 5 --trials 12`,
while the script only writes the bare `sweep_<universe>_<timeframe>.json`
for `candidates == 6 and trials == 8 and seed == 11 and period == "5y"`.
So **the default invocation never wrote the canonical artefact**: it
wrote a suffixed one instead, and the file the documentation pointed at
was never produced by the documented command.

**Why it was plausible:** the flag was plausible, the hard-coded list was
plausible, and the defaults were plausible. All three were checked
against the code only by reading the README, which is the one document
nothing verifies.

**How it was caught:** by a structural map that compared the documented
command line against the actual argparse definition.

**The rule:** a documented command is a claim about the program, and it
belongs to the same class as a documented number. Check it against the
parser, and make the defaults BE the canonical configuration, so the
command a reader copies produces the artefact the documentation names.

---

## 16. A missing optional dependency that said nothing

The live price path imported its two optional providers with bare
statements:

```python
if u.provider in ("yahoo", "mixed"):
    import yfinance as yf
...
if u.provider in ("ccxt", "mixed"):
    import ccxt
```

A user who ran `--live` without the extra got:

```
[crypto/swing] FAILED: ModuleNotFoundError: No module named 'ccxt'
```

A traceback that ends in a module name and says nothing else. The
agent-usability contract asks for better than that: an error message must
say exactly what to do and where to get it, so a reader -- or an AI --
can set itself up from the error alone.

**Why it was plausible:** a bare import inside a function is the ordinary
way to make an optional dependency optional, and it is what keeps
`yfinance` out of the core dependency set. The failure mode is only
visible to someone who opts into the live path, which by design is
nobody on the documented path.

**How it was caught:** by a background run of the live path against a
tree where the extra was not installed. The crash was not reproducible
against the current code -- the offline default had landed in the
meantime -- but the error message it exposed was still the one a reader
would have seen.

**Fix:** both imports now raise a `RuntimeError` that names the missing
package, offers the offline path (which needs nothing beyond the core
dependencies and reproduces the published numbers exactly), and gives the
install command.

**And a wrong instruction is worse than no instruction.** The first
version of that message said `pip install -e ".[live]"`. The extra in
`pyproject.toml` is called **`market-data`**. The command I wrote did not
exist. It was caught by reading the file the message points at, before
shipping it -- and now a test reads `pyproject.toml` and fails if the
message ever names an extra that is not declared.

**The rule:** an optional dependency is optional for the install, not
for the error message. Say what is missing, say what works without it,
and make sure the command you print is one that exists -- checked by a
test, because a plausible command line is not evidence of a real one.

---

## 17. A `--json` flag that produced JSON you could not parse

```
$ python scripts/run_research.py --json > out.json
$ json.load(open("out.json"))
json.decoder.JSONDecodeError: Expecting value: line 1 column 2 (char 1)
```

The flag existed, it printed JSON, and the JSON was surrounded by a
banner, six `[sweep]` progress lines, a `[saved]` path and a
cross-universe summary -- all on **stdout**.

The cause was spread across three files, which is why reading the
script did not find it: the script's own prints, the package's progress
prints in `pipeline.py`, and the data layer's retry diagnostics in
`data.py`. A consumer had to strip banners off stdout before it could
parse the thing the flag existed to produce.

**Impact:** high, and silent. Nothing errored. The tool appeared to
support machine consumption, and the first program that tried to use it
would have failed.

**Why it was plausible:** the human output and the JSON payload went to
the same place because `print()` is the obvious thing to write. The
convention that fixes it -- stdout is the machine payload, stderr is for
the human -- is easy to agree with and easy not to implement.

**How it was caught:** by running the documented command and parsing its
output. Not by reading the code, and not by looking at the flag.

**Fix:** every human-facing line now goes to stderr, in the script, in
`pipeline.py` and in `data.py`. `print_report()` takes a stream argument
so the caller can send it to stderr in `--json` mode.

`tests/test_json_output_contract.py` holds the line: it runs the real
entry point, parses stdout with `json.loads` **without stripping
anything**, checks the gate payload's shape, checks the human output is
on stderr, and checks the exit code is 1 (the gate rejected, which is a
result). `scripts/run_agent.py` had the same defect and is fixed the
same way.

**And the mechanical fix made it worse first.** A regex that inserted
`file=out` before a closing paren put the argument *inside* an f-string
expression in six places, which is a syntax error, and truncated one
line. The AST is what caught it:

    print calls with no file= argument: 0

**The rule:** a flag that exists is not a flag that works. The only way
to know `--json` works is to run it and parse the result, and that has to
be a test rather than a manual check. This is the same shape as the
Streamlit `width=` bug in the desk: the API was called, the call was
accepted, and the feature was not there.

---

## 18. A fixture rewritten with CRLF, and a git that could not see it

A background capture script re-downloaded the `mixed` universe and wrote
the fixture with `pandas.DataFrame.to_csv()`. On Windows that writes
**CRLF**. The repository's `.gitattributes` says `* text=auto eol=lf`, so
git normalises the line endings on the way in -- and therefore:

    git status --short          -> clean
    git diff --stat             -> empty
    but the bytes on disk       -> differ from HEAD

The file looked unchanged to every git command while being a different
file. The `.gitattributes` that makes the published bytes platform-
independent is exactly what hid the change.

**Impact:** high, and silent. The committed sweeps were generated from
the LF fixtures. A CRLF fixture is the same DATA, so the numbers would
still have matched -- until they did not, and nobody could tell why,
because git said the file had not changed.

**How it was caught:** by comparing the file's sha256 against
`git show HEAD:<path>`, because a `sha256sum` of the working tree and a
`sha256sum` of the piped `git show` disagreed while `git status` said
they were identical. All three answers cannot be true, so one of the
commands was lying, and `git status` was the one lying.

**The fix, and the rule:** the working tree was restored from `git show`
byte for byte, and every fixture was checked the same way. **A file that
`.gitattributes` normalises must be verified by CONTENT, not by
`git status`.** The check that belongs in the repository is a byte
comparison of the tracked file against `git show HEAD:<path>`, because
that is the only comparison that sees past the normalisation.

This is also why the published sweeps are committed as BYTES and compared
by sha256: a number that a reader can regenerate and diff is a claim; a
number in a file that a tool says is unchanged is not.

---

## 19. A cross-validation that had never run

```python
ml4t_stats = pytest.importorskip(
    "ml4t.diagnostic.evaluation.stats",
    reason="ml4t-diagnostic not installed; run with the 3.12 interpreter",
)
```

At module level, at the top of the file.

**Impact:** high, and invisible. This file checks our Sharpe, DSR and PBO
against an independent implementation -- the checks that could have found
a wrong number in the statistics that REJECT a strategy. It contributed
**zero tests in every environment**, and the total test count was
identical (135) whether `ml4t` was installed or not, so nothing looked
wrong.

**Why it never ran.** `ml4t-diagnostic` requires Python >= 3.12. Every
environment here was 3.11, so `pip install ml4t-diagnostic` refused. The
module-level `importorskip` then skipped the WHOLE FILE -- including the
tests -- and the suite reported the same number either way.

This is GOAL.md section 1d item 20 exactly: *"a module-level
`importorskip` makes the test count depend on which optional extras are
installed"*. The count did not vary, which is what made it invisible:
the failure mode is not a varying count, it is a count that is
**missing four tests that should be there**.

**How it was caught:** by counting the tests PER FILE rather than in
total, and noticing that `tests/test_crossvalidation.py` contributed
nothing at all. Then by checking the actual requirement:
`pip install ml4t-diagnostic` on Python 3.11 returns "No matching
distribution found" and lists every version as requiring >= 3.12.

**Fix, and the check.** The import moved inside the fixtures, so the
module always collects and the tests skip individually when the extra is
absent -- which is what a skippable optional dependency should do. And
the CI matrix runs Python 3.12, where `ml4t` installs, so the
cross-validation runs on every push.

On Python 3.12 with `ml4t-diagnostic` installed, **all 4 tests pass**:
our Sharpe, DSR and PBO agree with the independent implementation. The
published numbers were right; the check that would have proved it had
never executed.

**The rule:** *a skip that hides a whole file is not a skip, it is a
disabled test that still counts.* Import the optional dependency inside
the test that needs it, so the file always collects, and run the suite
on an interpreter where the dependency can actually be installed.

## 20. A dependency that was optional in behaviour and mandatory in the packaging

Entry 19 moved the `ml4t` skip out of module level, so the four
cross-validation tests skip cleanly where the package is absent and run
where it is present. The CODE was fixed. The PACKAGING was not.

`pyproject.toml` declared:

    dev = [
      "pytest>=8.0",
      "ml4t-diagnostic>=0.1.9",      # requires-python >=3.12,<3.15
      ...

**Impact:** high, and it broke the install for a whole interpreter. The
CI matrix runs 3.11 and 3.12. On 3.12 the install succeeded. On 3.11 pip
could not resolve `ml4t-diagnostic>=0.1.9` at all, so
`pip install -e ".[dev]"` aborted -- before a single test ran:

    ERROR: Could not find a version that satisfies the requirement
    ml4t-diagnostic>=0.1.9; extra == "dev" (from hedging-agent[dev])
    ERROR: No matching distribution found for ml4t-diagnostic>=0.1.9

The 3.11 job failed at "Install the package and the test dependencies".
A reader on Python 3.11 who followed the README could not install the
project.

**Why it was plausible:** the package is a genuine test dependency, and
listing a test dependency in `dev` is the obvious thing to do. The
project's own README already said the cross-validation "needs Python
3.12+", and the tests already skipped cleanly without it. So the
package was OPTIONAL in behaviour and MANDATORY in the declaration, and
the declaration was the part nobody re-read. It survived because the
3.12 job passed: one green job in a two-job matrix reads as "CI is
green" if you only look at the newest one.

**How it was caught:** by reading the CI log rather than the run
summary. The summary said "pytest (Python 3.11): failure"; the log said
the install step could not find a distribution. Reading only the summary
would have sent me looking for a broken test.

**Fix.** An environment marker, which says what is true -- the
dependency exists only where it can exist:

    "ml4t-diagnostic>=0.1.9; python_version >= '3.12' and python_version < '3.15'",

**Proved by breaking it, in both directions:**
- the install now succeeds on 3.11, and `ml4t-diagnostic` is correctly
  absent from the resolved set;
- the suite collects **the same number of tests on 3.11 and 3.12**, so
  the count a document may quote does not depend on the interpreter.
  (The exact number is not written here: a ledger that states one goes
  stale the next time a test is added, and the point of the entry is
  that the count is interpreter-INDEPENDENT, not what it currently is.
  `scripts/audit_claims.py` checks any count a document does state
  against what pytest actually collects, on every CI run.);
- on 3.11 the four cross-validation tests SKIP with the reason
  "ml4t-diagnostic not installed; needs Python >= 3.12";
- on 3.12 the same four tests **PASS** -- 4 passed -- which is the whole
  reason the package is there;
- the README now states the 3.12 requirement before the reader installs,
  because a constraint that is only discoverable by hitting it is not
  documentation.

**The rule:** *an optional dependency that is declared unconditionally is
not optional -- it is a hard floor wearing a soft hat.* A dependency
whose availability depends on the interpreter must say so in the
metadata, not only in prose. And when a matrix has more than one entry,
a green entry is not evidence about the others.

## 21. A checker that read the sentence's full stop as part of the path

`docs/DECISION-LAYER.md` contains a Python comment, inside a code
block:

    # Keys are the regimes this repository uses; see hedging_agent/regime.py.

That is a correct English sentence, and `hedging_agent/regime.py` is a
real file. The claim audit's path regex captured the trailing full stop
as part of the filename, because its character class was
`[A-Za-z0-9_./-]+` -- which contains a full stop.

**Impact:** low in substance, high in consequence. The audit reported

    FAIL  docs/DECISION-LAYER.md: references missing path
    'hedging_agent/regime.py.'

A correct document failed a correct check, and the CI job that runs the
audit went red on a sentence that ends in a full stop.

**Why it was plausible:** the regex is right about most paths. It
correctly finds a path in backticks, and correctly finds one at the start
of a line or after a space or an open bracket. The full stop only
becomes part of the filename when a path appears in running prose
without backticks -- which is exactly the case where the sentence
around it is most likely to END with one. The greedy class was doing
its job: it matched the longest possible path, and the longest possible
path included the punctuation.

**How it was caught:** by reading the CI log rather than the summary.
The summary said "Audit the documented claims: failure". The log named
the exact string it had captured -- `'hedging_agent/regime.py.'` -- with
the full stop visible. A summary that said "path check failed" would
have sent me looking for a deleted file.

**The fix is in the CHECKER, not the prose.** The document was right.
Deleting the full stop would have made the check pass by making the
sentence worse, and the next sentence ending in a full stop would have
failed the same way. The last character class may no longer include a
full stop:

    r"|(?:^|[\s(])((?:hedging_agent|scripts|...)/[A-Za-z0-9_./-]*[A-Za-z0-9_])"

The path must end in an alphanumeric or an underscore, which is what a
filename ends in.

**Proved by breaking it, in both directions.** A genuinely missing path
(`hedging_agent/definitely_not_here.py`) is still caught, with the same
message and a non-zero exit. The correct sentence is not. Restoring the
document returns the audit to exit 0.

**The rule:** *a checker that is wrong about punctuation will blame the
document.* When a check reports a path that looks right, look at the
character it ends with before you look for the file. And when a checker
fails on something that reads correctly, the checker is the more likely
of the two to be wrong -- because the document was written by someone
reading the sentence, and the checker was written by someone reading a
character class.

## 22. A statistic stored at sixteen digits, and a file that would not compare

The committed `us_equities` sweep stored

    "t_stat": 2.7487817806516186

A fresh run of the SAME commit produced

    "t_stat": 2.7487817806516133

The difference is 5e-15. It changes no published claim, and every number
the repository quotes is at four decimals, so no reader would ever notice
it. But it made a byte comparison of the committed artefact fail, and
"the published number can be reproduced" is the entire point of shipping
the artefact.

**Impact:** low in substance, high in consequence. It was the one place
in the repository where the file's bytes depended on something outside
the repository.

**Why it was plausible:** the sweep artefact is assembled from many
blocks, and most of them round explicitly -- `round(x, 4)` on the
metrics, `round(x, 3)` on the turnover, `round(x, 6)` on the regime
means. The significance block was the exception: `newey_west_tstat`
returned raw floats and `sharpe_difference_tstat` stored them as they
came. The SAME function already rounded its three siblings
(`sharpe_hedged`, `sharpe_unhedged`, `sharpe_gap`) to four decimals. One
block, one quantity, two precisions.

**How it was caught:** by regenerating the sweeps from the committed
code and comparing, rather than by reading the code. The hypothesis list
was long and every one of them was wrong:

- `PYTHONHASHSEED` order dependence -- disproved: two runs with seeds 1
  and 999 produced identical bytes;
- BLAS/OpenMP threading -- disproved: `np.dot` gave the same answer with
  `OMP_NUM_THREADS=1` as with the default;
- memory layout and pairwise summation -- disproved: a contiguous copy
  and a strided view of the same values agreed;
- a later commit changing the estimator -- disproved: regenerating at
  the very commit that published the sweeps still produced a different
  number;
- the estimator being impure -- disproved: recomputing the statistic
  from captured inputs in a fresh process reproduced the fresh value
  bit for bit.

The estimator was a pure function of its inputs. The inputs differed in
their last bits, because the platform summed the return series in a
different order.

**The fix, and why it is at the artefact boundary and not in the
estimator.** The estimator must stay at full precision: a statistic
should be computed as precisely as the arithmetic allows and rounded
once, where it is published. So the rounding moved to the pipeline, at
the point where the artefact is assembled, next to the rounding every
other block already does.

Rounding to DECIMAL places would have been wrong. The three fields span
six orders of magnitude: `mean_diff` is ~1e-4 and `se` is ~7e-5, so
`round(x, 4)` prints both as `0.0` and destroys the standard error the
gate reads. They are rounded to significant digits instead, and the
t-statistic to four decimals -- which is exactly what the promotion gate
already publishes and exactly what the research card prints. The file and
the gate now agree about the same number, and `se` keeps its meaning.

**What did not change:** every published claim. Sharpe, PBO, the
zero-hedge share, the trial count, the verdict and the failed-check
counts are byte-identical before and after. The research cards are
byte-identical. Only three raw floats per sweep moved, and they moved
to a rounded form.

**The lesson:** a file that stores one quantity at two precisions will
eventually disagree with itself. The disagreement is invisible until
someone tries to reproduce the file, and by then it looks like
nondeterminism -- which sends you looking for a race condition that is
not there.

## 23. A file that shipped the vendor it was supposed to leave behind

This repository ships a **seam**, not a service. The decision layer is a
`DecisionBackend` protocol with a keyless `LocalBackend` behind it, and
the point is that a reader can implement the protocol and swap the
backend. That promise holds only if no module in the tree quietly talks
to somebody else's API.

One did. It read `JEV_API_KEY`, `TYPESAFE_AI_API_KEY` and
`OPENROUTER_API_KEY`, posted to a hosted decision endpoint, and
registered a vendor backend on import. The file's own docstring said it
was not part of this copy. The README's module map listed it anyway.

**Impact:** the documented path was never affected -- the package's
`__init__.py` does not import it, so `pip install` and every command in
the README ran with no key, no account and no network. But the tree a
reader receives contained a module that does all three things the
repository promises it does not do, and a reader auditing the tree would
have found it and concluded the README was lying.

**Why it was plausible:** the check that ran afterwards asked whether
the published bytes contained personal strings. It did not ask whether
they contained the vendor. A privacy scanner cannot ask that question:
a hosted adapter with an environment-variable key is not a privacy leak.
It is a contract violation, and the contract is not a pattern list.

**The fix.** The adapter is not in this tree, the README's module map
names `decision.py` instead, and nothing in the package imports it.

**The lesson:** a scanner answers the question you asked it. It does not
answer the question you meant. Check the published tree against the
CONTRACT -- no key, no network, no vendor, no internal process record --
and not only against the patterns that happen to be in the scan.

## 24. A check that could not see the numbers a reader actually reads

The claim audit screens every numeric claim in the reader-facing
documents against the artefacts the repository produces. It passed. The
README's results table was wrong anyway.

**What the audit could see.** The prose pattern needs a unit word after
the number -- "10 checks", "48 trials" -- so it matches a number in a
sentence. Every results table in this repository is written the other
way: `| Newey-West HAC t | 2.7150 | > 3.00 | **FAIL** |`. A table cell
is followed by a pipe, not by a unit word, so **no table cell was ever
screened**. The one kind of claim the audit exists to catch -- a number
that has drifted from the artefact it describes -- was invisible to it,
in the one place the numbers live.

**What was actually wrong.** The README quoted the Sharpe, the PBO, the
HAC t-statistic, the decay and the failed-check counts of a superseded
sweep generation: 1.2751, 0.5416, 2.7150, and the mixed universe at a
Sharpe of 1.1283 when the committed artefact said 0.3838. The artefacts
were right; the prose was describing a run that no longer existed.

**Two more holes in the same check, found while fixing it.** The
comparison for a table cell used a one-decimal fallback, so a stale
1.2751 matched a current 1.276 (both "1.28") -- the loose form defeated
the strict one. And the row parser was fixed at four columns, so the
five-column universe table was skipped entirely.

**The fix is structural, not another check.** The tables are now
**generated** from the sweep artefacts by `scripts/make_readme_tables.py`,
the way the research cards already are, between two marker comments in
the README. A hand-written table asks "did the writer remember to
update it?" every time; a generated one asks the question once, at build
time. The audit was also taught to read table cells, so a number that
drifts into a table that is NOT generated is still caught.

**The lesson:** a checker has a blind spot, and the blind spot is usually
where the real content lives. Prose is easy to pattern-match; a table is
not. Check the shape your content actually has, and when the answer is
"generate it, do not check it", generate it.

## 25. A negative case that tested the wrong property

The D19 check compares the extension points the README names with the
ones AGENTS.md names. Its first negative case renamed a seam's LABEL --
"The benchmark set" became "Something else" -- kept its path, and the
audit passed.

**The check was right; the test was wrong.** The check compares PATHS,
because a path is what a caller actually edits. A label is prose, and
prose is free. Renaming a label changes nothing an agent would follow,
so two documents that differ only in a label do not disagree about where
the seams are.

**Why the wrong test is so easy to write.** "Does the check notice when
the two documents disagree?" sounds like one property, and the obvious
way to make them disagree is to change a word. But the property the
check claims is narrower: *the two documents must name the same
extension points*. The words in the table's first column are not
extension points; the paths in the second column are. A test that
changes the wrong column passes for the wrong reason, which is the
same trap as the desk's dead-code guard: the test exercised the guard
in front of the code it claimed to test.

**The test that actually proves it** removes a PATH from the AGENTS.md
seam table, and the check reports the missing seam by name. The other
half -- removing the "may not" list -- is proved the same way, by
renaming the `### Never` heading so the marker is absent.

**The lesson:** when a negative case passes, the first question is not
"why does the check not fire?" but "what property did I just try to
break, and is that the property the check claims?". A check that fires
on the wrong input and sleeps on the right one is worse than no check,
because it has been marked as proved.

## 26. A check that compared text where the artefact was bytes

The README generator's `--check` reported "the README's generated tables
are current" for a README that was full of CRLF line endings. The
committed README had to be LF, or a reader who regenerates it on a
different machine sees "the whole file changed".

**Why it could not see it.** The generator read the file with
`Path.read_text()`. That applies universal newlines: a CRLF file reads
as LF TEXT. So the comparison was between two texts that agreed, while
the thing being shipped -- the file's BYTES -- disagreed with what the
generator would write. The check was looking at the wrong representation
of its own subject.

**The general shape.** `read_text()` and `write_text()` translate
between bytes and text, and the translation is invisible at the call
site. A checker that reads a published file and compares it to a string
is comparing text, and it will happily pass a file whose bytes are
wrong. Read the bytes, compare the bytes, and write the bytes.

**Three bugs in one, and only the third was visible.** The first two
were in my own test harness: converting LF to CRLF with a naive
`bytes.replace(b"\n", b"\r\n")` on a file that already had CRLF
produced `\r\r\n`, and `read_text()` turned that into `\n\n` -- so
the blank lines doubled and the file grew from 315 lines to 617. I read
that as a generator bug. It was not: the round trip
`read_text()` -> `write_text(newline="\n")` is clean, and I had proved
that with a three-line probe before changing any code.

**The lesson:** when a measurement disagrees with the code, suspect the
measurement first. The doubling was in my harness; the CRLF blindness
was in the checker. Two different defects, and only one of them was
mine to fix.

## 27. A window that was requested but never delivered

The sweep artefact recorded the universe, the timeframe and the trial
count, and not the WINDOW. A Sharpe without the window it was measured
in is not a result, it is a number.

The gap was not visible in the numbers. It was visible in the file
names: `crypto_1d_5y.csv` holds **1000 daily bars**, which is 2.74
years, while `mixed_1d_5y.csv` holds 1827 bars and `us_equities_1d_5y.csv`
holds 1255 -- five years each. The crypto provider caps at 1000 bars, so
a 5y request silently returned less, and the artefact did not say so.
A reader comparing the three Sharpes would have compared a five-year
equity result with a 2.74-year crypto one.

**The fix.** The sweep now records the window it was measured over, read
from the data rather than from the request: `start`, `end`, `n_bars` and
`years`, plus a note when the window is materially shorter than the
period that was asked for. The research cards and the README's universe
table carry it too, both generated, so it cannot go stale.

**No published number changed.** The `window` block is the only
difference between the old and the new artefacts; the Sharpe, the PBO,
the zero-hedge share, the verdict and the failed-check counts are
byte-identical.

**The lesson:** a request is not a delivery. The period argument says
what was ASKED for; the artefact has to say what was GOT. A parameter
that is silently clamped by a provider is a claim the reader cannot
check, and it is exactly the kind of thing a fail-closed repository is
supposed to be loud about.

## 28. A mechanism described in prose that the code did not have

The published description of the PBO bug said we "divided the rank by N
instead of N+1", and that the buggy form counted "(N-1)/2 ranks instead of
N/2". The code said something else:

    theta = (rank_of_best - 1.0) / n_cfg_f - 0.5

a **zero-based** rank over N, not a one-based rank over N. The prose was
wrong twice: about the numerator and about the count.

**Why the wrong prose survived so long.** The NUMBERS were all correct.
The bias really was downward, it really was 0.25 at N=4, and the before
and after tables really did reproduce to the digit. A reader checking the
numbers found nothing wrong. Only a reader who implemented the published
description literally would have found the problem, and what they would
have found is that `omega / N - 0.5` flags N/2 ranks on even N -- that is,
**no bias at all**. The claim they were trying to check would have looked
fabricated.

**The mechanism, stated correctly.** The verdict is `theta > 0`, so what
matters is how many of the N ranks each form puts above the line:

| form | theta > 0 when | even N | odd N |
|---|---|---|---|
| Bailey & Lopez de Prado, `omega / (N+1) - 0.5` | `omega > (N+1)/2` | N/2 | (N-1)/2 |
| ours before the fix, `(omega-1)/N - 0.5` | `omega > N/2 + 1` | N/2 - 1 | (N-1)/2 |

One rank too few on even N, so PBO read exactly 1/N low: 0.25 at N=4,
0.17 at N=6, 0.10 at N=10, 0.08 at N=12. On odd N the two forms agree and
there is no bias at all -- so the old description of a "uniform downward
bias" was wrong about half the cases, in a way that made the mechanism
look simpler than it is.

**The fix.** The description now matches the code in
`hedging_agent/stats.py`, `docs/LITERATURE.md` section 11,
`notes/KNOWN-TRAPS.md` entry 1, the README and the research cards. The
arithmetic is pinned by `tests/test_pbo_divisor_mechanism.py`, which is
data-free: it counts ranks rather than sampling, so it cannot drift with
the data or the interpreter. One of its tests asserts the wrong
description directly -- that `omega / N` flags N/2 ranks on even N and is
therefore unbiased -- so the mistake cannot come back.

**The lesson:** a correct number under an incorrect mechanism is still a
checkable claim only if the reader can rebuild the mechanism from the prose.
We checked the numbers against the artefacts and never checked the prose
against the code. The checker that compares a documented NUMBER to an
artefact cannot see a documented FORMULA, because no artefact emits a
formula. Every claim in this repository that describes a formula now has a
test that reconstructs it, and the trap that records it says which test.

## 29. A declared Python floor that nothing checked

**The shape.** `pyproject.toml` said `requires-python = ">=3.10"`. The CI
matrix ran 3.11 and 3.12. So the repository made a promise about 3.10
that no job, no test and no reader-facing check ever verified. A
promised-but-unchecked floor is not a floor; it is a number in a file.

**Why it matters more than it sounds.** Python 3.12 (PEP 701) allows a
backslash inside an f-string's expression part:

    f"{data['a\'b']}"

That is a SyntaxError on 3.10 and 3.11. It is easy to write on 3.12, it
survives code review, and it is fatal at COLLECTION time: the whole
suite dies at import, not one test. A reader on the declared floor would
install the package and get nothing at all.

**The trap inside the trap.** The obvious guard is
`ast.parse(source, feature_version=(3, 10))`. It does not work, and we
measured it rather than assuming:

    feature_version=(3, 10): ACCEPTED
    feature_version=(3, 11): ACCEPTED
    feature_version=(3, 12): ACCEPTED

`feature_version` restricts CPython's grammar, but its f-string handling
is not versioned, so a 3.12-only f-string passes the check on every
interpreter. The first version of the guard test used exactly this
mechanism, and its own negative case -- a file containing precisely the
construct it was written to catch -- PASSED. The guard was useless, and
the only reason we know is that we tested the guard with the defect
rather than only against a clean tree.

**A second, blunter version of the same gap.** When the floor was
actually exercised, the suite could not even COLLECT on 3.10:

    tests/test_entry_points.py:25: in <module>
        import tomllib
    E   ModuleNotFoundError: No module named 'tomllib'

`tomllib` entered the standard library in 3.11. Two test files imported
it to read `pyproject.toml`, so on 3.10 pytest died during collection
and no test ran at all. The SHIPPED package never used it, so this was a
test-suite failure, not a runtime one -- but a reader who follows the
documented install on the declared floor gets a dead suite, which is the
same outcome as a dead package.

**The fix.** `tests/_toml_compat.py` imports `tomllib` on 3.11+ and the
`tomli` backport on 3.10, and `tomli` is now a declared test dependency
with a `python_version < '3.11'` marker. The suite collects and runs on
3.10.

**The fix for the floor itself.** `tests/test_supported_python_parses.py` compiles every file
in `hedging_agent/`, `tests/` and `scripts/` with a REAL 3.10
interpreter, found via `HERMES_PYTHON_310`, `python3.10` on PATH, or a
`uv`-managed 3.10. It reads the floor out of `pyproject.toml`, so the
test and the promise cannot drift apart. CI now runs 3.10 as well as 3.11
and 3.12, so the promise is checked on every push. It also checks the interpreter
it found really is the version it claims, because a `python3.10` that
quietly was 3.12 would make the whole file vacuous. CI now runs 3.10,
and on that entry runs the guard.

**The lesson:** a guard has to be tested with the defect it exists to
catch. A check that passes on a clean tree proves only that the clean
tree is clean. The negative case is the test of the test, and the
mechanism you expect to work is the one most likely to have a gap in
it.

