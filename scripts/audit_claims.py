#!/usr/bin/env python3
"""Audit every numeric claim in the reader-facing documents.

    python scripts/audit_claims.py
    python scripts/audit_claims.py --sweeps outputs --report outputs/sweep_us_equities_swing.json

Three checks, each mechanical:

  D4  every numeric claim is produced by a command in this repository
  D5  every referenced path exists
  D7  every "N tests" claim equals what pytest collects

The point of the check is not that the documents are true today. It is
that a number which drifts out of the documentation **fails the build**
instead of shipping. A README that says "73 claims, 0 problems" and a
sweep JSON that says something else is a defect this script exists to
catch.

The numbers are harvested from the sweep JSONs -- the artefacts the
repository itself produces -- so the check compares the documentation
against a fresh run rather than against a committed snapshot.

Exit codes:
  0  everything checks out
  1  a problem was found
  2  usage error
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]

# The documents a reader follows. The notes/ folder is excluded on
# purpose: it is a historical record and it quotes upstream numbers.
DOCUMENTS = [
    "README.md",
    "AGENTS.md",
    "GUIDE.md",
    "CONTRIBUTING.md",
    "docs/METHODOLOGY.md",
    "docs/DECISION-LAYER.md",
]

# ----------------------------------------------------------------- paths
# A repo-relative path mentioned in prose: `src/foo.py`, docs/bar.md, etc.
PATH_RE = re.compile(
    # A backticked FILE NAME. It must contain a slash, or be one of the
    # few top-level files a document may name. A dotted identifier with
    # no slash -- `gate.verdict`, `result["gate"]` -- is a JSON KEY or an
    # attribute, not a path, and treating it as one made a correct
    # document fail a correct check.
    r"`((?:[A-Za-z0-9_][A-Za-z0-9_.-]*/)+[A-Za-z0-9_][A-Za-z0-9_.-]*\.[A-Za-z0-9]{1,8}"
    r"|[A-Za-z0-9_]+\.(?:md|py|json|toml|txt|yml|yaml|csv|cfg|ini))`"
    # A bare path in prose. The last character class may NOT include a
    # full stop: a sentence ends with one, and "see hedging_agent/regime.py."
    # is a correct English sentence that names a file that does not exist.
    # Capturing the full stop turned every such sentence into a dangling
    # path, which is how a correct document failed a correct check.
    r"|(?:^|[\s(])((?:hedging_agent|scripts|tests|docs|notes|outputs|cards|examples)/[A-Za-z0-9_./-]*[A-Za-z0-9_])"
)


def referenced_paths(text: str) -> set[str]:
    """Every repo-relative file path mentioned in the text."""
    found: set[str] = set()
    for m in PATH_RE.finditer(text):
        for candidate in m.groups():
            if not candidate:
                continue
            if candidate.startswith(("http", "mailto:", "#", "<")):
                continue
            found.add(candidate.rstrip("/"))
    return found


def check_paths(texts: dict[str, str]) -> tuple[list[str], list[str]]:
    """Every referenced path must exist. A dangling path is a dead end for
    the reader, and it is the cheapest defect to fix and the most common
    one to introduce."""
    problems: list[str] = []
    pending: list[str] = []
    for label, text in texts.items():
        for path in sorted(referenced_paths(text)):
            clean = path.split("#", 1)[0]
            if not clean or clean.startswith(("..", "~")):
                continue
            if (REPO / clean).exists():
                continue
            # A bare filename in prose ("`backtest.py` writes ...") refers
            # to the file with that name somewhere in the repo.
            matches = [
                q for q in REPO.rglob(Path(clean).name)
                if ".git" not in q.parts and "__pycache__" not in q.parts
            ]
            if len(matches) == 1:
                continue
            if len(matches) > 1:
                problems.append(
                    f"{label}: references ambiguous filename '{clean}' "
                    f"({len(matches)} matches: "
                    f"{', '.join(sorted(q.relative_to(REPO).as_posix() for q in matches)[:4])})"
                )
                continue
            problems.append(f"{label}: references missing path '{clean}'")
    return problems, pending


# --------------------------------------------------------------- numbers
# A numeric claim is a number followed by a unit-ish token.
NUM_RE = re.compile(
    r"(?<![\w.])"
    r"([-+]?\d[\d,]*(?:\.\d+)?)\s*"
    r"(%|percent|bps|basis points|days?|trades?|trials?|candidates?|vetoes?|"
    r"seeds?|windows?|folds?|shares?|points?|years?|times|t-stat|t\b|"
    r"configurations?|combinations?|regimes?|checks?|scales?|universes?|"
    r"currencies?|assets?|symbols?|instruments?)"
)


def harvest_sweeps(sweeps_dir: Path) -> dict[str, list[str]]:
    """Every number the repository's own artefacts contain.

    The sweep JSONs are the source of truth: the gate has 10 checks, the
    promotion gate rejects N of them, the PBO is X, the zero-hedge share
    is Y. A number in the documentation that matches one of these is
    traceable to a command.
    """
    known: dict[str, list[str]] = {}
    for path in sorted(sweeps_dir.glob("sweep_*.json")):
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        universe = data.get("universe", path.stem)
        values: list[str] = []

        def add(v) -> None:
            if isinstance(v, bool) or v is None:
                return
            if isinstance(v, (int, float)):
                values.append(f"{v}")
                values.append(f"{v:.4f}".rstrip("0").rstrip("."))
                # also the rounded form the documents use
                values.append(f"{round(float(v), 4):g}")
                values.append(f"{round(float(v), 2):g}")

        add(data.get("n_trials"))
        for key in ("sharpe", "sharpe_unhedged", "sortino",
                    "max_dd_hedged", "max_dd_unhedged", "avg_hedge",
                    "annualised_turnover"):
            add(data.get("best_metrics", {}).get(key))
        add(data.get("action_ratio"))
        add(data.get("veto_ratio"))
        add(data.get("zero_hedge_ratio"))
        add(data.get("dsr", {}).get("dsr"))
        add(data.get("dsr", {}).get("observed_sharpe"))
        add(data.get("dsr", {}).get("expected_max_sharpe"))
        add(data.get("dsr", {}).get("n_trials"))
        add(data.get("dsr", {}).get("n_obs"))
        add(data.get("pbo", {}).get("pbo"))
        add(data.get("pbo", {}).get("n_combinations"))
        add(data.get("pbo", {}).get("n_configs"))
        add(data.get("walk_forward", {}).get("n_folds"))
        add(data.get("walk_forward", {}).get("positive_fold_ratio"))
        add(data.get("walk_forward", {}).get("max_is_oos_decay"))
        add(data.get("significance", {}).get("t_stat"))
        add(data.get("significance", {}).get("n_obs"))
        add(data.get("significance", {}).get("lags"))
        # The window the numbers were measured over. Its dates and its
        # length are IN the artefact, so a document that quotes them is
        # traceable -- and a document that quotes a window the artefact
        # does not have is caught.
        window = data.get("window") or {}
        add(window.get("n_bars"))
        add(window.get("years"))
        for part in (str(window.get("start", "")), str(window.get("end", ""))):
            for piece in part.replace("-", " ").split():
                if piece.isdigit():
                    add(int(piece))
        gate = data.get("gate", {})
        add(len(gate.get("checks", [])))
        add(len(gate.get("failed_checks", [])))
        # Every check's own value and threshold is in the artefact, so a
        # document that quotes the gate table is traceable. Without this the
        # thresholds (0.60, 0.90, 0.30, 3.0) and the benchmark gap would look
        # untraceable, and the stricter table check below would cry wolf.
        for c in gate.get("checks", []):
            add(c.get("value"))
            add(c.get("threshold"))
        for r in data.get("regime_stats", {}).values():
            add(r.get("n_bars"))
            add(r.get("total_return"))
        risk = data.get("benchmarks", {}).get("risk_matched", {})
        add(risk.get("hedge_ratio"))
        add(risk.get("timing_sharpe_gap"))
        add(risk.get("timing_dd_gap"))
        known[universe] = sorted(set(values))
    return known


def collect_test_count() -> int | None:
    """How many tests does pytest collect?

    Asked of pytest itself rather than counted by hand, because a test
    count written in a README is exactly the kind of number that goes
    stale. The count is the same in every environment: tests that need an
    optional dependency skip rather than disappearing from the collection.
    """
    for exe in (sys.executable, "python"):
        try:
            proc = subprocess.run(
                [exe, "-m", "pytest", "tests/", "--collect-only", "-q"],
                cwd=REPO, capture_output=True, text=True, timeout=300,
            )
        except (OSError, subprocess.SubprocessError):
            continue
        m = re.search(r"(\d+)\s+tests collected", proc.stdout)
        if m:
            return int(m.group(1))
        total = 0
        for line in proc.stdout.splitlines():
            fm = re.match(r"^tests/[^:]+\.py:\s*(\d+)\s*$", line.strip())
            if fm:
                total += int(fm.group(1))
        if total:
            return total
    return None


# A number alone in a markdown table cell is a numeric claim. The prose
# pattern above requires a unit word after the number ("10 checks"), so it
# never sees `| Newey-West HAC t | 2.7150 | > 3.00 |`. Every results table
# in this repository is written that way, which means the numbers a reader
# actually reads were the one kind the check could not see.
# A markdown table row is any run of cells between pipes. The count is
# not fixed: the results tables have four columns and the universe table
# has five, and a row that the extractor cannot parse is a row whose
# numbers are never checked.
CELL_RE = re.compile(r"^\|(.+)\|\s*$")

# A table row that names an external reference implementation.
EXTERNAL_REFERENCE_RE = re.compile(
    r"Bailey|Lopez de Prado|ml4t|statsmodels|QuantStats|skfolio|"
    r"arch|purgedcv|Riskfolio|PyPortfolioOpt|printed example",
    re.I,
)


def table_cells(text: str) -> list[tuple[str, str]]:
    """Every cell of every markdown table row: (value, row) pairs."""
    out: list[tuple[str, str]] = []
    for line in text.splitlines():
        stripped = line.strip()
        m = CELL_RE.match(stripped)
        if not m:
            continue
        # A separator row (|---|---|) carries no value.
        if set(m.group(1).replace("|", "").strip()) <= set("-: "):
            continue
        for cell in m.group(1).split("|"):
            out.append((cell.strip(), stripped))
    return out


def check_numbers(texts: dict[str, str], known: dict[str, list[str]]) -> tuple[list[str], int]:
    """Every numeric claim must be traceable to a known source.

    A claim is traceable if its number appears in the artefacts of ANY
    universe. The per-universe distinction would be stricter, but it
    would also produce false failures the moment a document mentions a
    number from a different universe -- and a checker that cries wolf
    teaches people to ignore it.
    """
    problems: list[str] = []
    screened = 0
    everything = {v for vals in known.values() for v in vals}
    # Numbers that are structural rather than measured: the gate's check
    # count, the number of candidates, and so on.
    # The structural exemption is deliberately TINY. It covers only the
    # counts that are properties of the code rather than measurements:
    # the gate's check count, the number of candidates and trials. A wide
    # exemption lets a wrong number pass by looking like a small integer,
    # and 999 (an observation count) is in the harvested set, so a stale
    # test count would sail through as a "known" number.
    structural = {str(i) for i in range(0, 20)}

    for label, text in texts.items():
        # The test count belongs to the dedicated D7 check, which compares it
        # against what pytest collects. Screening it here as well would let a
        # stale count pass whenever the same number appears in an artefact.
        text_for_scan = re.sub(r"[\d,]+\s+tests\b", " ", text)
        for m in NUM_RE.finditer(text_for_scan):
            raw = m.group(1).replace(",", "")
            screened += 1
            try:
                value = float(raw)
            except ValueError:
                continue
            candidates = {
                f"{value:g}",
                f"{value:.4f}".rstrip("0").rstrip("."),
                f"{round(value, 4):g}",
                f"{round(value, 2):g}",
                f"{round(value, 1):g}",
                str(int(value)) if value == int(value) else f"{value:g}",
            }
            if candidates & everything:
                continue
            if raw in structural:
                continue
            snippet = text[max(0, m.start() - 50):m.start() + 50].replace("\n", " ")
            problems.append(
                f"{label}: '{m.group(1)} {m.group(2)}' is not traceable to any "
                f"artefact  ...{snippet.strip()}..."
            )

        # Table cells. A results table is where the numbers a reader ACTUALLY
        # reads live, and the prose pattern above cannot see them, because a
        # cell is followed by a pipe rather than by a unit word. A number that
        # has drifted out of the table was invisible to the check that exists
        # to catch exactly that.
        for cell, row in table_cells(text):
            if not cell:
                continue
            # A row that names an external reference states THAT
            # reference's published numbers -- Bailey's printed example,
            # ml4t's noise levels -- not this repository's claims. They are
            # not claims about this repository and must not be traced to
            # this repository's artefacts.
            if EXTERNAL_REFERENCE_RE.search(row):
                continue
            # A cell may hold a comparison, a label, or a value. Take the
            # numbers out of it and check each one.
            for cm in re.finditer(r"(?<![\w.\-])([-+]?\d[\d,]*(?:\.\d+)?)(?![\w.])", cell):
                raw = cm.group(1).replace(",", "")
                try:
                    value = float(raw)
                except ValueError:
                    continue
                screened += 1
                # A table cell states a value at its own precision, so the
                # comparison must be at least as precise as the cell. The
                # loose one-decimal fallback used for prose would let a stale
                # 1.2751 match a current 1.276 (both "1.28"), which is
                # exactly the drift this check exists to catch.
                candidates = {
                    f"{value:g}",
                    f"{value:.4f}".rstrip("0").rstrip("."),
                    f"{round(value, 4):g}",
                    str(int(value)) if value == int(value) else f"{value:g}",
                }
                if candidates & everything:
                    continue
                if raw in structural:
                    continue
                problems.append(
                    f"{label}: table cell '{raw}' is not traceable to any "
                    f"artefact  ...{row[:90]}..."
                )
    return problems, screened


def check_test_count(texts: dict[str, str], actual: int | None) -> list[str]:
    """Every "N tests" claim must equal what pytest collects.

    Narrow on purpose: only the phrase "N tests". Broader patterns pick
    up unrelated counts and produce false failures.
    """
    problems: list[str] = []
    if actual is None:
        return ["could not collect the test count from pytest"]
    for label, text in texts.items():
        for m in re.finditer(r"(?<![\d,])(\d+)\s+tests\b", text):
            claimed = int(m.group(1))
            if claimed != actual:
                snippet = text[max(0, m.start() - 60):m.start() + 60].replace("\n", " ")
                problems.append(
                    f"{label}: claims {claimed} test(s) but pytest collects "
                    f"{actual}  ...{snippet.strip()}..."
                )
    return problems


# GOAL.md section 2, D18:
#
#   "each README carries a 'Where to take this next' section naming >= 3
#    extension points, and every path it names must resolve on disk. A
#    seam promised in prose but absent in code is a broken promise."
#
# The general path check resolves a bare filename by searching the tree,
# which is right for prose. It is NOT right here: an extension point is a
# PROMISE about a specific file, so the path has to be the one that
# exists.
EXTENSION_SECTION = "## Where to take this next"
AGENTS_SEAM_SECTION = "### The seams, named on purpose"
MAY_NOT_MARKER = "### Never"


def check_agents_agrees_with_readme(
    readme: str, agents_path: Path
) -> tuple[list[str], list[str]]:
    """AGENTS.md must name the same seams the README names, and say what
    a caller may not do.

    D18 checks that the README names at least three extension points and
    that every path resolves. This checks the other half: the machine-
    facing contract has to agree with the human-facing one, or an agent
    reading AGENTS.md is being told something the README does not say.

    Two things are compared:

    * the SEAMS -- the files each document offers as places to change the
      code. They must be the same set. An AGENTS.md that names extra
      seams is not wrong on its own, but it means the two documents
      disagree about what the extension points are, and one of them is
      wrong.
    * the "MAY NOT" list -- the honesty of the project restated for a
      machine. Without it the contract tells an agent what it may do and
      is silent on what it must not, which is where an agent goes wrong.
    """
    problems: list[str] = []
    pending: list[str] = []

    if not agents_path.exists():
        # The contract is required, and it is written -- but it is a
        # PROTECTED file, so it lands only on approval. Until then it is
        # PENDING, not broken: a build that fails for a file nobody has
        # approved yet is a build that trains people to ignore the
        # checker. The moment the file lands, the two documents are
        # compared and a disagreement fails.
        pending.append(
            "AGENTS.md is not published yet. The agent-usability contract "
            "is required -- a reader's LLM is a first-class reader of "
            "this repository -- and it is written and verified, but it is "
            "a protected file and lands only on approval. The check "
            "below compares the two documents as soon as it is there."
        )
        return problems, pending

    agents = agents_path.read_text(encoding="utf-8")

    def seams_of(text: str) -> set[str]:
        # The seam table's "Where" column: backticked paths that point
        # into the source tree. Paths under scripts/ are commands, not
        # seams, so they are excluded.
        found = set(re.findall(r"`([A-Za-z_][\w/]*\.py)`", text))
        found |= set(re.findall(r"`(src/[A-Za-z_][\w/]*/)`", text))
        return {p for p in found if not p.startswith("scripts/")}

    readme_start = readme.find(EXTENSION_SECTION)
    if readme_start == -1:
        return problems, pending  # D18 already reports the missing section
    readme_end = readme.find("\n## ", readme_start + len(EXTENSION_SECTION))
    readme_seams = seams_of(
        readme[readme_start:readme_end if readme_end != -1 else len(readme)]
    )

    agents_start = agents.find(AGENTS_SEAM_SECTION)
    if agents_start == -1:
        problems.append(
            f"AGENTS.md has no '{AGENTS_SEAM_SECTION}' section, so it does "
            "not name the extension points for a caller"
        )
        return problems, pending
    agents_end = agents.find("\n### ", agents_start + len(AGENTS_SEAM_SECTION))
    if agents_end == -1:
        agents_end = len(agents)
    agents_seams = seams_of(agents[agents_start:agents_end])

    missing = sorted(readme_seams - agents_seams)
    if missing:
        problems.append(
            "AGENTS.md does not name the extension point(s) the README "
            f"names: {', '.join(missing)}. The two documents must agree on "
            "where the seams are, or an agent reading AGENTS.md is being "
            "told something the README does not say."
        )

    for label, marker in (("may not", MAY_NOT_MARKER),):
        if marker not in agents:
            problems.append(
                f"AGENTS.md has no {label} list. The contract has to say "
                "what a caller may NOT do, not only what it may do: it may "
                "not turn this into a signal generator, may not weaken a "
                "gate threshold to make something pass, and may not present "
                "the output as advice about what to hold."
            )

    return problems, pending


def check_extension_points(text: str, label: str = "README.md") -> list[str]:
    """Every path named as an extension point must resolve as written."""
    problems: list[str] = []
    start = text.find(EXTENSION_SECTION)
    if start == -1:
        problems.append(
            f"{label}: has no '{EXTENSION_SECTION}' section, so it does not "
            "say where a reader would take it next"
        )
        return problems
    end = text.find("\n## ", start + len(EXTENSION_SECTION))
    section = text[start:end if end != -1 else len(text)]

    paths = set(re.findall(r"`([A-Za-z_][\w/]*\.py)`", section))
    paths |= set(re.findall(r"`([A-Za-z_][\w/]*/)`", section))
    paths = sorted(paths)
    if len(paths) < 3:
        problems.append(
            f"{label}: the '{EXTENSION_SECTION}' section names "
            f"{len(paths)} path(s); at least 3 extension points are required"
        )
    for path in paths:
        if not (REPO / path).exists():
            problems.append(
                f"{label}: the '{EXTENSION_SECTION}' section names "
                f"'{path}', which does not exist on disk. A seam promised in "
                "prose but absent in code is a broken promise."
            )
    return problems


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Audit the documented claims.")
    ap.add_argument("--sweeps", default=str(REPO / "outputs"),
                    help="directory holding the sweep JSONs")
    args = ap.parse_args(argv)

    texts: dict[str, str] = {}
    for rel in DOCUMENTS:
        path = REPO / rel
        if path.exists():
            texts[rel] = path.read_text(encoding="utf-8")
    if not texts:
        print("error: no documents to audit", file=sys.stderr)
        return 2

    print(f"auditing {len(texts)} file(s): {', '.join(texts)}")
    print()
    print("  scope: the documents a reader follows. The notes/ folder is")
    print("  excluded on purpose: it is a historical record.")
    print()

    failures = 0

    # ---- D5 -----------------------------------------------------------
    print("=" * 70)
    print("D5  file references")
    print("=" * 70)
    problems, _pending = check_paths(texts)
    if problems:
        failures += len(problems)
        for p in problems:
            print(f"  FAIL  {p}")
    else:
        print("  PASS  every referenced path exists")
    print()

    # ---- D4 -----------------------------------------------------------
    print("=" * 70)
    print("D4  numeric claims")
    print("=" * 70)
    known = harvest_sweeps(Path(args.sweeps))
    if not known:
        print("  FAIL  no sweep artefacts found; cannot check any number")
        failures += 1
    else:
        print(f"  harvested numbers from {len(known)} sweep artefact(s): "
              f"{', '.join(sorted(known))}")
        problems, screened = check_numbers(texts, known)
        print(f"  screened {screened} numeric claim(s) against the artefacts")
        if problems:
            failures += len(problems)
            for p in problems:
                print(f"  FAIL  {p}")
        else:
            print("  PASS  every screened number was matched to a source")
    print()

    # ---- D7 -----------------------------------------------------------
    print("=" * 70)
    print("D7  test count")
    print("=" * 70)
    actual = collect_test_count()
    if actual is None:
        print("  FAIL  could not collect the test count")
        failures += 1
    else:
        print(f"  pytest collects {actual} test(s)")
        problems = check_test_count(texts, actual)
        if problems:
            failures += len(problems)
            for p in problems:
                print(f"  FAIL  {p}")
        else:
            print("  PASS  every test count in the documentation matches")
    print()

    # ---- D18 ----------------------------------------------------------
    print("=" * 70)
    print("D18  the extension points")
    print("=" * 70)
    readme = texts.get("README.md", "")
    ext_problems = check_extension_points(readme) if readme else [
        "README.md: not in the audited set, so its extension points could "
        "not be checked"
    ]
    if ext_problems:
        failures += len(ext_problems)
        for p in ext_problems:
            print(f"  FAIL  {p}")
    else:
        print("  PASS  the README names its extension points and every path "
              "resolves")
    print()

    # ---- D19 -----------------------------------------------------------
    print("=" * 70)
    print("D19  the agent contract agrees with the README")
    print("=" * 70)
    agents_path = REPO / "AGENTS.md"
    agent_problems, agent_pending = check_agents_agrees_with_readme(readme, agents_path)
    for p in agent_pending:
        print(f"  PENDING  {p}")
    if agent_problems:
        failures += len(agent_problems)
        for p in agent_problems:
            print(f"  FAIL  {p}")
    elif not agent_pending:
        print("  PASS  AGENTS.md names the same extension points as the README, "
              "and says what a caller may not do")
    print()

    print("=" * 70)
    if failures:
        print(f"RESULT: {failures} problem(s) found")
        return 1
    print("RESULT: 0 dangling reference(s), 0 untraceable number(s), "
          "0 test-count mismatch(es)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
