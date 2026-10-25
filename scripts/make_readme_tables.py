#!/usr/bin/env python3
"""Render the README's results tables from the sweep artefacts.

    python scripts/make_readme_tables.py            # write the tables
    python scripts/make_readme_tables.py --check    # fail if they are stale

**Why this exists.** The README is the entry document, and its results
tables were hand-written. They drifted: the README quoted the Sharpe,
the PBO and the HAC t-statistic of a superseded sweep generation, while
the committed artefacts held the current ones. The claim audit did not
catch it, because the audit's prose pattern needs a unit word after a
number ("10 checks") and every one of those numbers sits in a table
cell, followed by a pipe.

Checking a hand-written table against an artefact asks a question every
time: did the writer remember to update it? Generating the table from
the artefact asks the question once, at build time, and cannot be
forgotten.

The same idea as `scripts/make_cards.py`, which generates the research
cards from the same sweep JSONs. The rule the repository already follows:
the machine artefact and the human artefact must not be able to drift.

The tables live between two marker comments in README.md:

    <!-- BEGIN GENERATED: gate-table -->
    ...rendered...
    <!-- END GENERATED: gate-table -->

Only the marked blocks are touched. The prose around them is written by
a human and is checked by `scripts/audit_claims.py`.

Exit codes:
  0  the tables are current (or were written)
  1  a table is stale
  2  usage error
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]

BEGIN = "<!-- BEGIN GENERATED: {name} -->"
END = "<!-- END GENERATED: {name} -->"


def load_sweeps(sweeps_dir: Path) -> dict[str, dict]:
    """Every committed sweep, keyed by universe."""
    out: dict[str, dict] = {}
    for path in sorted(sweeps_dir.glob("sweep_*.json")):
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        universe = data.get("universe")
        if universe:
            out[universe] = data
    return out


def fmt(value) -> str:
    """A number as the README states it."""
    if isinstance(value, float):
        text = f"{value:.10f}".rstrip("0").rstrip(".")
        return text if text else "0"
    return str(value)


def gate_table(sweeps: dict[str, dict]) -> str:
    """The promotion-gate table for the headline universe."""
    data = sweeps.get("us_equities")
    if not data:
        return ""
    checks = {c["check"]: c for c in data["gate"]["checks"]}

    def row(label: str, check: str, threshold: str) -> str:
        c = checks.get(check)
        if c is None:
            return ""
        verdict = "PASS" if c["pass"] else "**FAIL**"
        note = ""
        if check == "min_dsr":
            note = " (see below; do not trust it)"
        return f"| {label} | {fmt(c['value'])} | {threshold} | {verdict}{note} |"

    lines = [
        "| check | value | threshold | verdict |",
        "|---|---|---|---|",
    ]
    lines += [
        row("min Sharpe", "min_sharpe", ">= 0.60"),
        row("min DSR", "min_dsr", ">= 0.90"),
        row("max PBO", "max_pbo", "<= 0.30"),
        row("positive fold ratio", "positive_fold_ratio", ">= 0.60"),
        row("IS->OOS Sharpe decay", "max_is_oos_decay", "<= 0.50"),
        row("nonzero action ratio", "nonzero_action_ratio", ">= 0.02"),
        row("regimes positive", "min_regimes_positive", ">= 2"),
        row("beats risk-matched benchmark", "beats_simple_benchmark", "> 0"),
        row("Newey-West HAC t", "significance_t", "> 3.00"),
    ]
    return "\n".join(line for line in lines if line)


def universe_table(sweeps: dict[str, dict]) -> str:
    """One row per universe: Sharpe, PBO, the t-statistic, the verdict.

    The window is in the table because it is not the same for all three:
    the crypto provider caps at 1000 daily bars, so a 5y request covers
    2.74 years there and five years elsewhere. A reader comparing the
    three Sharpes needs to know that, and it is read from the artefact
    rather than typed.
    """
    lines = [
        "| universe | window | Sharpe hedged / unhedged | PBO | t | verdict |",
        "|---|---|---|---|---|---|",
    ]
    for universe in ("us_equities", "mixed", "crypto"):
        data = sweeps.get(universe)
        if not data:
            continue
        m = data["best_metrics"]
        failed = len(data["gate"]["failed_checks"])
        window = data.get("window") or {}
        if window.get("start"):
            win = (f"{window['start']} to {window['end']} "
                   f"({window['years']}y)")
        else:
            win = "not recorded"
        lines.append(
            f"| {universe} | {win} | {fmt(m['sharpe'])} / {fmt(m['sharpe_unhedged'])} "
            f"| {fmt(data['pbo']['pbo'])} | {fmt(data['significance']['t_stat'])} "
            f"| REJECTED ({failed} failed) |"
        )
    return "\n".join(lines)


def render(name: str, body: str, text: str) -> str:
    """Replace the marked block in `text` with `body`."""
    begin = BEGIN.format(name=name)
    end = END.format(name=name)
    pattern = re.compile(
        re.escape(begin) + r".*?" + re.escape(end), re.DOTALL)
    if not pattern.search(text):
        raise SystemExit(
            f"error: no marker block {name!r} in the README.\n"
            f"       expected {begin} ... {end}"
        )
    return pattern.sub(begin + "\n" + body + "\n" + end, text)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--sweeps", default=str(REPO / "outputs"),
                    help="directory holding the sweep JSONs")
    ap.add_argument("--readme", default=str(REPO / "README.md"),
                    help="the README whose tables are generated")
    ap.add_argument("--check", action="store_true",
                    help="do not write; fail if any table is out of date")
    args = ap.parse_args(argv)

    sweeps = load_sweeps(Path(args.sweeps))
    if not sweeps:
        print("error: no sweep artefacts found", file=sys.stderr)
        return 2

    readme = Path(args.readme)
    if not readme.exists():
        print(f"error: no README at {readme}", file=sys.stderr)
        return 2

    # Read the BYTES, not the text. `read_text()` applies universal
    # newlines, so a CRLF file reads as LF text and a text comparison
    # cannot see that the bytes on disk are wrong: the check reports
    # "current" for a file that is not. The published artefact is bytes.
    raw = readme.read_bytes()
    original = raw.decode("utf-8")
    updated = render("gate-table", gate_table(sweeps), original)
    updated = render("universe-table", universe_table(sweeps), updated)
    # LF, whatever the file on disk had. A file written on Windows and one
    # written on Linux must have the same bytes, or a reader who regenerates
    # it sees "the whole file changed". The .gitattributes normalises the
    # committed copy, but the generator must not depend on that: the
    # published bytes are the ones it writes.
    #
    # This runs even when the tables are unchanged, because a file that
    # arrived with CRLF is still wrong and the generator owns the README's
    # bytes.
    updated = updated.replace("\r\n", "\n")
    want = updated.encode("utf-8")

    if args.check:
        if raw == want:
            print("the README's generated tables are current")
            return 0
        if raw.replace(b"\r\n", b"\n") == want:
            print("FAIL: the README has CRLF line endings; the published bytes "
                  "must be the same on every machine", file=sys.stderr)
            print("  run: python scripts/make_readme_tables.py", file=sys.stderr)
            return 1
        print("FAIL: the README's generated tables are stale", file=sys.stderr)
        for name in ("gate-table", "universe-table"):
            begin = BEGIN.format(name=name)
            end = END.format(name=name)
            pat = re.compile(re.escape(begin) + r".*?" + re.escape(end), re.DOTALL)
            was = pat.search(original)
            now = pat.search(updated)
            if was and now and was.group(0) != now.group(0):
                print(f"  stale: {name}", file=sys.stderr)
                print(f"    in the file: {was.group(0)[:200]}", file=sys.stderr)
                print(f"    should be  : {now.group(0)[:200]}", file=sys.stderr)
        print("  run: python scripts/make_readme_tables.py", file=sys.stderr)
        return 1

    if raw == want:
        print("the README's generated tables were already current")
        return 0
    readme.write_bytes(want)
    print(f"wrote the generated tables into {readme.name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
