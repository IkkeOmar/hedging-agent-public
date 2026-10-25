#!/usr/bin/env python3
"""Generate a research card next to every sweep JSON.

    python scripts/make_cards.py                    # regenerate every card
    python scripts/make_cards.py --check            # fail if any card is stale
    python scripts/make_cards.py --out cards        # where to write them

The card is GENERATED from the sweep JSON, never written by hand. That is
the whole point: a hand-written card drifts from the artefact it
describes, and the drift is invisible because both look fine. Here the
card is a function of the JSON, so a test can regenerate every card and
diff it against the committed one -- and a stale card fails the build.

The section order is fixed by the brief in GOAL.md section 4:

  1. The idea                    (plain language, no jargon)
  2. The failure mode it targets
  3. The conditions              (plain words, formal name in parentheses once)
  4. Where it came from          (resolvable citation)
  5. What would have falsified it
  6. What we measured            (with N, always)
  7. Where it works              (the regime table)
  8. Where it does not           (failed regimes and failing checks)
  9. The verdict
  10. What we got wrong          (the entry from the defect ledger)

Exit codes:
  0  cards are current (or were written)
  1  a card is stale (with --check)
  2  usage error
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]

# The defect ledger, and the entry that applies to each universe. The
# mapping is by SUBSTRING of the universe name, so a new universe gets a
# sensible default rather than silently getting no entry.
TRAP_BY_UNIVERSE: list[tuple[str, str]] = [
    (
        "crypto",
        "**This universe's numbers were wrong twice before they were right.** "
        "The Sharpe and the borrow charge were both annualised with the "
        "**equity** factor (252 bars a year) instead of crypto's 365 -- a "
        "16.9% understatement of the Sharpe and a 45% overstatement of the "
        "financing cost. Both were found by noticing that one artefact "
        "reported two different Sharpes for the same return series. See "
        "traps 3 and 13.",
    ),
    (
        "mixed",
        "**This universe once contained no crypto at all.** Its declared "
        "assets were dropped by a filter that tested the wrong character, so "
        "the sweep ran on seven equity columns while the documentation "
        "described a mixed portfolio. Adding the crypto back moved the "
        "Sharpe from 1.1283 to 0.3838 and the zero-hedge share from 0.7966 "
        "to 0.0896. See trap 14.",
    ),
    (
        "us_equities",
        "**Our own PBO was biased low.** The out-of-sample rank was scaled "
        "by (rank - 1) / N instead of rank / (N + 1), which flagged one rank "
        "too few on even N and flattered every candidate by 1/N (0.25 at "
        "N=4). Found by cross-checking against an independent implementation. "
        "The published values here are post-fix. See trap 1.",
    ),
]

DEFAULT_TRAP = (
    "**See `notes/KNOWN-TRAPS.md`.** Every defect that was in this code is "
    "recorded there, with the mechanism and how it was caught."
)


def trap_for(universe: str) -> str:
    for key, text in TRAP_BY_UNIVERSE:
        if key in universe:
            return text
    return DEFAULT_TRAP


def fmt(value, digits: int = 4) -> str:
    """A number, or the word for 'there isn't one'."""
    if value is None:
        return "not measurable"
    if isinstance(value, bool):
        return "yes" if value else "no"
    if isinstance(value, (int,)):
        return str(value)
    if isinstance(value, float):
        if value != value:  # NaN
            return "not measurable"
        return f"{value:.{digits}f}"
    return str(value)


def yaml_list(values) -> str:
    if not values:
        return "[]"
    return "[" + ", ".join(json.dumps(v) for v in values) + "]"


def citation_for(spec: dict) -> str:
    """The citation for the winning spec.

    The template specs carry their source in the description's own words.
    We print the spec's name and the literature file, and the claim audit
    checks that the file exists -- a citation a reader cannot resolve is
    not a citation.
    """
    return (
        f"The candidate bank in `hedging_agent/strategies.py`, and the "
        f"thresholds and their primary sources in "
        f"[`docs/LITERATURE.md`](../docs/LITERATURE.md)."
    )


def conditions_block(spec: dict) -> list[str]:
    """The conditions, in plain words, with the formal name once."""
    rows = [
        f"Hold a target volatility of {fmt(spec['target_vol'], 3)} "
        f"(the `target_vol` parameter)",
        f"Scale the hedge by how far realised volatility sits above that "
        f"target, with a sensitivity of {fmt(spec['vol_gamma'], 3)} "
        f"(the `vol_gamma` parameter)",
        f"Only add protection once the portfolio is actually down "
        f"{fmt(abs(spec['dd_trigger']) * 100, 1)}% "
        f"(the `dd_trigger` drawdown trigger), ramping to full by "
        f"{fmt(abs(spec['dd_full']) * 100, 1)}% "
        f"(the `dd_full` level)",
        f"Act only when at least {fmt(spec['consensus_required'], 3)} of "
        f"the volatility scales agree (the `consensus_required` "
        f"agreement threshold)",
        f"Cap the hedge at {fmt(spec['max_hedge_fraction'], 3)} of the "
        f"portfolio (the `max_hedge_fraction` cap), and change it by no "
        f"more than {fmt(spec['max_daily_turnover'], 3)} a day "
        f"(the `max_daily_turnover` cap)",
    ]
    if spec.get("veto_on_expansion"):
        rows.append(
            "Hold no hedge at all while the market is expanding and the "
            "trend is intact (the `veto_on_expansion` rule)"
        )
    rows.append(
        f"Pay {fmt(spec['commission_bps'], 1)} basis points of commission and "
        f"{fmt(spec['spread_bps'], 1)} of spread on every trade, plus "
        f"{fmt(spec['borrow_bps_annual'], 1)} a year to borrow "
        f"(the cost model)"
    )
    return rows


def build_card(result: dict) -> str:
    """The whole card, as markdown."""
    spec = result["best_spec"]
    metrics = result["best_metrics"]
    dsr = result["dsr"]
    pbo = result["pbo"]
    wf = result["walk_forward"]
    gate = result["gate"]
    hedge = result.get("hedge", {})
    universe = result["universe"]
    timeframe = result["timeframe"]

    failed = gate["failed_checks"]
    verdict = gate["verdict"]

    # --- frontmatter ---------------------------------------------------
    lines = [
        "---",
        f"name: {json.dumps(spec['name'])}",
        f"universe: {json.dumps(universe)}",
        f"timeframe: {json.dumps(timeframe)}",
        f"n_trials: {result['n_trials']}",
        f"gate: {json.dumps(verdict)}",
        f"failed_checks: {yaml_list(failed)}",
        "---",
        "",
        f"# {spec['name']}",
        "",
    ]

    # --- 1. the idea ----------------------------------------------------
    lines += [f"## The idea", "", spec.get("description", "").strip(), ""]

    # --- 2. the failure mode -------------------------------------------
    lines += [
        "## The failure mode it targets",
        "",
        spec.get("failure_mode", "").strip(),
        "",
    ]

    # --- 3. the conditions ---------------------------------------------
    lines += ["## The conditions", ""]
    lines += [f"- {row}" for row in conditions_block(spec)]
    lines += [""]

    # --- 4. where it came from -----------------------------------------
    lines += ["## Where it came from", "", citation_for(spec), ""]

    # --- 5. what would have falsified it --------------------------------
    lines += [
        "## What would have falsified it",
        "",
        "Before running it, this strategy had to beat a constant hedge "
        "carrying the same realised risk, hold up out of sample, and work "
        "in more than one market regime. If it could not, it was not a "
        "strategy -- it was a lucky configuration.",
        "",
    ]

    # --- 6. what we measured --------------------------------------------
    n = result["n_trials"]
    # The window the numbers were measured over, read from the artefact.
    # A Sharpe without the window it was measured in is not a result, it is
    # a number -- and the window is not always what was asked for: the
    # crypto provider caps at 1000 daily bars, so a 5y request covers 2.74
    # years there.
    window = result.get("window") or {}
    if window.get("start") and window.get("end"):
        window_line = (
            f"{window['start']} to {window['end']} "
            f"({window['n_bars']} bars, {window['years']} years)"
        )
    else:
        window_line = "the window recorded in the sweep JSON"
    lines += [
        "## What we measured",
        "",
        f"Every figure below comes from one sweep of **{n} trials** over "
        f"the {universe} universe on the {timeframe} timeframe, measured over "
        f"**{window_line}**. The Sharpe "
        f"is the winner's, and it is only meaningful next to the number of "
        f"things tried to find it.",
        "",
        f"- **Sharpe (annualised):** {fmt(metrics['sharpe'])} hedged, "
        f"against {fmt(metrics['sharpe_unhedged'])} unhedged",
        f"- **Deflated Sharpe:** {fmt(dsr['dsr'])} -- the probability that "
        f"the observed Sharpe beats the best of {dsr['n_trials']} random "
        f"trials",
        f"- **Probability of backtest overfitting:** {fmt(pbo['pbo'])} over "
        f"{pbo['n_combinations']} combinations of {pbo['n_configs']} "
        f"configurations",
        f"- **Newey-West HAC t-statistic:** {fmt(result['significance']['t_stat'])} "
        f"against a hurdle of 3.0",
        f"- **In-sample to out-of-sample decay:** "
        f"{fmt(wf['max_is_oos_decay'])} (the worst fold)",
        f"- **Positive folds:** {fmt(wf['positive_fold_ratio'])} over "
        f"{wf['n_folds']} windows",
        f"- **Action ratio:** {fmt(result['action_ratio'])} -- the share of "
        f"days on which the hedge actually changed",
        f"- **Veto ratio:** {fmt(result['veto_ratio'])} -- the share of days "
        f"the risk rule said no",
        f"- **Zero-hedge share:** {fmt(result['zero_hedge_ratio'])} -- the "
        f"share of days on which the correct answer was to hold no hedge "
        f"at all",
        f"- **Max drawdown:** {fmt(metrics['max_dd_hedged'])} hedged, against "
        f"{fmt(metrics['max_dd_unhedged'])} unhedged",
        f"- **Hedge instrument:** {hedge.get('hedge_proxy', 'n/a')} "
        f"({hedge.get('hedge_side_label', 'n/a')}), correlated "
        f"{fmt(hedge.get('hedge_corr_to_portfolio'))} with the portfolio",
        f"- **Costs charged:** {fmt(metrics.get('annualised_turnover'), 2)}x "
        f"annualised turnover, at "
        f"{fmt(hedge.get('round_trip_cost_bps'), 1)} basis points of "
        f"commission and spread plus {fmt(hedge.get('borrow_bps_annual'), 1)} "
        f"a year to borrow",
        "",
    ]

    risk = result.get("benchmarks", {}).get("risk_matched", {})
    if risk:
        lines += [
            f"Against a constant hedge carrying the same realised risk: the "
            f"hedge ratio was {fmt(risk.get('hedge_ratio'))}, and the "
            f"timing was worth {fmt(risk.get('timing_sharpe_gap'))} of "
            f"Sharpe. That is the number that isolates *when* to hedge from "
            f"*how much*.",
            "",
        ]

    # --- 7. where it works ----------------------------------------------
    lines += ["## Where it works", ""]
    regimes = result.get("regime_stats", {})
    worked = []
    for name, r in regimes.items():
        if r.get("significant"):
            worked.append((name, r))
    if worked:
        lines += ["| Regime | Sharpe | Return | Max drawdown | Days |", "|---|---|---|---|---|"]
        for name, r in worked:
            lines.append(
                f"| {name} | {fmt(r['sharpe'])} | {fmt(r['total_return'])} | "
                f"{fmt(r['max_dd'])} | {r['n_bars']} |"
            )
    else:
        lines.append(
            "**Nowhere.** No regime produced a Sharpe that could be "
            "estimated with enough confidence to report."
        )
    lines += [""]

    # --- 8. where it does not -------------------------------------------
    lines += ["## Where it does not", ""]
    not_worked = [(n, r) for n, r in regimes.items() if not r.get("significant")]
    if not_worked:
        lines += [
            "| Regime | Days | Why there is no number |",
            "|---|---|---|",
        ]
        for name, r in not_worked:
            reason = (
                "too few bars to estimate the Sharpe"
                if r.get("sharpe") is None
                else "the Sharpe is not distinguishable from noise"
            )
            lines.append(f"| {name} | {r['n_bars']} | {reason} |")
        lines.append("")
    if failed:
        lines += ["The promotion gate failed these checks:", ""]
        lines += [f"- `{c}`" for c in failed]
        lines += [""]

    # --- 9. the verdict --------------------------------------------------
    if verdict == "PROMOTED":
        verdict_text = (
            "**Promoted.** It cleared every check of the promotion gate on "
            "walk-forward data."
        )
    else:
        verdict_text = (
            f"**Rejected.** It failed {len(failed)} of the "
            f"{len(gate['checks'])} checks in the promotion gate: "
            f"{', '.join('`' + c + '`' for c in failed)}. The gate is "
            f"fail-closed, so one failure is enough, and a rejected "
            f"candidate is a normal, publishable result."
        )
    lines += ["## The verdict", "", verdict_text, ""]

    # --- 10. what we got wrong -------------------------------------------
    lines += ["## What we got wrong", "", trap_for(universe), ""]

    return "\n".join(lines)


def card_path(out_dir: Path, result: dict) -> Path:
    return out_dir / f"card_{result['universe']}_{result['timeframe']}.md"


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Generate the research cards.")
    ap.add_argument("--sweeps", default=str(REPO / "outputs"),
                    help="directory holding the sweep JSONs")
    ap.add_argument("--out", default=str(REPO / "cards"),
                    help="directory to write the cards into")
    ap.add_argument("--check", action="store_true",
                    help="do not write; fail if any card is out of date")
    args = ap.parse_args(argv)

    sweeps = Path(args.sweeps)
    out_dir = Path(args.out)

    files = sorted(sweeps.glob("sweep_*.json"))
    if not files:
        print(f"error: no sweep JSONs in {sweeps}", file=sys.stderr)
        return 2

    if not args.check:
        out_dir.mkdir(parents=True, exist_ok=True)

    stale: list[str] = []
    written = 0
    for path in files:
        result = json.loads(path.read_text(encoding="utf-8"))
        card = build_card(result)
        target = card_path(out_dir, result)
        if args.check:
            if not target.exists():
                stale.append(f"{target.name}: missing")
                continue
            current = target.read_text(encoding="utf-8")
            if current != card:
                stale.append(f"{target.name}: out of date")
            continue
        target.write_text(card, encoding="utf-8", newline="\n")
        written += 1
        # relative_to() raises when --out points outside the repository, so
        # show the path as given and only shorten it when it is under the
        # repo. A generator that crashes on a legal argument is a defect.
        try:
            shown = target.relative_to(REPO)
        except ValueError:
            shown = target
        print(f"[card] {shown}  ({len(card)} bytes)")

    if args.check:
        if stale:
            print("\nSTALE CARDS:", file=sys.stderr)
            for s in stale:
                print(f"  ! {s}", file=sys.stderr)
            print("\nRun `python scripts/make_cards.py` to regenerate them.",
                  file=sys.stderr)
            return 1
        print(f"all {len(files)} card(s) are current")
        return 0

    print(f"\n{written} card(s) written to {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
