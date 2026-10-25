#!/usr/bin/env python3
"""Run the research loop and print the report. This is the entry point.

    python scripts/run_research.py --universe us_equities --timeframe swing
    python scripts/run_research.py --universe crypto --timeframe intraday
    python scripts/run_research.py --all
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import traceback

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from hedging_agent.data import UNIVERSES
from hedging_agent.pipeline import print_report, run_sweep


def result_filename(universe: str, timeframe: str, args) -> str:
    """Name the artifact after the config that produced it.

    A report whose filename does not encode its own settings is indistinguishable
    from every other report. That is not theoretical: a 4-trial diagnostic run
    overwrote the canonical 48-trial us_equities sweep, and the JSON on disk no
    longer described the numbers in the README. A reader could not tell which
    config had produced the file they were looking at.

    Only the canonical configuration (6 candidates x 8 trials, seed 11, 5y)
    keeps the bare `sweep_<universe>_<tf>.json` name, so the headline artifacts
    stay stable and greppable. Anything else is suffixed with the config that
    made it, and therefore cannot silently overwrite a headline.
    """
    canonical = (
        args.candidates == 6
        and args.trials == 8
        and args.seed == 11
        and args.period == "5y"
    )
    base = f"sweep_{universe}_{timeframe}"
    if canonical:
        return f"{base}.json"
    tag = (
        f"c{args.candidates}_t{args.trials}_s{args.seed}_{args.period}"
    )
    return f"{base}__{tag}.json"


def _say(message: str = "") -> None:
    """Print a human message.

    In --json mode stdout is reserved for the JSON payload and NOTHING
    else, so a program can do json.loads(stdout) without having to strip
    banners first. Every human-facing line goes to stderr instead.

    This is the convention the repository follows everywhere: stdout is
    for the machine, stderr is for the human.
    """
    print(message, file=sys.stderr)


def main() -> int:
    p = argparse.ArgumentParser(description="hedge agent research loop")
    p.add_argument("--universe", default="us_equities", choices=sorted(UNIVERSES))
    p.add_argument("--timeframe", default="swing", choices=["intraday", "swing", "position"])
    # These defaults ARE the canonical configuration: 6 candidates x 8
    # trials, seed 11. The script only writes the bare `sweep_<universe>_<tf>.json`
    # for that configuration, because a report whose filename does not encode
    # its own settings is indistinguishable from every other report. With
    # different defaults, the documented command produced a suffixed file and
    # the canonical artefact was never written.
    p.add_argument("--candidates", type=int, default=6)
    p.add_argument("--trials", type=int, default=8,
                   help="parameter trials per candidate")
    p.add_argument("--period", default="5y",
                   help="history window. The default is the one the committed "
                        "price fixtures cover, so the documented command runs "
                        "offline and reproduces the published numbers.")
    p.add_argument("--seed", type=int, default=11)
    p.add_argument("--no-llm", action="store_true",
                   help="skip the LLM and use the built-in template bank")
    p.add_argument("--all", action="store_true",
                   help="run every universe, for each timeframe in --timeframes")
    p.add_argument("--timeframes", nargs="+", default=None,
                   choices=["intraday", "swing", "position"],
                   help="timeframes to run with --all (default: swing). "
                        "The intraday timeframe needs at least 400 bars of "
                        "history, so a short fixture will refuse it.")
    p.add_argument("--out", default="outputs", help="where to write the JSON reports")
    p.add_argument("--offline", action="store_true",
                   help="read the committed price fixtures (no network). "
                        "This is the default: a live download cannot "
                        "reproduce the published numbers, because the crypto "
                        "provider returns the most recent 1000 daily bars and "
                        "the window moves with the calendar.")
    p.add_argument("--live", action="store_true",
                   help="download prices from the provider instead of using "
                        "the committed fixtures. Needs the network; the "
                        "results will NOT match the published numbers.")
    p.add_argument("--json", action="store_true",
                   help="print the result as JSON on stdout, for a program "
                        "to consume instead of a human to read")
    p.add_argument("--strict", action="store_true",
                   help="re-raise on the first failure instead of continuing")
    args = p.parse_args()

    os.makedirs(args.out, exist_ok=True)
    combos = []
    if args.all:
        # Every universe in the registry, and every requested timeframe.
        # The mixed universe used to be missing from this list, so the one
        # documented way to run "everything" never produced the mixed
        # sweep.
        combos = [(u, t) for u in sorted(UNIVERSES)
                  for t in (args.timeframes or ["swing"])]
    else:
        combos = [(args.universe, args.timeframe)]

    results = {}
    for universe, tf in combos:
        _say("\n" + "#" * 74)
        _say(f"# {universe} / {tf}  hedge proxy: {UNIVERSES[universe].hedge_proxy}")
        _say("#" * 74)
        try:
            res = run_sweep(
                universe=universe, timeframe=tf,
                n_candidates=args.candidates, trials_per_candidate=args.trials,
                seed=args.seed, period=args.period, use_llm=not args.no_llm,
                offline=not args.live,
            )
        except Exception as e:
            # NEVER swallow a traceback. An earlier version of this handler
            # printed only `type(e).__name__: e`, which reduced a real bug to
            # "ValueError: high - low < 0" with no file, no line, no context --
            # costing several debugging cycles guessing at a plotting bug that
            # did not exist. A research tool that hides its own failures is
            # worse than one that crashes loudly.
            _say(f"[{universe}/{tf}] FAILED: {type(e).__name__}: {e}")
            _say("-" * 74)
            traceback.print_exc(file=sys.stderr)
            _say("-" * 74)
            results[f"{universe}/{tf}"] = {
                "error": str(e),
                "error_type": type(e).__name__,
                "traceback": traceback.format_exc(),
            }
            if args.strict:
                raise
            continue
        if args.json:
            # stdout is the JSON payload and NOTHING else. The human
            # report goes to stderr so a program can do json.loads(stdout)
            # without stripping banners first.
            print(json.dumps(res.as_dict(), indent=2, default=str))
        else:
            print_report(res)
        results[f"{universe}/{tf}"] = res.as_dict()
        path = os.path.join(args.out, result_filename(universe, tf, args))
        # newline="\n" so the file on disk is byte-identical to the
        # committed version on every platform. Without it, Windows writes
        # CRLF, and .gitattributes normalises the difference away -- so
        # `git status` says the file is unchanged while the bytes differ.
        with open(path, "w", newline="\n", encoding="utf-8") as f:
            json.dump(res.as_dict(), f, indent=2, default=str)
        _say(f"[saved] {path}")

    if len(results) > 1 and not args.json:
        _say("\n" + "=" * 74)
        _say("CROSS-UNIVERSE SUMMARY (the honest headline)")
        _say("=" * 74)
        for k, v in results.items():
            if "error" in v:
                _say(f"  {k:<24} ERROR {v['error']}")
                continue
            g = v["gate"]
            _say(f"  {k:<24} trials {v['n_trials']:>4}  sharpe {v['best_metrics']['sharpe']:>6}  "
                  f"DSR {v['dsr']['dsr']:>6}  PBO {v['pbo']['pbo']:>5}  "
                  f"zero-hedge {v['zero_hedge_ratio']:>5}  {g['verdict']}")
        _say("A verdict of REJECTED is a normal, publishable result. It means the")
        _say("gate did its job.")
    if args.json:
        # A machine consumer wants the results, not a status. The verdict is
        # inside them. A non-zero exit would make the whole run look failed
        # when the gate did exactly its job.
        pass

    # Exit codes that mean something:
    #   0 = ran, and every gate that was reached passed
    #   1 = the gate REJECTED. This is a result, not an error.
    #   2 = usage error (argparse handles it)
    rejected = [
        k for k, v in results.items()
        if "error" not in v and not v.get("gate", {}).get("passed", False)
    ]
    failed = [k for k, v in results.items() if "error" in v]
    if failed:
        print(f"\n{len(failed)} run(s) failed: {', '.join(failed)}", file=sys.stderr)
        return 1
    if rejected:
        if not args.json:
            _say(f"\nREJECTED by the promotion gate: {', '.join(rejected)}")
            _say("That is a result, not a failure. The gate did its job.")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())