#!/usr/bin/env python3
"""Run a gate-passed spec forward with the decision layer in the loop.

Prints the journal: what the regime was, what the quantitative read wanted,
what the decision model said, and what the agent actually did. The veto line is
the point of the whole project, so it is a first-class column.

    python scripts/run_agent.py --universe us_equities
    python scripts/run_agent.py --universe us_equities --decision-every 5
"""

from __future__ import annotations

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from hedging_agent.data import UNIVERSES
from hedging_agent.pipeline import run_agent
from hedging_agent.strategies import generate_candidates, template_candidates, validate_spec


def _say(*args, **kwargs) -> None:
    """Print a human message to stderr.

    In --json mode stdout is reserved for the JSON payload and NOTHING
    else, so a program can do json.loads(stdout) without stripping banners
    first. Every human-facing line goes to stderr instead.
    """
    kwargs.setdefault("file", sys.stderr)
    print(*args, **kwargs)


def main() -> int:
    p = argparse.ArgumentParser(description="run the hedge agent forward")
    p.add_argument("--universe", default="us_equities", choices=sorted(UNIVERSES))
    p.add_argument("--timeframe", default="swing", choices=["intraday", "swing", "position"])
    p.add_argument("--period", default="5y",
                   help="history window. The default is the one the committed "
                        "price fixtures cover.")
    p.add_argument("--spec", default=None, help="path to a JSON spec; default = best template")
    p.add_argument("--no-decision", action="store_true",
                   help="run the quantitative policy only, no decision layer")
    p.add_argument("--decision-every", type=int, default=1,
                   help="call the decision model every N bars (cost control)")
    p.add_argument("--tail", type=int, default=30, help="rows of journal to print")
    p.add_argument("--out", default="outputs")
    p.add_argument("--offline", action="store_true",
                   help="read the committed price fixtures (no network)")
    p.add_argument("--live", action="store_true",
                   help="download prices from the provider instead")
    p.add_argument("--json", action="store_true",
                   help="print the result as JSON on stdout")
    args = p.parse_args()

    if args.spec:
        with open(args.spec) as f:
            from hedging_agent.strategies import HedgeStrategySpec
            spec = HedgeStrategySpec(**json.load(f))
    else:
        candidates = template_candidates(6)
        problems = [(s.name, validate_spec(s)) for s in candidates]
        good = [s for s in candidates if not validate_spec(s)]
        if not good:
            _say("no valid template spec")
            return 1
        spec = good[0]
        if problems:
            _say("[templates] rejected:",
                  {n: p[0] for n, p in problems if p})

    os.makedirs(args.out, exist_ok=True)
    res = run_agent(
        spec=spec, universe=args.universe, timeframe=args.timeframe,
        period=args.period, use_decision_layer=not args.no_decision,
        decision_every=max(1, args.decision_every),
        offline=not args.live,
    )
    if "error" in res:
        _say("ERROR:", res["error"])
        return 1

    log = res["log"]
    if args.json:
        print(json.dumps({
            "universe": res["universe"],
            "timeframe": res["timeframe"],
            "spec": res["spec"],
            "backend": res["backend"],
            "n_bars": res["n_bars"],
            "decision_calls": res["decision_calls"],
            "decision_cost_usd": res["decision_cost_usd"],
            "veto_ratio": res["veto_ratio"],
            "zero_hedge_ratio": res["zero_hedge_ratio"],
            "escalation_ratio": res["escalation_ratio"],
            "avg_hedge": res["avg_hedge"],
            "log": log.to_dict(orient="records"),
        }, indent=2, default=str))
        return 0

    _say(f"\nuniverse          {res['universe']} / {res['timeframe']}")
    _say(f"spec              {spec.name}")
    _say(f"decision backend  {res['backend']}")
    _say(f"decision calls    {res['decision_calls']} "
          f"(cost ${res['decision_cost_usd']})")
    _say(f"bars              {res['n_bars']}")
    _say(f"avg hedge         {res['avg_hedge']}")
    _say(f"ZERO-hedge share  {res['zero_hedge_ratio']}   <-- the headline number")
    _say(f"veto share        {res['veto_ratio']}")
    _say(f"escalation share  {res['escalation_ratio']}")

    if len(log):
        cols = ["date", "regime", "vol_ratio", "drawdown", "quant_hedge",
                "decision_hedge", "vetoed", "veto_reason"]
        with pd_option():
            _say("\n--- journal (last %d bars) ---" % args.tail)
            _say(log[cols].tail(args.tail).to_string(index=False))

    path = os.path.join(args.out, f"agent_{res['universe']}_{res['timeframe']}.csv")
    log.to_csv(path, index=False)
    _say(f"\n[saved] {path}")
    return 0


class pd_option:
    """Small context manager so we do not import pandas at module top."""

    def __enter__(self):
        import pandas as pd
        pd.set_option("display.width", 200)
        pd.set_option("display.max_columns", 40)
        return pd

    def __exit__(self, *a):
        return False


if __name__ == "__main__":
    raise SystemExit(main())