"""The worked example from `docs/DECISION-LAYER.md`, as a real module.

The guide is only worth anything if its example runs. This file IS that
example, and `tests/test_decision_layer.py` tests it. If the guide and
this file ever disagree, this file is the one that is true.

The idea: you have measured, offline, how often each regime label led to
a drawdown. That table IS the model. At run time we look up the current
regime and return the measured frequency, rather than asking anyone.

This is the shape of a *calibrated* backend: the probabilities are
frequencies from data, so a threshold on them means something.
"""

from __future__ import annotations

from typing import Any

from hedging_agent.decision import (
    Choice,
    Decision,
    DecisionBackend,
    Noul,
    Question,
    Score,
    register_backend,
)

# Measured frequencies, from a labelled out-of-sample set.
# Keys are the regimes this repository uses; see hedging_agent/regime.py.
P_DRAWDOWN_AFTER = {
    "crash": 0.62,
    "crash_rebound": 0.41,
    "distribution": 0.55,
    "expansion": 0.07,
    "high_vol_chop": 0.49,
    "trend_up": 0.11,
}
# The prior, for a regime we have not seen.
P_PRIOR = 0.25


class TableBackend:
    """Returns measured frequencies. Deterministic, dependency-free."""

    name = "table"
    # A calibrated backend: its probabilities are frequencies, so a
    # threshold on them is meaningful. LocalBackend sets this False.
    uncalibrated = False

    def decide(self, state: Any, questions: dict[str, Question]) -> dict[str, Decision]:
        regime = str((state or {}).get("regime", ""))
        p = P_DRAWDOWN_AFTER.get(regime, P_PRIOR)

        out: dict[str, Decision] = {}
        for key, q in questions.items():
            if isinstance(q, Noul):
                out[key] = Decision(
                    question=key, kind="noul", value=p,
                    probabilities={"true": p, "false": 1.0 - p},
                    confidence=abs(2 * p - 1),
                    model="table-v1", backend=self.name,
                )
            elif isinstance(q, Choice):
                # Put the mass on the option the measured frequency implies.
                # "none" (no hedge) is right when the drawdown rate is low.
                if key == "hedge_sizing":
                    if p >= 0.5:
                        probs = {"none": 0.05, "light": 0.25,
                                 "moderate": 0.35, "heavy": 0.35}
                    elif p >= 0.25:
                        probs = {"none": 0.30, "light": 0.40,
                                 "moderate": 0.25, "heavy": 0.05}
                    else:
                        probs = {"none": 0.75, "light": 0.20,
                                 "moderate": 0.04, "heavy": 0.01}
                    value = max(probs, key=probs.get)
                else:
                    probs = {"none": 1.0 - p, "risk": p}
                    value = "risk" if p > 0.5 else "none"
                ranked = sorted(probs.values(), reverse=True)
                conf = round(ranked[0] - ranked[1], 4) if len(ranked) > 1 else 0.0
                out[key] = Decision(
                    question=key, kind="choice", value=value,
                    probabilities=probs, confidence=conf,
                    model="table-v1", backend=self.name,
                )
            else:  # Score
                # Map the probability onto the rubric's positions.
                n = max(len(q.criteria), 1)
                idx = min(int(p * n), n - 1)
                probs = {str(i): (0.9 if i == idx else 0.1 / max(n - 1, 1))
                         for i in range(n)}
                out[key] = Decision(
                    question=key, kind="score", value=float(idx),
                    probabilities=probs, confidence=0.6,
                    model="table-v1", backend=self.name,
                )
        return out


# Registration is deliberately NOT done at import time. A module that
# registers a global as a side effect of being imported changes what every
# other module in the process sees, and a test that imports it cannot
# undo that without help. Call it explicitly instead:

#     from table_backend_example import TableBackend
#     register_backend("table", TableBackend)
