# Build your own decision layer

The decision layer is the part of this system that makes the **fast**
decisions. The LLM does the slow thinking — searching, drafting candidate
strategies, writing notes — and the decision model answers a small, fixed
set of questions at every bar.

It is **not** an LLM. It returns typed answers with probabilities, and you
can threshold on a probability. That difference is the whole point: a
language model will produce a confident-looking answer to anything, and a
confident-looking answer is indistinguishable from a good one at the
interface.

This guide shows you how to write your own and plug it in. The whole
contract is one Protocol with one method.

---

## 1. The protocol

```python
class DecisionBackend(Protocol):
    name: str

    def decide(self, state: Any, questions: dict[str, Question]) -> dict[str, Decision]: ...
```

That is the entire contract. It lives in
[`hedging_agent/decision.py`](../hedging_agent/decision.py) and it is
copied here verbatim so you can check it against the installed version.

### The question types

```python
@dataclass
class Choice:          # pick one of several named options
    instructions: str
    criteria: dict[str, str | None]

@dataclass
class Score:           # a position on an ordered rubric
    instructions: str
    criteria: list[str]

@dataclass
class Noul:            # yes or no
    instructions: str
    criteria: dict[str, str] | None = None
```

### The answer

```python
@dataclass
class Decision:
    question: str
    kind: str                    # "choice" | "score" | "noul"
    value: float | str           # the chosen option, or the score
    probabilities: dict[str, float] = field(default_factory=dict)
    confidence: float | None = None
    cost_usd: float = 0.0
    input_tokens: int = 0
    model: str = ""
    backend: str = ""

    def p(self, key: str) -> float:
        """The probability of one option."""
        return float(self.probabilities.get(key, 0.0))
```

**`probabilities` is the part that matters.** A decision without a
distribution over its options cannot be thresholded, and a threshold on a
bare label is just an opinion with extra steps.

---

## 2. The hard rule

The decision layer may **veto** the quantitative read, and it may
**scale** it. It may **not invent a hedge where the quantitative read says
zero**, unless it is genuinely confident.

This is enforced in `hedging_agent/policy.py`, not left to your
judgement:

> *"An LLM that can raise a hedge from nothing is an LLM that can sell
> you an overlay."*

Concretely: the quantitative read produces a hedge notional from the
regime state. Your decision layer can push that down to zero (a veto), or
scale it. What it cannot do is raise it from zero to something meaningful
unless its confidence clears the escalation threshold.

---

## 3. A complete worked example

Copy this, change what you like, register it. It is deliberately
dependency-free and deterministic, so you can test it without a network or
a key — and it is the shape a real backend should have: it returns a
*distribution*, not a label.

```python
"""A decision backend that reads a calibrated probability table.

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


# Register it, so build_backend can find it by name.
register_backend("table", TableBackend)
```

### Plugging it in

```python
from hedging_agent.config import DecisionConfig
from hedging_agent.decision import build_backend

backend = build_backend(DecisionConfig(backend="table"))
```

Or, from the command line, the backend name comes from the config. The
default (`auto`) uses a registered backend when one works and falls back
to the keyless local backend otherwise.

---

## 4. Test it

A backend that returns a distribution is easy to test, and it should be.
Here is the shape of the test — note that it asserts on the *probabilities*,
not just on the chosen label, because a label is the part that is easy to
get right and hard to get right *usefully*.

```python
def test_table_backend_returns_a_distribution():
    from hedging_agent.decision import hedge_questions, register_backend
    from table_backend import TableBackend

    register_backend("table", TableBackend)
    backend = TableBackend()
    out = backend.decide({"regime": "distribution"}, hedge_questions())

    # every question got an answer
    assert set(out) == set(hedge_questions())

    # the probabilities are a distribution
    for key, d in out.items():
        assert set(d.probabilities), f"{key} has no probabilities"
        assert abs(sum(d.probabilities.values()) - 1.0) < 0.02, key

    # and the answer follows the measured frequency
    assert out["should_hedge"].p("true") == 0.55
```

---

## 5. The local fallback is not calibrated

The backend that ships, `LocalBackend`, reads keywords out of the state
and turns them into a distribution. It is deterministic and it needs
nothing — but it is **not calibrated**: its probabilities are scores, not
frequencies, and it marks itself `uncalibrated = True` so the policy can
treat it differently.

**Any threshold you take from it must be marked as such.** That is the
whole argument of this project in one line: uncalibrated confidence is a
lie, and a system that says so about itself is worth more than one that
does not.

If you want probabilities you can threshold on, measure them, like
`TableBackend` does above.

---

## 6. What a backend may and may not do

**May:**

- veto the quantitative read, when its probabilities say the risk is not
  compensated;
- scale the hedge within the caps in `HedgeConfig`;
- abstain, by returning a low confidence on `confidence_gate` — the policy
  will hold the previous hedge and mark the bar as an escalation;
- report its own cost, so a caller can see what the decisions cost.

**May not:**

- raise a hedge from zero without genuine confidence;
- return a bare label with no distribution;
- be silently substituted for another backend — if you ask for one by
  name and it cannot be built, you get an error, not a different model;
- make the harness's arithmetic decisions. The hedge ratio, the sizing,
  the costs and the gates are never a model's job.
