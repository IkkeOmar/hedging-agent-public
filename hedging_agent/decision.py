"""The decision layer: a seam, not a vendor.

The decision layer is the part of this system that makes the fast
decisions. The LLM does the slow thinking -- searching, drafting
candidates, writing notes -- and the decision model answers a small,
fixed set of questions at every bar. It is **not** an LLM: it returns
typed answers with calibrated probabilities, and you can threshold on a
probability.

This module ships:

* the question primitives (`Choice`, `Score`, `Noul`);
* the `Decision` dataclass, with the provenance you need to audit it;
* the `DecisionBackend` **Protocol** -- the contract;
* `LocalBackend`, a keyless, deterministic fallback;
* `build_backend`, which resolves the backend by name.

**There is no hosted service here.** No API key, no account, no network
call, no third party. That is deliberate: a repository whose central
promise is "no key, no account, no network" should not contain a vendor
that needs a key. If you want a hosted decision model, or your own, write
one -- the protocol is the contract, and
[`docs/DECISION-LAYER.md`](../docs/DECISION-LAYER.md) is a worked
example you can copy.

**The hard rule, enforced in `policy.py`:** the decision layer may veto
the quantitative read, and it may scale it. It may **not invent a hedge
where the quantitative read says zero**, unless it is genuinely confident.

> *"An LLM that can raise a hedge from nothing is an LLM that can sell you
> an overlay."*
"""

from __future__ import annotations

import json
import math
import os
from dataclasses import dataclass, field
from typing import Any, Protocol

from .config import DecisionConfig

__all__ = [
    "Choice",
    "Score",
    "Noul",
    "Question",
    "Decision",
    "DecisionBackend",
    "LocalBackend",
    "build_backend",
    "hedge_questions",
    "register_backend",
    "unregister_backend",
    "available_backends",
]


# --------------------------------------------------------------- primitives
@dataclass
class Choice:
    """A question with named options. The answer is one option plus a
    probability for each."""

    instructions: str
    criteria: dict[str, str | None]


@dataclass
class Score:
    """A question scored on an ordered rubric. The answer is a position on
    the rubric plus a probability for each position."""

    instructions: str
    criteria: list[str]


@dataclass
class Noul:
    """A yes/no question. The answer is a boolean plus the probability of
    'true'."""

    instructions: str
    criteria: dict[str, str] | None = None


Question = Choice | Score | Noul


def _serialise(q: Question) -> dict[str, Any]:
    """The wire form, for a backend that talks to a service."""
    if isinstance(q, Choice):
        return {"type": "choice", "instructions": q.instructions, "criteria": q.criteria}
    if isinstance(q, Score):
        return {"type": "score", "instructions": q.instructions, "criteria": q.criteria}
    return {"type": "noul", "instructions": q.instructions, "criteria": q.criteria}


@dataclass
class Decision:
    """One resolved answer, plus the provenance you need to audit it.

    `probabilities` is the part that matters. A decision without a
    distribution over its options cannot be thresholded, and a threshold
    on a bare label is just an opinion with extra steps.
    """

    question: str
    kind: str
    value: float | str
    probabilities: dict[str, float] = field(default_factory=dict)
    confidence: float | None = None
    cost_usd: float = 0.0
    input_tokens: int = 0
    model: str = ""
    backend: str = ""

    def p(self, key: str) -> float:
        """The probability of one option."""
        return float(self.probabilities.get(key, 0.0))


class DecisionBackend(Protocol):
    """The contract a decision layer must satisfy.

    A backend takes the current state and a set of questions, and returns
    one `Decision` per question, keyed by the question's name.

    The protocol is the whole point of this module: it is what lets you
    replace the decision layer without touching anything else. Implement
    it, register the implementation, and the pipeline will use yours.
    """

    name: str

    def decide(self, state: Any, questions: dict[str, Question]) -> dict[str, Decision]: ...


# ------------------------------------------------------------------ backends
class LocalBackend:
    """A keyless, deterministic fallback.

    It reads keywords out of the state and turns them into a
    distribution. It is honest about what it is: **it is not calibrated.**
    A probability from this backend is a score, not a frequency, and any
    threshold taken from it must be marked as such -- which is why the
    policy treats it differently from a real model.

    The `uncalibrated` marker below is not decoration. The whole argument
    of this project is that uncalibrated confidence is a lie.
    """

    name = "local"
    uncalibrated = True

    def decide(self, state: Any, questions: dict[str, Question]) -> dict[str, Decision]:
        text = json.dumps(state, default=str) if not isinstance(state, str) else state
        low = text.lower()
        out: dict[str, Decision] = {}

        for key, q in questions.items():
            if isinstance(q, Noul):
                p = _local_noul(q.instructions, low)
                out[key] = Decision(key, "noul", round(p, 4), {"true": round(p, 4)},
                                    confidence=round(min(p, 1 - p) * 2, 4),
                                    model="local-logistic", backend=self.name)
            elif isinstance(q, Choice):
                scores = {k: _local_choice(k, desc or k, low) for k, desc in q.criteria.items()}
                tot = sum(scores.values()) or 1.0
                probs = {k: round(v / tot, 4) for k, v in scores.items()}
                best = max(probs, key=probs.get)
                ranked = sorted(probs.values(), reverse=True)
                conf = round(ranked[0] - ranked[1], 4) if len(ranked) > 1 else 0.0
                out[key] = Decision(key, "choice", best, probs, confidence=conf,
                                    model="local-logistic", backend=self.name)
            else:
                # the score primitive: map regime keywords onto the rubric
                pos = sum(1 for w in ("expansion", "uptrend", "quiet", "calm") if w in low)
                neg = sum(1 for w in ("crash", "distribution", "drawdown", "collapse", "stress")
                          if w in low)
                n = max(len(q.criteria), 1)
                mid = (n - 1) / 2
                s = mid + (pos - neg) * 0.8
                s = float(max(0.0, min(n - 1 + 1e-6, s)))
                out[key] = Decision(key, "score", round(s, 4),
                                    {str(i): round(1 - min(abs(i - s), 1.0), 4)
                                     for i in range(n)},
                                    confidence=0.5, model="local-logistic", backend=self.name)
        return out


def _local_noul(instructions: str, low: str) -> float:
    """A crude keyword scorer. Honest about its crudeness."""
    text = instructions.lower()
    risk_words = ("hedge", "crash", "risk", "stress", "downside", "loss", "drawdown", "draw down")
    calm_words = ("expansion", "uptrend", "calm", "quiet", "advance", "rally")
    if "veto" in text or "no hedge" in text or "do nothing" in text:
        c, r = calm_words, risk_words
    else:
        c, r = risk_words, calm_words
    hits_c = sum(w in low for w in c)
    hits_r = sum(w in low for w in r)
    return 1.0 / (1.0 + math.exp(-(hits_c - hits_r) * 1.2))


def _local_choice(key: str, desc: str, low: str) -> float:
    key_w = key.replace("_", " ").lower()
    desc_w = desc.lower()
    score = 1.0
    if key_w in low:
        score += 4.0
    for tok in key_w.split():
        if tok in low:
            score += 0.7
    for tok in desc_w.split()[:6]:
        if len(tok) > 4 and tok in low:
            score += 0.25
    return score


# ------------------------------------------------------------------ registry
# Third-party backends register themselves here. That is how a reader's own
# decision layer becomes usable without editing this file: import it, call
# register_backend, and it is available by name.
_REGISTRY: dict[str, Any] = {"local": LocalBackend}


def register_backend(name: str, factory: Any) -> None:
    """Register a decision backend under a name.

    `factory` is called with no arguments and must return an object
    satisfying `DecisionBackend`.
    """
    if not callable(factory):
        raise TypeError(f"backend factory for {name!r} is not callable")
    _REGISTRY[name] = factory


def available_backends() -> list[str]:
    """The names `build_backend` understands."""
    return sorted(_REGISTRY)


def unregister_backend(name: str) -> None:
    """Remove a registered backend.

    Registration is global state, so a test that registers a backend has
    to be able to take it out again -- otherwise it changes what every
    LATER test sees. That is not a theoretical hazard: the worked example
    in the guide registers "table" on import, and without this a test
    asserting that "auto" falls back to the local backend started failing
    because of a test that ran earlier.
    """
    _REGISTRY.pop(name, None)


# ------------------------------------------------------------------ factory
def build_backend(cfg: DecisionConfig | None = None) -> DecisionBackend:
    """Resolve the backend by name.

    `backend="auto"` is the default, and it means: **use a registered
    backend if one works, otherwise fall back to the keyless local
    backend.** With nothing registered -- which is the case in the public
    copy -- that is simply the local backend, and it needs no key, no
    account and no network.

    A backend named explicitly is called, and its error is allowed to
    propagate. If you asked for a backend by name, you want to know that
    it could not be built, rather than being silently handed a different
    one.
    """
    cfg = cfg or DecisionConfig()
    name = (cfg.backend or "auto").lower()

    if name == "auto":
        for candidate in available_backends():
            if candidate == "local":
                continue
            try:
                return _REGISTRY[candidate]()
            except RuntimeError:
                # the backend could not be built -- most often because its
                # key is not set. Fall through to the next one.
                continue
        return LocalBackend()

    factory = _REGISTRY.get(name)
    if factory is None:
        raise KeyError(
            f"unknown decision backend {name!r}. Available: "
            f"{', '.join(available_backends())}. Register yours with "
            f"hedging_agent.decision.register_backend(name, factory)."
        )
    return factory()


# --------------------------------------------------------- hedge question set
def hedge_questions() -> dict[str, Question]:
    """The standing question set the hedge agent asks every bar.

    Batched into one call on purpose: the questions are independent, they
    run in parallel, and they share the state cost. Four separate calls
    would cost 4x.
    """
    return {
        "should_hedge": Noul(
            instructions=(
                "Should a dynamic hedging overlay add protection now? Hedging is "
                "correct when forward risk is elevated: volatility expanding, "
                "trend breaking down, drawdown deepening, or correlation rising. "
                "Hedging is wrong when the tape is calm and expanding, because a "
                "constant hedge in an uptrend caps the upside you would otherwise "
                "keep."
            ),
            criteria={
                "true": "forward risk is elevated and protection is worth its cost",
                "false": "protection would cost more upside than it saves in risk",
            },
        ),
        "primary_risk": Choice(
            instructions="What is the dominant source of risk in the current state?",
            criteria={
                "volatility_expansion": "realised volatility is rising across scales",
                "trend_break": "trend has broken down on the majority of scales",
                "correlation_spike": "assets are moving together, diversification has failed",
                "drawdown": "the portfolio is already in a deepening drawdown",
                "none": "no dominant risk; the portfolio is behaving normally",
            },
        ),
        "risk_severity": Score(
            instructions="How severe is the risk right now?",
            criteria=[
                "no measurable risk; volatility compressed and trend intact",
                "mild elevation; volatility drifting up on the short scales",
                "moderate; half the scales disagree about direction",
                "elevated; volatility expanding and trend rolling over",
                "severe; multiple scales confirm a fast, directional move down",
            ],
        ),
        "hedge_sizing": Choice(
            instructions="How much of the portfolio should the overlay hedge?",
            criteria={
                "none": "no hedge; the veto case, risk is not compensated",
                "light": "hedge a small fraction, under one quarter of notional",
                "moderate": "hedge a substantial fraction, roughly one quarter to one half",
                "heavy": "hedge most of notional, above one half",
            },
        ),
        "confidence_gate": Noul(
            instructions=(
                "Is the evidence strong enough to act autonomously without a human "
                "review? Answer no if the scales disagree or the state is ambiguous."
            ),
            criteria={
                "true": "scales agree and the signal is decisive",
                "false": "scales disagree, or the evidence is ambiguous",
            },
        ),
    }
