"""LLM strategy generation.

The LLM does not trade and the LLM does not decide whether to trade. It writes
candidate hedge rule sets, and every candidate has to survive the promotion gate
in stats.py before it is allowed near the decision layer.

That separation is the entire thesis. Anyone shipping "an LLM that picks hedges"
without a gate is shipping the critique, not the agent.

Implemented providers, in preference order:
  anthropic | openrouter | ollama | none
`none` emits deterministic template candidates so the pipeline is runnable with
zero API keys.
"""

from __future__ import annotations

import json
import os
import re
import textwrap
from dataclasses import dataclass, field, asdict

# --------------------------------------------------------------- spec model
@dataclass
class HedgeStrategySpec:
    """A hedge strategy expressed as parameters, not as code.

    Parameters are ranges, not points. The market regime changes in a fractal
    environment, so a strategy that has a single hard-coded threshold has
    already lost to the market. The search space is the strategy.
    """

    name: str
    description: str
    # the specific failure this hedge exists to absorb. Required, because a
    # hedge without a named failure mode is a cost centre with a UI.
    failure_mode: str = "unspecified"
    # target volatility the overlay defends against, annualised
    target_vol: float = 0.18
    # sensitivity of hedge size to the vol ratio
    vol_gamma: float = 0.5
    vol_cap_mult: float = 2.5
    # drawdown governor
    dd_trigger: float = -0.05
    dd_full: float = -0.15
    dd_max_mult: float = 2.0
    # regime consensus requirement
    consensus_required: float = 0.5
    veto_on_expansion: bool = True
    max_hedge_fraction: float = 0.75
    max_daily_turnover: float = 0.25
    # hedge instrument / proxy used to express the hedge
    hedge_instrument: str = "short_beta"

    # cost model, in basis points of notional traded. These are POLICY knobs,
    # not constants: a hedge strategy that cannot state what it assumes about
    # costs is not auditable, and a cost-sensitivity sweep is impossible
    # without them. commission 1 + spread 2 = 3bps round trip, plus borrow.
    commission_bps: float = 1.0
    spread_bps: float = 2.0
    borrow_bps_annual: float = 10.0
    # rebalance cadence in bars. 0 = re-evaluate every bar, act only on change.
    rebalance_every_bars: int = 1
    # parameters the LLM is allowed to vary, as (low, high) search bounds
    search_space: dict[str, tuple[float, float]] = field(default_factory=dict)

    def as_dict(self) -> dict:
        return asdict(self)


# The search space is the actual degree of freedom in this project. It is
# deliberately large enough that picking the best point in it is a
# multiple-testing problem, which is why DSR gets a real n_trials.
BASE_SEARCH_SPACE: dict[str, tuple[float, float]] = {
    "target_vol": (0.10, 0.30),
    "vol_gamma": (0.1, 1.2),
    "vol_cap_mult": (1.5, 3.5),
    "dd_trigger": (-0.02, -0.10),
    "dd_full": (-0.08, -0.25),
    "dd_max_mult": (1.2, 3.0),
    "consensus_required": (0.34, 0.84),
    "max_hedge_fraction": (0.30, 0.90),
    "max_daily_turnover": (0.08, 0.40),
}


def validate_spec(spec: HedgeStrategySpec) -> list[str]:
    """Reject structurally broken specs before they burn a backtest."""
    problems: list[str] = []
    if not 0.05 <= spec.target_vol <= 0.60:
        problems.append(f"target_vol {spec.target_vol} outside [0.05, 0.60]")
    if spec.vol_gamma < 0:
        problems.append("vol_gamma must be >= 0")
    if spec.dd_trigger <= spec.dd_full:
        problems.append(
            "dd_trigger must be greater (less negative) than dd_full; "
            "the ladder ramps from dd_trigger down to dd_full"
        )
    if spec.max_daily_turnover <= 0:
        problems.append("max_daily_turnover must be > 0 or the hedge can never move")
    if not 0 < spec.consensus_required <= 1:
        problems.append("consensus_required must be in (0, 1]")
    if spec.veto_on_expansion is False:
        problems.append(
            "veto_on_expansion=False turns the overlay into a constant hedge, "
            "which is the exact failure mode this project exists to avoid"
        )
    return problems


# ------------------------------------------------------------ prompt design
GENERATION_PROMPT = textwrap.dedent("""
    You are a systematic-risk researcher writing hedge-policy candidates for a
    portfolio overlay. You do NOT write trading signals. You write the RISK
    side: how much protection to hold, and when to hold none at all.

    Hard constraints for every candidate you produce:
    1. The target hedge is allowed to be 0. A candidate that always hedges is
       rejected automatically, because a constant overlay taxes every good day
       to insure the drawdowns you already predicted.
    2. Parameters are search RANGES, not fixed values. Market regimes are
       fractal and change; a single hard-coded threshold cannot survive that.
    3. veto_on_expansion must be true. In a calm, expanding, intact tape, the
       hedge is a pure carry cost and the correct position is zero.
    4. Turnover must be capped. The dominant hidden cost in overlay strategies
       is the cost of re-hedging, not the cost of holding the hedge.
    5. Every candidate must state the mechanism in one sentence, and the
       failure mode it is expected to protect against.

    Return ONLY a JSON array. Each element:
    {{
      "name": "short_snake_case_name",
      "description": "one sentence on the mechanism",
      "failure_mode": "what it protects against",
      "target_vol": 0.0-0.30,
      "vol_gamma": 0.1-1.2,
      "vol_cap_mult": 1.5-3.5,
      "dd_trigger": -0.02 to -0.10,
      "dd_full": -0.08 to -0.25,
      "dd_max_mult": 1.2-3.0,
      "consensus_required": 0.34-0.84,
      "veto_on_expansion": true,
      "max_hedge_fraction": 0.30-0.90,
      "max_daily_turnover": 0.08-0.40,
      "hedge_instrument": "short_beta|put_overlay|cash_buffer|vol_aware_short"
    }}

    Produce {n} genuinely different candidates. Vary the MECHANISM, not just
    the numbers: one vol-scaling, one drawdown-laddered, one consensus-gated,
    one asymmetric (fast-on/slow-off), one horizon-weighted. Do not return N
    copies of one idea with jittered constants.
""").strip()


def _postprocess(raw: str, n: int) -> list[HedgeStrategySpec]:
    """Parse LLM output into specs. Malformed candidates are dropped, not
    silently repaired -- a repaired candidate is a candidate nobody audited."""
    text = raw.strip()
    if "```" in text:
        text = re.sub(r"^```[a-zA-Z]*\n?|```$", "", text, flags=re.M).strip()
    start = text.find("[")
    end = text.rfind("]")
    if start == -1 or end == -1:
        return []
    try:
        items = json.loads(text[start : end + 1])
    except json.JSONDecodeError:
        return []
    if not isinstance(items, list):
        return []

    specs: list[HedgeStrategySpec] = []
    for it in items[:n]:
        if not isinstance(it, dict):
            continue
        it.setdefault("search_space", dict(BASE_SEARCH_SPACE))
        allowed = {f for f in HedgeStrategySpec.__dataclass_fields__}
        clean = {k: v for k, v in it.items() if k in allowed}
        clean["search_space"] = it.get("search_space", dict(BASE_SEARCH_SPACE))
        try:
            spec = HedgeStrategySpec(**clean)
        except TypeError:
            continue
        if validate_spec(spec):
            continue
        specs.append(spec)
    return specs


# ------------------------------------------------------------- providers
def generate_candidates(
    n: int = 5,
    provider: str | None = None,
    model: str | None = None,
    context: dict | None = None,
) -> tuple[list[HedgeStrategySpec], dict]:
    """Return (specs, meta). meta records exactly what produced the specs,
    because an unlabelled candidate list is not reproducible."""
    provider = (provider or os.environ.get("LLM_PROVIDER") or "none").lower()
    context = context or {}
    prompt = GENERATION_PROMPT.replace("{n}", str(n))
    if context:
        prompt += "\n\nObserved context for this generation round:\n" + json.dumps(
            context, indent=2, default=str
        )[:4000]

    meta = {"provider": provider, "model": model, "requested": n,
            "rejected": 0, "source": None, "error": None}

    if provider == "anthropic":
        return _anthropic(prompt, n, model or "claude-sonnet-4-6", meta)
    if provider == "openrouter":
        return _openai_compatible(
            prompt, n, model or "anthropic/claude-sonnet-4.6",
            os.environ.get("OPENROUTER_API_KEY"),
            "https://openrouter.ai/api/v1", meta,
        )
    if provider == "ollama":
        return _openai_compatible(
            prompt, n, model or os.environ.get("OLLAMA_MODEL", "qwen3-coder-next"),
            os.environ.get("OLLAMA_API_KEY", "ollama"),
            os.environ.get("OLLAMA_BASE_URL", "http://127.0.0.1:11434/v1"), meta,
        )
    specs = template_candidates(n)
    meta["source"] = "template"
    return specs, meta


def _anthropic(prompt: str, n: int, model: str, meta: dict):
    try:
        import anthropic
    except ImportError:
        meta["error"] = "anthropic package not installed"
        return template_candidates(n), meta
    key = os.environ.get("ANTHROPIC_API_KEY")
    if not key:
        meta["error"] = "ANTHROPIC_API_KEY unset"
        return template_candidates(n), meta
    try:
        client = anthropic.Anthropic(api_key=key)
        resp = client.messages.create(
            model=model, max_tokens=4000,
            system="You generate hedge-policy candidates as strict JSON. No prose.",
            messages=[{"role": "user", "content": prompt}],
        )
        raw = "".join(b.text for b in resp.content if getattr(b, "type", "") == "text")
        specs = _postprocess(raw, n)
        meta.update(source="anthropic", model=model, produced=len(specs))
        return (specs or template_candidates(n)), meta
    except Exception as e:  # provider failures must degrade, not crash the loop
        meta["error"] = f"{type(e).__name__}: {e}"
        return template_candidates(n), meta


def _openai_compatible(prompt: str, n: int, model: str, key, base_url: str, meta: dict):
    try:
        import requests
    except ImportError:
        meta["error"] = "requests not installed"
        return template_candidates(n), meta
    if not key:
        meta["error"] = "api key unset"
        return template_candidates(n), meta
    try:
        r = requests.post(
            f"{base_url.rstrip('/')}/chat/completions",
            headers={"Authorization": f"Bearer {key}"},
            json={
                "model": model, "temperature": 0.8,
                "messages": [
                    {"role": "system",
                     "content": "You generate hedge-policy candidates as strict JSON. No prose."},
                    {"role": "user", "content": prompt},
                ],
            },
            timeout=120,
        )
        if r.status_code != 200:
            meta["error"] = f"HTTP {r.status_code}: {r.text[:200]}"
            return template_candidates(n), meta
        raw = r.json()["choices"][0]["message"]["content"]
        specs = _postprocess(raw, n)
        meta.update(source=base_url, model=model, produced=len(specs))
        return (specs or template_candidates(n)), meta
    except Exception as e:
        meta["error"] = f"{type(e).__name__}: {e}"
        return template_candidates(n), meta


# --------------------------------------------- keyless template candidates
def template_candidates(n: int = 5) -> list[HedgeStrategySpec]:
    """Deterministic, mechanism-diverse fallbacks. Not a placeholder: these
    are five genuinely different hedges that the gate can actually accept or
    reject, so the pipeline is meaningful with zero keys."""
    bank: list[HedgeStrategySpec] = [
        HedgeStrategySpec(
            name="vol_scaled_beta_hedge",
            description="Scale a beta hedge inversely to realised volatility across all scales.",
            failure_mode="a volatility regime that is expanding faster than the portfolio can absorb",
            target_vol=0.16, vol_gamma=0.6, vol_cap_mult=2.2,
            dd_trigger=-0.04, dd_full=-0.12, dd_max_mult=1.8,
            consensus_required=0.5, max_hedge_fraction=0.60,
            max_daily_turnover=0.20,
            search_space=dict(BASE_SEARCH_SPACE),
        ),
        HedgeStrategySpec(
            name="drawdown_ladder",
            description="A three-rung ladder that adds hedge depth as the drawdown deepens.",
            failure_mode="a slow bleed that shallow overlays never catch",
            target_vol=0.20, vol_gamma=0.3, vol_cap_mult=2.0,
            dd_trigger=-0.03, dd_full=-0.18, dd_max_mult=2.6,
            consensus_required=0.45, max_hedge_fraction=0.75,
            max_daily_turnover=0.25,
            search_space=dict(BASE_SEARCH_SPACE),
        ),
        HedgeStrategySpec(
            name="consensus_gated_trend",
            description="Hedge only when a strict majority of fractal scales agree the trend is breaking.",
            failure_mode="false alarms from a single noisy timeframe",
            target_vol=0.18, vol_gamma=0.7, vol_cap_mult=2.6,
            dd_trigger=-0.05, dd_full=-0.15, dd_max_mult=2.0,
            consensus_required=0.70, max_hedge_fraction=0.55,
            max_daily_turnover=0.15,
            search_space=dict(BASE_SEARCH_SPACE),
        ),
        HedgeStrategySpec(
            name="asymmetric_fast_on_slow_off",
            description="Add hedge quickly on the vol breach, release it slowly so the hedge does not churn.",
            failure_mode="turnover cost from re-hedging on every vol oscillation",
            target_vol=0.17, vol_gamma=0.9, vol_cap_mult=2.4,
            dd_trigger=-0.04, dd_full=-0.14, dd_max_mult=2.2,
            consensus_required=0.55, max_hedge_fraction=0.70,
            max_daily_turnover=0.10,
            search_space=dict(BASE_SEARCH_SPACE),
        ),
        HedgeStrategySpec(
            name="horizon_weighted_overlay",
            description="Weight short-horizon volatility more heavily, so intraday stress triggers faster than swing stress.",
            failure_mode="a slow regime shift that long-lookback detectors miss for weeks",
            target_vol=0.19, vol_gamma=0.55, vol_cap_mult=2.8,
            dd_trigger=-0.06, dd_full=-0.16, dd_max_mult=2.1,
            consensus_required=0.50, max_hedge_fraction=0.65,
            max_daily_turnover=0.30,
            search_space=dict(BASE_SEARCH_SPACE),
        ),
        HedgeStrategySpec(
            name="cash_buffer_preference",
            description="Express the hedge partly as cash rather than short beta, accepting the drag to reduce gap risk.",
            failure_mode="overnight gap risk that a same-session short cannot capture",
            target_vol=0.15, vol_gamma=0.45, vol_cap_mult=1.9,
            dd_trigger=-0.03, dd_full=-0.13, dd_max_mult=1.7,
            consensus_required=0.48, max_hedge_fraction=0.50,
            max_daily_turnover=0.18,
            hedge_instrument="cash_buffer",
            search_space=dict(BASE_SEARCH_SPACE),
        ),
    ]
    return bank[:n]