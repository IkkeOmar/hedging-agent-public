"""The hedge policy: regime in, hedge-notional out.

Design stance, which is the whole thesis of this project:

    A constant hedge overlay is a bet that you know the future. It pays for
    itself in the drawdowns you predicted and quietly taxes every good day in
    between. Almost every "low volatility, high Sharpe" result in the overlay
    literature is that trade, and it is rarely reported net of the cost.

    So the correct target hedge is frequently ZERO, and a hedge engine that has
    never returned zero has not been tested.

Every multiplier below is computed from state. There is no fixed hedge ratio
anywhere in this file.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict

import numpy as np

from .config import HedgeConfig
from .regime import Regime, RegimeState
from .decision import Decision


@dataclass
class HedgeAction:
    date: object
    target_hedge: float          # fraction of notional to hedge, 0..max
    prev_hedge: float
    regime: str
    vol_ratio: float
    drawdown: float
    multiplier: dict[str, float]
    decision: dict | None        # provenance of the decision-layer call
    vetoed: bool
    veto_reason: str | None
    escal: bool

    def as_dict(self) -> dict:
        return asdict(self)


def vol_multiplier(vol_ratio: float, cfg: HedgeConfig) -> float:
    """Inverse-vol response. Below target -> de-hedge, above target -> add."""
    if vol_ratio <= 0:
        return cfg.vol_floor_mult
    m = 1.0 + cfg.vol_gamma * (vol_ratio - 1.0)
    return float(np.clip(m, cfg.vol_floor_mult, cfg.vol_cap_mult))


def drawdown_multiplier(drawdown: float, cfg: HedgeConfig) -> float:
    """Ramp only after real damage. Hedges that activate before the drawdown
    are just leverage with extra steps."""
    if drawdown >= cfg.dd_trigger:
        return 1.0
    span = cfg.dd_trigger - cfg.dd_full
    if span <= 0:
        return cfg.dd_max_mult
    frac = (cfg.dd_trigger - drawdown) / span
    return float(1.0 + (cfg.dd_max_mult - 1.0) * np.clip(frac, 0.0, 1.0))


def consensus_multiplier(state: RegimeState, cfg: HedgeConfig) -> float:
    """Act only when enough scales agree. Disagreement is a reason to do less,
    not to guess."""
    if state.consensus >= cfg.consensus_required:
        return 1.0
    # scale the response down by the degree of disagreement
    return float(max(0.0, state.consensus / max(cfg.consensus_required, 1e-9)))


def veto(state: RegimeState, drawdown: float, cfg: HedgeConfig) -> tuple[bool, str | None]:
    """The veto. This is the mechanism that prevents a constant hedge.

    In an expansion regime with an intact trend, the hedge is priced as pure
    cost. Return zero and say so in the log.

    The veto is SUSPENDED on the crash-rebound state. Daniel & Moskowitz (2015)
    show the documented crash is high volatility + recent decline + a sharp
    positive burst -- i.e. the tape that superficially looks like "trend is up,
    release the hedge". Releasing protection in that bar is the mechanism-level
    failure this function exists to prevent.
    """
    if not cfg.veto_on_expansion:
        return False, None
    if state.rebound or state.label is Regime.CRASH_REBOUND:
        return False, (
            "veto suspended: crash-rebound state (elevated vol after a decline "
            "with a positive burst). Daniel & Moskowitz 2015 documents crashes "
            "as contemporaneous with rebounds."
        )
    if state.label is Regime.EXPANSION and state.trend_sign >= 0:
        return True, "expansion regime: hedge would cap intact upside for no risk reduction"
    if drawdown >= 0 and state.label in (Regime.EXPANSION, Regime.TREND_UP) and state.vol_ratio < 1.0:
        return True, "no drawdown and volatility below target: hedge is a pure carry cost"
    return False, None


SIZING = {
    "none": 0.0,
    "light": 0.20,
    "moderate": 0.40,
    "heavy": 0.65,
}


@dataclass
class DecisionThresholds:
    """Where the decision layer's probabilities cut.

    These are policy knobs, NOT a HedgeConfig concern: HedgeConfig is the
    quantitative hedge sizing, this is the arbitration layer. Keeping them
    separate is what stops a hedge-policy tuning run from silently retuning
    the decision thresholds.
    """

    act: float = 0.70
    veto: float = 0.30
    escalate: float = 0.50


def decide_hedge(
    state: RegimeState,
    drawdown: float,
    prev_hedge: float,
    cfg: HedgeConfig,
    decisions: dict[str, Decision] | None = None,
    thresholds: DecisionThresholds | None = None,
) -> HedgeAction:
    """Fuse the regime read, the drawdown governor and the decision layer.

    The decision layer can *veto* and can *scale* the quantitative read, but it
    cannot invent a hedge where the quantitative read says zero unless the
    decision layer is genuinely confident. Rationale: an LLM that can raise a
    hedge from nothing is an LLM that can sell you an overlay.
    """
    th = thresholds or DecisionThresholds()
    m_vol = vol_multiplier(state.vol_ratio, cfg)
    m_dd = drawdown_multiplier(drawdown, cfg)
    m_cons = consensus_multiplier(state, cfg)
    quant = float(np.clip(m_vol * m_dd * m_cons, 0.0, cfg.max_hedge_fraction))

    vetoed, veto_reason = veto(state, drawdown, cfg)
    escal = False
    decision_payload = None
    target = quant

    if decisions:
        decision_payload = {k: d.as_dict() if hasattr(d, "as_dict") else d.__dict__
                            for k, d in decisions.items()}
        should = decisions.get("should_hedge")
        gate = decisions.get("confidence_gate")
        sizing = decisions.get("hedge_sizing")

        # confidence gate: below threshold, do not act autonomously
        if gate is not None:
            p_true = gate.p("true") if gate.probabilities else (
                1.0 if float(gate.value) >= th.escalate else 0.0
            )
            if p_true < th.escalate and gate.backend != "local":
                escal = True
                target = prev_hedge  # hold, do not act

        if should is not None and not escal:
            p_hedge = should.p("true") if should.probabilities else float(should.value)
            if p_hedge < th.veto and quant < 0.15:
                vetoed, veto_reason = True, (
                    f"decision layer vetoed (p_hedge={p_hedge:.2f} < {th.veto:.2f})"
                )
            elif p_hedge >= th.act:
                if sizing is not None and isinstance(sizing.value, str):
                    target = max(quant, SIZING.get(sizing.value, quant))
                else:
                    target = min(cfg.max_hedge_fraction, quant * 1.25)
            else:
                # ambiguous band: blend toward the quantitative read
                blend = (p_hedge - th.veto) / max(th.act - th.veto, 1e-9)
                target = quant * float(np.clip(blend + 0.35, 0.0, 1.0))

    if vetoed:
        target = 0.0
    target = float(np.clip(target, 0.0, cfg.max_hedge_fraction))

    # turnover cap: the biggest silent killer of overlay strategies
    delta = target - prev_hedge
    max_delta = cfg.max_daily_turnover
    if abs(delta) > max_delta:
        target = prev_hedge + float(np.sign(delta) * max_delta)

    return HedgeAction(
        date=getattr(state, "date", None),
        target_hedge=round(target, 6),
        prev_hedge=round(prev_hedge, 6),
        regime=state.label.value,
        vol_ratio=round(state.vol_ratio, 4),
        drawdown=round(drawdown, 4),
        multiplier={"vol": round(m_vol, 4), "drawdown": round(m_dd, 4),
                    "consensus": round(m_cons, 4)},
        decision=decision_payload,
        vetoed=vetoed,
        veto_reason=veto_reason,
        escal=escal,
    )