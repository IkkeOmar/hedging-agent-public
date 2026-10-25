"""Multi-scale regime detection.

Fractal premise: there is no privileged timeframe. The same detector stack runs
at N scales, each scale votes, and the consensus is the regime signal. A
single-scale regime detector is a look-ahead machine that happens to be slow.

Inspired by the 2-of-4 orthogonal consensus in tanishhky/regime-adaptive-portfolio,
extended here to (a) more scales, (b) fractal cross-scale agreement, and
(c) emitting a calibrated *distribution* over regimes rather than a label.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

import numpy as np
import pandas as pd

from .config import TimeframeSpec


class Regime(str, Enum):
    EXPANSION = "expansion"      # vol compressing, trend positive
    TREND_UP = "trend_up"
    HIGH_VOL_CHOP = "high_vol_chop"
    DISTRIBUTION = "distribution"  # vol rising, trend negative
    CRASH = "crash"
    # Daniel & Moskowitz (2015), "Momentum Crashes" (JF 2015): the dangerous
    # state is NOT the fall. It is high volatility + a recent decline + a sharp
    # POSITIVE return burst. Vol is still elevated, direction has flipped, and
    # any strategy with a conditionally negative beta crashes exactly there.
    # A volatility-gated overlay that releases its hedge on "trend is up" fires
    # the release in the one bar it must hold on. This state exists so the veto
    # can be suspended there.
    CRASH_REBOUND = "crash_rebound"


# order matters: the score scale maps onto these levels 0..4
REGIME_SCALE = [
    "a quiet, low-volatility tape with flat or negative trend",
    "an expanding tape: trend positive and volatility compressing",
    "a directional uptrend at normal volatility",
    "high volatility with no consistent direction",
    "a directional downtrend with expanding volatility",
]


@dataclass
class ScaleVote:
    scale: int
    regime: Regime
    vol_ratio: float        # realised vol / target vol at this scale
    trend_sign: int         # -1, 0, +1
    strength: float         # 0..1, how decisive this scale is
    weight: float           # 1/sqrt(scale) style weight, short scales count more


@dataclass
class RegimeState:
    """The full multi-scale read. This is what the decision layer sees."""

    label: Regime
    score: float                       # 0..4 probability-weighted
    consensus: float                   # fraction of scales agreeing with label
    votes: list[ScaleVote]
    vol_ratio: float                   # blended across scales
    trend_sign: int
    distribution: dict[str, float]     # calibrated probability per regime
    n_scales: int
    # True when the detector is in the Daniel-Moskowitz crash-rebound state.
    # The veto must not fire here. Set by the detector, never by the policy.
    rebound: bool = False

    def summary(self) -> str:
        top = sorted(self.distribution.items(), key=lambda kv: -kv[1])[:2]
        return (
            f"{self.label.value} (score {self.score:.2f}/4, "
            f"consensus {self.consensus:.2f} across {self.n_scales} scales; "
            f"vol/target {self.vol_ratio:.2f}; "
            f"trend {'+' if self.trend_sign > 0 else '-' if self.trend_sign < 0 else '0'}; "
            f"next: {top[0][0]} {top[0][1]:.2f}, {top[1][0]} {top[1][1]:.2f})"
        )

    def as_dict(self) -> dict:
        return {
            "label": self.label.value,
            "score": round(self.score, 4),
            "consensus": round(self.consensus, 4),
            "vol_ratio": round(self.vol_ratio, 4),
            "trend_sign": self.trend_sign,
            "distribution": {k: round(v, 4) for k, v in self.distribution.items()},
            "n_scales": self.n_scales,
            "rebound": self.rebound,
            "votes": [
                {
                    "scale": v.scale,
                    "regime": v.regime.value,
                    "vol_ratio": round(v.vol_ratio, 4),
                    "trend_sign": v.trend_sign,
                    "strength": round(v.strength, 4),
                    "weight": round(v.weight, 4),
                }
                for v in self.votes
            ],
        }


def _annualised_vol(returns: pd.Series, bars_per_year: int) -> float:
    return float(returns.std(ddof=1) * np.sqrt(bars_per_year))


def detect_regime(
    prices: pd.Series | pd.DataFrame,
    spec: TimeframeSpec,
    target_vol: float = 0.18,
    annual_rf: float = 0.04,
) -> RegimeState:
    """Multi-scale regime read on a price series.

    prices: pd.Series (a single asset close) or pd.DataFrame (asset x bar).
            Multi-asset uses the mean pairwise correlation as a contagion term,
            which is the part a single asset cannot see.
    """
    if isinstance(prices, pd.Series):
        closes, rets = prices, prices.pct_change()
    else:
        # portfolio proxy: equal-weight basket of the columns
        closes = prices.mean(axis=1)
        rets = prices.pct_change().mean(axis=1)

    votes: list[ScaleVote] = []
    vrs = []
    strengths = []
    weights = []
    weights_sum = 0.0

    rets_np = rets.to_numpy(dtype=float)
    # current-bar burst and prior-bar direction, both past-only
    _last = rets_np[-1] if len(rets_np) else 0.0
    _prev = rets_np[-2] if len(rets_np) > 1 else 0.0
    burst_cut = 1.5 * float(np.nanstd(rets_np[-20:])) if len(rets_np) > 20 else 0.0

    for scale in sorted(set(spec.vol_lookbacks) | set(spec.trend_lookbacks)):
        if len(closes) < scale + 5:
            continue
        vol = _annualised_vol(rets.tail(scale), spec.bars_per_year)
        vol_ratio = vol / max(target_vol, 1e-9)

        # trend: sign of the drawup over the scale window, normalised
        window = closes.tail(scale)
        trend = float(window.iloc[-1] / window.iloc[0] - 1.0)
        trend_sign = 1 if trend > 0.01 else (-1 if trend < -0.01 else 0)

        # strength: how far vol is from target, clipped. vol_ratio == 1 -> 0
        strength = float(np.clip(abs(np.log(max(vol_ratio, 1e-9))) / np.log(2.5), 0.0, 1.0))
        # trend agreement adds to strength
        if trend_sign != 0:
            strength = float(np.clip(strength + 0.25, 0.0, 1.0))

        regime = _classify(
            vol_ratio, trend_sign, trend,
            rebound=bool(burst_cut > 0 and _last > burst_cut),
            prior_trend=int(np.sign(_prev)) if abs(_prev) > burst_cut else 0,
        )
        # shorter scales react faster, so they get more weight, but not linearly
        weight = 1.0 / np.sqrt(np.log2(scale + 1))
        weights_sum += weight

        votes.append(
            ScaleVote(scale=scale, regime=regime, vol_ratio=vol_ratio,
                      trend_sign=trend_sign, strength=strength, weight=weight)
        )
        vrs.append(vol_ratio)
        strengths.append(strength)
        weights.append(weight)

    if not votes:
        return RegimeState(Regime.HIGH_VOL_CHOP, 2.0, 0.0, [], 1.0, 0,
                           {r.value: 0.0 for r in Regime}, 0)

    w = np.array(weights) / weights_sum
    vr_arr = np.array(vrs)
    blend_vol_ratio = float(np.sum(w * vr_arr))

    # consensus: weighted share backing the winning regime
    wreg = np.array([v.weight for v in votes])
    wreg = wreg / wreg.sum()
    counts: dict[str, float] = {}
    for v, wt in zip(votes, wreg):
        counts[v.regime.value] = counts.get(v.regime.value, 0.0) + float(wt)
    label = max(counts, key=counts.get)
    consensus = counts[label]

    idx = {r.value: i for i, r in enumerate(Regime)}
    dist = {r.value: float(counts.get(r.value, 0.0)) for r in Regime}
    # CRASH_REBOUND has no place on the 0..4 severity scale; it is an overlay
    # on top of whatever severity is present, so it scores as the underlying
    # severity rather than shifting the scale.
    score = float(sum(dist[r.value] * (idx[r.value] if r is not Regime.CRASH_REBOUND
                                       else 3.0) for r in Regime))

    trend_blend = float(np.sum(wreg * np.array([v.trend_sign for v in votes])))

    return RegimeState(
        label=Regime(label),
        score=score,
        consensus=consensus,
        votes=votes,
        vol_ratio=blend_vol_ratio,
        trend_sign=int(np.sign(round(trend_blend, 3))),
        distribution=dist,
        n_scales=len(votes),
        rebound=bool(label == Regime.CRASH_REBOUND.value),
    )


def _classify(
    vol_ratio: float,
    trend_sign: int,
    trend: float,
    rebound: bool = False,
    prior_trend: int = 0,
) -> Regime:
    """Deterministic regime map. Thresholds live here on purpose, not hidden.

    The crash_rebound branch comes FIRST and is deliberately sticky. It requires
    three conditions together, all from past data only:
      1. volatility still elevated (vol_ratio > 1.15)
      2. the immediately preceding scale was negative (the decline)
      3. the current bar is a sharp positive burst (rebound)

    Daniel & Moskowitz: momentum crashes are "contemporaneous with market
    rebounds" and occur "when market volatility is high". Requiring all three
    keeps this from firing on every healthy rally inside a calm tape.
    """
    if rebound and vol_ratio > 1.15 and prior_trend < 0:
        return Regime.CRASH_REBOUND
    if vol_ratio < 0.85:
        return Regime.EXPANSION if trend_sign >= 0 else Regime.HIGH_VOL_CHOP
    if vol_ratio < 1.25:
        return Regime.TREND_UP if trend_sign > 0 else Regime.DISTRIBUTION
    if trend < -0.06 and vol_ratio > 1.45:
        return Regime.CRASH
    if trend_sign < 0 and vol_ratio > 1.2:
        return Regime.DISTRIBUTION
    return Regime.HIGH_VOL_CHOP


def rolling_regime(
    prices: pd.Series | pd.DataFrame,
    spec: TimeframeSpec,
    target_vol: float = 0.18,
    annual_rf: float = 0.04,
    warmup: int | None = None,
) -> pd.DataFrame:
    """Vectorised-ish walk over history producing one RegimeState per bar.

    Naive by design. The regime read at bar t uses bars <= t only. This is the
    property the whole project rests on, so it is the slowest thing here on
    purpose.
    """
    warmup = warmup or spec.min_bars
    if isinstance(prices, pd.Series):
        closes, rets = prices, prices.pct_change()
    else:
        closes = prices.mean(axis=1)
        rets = prices.pct_change().mean(axis=1)

    scales = sorted(set(spec.vol_lookbacks) | set(spec.trend_lookbacks))
    max_scale = max(scales)
    rows = []
    vol_full = rets.rolling(max_scale).std(ddof=1) * np.sqrt(spec.bars_per_year)

    for i in range(len(closes)):
        if i < warmup:
            rows.append({
                "date": closes.index[i], "regime": None, "score": np.nan,
                "consensus": np.nan, "vol_ratio": np.nan, "trend_sign": 0,
                "rebound": False,
            })
            continue
        state = detect_regime(
            closes.iloc[: i + 1].tail(max_scale + 5),
            spec, target_vol=target_vol, annual_rf=annual_rf,
        )
        rows.append({
            "date": closes.index[i],
            "regime": state.label.value,
            "score": state.score,
            "consensus": state.consensus,
            "vol_ratio": state.vol_ratio,
            "trend_sign": state.trend_sign,
            "rebound": bool(state.label is Regime.CRASH_REBOUND),
        })

    df = pd.DataFrame(rows).set_index("date")
    # honest: the warmup window has no verdict, and that is reported as NaN
    # rather than backfilled from the future.
    return df