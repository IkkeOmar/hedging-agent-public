"""Central configuration. Nothing magic, everything inspectable.

The only numbers that are allowed to be constants live here or in a strategy
spec. Everything else is derived from state.
"""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Any


# ---------------------------------------------------------------- timeframes
@dataclass(frozen=True)
class TimeframeSpec:
    """A horizon defines both the bar size and the regime lookback stack.

    Fractal principle: the same detector stack runs at every scale, and the
    scales vote. There is no privileged "the" timeframe.
    """

    name: str
    bar: str                      # yfinance / ccxt style: 5m, 1h, 1d
    # multi-scale lookbacks expressed in BARS of this timeframe
    vol_lookbacks: tuple[int, ...] = (5, 10, 20, 60, 120, 250)
    trend_lookbacks: tuple[int, ...] = (10, 20, 60, 120, 250)
    # minimum bars required before the regime detector will emit a verdict
    min_bars: int = 320
    # annualisation factor for converting per-bar variance to annual variance.
    # This is the EQUITY default. Crypto must use 365 (see BARS_PER_YEAR);
    # callers holding a universe should prefer bars_per_year(asset_class, name)
    # over reading this field directly.
    bars_per_year: int = 252

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


# Annualisation is a property of the ASSET CLASS, not of the timeframe.
# Equities trade ~252 sessions/year (260 weekdays minus ~10 NYSE holidays).
# Crypto trades every calendar day: 365. Using 252 for crypto understates
# annualised volatility by sqrt(365/252) = 1.20 -- a 20% understatement that
# silently makes every crypto Sharpe look 20% worse and every vol target too
# loose. ml4t/diagnostic makes exactly this point in its docs:
# "Daily crypto return series usually need periods_per_year=365, not 252."
#
# Annualisation must match the CADENCE OF THE RETURN SERIES being annualised,
# not the rebalance cadence that produced it.
BARS_PER_YEAR: dict[str, dict[str, int]] = {
    # asset class -> timeframe name -> bars per year
    "equity": {"swing": 252, "position": 52, "intraday": 252 * 7},
    "crypto": {"swing": 365, "position": 52, "intraday": 365 * 24},
    "mixed":  {"swing": 252, "position": 52, "intraday": 252 * 7},
}


def bars_per_year(asset_class: str, timeframe: str) -> int:
    """Resolve the annualisation factor, failing loudly on anything unknown."""
    try:
        return BARS_PER_YEAR[asset_class][timeframe]
    except KeyError as exc:  # pragma: no cover - defensive
        raise KeyError(
            f"no annualisation for asset_class={asset_class!r} "
            f"timeframe={timeframe!r}; add it to BARS_PER_YEAR"
        ) from exc


TIMEFRAMES: dict[str, TimeframeSpec] = {
    "intraday": TimeframeSpec(
        name="intraday",
        bar="1h",
        vol_lookbacks=(6, 12, 24, 72, 120, 240),
        trend_lookbacks=(12, 24, 72, 120, 240),
        min_bars=400,
        bars_per_year=252 * 7,
    ),
    "swing": TimeframeSpec(
        name="swing",
        bar="1d",
        vol_lookbacks=(5, 10, 20, 60, 120, 250),
        trend_lookbacks=(10, 20, 60, 120, 250),
        min_bars=320,
        bars_per_year=252,
    ),
    "position": TimeframeSpec(
        name="position",
        bar="1wk",
        vol_lookbacks=(4, 8, 13, 26, 52, 104),
        trend_lookbacks=(8, 13, 26, 52, 104),
        min_bars=260,
        bars_per_year=52,
    ),
}


# --------------------------------------------------------------- hedge rules
@dataclass
class HedgeConfig:
    """Dynamic hedge sizing. Every field is a *policy knob*, not an opinion.

    veto_floor exists because the whole point of the project is that the
    correct hedge is frequently zero. An agent that never hedges cannot be
    right about when to hedge.
    """

    # volatility the portfolio is willing to carry
    target_vol: float = 0.18

    # the hedge is scaled by how far realised vol sits above/below target.
    # below target -> multiplier < 1 (reduce hedge), above -> > 1 (add)
    vol_gamma: float = 0.5
    vol_floor_mult: float = 0.0
    vol_cap_mult: float = 2.5

    # drawdown governor: hedge ramps only after the portfolio is actually hurt
    dd_trigger: float = -0.05
    dd_full: float = -0.15
    dd_max_mult: float = 2.0

    # regime vote threshold: fraction of scales that must agree to act
    consensus_required: float = 0.5

    # the veto. if the regime stack says risk-on expansion, target hedge is 0.
    veto_on_expansion: bool = True

    # hard caps on notional
    max_hedge_fraction: float = 0.75
    max_daily_turnover: float = 0.25

    # cost model (fraction of notional traded)
    commission_bps: float = 1.0
    spread_bps: float = 2.0
    borrow_bps_annual: float = 10.0

    # rebalance cadence in bars. 0 = re-evaluate every bar, act only on change
    rebalance_every_bars: int = 1

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class GateConfig:
    """Promotion gate. A candidate strategy cannot be handed to the decision
    layer until it has passed every one of these on walk-forward data.

    These thresholds are the project's claim to being able to tell the
    difference between "we found an edge" and "we found the best of N".
    """

    min_trades: int = 30
    min_sharpe: float = 0.6
    min_dsr: float = 0.90           # deflated Sharpe probability
    max_pbo: float = 0.30           # probability of backtest overfitting
    max_is_oos_sharpe_decay: float = 0.50
    min_folds: int = 6
    require_positive_fold_ratio: float = 0.60
    # a strategy that never hedges is not a hedge strategy
    require_nonzero_action_ratio: float = 0.02
    # regime coverage: it must survive in more than one regime to be believed
    min_regimes_positive: int = 2

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class DecisionConfig:
    """Decision-layer config. Works with no key at all (the local fallback)."""

    backend: str = "auto"          # auto | local | a backend you registered
    model: str = "local-logistic"  # the model id your backend reports
    # thresholds are tuned on labelled data, never on a round number
    act_threshold: float = 0.70
    veto_threshold: float = 0.30
    escalate_threshold: float = 0.50
    pinned_model_in_production: bool = True
    timeout_s: float = 30.0

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class Config:
    hedge: HedgeConfig = field(default_factory=HedgeConfig)
    gate: GateConfig = field(default_factory=GateConfig)
    decision: DecisionConfig = field(default_factory=DecisionConfig)
    timeframe: str = "swing"
    asset_class: str = "equity"
    annual_rf: float = 0.04

    def tf(self) -> TimeframeSpec:
        return TIMEFRAMES[self.timeframe]

    def periods(self) -> int:
        """Annualisation factor for the configured asset class and timeframe.

        Prefer this over TIMEFRAMES[...].bars_per_year, which is the equity
        default and understates crypto annualised vol by ~20%.
        """
        return bars_per_year(self.asset_class, self.timeframe)


DEFAULT = Config()