"""hedging-agent: LLM-generated hedge policies, gated by walk-forward backtest,
sized dynamically by a fractal regime detector, arbitrated by a swappable
decision layer.

The claim this package makes, and tries to falsify on itself:

    The useful hedge is usually zero. An agent that cannot return zero has not
    been tested. So the correct target hedge is a function of state, sized by
    the regime, gated by a backtest that discloses its own multiple-testing, and
    vetoed by a calibrated decision model when the evidence is not there.
"""

from .config import (
    Config, HedgeConfig, GateConfig, DecisionConfig, TIMEFRAMES, TimeframeSpec,
)
from .regime import Regime, RegimeState, detect_regime, rolling_regime
from .policy import (
    DecisionThresholds, HedgeAction, decide_hedge, veto,
    vol_multiplier, drawdown_multiplier, consensus_multiplier,
)
from .decision import (
    Choice, Decision, DecisionBackend, LocalBackend, Score, Noul,
    build_backend, hedge_questions, register_backend, available_backends,
)
from .stats import (
    deflated_sharpe_ratio, pbo_cscv, evaluate_walk_forward, walk_forward_folds,
    expected_max_sharpe, probabilistic_sharpe_ratio, newey_west_tstat,
    sharpe_difference_tstat,
    gate_report, regime_breakdown, sharpe, sortino, max_drawdown, calmar,
)
from .benchmarks import benchmark_table
from .strategies import HedgeStrategySpec, generate_candidates, template_candidates, validate_spec
from .pipeline import run_sweep, run_agent, simulate_hedge, print_report

__version__ = "0.1.0"

__all__ = [
    "Config", "HedgeConfig", "GateConfig", "DecisionConfig", "TIMEFRAMES", "TimeframeSpec",
    "Regime", "RegimeState", "detect_regime", "rolling_regime",
    "DecisionThresholds", "HedgeAction", "decide_hedge", "veto", "vol_multiplier",
    "drawdown_multiplier", "consensus_multiplier",
    "Choice", "Decision", "DecisionBackend", "LocalBackend", "Score", "Noul",
    "build_backend", "hedge_questions", "register_backend", "available_backends",
    "deflated_sharpe_ratio", "pbo_cscv", "evaluate_walk_forward", "walk_forward_folds",
    "expected_max_sharpe", "probabilistic_sharpe_ratio", "newey_west_tstat",
    "sharpe_difference_tstat", "benchmark_table",
    "gate_report", "regime_breakdown", "sharpe", "sortino", "max_drawdown", "calmar",
    "HedgeStrategySpec", "generate_candidates", "template_candidates", "validate_spec",
    "run_sweep", "run_agent", "simulate_hedge", "print_report",
    "__version__",
]