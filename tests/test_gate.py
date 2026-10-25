"""The gate must be fail-closed.

A promotion gate that passes when the evidence is missing is decoration. These
tests pin that behaviour for the two checks the literature forced us to add.
"""

from __future__ import annotations

import numpy as np
import pytest

from hedging_agent.benchmarks import benchmark_table, constant_hedge
from hedging_agent.config import GateConfig, HedgeConfig
from hedging_agent.stats import (
    WalkForwardResult, gate_report, min_bars_for_sharpe,
    sharpe_difference_tstat,
)


def _fixture(n=900, seed=19):
    rng = np.random.default_rng(seed)
    p = rng.normal(0.0004, 0.011, n)
    h = -0.85 * p + rng.normal(0.0, 0.002, n)
    return p, h


def _wf():
    return WalkForwardResult(
        folds=[], n_folds=8, positive_fold_ratio=0.75,
        max_is_oos_decay=0.1, mean_oos_sharpe=1.2, aggregate_sharpe=1.1,
    )


def _base_kwargs(**over):
    kw = dict(
        metrics={"sharpe": 1.1},
        dsr={"dsr": 0.95},
        pbo={"pbo": 0.25},
        wf=_wf(),
        # n_bars is load-bearing: the gate only counts regimes with enough
        # bars to have an estimable Sharpe (>= 63 at periods=252)
        # sharpe_se is load-bearing: the gate reads the per-universe
        # estimability bound off each regime rather than assuming the equity
        # default of 63 bars (crypto needs 92).
        regime_stats={"trend_up": {"sharpe": 0.4, "n_bars": 200,
                                   "sharpe_se": 1.12},
                      "crash": {"sharpe": 0.2, "n_bars": 200,
                                "sharpe_se": 1.12}},
        action_ratio=0.6,
        cfg=GateConfig(),
    )
    kw.update(over)
    return kw


def test_gate_ignores_regimes_whose_sharpe_is_not_estimable():
    """A regime too thin to estimate a Sharpe gets no vote.

    Found in a crypto sweep printing "crash_rebound sharpe 417.3691 n 2".
    The gate counted sharpe > 0 toward min_regimes_positive, so that one
    two-bar regime satisfied the check by itself.
    """
    # every regime too thin to estimate: sharpe is None, nothing counts
    out = gate_report(**_base_kwargs(
        regime_stats={"stub": {"n_bars": 2, "sharpe": None,
                               "sharpe_se": 13.5, "significant": False,
                               "total_return": 0.1, "max_dd": 0.0}},
        benchmarks=_passing_benchmarks(),
        significance={"t_stat": 5.0},
    ))
    check = next(c for c in out["checks"]
                 if c["check"] == "min_regimes_positive")
    assert check["value"] == 0
    assert check["pass"] is False

    # a fat regime counts, the 417-sharpe stub does not
    out2 = gate_report(**_base_kwargs(
        regime_stats={
            "real": {"n_bars": 300, "sharpe": 1.4, "sharpe_se": 0.92,
                     "significant": True, "total_return": 0.4, "max_dd": -0.1},
            # n_bars=2 is what disqualifies it, NOT the absence of a flag --
            # a caller could hand the gate any dict, so the bar count is
            # what the guard has to test
            "stub": {"n_bars": 2, "sharpe": 417.4, "sharpe_se": 13.5,
                     "significant": False, "total_return": 0.1, "max_dd": 0.0},
        },
        benchmarks=_passing_benchmarks(),
        significance={"t_stat": 5.0},
    ))
    check2 = next(c for c in out2["checks"]
                  if c["check"] == "min_regimes_positive")
    assert check2["value"] == 1  # the 417.4 stub is ignored
    assert check2["pass"] is False  # threshold is 2


def test_gate_uses_the_universe_own_threshold_not_the_equity_default():
    """Crypto's bar threshold is 92, not 63, and the gate must use the former.

    The guard in section 15 initially called min_bars_for_sharpe() with its
    equity default. Crypto annualises over 365, so it needs 92 bars -- and its
    expansion regime sits at exactly 90. The live sweep consequently reported
    zero positive regimes for crypto when it had one. The estimability bound
    now travels on each regime as sharpe_se, computed with that universe's own
    periods, so the gate cannot silently apply the wrong universe's default.
    """
    out = gate_report(**_base_kwargs(
        # 90 bars: below crypto's 92-bar threshold, above equity's 63
        regime_stats={"expansion": {"n_bars": 90, "sharpe": 5.7,
                                    "sharpe_se": 2.014, "significant": False,
                                    "total_return": 1.09, "max_dd": -0.14}},
        benchmarks=_passing_benchmarks(),
        significance={"t_stat": 5.0},
    ))
    c = next(x for x in out["checks"] if x["check"] == "min_regimes_positive")
    assert c["value"] == 0  # SE 2.014 > 2.0 -> excluded

    # same universe, 100 bars: SE = sqrt(365/100) = 1.91, now it counts
    out2 = gate_report(**_base_kwargs(
        regime_stats={"expansion": {"n_bars": 100, "sharpe": 5.7,
                                    "sharpe_se": 1.911, "significant": True,
                                    "total_return": 1.09, "max_dd": -0.14}},
        benchmarks=_passing_benchmarks(),
        significance={"t_stat": 5.0},
    ))
    c2 = next(x for x in out2["checks"] if x["check"] == "min_regimes_positive")
    assert c2["value"] == 1

    # and confirm the threshold arithmetic itself
    assert min_bars_for_sharpe(365) == 92
    assert min_bars_for_sharpe(252) == 63


def test_gate_rejects_when_no_benchmark_is_supplied():
    out = gate_report(**_base_kwargs())
    assert out["passed"] is False
    assert "beats_simple_benchmark" in out["failed_checks"]


def test_gate_rejects_when_no_tstat_is_supplied():
    out = gate_report(**_base_kwargs())
    assert out["passed"] is False
    assert "significance_t" in out["failed_checks"]


def test_gate_rejects_when_the_candidate_only_matches_a_constant_hedge():
    """A candidate identical to the constant hedge has zero edge."""
    p, h = _fixture()
    bm = benchmark_table(p, h, constant_hedge(p, h, 0.5), name="candidate")
    assert bm["verdict"]["beats_best_simple_on_sharpe"] is False
    out = gate_report(**_base_kwargs(benchmarks=bm,
                                    significance={"t_stat": 5.0}))
    assert out["passed"] is False
    assert "beats_simple_benchmark" in out["failed_checks"]


def test_a_constant_hedge_candidate_loses_to_the_risk_matched_control():
    """A candidate that IS a constant hedge has exactly zero timing edge."""
    p, h = _fixture()
    bm = benchmark_table(p, h, constant_hedge(p, h, 0.5), name="candidate")
    assert bm["risk_matched"]["timing_sharpe_gap"] == pytest.approx(0.0, abs=1e-6), (
        "a constant hedge must tie its own risk-matched self, to numerical noise"
    )


def test_gate_rejects_a_tstat_below_the_harvey_hurdle():
    out = gate_report(**_base_kwargs(benchmarks=_passing_benchmarks(),
                                    significance={"t_stat": 2.4}))
    assert out["passed"] is False
    assert "significance_t" in out["failed_checks"]


def _passing_benchmarks(gap: float = 0.4) -> dict:
    return {
        "verdict": {
            "beats_best_simple_on_sharpe": True,
            "sharpe_edge_over_best_simple": gap,
            "best_simple_sharpe": 1.0, "candidate_sharpe": 1.4,
            "n_benchmarks_beaten_on_sharpe": 5, "n_benchmarks": 6,
        },
        "risk_matched": {
            "hedge_ratio": 0.42,
            "candidate_vol": 0.0081,
            "matched_vol": 0.0081,
            "timing_sharpe_gap": gap,
            "timing_dd_gap": -0.02,
        },
    }


def test_gate_accepts_only_when_both_evidence_pieces_are_present():
    out = gate_report(**_base_kwargs(benchmarks=_passing_benchmarks(),
                                    significance={"t_stat": 4.1}))
    assert out["passed"] is True, out["failed_checks"]


def test_gate_rejects_when_the_timing_gap_is_negative():
    """Beating buy-and-hold while losing to a RISK-MATCHED constant hedge is
    not a pass. This is the failure mode Maguire documents."""
    out = gate_report(**_base_kwargs(benchmarks=_passing_benchmarks(gap=-0.11),
                                    significance={"t_stat": 9.9}))
    assert out["passed"] is False
    assert "beats_simple_benchmark" in out["failed_checks"]


def test_gate_rejects_a_benchmark_table_without_the_risk_matched_control():
    out = gate_report(**_base_kwargs(benchmarks={"verdict": {
        "beats_best_simple_on_sharpe": True, "sharpe_edge_over_best_simple": 0.4,
        "best_simple_sharpe": 1.0, "candidate_sharpe": 1.4,
        "n_benchmarks_beaten_on_sharpe": 5, "n_benchmarks": 5}},
        significance={"t_stat": 9.9}))
    assert out["passed"] is False, (
        "a drawdown claim without a risk-matched control must not promote"
    )
    assert "beats_simple_benchmark" in out["failed_checks"]


def test_nan_tstat_never_passes():
    bm = {"verdict": {
        "beats_best_simple_on_sharpe": True,
        "sharpe_edge_over_best_simple": 0.4,
        "best_simple_sharpe": 1.0, "candidate_sharpe": 1.4,
        "n_benchmarks_beaten_on_sharpe": 5, "n_benchmarks": 5}}
    out = gate_report(**_base_kwargs(benchmarks=_passing_benchmarks(),
                                    significance={"t_stat": float("nan")}))
    assert out["passed"] is False, "a NaN t-statistic must never satisfy a threshold"


def test_significance_cites_its_hurdle():
    bm = _passing_benchmarks()
    out = gate_report(**_base_kwargs(benchmarks=_passing_benchmarks(),
                                    significance={"t_stat": 4.1}))
    chk = next(c for c in out["checks"] if c["check"] == "significance_t")
    assert "Harvey" in chk["detail"], "every threshold must cite its source"
    assert chk["threshold"] == 3.0


def test_tstat_from_a_real_fixture_is_computable_end_to_end():
    p, h = _fixture()
    cand = constant_hedge(p, h, 0.5)
    sig = sharpe_difference_tstat(cand, p)
    assert sig["n_obs"] == len(p)
    assert not np.isnan(sig["t_stat"])
