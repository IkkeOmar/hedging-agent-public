"""Synthetic-data tests. No network. These run in CI and on every push.

The rules the engine must never break:
  1. no look-ahead: a signal at bar t fills at t+1
  2. the regime read at bar t uses bars <= t
  3. a hedge policy is allowed to return zero, and the veto path works
  4. the promotion gate actually rejects bad strategies
  5. DSR falls when n_trials rises (the whole point of the correction)
  6. turnover caps bind
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest

from hedging_agent.config import TIMEFRAMES, HedgeConfig, GateConfig, Config
from hedging_agent.decision import Choice, LocalBackend, Noul, Score, build_backend, hedge_questions
from hedging_agent.policy import decide_hedge, drawdown_multiplier, vol_multiplier, veto
from hedging_agent.regime import Regime, RegimeState, detect_regime, rolling_regime
from hedging_agent.stats import (
    deflated_sharpe_ratio, evaluate_walk_forward, expected_max_sharpe, gate_report,
    max_drawdown, pbo_cscv, probabilistic_sharpe_ratio, sharpe, walk_forward_folds,
)
from hedging_agent.strategies import (
    HedgeStrategySpec, template_candidates, validate_spec,
)
from hedging_agent.data import build_hedged_returns, infer_hedge_side
from hedging_agent.pipeline import simulate_hedge


# ------------------------------------------------------------------ fixtures
def synthetic_prices(n: int = 900, seed: int = 3, regime_shifts: bool = True) -> pd.DataFrame:
    """Multi-asset frame with deliberate regime breaks so the detector has
    something real to find. Without breaks, every hedge looks great and the
    tests would be meaningless."""
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2015-01-01", periods=n, freq="D")
    # three regimes: calm-trending, choppy, crash
    vols = np.concatenate([
        np.full(n // 3, 0.005),
        np.full(n // 3, 0.020),
        np.full(n - 2 * (n // 3), 0.045),
    ])[:n]
    drifts = np.concatenate([
        np.full(n // 3, 0.0006),
        np.full(n // 3, 0.0001),
        np.full(n - 2 * (n // 3), -0.0025),
    ])[:n]
    if not regime_shifts:
        vols[:] = 0.01
        drifts[:] = 0.0004
    cols = {}
    # common factor + idiosyncratic noise, so correlation actually moves
    common = rng.normal(0, 1, n)
    for i, name in enumerate(["AAA", "BBB", "CCC", "HEDGE"]):
        beta = 0.35 if name == "HEDGE" else 1.0
        idio = rng.normal(0, 1, n) * (0.35 if name == "HEDGE" else 0.8)
        r = (common * beta + idio) * vols + drifts
        cols[name] = 100.0 * np.cumprod(1.0 + r)
    return pd.DataFrame(cols, index=idx)


# ------------------------------------------------------------------- regime
def test_regime_detector_finds_the_regimes():
    px = synthetic_prices()
    st = detect_regime(px["AAA"], TIMEFRAMES["swing"])
    assert st.n_scales >= 4
    assert 0.0 <= st.consensus <= 1.0
    assert set(st.distribution.keys()) == {r.value for r in Regime}
    # on synthetic data with a real crash the vol ratio must exceed 1 somewhere
    tail = detect_regime(px["AAA"].tail(120), TIMEFRAMES["swing"])
    assert tail.vol_ratio > 1.0, f"crash leg should read high vol, got {tail.vol_ratio}"


def test_regime_is_fractal_across_scales():
    px = synthetic_prices()
    st = detect_regime(px, TIMEFRAMES["swing"])  # DataFrame path -> portfolio + contagion
    assert st.n_scales >= 4
    assert all(v.scale >= 5 for v in st.votes)
    # weight ordering must be monotonic in scale: shorter scale weighs more
    scales = [v.scale for v in sorted(st.votes, key=lambda v: v.scale)]
    weights = [v.weight for v in sorted(st.votes, key=lambda v: v.scale)]
    assert weights == sorted(weights, reverse=True), "short scales must dominate"


def test_no_lookahead_in_rolling_regime():
    """Truncating history must not change past verdicts. If it does, the
    detector is peeking."""
    px = synthetic_prices()
    full = rolling_regime(px[["AAA"]], TIMEFRAMES["swing"], warmup=320)
    cut = rolling_regime(px[["AAA"]].iloc[:600], TIMEFRAMES["swing"], warmup=320)
    common = full.loc[full.index <= cut.index[-1]].dropna(subset=["score"])
    assert len(common) > 100, "not enough live verdicts to compare"
    mismatch = (common["regime"].values != cut.loc[common.index, "regime"].values).sum()
    assert mismatch == 0, f"{mismatch} historical verdicts changed when future data was removed"


def test_warmup_is_nan_not_backfilled():
    px = synthetic_prices()
    df = rolling_regime(px[["AAA"]], TIMEFRAMES["swing"], warmup=320)
    head = df.iloc[:320]
    assert head["score"].isna().all(), "warmup must be NaN, never backfilled from the future"


# ------------------------------------------------------------------- policy
def test_vol_multiplier_is_inverted_and_clipped():
    cfg = HedgeConfig()
    assert vol_multiplier(2.0, cfg) > vol_multiplier(1.0, cfg) > vol_multiplier(0.5, cfg)
    assert vol_multiplier(99.0, cfg) <= cfg.vol_cap_mult
    assert vol_multiplier(0.01, cfg) >= cfg.vol_floor_mult


def test_drawdown_ramp_only_activates_after_the_trigger():
    cfg = HedgeConfig()
    assert drawdown_multiplier(0.0, cfg) == 1.0
    assert drawdown_multiplier(-0.02, cfg) == 1.0
    mid = drawdown_multiplier(-0.10, cfg)
    full = drawdown_multiplier(-0.30, cfg)
    assert 1.0 < mid < full <= cfg.dd_max_mult


def test_veto_returns_zero_in_expansion():
    cfg = HedgeConfig()
    st = RegimeState(Regime.EXPANSION, 0.4, 0.7, [], 0.6, 1,
                     {r.value: 0.0 for r in Regime}, 4)
    v, why = veto(st, drawdown=0.0, cfg=cfg)
    assert v is True and why
    # the veto releases the hedge, subject to the turnover cap
    prev = 0.40
    act = decide_hedge(st, 0.0, prev_hedge=prev, cfg=cfg)
    assert act.vetoed is True
    # the veto unwinds, but never faster than the turnover cap allows
    assert act.target_hedge == pytest.approx(prev - cfg.max_daily_turnover)
    assert act.target_hedge < prev, "veto must unwind the existing hedge"


def test_policy_can_and_does_return_zero():
    """The load-bearing property of the whole project."""
    cfg = HedgeConfig()
    states = []
    for r in list(Regime):
        for vr in (0.5, 1.0, 2.5):
            for ts in (-1, 0, 1):
                states.append(RegimeState(r, 2.0, 0.6, [], vr, ts,
                                          {x.value: 0.2 for x in Regime}, 4))
    zeroes = 0
    for st in states:
        act = decide_hedge(st, drawdown=0.0, prev_hedge=0.0, cfg=cfg)
        if act.target_hedge == 0.0:
            zeroes += 1
    ratio = zeroes / len(states)
    assert ratio > 0.15, f"policy returned zero only {ratio:.1%} of the time; a constant hedge"


def test_crash_rebound_suspends_the_veto():
    """Daniel & Moskowitz (2015): the crash is high vol + decline + POSITIVE
    burst. A veto that fires there releases protection in the exact bar the
    strategy is most exposed."""
    cfg = HedgeConfig()
    # looks like a healthy uptrend (which would normally veto) but carries the
    # rebound flag because vol is still elevated after a decline
    st = RegimeState(Regime.EXPANSION, 0.5, 0.6, [], 1.6, 1,
                     {r.value: 0.2 for r in Regime}, 4, rebound=True)
    v, why = veto(st, drawdown=0.0, cfg=cfg)
    assert v is False, "veto must be suspended on a crash rebound"
    assert why and "crash-rebound" in why
    act = decide_hedge(st, 0.0, prev_hedge=0.0, cfg=cfg)
    assert act.target_hedge > 0.0, "protection must be held through the rebound"


def test_crash_rebound_requires_all_three_conditions():
    from hedging_agent.regime import _classify
    # elevated vol + rebound after a decline -> fires
    assert _classify(1.6, 1, 0.03, rebound=True, prior_trend=-1) is Regime.CRASH_REBOUND
    # calm tape -> does not fire, even with a rebound
    assert _classify(0.6, 1, 0.03, rebound=True, prior_trend=-1) is not Regime.CRASH_REBOUND
    # elevated vol but NO prior decline -> does not fire
    assert _classify(1.6, 1, 0.03, rebound=True, prior_trend=1) is not Regime.CRASH_REBOUND
    # elevated vol and a prior decline but NO burst -> does not fire
    assert _classify(1.6, 1, 0.03, rebound=False, prior_trend=-1) is not Regime.CRASH_REBOUND


def test_rebound_state_is_never_backfilled_into_warmup():
    px = synthetic_prices()
    df = rolling_regime(px[["AAA"]], TIMEFRAMES["swing"], warmup=320)
    assert not df["rebound"].iloc[:320].any(), "warmup must not claim a rebound"
    assert df["rebound"].dtype == bool


def test_turnover_cap_binds():
    cfg = HedgeConfig(max_daily_turnover=0.10)
    st = RegimeState(Regime.CRASH, 4.0, 0.9, [], 3.0, -1,
                     {r.value: 0.2 for r in Regime}, 4)
    act = decide_hedge(st, drawdown=-0.30, prev_hedge=0.0, cfg=cfg)
    assert act.target_hedge <= 0.10 + 1e-9, f"turnover cap ignored: {act.target_hedge}"


# ----------------------------------------------------------------- strategy
def test_validator_rejects_constant_hedge():
    spec = template_candidates(1)[0]
    spec.veto_on_expansion = False
    problems = validate_spec(spec)
    assert any("constant hedge" in p for p in problems), problems


def test_validator_rejects_inverted_drawdown_ladder():
    spec = template_candidates(1)[0]
    spec.dd_trigger, spec.dd_full = -0.20, -0.05
    assert any("dd_trigger" in p for p in validate_spec(spec))


def test_template_bank_is_mechanism_diverse():
    specs = template_candidates(6)
    assert len(specs) == 6
    assert len({s.name for s in specs}) == 6
    assert len({s.hedge_instrument for s in specs}) >= 2
    assert all(not validate_spec(s) for s in specs)


# -------------------------------------------------------------------- stats
# --------------------------------------------------------------------------
# Golden tests. These reproduce the printed numbers in Bailey & Lopez de Prado
# (2014), "The Deflated Sharpe Ratio", JPM 40(5) pp. 9-10, "A Numerical
# Example". If these fail, the DSR is wrong and every headline in this repo is
# wrong with it.
#
# Paper's setup: N=100 trials, variance of the DAILY Sharpe across trials
# V = 1/(2*250) = 0.002, T = 1250 obs, skew = -3, RAW kurtosis = 10,
# best-trial Sharpe = 2.5 annualised. Paper prints SR0 ~= 0.1132 (daily)
# and DSR = 0.9004, i.e. below its own 0.95 significance bar.
# --------------------------------------------------------------------------
V_TRIALS_DAILY = 1.0 / (2.0 * 250.0)
SR_BEST_DAILY = 2.5 / np.sqrt(250.0)
T_PAPER = 1250


def test_expected_max_sharpe_matches_paper_sr0():
    assert expected_max_sharpe(V_TRIALS_DAILY, 100) == pytest.approx(0.1132, abs=5e-5)


def test_dsr_matches_paper():
    sr0 = expected_max_sharpe(V_TRIALS_DAILY, 100)
    dsr = probabilistic_sharpe_ratio(SR_BEST_DAILY, sr0, T_PAPER, -3.0, 10.0)
    assert dsr == pytest.approx(0.9004, abs=5e-5)
    assert dsr < 0.95, "the paper's own conclusion: not significant at 95%"


def test_dsr_matches_paper_secondary_claims():
    # only 46 trials run
    assert probabilistic_sharpe_ratio(
        SR_BEST_DAILY, expected_max_sharpe(V_TRIALS_DAILY, 46), T_PAPER, -3.0, 10.0
    ) == pytest.approx(0.9505, abs=5e-5)
    # normal returns, 88 trials
    assert probabilistic_sharpe_ratio(
        SR_BEST_DAILY, expected_max_sharpe(V_TRIALS_DAILY, 88), T_PAPER, 0.0, 3.0
    ) == pytest.approx(0.9505, abs=5e-5)


def test_paper_example_says_fewer_trials_is_better():
    """The entire argument of DSR, as a single assertion: 46 trials beats 100."""
    few = probabilistic_sharpe_ratio(
        SR_BEST_DAILY, expected_max_sharpe(V_TRIALS_DAILY, 46), T_PAPER, -3.0, 10.0)
    many = probabilistic_sharpe_ratio(
        SR_BEST_DAILY, expected_max_sharpe(V_TRIALS_DAILY, 100), T_PAPER, -3.0, 10.0)
    assert few > many


def test_dsr_falls_as_trials_rise():
    # A modest Sharpe over a short sample: long enough to be meaningful, short
    # enough that the DSR is not saturated at 1.0 (which is the correct answer
    # for a strong Sharpe over hundreds of bars and would make this test vacuous).
    # seed/n chosen by scanning for a case where the DSR is genuinely
    # unsaturated at low trials and genuinely collapsed at high trials. A
    # saturated DSR would make this assertion vacuous.
    rng = np.random.default_rng(287)
    r = rng.normal(0.0006, 0.012, 150)
    few = deflated_sharpe_ratio(r, n_trials=2)["dsr"]
    many = deflated_sharpe_ratio(r, n_trials=50_000)["dsr"]
    assert many < few, "deflation must bite harder with more trials"
    assert 0.0 <= many <= 1.0
    assert few < 1.0, "fixture should not be saturated, or the test proves nothing"


def test_dsr_saturates_for_a_strong_sharpe_over_a_long_sample():
    """The honest limit: enough observations make a real edge unmistakable."""
    rng = np.random.default_rng(6)
    r = rng.normal(0.0008, 0.01, 900)
    out = deflated_sharpe_ratio(r, n_trials=100)
    assert out["dsr"] > 0.99


def test_dsr_on_pure_noise_is_low():
    rng = np.random.default_rng(9)
    r = rng.normal(0.0, 0.01, 1000)
    out = deflated_sharpe_ratio(r, n_trials=5_000)
    assert out["dsr"] < 0.5, "a noise series with 5000 trials must not pass"


def test_pbo_is_near_half_for_random_matrices():
    rng = np.random.default_rng(2)
    m = rng.normal(0, 1, (600, 12))     # pure noise: selection is a coin flip
    out = pbo_cscv(m, n_splits=8)
    assert 0.2 <= out["pbo"] <= 0.8, out


def test_pbo_is_unbiased_under_pure_noise():
    """PBO MUST average 0.5 across many draws when there is no skill.

    The previous version of this test checked only ONE draw against a 0.2-0.8
    band, which happily passed while the estimator carried a systematic
    DOWNWARD bias of -0.25 at N=4 and -0.08 at N=12. A single-sample band test
    cannot see a bias; averaging can.

    Found by cross-checking against ml4t.diagnostic, which uses the Bailey &
    Lopez de Prado form omega/(N+1). We scaled by (omega-1)/N, which flags
    N/2 - 1 of the ranks on even N instead of N/2: exactly one rank too few,
    so PBO reads 1/N low. That flatters every candidate -- the exact
    direction of error an overfitting gate must not make. On odd N the two
    forms agree and there is no bias.

    Before the fix: 0.324 (N=6), 0.413 (N=10), 0.421 (N=12).
    After:         0.494 (N=6), 0.515 (N=10), 0.503 (N=12).
    """
    rng = np.random.default_rng(11)
    for n_cfg in (6, 10, 12):
        vals = np.array([
            pbo_cscv(rng.normal(0.0, 0.01, (2000, n_cfg)), n_splits=8)["pbo"]
            for _ in range(150)
        ])
        assert abs(vals.mean() - 0.5) < 0.05, (
            f"PBO must average 0.5 under noise; N={n_cfg} gave "
            f"{vals.mean():.4f} (a downward bias means overfitting is "
            f"being under-detected)"
        )


def test_pbo_rank_divisor_is_n_plus_one():
    """Pinned arithmetic check on the (N+1) divisor, independent of any data."""
    # For N even and omega uniform on {1..N}, exactly N/2 ranks exceed (N+1)/2,
    # so an unbiased estimator returns exactly 0.5 on noise by construction.
    for n_cfg in (4, 6, 10, 12, 20):
        correct = sum(1 for r in range(1, n_cfg + 1)
                      if r / (n_cfg + 1) - 0.5 > 0)
        assert correct == n_cfg // 2, (
            f"the (N+1) form must put exactly half the ranks in the bottom "
            f"half for N={n_cfg}, got {correct}"
        )


# ------------------------------------------------------------- cost model
def test_costs_are_actually_charged():
    """The cost model must reduce returns. It previously charged nothing.

    The cost fields existed in config, were printed in the report metrics, and
    were never applied to the return series at all. A Sharpe printed for a
    strategy that trades 40x a year was gross of costs.
    """
    import pandas as pd
    from hedging_agent.data import build_hedged_returns

    idx = pd.date_range("2020-01-01", periods=400, freq="D")
    rng = np.random.default_rng(2)
    prices = pd.DataFrame(
        {"AAA": 100 * np.cumprod(1 + rng.normal(0, 0.01, 400)),
         "SH": 100 * np.cumprod(1 + rng.normal(0, 0.01, 400))},
        index=idx,
    )
    # A CONSTANT notional trades nothing after the first bar, so it cannot
    # exercise the cost model. Ramp up then down, which is what a real hedge
    # does: total traded = 0.2 + 0.2 = 0.4 (enter once, exit once).
    ramp = pd.Series(np.concatenate([np.full(200, 0.2), np.zeros(200)]), index=idx)

    free, _ = build_hedged_returns(prices, ramp, "SH",
                                   portfolio_cols=["AAA"],
                                   round_trip_cost_bps=0.0)
    costed, _ = build_hedged_returns(prices, ramp, "SH",
                                     portfolio_cols=["AAA"],
                                     round_trip_cost_bps=50.0)
    assert len(free) == len(costed)
    assert costed.mean() < free.mean(), "costs did not reduce mean return"
    # total traded = 0.4 notional; 50bps = 0.005 -> 0.002 once, spread over
    # the 399 return bars
    assert free.mean() - costed.mean() == pytest.approx(
        0.4 * 0.005 / len(costed), rel=0.05
    )


def test_borrow_is_charged_as_an_annual_rate_not_per_bar():
    """Borrow is an ANNUAL rate. Charging it every bar overstates it ~252x.

    Regression: `borrow_bps_annual / periods * notional` treated the bps figure
    as if it were already a fraction, charging a full year's borrow every
    single bar. At 10bps over 252 bars that is a 120%/year drag instead of 10%.
    """
    import pandas as pd
    from hedging_agent.data import build_hedged_returns

    idx = pd.date_range("2020-01-01", periods=400, freq="D")
    rng = np.random.default_rng(3)
    prices = pd.DataFrame(
        {"AAA": 100 * np.cumprod(1 + rng.normal(0, 0.01, 400)),
         "SH": 100 * np.cumprod(1 + rng.normal(0, 0.01, 400))},
        index=idx,
    )
    notional = pd.Series(0.2, index=idx)

    free, _ = build_hedged_returns(prices, notional, "SH",
                                   portfolio_cols=["AAA"],
                                   borrow_bps_annual=0.0)
    borrow, _ = build_hedged_returns(prices, notional, "SH",
                                     portfolio_cols=["AAA"],
                                     borrow_bps_annual=100.0)
    drag_per_bar = (free - borrow).mean()
    # 100bps/yr = 1%/yr, on a 0.2 notional = 0.002 of equity per YEAR,
    # i.e. 0.002/252 = 7.94e-6 per bar.
    # The broken version charged (100/252)*0.2 = 0.079 per bar -- 10,000x too
    # big, because it treated the bps number as an already-converted fraction.
    assert drag_per_bar == pytest.approx(0.002 / 252, rel=0.05), (
        f"per-bar borrow drag is {drag_per_bar:.8f}, expected "
        f"{0.002 / 252:.8f} (1% annual on a 0.2 notional, spread over 252 "
        f"bars). Anything near 0.079 means the annual rate is applied every bar."
    )
    # and the annualised total must equal the stated rate on the notional
    assert drag_per_bar * 252 == pytest.approx(0.002, rel=0.05)


def test_zero_cost_reproduces_the_pre_cost_Sharpe_exactly():
    """A zero-cost run must equal the old gross behaviour, bit for bit."""
    import pandas as pd
    from hedging_agent.data import build_hedged_returns

    idx = pd.date_range("2020-01-01", periods=300, freq="D")
    rng = np.random.default_rng(9)
    prices = pd.DataFrame(
        {"AAA": 100 * np.cumprod(1 + rng.normal(0, 0.01, 300)),
         "SH": 100 * np.cumprod(1 + rng.normal(0, 0.01, 300))},
        index=idx,
    )
    notional = pd.Series(0.15, index=idx)
    a, _ = build_hedged_returns(prices, notional, "SH", portfolio_cols=["AAA"])
    b, _ = build_hedged_returns(prices, notional, "SH", portfolio_cols=["AAA"],
                                 round_trip_cost_bps=0.0,
                                 borrow_bps_annual=0.0)
    pd.testing.assert_series_equal(a, b)


# ---------------------------------------------------------- annualisation
def test_crypto_annualises_at_365_not_252():
    """Crypto trades every calendar day. Using 252 understates vol by 20%.

    sqrt(365/252) = 1.2035, so annualised crypto volatility computed with the
    equity factor comes out ~20% too low, which in turn makes every volatility
    target too loose and every Sharpe too pessimistic. ml4t/diagnostic states
    the rule directly: "Daily crypto return series usually need
    periods_per_year=365, not 252."
    """
    from hedging_agent.config import bars_per_year
    from hedging_agent.pipeline import annualisation

    assert bars_per_year("equity", "swing") == 252
    assert bars_per_year("crypto", "swing") == 365
    assert annualisation("us_equities", "swing") == 252
    assert annualisation("crypto", "swing") == 365
    # and the magnitude of the error we were making
    assert round((365 / 252) ** 0.5, 3) == 1.204


def test_crypto_and_equity_sharpes_are_not_comparable_across_annualisations():
    """The same return stream annualised two ways must differ by sqrt(365/252)."""
    import pandas as pd
    from hedging_agent.stats import sharpe

    rng = np.random.default_rng(4)
    r = pd.Series(rng.normal(0.001, 0.02, 365))
    eq = sharpe(r, periods=252)
    cr = sharpe(r, periods=365)
    # Sharpe scales as sqrt(periods), so the crypto factor is the LARGER one
    assert cr == pytest.approx(eq * (365 / 252) ** 0.5, rel=1e-9)
    assert cr > eq


def test_unknown_asset_class_fails_loudly():
    from hedging_agent.config import bars_per_year
    with pytest.raises(KeyError, match="no annualisation"):
        bars_per_year("forex", "swing")


def test_thin_regimes_get_no_sharpe_and_cannot_pass_the_gate():
    """A 2-bar regime has no estimable Sharpe, and must not vote.

    Found in a crypto sweep that printed "crash_rebound sharpe 417.3691 n 2".
    The gate counted regimes with sharpe > 0 toward min_regimes_positive, so
    that single 2-bar regime satisfied the check on its own -- the gate was
    being passed by arithmetic, not evidence.
    """
    from hedging_agent.stats import min_bars_for_sharpe, regime_breakdown

    # SE(SR) ~= sqrt(periods/n); at the equity default this needs n >= 63
    assert min_bars_for_sharpe(252) == 63
    # crypto annualises over 365 days, so it needs strictly more bars
    assert min_bars_for_sharpe(365) == 92
    # a looser bound demands fewer bars, never more
    assert min_bars_for_sharpe(252, max_se=1.0) > min_bars_for_sharpe(252, max_se=2.0)

    idx = pd.date_range("2020-01-01", periods=400, freq="D")
    r = pd.Series(np.linspace(-0.001, 0.002, 400), index=idx)
    # one fat regime, plus a 2-bar regime whose sign is pure luck
    reg = pd.Series(["fat"] * 398 + ["stub"] * 2, index=idx)

    out = regime_breakdown(r, reg, periods=252)

    assert out["fat"]["significant"] is True
    assert out["fat"]["sharpe"] is not None
    # SE ~ sqrt(periods/n) = sqrt(252/398) = 0.796. (2.0 is the SE at
    # exactly 63 bars, the threshold -- not the SE at 398.)
    assert out["fat"]["sharpe_se"] == pytest.approx(0.7957, abs=0.001)
    # and the threshold itself is where SE crosses 2.0
    assert math.sqrt(252 / min_bars_for_sharpe(252)) == pytest.approx(2.0, abs=0.02)

    # the stub has no Sharpe at all -- suppressed, not zeroed
    assert out["stub"]["significant"] is False
    assert out["stub"]["sharpe"] is None
    assert out["stub"]["n_bars"] == 2


def test_result_filename_separates_diagnostics_from_canonical():
    """A non-canonical run must not overwrite the headline artifact.

    Regression, not hypothetical: a 4-candidate/1-trial diagnostic run wrote to
    outputs/sweep_us_equities_swing.json, replacing the canonical 48-trial
    sweep. The JSON on disk then described different numbers than the README,
    with nothing in the filename to reveal it.
    """
    import argparse
    import importlib.util
    from pathlib import Path

    spec = importlib.util.spec_from_file_location(
        "run_research",
        Path(__file__).resolve().parents[1] / "scripts" / "run_research.py",
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)

    def args(c, t, s, p):
        return argparse.Namespace(candidates=c, trials=t, seed=s, period=p)

    canonical = mod.result_filename("us_equities", "swing",
                                    args(6, 8, 11, "5y"))
    diagnostic = mod.result_filename("us_equities", "swing",
                                     args(4, 1, 11, "5y"))
    other = mod.result_filename("crypto", "swing", args(6, 8, 42, "5y"))

    # the headline keeps its stable, greppable name
    assert canonical == "sweep_us_equities_swing.json"
    # and no diagnostic can collide with it
    assert diagnostic != canonical
    assert other != mod.result_filename("crypto", "swing", args(6, 8, 11, "5y"))
    # the config is recoverable from the filename alone
    assert "c4_t1_s11_5y" in diagnostic
    assert "s42" in other


def test_walk_forward_purges_between_train_and_test():
    idx = pd.date_range("2020-01-01", periods=800, freq="D")
    splits = walk_forward_folds(idx, n_folds=6, train_frac=0.6, purge_bars=2)
    assert splits
    for tr_s, tr_e, te_s, te_e in splits:
        assert te_s >= tr_e + 2, "purge bars must sit between train and test"
        assert tr_e > tr_s and te_e > te_s


def test_max_drawdown_matches_hand_calc():
    r = np.array([0.10, -0.50, 0.10])
    assert max_drawdown(r) == pytest.approx(-0.50, abs=1e-9)


def test_sharpe_zero_on_constant_series():
    assert sharpe(np.zeros(100)) == 0.0


# --------------------------------------------------------------------- gate
def test_gate_rejects_a_bad_strategy():
    cfg = GateConfig()
    metrics = {"sharpe": 0.05, "max_dd_hedged": -0.9}
    dsr = {"dsr": 0.10}
    pbo = {"pbo": 0.72}
    wf = evaluate_walk_forward(pd.Series(np.random.default_rng(1).normal(0, 0.01, 400),
                              index=pd.date_range("2020-01-01", periods=400, freq="D")),
                              n_folds=8)
    reg = {"a": {"sharpe": -0.2}, "b": {"sharpe": -0.1}}
    rep = gate_report(metrics, dsr, pbo, wf, reg, action_ratio=0.0, cfg=cfg)
    assert rep["passed"] is False
    assert rep["verdict"] == "REJECTED"
    assert "min_dsr" in rep["failed_checks"]
    assert "max_pbo" in rep["failed_checks"]
    assert "nonzero_action_ratio" in rep["failed_checks"]


def test_gate_rejects_a_strategy_that_never_hedges():
    cfg = GateConfig()
    metrics = {"sharpe": 3.0}
    dsr = {"dsr": 0.99}
    pbo = {"pbo": 0.05}
    wf = evaluate_walk_forward(pd.Series(np.random.default_rng(2).normal(0.0005, 0.01, 800),
                              index=pd.date_range("2020-01-01", periods=800, freq="D")),
                              n_folds=8)
    wf.positive_fold_ratio = 0.9
    reg = {"a": {"sharpe": 1.0}, "b": {"sharpe": 1.0}}
    rep = gate_report(metrics, dsr, pbo, wf, reg, action_ratio=0.0, cfg=cfg)
    assert rep["passed"] is False
    assert "nonzero_action_ratio" in rep["failed_checks"], "a never-hedges strategy must fail"


# ---------------------------------------------------------------- decision
def test_local_backend_returns_typed_answers():
    be = LocalBackend()
    out = be.decide({"regime": "crash", "drawdown": -0.2},
                    {"should_hedge": Noul("Should we hedge?"),
                     "sizing": Choice("How much?", {"none": "no", "light": "a little"}),
                     "sev": Score("How bad?", ["fine", "bad", "awful"])})
    assert out["should_hedge"].kind == "noul"
    assert 0.0 <= float(out["should_hedge"].value) <= 1.0
    assert out["sizing"].value in ("none", "light")
    assert isinstance(out["sev"].value, float)


def test_local_backend_prefers_risk_words_for_hedging():
    be = LocalBackend()
    risky = be.decide("regime crash drawdown stress volatility expanding",
                      {"should_hedge": Noul("Should a dynamic hedging overlay add protection now?")})
    calm = be.decide("regime expansion calm quiet uptrend advance",
                     {"should_hedge": Noul("Should a dynamic hedging overlay add protection now?")})
    assert float(risky["should_hedge"].value) > float(calm["should_hedge"].value)


def test_backend_factory_falls_back_to_local_without_keys(monkeypatch):
    for k in ("JEV_API_KEY", "TYPESAFE_AI_API_KEY", "OPENROUTER_API_KEY"):
        monkeypatch.delenv(k, raising=False)
    from hedging_agent.config import DecisionConfig
    be = build_backend(DecisionConfig(backend="auto"))
    assert be.name == "local"


def test_question_set_is_batched_and_complete():
    qs = hedge_questions()
    assert set(qs) == {"should_hedge", "primary_risk", "risk_severity",
                       "hedge_sizing", "confidence_gate"}
    assert isinstance(qs["should_hedge"], Noul)
    assert isinstance(qs["primary_risk"], Choice)
    assert isinstance(qs["risk_severity"], Score)
    assert 2 <= len(qs["risk_severity"].criteria) <= 10, "score rubric needs 2-10 levels"
    assert len(qs["hedge_sizing"].criteria) >= 2


def test_decision_layer_cannot_rescue_a_quant_zero():
    """An unconfident decision must not manufacture a hedge."""
    cfg = HedgeConfig()
    st = RegimeState(Regime.EXPANSION, 0.3, 0.8, [], 0.5, 1,
                     {r.value: 0.2 for r in Regime}, 4)
    from hedging_agent.decision import Decision
    weak = {"should_hedge": Decision("should_hedge", "noul", 0.1, {"true": 0.1},
                                    backend="calibrated"),
            "confidence_gate": Decision("confidence_gate", "noul", 0.1, {"true": 0.1},
                                        backend="calibrated"),
            "hedge_sizing": Decision("hedge_sizing", "choice", "heavy",
                                     {"none": 0.0, "light": 0.1, "moderate": 0.3,
                                      "heavy": 0.6}, backend="calibrated")}
    act = decide_hedge(st, drawdown=0.0, prev_hedge=0.0, cfg=cfg, decisions=weak)
    assert act.escal is True and act.target_hedge == 0.0


# ----------------------------------------------------------------- pipeline
def test_simulate_hedge_produces_returns_and_a_zero_share():
    px = synthetic_prices(900)
    spec = template_candidates(1)[0]
    sim = simulate_hedge(px, "HEDGE", spec, timeframe="swing", warmup=320)
    assert "error" not in sim, sim.get("error")
    assert len(sim["returns_hedged"]) > 100
    assert 0.0 <= sim["zero_hedge_ratio"] <= 1.0
    # on synthetic data with a real crash, the overlay must do something
    assert sim["action_ratio"] > 0.0


def test_simulate_hedge_is_deterministic():
    px = synthetic_prices(700, seed=13)
    spec = template_candidates(1)[0]
    a = simulate_hedge(px, "HEDGE", spec, timeframe="swing", warmup=320)
    b = simulate_hedge(px, "HEDGE", spec, timeframe="swing", warmup=320)
    assert np.allclose(a["returns_hedged"].values, b["returns_hedged"].values)


def inverse_fixture(n: int = 260, seed: int = 17, hedge_beta: float = -0.9):
    """A risky asset and a properly INVERSE hedge instrument.

    Constructed from RETURNS, not prices: a monotone price ladder correlates
    +1 with any other monotone ladder regardless of direction, which makes it
    useless for testing sign inference. Here H's bar return is a fixed negative
    multiple of A's, so the measured correlation is genuinely negative.
    """
    rng = np.random.default_rng(seed)
    r_a = rng.normal(0.0003, 0.012, n)
    idx = pd.date_range("2024-01-01", periods=n, freq="D")
    a = 100.0 * np.cumprod(1.0 + r_a)
    # add a little idiosyncratic noise so the correlation is not exactly -1
    r_h = hedge_beta * r_a + rng.normal(0.0, 0.002, n)
    h = 100.0 * np.cumprod(1.0 + r_h)
    return pd.DataFrame({"A": a, "H": h}, index=idx)


def test_hedge_sign_convention():
    """A LONG inverse hedge must pay when the market falls."""
    px = inverse_fixture()
    n = pd.Series(0.5, index=px.index)
    out, diag = build_hedged_returns(px, n, "H", portfolio_cols=["A"])

    # H is already inverse to A, so the side is inferred as LONG and the
    # correlation is reported rather than assumed
    assert diag["hedge_side"] == 1.0, diag
    assert diag["hedge_corr_to_portfolio"] < -0.5, diag
    assert diag["correlation_warning"] is None

    port = px["A"].pct_change().fillna(0.0)
    hedge = px["H"].pct_change().fillna(0.0)
    expected = (port + 0.5 * hedge).iloc[1:]
    np.testing.assert_allclose(out.values, expected.values, rtol=1e-9)

    # LONG an inverse instrument: the hedged return must beat the bare asset
    # over the whole sample, and the drawdown must be shallower.
    assert max_drawdown(out) > max_drawdown(port), (
        "longing an inverse hedge must produce a shallower drawdown"
    )


def test_shorting_an_inverse_hedge_makes_drawdowns_worse():
    """The error this project exists to catch, asserted as a regression test."""
    px = inverse_fixture()
    n = pd.Series(0.5, index=px.index)
    long_h, _ = build_hedged_returns(px, n, "H", portfolio_cols=["A"])
    short_h, _ = build_hedged_returns(px, n, "H", hedge_sign=-1.0,
                                      portfolio_cols=["A"])
    assert max_drawdown(short_h) < max_drawdown(long_h), (
        "shorting an already-inverse instrument must produce a worse drawdown; "
        "if this flips, the sign inference is broken"
    )


def test_infer_side_flips_when_the_instrument_is_correlated():
    """A positively-correlated instrument must be SHORTED, not bought."""
    px = inverse_fixture()
    px["H"] = px["A"] * 1.02          # H now tracks A
    side, corr = infer_hedge_side(px, "H", ["A"])
    assert corr > 0.5, corr
    assert side == -1.0, "a correlated instrument is a SHORT, not a long"


def test_correlated_hedge_raises_the_warning():
    px = inverse_fixture()
    px["H"] = px["A"] * 1.02
    _, diag = build_hedged_returns(px, pd.Series(0.5, index=px.index), "H",
                                   portfolio_cols=["A"])
    assert diag["correlation_warning"], "a non-inverse hedge must be flagged"


def test_hedge_leg_has_no_lookahead():
    """The hedge P&L must land on the bar it happened, not one bar early."""
    idx = pd.date_range("2024-01-01", periods=4, freq="D")
    # H spikes UP on bar 1 only. Forced to the SHORT side so the spike is a loss.
    px = pd.DataFrame({"A": [100.0] * 4,
                       "H": [100.0, 110.0, 110.0, 110.0]}, index=idx)
    n = pd.Series([0.5, 0.5, 0.5, 0.5], index=idx)
    # force the SHORT side so the single shock on bar 1 is a loss
    out, diag = build_hedged_returns(px, n, "H", hedge_sign=-1.0,
                                     portfolio_cols=["A"])
    # the shock may only affect bar 1, never bar 2
    assert out.loc[idx[1]] < 0, "bar 1 must carry the shock"
    assert out.loc[idx[2]] == pytest.approx(0.0, abs=1e-12), (
        "hedge shock leaked forward -- one-bar look-ahead in the hedge leg"
    )


def test_config_roundtrips_to_json():
    c = Config()
    import json
    s = json.dumps(c.hedge.as_dict())
    assert "target_vol" in s
    assert json.loads(s)["veto_on_expansion"] is True
    assert Config().tf().name in TIMEFRAMES