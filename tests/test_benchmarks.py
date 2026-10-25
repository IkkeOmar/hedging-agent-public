"""Validation of the HAC t-stat against statsmodels, and the benchmark suite.

The Newey-West implementation is checked against statsmodels' cov_hac on a
generated autocorrelated series. Two independent implementations agreeing is
evidence; my own code agreeing with my own code is not.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from hedging_agent.benchmarks import (
    benchmark_table, buy_and_hold, constant_hedge, drawdown_control,
    inverse_vol_overlay, trend_following,
)
from hedging_agent.stats import (
    max_drawdown, newey_west_tstat, sharpe, sharpe_difference_tstat,
)


# ---------------------------------------------------------------- HAC checks
def _statsmodels_hac_t(x: np.ndarray, y: np.ndarray, lags: int) -> tuple[float, float]:
    """Reference HAC t and se from statsmodels on the mean of the difference.

    NOTE: build the design matrix as an explicit (n,1) array of ones.
    sm.add_constant() on a 1-D array does NOT prepend an intercept column, so
    statsmodels would silently t-test a zero-slope regression instead of the
    mean. That mistake cost one debug cycle and is pinned here so nobody
    repeats it.
    """
    sm = pytest.importorskip("statsmodels.api")
    diff = x - y
    X = np.ones((len(diff), 1))
    res = sm.OLS(diff, X).fit(
        cov_type="HAC",
        cov_kwds={"maxlags": int(lags), "use_correction": False},
    )
    return float(res.tvalues[0]), float(res.bse[0])


@pytest.mark.parametrize("seed", [1, 7, 13, 42])
@pytest.mark.parametrize("lags", [0, 3, 7])
def test_newey_west_matches_statsmodels(seed, lags):
    rng = np.random.default_rng(seed)
    n = 400
    # autocorrelated difference: exactly the case HAC exists for
    e = rng.normal(0, 0.01, n)
    d = np.zeros(n)
    for i in range(1, n):
        d[i] = 0.45 * d[i - 1] + e[i]
    x = 0.0005 + d + rng.normal(0, 0.005, n)
    y = rng.normal(0.0002, 0.01, n)

    mine = newey_west_tstat(x, y, lags=lags)
    theirs_t, theirs_se = _statsmodels_hac_t(x, y, lags)
    assert mine["se"] == pytest.approx(theirs_se, rel=1e-9, abs=1e-15), (
        f"seed={seed} lags={lags}: HAC se {mine['se']} vs statsmodels {theirs_se}"
    )
    assert mine["t_stat"] == pytest.approx(theirs_t, rel=1e-9, abs=1e-12), (
        f"seed={seed} lags={lags}: HAC t {mine['t_stat']} vs statsmodels {theirs_t}"
    )


def test_newey_west_ignores_lags_when_iid():
    rng = np.random.default_rng(3)
    n = 2000
    x = rng.normal(0.001, 0.01, n)
    y = rng.normal(0.0, 0.01, n)
    a = newey_west_tstat(x, y, lags=0)
    b = newey_west_tstat(x, y, lags=10)
    assert a["t_stat"] == pytest.approx(b["t_stat"], rel=0.25), (
        "under iid, adding HAC lags should barely move the t-stat"
    )


def test_newey_west_penalises_persistent_autocorrelation():
    rng = np.random.default_rng(11)
    n = 600
    x = np.zeros(n)
    for i in range(1, n):                     # random walk: strong autocorrelation
        x[i] = x[i - 1] + rng.normal(0.0006, 0.01)
    y = np.zeros(n)
    naive = newey_west_tstat(x, y, lags=0)
    hac = newey_west_tstat(x, y)
    assert not np.isnan(hac["t_stat"])
    assert abs(naive["t_stat"]) > abs(hac["t_stat"]), (
        "HAC must shrink the t-stat on a strongly autocorrelated difference"
    )


def test_tstat_is_too_few_to_measure():
    out = newey_west_tstat(np.array([1.0, 2, 3, 4, 5]), np.array([1.0, 2, 3]))
    assert np.isnan(out["t_stat"]), "a 3-observation HAC t-stat must not be printed"
    assert out["n_obs"] == 3


def test_tstat_right_aligns_unequal_lengths():
    """Unequal lengths must align on the most recent bars, never crash."""
    rng = np.random.default_rng(3)
    x = rng.normal(0.001, 0.01, 500)
    y = rng.normal(0.0, 0.01, 300)
    out = newey_west_tstat(x, y)
    assert out["n_obs"] == 300
    ref = newey_west_tstat(x[-300:], y)
    assert out["t_stat"] == pytest.approx(ref["t_stat"], rel=1e-12)


def test_tstat_handles_nan_pairs():
    x = np.array([0.01, np.nan, 0.02, 0.03, np.nan] + [0.01] * 30)
    y = np.array([0.00, 0.01, 0.02, np.nan, 0.00] + [0.00] * 30)
    out = newey_west_tstat(x, y)
    assert out["n_obs"] == 32, out
    assert not np.isnan(out["t_stat"])


def test_sharpe_difference_reports_both_sharpes_and_the_gap():
    rng = np.random.default_rng(5)
    n = 750
    base = rng.normal(0.0003, 0.01, n)
    better = base + rng.normal(0.0, 0.004, n)
    out = sharpe_difference_tstat(better, base)
    assert out["sharpe_hedged"] > out["sharpe_unhedged"]
    assert out["sharpe_gap"] == pytest.approx(
        out["sharpe_hedged"] - out["sharpe_unhedged"], abs=1e-4)
    assert out["hac"] == "Newey-West Bartlett"
    assert out["n_obs"] == n


# ----------------------------------------------------------- benchmark suite
def _fixture(n=900, seed=19):
    rng = np.random.default_rng(seed)
    p = rng.normal(0.0004, 0.011, n)
    h = -0.85 * p + rng.normal(0.0, 0.002, n)
    return p, h


def test_buy_and_hold_is_the_identity():
    p, _ = _fixture()
    np.testing.assert_allclose(buy_and_hold(p), p)


def test_constant_hedge_reduces_vol_when_the_instrument_is_inverse():
    p, h = _fixture()
    c = constant_hedge(p, h, 0.5)
    assert np.std(c) < np.std(p), "a constant 50% inverse hedge must cut volatility"


def test_every_benchmark_is_a_series_on_the_same_grid():
    p, h = _fixture()
    cand = constant_hedge(p, h, 0.5)
    out = benchmark_table(p, h, cand, name="candidate")
    n_bars = {v["n_bars"] for v in out["table"].values()}
    assert len(n_bars) == 1, f"benchmarks compared on different samples: {out['table']}"
    assert len(out["table"]) == 7, sorted(out["table"])
    assert "risk_matched_constant" in out["table"]


def test_verdict_counts_only_simple_benchmarks():
    p, h = _fixture()
    out = benchmark_table(p, h, constant_hedge(p, h, 0.5), name="candidate")
    v = out["verdict"]
    assert v["n_benchmarks"] == 6
    assert 0 <= v["n_benchmarks_beaten_on_sharpe"] <= 6


def test_trend_following_uses_no_future_information():
    """Truncating the series must not change any earlier bar's value."""
    p, h = _fixture(n=600)
    full = trend_following(p, h, 200)
    cut = trend_following(p[:400], h[:400], 200)
    np.testing.assert_allclose(full[:400], cut, rtol=1e-12, atol=1e-15)


def test_trend_following_only_holds_the_hedge_below_the_moving_average():
    p, h = _fixture(n=500)
    out = trend_following(p, h, lookback=50)
    p_s = pd.Series(p)
    ma = p_s.rolling(50).mean().shift(1)
    for i in range(60, len(out)):
        expected = 0.0 if p_s.iloc[i] >= ma.iloc[i] else 0.5
        got = (out[i] - p_s.iloc[i]) / h[i] if abs(h[i]) > 1e-12 else 0.0
        if abs(h[i]) > 1e-12:
            assert got == pytest.approx(expected, abs=1e-9), f"bar {i}"


def test_drawdown_control_holds_only_in_drawdown():
    p, h = _fixture(n=500)
    out = drawdown_control(p, h, target_dd=0.10)
    curve = (1 + pd.Series(p)).cumprod()
    dd = curve / curve.cummax() - 1.0
    for i in range(len(out)):
        if abs(h[i]) > 1e-12:
            expected = 0.5 if dd.iloc[i] <= -0.10 else 0.0
            got = (out[i] - p[i]) / h[i]
            assert got == pytest.approx(expected, abs=1e-9), f"bar {i}"


def test_inverse_vol_scales_down_when_vol_is_high():
    p, h = _fixture(n=600)
    out = inverse_vol_overlay(p, h, lookback=20)
    p_s, h_s = pd.Series(p), pd.Series(h)
    rv = p_s.rolling(20).std().shift(1)
    # causal target: expanding median, NOT rv.median() over the whole sample
    target = rv.expanding(min_periods=20).median()
    for i in range(25, len(out)):
        if abs(h_s.iloc[i]) <= 1e-12:
            continue
        got = (out[i] - p_s.iloc[i]) / h_s.iloc[i]
        # before `lookback` realised-vol windows exist the target is undefined and
        # the benchmark deliberately falls back to a constant 0.5 hedge. That is
        # a design decision, not a NaN leak, so it is asserted explicitly.
        if not np.isfinite(target.iloc[i]):
            assert got == pytest.approx(0.5, abs=1e-9), (
                f"bar {i}: expected the constant-hedge fallback"
            )
        else:
            expected = 0.5 * float(min(target.iloc[i] / rv.iloc[i], 2.0))
            assert got == pytest.approx(expected, abs=1e-9), f"bar {i}"


def test_benchmarks_are_causal_in_the_past_sense():
    """A benchmark must never react to a bar before the one it is scoring.

    Truncate mid-sample: the shared prefix must be bit-identical. This is the
    strongest available statement of causality for a return-series strategy --
    if bar i's value depends only on bars <= i, then cutting the series at 400
    cannot change any bar before 400.
    """
    for fn in (trend_following, inverse_vol_overlay, drawdown_control):
        p, h = _fixture(n=800, seed=29)
        full = fn(p, h)
        cut = fn(p[:500], h[:500])
        np.testing.assert_allclose(full[:500], cut, rtol=1e-12, atol=1e-15), (
            f"{fn.__name__} changed its past when the future was removed"
        )


def test_no_benchmark_uses_a_whole_sample_statistic():
    """Guard against the exact bug this module was written to avoid.

    A full-sample median/std is a look-ahead in a time-ordered strategy. Each
    benchmark must be built from rolling/expanding windows only, so truncating
    the series cannot change any earlier value. This test is the reason
    inverse_vol_overlay's target is an EXPANDING median.
    """
    p, h = _fixture(n=900, seed=101)
    for name, fn in (("trend_following", trend_following),
                     ("inverse_vol_overlay", inverse_vol_overlay),
                     ("drawdown_control", drawdown_control)):
        full = fn(p, h)
        cut = fn(p[:600], h[:600])
        np.testing.assert_allclose(full[:600], cut, rtol=1e-12, atol=1e-15), (
            f"{name} depends on the whole sample -- that is look-ahead"
        )


def test_risk_matched_constant_carries_the_same_volatility():
    """The control must actually match risk, or it proves nothing."""
    from hedging_agent.benchmarks import risk_matched_constant_hedge
    p, h = _fixture(n=800, seed=53)
    cand = constant_hedge(p, h, 0.5)
    matched, ratio = risk_matched_constant_hedge(p, h, cand)
    assert 0.0 <= ratio <= 4.0
    assert float(np.std(matched, ddof=1)) == pytest.approx(
        float(np.std(cand, ddof=1)), rel=1e-6), (
        "risk-matched control does not carry the candidate's realised volatility"
    )


def test_risk_matched_inverts_relative_to_a_lonely_exposure():
    """Documenting the sign, because the obvious intuition is the WRONG way round.

    The hedged book's exposure is p + r*h with h inverse to p, so volatility
    FALLS as r rises toward the hedge's own beta and then rises again past it.
    Two consequences that a test must not get backwards:

      * a candidate LESS volatile than p*0.5 needs a LARGER r to match
        (typically it is below the minimum, so the solver returns the
        minimum-variance ratio itself);
      * r=0 is what matches an unhedged book, and it must come back as 0.

    The earlier version of this test asserted ratio_low < ratio_high, which is
    false for an inverse hedge and would have pinned the wrong behaviour.
    """
    from hedging_agent.benchmarks import risk_matched_constant_hedge
    p, h = _fixture(n=800, seed=53)
    _, r_full = risk_matched_constant_hedge(p, h, p)          # unhedged
    assert r_full == pytest.approx(0.0, abs=1e-6), (
        f"an unhedged book must be matched by NO hedge, got r={r_full}"
    )
    _, r_half = risk_matched_constant_hedge(p, h, constant_hedge(p, h, 0.5))
    assert r_half == pytest.approx(0.5, abs=1e-3)
    # more hedging -> less volatility, until the minimum
    vols = [float(np.std(constant_hedge(p, h, r), ddof=1)) for r in (0.0, 0.5, 1.0)]
    assert vols[0] > vols[1] > vols[2], f"not monotone decreasing toward the min: {vols}"


def test_risk_matched_ratio_is_a_whole_sample_estimator_by_design():
    """The matched hedge ratio is estimated once from the candidate's OWN history.

    It is a benchmarking statistic, not a tradable signal: it is computed after
    the fact from the candidate's realised volatility. So it is NOT expected to
    be invariant to truncation -- the full sample legitimately has a slightly
    different estimate. What IS required is that it depends only on the
    candidate's realised volatility, never on any other input.
    """
    from hedging_agent.benchmarks import risk_matched_constant_hedge
    p, h = _fixture(n=900, seed=61)
    cand = constant_hedge(p, h, 0.5)
    _, r_full = risk_matched_constant_hedge(p, h, cand)
    _, r_cut = risk_matched_constant_hedge(p[:600], h[:600], cand[:600])
    # the two agree closely: truncation must not cause a jump
    assert r_cut == pytest.approx(r_full, rel=0.02), (
        f"matched ratio unstable under truncation: {r_cut} vs {r_full}"
    )
    # and it is genuinely risk-matched on BOTH samples
    for k in (900, 600):
        m, _r = risk_matched_constant_hedge(p[:k], h[:k], cand[:k])
        assert float(np.std(m, ddof=1)) == pytest.approx(
            float(np.std(cand[:k], ddof=1)), rel=1e-4)


def test_risk_matched_ratio_solver_handles_non_monotonic_volatility():
    """std(p + r*h) is NOT monotone in r -- it has a minimum at the hedge's beta.

    A bisection-based solver converges to a ratio on the FAR side of that
    minimum and returns a control with the right volatility for entirely the
    wrong reason. This test pins the failure that actually happened.
    """
    from hedging_agent.benchmarks import _solve_matched_ratio, risk_matched_constant_hedge
    rng = np.random.default_rng(19)
    p = pd.Series(rng.normal(0.0004, 0.011, 900))
    h = pd.Series(-0.85 * p.values + rng.normal(0.0, 0.002, 900))
    vols = [float((p + r * h).std(ddof=1)) for r in (0.0, 0.5, 1.0, 1.5, 2.0)]
    assert vols[2] < vols[0] and vols[4] > vols[2], (
        f"fixture no longer has a volatility minimum; vols={vols}"
    )
    # matching the 0.5 constant hedge must return 0.5, not the far-side root
    cand = constant_hedge(p, h, 0.5)
    m, ratio = risk_matched_constant_hedge(p, h, cand)
    assert ratio == pytest.approx(0.5, abs=1e-3), (
        f"solver picked the wrong root: {ratio} (vol curve {vols})"
    )
    # and directly: solve for the 0.5 volatility
    assert _solve_matched_ratio(p, h, vols[1]) == pytest.approx(0.5, abs=1e-3)


def test_risk_matched_prefers_the_smallest_matching_ratio():
    from hedging_agent.benchmarks import _solve_matched_ratio
    rng = np.random.default_rng(19)
    p = pd.Series(rng.normal(0.0004, 0.011, 900))
    h = pd.Series(-0.85 * p.values + rng.normal(0.0, 0.002, 900))
    # a volatility between the r=0.25 and r=0.75 levels has TWO solutions on the
    # curve; we must take the lower (least hedging)
    target = float((p + 0.4 * h).std(ddof=1))
    r = _solve_matched_ratio(p, h, target)
    assert r < 0.45, f"expected the lower root near 0.4, got {r}"
    assert float((p + r * h).std(ddof=1)) == pytest.approx(target, rel=1e-6)


def test_risk_matched_handles_a_perfect_hedge():
    """A candidate less volatile than ANY constant hedge must not crash."""
    from hedging_agent.benchmarks import risk_matched_constant_hedge
    rng = np.random.default_rng(5)
    n = 500
    p = pd.Series(rng.normal(0.0004, 0.011, n))
    h = pd.Series(-1.0 * p.values)                 # exact inverse
    near_zero = p * 0.02                           # almost perfectly hedged
    m, ratio = risk_matched_constant_hedge(p, h, near_zero)
    assert np.isfinite(ratio)
    assert float(np.std(m, ddof=1)) > 0


def test_volatility_drag_matches_the_sigma_squared_over_two_identity():
    """drag = arith_mean - cagr, and the volatility tax is ~= sigma^2/2.

    Definition checked against Wikipedia "Volatility tax" / Ashburn (2003) /
    Spitznagel: geometric mean ~= mu - sigma^2/2, so the tax is sigma^2/2.

    We use the LOG-EQUIVALENT growth rate exp(E[log(1+r)])*periods - 1,
    because that is the convention under which the identity actually holds.
    The POWER convention (QuantStats/empyrical: prod(1+r)**(1/years)-1)
    overstates growth when volatility is high and is reported separately as
    `power_cagr`.
    """
    from hedging_agent.benchmarks import volatility_drag
    rng = np.random.default_rng(0)
    r = rng.normal(0.0010, 0.012, 5040)          # 20y of daily returns
    out = volatility_drag(r, periods=252)
    mu = r.mean() * 252
    sig = r.std(ddof=1) * np.sqrt(252)

    assert out["drag"] > 0, "variance must cost compounding (drag > 0)"
    assert out["arith_mean"] == pytest.approx(mu, abs=1e-6)
    assert out["vol"] == pytest.approx(sig, abs=1e-6)

    # THE identity, and the bug this test exists to catch: the tax is defined
    # against the LOGARITHMIC mean, NOT against exp(log mean) - 1. Differencing
    # against the exponentiated form reports a NEGATIVE tax on a sample that
    # has a genuinely positive one, because exp() re-adds the drag.
    log_cagr = float(np.mean(np.log1p(r))) * 252
    assert out["cagr_log"] == pytest.approx(log_cagr, abs=1e-6)
    assert out["drag"] == pytest.approx(mu - log_cagr, abs=1e-6), (
        "drag must be arith_mean minus the LOG mean"
    )
    assert out["cagr"] == pytest.approx(np.exp(log_cagr) - 1.0, abs=1e-6)
    # cagr (simple, exponentiated) sits ABOVE the arithmetic mean here: that is
    # not an error, it is why the tax must be taken in log space.
    assert out["cagr"] > out["arith_mean"]
    # and the tax matches sigma^2/2 to the third moment
    assert out["drag"] == pytest.approx(sig**2 / 2, abs=0.001), (
        f"tax {out['drag']} vs sigma^2/2 {sig**2/2}"
    )
    assert abs(out["drag_residual"]) < 0.001
    # the power convention is the most optimistic of the three
    assert out["power_cagr"] >= out["cagr"] - 1e-9


def test_power_and_exp_conventions_are_mathematically_identical():
    """CORRECTION to an earlier, wrong claim in this file.

    An earlier version of the docstring asserted that the QuantStats/empyrical
    "power" convention (prod(1+r)**(periods/n) - 1) OVERSTATES growth relative
    to exp(mean(log(1+r))*periods) - 1, and that the gap widens with volatility.
    Both halves of that are false. Measured across sd = 0.002 .. 0.030:

        sd      log_cagr   cagr=exp-1   power        power - cagr
        0.0020   0.24213    0.27396     0.27396     0.00000
        0.0120   0.17851    0.19543     0.19543     0.00000
        0.0200   0.10974    0.11599     0.11599     0.00000

    prod(1+r)**(1/y) is exp((1/y) * sum(log(1+r))) = exp(E[log(1+r)]*periods).
    They are the same number to floating-point precision. What shrinks as
    volatility rises is log-CAGR ITSELF -- the volatility tax -- not a
    difference between conventions.

    So the volatility drag is arithmetic-mean minus geometric-mean. Nothing here
    needs a second convention to be explained, and no strategy is being
    flattered by picking one.
    """
    from hedging_agent.benchmarks import volatility_drag
    for sd in (0.002, 0.008, 0.020):
        rng = np.random.default_rng(7)
        out = volatility_drag(rng.normal(0.0010, sd, 5040))
        assert out["power_cagr"] == pytest.approx(out["cagr"], rel=1e-9), (
            f"power and exp conventions differ at sd={sd}: "
            f"{out['power_cagr']} vs {out['cagr']}"
        )


def test_volatility_tax_shrinks_as_vol_rises_for_a_fixed_arithmetic_mean():
    """The actual monotone relationship: higher vol -> lower log-CAGR."""
    from hedging_agent.benchmarks import volatility_drag
    log_cagrs = []
    for sd in (0.002, 0.008, 0.020):
        rng = np.random.default_rng(7)
        log_cagrs.append(volatility_drag(rng.normal(0.0010, sd, 5040))["cagr_log"])
    assert log_cagrs[0] > log_cagrs[1] > log_cagrs[2], (
        f"log-CAGR must fall as vol rises at constant mu, got {log_cagrs}"
    )


def test_volatility_drag_is_larger_for_a_more_volatile_book():
    """The whole point: the tax grows ~quadratically in sigma, so vol cuts pay."""
    from hedging_agent.benchmarks import volatility_drag
    rng = np.random.default_rng(4)
    low = rng.normal(0.0010, 0.006, 5040)        # sigma 0.095
    high = rng.normal(0.0010, 0.018, 5040)       # sigma 0.285
    d_low = volatility_drag(low)["drag"]
    d_high = volatility_drag(high)["drag"]
    assert d_high > d_low, f"higher vol must pay more tax: {d_high} vs {d_low}"
    # roughly quadratic: the ratio should be much bigger than the sigma ratio
    sig_lo = low.std(ddof=1) * np.sqrt(252)
    sig_hi = high.std(ddof=1) * np.sqrt(252)
    assert (d_high / d_low) > (sig_hi / sig_lo) ** 1.5, (
        "drag is not scaling ~quadratically in sigma"
    )


def test_zero_volatility_pays_no_tax():
    """A constant return series has no variance, hence no volatility tax.

    Note this requires periods == len(series). With periods=252 and a 500-bar
    series, exp(mean(log(1+r))*252) annualises a PARTIAL year of a constant
    return and the exponent step introduces a difference that is an
    annualisation artefact, not variance. That is why the assertion pins
    periods to the sample length.
    """
    from hedging_agent.benchmarks import volatility_drag
    r = np.full(500, 0.0004)
    out = volatility_drag(r, periods=500)
    # tolerance reflects the 6-decimal rounding in the returned dict, not
    # imprecision in the computation
    assert out["drag"] == pytest.approx(0.0, abs=1e-4), (
        f"a constant return series pays no tax, got {out['drag']}"
    )
    assert out["vol"] == pytest.approx(0.0, abs=1e-9)
    assert out["drag_sigma2_over_2"] == pytest.approx(0.0, abs=1e-12)


def test_benchmark_table_reports_the_drag_decomposition():
    p, h = _fixture(n=900, seed=19)
    out = benchmark_table(p, h, constant_hedge(p, h, 0.5), name="candidate")
    vd = out["vol_drag"]
    for key in ("candidate", "buy_and_hold", "risk_matched_constant",
                "drag_saved_vs_bnh", "vol_reduction", "cagr_gain_vs_bnh"):
        assert key in vd, sorted(vd)
    assert vd["candidate"]["drag"] < vd["buy_and_hold"]["drag"], (
        "hedging must REDUCE the volatility tax relative to buy-and-hold; "
        "drag is arith_mean - cagr so a lower-vol book pays less"
    )
    assert vd["vol_reduction"] > 0, "the hedge must actually cut realised vol"


def test_risk_matched_with_a_fixed_ratio_is_exactly_reproducible():
    """With the ratio pinned, the control is pure arithmetic and fully causal."""
    from hedging_agent.benchmarks import risk_matched_constant_hedge
    p, h = _fixture(n=900, seed=61)
    cand = constant_hedge(p, h, 0.5)
    m_full, r = risk_matched_constant_hedge(p, h, cand, hedge_ratio=0.75)
    m_cut, _ = risk_matched_constant_hedge(p[:600], h[:600], cand[:600],
                                            hedge_ratio=0.75)
    assert r == 0.75
    np.testing.assert_allclose(m_full[:600], m_cut, rtol=1e-12, atol=1e-15)


def test_trend_following_needs_no_warmup_to_reach_a_valid_state():
    p, h = _fixture(n=500)
    out = trend_following(p, h, lookback=200)
    assert np.isfinite(out).all()
    # before the MA exists there is no trend opinion, so no hedge
    assert (out[:200] == p[:200]).all(), "must not hold a hedge before any MA exists"
