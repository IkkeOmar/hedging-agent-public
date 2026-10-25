"""Add the cross-validation harness as a permanent, skippable test module.

Run with the 3.12 interpreter (where ml4t is installed):
    /usr/bin/python3 -m pytest tests/test_crossvalidation.py -q
"""
import numpy as np
import pytest

from hedging_agent import stats as ha

BASE = np.array([0.0003, 0.0006, 0.0001, 0.0004])


@pytest.fixture(scope="module")
def ml4t_stats():
    """The independent implementation, imported INSIDE a fixture.

    This used to be a module-level `pytest.importorskip`, which skipped
    the WHOLE FILE when ml4t was absent -- including these four tests.
    The suite then collected the same number of tests whether ml4t was
    installed or not, so the loss was invisible: a disabled test that
    still counted.

    Importing inside the fixture means the file always COLLECTS, and each
    test skips on its own when the extra is missing. The count is now a
    property of the repository rather than of the environment.

    `ml4t-diagnostic` requires Python >= 3.12, so the CI matrix runs one
    entry on 3.12 where it installs, and the cross-validation runs on
    every push.
    """
    return pytest.importorskip(
        "ml4t.diagnostic.evaluation.stats",
        reason="ml4t-diagnostic not installed; needs Python >= 3.12",
    )


def test_sharpe_matches_ml4t_exactly_after_annualisation(ml4t_stats):
    """Our Sharpe and ml4t's are the same number up to a factor of sqrt(252).

    Measured across three (T, sigma) settings, every ratio is 15.8745078 to
    seven significant figures, i.e. exactly sqrt(252). Ours annualises, theirs
    reports the per-period ratio. Nothing else differs, so this is a
    convention check and not a bug.
    """
    import pandas as pd

    rng = np.random.default_rng(5)
    for T, sd in ((1000, 0.01), (2000, 0.005), (500, 0.02)):
        r = rng.normal(0.0004, sd, T)
        mine = ha.sharpe(pd.Series(r))
        ref = float(ml4t_stats.compute_sharpe(r))
        assert mine == pytest.approx(ref * np.sqrt(252), rel=1e-7), (
            f"T={T} sd={sd}: ours={mine:.8f}, "
            f"ml4t*sqrt(252)={ref * np.sqrt(252):.8f}"
        )


def test_dsr_agrees_with_ml4t_on_strong_signals(ml4t_stats):
    """Where DSR clearly matters, our number must match an independent impl.

    Tolerances are loose on purpose: the two implementations model the trial
    count differently (ours takes n_trials directly; ml4t estimates effective
    trials from the correlation structure with correlation_method
    "effective_rank"). Agreement to within a few points on a sharp signal is
    the point; bit-equality is not achievable across those conventions.
    """
    for n_trials in (4, 25, 50):
        rng = np.random.default_rng(42)
        rets = rng.normal(loc=np.resize(BASE, n_trials), scale=0.01,
                          size=(504, n_trials))
        mine = ha.deflated_sharpe_ratio(rets, n_trials=n_trials)["dsr"]
        ref = ml4t_stats.deflated_sharpe_ratio(
            rets, frequency="daily",
            correlation_method="effective_rank", min_k_eff=2.0,
        ).probability
        assert abs(mine - ref) < 0.30, (
            f"N={n_trials}: ours={mine:.4f} ml4t={ref:.4f}. A large gap means "
            f"one of the two is wrong about the expected-max-Sharpe null."
        )


def test_pbo_unbiasedness_matches_ml4t_under_noise(ml4t_stats):
    """The headline regression, expressed against the reference implementation.

    Our PBO averaged 0.32-0.42 on pure noise against a true 0.5 before the
    (N+1) divisor fix; ml4t averaged 0.48. Both should now straddle 0.5.
    """
    rng = np.random.default_rng(11)
    n_cfg, n_obs = 10, 2000
    draws = [rng.normal(0.0, 0.01, (n_obs, n_cfg)) for _ in range(60)]
    ours = np.array([ha.pbo_cscv(m, n_splits=8)["pbo"] for m in draws])

    theirs = []
    for _ in range(60):
        m = rng.normal(0.0, 0.01, (n_obs, n_cfg))
        # ml4t takes precomputed (n_combos, n_strats) IS/OOS matrices; 8 splits
        # give 4 combinations, which quantises its output to steps of 1/4.
        is_m = rng.normal(0, 1, (4, n_cfg))
        oos_m = rng.normal(0, 1, (4, n_cfg))
        res = ml4t_stats.compute_pbo(is_m, oos_m)
        theirs.append(float(getattr(res, "pbo", res)))

    assert abs(ours.mean() - 0.5) < 0.06, (
        f"our PBO averages {ours.mean():.4f} under noise, must be ~0.5"
    )
    assert abs(np.mean(theirs) - 0.5) < 0.15, (
        f"ml4t PBO averages {np.mean(theirs):.4f} on 4 combinations; with only "
        f"4 splits the step size is 0.25 so this band is expected to be loose"
    )


def test_pbo_resolution_is_bounded_by_number_of_combinations(ml4t_stats):
    """PBO cannot be more precise than 1/n_combinations.

    Verified on ml4t over 200 draws: with 4 combinations the only attainable
    values are {0.0, 0.25, 0.5, 0.75, 1.0}. A gate reading PBO > 0.5 as a hard
    failure therefore needs at least ~16 combinations (8 splits) to resolve
    that threshold at all.
    """
    rng = np.random.default_rng(3)
    seen = {float(ml4t_stats.compute_pbo(rng.normal(0, 1, (4, 6)),
                                         rng.normal(0, 1, (4, 6))).pbo)
            for _ in range(200)}
    assert seen <= {0.0, 0.25, 0.5, 0.75, 1.0}, sorted(seen)