"""Pin the exact mechanism of the PBO divisor bug we shipped and then fixed.

The published claim (README, docs/LITERATURE.md section 11, and
notes/KNOWN-TRAPS.md entry 1) is specific:

  * the pre-fix estimator scaled the out-of-sample rank by ``(rank - 1) / N``
    instead of Bailey & Lopez de Prado's ``rank / (N + 1)``;
  * the verdict is ``theta > 0``, so on EVEN N that flags one rank too few
    (N/2 - 1 instead of N/2) and PBO reads exactly 1/N low;
  * on ODD N the two forms agree and there is no bias at all.

Those are arithmetic claims about two formulas, so they are checked here
without any data, without any RNG, and without any statistics library. If
someone "fixes" the description in the docs but the formula in the code
drifts, these assertions fail.

The pre-fix formula is reproduced here as a local reference implementation
rather than imported, because the shipped code no longer contains it.
"""
from __future__ import annotations

import numpy as np
import pytest


def _fixed_theta(omega: float, n_cfg: int) -> float:
    """Bailey & Lopez de Prado: theta = omega / (N + 1) - 0.5."""
    return omega / (n_cfg + 1.0) - 0.5


def _pre_fix_theta(omega: float, n_cfg: int) -> float:
    """What we shipped before the fix: theta = (omega - 1) / N - 0.5."""
    return (omega - 1.0) / n_cfg - 0.5


def _flagged(theta_fn, n_cfg: int) -> int:
    """How many of the N ranks the verdict `theta > 0` puts above the line."""
    return sum(1 for omega in range(1, n_cfg + 1) if theta_fn(omega, n_cfg) > 0)


@pytest.mark.parametrize("n_cfg", [4, 6, 8, 10, 12, 20, 48])
def test_pre_fix_form_flags_one_rank_too_few_on_even_n(n_cfg: int) -> None:
    """The core mechanism: on even N we flagged N/2 - 1, not N/2."""
    assert n_cfg % 2 == 0, "this property is specific to even N"
    assert _flagged(_fixed_theta, n_cfg) == n_cfg // 2
    assert _flagged(_pre_fix_theta, n_cfg) == n_cfg // 2 - 1
    assert (
        _flagged(_fixed_theta, n_cfg) - _flagged(_pre_fix_theta, n_cfg) == 1
    ), "the pre-fix form must flag exactly one rank fewer"


@pytest.mark.parametrize("n_cfg", [3, 5, 7, 9, 11, 13])
def test_pre_fix_form_is_unbiased_on_odd_n(n_cfg: int) -> None:
    """On odd N the two forms agree exactly, so there is no bias at all.

    This is the part an earlier draft of the documentation got wrong: it
    claimed a uniform downward bias. The bias exists only on even N.
    """
    assert n_cfg % 2 == 1, "this property is specific to odd N"
    assert _flagged(_pre_fix_theta, n_cfg) == _flagged(_fixed_theta, n_cfg)
    assert _flagged(_fixed_theta, n_cfg) == (n_cfg - 1) // 2


@pytest.mark.parametrize(
    "n_cfg,expected_bias",
    [(4, 0.25), (6, 1 / 6), (10, 0.10), (12, 1 / 12), (48, 1 / 48)],
)
def test_the_bias_is_exactly_one_over_n(n_cfg: int, expected_bias: float) -> None:
    """The published magnitude: PBO reads exactly 1/N low on even N.

    The README and the trap ledger both say "0.25 at N=4". That number is
    only right if the bias is exactly 1/N, so the two are pinned together.
    """
    fixed = _flagged(_fixed_theta, n_cfg) / n_cfg
    pre_fix = _flagged(_pre_fix_theta, n_cfg) / n_cfg
    assert fixed - pre_fix == pytest.approx(expected_bias)
    assert fixed == pytest.approx(0.5), "the correct form is unbiased on even N"


def test_the_two_forms_differ_only_by_the_rank_shift() -> None:
    """The fix is the (N+1) divisor together with the 1-based rank.

    A reader who implemented the published description literally ("divide the
    rank by N instead of N+1") would have measured NO bias on even N, and the
    published claim would have looked false. This pins the difference to the
    two changes that actually distinguish the forms.
    """
    n_cfg = 10
    # omega/N - 0.5 is what a literal reading of the old description gives.
    def literal(omega: float, n: int) -> float:
        return omega / n - 0.5

    assert _flagged(literal, n_cfg) == n_cfg // 2, (
        "omega/N flags N/2 ranks on even N, so it is unbiased and cannot be "
        "the bug we shipped"
    )
    # The shipped form is the zero-based rank over N.
    assert _flagged(_pre_fix_theta, n_cfg) == n_cfg // 2 - 1


def test_the_shipped_estimator_uses_the_fixed_form() -> None:
    """The code must match the form the documentation describes.

    This is the assertion that stops the docs and the code drifting apart:
    the shipped estimator's rank-count is read back out of its own output.
    """
    from hedging_agent.stats import pbo_cscv

    # Pure noise: every configuration is exchangeable, so the winner is a
    # coin flip and the estimator must average 0.5 over many draws.
    rng = np.random.default_rng(20261025)
    values = [
        pbo_cscv(rng.normal(0.0, 0.01, (2000, 8)), n_splits=8)["pbo"]
        for _ in range(150)
    ]
    mean = float(np.mean(values))
    assert abs(mean - 0.5) < 0.06, (
        f"the shipped estimator averaged {mean:.4f} under noise; the "
        f"pre-fix form would have read about 0.375 on even N=8"
    )
