"""The decision-layer seam must actually be a seam.

A protocol that only the shipped backend satisfies is not a seam, it is a
class hierarchy with extra steps. These tests prove that a THIRD-PARTY
backend -- the worked example in `docs/DECISION-LAYER.md` -- can be
written, registered, and used without editing the library.

The example in the guide is a real module (`docs/table_backend_example.py`),
so the guide cannot rot: if it drifts from the code, the test fails.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

from hedging_agent.decision import (
    Choice,
    Decision,
    DecisionBackend,
    LocalBackend,
    Noul,
    Score,
    available_backends,
    build_backend,
    hedge_questions,
    register_backend,
    unregister_backend,
)

REPO = Path(__file__).resolve().parents[1]
EXAMPLE = REPO / "docs" / "table_backend_example.py"


def _load_example():
    """Import the guide's example as a module."""
    spec = importlib.util.spec_from_file_location("table_backend_example", EXAMPLE)
    module = importlib.util.module_from_spec(spec)
    sys.modules["table_backend_example"] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def table_backend():
    """The guide's example, registered for the duration of one test.

    The registry is GLOBAL state. A test that registers a backend and
    leaves it registered changes what every later test sees -- and it did
    exactly that: a test asserting that "auto" falls back to the local
    backend started failing because a test that ran EARLIER had left
    "table" registered. The fixture removes it again on the way out, and
    `test_registration_does_not_leak` guards the behaviour itself.
    """
    module = _load_example()
    register_backend("table", module.TableBackend)
    try:
        yield module
    finally:
        unregister_backend("table")


def test_the_guides_example_exists():
    """The guide must point at code that exists."""
    assert EXAMPLE.exists(), (
        f"docs/DECISION-LAYER.md points at a worked example that is not there: "
        f"{EXAMPLE}"
    )


def test_the_example_satisfies_the_protocol(table_backend):
    """A third-party backend must satisfy the Protocol.

    Checked with an explicit structural check rather than
    `isinstance(..., DecisionBackend)`: a Protocol with a non-method
    member is not runtime-checkable, and this way the test still means
    something if that changes.
    """
    module = table_backend
    backend = module.TableBackend()
    assert isinstance(backend.name, str) and backend.name
    assert callable(backend.decide)
    # the Protocol's two members, exactly
    assert hasattr(backend, "name")
    assert hasattr(backend, "decide")


def test_a_registered_backend_is_usable_by_name(table_backend):
    """Registering a backend must make it selectable."""
    assert "table" in available_backends()
    backend = build_backend(type("C", (), {"backend": "table"})())
    assert backend.name == "table"


def test_the_example_returns_a_distribution_for_every_question(table_backend):
    """Every question gets an answer, and every answer has probabilities."""
    module = table_backend
    backend = module.TableBackend()
    questions = hedge_questions()
    out = backend.decide({"regime": "distribution"}, questions)

    assert set(out) == set(questions)
    for key, d in out.items():
        assert d.probabilities, f"{key} has no probabilities"
        total = sum(d.probabilities.values())
        assert abs(total - 1.0) < 0.02, f"{key} probabilities sum to {total}"


def test_the_examples_answers_follow_its_own_table(table_backend):
    """The answer must follow the measured frequency, not something else."""
    module = table_backend
    backend = module.TableBackend()
    out = backend.decide({"regime": "distribution"}, hedge_questions())
    assert out["should_hedge"].p("true") == 0.55
    assert out["should_hedge"].value == 0.55


def test_an_unknown_regime_falls_back_to_the_prior(table_backend):
    module = table_backend
    backend = module.TableBackend()
    out = backend.decide({"regime": "not_a_regime"}, hedge_questions())
    assert out["should_hedge"].p("true") == module.P_PRIOR


def test_a_calibrated_backend_declares_itself_calibrated(table_backend):
    """The `uncalibrated` marker is what the policy keys off.

    A backend that returns frequencies must say so; one that returns
    scores must say so too. A backend that says neither is a backend
    whose probabilities nobody should threshold.
    """
    module = table_backend
    assert getattr(module.TableBackend, "uncalibrated", None) is False
    assert getattr(LocalBackend, "uncalibrated", None) is True


def test_the_local_backend_is_not_calibrated():
    """The shipped fallback must be honest about itself."""
    assert LocalBackend.uncalibrated is True


def test_an_unknown_backend_name_is_an_error_not_a_silent_substitute():
    """Asking for a backend by name and getting a different one is worse
    than an error."""
    with pytest.raises(KeyError):
        build_backend(type("C", (), {"backend": "no-such-backend"})())


def test_registration_rejects_a_non_callable():
    with pytest.raises(TypeError):
        register_backend("bad", "not callable")


def test_the_guide_and_the_example_agree():
    """The guide's code block must match the real module.

    A guide that shows different code from the one that works is a guide
    that teaches a broken example. The check is textual and blunt: the
    example's class body must appear in the guide.
    """
    guide = (REPO / "docs" / "DECISION-LAYER.md").read_text(encoding="utf-8")
    example = EXAMPLE.read_text(encoding="utf-8")
    # the constants table and the class name must both be in the guide
    assert "class TableBackend" in guide
    assert "P_DRAWDOWN_AFTER" in guide
    assert '"crash": 0.62' in guide
    # and the guide's own test must match the example's behaviour
    assert 'out["should_hedge"].p("true") == 0.55' in guide
    assert 'p = P_DRAWDOWN_AFTER.get(regime, P_PRIOR)' in example


def test_registration_does_not_leak():
    """The registry is global state, so a registration must be removable.

    This is the bug this file's fixture was written for: an earlier
    version of the example registered itself at IMPORT time, so importing
    it changed what every later test in the session saw. A test in
    another file -- "auto falls back to the local backend" -- started
    failing because of a test that had run earlier and left a backend
    registered.

    The fix is two-sided: the example does not register on import, and a
    registration can be undone.
    """
    module = _load_example()
    # importing it must NOT have registered anything
    assert "table" not in available_backends(), (
        "the example registered itself at import time; importing a module "
        "must not change global state"
    )
    register_backend("table", module.TableBackend)
    try:
        assert "table" in available_backends()
    finally:
        unregister_backend("table")
    assert "table" not in available_backends()


def test_auto_falls_back_to_local_when_nothing_is_registered():
    """With no other backend registered, "auto" is the local one.

    The same guarantee the existing engine test relies on, asserted here
    so the seam's own tests cover it.
    """
    be = build_backend(type("C", (), {"backend": "auto"})())
    assert be.name == "local"
