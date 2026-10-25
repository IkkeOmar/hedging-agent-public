"""The path checker must read a PATH, not a dotted identifier.

`gate.verdict` and `gate.passed` are JSON KEYS. The claim audit's path
pattern used to accept any backticked dotted identifier as a filename,
so a document that named a key of the sweep JSON was reported as
referencing a missing file. A correct document failed a correct check.

The rule, taken from the desk's checker, which already had it: a
backticked path must contain a slash, or be a file with a known
extension at the top level.

These tests pin both halves. A checker that cannot fail is not a
checker, so each one also asserts that a genuinely missing path IS
caught.
"""

import importlib.util
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]


def _load():
    spec = importlib.util.spec_from_file_location(
        "audit_claims_under_test", REPO / "scripts" / "audit_claims.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def audit():
    return _load()


def test_a_json_key_is_not_a_path(audit):
    paths = audit.referenced_paths("The card's `gate.verdict` and `gate.passed` are verbatim.")
    assert not paths, f"a JSON key was read as a path: {paths}"


def test_a_dotted_attribute_is_not_a_path(audit):
    paths = audit.referenced_paths("See `config.max_hedge_fraction` for the limit.")
    assert not paths, f"an attribute was read as a path: {paths}"


def test_a_real_path_is_a_path(audit):
    paths = audit.referenced_paths("Run `hedging_agent/strategies.py`.")
    assert "hedging_agent/strategies.py" in paths


def test_a_top_level_file_is_a_path(audit):
    paths = audit.referenced_paths("Read `README.md` and `docs/METHODOLOGY.md`.")
    assert "README.md" in paths
    assert "docs/METHODOLOGY.md" in paths


def test_a_missing_path_is_caught(audit, tmp_path):
    """The negative case: the checker must still find a dead reference."""
    text = "Run `hedging_agent/does_not_exist.py`."
    problems, _pending = audit.check_paths({"README.md": text})
    assert problems, "a missing path was not caught"
    assert "does_not_exist.py" in problems[0]
