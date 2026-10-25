"""A path checker must not read a sentence's full stop as part of a path.

`docs/DECISION-LAYER.md` contains a Python comment inside a code block:

    # Keys are the regimes this repository uses; see hedging_agent/regime.py.

That sentence is correct English and the file is real. The path regex
used to capture the trailing full stop as part of the filename, so a
correct document failed a correct check and the audit CI job went red.

These tests pin both halves of the fix: the punctuation is not part of
the path, and a genuinely missing path is still caught.
"""

from __future__ import annotations

import importlib.util
import re
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]


def _load_audit_module():
    """Import scripts/audit_claims.py by path.

    It is a script, not a package module, and importing it runs its
    top-level code -- so it is loaded by path rather than by name.
    """
    path = REPO / "scripts" / "audit_claims.py"
    spec = importlib.util.spec_from_file_location("lab_audit_claims", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def audit():
    return _load_audit_module()


def _captured(audit, text: str) -> list[str]:
    """Every path the regex finds in a piece of text."""
    found = []
    for m in audit.PATH_RE.finditer(text):
        for group in m.groups():
            if group:
                found.append(group)
    return found


def test_a_trailing_full_stop_is_not_part_of_the_path(audit):
    """The sentence that broke the build must not break it again."""
    line = "# Keys are the regimes this repository uses; see hedging_agent/regime.py."
    captured = _captured(audit, line)
    assert "hedging_agent/regime.py" in captured, captured
    assert "hedging_agent/regime.py." not in captured, captured


def test_a_path_in_backticks_is_still_found(audit):
    """Backticked paths are the common case and must keep working."""
    captured = _captured(audit, "see `hedging_agent/policy.py`, not left to you")
    assert "hedging_agent/policy.py" in captured, captured


def test_a_bare_path_is_still_found(audit):
    """A path at the start of a line or after a space is found."""
    for line in (
        "hedging_agent/regime.py",
        "see hedging_agent/regime.py",
        "the file scripts/run_research.py",
        "(hedging_agent/decision.py)",
    ):
        captured = _captured(audit, line)
        assert captured, f"nothing captured from {line!r}"


def test_a_genuinely_missing_path_is_still_caught(audit, tmp_path):
    """The check must not have been weakened into uselessness.

    A path that does not exist is still reported, with a non-zero exit
    from the audit. This is the half of the fix that matters most: a
    checker that cannot fail is not a check.
    """
    doc = REPO / "docs" / "DECISION-LAYER.md"
    original = doc.read_text(encoding="utf-8")
    try:
        doc.write_text(
            original + "\n# See hedging_agent/definitely_not_here.py for details.\n",
            encoding="utf-8",
            newline="\n",
        )
        problems, _pending = audit.check_paths(
            {"docs/DECISION-LAYER.md": doc.read_text(encoding="utf-8")}
        )
        assert problems, "a genuinely missing path must be reported"
        assert any("definitely_not_here.py" in p for p in problems), problems
    finally:
        doc.write_text(original, encoding="utf-8", newline="\n")


def test_the_document_itself_is_clean(audit):
    """The real document must pass the real check.

    If this fails, the shipped guide names a path that does not exist.
    """
    problems, _pending = audit.check_paths({
        "docs/DECISION-LAYER.md": (REPO / "docs" / "DECISION-LAYER.md").read_text(
            encoding="utf-8"
        )
    })
    assert problems == [], problems
