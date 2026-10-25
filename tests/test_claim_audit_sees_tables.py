"""The claim audit must be able to SEE the numbers in a results table.

The audit's prose pattern needs a unit word after the number -- "10
checks" -- so it never saw `| Newey-West HAC t | 2.7150 | > 3.00 |`. Every
results table in this repository is written that way, which meant the
numbers a reader actually reads were the one kind of claim the check
could not see. And it did not: the README carried a table of numbers
from a superseded sweep generation for long enough to be published.

These tests pin the two properties that make the check work:

  * a number in a table cell is screened, at the cell's own precision;
  * a stale number in a table FAILS the audit.

The second is the one that matters. A check that cannot fail is not a
check.
"""

import importlib.util
import json
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]


def _load_audit():
    spec = importlib.util.spec_from_file_location(
        "audit_claims_under_test", REPO / "scripts" / "audit_claims.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def audit():
    return _load_audit()


def test_a_table_row_is_split_into_its_cells(audit):
    """A results table has four columns and a universe table has five.

    A parser that only reads four-column rows silently ignores every
    number in the second table.
    """
    row5 = "| crypto | 0.2124 / 0.0841 | 0.2857 | 0.4692 | REJECTED (5 failed) |"
    cells = [c for c, _ in audit.table_cells(row5)]
    assert "0.2124 / 0.0841" in cells
    assert "0.4692" in cells
    assert len(cells) == 5


def test_a_separator_row_contributes_no_cells(audit):
    assert audit.table_cells("|---|---|---|---|") == []


def test_a_number_in_a_cell_is_found_at_the_cells_precision(audit):
    cells = [c for c, _ in audit.table_cells("| min Sharpe | 1.2751 | >= 0.60 | PASS |")]
    assert "1.2751" in cells


def test_a_stale_number_in_a_table_fails_the_audit(audit, tmp_path):
    """The negative case. A stale number must FAIL, not pass quietly.

    The README's table is the real case this exists for: it carried the
    numbers of a superseded sweep generation, and the audit was green.
    """
    known = audit.harvest_sweeps(REPO / "outputs")
    assert known, "no sweep artefacts to check against"

    stale_text = {
        "README.md": (
            "| check | value | threshold | verdict |\n"
            "|---|---|---|---|\n"
            "| min Sharpe | 1.2751 | >= 0.60 | PASS |\n"
            "| Newey-West HAC t | 2.7150 | > 3.00 | **FAIL** |\n"
        )
    }
    problems, screened = audit.check_numbers(stale_text, known)
    assert problems, "a stale number in a table passed the audit"
    assert screened >= 2
    joined = " ".join(problems)
    assert "1.2751" in joined
    assert "2.7150" in joined


def test_the_current_numbers_pass(audit):
    """The same shape, with the values the artefacts actually hold."""
    known = audit.harvest_sweeps(REPO / "outputs")
    current = json.loads(
        (REPO / "outputs" / "sweep_us_equities_swing.json").read_text(encoding="utf-8"))
    text = {
        "README.md": (
            "| check | value | threshold | verdict |\n"
            "|---|---|---|---|\n"
            f"| min Sharpe | {current['best_metrics']['sharpe']} | >= 0.60 | PASS |\n"
            f"| Newey-West HAC t | {current['significance']['t_stat']} | > 3.00 | **FAIL** |\n"
        )
    }
    problems, _ = audit.check_numbers(text, known)
    assert not problems, problems


def test_an_external_reference_is_not_required_to_be_traceable(audit):
    """A row quoting a published reference states THAT reference's numbers.

    Bailey's printed example and ml4t's noise levels are not claims about
    this repository, so they must not be traced to this repository's
    artefacts -- and a check that demands it would cry wolf on a correct
    document.
    """
    known = audit.harvest_sweeps(REPO / "outputs")
    text = {
        "README.md": (
            "| stat | reference | result |\n"
            "|---|---|---|\n"
            "| DSR | Bailey & Lopez de Prado printed example | `SR0 = 0.1132`, `DSR = 0.9004` exactly |\n"
            "| PBO | `ml4t/diagnostic` `compute_pbo` | unbiased under noise to 0.494/0.515/0.503 vs true 0.5 |\n"
        )
    }
    problems, _ = audit.check_numbers(text, known)
    assert not problems, problems
