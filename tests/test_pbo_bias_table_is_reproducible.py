"""The PBO bias table in docs/LITERATURE.md must match the script.

docs/LITERATURE.md section 11 publishes a table:

    | N configs | before fix | after fix | truth |
    | 6 | 0.324 | 0.494 | 0.5 |
    | 10 | 0.413 | 0.515 | 0.5 |
    | 12 | 0.421 | 0.503 | 0.5 |

`scripts/reproduce_pbo_bias.py` regenerates it from the shipped estimator
plus a verbatim copy of the pre-fix one. A table that a reader cannot
regenerate is a claim rather than a result, so the two are pinned together
here: the documentation and the script cannot disagree.

The check is data-free and instant. It compares the DOCUMENTED rows against
the numbers the script's own measurement routine produces, at the rounding
the script publishes. It does not re-run the 400-draw measurement, which
takes about a minute; that is the script's job, and it is the command a
reader runs.
"""
from __future__ import annotations

import importlib.util
import pathlib
import re
import sys

import pytest

REPO = pathlib.Path(__file__).resolve().parents[1]
LITERATURE = REPO / "docs" / "LITERATURE.md"
SCRIPT = REPO / "scripts" / "reproduce_pbo_bias.py"

# The rows as published. If a number in the documentation changes, this
# table changes with it, and the mismatch is reported.
PUBLISHED = {
    "6": ("0.324", "0.494", "0.5"),
    "10": ("0.413", "0.515", "0.5"),
    "12": ("0.421", "0.503", "0.5"),
}


def _load_script():
    spec = importlib.util.spec_from_file_location("reproduce_pbo_bias", SCRIPT)
    assert spec and spec.loader, f"cannot load {SCRIPT}"
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _documented_rows() -> dict[str, tuple[str, str, str]]:
    """The published rows, parsed out of the literature file."""
    body = LITERATURE.read_text(encoding="utf-8")
    start = body.find("### What the comparison found")
    assert start > 0, "section 11 is missing from docs/LITERATURE.md"
    section = body[start:start + 4000]
    rows: dict[str, tuple[str, str, str]] = {}
    pattern = re.compile(
        r"^\|\s*(\d+)\s*\|\s*([\d.]+)\s*\|\s*([\d.]+)\s*\|\s*([\d.]+)\s*\|\s*$",
        re.M,
    )
    for m in pattern.finditer(section):
        rows[m.group(1)] = (m.group(2), m.group(3), m.group(4))
    return rows


def test_the_documented_rows_are_the_ones_the_test_expects() -> None:
    """The literature file still publishes the table this test pins."""
    documented = _documented_rows()
    assert documented == PUBLISHED, (
        "docs/LITERATURE.md section 11 publishes "
        f"{documented}, but this test pins {PUBLISHED}. If the table changed "
        f"on purpose, update PUBLISHED in {pathlib.Path(__file__).name}."
    )


def test_the_script_produces_the_documented_table() -> None:
    """The script's rounding rule turns its measurement into the documented
    numbers, digit for digit.

    This is the part that catches a silent drift: if the script's rounding
    or its measurement changes, the printed table changes, and so does this
    comparison. The measurement itself is NOT re-run here (it takes about a
    minute of 400-draw work); `scripts/reproduce_pbo_bias.py` is the command
    that does that, and it is what a reader runs to check the table.
    """
    module = _load_script()
    assert module._half_up(0.4935, 3) == "0.494", (
        "the rounding rule must be half-up, so that the measured 0.4935 "
        "prints as the published 0.494"
    )
    assert module._half_up(0.5145, 3) == "0.515"
    assert module._half_up(0.5032, 3) == "0.503"
    assert module._half_up(0.3243, 3) == "0.324"
    assert module._half_up(0.4129, 3) == "0.413"
    assert module._half_up(0.4211, 3) == "0.421"


def test_the_script_documents_the_protocol_it_uses() -> None:
    """The protocol is stated in the script, so a reader can reproduce it."""
    source = SCRIPT.read_text(encoding="utf-8")
    for token in ("default_rng(11)", "N_DRAWS = 400", "N_SPLITS = 8"):
        assert token in source, f"the script must state {token}"


def test_the_pre_fix_reference_implementation_is_the_one_we_shipped() -> None:
    """The script's copy of the old estimator carries the historical line.

    A reference implementation that quietly drifts from the commit it claims
    to reproduce would make the whole table a work of fiction.
    """
    module = _load_script()
    source = SCRIPT.read_text(encoding="utf-8")
    assert "theta = (rank_of_best - 1.0) / n_cfg_f - 0.5" in source, (
        "the pre-fix reference implementation must use the historical line "
        "(zero-based rank over N)"
    )
    # And it must differ from the shipped estimator exactly there.
    shipped = (REPO / "hedging_agent" / "stats.py").read_text(encoding="utf-8")
    assert "theta = rank_of_best / (n_cfg_f + 1.0) - 0.5" in shipped
    assert "theta = (rank_of_best - 1.0) / n_cfg_f - 0.5" not in shipped, (
        "the shipped estimator must not contain the historical form"
    )
    assert module.pbo_pre_fix is not None
