"""The sweep artefact must not store DERIVED STATISTICS at full precision.

A committed sweep is the machine half of the published claim, so a reader
has to be able to regenerate it and compare. At full float precision that
failed: the committed `us_equities` sweep stored `t_stat`
2.7487817806516186 while a fresh run of the SAME commit produced
2.7487817806516133. The gap is 5e-15 and it changed no published claim,
but it made the byte comparison fail.

The cause was precision, not nondeterminism. Every other number the
repository publishes is rounded (`round(x, 4)`), and the significance
block was the one place that stored raw floats.

Two blocks still keep full precision on purpose, and both are stable
across runs and across machines:

  * `best_spec` holds the winning strategy's parameters. Those are draws
    from a seeded RNG and are exactly representable, so they are bit
    identical every time -- rounding them would LOSE information.
  * `regime_stats` holds per-regime statistics at six decimals. Their
    values are computed once and rounded at the source.

These tests therefore pin the block that was actually unstable.
"""

import json
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
SWEEPS = sorted((REPO / "outputs").glob("sweep_*.json"))

# Blocks whose full precision is intentional, with the reason.
FULL_PRECISION_ALLOWED = {
    "best_spec": "seeded-RNG parameter draws; exactly representable",
    "regime_stats": "rounded at the source to six decimals",
}


def test_there_are_sweeps_to_check():
    assert SWEEPS, "no committed sweep JSONs to check"


def _significant_digits(value: float) -> int:
    text = repr(float(value)).replace("-", "").replace(".", "").lstrip("0")
    return len(text)


@pytest.mark.parametrize("path", SWEEPS, ids=lambda p: p.name)
def test_the_significance_block_is_rounded(path):
    """The three fields that drifted must not keep more than 10 digits."""
    doc = json.loads(path.read_text(encoding="utf-8"))
    sig = doc["significance"]
    for key in ("mean_diff", "se", "t_stat"):
        value = sig[key]
        if value is None:
            continue
        assert _significant_digits(value) <= 10, (
            f"{path.name}: significance.{key} = {value!r} keeps "
            f"{_significant_digits(value)} significant digits"
        )


@pytest.mark.parametrize("path", SWEEPS, ids=lambda p: p.name)
def test_no_other_block_stores_a_raw_derived_statistic(path):
    """Only the documented blocks may keep full precision."""
    doc = json.loads(path.read_text(encoding="utf-8"))

    def leaves(obj, prefix=""):
        if isinstance(obj, dict):
            for key, value in obj.items():
                yield from leaves(value, f"{prefix}.{key}" if prefix else key)
        elif isinstance(obj, list):
            for i, value in enumerate(obj):
                yield from leaves(value, f"{prefix}[{i}]")
        elif isinstance(obj, float):
            yield prefix, obj

    offenders = []
    for path_in_doc, value in leaves(doc):
        block = path_in_doc.split(".")[0].split("[")[0]
        if block in FULL_PRECISION_ALLOWED:
            continue
        if value != 0.0 and _significant_digits(value) > 10:
            offenders.append((path_in_doc, value))
    assert not offenders, (
        f"{path.name} stores derived statistics at more than 10 significant "
        f"digits: {offenders[:5]}"
    )


@pytest.mark.parametrize("path", SWEEPS, ids=lambda p: p.name)
def test_the_significance_t_stat_matches_the_gate_value(path):
    """The file and the gate must not disagree about the same number.

    The promotion gate publishes `significance_t` rounded to four
    decimals and the research card prints that value. The significance
    block stores the same quantity. If they are stored differently, the
    artefact says two things about one number.
    """
    doc = json.loads(path.read_text(encoding="utf-8"))
    gate_value = next(c["value"] for c in doc["gate"]["checks"]
                      if c["check"] == "significance_t")
    assert doc["significance"]["t_stat"] == gate_value


@pytest.mark.parametrize("path", SWEEPS, ids=lambda p: p.name)
def test_the_significance_block_keeps_its_small_numbers_meaningful(path):
    """Rounding to DECIMAL places would print `se` as 0.0.

    `mean_diff` is ~1e-4 and `se` is ~7e-5, so they are rounded to
    significant digits. A zero here would be a silent loss of the
    standard error, and the gate would be reading a number that no
    longer carries its own uncertainty.
    """
    doc = json.loads(path.read_text(encoding="utf-8"))
    sig = doc["significance"]
    assert sig["se"] != 0.0, f"{path.name}: the standard error was rounded away"
    assert sig["mean_diff"] != 0.0, f"{path.name}: the mean difference was rounded away"
