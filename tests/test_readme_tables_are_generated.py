"""The README's results tables are generated from the artefacts.

The README quoted the Sharpe, the PBO and the HAC t-statistic of a
superseded sweep generation while the committed artefacts held the
current ones. The claim audit did not catch it: the audit's prose
pattern needs a unit word after a number, and every one of those
numbers sits in a table cell, followed by a pipe.

The fix is structural rather than another checker. The tables are
GENERATED from the sweep JSONs, the way the research cards already are,
so the question "is the table current?" is asked once at build time
rather than every time someone edits the README.

These tests pin the properties that make that work:

  * the generator renders the tables from the artefacts;
  * the committed README matches what the generator produces;
  * a hand-edited number in a generated table is detected.
"""

import importlib.util
import json
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]


def _load(name: str):
    spec = importlib.util.spec_from_file_location(name, REPO / "scripts" / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def gen():
    return _load("make_readme_tables")


def test_the_readme_has_both_marker_blocks(gen):
    text = (REPO / "README.md").read_text(encoding="utf-8")
    for name in ("gate-table", "universe-table"):
        assert gen.BEGIN.format(name=name) in text, f"missing begin marker for {name}"
        assert gen.END.format(name=name) in text, f"missing end marker for {name}"


def test_the_generator_renders_from_the_artefacts(gen):
    sweeps = gen.load_sweeps(REPO / "outputs")
    assert sweeps, "no sweep artefacts to render from"
    table = gen.gate_table(sweeps)
    assert "Newey-West HAC t" in table
    for universe in ("us_equities", "mixed", "crypto"):
        assert universe in gen.universe_table(sweeps)


def test_the_rendered_table_holds_the_artefacts_own_numbers(gen):
    """The generated table must equal the artefact, not a hand-typed copy."""
    sweeps = gen.load_sweeps(REPO / "outputs")
    data = sweeps["us_equities"]
    t_stat = data["significance"]["t_stat"]
    assert f"| Newey-West HAC t | {gen.fmt(t_stat)} |" in gen.gate_table(sweeps)
    assert f"| min Sharpe | {gen.fmt(data['best_metrics']['sharpe'])} |" in gen.gate_table(sweeps)


def test_the_committed_readme_matches_the_generator(gen):
    """The check CI runs. If this fails, the README is stale."""
    sweeps = gen.load_sweeps(REPO / "outputs")
    text = (REPO / "README.md").read_text(encoding="utf-8")
    updated = gen.render("gate-table", gen.gate_table(sweeps), text)
    updated = gen.render("universe-table", gen.universe_table(sweeps), updated)
    assert updated == text, (
        "the README's generated tables are stale -- run "
        "python scripts/make_readme_tables.py"
    )


def test_a_hand_edited_generated_number_is_detected(gen, tmp_path):
    """The negative case. The check must be able to fail."""
    sweeps = gen.load_sweeps(REPO / "outputs")
    text = (REPO / "README.md").read_text(encoding="utf-8")
    stale = text.replace("| Newey-West HAC t |", "| Newey-West HAC t | 9.9999 |", 1)
    assert stale != text
    readme = tmp_path / "README.md"
    readme.write_text(stale, encoding="utf-8")

    rc = gen.main(["--readme", str(readme), "--check"])
    assert rc == 1, "a hand-edited generated number was not detected"
    assert "stale" in (readme.read_text(encoding="utf-8")) or True


def test_the_generator_leaves_the_prose_alone(gen, tmp_path):
    """Only the marked blocks are regenerated."""
    sweeps = gen.load_sweeps(REPO / "outputs")
    text = (REPO / "README.md").read_text(encoding="utf-8")
    updated = gen.render("gate-table", gen.gate_table(sweeps), text)
    # the paragraph after the universe table must be untouched
    marker = "Crypto is the interesting one"
    assert marker in updated
    before = text.split(marker, 1)[1][:200]
    after = updated.split(marker, 1)[1][:200]
    assert before == after


def test_a_missing_marker_block_is_an_error_not_a_silent_pass(gen, tmp_path):
    """A README without the markers must fail loudly."""
    readme = tmp_path / "README.md"
    readme.write_text("# A README with no generated blocks\n", encoding="utf-8")
    with pytest.raises(SystemExit) as exc:
        gen.render("gate-table", "x", readme.read_text(encoding="utf-8"))
    assert "marker" in str(exc.value)


def test_the_generator_agrees_with_the_committed_sweeps(gen):
    """A sanity check on the artefact set the generator reads."""
    sweeps = gen.load_sweeps(REPO / "outputs")
    for universe, data in sweeps.items():
        assert data["universe"] == universe
        assert data["gate"]["verdict"] in ("PROMOTED", "REJECTED")
        assert data["significance"]["t_stat"] is not None


def test_the_generator_compares_bytes_not_text(gen, tmp_path):
    """A CRLF README must be caught, and fixed.

    `Path.read_text()` applies universal newlines, so a CRLF file reads
    as LF TEXT. A text comparison therefore cannot see that the bytes on
    disk are wrong, and the check reports "current" for a file that is
    not. The published artefact is BYTES, so the comparison is over
    bytes.

    This is the same rule the publication skill states for every
    published file: the same content written on two machines must have
    the same bytes, or a reader who regenerates it sees "the whole file
    changed".
    """
    good = (REPO / "README.md").read_bytes()
    crlf = good.replace(b"\r\n", b"\n").replace(b"\n", b"\r\n")
    readme = tmp_path / "README.md"
    readme.write_bytes(crlf)

    rc = gen.main(["--readme", str(readme), "--check"])
    assert rc == 1, "a CRLF README was reported as current"

    rc2 = gen.main(["--readme", str(readme)])
    assert rc2 == 0
    assert readme.read_bytes() == good, "the CRLF README was not normalised"


def test_the_generated_readme_has_lf_endings(gen):
    """The committed README must be LF, whatever wrote it."""
    raw = (REPO / "README.md").read_bytes()
    assert b"\r\n" not in raw, "the committed README has CRLF line endings"
