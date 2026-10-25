"""The sweep artefact must record the window it was measured over.

A Sharpe without the window it was measured in is not a result, it is a
number. The gap was visible in the file names: `crypto_1d_5y.csv` holds
1000 daily bars, which is 2.74 years, while the other two fixtures hold
five years. The crypto provider caps at 1000 bars, so a 5y request
silently returned less and the artefact did not say so.

The window is read from the DATA, not from the request: a request is
not a delivery.
"""

import json
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
SWEEPS = sorted((REPO / "outputs").glob("sweep_*.json"))


def test_there_are_sweeps_to_check():
    assert SWEEPS, "no committed sweep JSONs to check"


@pytest.mark.parametrize("path", SWEEPS, ids=lambda p: p.name)
def test_the_window_is_recorded(path):
    doc = json.loads(path.read_text(encoding="utf-8"))
    window = doc.get("window")
    assert window, f"{path.name} does not record the window it was measured over"
    for key in ("start", "end", "n_bars", "years"):
        assert key in window, f"{path.name}: the window has no {key}"


@pytest.mark.parametrize("path", SWEEPS, ids=lambda p: p.name)
def test_the_window_dates_are_ordered_and_the_bars_are_counted(path):
    doc = json.loads(path.read_text(encoding="utf-8"))
    window = doc["window"]
    assert window["start"] < window["end"], f"{path.name}: the window runs backwards"
    assert window["n_bars"] > 0
    # the years must be consistent with the dates, not asserted separately
    from datetime import date
    d0 = date.fromisoformat(window["start"])
    d1 = date.fromisoformat(window["end"])
    years = round((d1 - d0).days / 365.25, 2)
    assert window["years"] == years, (
        f"{path.name}: the window says {window['years']} years but its dates "
        f"span {years}"
    )


@pytest.mark.parametrize("path", SWEEPS, ids=lambda p: p.name)
def test_a_short_window_says_why(path):
    """A window materially shorter than the request must be explained.

    The crypto provider caps at 1000 daily bars, so a 5y request covers
    2.74 years there. A reader who is not told cannot tell that a
    five-year equity result is being compared with a 2.74-year crypto
    one.
    """
    doc = json.loads(path.read_text(encoding="utf-8"))
    window = doc["window"]
    if window["years"] < 4.0:
        assert window.get("note"), (
            f"{path.name}: the window covers {window['years']} years of a "
            f"{window['requested_period']} request and does not say why"
        )
        assert "bars" in window["note"]


@pytest.mark.parametrize("path", SWEEPS, ids=lambda p: p.name)
def test_the_window_matches_the_committed_fixture(path):
    """The recorded window must match the price file the sweep ran on.

    Otherwise the artefact states a window the data does not have, which
    is worse than stating none.
    """
    doc = json.loads(path.read_text(encoding="utf-8"))
    window = doc["window"]
    universe = doc["universe"]
    fixture = REPO / "hedging_agent" / "fixtures" / f"{universe}_1d_5y.csv"
    if not fixture.exists():
        pytest.skip(f"no committed fixture for {universe}")
    with fixture.open(encoding="utf-8") as fh:
        header = fh.readline()
        first = fh.readline().split(",")[0][:10]
        n = 1
        for _ in fh:
            n += 1
    assert window["start"] == first, (
        f"{path.name}: the window starts {window['start']} but the fixture "
        f"starts {first}"
    )
    assert window["n_bars"] == n, (
        f"{path.name}: the window counts {window['n_bars']} bars but the "
        f"fixture has {n}"
    )
