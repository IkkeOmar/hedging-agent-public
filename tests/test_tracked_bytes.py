"""The tracked fixtures must match HEAD byte for byte.

This exists because `git status` could not see a real change.

A background capture script rewrote a committed fixture with
`pandas.DataFrame.to_csv()`, which writes CRLF on Windows. The
repository's `.gitattributes` says `* text=auto eol=lf`, so git
normalises the line endings on the way in:

    git status --short   -> clean
    git diff --stat      -> empty
    but the bytes on disk -> differ from HEAD

The `.gitattributes` that makes the published bytes platform-independent
is exactly what hid the change. The published sweeps are generated from
these fixtures, so a fixture that differs by a single byte is a
published number that cannot be reproduced.

So the check compares CONTENT, not git's opinion of it: the working-tree
file against `git show HEAD:<path>`. That is the only comparison that
sees past the normalisation.

The same check covers the committed sweep results, for the same reason:
they are the artefacts the cards are generated from, and a card that
cannot be regenerated from the committed sweep is a card that has
drifted.
"""

from __future__ import annotations

import hashlib
import subprocess
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]

TRACKED = [
    "hedging_agent/fixtures/crypto_1d_5y.csv",
    "hedging_agent/fixtures/mixed_1d_5y.csv",
    "hedging_agent/fixtures/us_equities_1d_5y.csv",
    "outputs/sweep_crypto_swing.json",
    "outputs/sweep_mixed_swing.json",
    "outputs/sweep_us_equities_swing.json",
    "cards/card_crypto_swing.md",
    "cards/card_mixed_swing.md",
    "cards/card_us_equities_swing.md",
]


def _head_bytes(path: str) -> bytes | None:
    proc = subprocess.run(
        ["git", "-C", str(REPO), "show", f"HEAD:{path}"],
        capture_output=True,
    )
    if proc.returncode != 0:
        return None
    return proc.stdout


@pytest.mark.parametrize("rel", TRACKED)
def test_the_working_tree_matches_head_byte_for_byte(rel: str):
    """The bytes on disk must equal the bytes in HEAD, exactly."""
    disk_path = REPO / rel
    assert disk_path.exists(), f"{rel} is missing from the working tree"

    head = _head_bytes(rel)
    if head is None:
        pytest.skip(f"{rel} is not tracked in HEAD (a new file)")

    disk = disk_path.read_bytes()
    assert disk == head, (
        f"{rel} differs from HEAD.\n"
        f"  working tree: {hashlib.sha256(disk).hexdigest()[:16]} "
        f"({len(disk)} bytes)\n"
        f"  HEAD        : {hashlib.sha256(head).hexdigest()[:16]} "
        f"({len(head)} bytes)\n"
        "  `git status` may say the file is unchanged: a .gitattributes "
        "normalisation hides line-ending differences. The published "
        "artefacts are generated from these files, so a byte difference "
        "means the published numbers cannot be reproduced."
    )


def test_the_fixtures_use_lf_line_endings():
    """The fixtures are CSV files read by pandas.

    CRLF is the pandas default on Windows, and it is what a capture
    script writes. The committed form is LF, and it must stay that way
    in the working tree too, because the working tree is where a reader
    runs the code.
    """
    for path in sorted((REPO / "hedging_agent" / "fixtures").glob("*.csv")):
        raw = path.read_bytes()
        n_crlf = raw.count(b"\r\n")
        assert n_crlf == 0, (
            f"{path.name} contains {n_crlf} CRLF line endings. The "
            "committed form is LF. A capture script's to_csv() writes CRLF "
            "on Windows; write the file with newline='\\n' instead."
        )
