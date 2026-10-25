"""The declared Python floor must be real.

`pyproject.toml` says `requires-python = ">=3.10"`. That is a promise to
every reader: this repository works on 3.10. A promise nobody checks is
just a number in a file.

The CI matrix runs 3.11 and 3.12, so a construct that only parses on 3.12
would pass CI and then break at a reader's install, where it costs them
the whole suite rather than one job.

The failure this guards against is specific. PEP 701 (Python 3.12) allows
a backslash inside an f-string's expression part:

    f"{data['a\\'b']}"        # a SyntaxError on 3.10 and 3.11

It is easy to write on 3.12, invisible in review, and fatal on
collection -- the whole suite dies at import time, not one test.

## Why this test invokes a real interpreter

The obvious tool, `ast.parse(source, feature_version=(3, 10))`, does NOT
catch this, and that was measured rather than assumed:

    feature_version=(3, 10): ACCEPTED
    feature_version=(3, 11): ACCEPTED
    feature_version=(3, 12): ACCEPTED

`feature_version` restricts the parser's grammar, but PEP 701's f-string
handling is not versioned in CPython's parser, so a 3.12-only f-string
passes the check even on a 3.12 interpreter. The same file under a real
3.10 interpreter is a SyntaxError. So the check that actually works is to
compile every file with a REAL 3.10, and that is what this test does.

## How it finds a 3.10

In order: an explicit `HERMES_PYTHON_310` environment variable, then a
`python3.10` / `python3.10.exe` on PATH, then a `uv`-managed 3.10.

If no 3.10 can be found the test SKIPS rather than failing, because the
absence of an interpreter is not a defect in the repository. CI pins 3.10
explicitly, so there the check always runs.
"""

from __future__ import annotations

import os
import py_compile
import shutil
import subprocess
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]

# The floor declared in pyproject.toml. If that changes, this changes.
DECLARED_FLOOR = (3, 10)

# The directories a reader's install and the documented commands touch.
SCANNED = ("hedging_agent", "tests", "scripts")


def _declared_floor() -> tuple[int, int]:
    """Read the floor from pyproject.toml, so the test cannot drift."""
    text = (REPO / "pyproject.toml").read_text(encoding="utf-8")
    for line in text.splitlines():
        stripped = line.strip()
        if stripped.startswith("requires-python"):
            spec = stripped.split("=", 1)[1].strip().strip('"').strip("'")
            digits = spec.lstrip(">=~ ")
            parts = digits.split(".")
            if len(parts) >= 2 and parts[0].isdigit() and parts[1].isdigit():
                return (int(parts[0]), int(parts[1]))
    pytest.fail("pyproject.toml does not declare requires-python")


def _python_files() -> list[Path]:
    out: list[Path] = []
    for name in SCANNED:
        directory = REPO / name
        if not directory.is_dir():
            continue
        out.extend(sorted(directory.rglob("*.py")))
    return out


def _floor_interpreter() -> str | None:
    """Find a REAL interpreter for the declared floor, or None."""
    want = f"{DECLARED_FLOOR[0]}.{DECLARED_FLOOR[1]}"

    explicit = os.environ.get("HERMES_PYTHON_310")
    if explicit and Path(explicit).exists():
        return explicit

    for name in (f"python{want}", f"python{want}.exe"):
        found = shutil.which(name)
        if found:
            return found

    # uv's managed interpreters, if uv is on PATH.
    uv = shutil.which("uv")
    if uv:
        try:
            r = subprocess.run(
                [uv, "python", "find", want],
                capture_output=True,
                text=True,
                timeout=120,
            )
            if r.returncode == 0:
                candidate = r.stdout.strip()
                if candidate and Path(candidate).exists():
                    return candidate
        except (OSError, subprocess.SubprocessError):
            pass

    return None


def _version_of(interpreter: str) -> tuple[int, int] | None:
    try:
        r = subprocess.run(
            [interpreter, "-c", "import sys; print('%d.%d' % sys.version_info[:2])"],
            capture_output=True,
            text=True,
            timeout=120,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    if r.returncode != 0:
        return None
    try:
        major, minor = r.stdout.strip().split(".")[:2]
        return (int(major), int(minor))
    except ValueError:
        return None


FLOOR = _floor_interpreter()
FLOOR_VERSION = _version_of(FLOOR) if FLOOR else None

requires_floor = pytest.mark.skipif(
    FLOOR_VERSION != DECLARED_FLOOR,
    reason=(
        f"no Python {DECLARED_FLOOR[0]}.{DECLARED_FLOOR[1]} interpreter found "
        f"to parse the tree with; set HERMES_PYTHON_310 to one"
    ),
)


def test_pyproject_declares_the_floor_this_test_enforces():
    """The constant and the file must agree, or this test checks nothing."""
    assert _declared_floor() == DECLARED_FLOOR, (
        f"pyproject.toml declares {_declared_floor()} but this test enforces "
        f"{DECLARED_FLOOR}. Update DECLARED_FLOOR when the floor changes."
    )


def test_there_are_files_to_scan():
    """A path typo would make this file pass vacuously."""
    files = _python_files()
    assert len(files) > 30, f"expected the tree's Python files, found {len(files)}"
    assert any(f.name == "stats.py" for f in files), "hedging_agent/ was not scanned"


def test_the_floor_interpreter_really_is_the_floor():
    """The interpreter we found must be the version we think it is.

    Without this, `python3.10` could quietly be 3.12 and the whole check
    would pass while testing nothing -- which is exactly the failure this
    file exists to prevent, one level up.
    """
    if FLOOR is None:
        pytest.skip("no floor interpreter available")
    assert FLOOR_VERSION == DECLARED_FLOOR, (
        f"{FLOOR} reports {FLOOR_VERSION}, not {DECLARED_FLOOR}"
    )


@pytest.mark.parametrize(
    "path",
    _python_files(),
    ids=lambda p: p.relative_to(REPO).as_posix(),
)
@requires_floor
def test_file_compiles_under_the_declared_floor(path: Path, tmp_path: Path):
    """The file must compile under the OLDEST supported interpreter.

    This is the check that catches a PEP 701 f-string, which
    `ast.parse(feature_version=...)` does NOT.
    """
    target = tmp_path / (path.stem + ".pyc")
    result = subprocess.run(
        [
            FLOOR,
            "-c",
            "import py_compile,sys; py_compile.compile(sys.argv[1], "
            "cfile=sys.argv[2], doraise=True)",
            str(path),
            str(target),
        ],
        capture_output=True,
        text=True,
        timeout=300,
    )
    if result.returncode != 0:
        pytest.fail(
            f"{path.relative_to(REPO).as_posix()} does not compile under "
            f"Python {DECLARED_FLOOR[0]}.{DECLARED_FLOOR[1]}, which is the "
            f"version pyproject.toml promises to support:\n"
            f"{result.stderr.strip()}\n"
            f"A reader on the declared floor would fail at import time, not "
            f"on one test."
        )


@pytest.mark.parametrize(
    "path",
    _python_files(),
    ids=lambda p: p.relative_to(REPO).as_posix(),
)
def test_file_compiles_under_this_interpreter(path: Path, tmp_path: Path):
    """And it must compile under the interpreter actually running the suite."""
    target = tmp_path / (path.stem + ".pyc")
    try:
        py_compile.compile(
            str(path), cfile=str(target), doraise=True, quiet=1
        )
    except py_compile.PyCompileError as exc:
        pytest.fail(f"{path.relative_to(REPO).as_posix()} does not compile:\n{exc}")
