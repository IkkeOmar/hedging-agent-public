"""`--json` must produce JSON on stdout and nothing else.

This was broken and nothing said so. The flag existed, it printed JSON,
and it printed the JSON surrounded by a banner, six `[sweep]` progress
lines, a `[saved]` path and a cross-universe summary -- all on stdout. So:

    $ python scripts/run_research.py --json > out.json
    $ json.load(open("out.json"))
    json.decoder.JSONDecodeError: Expecting value: line 1 column 2 (char 1)

The agent-usability contract in GOAL.md section 5 requires `--json` on
every script "so the output can be consumed without parsing prose", and
the convention settled by the R2 survey is that **stdout is the machine
payload and stderr is for the human**. This test holds that line.

It runs the real entry point on the real fixture. The sweep is slow, so
the module-scoped fixture runs it once and the tests read the captured
output.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
SCRIPT = REPO / "scripts" / "run_research.py"


@pytest.fixture(scope="module")
def json_run(tmp_path_factory):
    """Run the entry point with --json and capture both streams."""
    out = tmp_path_factory.mktemp("json-run")
    proc = subprocess.run(
        [sys.executable, str(SCRIPT),
         "--universe", "us_equities",
         "--timeframe", "swing",
         "--candidates", "6",
         "--trials", "8",
         "--seed", "11",
         "--period", "5y",
         "--no-llm",
         "--json",
         "--out", str(out)],
        cwd=str(REPO), capture_output=True, text=True, timeout=900,
    )
    return proc


def test_stdout_is_pure_json(json_run):
    """stdout must parse with json.loads, with no stripping first."""
    try:
        payload = json.loads(json_run.stdout)
    except json.JSONDecodeError as exc:
        pytest.fail(
            f"stdout did not parse as JSON: {exc}\n"
            f"stdout began with:\n{json_run.stdout[:600]!r}\n"
            "stdout must contain the JSON payload and NOTHING else; every "
            "human-readable line belongs on stderr."
        )
    assert isinstance(payload, dict)
    assert "gate" in payload


def test_the_json_payload_has_the_documented_shape(json_run):
    """The gate is a dict: {passed, verdict, checks, failed_checks}."""
    payload = json.loads(json_run.stdout)
    gate = payload["gate"]
    assert isinstance(gate, dict)
    assert set(gate) >= {"passed", "verdict", "checks", "failed_checks"}
    assert gate["verdict"] in ("PROMOTED", "REJECTED")
    assert isinstance(gate["checks"], list)
    assert isinstance(gate["failed_checks"], list)


def test_the_human_output_is_on_stderr(json_run):
    """The report and the progress lines must be on stderr."""
    assert json_run.stderr.strip(), (
        "expected the human-readable output on stderr; stdout must be the "
        "JSON payload alone"
    )
    assert "[saved]" in json_run.stderr, (
        "the '[saved] <path>' line tells the caller where the artefact is; "
        "it belongs on stderr"
    )


def test_the_exit_code_is_one_because_the_gate_rejected(json_run):
    """1 means "the gate rejected", which is a RESULT, not an error.

    0 would mean the gate passed, which would be a false claim about this
    repository. 2 would mean a usage error.
    """
    assert json_run.returncode == 1, (
        f"expected exit 1 (the promotion gate rejects every universe); got "
        f"{json_run.returncode}\n"
        f"stderr tail:\n{json_run.stderr[-800:]}"
    )


def test_the_artefact_is_written_regardless_of_the_verdict(json_run, tmp_path):
    """A report that only exists on success is a report that hides its
    own failures."""
    # The fixture wrote into its own tmp dir; find it.
    written = list(Path(json_run.args[-1]).glob("sweep_*.json"))
    assert written, "no sweep artefact was written"
    data = json.loads(written[0].read_text(encoding="utf-8"))
    assert data["gate"]["verdict"] == "REJECTED"


def test_the_json_payload_matches_the_written_artefact(json_run):
    """What is printed and what is saved must be the same thing."""
    payload = json.loads(json_run.stdout)
    written = list(Path(json_run.args[-1]).glob("sweep_*.json"))
    assert written
    data = json.loads(written[0].read_text(encoding="utf-8"))
    assert payload["gate"]["verdict"] == data["gate"]["verdict"]
    assert payload["n_trials"] == data["n_trials"]
