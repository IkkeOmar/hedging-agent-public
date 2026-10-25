"""The console scripts must resolve to something that exists.

`pyproject.toml` declared:

    [project.scripts]
    hedge-research = "hedging_agent.cli:main"

and there was no `hedging_agent/cli.py`. `pip install` created the
console script happily, and every invocation died with:

    ModuleNotFoundError: No module named 'hedging_agent.cli'

An entry point that resolves to nothing is worse than no entry point: it
looks like an installed command and fails the moment it is used. Nothing
caught it, because the test suite never invoked the installed command --
it called `scripts/run_research.py` directly.

So this test does what the suite did not: it imports the module the way
`pip` does and checks that the target exists and is callable.
"""

from __future__ import annotations

import importlib
from _toml_compat import tomllib
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]


def _declared_scripts() -> dict[str, str]:
    data = tomllib.loads((REPO / "pyproject.toml").read_text(encoding="utf-8"))
    return data["project"].get("scripts", {})


def test_the_project_declares_console_scripts():
    """A project with no console script has nothing to break. This one
    does, so the rest of this file has something to check."""
    assert _declared_scripts(), "no [project.scripts] were declared"


@pytest.mark.parametrize("name,target", sorted(_declared_scripts().items()))
def test_the_entry_point_resolves_to_a_callable(name: str, target: str):
    """`module:attr` must import, and the attribute must be callable."""
    module_name, _, attr = target.partition(":")
    assert module_name, f"{name} has no module part"
    assert attr, f"{name} has no attribute part"

    try:
        module = importlib.import_module(module_name)
    except ModuleNotFoundError as exc:
        pytest.fail(
            f"the console script {name!r} points at {target!r}, but "
            f"{module_name!r} cannot be imported: {exc}. "
            "An entry point that resolves to nothing installs a command "
            "that dies on first use."
        )

    assert hasattr(module, attr), (
        f"{module_name!r} has no attribute {attr!r}, so {name!r} would fail "
        f"when it is run."
    )
    assert callable(getattr(module, attr)), f"{target!r} is not callable"


@pytest.mark.parametrize("name,target", sorted(_declared_scripts().items()))
def test_the_entry_point_survives_an_import_without_arguments(name: str, target: str):
    """Importing the module must not run the whole research sweep.

    A console-script module is imported by `pip` at install time to find
    its entry point. If importing it starts a 90-second sweep, the
    install is unusable.
    """
    module_name = target.partition(":")[0]
    module = importlib.import_module(module_name)
    assert module.__doc__, f"{module_name} has no docstring"
