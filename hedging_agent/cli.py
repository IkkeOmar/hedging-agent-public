"""The console-script entry point.

``pyproject.toml`` declares::

    [project.scripts]
    hedge-research = "hedging_agent.cli:main"

and that module did not exist. ``pip install`` created the console script
happily, and every invocation died with::

    ModuleNotFoundError: No module named 'hedging_agent.cli'

An entry point that resolves to nothing is worse than no entry point: it
looks like an installed command and fails when it is used. This module
is the fix -- a thin shell around the scripts, so the argument parsing
and the sweep logic stay in one place and the console script is a real
thing that works.

The scripts live in ``scripts/``, which is not a package, so they cannot
be the target of a console script directly. They are imported by path
here, once, at import time.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from typing import Any


def _load(name: str, filename: str) -> Any:
    """Import a script from the ``scripts/`` directory by path."""
    path = Path(__file__).resolve().parent.parent / "scripts" / filename
    if not path.exists():
        raise ImportError(
            f"the {name} entry point is missing: expected {path}. The "
            "scripts/ directory is part of the distribution; if you "
            "installed with --no-install-project or from a partial copy, "
            "the repository is incomplete."
        )
    spec = importlib.util.spec_from_file_location(f"_hedge_{name}", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"could not load {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def main() -> int:
    """Run the research sweep. See ``scripts/run_research.py --help``."""
    module = _load("research", "run_research.py")
    return int(module.main())


def agent_main() -> int:
    """Run the forward simulation. See ``scripts/run_agent.py --help``."""
    module = _load("agent", "run_agent.py")
    return int(module.main())


if __name__ == "__main__":
    raise SystemExit(main())
