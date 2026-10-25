"""Read `pyproject.toml` on every supported interpreter.

`tomllib` is in the standard library from Python 3.11. On 3.10 the
backport `tomli` provides the same API, and `tomli` is a declared test
dependency, so the same test runs on the declared floor.

Without this, `pytest` cannot even COLLECT the two tests that use this
module on 3.10, so the suite dies before it starts rather than on one
assertion.
"""

from __future__ import annotations

import sys

if sys.version_info >= (3, 11):
    import tomllib
else:  # Python 3.10
    import tomli as tomllib

__all__ = ["tomllib"]
