"""A missing optional dependency must say what to do.

GOAL.md section 5 (the agent-usability contract) requires it:

    "Zero keys required. Every *_API_KEY is optional, and the error
     message when one is missing must say exactly what to do and where to
     get it -- so an AI can set it up from the error alone."

The same rule applies to an optional *package*. The live price path needs
yfinance and ccxt; the offline path needs neither. A bare
`ModuleNotFoundError: No module named 'ccxt'` is a traceback that ends
in a module name and says nothing else, and it is what a user who opted
into `--live` without the extra actually saw.

These tests pin the message: it must name the missing package, name the
offline alternative, and name the extra to install. The last one matters
most, because a wrong instruction is worse than no instruction -- the
extra is called `market-data`, and telling a reader to install
`".[live]"` sends them to a command that does not exist.
"""

from __future__ import annotations

import builtins

import pytest

from hedging_agent import data as D


def _without(missing: str):
    """Make `import <missing>` fail, the way a fresh install would."""

    real_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name == missing or name.startswith(f"{missing}."):
            raise ImportError(f"No module named {missing!r}")
        return real_import(name, *args, **kwargs)

    return fake_import


def test_missing_yfinance_says_what_to_do(monkeypatch):
    """The Yahoo live path must not die on a bare ModuleNotFoundError."""
    monkeypatch.setattr(builtins, "__import__", _without("yfinance"))
    with pytest.raises(RuntimeError) as exc:
        D.load_universe("us_equities", period="1y", use_cache=False)
    msg = str(exc.value)
    assert "yfinance" in msg
    assert "offline" in msg.lower(), (
        "the message must offer the offline path, which needs no extra"
    )
    assert "market-data" in msg, (
        "the message must name the REAL extra; `market-data` is what "
        "pyproject.toml declares"
    )


def test_missing_ccxt_says_what_to_do(monkeypatch):
    """The crypto live path must not die on a bare ModuleNotFoundError."""
    monkeypatch.setattr(builtins, "__import__", _without("ccxt"))
    with pytest.raises(RuntimeError) as exc:
        D.load_universe("crypto", period="1y", use_cache=False)
    msg = str(exc.value)
    assert "ccxt" in msg
    assert "offline" in msg.lower()
    assert "market-data" in msg


def test_the_message_names_an_install_command_that_exists(monkeypatch):
    """The instruction must be a command that actually works.

    A wrong extra name is worse than no message: the reader runs it, it
    fails, and now they know less than before.
    """
    monkeypatch.setattr(builtins, "__import__", _without("ccxt"))
    with pytest.raises(RuntimeError) as exc:
        D.load_universe("crypto", period="1y", use_cache=False)

    msg = str(exc.value)
    assert "pip install" in msg

    # The extra named in the message must be a real one.
    import re
    from _toml_compat import tomllib
    from pathlib import Path

    pyproject = Path(__file__).resolve().parents[1] / "pyproject.toml"
    data = tomllib.loads(pyproject.read_text(encoding="utf-8"))
    extras = data["project"]["optional-dependencies"]

    named = set(re.findall(r"\.\[([a-z-]+)\]", msg))
    assert named, "the message must name an extra in the form .[name]"
    for name in named:
        assert name in extras, (
            f"the message names the extra {name!r}, which pyproject.toml does "
            f"not declare; the real extras are {sorted(extras)}"
        )


def test_the_offline_path_needs_neither_package(monkeypatch):
    """The documented default path must work with no optional extra.

    This is the path D2 promises: the committed fixtures reproduce the
    published numbers, with no network and no key.
    """
    monkeypatch.setattr(builtins, "__import__", _without("yfinance"))
    monkeypatch.setattr(builtins, "__import__", _without("ccxt"))
    df = D.load_universe("crypto", interval="1d", period="5y", offline=True)
    assert len(df) > 0
    assert "BTC/USDT" in df.columns
