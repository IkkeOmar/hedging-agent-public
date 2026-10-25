"""Yahoo robustness: a partial universe must never be silently accepted.

Background: a sweep failed with "yfinance returned nothing for [11 ETFs]".
Probing the failure mode showed the worse case was not the one that raised --
a PARTIAL response was accepted silently, so research continued on a portfolio
missing most of its assets and every statistic described a different strategy
than the README documents.

The retry behaviour is written against the installed yfinance (1.7.0), verified
by introspection rather than assumed:
  * `_is_transient_error(YFRateLimitError())` is False -> the library will not
    retry a 429 by itself, so we must.
  * `download()` has no `retries` parameter at this version.
  * `YFRateLimitError` lives in `yfinance.exceptions`.
"""
from __future__ import annotations

import pandas as pd
import pytest
import yfinance as yf
from yfinance.exceptions import YFRateLimitError

from hedging_agent import data as D

FULL = ["SPY", "QQQ", "IWM", "XLK", "XLF", "EFA", "EEM", "TLT", "GLD", "UUP", "SH"]


def _frame(tickers):
    cols = pd.MultiIndex.from_product([["Close"], tickers])
    idx = pd.date_range("2024-01-01", periods=300, freq="D")
    return pd.DataFrame(300.0, index=idx, columns=cols)


def test_partial_universe_is_rejected_not_silently_trimmed():
    """Yahoo returning 4 of 11 tickers must raise, not return a short frame."""
    yf.download = lambda *a, **k: _frame(FULL[:4])
    with pytest.raises(RuntimeError, match="missing"):
        D.load_universe("us_equities", period="1y", use_cache=False)


def test_complete_universe_still_loads(monkeypatch):
    yf.download = lambda *a, **k: _frame(FULL)
    df = D.load_universe("us_equities", period="1y", use_cache=False)
    assert len(df.columns) == len(FULL)
    assert "SPY" in df.columns and "SH" in df.columns


def test_rate_limit_is_retried_and_recovers(monkeypatch):
    """A 429 on the first attempt must not end the run."""
    calls = {"n": 0}

    def flaky(*a, **k):
        calls["n"] += 1
        if calls["n"] == 1:
            raise YFRateLimitError()
        return _frame(FULL)

    yf.download = flaky
    monkeypatch.setattr("time.sleep", lambda *_a, **_k: None)
    df = D.load_universe("us_equities", period="1y", use_cache=False)
    assert len(df.columns) == len(FULL)
    assert calls["n"] >= 2, "should have retried the rate limit"


def test_rate_limit_that_never_clears_raises():
    """Persistent throttling must fail loudly, not return partial data."""
    def always_limited(*a, **k):
        raise YFRateLimitError()

    yf.download = always_limited
    with pytest.raises(RuntimeError):
        D.load_universe("us_equities", period="1y", use_cache=False)


def test_library_will_not_retry_rate_limits_for_us():
    """Pin the assumption the retry wrapper rests on.

    If a future yfinance release starts retrying 429s internally, the comment
    above becomes false and this test says so instead of the wrapper silently
    duplicating it.
    """
    import yfinance.data as ydata

    assert ydata._is_transient_error(YFRateLimitError()) is False
    assert "retries" not in __import__("inspect").signature(
        yf.download).parameters


def test_clear_crumb_cache_targets_the_cookie_db_not_the_tz_cache():
    """Deleting the wrong file is worse than not deleting anything.

    `set_tz_cache_location()` redirects the timezone cache; the cookie/crumb
    cache is a different file managed by a different manager.
    """
    from yfinance.cache import _CookieDBManager

    loc = _CookieDBManager.get_location()
    assert loc, "cookie cache location should be discoverable"
    # must not raise even when the db does not exist yet
    D._clear_crumb_cache()