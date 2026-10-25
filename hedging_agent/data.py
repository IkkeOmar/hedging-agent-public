"""Market data.

Two providers, one interface:
  yfinance  -- equities and ETFs, no key
  ccxt      -- crypto spot/perps, no key

Both return a tidy frame: index=timestamp, columns=assets, values=close.
That frame is the only thing the rest of the system sees, so swapping providers
does not ripple.
"""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd


@dataclass
class Universe:
    name: str
    assets: list[str]
    provider: str      # yahoo | ccxt
    kind: str          # equities | crypto
    hedge_proxy: str   # the instrument expressing the hedge
    note: str = ""

    def as_dict(self) -> dict:
        return {"name": self.name, "assets": self.assets, "provider": self.provider,
                "kind": self.kind, "hedge_proxy": self.hedge_proxy, "note": self.note}


# The unhedged portfolio and the hedge instrument are deliberately different
# exposures. A hedge that is the same trade as the portfolio is not a hedge.
UNIVERSES: dict[str, Universe] = {
    "us_equities": Universe(
        name="us_equities",
        assets=["SPY", "QQQ", "IWM", "XLK", "XLF", "EFA", "EEM", "TLT", "GLD", "UUP"],
        provider="yahoo", kind="equities", hedge_proxy="SH",  # ProShares Short S&P 500
        note="SH is a short-S&P futures proxy. It is negatively correlated to the "
             "equity sleeve, which is the entire point of picking it.",
    ),
    "crypto": Universe(
        name="crypto",
        assets=["BTC/USDT", "ETH/USDT", "SOL/USDT", "AVAX/USDT", "LINK/USDT",
                "DOT/USDT", "ADA/USDT"],
        provider="ccxt", kind="crypto", hedge_proxy="BTC/USDT",
        note="Shorting the majors hedges beta but not idiosyncratic alt risk. "
             "That gap is real and is reported, not hidden.",
    ),
    "mixed": Universe(
        name="mixed",
        assets=["SPY", "QQQ", "IWM", "EFA", "TLT", "GLD",
                "BTC-USD", "ETH-USD", "SOL-USD"],
        provider="mixed", kind="mixed", hedge_proxy="SH",
        note="Crypto trades 24/7 and equities do not. The mismatch is handled by "
             "aligning on the equity calendar and forward-filling crypto, which "
             "introduces its own staleness. Disclosed, not hidden.",
    ),
}


def _clear_crumb_cache() -> None:
    """Drop yfinance's persisted cookie/crumb cache.

    yfinance stores authentication cookies and crumbs in `cookies.db` inside
    its `py-yfinance` cache directory (yfinance/cache.py, `_CookieDBManager`),
    and a cached crumb whose text is "Too Many Requests" keeps a throttled
    client throttled -- waiting alone does not clear it
    (ranaroussi/yfinance#2441, fixed upstream in PR #2491).

    The location is read from the library's own manager rather than guessed,
    so a user who relocated the cache is handled. `set_tz_cache_location()`
    would be wrong here: it redirects the TIMEZONE cache, not the cookie one.
    Best-effort -- if anything is missing, the retry still proceeds.
    """
    import os

    try:
        from yfinance.cache import _CookieDBManager

        loc = _CookieDBManager.get_location()
        if not loc:
            return
        db = os.path.join(loc, "cookies.db")
        if not os.path.exists(db):
            return
        # close the handle first: SQLite will not delete an open file on all
        # platforms, and yfinance may hold the connection cached
        try:
            _CookieDBManager.close_db()
        except Exception:
            pass
        os.remove(db)
    except Exception:
        pass


def _download_with_retry(yf, assets: list[str], period: str,
                         interval: str, attempts: int = 4):
    """Fetch from Yahoo, tolerating the rate limiting Yahoo actually does.

    Grounded in the installed yfinance (1.7.0) rather than guessed:

      * `_is_transient_error()` covers only timeout/socket/OSError. A
        `YFRateLimitError` returns False for it, so the library does NOT
        retry a 429 on its own -- verified by calling it directly.
      * `download()` has no `retries` parameter at this version (that landed
        on `dev`, PR ranaroussi/yfinance#2627), so there is nothing to pass.
      * `YFRateLimitError` lives in `yfinance.exceptions`, not on the
        top-level `yf` namespace.
      * Backoff convention matches the library's own transient retry
        (`2 ** attempt` seconds).

    Retrying a rate limit also means dropping any cached crumb/session: a
    sticky cached crumb is what keeps a throttled client throttled
    (ranaroussi/yfinance#2441).
    """
    import time

    from yfinance.exceptions import YFRateLimitError

    last = None
    raw = None
    for i in range(max(1, attempts)):
        try:
            raw = yf.download(assets, period=period, interval=interval,
                              progress=False, auto_adjust=True, threads=True)
        except YFRateLimitError as e:
            last = e
            raw = None
            # Backoff alone is not always enough: a cached crumb whose text is
            # literally "Too Many Requests" keeps a throttled client
            # throttled (ranaroussi/yfinance#2441). yfinance 1.7.0 exposes no
            # public reset for it -- YfData has no clear/reset method and no
            # class-level instance handle -- so the crumb cache on disk is
            # cleared instead, which is the durable part of the state.
            _clear_crumb_cache()
            print(f"[data] rate limited (attempt {i+1}/{attempts}), "
                  f"backing off {2 ** i}s", file=sys.stderr)
        except Exception as e:            # network hiccup, not a data problem
            last = e
            raw = None
        if raw is not None and not raw.empty:
            return raw
        time.sleep(2 ** i)

    # last resort: one at a time, no thread pool
    try:
        raw = yf.download(assets, period=period, interval=interval,
                          progress=False, auto_adjust=True, threads=False)
        if raw is not None and not raw.empty:
            return raw
    except Exception as e:
        last = e
    if last is not None:
        print(f"[data] yfinance failed after {attempts} attempts: {last}", file=sys.stderr)
    return raw


FIXTURES_DIR = Path(__file__).resolve().parent / "fixtures"


def fixture_path(universe: str, interval: str, period: str) -> Path:
    """The committed price file for a universe, or a path that does not exist.

    The fixtures are committed so that the default path needs no network
    and reproduces the published numbers exactly. A live download cannot do
    that: the crypto provider returns the LAST 1000 daily bars, so the
    window ends at today's date and a re-run three days later is a
    DIFFERENT experiment rather than a reproduction of the same one.
    """
    return FIXTURES_DIR / f"{universe}_{interval}_{period}.csv"


def load_fixture(universe: str, interval: str = "1d", period: str = "5y") -> pd.DataFrame:
    """Read a committed price file. No network, no cache, no clock."""
    path = fixture_path(universe, interval, period)
    if not path.exists():
        available = sorted(p.name for p in FIXTURES_DIR.glob("*.csv"))
        raise FileNotFoundError(
            f"no committed price file for {universe} ({interval}, {period}). "
            f"Expected {path}. Available: {available}. "
            f"Run with --live to download from the provider instead."
        )
    df = pd.read_csv(path, index_col=0, parse_dates=True)
    # The crypto fixture is written by the ccxt branch and carries an
    # unnamed first column; the equity ones carry a 'Date' column. Normalise
    # both to the same shape so the frame is identical either way.
    df.index = pd.to_datetime(df.index, utc=False)
    if isinstance(df.index, pd.DatetimeIndex) and df.index.tz is not None:
        df.index = df.index.tz_localize(None)
    df = df.sort_index()
    return df


def load_universe(
    universe: str,
    period: str = "3y",
    interval: str = "1d",
    cache_dir: str | None = None,
    use_cache: bool = True,
    offline: bool = False,
) -> pd.DataFrame:
    """Returns a close-price frame, index=DatetimeIndex, columns=assets.

    `offline=True` reads the committed fixture and never touches the
    network. It is the DEFAULT for the documented path, because a live
    download cannot reproduce the published numbers: the crypto provider
    returns the most recent 1000 daily bars, so the window moves with the
    calendar and the same command produces a different experiment.
    """
    u = UNIVERSES[universe]
    if offline:
        return load_fixture(universe, interval=interval, period=period)
    cache_dir = cache_dir or os.path.expanduser("~/.cache/hedging-agent")
    os.makedirs(cache_dir, exist_ok=True)
    cache = os.path.join(cache_dir, f"{universe}_{interval}_{period}.csv")

    if use_cache and os.path.exists(cache):
        df = pd.read_csv(cache, index_col=0, parse_dates=True)
        return df

    frames: dict[str, pd.Series] = {}
    if u.provider in ("yahoo", "mixed"):
        try:
            import yfinance as yf
        except ImportError as exc:
            raise RuntimeError(
                "the live price path needs yfinance, which is not installed. "
                "Either: (a) run the OFFLINE path, which is the default and "
                "needs nothing beyond the core dependencies -- the committed "
                "price fixtures reproduce the published numbers exactly; or "
                "(b) install the live extra with "
                "`pip install -e \".[market-data]\"`. The two providers are told "
                "apart by the asset's SLASH, not its dash."
            ) from exc
        # the hedge instrument is fetched alongside the portfolio: a hedge you
        # cannot price is a wish
        # "/" is the ccxt pair separator ("BTC/USDT"); "-" is the Yahoo
        # ticker separator ("BTC-USD"). The two providers are told apart by
        # the SLASH, not by the dash.
        #
        # This used to test for "-", which silently removed every crypto
        # asset from the mixed universe: Yahoo dropped "BTC-USD" because it
        # contained a dash, and the ccxt branch skipped it for the same
        # reason. The mixed universe was therefore pure equities while its
        # own note claimed crypto was forward-filled onto the equity
        # calendar. A filter that is right for one provider and wrong for
        # the other removes data without saying so.
        yahoo_assets = [a for a in u.assets if "/" not in a]
        if u.hedge_proxy not in yahoo_assets and "-" not in u.hedge_proxy:
            yahoo_assets = yahoo_assets + [u.hedge_proxy]
        # threads=True fires every ticker at once, which is precisely what
        # earns a 429 from Yahoo. A transient throttle used to kill all 11
        # tickers and the whole universe with it, so retry with backoff and
        # fall back to serial requests if the parallel one keeps failing.
        raw = _download_with_retry(yf, yahoo_assets, period, interval)
        if raw is None or raw.empty:
            raise RuntimeError(f"yfinance returned nothing for {yahoo_assets}")
        close = (raw["Close"] if isinstance(raw.columns, pd.MultiIndex)
                 else raw[["Close"]].rename(columns={"Close": yahoo_assets[0]}))

        # A PARTIAL response is the dangerous case. Yahoo will happily return
        # 4 of 11 tickers under rate limiting, and the old loop simply skipped
        # the rest -- so the run continued on a silently different portfolio
        # than the one the README documents. Every statistic would describe a
        # strategy nobody asked about. Fail loudly instead.
        missing = [a for a in yahoo_assets if a not in close.columns]
        if missing:
            raise RuntimeError(
                f"yfinance returned {len(close.columns)}/{len(yahoo_assets)} "
                f"tickers; missing {missing}. Refusing to run on a partial "
                f"universe -- the results would not describe the requested "
                f"portfolio. Retry later or pass use_cache=False."
            )
        for a in yahoo_assets:
            frames[a] = close[a].dropna()

    if u.provider in ("ccxt", "mixed"):
        try:
            import ccxt
        except ImportError as exc:
            raise RuntimeError(
                "the live price path needs ccxt, which is not installed. "
                "Either: (a) run the OFFLINE path, which is the default and "
                "needs nothing beyond the core dependencies -- the committed "
                "price fixtures reproduce the published numbers exactly; or "
                "(b) install the live extra with "
                "`pip install -e \".[market-data]\"`."
            ) from exc
        ex = ccxt.binance({"enableRateLimit": True})
        for a in u.assets:
            if "/" not in a:        # a Yahoo ticker, not a ccxt pair
                continue
            try:
                ohlc = ex.fetch_ohlcv(a, timeframe=interval, limit=1000)
            except Exception:
                continue
            if not ohlc:
                continue
            # ccxt.fetch_ohlcv returns a LIST of lists (each row
            # [ts, o, h, l, c, v]), not a numpy array. Indexing it as
            # ohlc[:, 0] raises "TypeError: list indices must be integers or
            # slices, not tuple" -- which the research entry point used to hide
            # behind a bare except, so the entire crypto universe was broken
            # and looked like an unrelated plotting error.
            ohlc = np.asarray(ohlc, dtype=float)
            if ohlc.ndim != 2 or ohlc.shape[1] < 5:
                continue
            idx = pd.to_datetime(ohlc[:, 0], unit="ms", utc=True)
            closes = pd.Series(ohlc[:, 4], index=idx)
            # ccxt occasionally returns zero or missing prints for illiquid
            # pairs; a zero close makes pct_change explode to -100% and
            # silently poisons every downstream statistic
            closes = closes.replace(0.0, np.nan).dropna()
            if len(closes):
                frames[a] = closes

    if not frames:
        raise RuntimeError("no data loaded; check network or provider packages")

    df = pd.DataFrame(frames).sort_index()
    if u.provider == "mixed":
        # align on the union calendar, forward-fill the 24/7 series
        df = df.ffill().dropna(how="all")
    df = df.dropna(how="all").ffill()
    df.to_csv(cache)
    return df


def hedge_returns(prices: pd.DataFrame, hedge_proxy: str) -> pd.Series:
    """Return series of the hedge INSTRUMENT as traded (a long position).

    This is deliberately sign-free: it is what the instrument did. Whether the
    portfolio wants the long or the short side is decided by `hedge_side`, which
    is derived from the measured correlation rather than assumed.

    Getting this backwards is the single most expensive mistake in an overlay:
    shorting an already-inverse ETF makes every drawdown worse while the
    backtest still reports a plausible number.
    """
    if hedge_proxy in prices.columns:
        # NOTE: keep the leading NaN. pct_change() is defined on the ORIGINAL
        # bar grid; dropping it shifts every later value onto the wrong date,
        # which is a one-bar look-ahead in the hedge leg. It is the kind of bug
        # that improves a backtest and is invisible in a chart.
        return prices[hedge_proxy].pct_change()
    raise KeyError(
        f"hedge proxy {hedge_proxy!r} not in price frame. Columns: {list(prices.columns)}"
    )


def infer_hedge_side(
    prices: pd.DataFrame,
    hedge_proxy: str,
    portfolio_cols: list[str] | None = None,
) -> tuple[float, float]:
    """Decide whether to go LONG or SHORT the hedge instrument, from data.

    Returns (side, correlation) where side = +1 means hold the instrument long,
    -1 means hold it short.

    Convention, and it is the important part:
      * If the instrument is already negatively correlated to the portfolio
        (SH, a short-equity ETF), then LONG it. You want it to pay when the
        portfolio falls.
      * If it is positively correlated (SPY itself, used as a crude beta hedge),
        then SHORT it.

    The caller is expected to REPORT this correlation. A hedge whose sign is
    chosen by hand is a hedge nobody checked.
    """
    cols = portfolio_cols if portfolio_cols is not None else [
        c for c in prices.columns if c != hedge_proxy
    ]
    if hedge_proxy not in prices.columns:
        raise KeyError(f"hedge proxy {hedge_proxy!r} not in price frame")
    r = prices[cols].pct_change().mean(axis=1)
    h = prices[hedge_proxy].pct_change()
    j = pd.concat([r, h], axis=1).dropna()
    corr = float(j.iloc[:, 0].corr(j.iloc[:, 1]))
    side = 1.0 if corr < 0 else -1.0
    return side, corr


def build_hedged_returns(
    prices: pd.DataFrame,
    hedge_notional: pd.Series,
    hedge_proxy: str,
    hedge_sign: float | None = None,
    portfolio_cols: list[str] | None = None,
    round_trip_cost_bps: float = 0.0,
    borrow_bps_annual: float = 0.0,
    periods: int = 252,
) -> tuple[pd.Series, dict]:
    """Portfolio returns + hedge P&L at the requested notional.

    hedge_sign=None means "infer it from the measured correlation". Pass a
    float only to override deliberately, e.g. to stress the opposite sign.

    portfolio_cols: the columns that constitute the risky portfolio. Pass it
    when the frame also carries the hedge instrument -- otherwise the mean
    silently includes the hedge and the two legs cancel. That mistake is
    invisible in a chart and fatal in a Sharpe.

    Returns (returns, diagnostics). The diagnostics carry the side and the
    correlation so the report can print them.
    """
    cols = portfolio_cols if portfolio_cols is not None else [
        c for c in prices.columns if c != hedge_proxy
    ]
    if not cols:
        raise ValueError("no portfolio columns after excluding the hedge proxy")
    port = prices[cols].pct_change().mean(axis=1).fillna(0.0)
    proxy = hedge_returns(prices, hedge_proxy).fillna(0.0)
    n = hedge_notional.reindex(port.index).fillna(0.0)

    if hedge_sign is None:
        side, corr = infer_hedge_side(prices, hedge_proxy, cols)
    else:
        side, corr = float(hedge_sign), float("nan")

    gross = port + side * n * proxy

    # Transaction costs. Charged on the CHANGE in hedge notional, which is what
    # is actually traded: a hedge held steady costs nothing to maintain beyond
    # financing, while a hedge that ramps and unwraps pays on every step.
    #
    # This was previously modelled in config and reported in the metrics but
    # NEVER APPLIED, so every Sharpe in every sweep was gross of costs. At 36x
    # annualised turnover that is not a rounding error.
    if round_trip_cost_bps > 0 or borrow_bps_annual > 0:
        # Cost of the trade placed AT bar t is charged at bar t+1, matching the
        # convention that a position placed at t only affects returns from
        # t+1. Without this shift the return window (.iloc[1:], which drops bar
        # 0) silently discarded the entry cost and every position was entered
        # for free.
        traded = (n - n.shift(1)).abs().fillna(n.abs())
        cost_per_bps = (traded / 10_000.0).shift(1).fillna(0.0)
        # Borrow is an ANNUAL rate, so it must be divided down to a per-period
        # charge before being applied each bar. Dividing the bps figure by
        # `periods` and then multiplying by |notional| (a FRACTION of equity,
        # ~0.12) silently double-counts: it charged 10bps/252 * 0.12 every day,
        # i.e. a full annual 10% on the notional EVERY BAR -> 120% a year.
        # The annual rate applies to the notional; the per-bar charge is the
        # annual rate divided by periods, times the notional.
        borrow_per_bar = borrow_bps_annual / 10_000.0 / periods * n.abs()
        gross = gross - (round_trip_cost_bps * cost_per_bps
                         + borrow_per_bar.shift(1).fillna(0.0))

    out = gross.iloc[1:].dropna()
    diag = {
        "round_trip_cost_bps": round_trip_cost_bps,
        "borrow_bps_annual": borrow_bps_annual,
        "hedge_proxy": hedge_proxy,
        "hedge_side": side,
        "hedge_side_label": "LONG (instrument already inverse to portfolio)"
                            if side > 0 else "SHORT (instrument correlated with portfolio)",
        "hedge_corr_to_portfolio": round(corr, 4) if corr == corr else None,
        "portfolio_cols": cols,
        "correlation_warning": (
            "hedge instrument is NOT inverse to the portfolio; a hedge here "
            "adds risk rather than removing it" if corr == corr and corr > -0.1 else None
        ),
    }
    return out, diag