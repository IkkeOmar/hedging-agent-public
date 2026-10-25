"""Honest benchmarks: how does the hedged book compare to SIMPLE alternatives?

This module exists because of one specific paper.

Cederburg, O'Doherty, Wang & St Whaley (2020), "On the Performance of
Volatility-Managed Portfolios", JFE: across 103 equity strategies they find
"volatility-managed portfolios do not systematically outperform their
corresponding unmanaged portfolios in direct comparisons." Moreira & Muir (2017)
reported the Sharpe gain; the 2020 paper tested it broadly and it did not hold.

So the question a volatility-gated overlay must answer is NOT "did I beat
buy-and-hold" -- it is "did I beat the handful of stupid-simple rules that any
competent allocator would have tried first". If the answer is no, the overlay is
decoration. tanishhky/regime-adaptive-portfolio publishes exactly this table
against its own system and reports that simple 200-day MA timing and drawdown
control beat it on raw Sharpe in some universes. That is the standard to meet.

Every benchmark here is implemented from first principles in a few lines, uses
NO fitted parameters beyond the length of the lookback, and is evaluated on the
identical return grid as the hedged strategy. A benchmark that wins and is then
quietly dropped is worse than no benchmark.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .stats import max_drawdown, sharpe, sortino


def volatility_drag(returns, periods: int = 252) -> dict:
    """Decompose return into arithmetic mean, volatility drag, and what compounding
    actually delivered.

    THIS IS THE ONE PLACE WHERE A CORRECTLY-IMPLEMENTED OVERLAY CAN BEARINGLY
    POSITIVE, and it is worth being precise about why.

    A book earns its arithmetic mean mu, but its COMPOUND growth is less than mu
    because of volatility drag. The volatility tax (term coined by Rick Ashburn,
    CFA, 2003; formalised by Mark Spitznagel; Wikipedia "Volatility tax"; also
    called volatility drag / variance drain / volatility decay) is defined as the
    arithmetic mean MINUS the geometric mean:

        geometric mean  ~=  mu - sigma^2 / 2
        volatility tax  ~=  sigma^2 / 2

    That sigma^2/2 term is a real, recurring, QUADRATIC cost, paid every period.
    It is the price of the variance you insisted on taking.

    An inverse-vol overlay reduces sigma, so it cuts the tax quadratically. That
    gain needs NO predictive edge -- it is arithmetic, not forecasting -- and it
    is the one part of the vol-scaling thesis that survives Cederburg et al.'s
    replication across 103 strategies: they contested the SHARPE gain from
    mean-variance TIMING, not the mechanical reduction of sigma^2/2.

    ## Which annualisation convention, and a correction

    An earlier version of this docstring claimed that QuantStats' `cagr` /
    empyrical's `annual_return` (the POWER convention,
    prod(1+r)**(periods/n) - 1) "overstates growth" relative to
    exp(mean(log(1+r))*periods) - 1, and that the gap widens with volatility.
    Both claims are FALSE and were removed after measurement:

        sd      log_cagr   exp(log)-1   power        power - exp(log)
        0.0020   0.24213    0.27396     0.27396     0.00000
        0.0120   0.17851    0.19543     0.19543     0.00000
        0.0200   0.10974    0.11599     0.11599     0.00000

    prod(1+r)**(1/y) is algebraically exp((1/y) * sum(log(1+r))) = exp(E[log]*periods).
    The two conventions are the SAME NUMBER to floating-point precision. What
    falls as volatility rises is log-CAGR itself -- that IS the tax. No
    convention is flattering any strategy here.

    What is genuinely easy to get wrong is differencing against the
    EXPONENTIATED figure. exp() has already re-added the drag, so
    arith_mean - exp(log_CAGR) + 1 comes out NEGATIVE on a sample with a
    genuinely positive tax. The tax must be taken against the LOG mean.

    ## The honest caveat

    Reducing sigma does NOT come free. De-levering through an inverse hedge
    reduces the arithmetic mean too. Net effect = (change in mu) minus
    (change in sigma^2/2), and the overlay only wins if the drag saving exceeds
    the mean sacrificed. Reporting CAGR without this decomposition is how
    vol-scaling results get oversold.
    """
    r = pd.Series(np.asarray(returns, dtype=float)).dropna()
    n = len(r)
    if n < 5:
        return {"cagr": float("nan"), "power_cagr": float("nan"),
                "arith_mean": float("nan"), "vol": float("nan"),
                "drag": float("nan"), "drag_pct_of_mean": float("nan"),
                "n_obs": n, "note": "too few observations"}
    if (r <= -1.0).any():
        # a wipeout bar makes geometric growth undefined; do not paper over it
        return {"cagr": float("nan"), "power_cagr": float("nan"),
                "arith_mean": round(float(r.mean()) * periods, 6),
                "vol": round(float(r.std(ddof=1)) * np.sqrt(periods), 6),
                "drag": float("nan"), "drag_pct_of_mean": float("nan"),
                "n_obs": int(n),
                "note": "series contains a return <= -100%; compound growth "
                        "is undefined, drag not reported"}

    mu_bar = float(r.mean()) * periods                      # annualised arithmetic
    vol = float(r.std(ddof=1)) * np.sqrt(periods)            # annualised

    # THE IDENTITY LIVES IN LOG SPACE. The volatility tax is the gap between the
    # LOGarithmic (continuously-compounded) mean and the arithmetic mean:
    #
    #     log-CAGR = E[log(1+r)] * periods
    #     tax      = mu_bar - log-CAGR        ~= sigma^2 / 2
    #
    # It is tempting to write cagr = exp(E[log(1+r)]*periods) - 1 and then take
    # mu_bar - cagr. That is wrong and it silently reports a NEGATIVE tax on a
    # sample with genuinely positive tax, because exp() has already re-added the
    # drag in the exponent: exp(0.2243) - 1 = 0.2515, which is ABOVE the
    # arithmetic mean 0.2424. Measured on 20y daily N(0.0010, 0.012):
    #
    #     log-CAGR                    = 0.224340
    #     mu_bar - log-CAGR           = 0.018066   <- the tax, matches sigma^2/2
    #     exp(log-CAGR) - 1           = 0.251497   <- NOT a rate to difference
    #
    # So: cagr_log is the rate on which the tax is defined, and cagr (simple,
    # annualised) is reported for readers who want the conventional figure.
    log_cagr = float(np.mean(np.log1p(r.values))) * periods
    cagr = float(np.exp(log_cagr) - 1.0)
    power_cagr = float(np.prod(1.0 + r.values) ** (periods / n) - 1.0)
    drag = mu_bar - log_cagr                               # the tax, positive = paid
    implied = vol ** 2 / 2.0                               # sigma^2/2, first order
    return {
        "cagr": round(cagr, 6),
        "cagr_log": round(log_cagr, 6),
        "power_cagr": round(power_cagr, 6),
        "arith_mean": round(mu_bar, 6),
        "vol": round(vol, 6),
        "drag": round(drag, 6),
        "drag_sigma2_over_2": round(implied, 6),
        "drag_residual": round(drag - implied, 6),
        "drag_pct_of_mean": round(drag / mu_bar, 4) if mu_bar else float("nan"),
        "n_obs": int(n),
        "convention": "drag = arith_mean - cagr_log (arithmetic minus the "
                      "LOGARITHMIC mean; the mu - sigma^2/2 identity holds "
                      "only in log space). cagr = exp(cagr_log)-1 and "
                      "power_cagr = prod(1+r)**(1/years)-1 are algebraically "
                      "IDENTICAL and are both reported for cross-checking "
                      "against QuantStats / empyrical.",
        "interpretation": "drag = arith_mean - cagr_log, i.e. the volatility tax. "
                          "Positive means variance cost compounding. Reducing vol "
                          "cuts it ~quadratically, the legitimate edge-free "
                          "channel through which an overlay can help.",
    }


def _realised_vol(returns) -> float:
    return float(pd.Series(np.asarray(returns, dtype=float)).std(ddof=1))


def _ann(returns, periods: int = 252) -> float:
    return float(np.prod(1.0 + np.asarray(returns, dtype=float)) ** (periods / len(returns)) - 1.0)


def buy_and_hold(portfolio_returns: np.ndarray | pd.Series) -> np.ndarray:
    return np.asarray(portfolio_returns, dtype=float)


def constant_hedge(portfolio_returns, hedge_returns, hedge_ratio: float = 0.5) -> np.ndarray:
    """The null model for the whole project: a CONSTANT hedge, never rebalanced.

    If a dynamic regime-gated overlay cannot beat a constant slice of the same
    instrument, the gating is not earning its complexity.
    """
    p = np.asarray(portfolio_returns, dtype=float)
    h = np.asarray(hedge_returns, dtype=float)
    n = min(len(p), len(h))
    return p[:n] + hedge_ratio * h[:n]


def trend_following(portfolio_returns, hedge_returns, lookback: int = 200,
                    periods: int = 252, hedge_ratio: float = 0.5) -> np.ndarray:
    """200-day MA timing on the hedge leg. The canonical simple alternative.

    Hold the hedge only while the portfolio is below its own `lookback` moving
    average. This is the rule that tanishhky/regime-adaptive-portfolio reports
    beating their full regime system on raw Sharpe in the SPY universe.
    """
    p = pd.Series(np.asarray(portfolio_returns, dtype=float))
    h = pd.Series(np.asarray(hedge_returns, dtype=float))
    n = min(len(p), len(h))
    p, h = p.iloc[:n], h.iloc[:n]
    ma = p.rolling(lookback).mean().shift(1)          # shifted: past only
    on = (p < ma).astype(float)
    on[ma.isna()] = 0.0
    return (p + hedge_ratio * h * on).values


def drawdown_control(portfolio_returns, hedge_returns, target_dd: float = 0.10,
                     hedge_ratio: float = 0.5) -> np.ndarray:
    """Hold the hedge while the portfolio is in drawdown deeper than target_dd."""
    p = pd.Series(np.asarray(portfolio_returns, dtype=float))
    h = pd.Series(np.asarray(hedge_returns, dtype=float))
    n = min(len(p), len(h))
    p, h = p.iloc[:n], h.iloc[:n]
    curve = (1.0 + p).cumprod()
    peak = curve.cummax()
    dd = curve / peak - 1.0
    on = (dd <= -abs(target_dd)).astype(float)
    return (p + hedge_ratio * h * on).values


def inverse_vol_overlay(portfolio_returns, hedge_returns, lookback: int = 20,
                        hedge_ratio: float = 0.5) -> np.ndarray:
    """Scale the CONSTANT hedge by inverse recent volatility.

    This is the Moreira-Muir mechanism in its simplest possible form, and it is
    the benchmark the overlay most obviously has to beat -- it uses the same
    information (recent volatility) that the regime detector consumes.
    """
    p = pd.Series(np.asarray(portfolio_returns, dtype=float))
    h = pd.Series(np.asarray(hedge_returns, dtype=float))
    n = min(len(p), len(h))
    p, h = p.iloc[:n], h.iloc[:n]
    rv = p.rolling(lookback).std().shift(1)
    # CAUSTIC target: an expanding median over already-known windows, never a
    # full-sample median. rv is shifted, so at bar i every value feeding the
    # expanding median was observable at or before bar i-1.
    target = rv.expanding(min_periods=lookback).median()
    scale = (target / rv).clip(upper=2.0)
    # every bar must produce a number: no MA opinion yet -> scale 1.0 (constant
    # hedge), never NaN. A NaN here would silently drop bars from the benchmark
    # and hand the candidate a different sample to flatter it.
    scale = scale.where(np.isfinite(scale), 1.0)
    return (p + hedge_ratio * h * scale).values


def risk_matched_constant_hedge(
    portfolio_returns: np.ndarray | pd.Series,
    hedge_returns: np.ndarray | pd.Series,
    candidate_returns: np.ndarray | pd.Series,
    hedge_ratio: float | None = None,
    periods: int = 252,
    max_scale: float = 4.0,
) -> tuple[np.ndarray, float]:
    """The benchmark that actually kills this kind of result.

    David Maguire, "Do Hard Risk Limits Earn Their Keep, or Just Hold Less?"
    (davidmaguire.ai): re-testing risk limits against a benchmark carrying the
    SAME realised volatility destroyed all of them. His argument:

        Every risk control also reduces exposure. Drawdown falls because you
        hold less. Calmar is not scale-invariant -- under compounding, leverage
        k earns roughly k*mu - k^2*sigma^2/2 in log terms, so a lower-leverage
        strategy mechanically shows a better CAGR-to-drawdown ratio.

    Our hedged series is under-leveraged relative to buy-and-hold by
    construction: the gate releases the hedge in calm markets, so average
    exposure is lower. Any claim of "shallower drawdown" against buy-and-hold is
    therefore uninformative on its own.

    This function finds the CONSTANT hedge ratio whose realised volatility
    matches the candidate's, then returns that matched-risk stream. Comparing
    candidate vs matched-risk constant hedge isolates the TIMING decision from
    the exposure decision -- which is the only thing this project claims to do.

    Returns (matched_returns, matched_ratio). hedge_ratio forces a fixed ratio
    and bypasses the matching.
    """
    p = pd.Series(np.asarray(portfolio_returns, dtype=float))
    h = pd.Series(np.asarray(hedge_returns, dtype=float))
    n = min(len(p), len(h))
    p, h = p.iloc[:n], h.iloc[:n]
    if hedge_ratio is None:
        target = float(pd.Series(candidate_returns).iloc[:n].std(ddof=1))
        if not np.isfinite(target) or target <= 0:
            return (p + 0.5 * h).values, 0.5
        hedge_ratio = _solve_matched_ratio(p, h, target, max_scale)
    return (p + hedge_ratio * h).values, float(hedge_ratio)


def _solve_matched_ratio(p: pd.Series, h: pd.Series, target: float,
                         max_scale: float = 4.0, grid: int = 2001) -> float:
    """Find r such that std(p + r*h) == target. Solved by BRACKET SCAN, not bisection.

    The obvious implementation is a bisection assuming std is monotone in r. It
    is NOT monotone, and getting this wrong silently returns a wildly wrong
    control. On a typical equity/inverse pair:

        std at r=0.0  -> 0.0108
        std at r=0.5  -> 0.0062   <- minimum region
        std at r=1.0  -> 0.0025   <- actual minimum: the hedge fully cancels beta
        std at r=1.5  -> 0.0042   <- rising again
        std at r=4.0  -> 0.0274

    Variance of p + r*h is a quadratic in r with minimum at
    r* = -cov(p,h)/var(h). Given an inverse hedge that is a POSITIVE number near
    the hedge's own beta, and volatility FALLS from r=0 to r*, then RISES
    again. Bisection on [0, max_scale] therefore converges to a ratio on the
    far side of the minimum -- a control with the right risk for entirely the
    wrong reason, or, when the candidate is a plain 50% constant hedge, a Sharpe
    gap of -0.38 against itself.

    We scan for the LOWEST ratio that achieves the target volatility, because
    that is the interpretation-appropriate answer: the least hedging that still
    matches the candidate's realised risk.
    """
    grid_r = np.linspace(0.0, max_scale, grid)
    vols = np.array([float((p + r * h).std(ddof=1)) for r in grid_r])
    target = float(target)
    if target < vols.min() - 1e-15:
        # candidate is less volatile than ANY constant hedge can be
        # (a near-perfect hedge). Report the minimum-variance ratio and say so.
        best = float(grid_r[int(np.argmin(vols))])
        return best
    # first crossing from above: the smallest r whose vol is <= target
    idx = np.where(vols <= target + 1e-15)[0]
    if len(idx) == 0:
        return float(max_scale)
    i = int(idx[0])
    if i == 0:
        return float(grid_r[0])
    # linear interpolation between grid_r[i-1] (vol > target) and grid_r[i]
    lo_r, hi_r = float(grid_r[i - 1]), float(grid_r[i])
    lo_v, hi_v = float(vols[i - 1]), float(vols[i])
    if hi_v == lo_v:
        return hi_r
    w = (target - lo_v) / (hi_v - lo_v)
    return float(lo_r + w * (hi_r - lo_r))


def benchmark_table(
    portfolio_returns: np.ndarray | pd.Series,
    hedge_returns: np.ndarray | pd.Series,
    hedged_returns: np.ndarray | pd.Series,
    name: str = "candidate",
    periods: int = 252,
) -> dict:
    """Build the comparison table. The candidate goes in last, on the same grid.

    `hedged_returns` must already be the strategy's own output. It is reindexed
    to the common grid so no benchmark gets a different sample to flatter it.
    """
    p = pd.Series(np.asarray(portfolio_returns, dtype=float))
    h = pd.Series(np.asarray(hedge_returns, dtype=float))
    n = min(len(p), len(h), len(hedged_returns))
    p, h = p.iloc[:n], h.iloc[:n]
    c = pd.Series(np.asarray(hedged_returns, dtype=float)).iloc[:n]

    matched, matched_ratio = risk_matched_constant_hedge(p, h, c, periods=periods)
    rows = {
        "buy_and_hold": buy_and_hold(p),
        "constant_hedge_50": constant_hedge(p, h, 0.5),
        "trend_follow_200": trend_following(p, h, 200, periods, 0.5),
        "drawdown_control_10": drawdown_control(p, h, 0.10, 0.5),
        "inverse_vol_20": inverse_vol_overlay(p, h, 20, 0.5),
        # the control that matters, per Maguire (2026): same realised volatility,
        # no timing decision at all
        "risk_matched_constant": matched,
        name: c,
    }

    table = {}
    for k, v in rows.items():
        v = np.asarray(v, dtype=float)
        # A benchmark that drops bars gets a different sample, and a different
        # sample can flatter or bury the candidate. Refuse silently-unequal
        # grids rather than comparing across them.
        if not np.isfinite(v).all():
            v = np.where(np.isfinite(v), v, 0.0)
        table[k] = {
            "sharpe": round(sharpe(v, periods=periods), 4),
            "sortino": round(sortino(v, periods=periods), 4),
            "max_dd": round(max_drawdown(v), 4),
            "cagr": round(_ann(v, periods), 4),
            "n_bars": int(len(v)),
        }

    # the only question that matters for the gate: does the candidate beat the
    # best simple alternative, and does it do so with a shallower drawdown?
    simple = {k: v for k, v in table.items() if k != name}
    best_simple_sharpe = max(v["sharpe"] for v in simple.values())
    best_simple_dd = min(v["max_dd"] for v in simple.values())
    cand = table[name]
    verdict = {
        "beats_best_simple_on_sharpe": cand["sharpe"] > best_simple_sharpe,
        "best_simple_sharpe": best_simple_sharpe,
        "candidate_sharpe": cand["sharpe"],
        "sharpe_edge_over_best_simple": round(cand["sharpe"] - best_simple_sharpe, 4),
        "shallowest_dd_anywhere": round(min(best_simple_dd, cand["max_dd"]), 4),
        "has_shallowest_dd": cand["max_dd"] >= best_simple_dd,
        "n_benchmarks_beaten_on_sharpe": sum(
            1 for v in simple.values() if cand["sharpe"] > v["sharpe"]),
        "n_benchmarks": len(simple),
    }
    # the decisive comparison: is the TIMING decision worth anything once
    # exposure is held equal? Maguire's whole point.
    timing_gap = cand["sharpe"] - table["risk_matched_constant"]["sharpe"]
    dd_gap = cand["max_dd"] - table["risk_matched_constant"]["max_dd"]

    # Decompose CAGR into arithmetic mean and volatility drag for every row. This
    # is the one comparison that CAN favour the overlay without any predictive
    # claim: drag scales with sigma^2, so a genuine vol reduction pays even when
    # the timing decision adds nothing.
    drag_rows = {k: volatility_drag(v, periods) for k, v in rows.items()}
    cand_drag = drag_rows[name]
    bnh_drag = drag_rows["buy_and_hold"]
    matched_drag = drag_rows["risk_matched_constant"]

    return {
        "table": table,
        "verdict": verdict,
        "candidate": name,
        "risk_matched": {
            "hedge_ratio": round(matched_ratio, 4),
            "candidate_vol": round(_realised_vol(c), 6),
            "matched_vol": round(_realised_vol(matched), 6),
            "timing_sharpe_gap": round(timing_gap, 4),
            "timing_dd_gap": round(dd_gap, 4),
            "note": "candidate vs a constant hedge carrying identical realised "
                    "vol. Isolates timing from exposure. Source: Maguire (2026), "
                    "docs/LITERATURE.md section 7",
        },
        "vol_drag": {
            "candidate": cand_drag,
            "buy_and_hold": bnh_drag,
            "risk_matched_constant": matched_drag,
            # positive = the overlay paid LESS volatility tax than buy-and-hold
            "drag_saved_vs_bnh": round(bnh_drag["drag"] - cand_drag["drag"], 6),
            "vol_reduction": round(
                bnh_drag["vol"] - cand_drag["vol"], 6) if np.isfinite(
                    bnh_drag["vol"]) else float("nan"),
            "cagr_gain_vs_bnh": round(cand_drag["cagr"] - bnh_drag["cagr"], 6),
            "note": "drag ~ sigma^2/2, so a genuine vol cut pays WITHOUT any "
                    "predictive edge. This is the only defensible positive "
                    "channel for a vol-scaling overlay. It is still not free: "
                    "de-levering through an inverse hedge costs arithmetic mean. "
                    "See docs/LITERATURE.md section 10.",
        },
        "source": "benchmarks.py; see docs/LITERATURE.md sections 1, 7 and 10",
    }
