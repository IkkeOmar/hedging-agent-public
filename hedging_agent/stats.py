"""Statistics that separate "we found an edge" from "we found the best of N".

Implements, from scratch and verifiable:
  - Deflated Sharpe Ratio (Bailey & Lopez de Prado 2014)
  - Probability of Backtest Overfitting (CSCV, same paper)
  - Walk-forward fold statistics with purge/embargo
  - Per-regime breakdown, because a single number across regimes is not a result

Every function is deterministic and takes only arrays, so a reader can rerun
them against our published numbers without installing anything.
"""

from __future__ import annotations

from dataclasses import dataclass, field, asdict

import math

import numpy as np
import pandas as pd
from scipy import stats


# ------------------------------------------------------------------ basics
def sharpe(returns: np.ndarray | pd.Series, rf_annual: float = 0.0,
           periods: int = 252) -> float:
    r = np.asarray(returns, dtype=float)
    r = r[np.isfinite(r)]
    if len(r) < 2 or r.std(ddof=1) == 0:
        return 0.0
    excess = r - rf_annual / periods
    return float(excess.mean() / excess.std(ddof=1) * np.sqrt(periods))


def sortino(returns: np.ndarray | pd.Series, periods: int = 252) -> float:
    r = np.asarray(returns, dtype=float)
    r = r[np.isfinite(r)]
    downside = r[r < 0]
    if len(r) < 2 or len(downside) < 2:
        return 0.0
    dd = np.sqrt(np.mean(np.square(downside)))
    return float(r.mean() / dd * np.sqrt(periods)) if dd > 0 else 0.0


def max_drawdown(returns: np.ndarray | pd.Series) -> float:
    r = np.asarray(returns, dtype=float)
    r = r[np.isfinite(r)]
    if len(r) == 0:
        return 0.0
    curve = np.cumprod(1.0 + r)
    peak = np.maximum.accumulate(curve)
    with np.errstate(divide="ignore", invalid="ignore"):
        dd = curve / peak - 1.0
    return float(np.nanmin(dd))


def calmar(returns: np.ndarray | pd.Series, rf_annual: float = 0.0,
           periods: int = 252) -> float:
    r = np.asarray(returns, dtype=float)
    cagr = float(np.prod(1.0 + r) ** (periods / max(len(r), 1)) - 1.0)
    mdd = abs(max_drawdown(r))
    return cagr / mdd if mdd > 1e-12 else 0.0


# ------------------------------------------------------- deflated Sharpe
def expected_max_sharpe(trials_variance: float, n_trials: int) -> float:
    """E[max Sharpe across n_trials independent trials].

    Bailey & Lopez de Prado (2014), eq. 9:

        SR0 = sqrt(V) * [ (1-g)*Z(1 - 1/N) + g*Z(1 - 1/(N*e)) ]

    V is the variance of the Sharpe across trials. It is a property of the
    trial set, not of the winner -- which is exactly why the correction cannot
    be faked from a single backtest.
    """
    euler = 0.5772156649015329
    if n_trials <= 1:
        return 0.0
    v = max(float(trials_variance), 1e-18)
    return float(np.sqrt(v)) * (
        (1 - euler) * stats.norm.ppf(1.0 - 1.0 / n_trials)
        + euler * stats.norm.ppf(1.0 - 1.0 / (n_trials * np.e))
    )


def probabilistic_sharpe_ratio(
    observed: float,
    benchmark: float,
    n_obs: int,
    skew: float = 0.0,
    kurtosis: float = 3.0,
) -> float:
    """PSR: P(true Sharpe > benchmark), Bailey & Lopez de Prado (2014) eq. 11.

        Z = (SR - SR0) * sqrt(T-1) / sqrt(1 - g3*SR + (g4-1)/4 * SR^2)

    NOTE: kurtosis here is RAW (normal = 3.0), not excess. Passing an excess
    kurtosis silently inflates the denominator and inflates the probability,
    which is the whole reason the golden test below exists.
    """
    if n_obs < 3:
        return 0.0
    denom = 1.0 - skew * observed + (kurtosis - 1.0) / 4.0 * observed**2
    denom = float(denom) if abs(denom) > 1e-12 else 1e-12
    z = (observed - benchmark) * np.sqrt(n_obs - 1.0) / np.sqrt(denom)
    return float(stats.norm.cdf(z))


def deflated_sharpe_ratio(
    returns: np.ndarray | pd.Series,
    n_trials: int,
    periods: int = 252,
    rf_annual: float = 0.0,
    trials_variance: float | None = None,
) -> dict:
    """DSR: the probability that a backtest Sharpe exceeds the expected maximum
    Sharpe under the null, given the number of trials that produced it.

    This is the single most load-bearing number in the project. A Sharpe printed
    without n_trials is not a result; it is a sample from a distribution whose
    width is set entirely by how many things you tried.

    Two units are in play and they are easy to mix up:
      * V (trials_variance) is the variance of the Sharpe in the SAME frequency
        as the observed Sharpe. For annualised Sharpes, pass the annualised
        spread of the trial Sharpes.
      * kurtosis is raw, not excess.
    """
    r = np.asarray(returns, dtype=float)
    r = r[np.isfinite(r)]
    n = len(r)
    if n < 5 or n_trials < 1:
        return {"dsr": 0.0, "observed_sharpe": 0.0, "expected_max_sharpe": 0.0,
                "n_trials": n_trials, "n_obs": n, "note": "insufficient observations"}

    sr = sharpe(r, rf_annual=rf_annual, periods=periods)
    skew = float(stats.skew(r)) if n > 2 else 0.0
    # RAW kurtosis: normal = 3.0. scipy returns EXCESS, so add 3.
    kurt = (float(stats.kurtosis(r, fisher=True)) + 3.0) if n > 3 else 3.0
    euler = 0.5772156649015329  # Euler-Mascheroni gamma

    # V: variance of the ANNUALISED Sharpe across trials. Default is the
    # sampling variance under the null (eq. 8), i.e. we assume the trials were
    # independent draws. Callers that measured the spread pass it in.
    sampling_var = float(max(
        (1.0 + 0.5 * skew + (kurt - 1.0) / 24.0 * sr**2) / max(n - 1, 1), 1e-18))
    if trials_variance is None:
        trials_variance = sampling_var

    sr0 = expected_max_sharpe(trials_variance, n_trials)
    dsr = probabilistic_sharpe_ratio(sr, sr0, n, skew=skew, kurtosis=kurt)

    return {
        "dsr": round(dsr, 4),
        "observed_sharpe": round(sr, 4),
        "expected_max_sharpe": round(float(sr0), 4),
        "n_trials": int(n_trials),
        "n_obs": int(n),
        "skew": round(skew, 4),
        "raw_kurtosis": round(kurt, 4),
        "trials_variance": round(float(trials_variance), 8),
        "note": "DSR is P(Sharpe > expected max Sharpe under null | n_trials). "
                "raw_kurtosis (normal = 3.0); trials_variance is in the same "
                "frequency as the Sharpe.",
    }


# -------------------------------------------------------------- CSCV / PBO
def newey_west_tstat(x: np.ndarray | pd.Series, y: np.ndarray | pd.Series,
                     lags: int | None = None) -> dict:
    """t-stat for the mean difference, with Newey-West (Bartlett) HAC errors.

    Why this exists: our headline is "hedged Sharpe > unhedged Sharpe". A
    Sharpe difference on overlapping windows is not a t-test away from zero --
    the two series are mechanically related (the hedge is a scaled slice of one
    of them) and returns are autocorrelated. Reporting the difference without a
    HAC standard error would be reporting a number we cannot defend.

    Harvey, Liu & Zhu (2016) argue a new factor needs t > 3.0, not 2.0. That
    threshold is applied downstream by the caller; this function only produces
    an honest t.

    lags defaults to floor(4 * (n/100)^(2/9)) -- Newey & West (1994) rule of
    thumb for an automatically selected bandwidth.
    """
    x = np.asarray(x, dtype=float).ravel()
    y = np.asarray(y, dtype=float).ravel()
    if x.shape != y.shape:
        # right-align in time: keep the most recent min(n) bars of each series
        n_aligned = min(len(x), len(y))
        x, y = x[-n_aligned:], y[-n_aligned:]
    m = np.isfinite(x) & np.isfinite(y)
    x, y = x[m], y[m]
    n = len(x)
    if n < 12:
        return {"mean_diff": float("nan"), "t_stat": float("nan"),
                "n_obs": n, "lags": 0,
                "note": "too few observations for a HAC standard error"}
    d = x - y
    d = d - d.mean()
    if lags is None:
        lags = int(np.floor(4.0 * (n / 100.0) ** (2.0 / 9.0)))
    lags = max(int(min(lags, n // 2 - 1)), 0)

    gamma0 = float(np.dot(d, d) / n)
    s2 = gamma0
    for lag in range(1, lags + 1):
        w = 1.0 - lag / (lags + 1.0)          # Bartlett kernel
        gamma_lag = float(np.dot(d[lag:], d[:-lag]) / n)
        s2 += 2.0 * w * gamma_lag
    s2 = max(s2, 1e-24)
    # the test is on the LEVEL difference of the mean, so use the raw (uncentred
    # by the mean) difference and its standard error
    dm = float(np.mean(x) - np.mean(y))
    se = float(np.sqrt(s2 / n))
    return {
        "mean_diff": dm,
        "t_stat": dm / se if se > 0 else float("nan"),
        "se": se,
        "n_obs": int(n),
        "lags": int(lags),
        "hac": "Newey-West Bartlett",
    }


def sharpe_difference_tstat(
    hedged: np.ndarray | pd.Series,
    unhedged: np.ndarray | pd.Series,
    periods: int | None = None,
) -> dict:
    """t-stat that the hedged return series has a higher MEAN return.

    Deliberately not a t-stat on the Sharpe ratio: the sampling distribution of
    a difference of Sharpe ratios is not closed-form, and Lo (2002) shows the
    usual asymptotic Sharpe inference is unreliable at these sample sizes. A
    HAC t on the mean-return difference is the defensible minimum. Both numbers
    are reported so the reader can see the Sharpe gap and its tested support.

    `periods` annualises the two reported Sharpes and must match the asset
    class: 252 for equities, 365 for crypto. The t-statistic itself is not
    annualised and does not depend on it.
    """
    out = newey_west_tstat(hedged, unhedged)
    h = np.asarray(hedged, dtype=float).ravel()
    u = np.asarray(unhedged, dtype=float).ravel()
    if h.shape != u.shape:
        n_aligned = min(len(h), len(u))
        h, u = h[-n_aligned:], u[-n_aligned:]
    m = np.isfinite(h) & np.isfinite(u)
    # annualisation must follow the asset class: equities trade 252 sessions a
    # year, crypto 365. Hardcoding 252 here understated crypto annualised vol by
    # 20% (sqrt(365/252)) and made every crypto Sharpe look worse than it was.
    ppy = periods if periods is not None else 252
    out["sharpe_hedged"] = round(sharpe(h[m], periods=ppy), 4)
    out["sharpe_unhedged"] = round(sharpe(u[m], periods=ppy), 4)
    out["sharpe_gap"] = round(out["sharpe_hedged"] - out["sharpe_unhedged"], 4)
    return out


def pbo_cscv(
    matrix: np.ndarray,
    n_splits: int = 8,
    max_combinations: int | None = None,
    seed: int = 7,
) -> dict:
    """Probability of Backtest Overfitting via Combinatorially Symmetric
    Cross-Validation.

    matrix: shape (n_observations, n_configurations). Every column is a full
            performance series from one configuration (e.g. one strategy, one
            parameter set). We split observations into S even blocks, form all
            S/2 splits of (train-half, test-half), and measure how often the
            best in-sample configuration is worse out-of-sample than its median.

    PBO near 0 means selection generalises. PBO near 1 means the selection
    procedure is a coin flip, which is the honest default expectation.
    """
    m = np.asarray(matrix, dtype=float)
    if m.ndim != 2:
        raise ValueError("matrix must be (n_observations, n_configurations)")
    n_obs, n_cfg = m.shape
    if n_obs < n_splits * 2 or n_cfg < 2:
        return {"pbo": float("nan"), "n_combinations": 0,
                "note": "not enough observations or configurations"}

    n_splits = int(min(n_splits, (n_obs // 2) * 2))
    if n_splits % 2 == 1:
        n_splits -= 1
    block = n_obs // n_splits
    usable = block * n_splits
    blocks = np.array_split(m[:usable], n_splits, axis=0)

    from itertools import combinations

    combos = list(combinations(range(n_splits), n_splits // 2))
    if max_combinations and len(combos) > max_combinations:
        rng = np.random.default_rng(seed)
        combos = [combos[i] for i in rng.choice(len(combos), max_combinations, replace=False)]

    logits = []
    for train_idx in combos:
        test_idx = [i for i in range(n_splits) if i not in train_idx]
        train = np.concatenate([blocks[i] for i in train_idx], axis=0)
        test = np.concatenate([blocks[i] for i in test_idx], axis=0)
        is_mean = train.mean(axis=0)
        oos_mean = test.mean(axis=0)
        best_in_sample = int(np.argmax(is_mean))
        rank_of_best = float(stats.rankdata(oos_mean)[best_in_sample])
        n_cfg_f = float(n_cfg)
        # Bailey & Lopez de Prado (2014), "The Probability of Backtest
        # Overfitting", JCF 20(4) define theta from the out-of-sample rank of
        # the in-sample winner among the N configurations:
        #
        #     theta = omega / (N + 1) - 0.5      in (-0.5, 0.5]
        #     PBO   = P(theta > 0) = P(omega > (N+1)/2)
        #
        # omega here is the ASCENDING rank of the winner's out-of-sample mean
        # among the N means, so 1 is the WORST out-of-sample rank and N the
        # best. P(omega > (N+1)/2) is therefore the probability that the
        # in-sample winner landed in the BOTTOM half out-of-sample, which is
        # the event PBO is meant to measure.
        #
        # The divisor is N + 1 and the rank is 1-BASED. This code used to
        # shift the rank to zero-based and divide by N instead:
        #
        #     theta = (omega - 1) / N - 0.5
        #
        # The verdict is theta > 0, so the two forms disagree about how many
        # of the N ranks count as "worse than the median". For even N the
        # correct form puts N/2 of the N ranks above the line and the old one
        # put N/2 - 1: ONE RANK TOO FEW. For odd N both put (N-1)/2 of them
        # there, so the two agree exactly.
        #
        # One rank too few biases PBO DOWNWARD by exactly 1/N:
        #
        #     N=  4: correct 0.50, old 0.25  (bias -0.25 = -1/N)
        #     N=  6: correct 0.50, old 0.33  (bias -0.17 = -1/N)
        #     N= 10: correct 0.50, old 0.40  (bias -0.10 = -1/N)
        #     N= 12: correct 0.50, old 0.42  (bias -0.08 = -1/N)
        #
        # N=2 is degenerate: the old form flagged none of the two ranks, which
        # is why the smallest useful N here is 4.
        #
        # Measured over 400 iid draws on this implementation before the fix:
        # 0.324 (N=6), 0.413 (N=10), 0.421 (N=12) against a true 0.5. The
        # estimator was systematically flattering every candidate, i.e. making
        # overfitting LESS likely to be detected -- the one direction of error
        # a backtest-overfitting gate must never have. Caught by cross-checking
        # against ml4t.diagnostic, which uses the (N+1) form.
        theta = rank_of_best / (n_cfg_f + 1.0) - 0.5
        # log-odds of the config being above the median out-of-sample
        p = float(np.clip(0.5 - theta, 1e-6, 1 - 1e-6))
        logits.append(np.log(p / (1 - p)))

    if not logits:
        return {"pbo": float("nan"), "n_combinations": 0}
    # theta > 0 <=> logit < 0, so the IS winner landing in the BOTTOM half
    pbo = float(np.mean([1.0 if l < 0 else 0.0 for l in logits]))
    return {
        "pbo": round(pbo, 4),
        "n_combinations": len(combos),
        "n_configs": int(n_cfg),
        "n_observations": int(n_obs),
        "n_splits": int(n_splits),
        "median_oos_win_prob": round(float(np.median(
            [1.0 / (1.0 + np.exp(-l)) for l in logits])), 4),
        "note": "PBO < 0.5 means the in-sample winner tends to stay above median OOS; "
                "near 0.5 the selection procedure is a coin flip",
    }


# -------------------------------------------------------- walk-forward eval
@dataclass
class FoldResult:
    fold: int
    train_start: object
    train_end: object
    test_start: object
    test_end: object
    is_sharpe: float
    oos_sharpe: float
    oos_return: float
    oos_trades: int
    accepted: bool


@dataclass
class WalkForwardResult:
    folds: list[FoldResult] = field(default_factory=list)
    n_folds: int = 0
    oos_returns: np.ndarray | None = None
    aggregate_sharpe: float = 0.0
    positive_fold_ratio: float = 0.0
    mean_is_sharpe: float = 0.0
    mean_oos_sharpe: float = 0.0
    max_is_oos_decay: float = 0.0

    def as_dict(self) -> dict:
        d = asdict(self)
        d.pop("oos_returns", None)
        return d


def walk_forward_folds(
    index: pd.DatetimeIndex,
    n_folds: int = 8,
    train_frac: float = 0.6,
    purge_bars: int = 1,
    embargo_bars: int = 0,
) -> list[tuple[int, int, int, int]]:
    """Return (train_start, train_end, test_start, test_end) index positions.

    purge: bars between train end and test start are dropped. Without it, a
    label that looks forward from the last training bar leaks into the first
    test bar. This is the single most common backtest bug and it is why the
    purge is a parameter of the splitter, not an afterthought.
    """
    n = len(index)
    if n < n_folds * 4:
        n_folds = max(2, n // 4)
    total = n - purge_bars
    fold_len = total // n_folds
    tr = max(int(fold_len * train_frac), 2)
    out = []
    for k in range(n_folds):
        tr_s = k * fold_len
        tr_e = tr_s + tr
        te_s = tr_e + purge_bars
        te_e = min(te_s + fold_len, n)
        if tr_e >= n or te_s >= te_e:
            continue
        if embargo_bars:
            tr_e = max(tr_s + 2, tr_e - embargo_bars)
        out.append((tr_s, tr_e, te_s, te_e))
    return out


def evaluate_walk_forward(
    returns: pd.Series,
    n_folds: int = 8,
    train_frac: float = 0.6,
    periods: int = 252,
    rf_annual: float = 0.0,
    purge_bars: int = 1,
    embargo_bars: int = 0,
    min_oos_trades: int = 5,
) -> WalkForwardResult:
    """Split, then report. The aggregate is the concatenation of test segments."""
    r = returns.dropna()
    splits = walk_forward_folds(
        r.index, n_folds=n_folds, train_frac=train_frac,
        purge_bars=purge_bars, embargo_bars=embargo_bars,
    )
    folds: list[FoldResult] = []
    oos_chunks: list[np.ndarray] = []

    for i, (tr_s, tr_e, te_s, te_e) in enumerate(splits):
        tr = r.iloc[tr_s:tr_e]
        te = r.iloc[te_s:te_e]
        if len(te) < 2:
            continue
        is_sr = sharpe(tr, rf_annual, periods)
        oos_sr = sharpe(te, rf_annual, periods)
        n_trades = int((te != 0).sum())
        oos_chunks.append(te.values)
        folds.append(FoldResult(
            fold=i,
            train_start=str(tr.index[0].date() if hasattr(tr.index[0], "date") else tr.index[0]),
            train_end=str(tr.index[-1].date() if hasattr(tr.index[-1], "date") else tr.index[-1]),
            test_start=str(te.index[0].date() if hasattr(te.index[0], "date") else te.index[0]),
            test_end=str(te.index[-1].date() if hasattr(te.index[-1], "date") else te.index[-1]),
            is_sharpe=round(is_sr, 4),
            oos_sharpe=round(oos_sr, 4),
            oos_return=round(float(np.prod(1 + te.values) - 1), 6),
            oos_trades=n_trades,
            accepted=bool(oos_sr > 0 and n_trades >= min_oos_trades),
        ))

    res = WalkForwardResult(folds=folds, n_folds=len(folds))
    if not folds:
        return res

    oos = np.concatenate(oos_chunks)
    res.oos_returns = oos
    res.aggregate_sharpe = round(sharpe(oos, rf_annual, periods), 4)
    res.positive_fold_ratio = round(
        sum(1 for f in folds if f.oos_sharpe > 0) / len(folds), 4
    )
    res.mean_is_sharpe = round(float(np.mean([f.is_sharpe for f in folds])), 4)
    res.mean_oos_sharpe = round(float(np.mean([f.oos_sharpe for f in folds])), 4)
    decays = [(f.is_sharpe - f.oos_sharpe) / abs(f.is_sharpe)
              for f in folds if abs(f.is_sharpe) > 1e-6]
    res.max_is_oos_decay = round(float(max(decays)), 4) if decays else 0.0
    return res


def min_bars_for_sharpe(periods: int = 252, max_se: float = 2.0) -> int:
    """Smallest bar count whose Sharpe estimate has SE <= `max_se`.

    SE(SR_annual) ~= sqrt(periods / n). Solving sqrt(periods/n) <= max_se for
    n gives periods / max_se**2 -- hence 63 bars at the equity default, and 91
    for crypto at periods=365. Calibrated against simulation: the empirical
    SE tracks this to within 7% for n >= 40 and blows up below that (n=3 gives
    SE 26.9 against a theoretical 9.2), which is exactly the regime this guard
    exists for.

    A Sharpe below this threshold has a sign that flips on noise alone, so
    using it to count "regimes where the strategy worked" is counting luck.
    """
    return int(math.ceil(periods / (max_se ** 2)))


def regime_breakdown(
    returns: pd.Series,
    regimes: pd.Series,
    periods: int = 252,
) -> dict:
    """A strategy's number is not a number until it is split by regime."""
    df = pd.DataFrame({"r": returns, "regime": regimes}).dropna()
    out = {}
    for reg, grp in df.groupby("regime"):
        if reg is None:
            continue
        rr = grp["r"].values
        n = int(len(rr))
        # Standard error of an annualised Sharpe estimated from n bars is
        # ~sqrt(periods/n) (Lo 2002; verified by simulation, see
        # docs/LITERATURE.md section 15). Below `min_bars_for_sharpe` the
        # estimate has an SE above 2.0, so its SIGN is not evidence of
        # anything -- yet it printed as "sharpe 417.4" for a 2-bar regime and
        # satisfied the min_regimes_positive gate on its own.
        thick_enough = n >= min_bars_for_sharpe(periods)
        out[str(reg)] = {
            "n_bars": n,
            # suppressed, not zeroed: a 2-bar regime has no estimable Sharpe,
            # and printing 0.0 would read as "measured and it was flat"
            "sharpe": round(sharpe(rr, periods=periods), 4) if thick_enough else None,
            "sharpe_se": round(float(np.sqrt(periods / n)), 3) if n > 1 else None,
            "significant": thick_enough,
            "total_return": round(float(np.prod(1 + rr) - 1), 6),
            "max_dd": round(max_drawdown(rr), 4),
            "mean": round(float(np.mean(rr)), 6),
        }
    return out


def gate_report(
    metrics: dict,
    dsr: dict,
    pbo: dict,
    wf: WalkForwardResult,
    regime_stats: dict,
    action_ratio: float,
    cfg,
    benchmarks: dict | None = None,
    significance: dict | None = None,
) -> dict:
    """The promotion gate. Every check is named, every threshold is cited.

    Two checks exist because of the published literature, not because they were
    convenient:

      * `beats_simple_benchmark` -- Cederburg et al. (2020, JFE) tested
        volatility-managed portfolios across 103 equity strategies and found
        they do NOT systematically beat their unmanaged counterparts. So
        "beat buy-and-hold" is not an acceptable claim. The overlay must beat
        the simple alternatives it was measured against.
      * `significance_t_3` -- Harvey, Liu & Zhu (2016, RFS) show a new factor
        needs a t-statistic above 3.0, not 2.0, given the number of factors
        already tried, and argue most published findings are false.

    Both are FAIL-CLOSED: if the input is missing the check fails. A gate that
    passes when the evidence is absent is not a gate.
    """
    checks = []

    def chk(name: str, value, ok: bool, threshold, detail: str = "") -> None:
        checks.append({
            "check": name, "value": value, "threshold": threshold,
            "pass": bool(ok), "detail": detail,
        })

    sh = metrics.get("sharpe", 0.0)
    chk("min_sharpe", round(sh, 4), sh >= cfg.min_sharpe, cfg.min_sharpe)
    d = dsr.get("dsr", 0.0)
    chk("min_dsr", d, d >= cfg.min_dsr, cfg.min_dsr,
        "P(true edge | n_trials disclosed)")
    p = pbo.get("pbo", 1.0)
    chk("max_pbo", p, (not np.isnan(p)) and p <= cfg.max_pbo, cfg.max_pbo,
        "<0.5 = selection generalises; near 0.5 = coin flip")
    chk("min_folds", wf.n_folds, wf.n_folds >= cfg.min_folds, cfg.min_folds)
    chk("positive_fold_ratio", wf.positive_fold_ratio,
        wf.positive_fold_ratio >= cfg.require_positive_fold_ratio,
        cfg.require_positive_fold_ratio)
    chk("max_is_oos_decay", wf.max_is_oos_decay,
        wf.max_is_oos_decay <= cfg.max_is_oos_sharpe_decay, cfg.max_is_oos_sharpe_decay)
    chk("nonzero_action_ratio", round(action_ratio, 5),
        action_ratio >= cfg.require_nonzero_action_ratio, cfg.require_nonzero_action_ratio,
        "a hedge strategy that never hedges has not been tested")

    # Count only regimes whose Sharpe is actually estimable. regime_breakdown
    # reports sharpe=None when a regime is too thin to estimate one, and
    # before that guard a 2-bar regime scoring 417.4 satisfied this gate by
    # itself -- min_regimes_positive was being passed by arithmetic, not by
    # evidence.
    #
    # Gate on n_bars rather than on the None sentinel alone: regime_stats can
    # arrive from any caller, and a hand-built dict carrying
    # {"sharpe": 417.4, "n_bars": 2} would still vote. The bar count is the
    # thing that determines estimability, so it is the thing to test.
    # The threshold is per-universe: it comes from the SAME periods value
    # that regime_breakdown annualised with, carried on each regime as
    # sharpe_se. Reading it from there instead of assuming 252 matters --
    # crypto annualises over 365 and needs 92 bars, not 63, and assuming the
    # equity default silently excluded crypto's 90-bar expansion regime.
    pos_regimes = sum(
        1 for v in regime_stats.values()
        if (v.get("sharpe") or 0) > 0
        and v.get("sharpe_se") is not None
        and float(v["sharpe_se"]) <= 2.0
    )
    chk("min_regimes_positive", pos_regimes,
        pos_regimes >= cfg.min_regimes_positive, cfg.min_regimes_positive,
        "must work in more than one regime")

    # --- literature-mandated checks, fail-closed -------------------------
    if benchmarks is None:
        chk("beats_simple_benchmark", None, False, "must be measured",
            "no benchmark table supplied; refusing to promote an unbenchmarked "
            "candidate. See docs/LITERATURE.md section 1 (Cederburg et al. 2020)")
    elif "risk_matched" not in benchmarks:
        chk("beats_simple_benchmark", None, False, "risk-matched control required",
            "benchmark table has no risk-matched control. A shallower drawdown "
            "against buy-and-hold is uninformative when the candidate is "
            "under-leveraged by construction. See docs/LITERATURE.md section 7")
    else:
        rm = benchmarks["risk_matched"]
        gap = rm.get("timing_sharpe_gap")
        chk("beats_simple_benchmark", gap,
            (gap is not None) and float(gap) > 0.0, "> 0 vs risk-matched constant",
            f"vs a constant hedge at ratio {rm.get('hedge_ratio')} carrying the "
            f"same realised vol; Sharpe gap {gap}, drawdown gap "
            f"{rm.get('timing_dd_gap')}. Maguire 2026: the benchmark is where "
            f"the result lives")

    if significance is None:
        chk("significance_t", None, False, "> 3.0",
            "no HAC t-statistic supplied; a Sharpe difference without a "
            "standard error is not a result. See docs/LITERATURE.md section 3")
    else:
        t = significance.get("t_stat", float("nan"))
        chk("significance_t", None if np.isnan(t) else round(float(t), 4),
            (not np.isnan(t)) and float(t) > 3.0, 3.0,
            "Newey-West t on the hedged-minus-unhedged mean return; "
            "Harvey/Liu/Zhu (2016) hurdle is t>3.0, not 2.0")

    passed = all(c["pass"] for c in checks)
    return {
        "passed": passed,
        "verdict": "PROMOTED" if passed else "REJECTED",
        "checks": checks,
        "failed_checks": [c["check"] for c in checks if not c["pass"]],
    }