"""The pipeline: LLM candidate -> backtest -> promotion gate -> decision layer.

This module is the honest part of the project. It runs the sweep, it counts
every trial, and it refuses to promote anything that does not clear the gate.

Two modes:
  sweep   -- search the parameter space, report n_trials and the deflated Sharpe
             of the WINNER. This is the number that matters, and it is always
             paired with how many things were tried to find it.
  live    -- take a gate-passed spec, run it forward bar by bar, and hand the
             state to the decision layer (a swappable backend) at each step.
"""

from __future__ import annotations

import json
import sys
from dataclasses import dataclass, field, asdict

import numpy as np
import pandas as pd

from .config import Config, TIMEFRAMES, HedgeConfig, bars_per_year
from .data import (
    Universe, UNIVERSES, build_hedged_returns, hedge_returns, load_universe,
)
from .decision import build_backend, hedge_questions
from .policy import HedgeAction, decide_hedge
from .regime import RegimeState, detect_regime, rolling_regime
from .stats import (
    WalkForwardResult, deflated_sharpe_ratio, evaluate_walk_forward,
    gate_report, max_drawdown, pbo_cscv, regime_breakdown, sharpe, sortino,
    sharpe_difference_tstat,
)
from .benchmarks import benchmark_table
from .strategies import HedgeStrategySpec, generate_candidates, validate_spec


# ------------------------------------------------------------------ helpers
def _hedge_cfg_from_spec(spec: HedgeStrategySpec) -> HedgeConfig:
    return HedgeConfig(
        target_vol=spec.target_vol,
        vol_gamma=spec.vol_gamma,
        vol_floor_mult=0.0,
        vol_cap_mult=spec.vol_cap_mult,
        dd_trigger=spec.dd_trigger,
        dd_full=spec.dd_full,
        dd_max_mult=spec.dd_max_mult,
        consensus_required=spec.consensus_required,
        veto_on_expansion=spec.veto_on_expansion,
        max_hedge_fraction=spec.max_hedge_fraction,
        max_daily_turnover=spec.max_daily_turnover,
        # Cost model. Previously these were left at HedgeConfig defaults, which
        # meant a strategy could not vary its own cost assumption and a
        # cost-sensitivity analysis was impossible without editing the library.
        # They are policy knobs like every other field, so they belong here.
        commission_bps=spec.commission_bps,
        spread_bps=spec.spread_bps,
        borrow_bps_annual=spec.borrow_bps_annual,
        rebalance_every_bars=spec.rebalance_every_bars,
    )


def simulate_hedge(
    prices: pd.DataFrame,
    hedge_proxy: str,
    spec: HedgeStrategySpec,
    timeframe: str = "swing",
    warmup: int | None = None,
    periods: int = 252,
) -> dict:
    """Walk the regime detector forward and size a hedge at each bar.

    `periods` is the universe's own bars-per-year. It matters for the
    BORROW charge: the borrow rate is annual, so it is divided by the
    number of bars in a year before being charged to each bar. Passing the
    equity default of 252 for the crypto universe charges 365/252 = 1.45x
    the correct borrow -- a 45% overstatement that made every crypto
    result look worse than it was.
    """
    """Walk the regime detector forward and size a hedge at each bar.

    No look-ahead: the regime at bar t is computed from bars <= t only, and the
    hedge placed at t only affects returns from t+1 onward.
    """
    tf = TIMEFRAMES[timeframe]
    cfg = _hedge_cfg_from_spec(spec)
    warmup = warmup or tf.min_bars

    regime_df = rolling_regime(prices, tf, target_vol=cfg.target_vol, warmup=warmup)

    rows: list[HedgeAction] = []
    port_rets = prices.pct_change().mean(axis=1)
    curve = (1.0 + port_rets.fillna(0.0)).cumprod()
    drawdown = curve / curve.cummax() - 1.0

    prev_hedge = 0.0
    for dt, row in regime_df.iterrows():
        if row["regime"] is None or pd.isna(row.get("score")):
            continue
        window = prices.loc[:dt]
        # re-derive the state cheaply from the stored row so the cost stays linear
        state = _state_from_row(row, window, tf, cfg.target_vol)
        if state is None:
            continue
        dd = float(drawdown.loc[dt]) if dt in drawdown.index else 0.0
        act = decide_hedge(state, dd, prev_hedge, cfg)
        act.date = dt
        rows.append(act)
        prev_hedge = act.target_hedge

    if not rows:
        return {"error": "no regime verdicts produced; data too short for warmup",
                "warmup_required": warmup, "n_bars": len(prices)}

    notional = pd.Series({r.date: r.target_hedge for r in rows}).sort_index()
    port_cols = [c for c in prices.columns if c != hedge_proxy]
    hedged, hedge_diag = build_hedged_returns(
        prices, notional, hedge_proxy, portfolio_cols=port_cols,
        # commission + spread, applied to every trade actually placed.
        round_trip_cost_bps=cfg.commission_bps + cfg.spread_bps,
        borrow_bps_annual=cfg.borrow_bps_annual,
        periods=periods,
    )
    unhedged = (prices[port_cols].pct_change().mean(axis=1)
                .reindex(hedged.index).dropna())

    action_ratio = float((notional.diff().abs() > 1e-9).sum() / max(len(notional), 1))
    veto_ratio = float(sum(1 for r in rows if r.vetoed) / len(rows))
    zero_ratio = float(sum(1 for r in rows if r.target_hedge == 0.0) / len(rows))

    regimes = pd.Series({r.date: r.regime for r in rows}).sort_index()
    return {
        "spec": spec.as_dict(),
        "timeframe": timeframe,
        "hedge": hedge_diag,
        "n_bars": int(len(hedged)),
        "returns_hedged": hedged,
        "returns_unhedged": unhedged,
        # raw proxy returns on the hedged index: the benchmark table needs the
        # hedge leg alone, not the gated notional
        "hedge_returns": hedge_returns(prices, hedge_proxy).reindex(hedged.index),
        "hedge_notional": notional,
        "regimes": regimes,
        "actions": rows,
        "action_ratio": action_ratio,
        "veto_ratio": veto_ratio,
        "zero_hedge_ratio": zero_ratio,
        "avg_hedge": float(notional.mean()),
        "turnover_annualised": float(notional.diff().abs().sum()),
    }


def _state_from_row(row: pd.Series, window: pd.DataFrame, tf, target_vol: float):
    """Rebuild a minimal RegimeState from a cached rolling row (linear cost)."""
    from .regime import Regime, ScaleVote
    label = row["regime"]
    if label is None:
        return None
    dist = {r.value: 0.0 for r in Regime}
    dist[label] = float(np.clip(row.get("consensus") or 0.0, 0.0, 1.0))
    remaining = max(1.0 - dist[label], 1e-9)
    for r in Regime:
        if r.value != label:
            dist[r.value] = remaining / (len(Regime) - 1)
    return RegimeState(
        label=Regime(label),
        score=float(row.get("score") or 0.0),
        consensus=float(row.get("consensus") or 0.0),
        votes=[],
        vol_ratio=float(row.get("vol_ratio") or 1.0),
        trend_sign=int(row.get("trend_sign") or 0),
        distribution=dist,
        n_scales=len(tf.vol_lookbacks),
        rebound=bool(row.get("rebound", False)) or label == Regime.CRASH_REBOUND.value,
    )


# -------------------------------------------------------------------- sweep
@dataclass
class SweepResult:
    universe: str
    timeframe: str
    n_trials: int
    best_spec: dict
    best_metrics: dict
    dsr: dict
    pbo: dict
    walk_forward: dict
    regime_stats: dict
    gate: dict
    action_ratio: float
    veto_ratio: float
    zero_hedge_ratio: float
    top_candidates: list = field(default_factory=list)
    hedge: dict = field(default_factory=dict)
    # Evidence that makes the gate verdict checkable after the run. Without
    # these persisted, the JSON shows a FAIL on beats_simple_benchmark and
    # nobody can see which simple rule won.
    benchmarks: dict = field(default_factory=dict)
    significance: dict = field(default_factory=dict)
    # The window the numbers were measured over. A Sharpe without the
    # window it was measured in is not a result, it is a number. This
    # matters here because the window is NOT what the period argument
    # asks for: the crypto provider caps at 1000 daily bars, so a "5y"
    # request covers 2.74 years. A reader who cannot see the window
    # cannot see that.
    window: dict = field(default_factory=dict)

    def as_dict(self) -> dict:
        return asdict(self)


def _window_of(prices, requested_period: str = "") -> dict:
    """The window the sweep was measured over.

    The period argument says what to ASK for; this says what was GOT.
    The crypto provider returns at most 1000 daily bars, so a "5y"
    request covers 2.74 years of crypto while the same request covers
    five years of equities. A reader comparing the three universes needs
    to know that, and the only place it can be read is the artefact.

    The bars are counted, not assumed: `n_bars` is the number of price
    rows that went in, and `years` is derived from the dates, so the two
    cannot drift apart silently.
    """
    index = prices.index
    start = str(index[0].date())
    end = str(index[-1].date())
    days = (index[-1] - index[0]).days
    return {
        "start": start,
        "end": end,
        "n_bars": int(len(index)),
        "years": round(days / 365.25, 2),
        "requested_period": requested_period,
        # Only say something when the window is materially shorter than
        # the request. The crypto provider caps at 1000 daily bars, so a
        # 5y request covers 2.74 years there.
        "note": (
            f"the provider caps at 1000 daily bars, so a {requested_period} "
            f"request covers {round(days / 365.25, 2)} years here"
            if days < 365 * 4 else ""
        ),
    }


def _sig(value: float, digits: int = 6) -> float:
    """Round to SIGNIFICANT digits, not decimal places.

    The sweep artefact stores three numbers that span six orders of
    magnitude: `mean_diff` is ~1e-4, `se` is ~7e-5 and `t_stat` is ~2.7.
    Rounding them to four DECIMAL places would print the first two as
    0.0 -- so they are rounded to six significant digits instead, which
    keeps every one of them meaningful and keeps the file's bytes
    independent of the order in which the platform summed the return
    series.

    Why this matters: at full precision the committed sweep for
    `us_equities` stored `t_stat` 2.7487817806516186, while a fresh run of
    the SAME commit produced 2.7487817806516133. The difference is 5e-15
    and it changes no published claim -- every claim in the repository is
    quoted at four decimals -- but it made a byte comparison of the
    committed artefact fail, and "the published number can be reproduced"
    is the whole point of shipping the artefact.
    """
    v = float(value)
    if v == 0.0 or not np.isfinite(v):
        return v
    return float(f"%.{digits}g" % v)


def _published_significance(significance: dict) -> dict:
    """The significance block as the artefact stores it.

    `t_stat` is rounded to four decimals -- exactly what the promotion
    gate already publishes and what the research card prints -- so the
    file and the card cannot disagree about the same number.
    """
    out = dict(significance)
    for key in ("mean_diff", "se"):
        if key in out:
            out[key] = _sig(out[key])
    if "t_stat" in out:
        out["t_stat"] = (None if out["t_stat"] is None
                         else round(float(out["t_stat"]), 4))
    return out


def _sample_search_space(space: dict[str, tuple[float, float]], n: int,
                         rng: np.random.Generator) -> list[dict]:
    keys = list(space.keys())
    out = []
    for _ in range(n):
        row = {}
        for k in keys:
            lo, hi = float(space[k][0]), float(space[k][1])
            # bounds are written in the domain convention (e.g. dd_trigger
            # runs -0.02 -> -0.10, i.e. "less negative to more negative"), which
            # is not always low -> high. Sorting here keeps the spec readable
            # and the sampler total.
            row[k] = float(rng.uniform(min(lo, hi), max(lo, hi)))
        out.append(row)
    return out


def annualisation(universe: str, timeframe: str) -> int:
    """Bars per year for a universe, resolved by ASSET CLASS.

    'us_equities' trades 252 sessions; 'crypto' trades 365 calendar days;
    'mixed' holds equities so its headline figures stay on the equity basis and
    the crypto legs are reported separately. Getting this wrong understates
    crypto annualised vol by 20%.
    """
    asset_class = {
        "us_equities": "equity",
        "crypto": "crypto",
        "mixed": "mixed",
    }.get(universe, "equity")
    return bars_per_year(asset_class, timeframe)


def run_sweep(
    universe: str = "us_equities",
    timeframe: str = "swing",
    n_candidates: int = 5,
    trials_per_candidate: int = 12,
    seed: int = 11,
    period: str = "3y",
    interval: str | None = None,
    use_llm: bool = True,
    verbose: bool = True,
    offline: bool = False,
) -> SweepResult:
    """Full research loop. The output deliberately leads with n_trials.

    The winner's Sharpe on its own is not a result. The winner's DSR, given how
    many configurations were tried, is the result.
    """
    u = UNIVERSES[universe]
    tf = TIMEFRAMES[timeframe]
    interval = interval or tf.bar
    prices = load_universe(universe, period=period, interval=interval,
                            offline=offline)

    rng = np.random.default_rng(seed)
    if use_llm:
        specs, meta = generate_candidates(
            n=n_candidates,
            context={
                "universe": u.as_dict(),
                "timeframe": timeframe,
                "bars": len(prices),
                "date_range": [str(prices.index[0].date()), str(prices.index[-1].date())],
            },
        )
    else:
        from .strategies import template_candidates
        specs, meta = template_candidates(n_candidates), {"provider": "template"}
    if verbose:
        _diag(f"[sweep] {n_candidates} candidate specs via {meta.get('provider')} "
              f"(source={meta.get('source')})")
        if meta.get("error"):
            _diag(f"[sweep] provider degraded: {meta['error']}")

    results = []
    total_trials = 0
    all_oos = []

    # The annualisation factor for THIS universe, resolved before the
    # candidate loop. The selection Sharpe below must use it: the crypto
    # universe trades 365 days a year, and annualising its daily Sharpe
    # at the equity default of 252 understates it by sqrt(365/252) =
    # 1.2035, i.e. by 16.9%. A constant factor does not change WHICH
    # candidate wins, but it does change the number the repository
    # publishes -- and the same sweep file then reports two different
    # Sharpes for the same return series, which is worse than either.
    ppy = annualisation(universe, timeframe)

    for spec in specs:
        problems = validate_spec(spec)
        if problems:
            if verbose:
                _diag(f"[sweep] reject {spec.name}: {problems[0]}")
            continue
        variants = _sample_search_space(spec.search_space or {}, trials_per_candidate, rng)
        best = None
        for v in variants:
            trial_spec = HedgeStrategySpec(
                **{**spec.as_dict(), **v,
                   "name": f"{spec.name}", "search_space": spec.search_space}
            )
            sim = simulate_hedge(prices, u.hedge_proxy, trial_spec, timeframe,
                                 periods=ppy)
            if "error" in sim:
                continue
            sh = sharpe(sim["returns_hedged"], periods=ppy)
            total_trials += 1
            all_oos.append(sim["returns_hedged"].values)
            if best is None or sh > best[0]:
                best = (sh, trial_spec, sim)
        if best is None:
            continue
        sh, trial_spec, sim = best
        results.append((sh, trial_spec, sim))
        if verbose:
            _diag(f"[sweep]   {spec.name:<32} best sharpe {sh:6.3f}  "
                  f"avg_hedge {sim['avg_hedge']:.3f}  veto {sim['veto_ratio']:.2f}")

    if not results:
        raise RuntimeError(
            "sweep produced no results. Data too short for the requested "
            f"timeframe warmup ({tf.min_bars} bars). Try period='5y' or a lower "
            "timeframe."
        )

    results.sort(key=lambda t: -t[0])
    best_sh, best_spec, best_sim = results[0]

    # CSCV across the surviving candidates: PBO of the SELECTION, not of one run
    min_len = min(len(r) for r in all_oos)
    pbo_matrix = np.column_stack([r[:min_len] for r in all_oos])
    pbo = pbo_cscv(pbo_matrix, n_splits=8)

    # Deflate the winner. trials_variance is the across-trial Sharpe spread.
    win_sharpes = [r[0] for r in results]
    wf = evaluate_walk_forward(best_sim["returns_hedged"], n_folds=8,
                               periods=ppy, purge_bars=1)
    dsr = deflated_sharpe_ratio(
        best_sim["returns_hedged"],
        n_trials=max(total_trials, 1),
        periods=ppy,
    )
    reg_stats = regime_breakdown(best_sim["returns_hedged"], best_sim["regimes"],
                                periods=ppy)

    metrics = {
        "sharpe": round(best_sh, 4),
        "sharpe_unhedged": round(sharpe(best_sim["returns_unhedged"],
                                        periods=ppy), 4),
        "sortino": round(sortino(best_sim["returns_hedged"], periods=ppy), 4),
        "max_dd_hedged": round(max_drawdown(best_sim["returns_hedged"]), 4),
        "max_dd_unhedged": round(max_drawdown(best_sim["returns_unhedged"]), 4),
        "avg_hedge": round(best_sim["avg_hedge"], 4),
        "annualised_turnover": round(best_sim["turnover_annualised"], 3),
    }

    # --- the two checks the published literature makes mandatory -----------
    # Cederburg et al. (2020, JFE): a volatility-gated overlay must be shown to
    # beat the simple alternatives, not just buy-and-hold.
    # Harvey/Liu/Zhu (2016, RFS): report a HAC t-statistic and hold it to t>3.0.
    significance = sharpe_difference_tstat(
        best_sim["returns_hedged"], best_sim["returns_unhedged"], periods=ppy,
    )
    try:
        benchmarks = benchmark_table(
            best_sim["returns_unhedged"],
            best_sim["hedge_returns"],
            best_sim["returns_hedged"],
            name="regime_gated_overlay",
            periods=ppy,
        )
    except (KeyError, ValueError) as exc:
        benchmarks = None
        _diag(f"  ! benchmark table unavailable: {exc}")

    gate = gate_report(metrics, dsr, pbo, wf, reg_stats, best_sim["action_ratio"],
                       Config().gate, benchmarks=benchmarks,
                       significance=significance)

    return SweepResult(
        universe=universe, timeframe=timeframe, n_trials=total_trials,
        best_spec=best_spec.as_dict(), best_metrics=metrics, dsr=dsr, pbo=pbo,
        walk_forward=wf.as_dict(), regime_stats=reg_stats, gate=gate,
        action_ratio=round(best_sim["action_ratio"], 4),
        veto_ratio=round(best_sim["veto_ratio"], 4),
        zero_hedge_ratio=round(best_sim["zero_hedge_ratio"], 4),
        top_candidates=[
            {"name": s.name, "sharpe": round(sh, 4), "avg_hedge": round(sim["avg_hedge"], 4)}
            for sh, s, sim in results[:5]
        ],
        hedge=best_sim["hedge"],
        benchmarks=benchmarks or {},
        significance=_published_significance(significance),
        window=_window_of(prices, period),
    )


# -------------------------------------------------------------- live agent
@dataclass
class StepLog:
    date: str
    regime: str
    consensus: float
    vol_ratio: float
    drawdown: float
    quant_hedge: float
    decision_hedge: float
    vetoed: bool
    escal: bool
    veto_reason: str | None
    backend: str
    model: str
    cost_usd: float


def run_agent(
    spec: HedgeStrategySpec,
    universe: str = "us_equities",
    timeframe: str = "swing",
    period: str = "3y",
    interval: str | None = None,
    use_decision_layer: bool = True,
    decision_every: int = 1,
    verbose: bool = True,
    offline: bool = False,
) -> dict:
    """Forward run of a gate-passed spec with the decision layer in the loop.

    decision_every > 1 is a real cost control: the quantitative hedge is
    recomputed every bar, and the (paid) decision call is made only every N
    bars. Between calls the policy holds the last decision. This is the honest
    version of "the agent thinks continuously".
    """
    u = UNIVERSES[universe]
    tf = TIMEFRAMES[timeframe]
    interval = interval or tf.bar
    prices = load_universe(universe, period=period, interval=interval,
                            offline=offline)

    problems = validate_spec(spec)
    if problems:
        return {"error": f"spec rejected by validator: {problems}"}

    cfg = _hedge_cfg_from_spec(spec)
    backend = build_backend(Config().decision) if use_decision_layer else None
    questions = hedge_questions()

    regime_df = rolling_regime(prices, tf, target_vol=cfg.target_vol)
    port_rets = prices.pct_change().mean(axis=1)
    curve = (1.0 + port_rets.fillna(0.0)).cumprod()
    drawdown = curve / curve.cummax() - 1.0

    logs: list[StepLog] = []
    prev_hedge = 0.0
    last_decision = None
    total_cost = 0.0
    n_calls = 0

    for i, (dt, row) in enumerate(regime_df.iterrows()):
        if row["regime"] is None or pd.isna(row.get("score")):
            continue
        state = _state_from_row(row, prices.loc[:dt], tf, cfg.target_vol)
        if state is None:
            continue
        dd = float(drawdown.loc[dt]) if dt in drawdown.index else 0.0

        decisions = None
        if backend is not None:
            if last_decision is None or (i % decision_every == 0):
                # state is trimmed to what the decision needs: you pay per token
                state_payload = {
                    "regime": state.label.value,
                    "regime_score_0_4": round(state.score, 3),
                    "scale_consensus": round(state.consensus, 3),
                    "vol_ratio_to_target": round(state.vol_ratio, 3),
                    "trend_sign": state.trend_sign,
                    "portfolio_drawdown": round(dd, 4),
                    "current_hedge_notional": round(prev_hedge, 4),
                    "timeframe": timeframe,
                }
                try:
                    last_decision = backend.decide(state_payload, questions)
                    n_calls += 1
                    total_cost += sum(d.cost_usd for d in last_decision.values())
                except Exception as e:  # a decision failure must not become a blind trade
                    if verbose:
                        _diag(f"[agent] decision call failed at {dt.date()}: "
                              f"{type(e).__name__}: {e}. Holding previous hedge.")
                    last_decision = None
            decisions = last_decision

        act = decide_hedge(state, dd, prev_hedge, cfg, decisions)
        logs.append(StepLog(
            date=str(dt.date()),
            regime=act.regime,
            consensus=round(state.consensus, 3),
            vol_ratio=act.vol_ratio,
            drawdown=act.drawdown,
            quant_hedge=round(
                min(cfg.max_hedge_fraction,
                    act.multiplier["vol"] * act.multiplier["drawdown"]
                    * act.multiplier["consensus"]), 4),
            decision_hedge=act.target_hedge,
            vetoed=act.vetoed,
            escal=act.escal,
            veto_reason=act.veto_reason,
            backend=(next(iter(decisions.values())).backend if decisions else "none"),
            model=(next(iter(decisions.values())).model if decisions else ""),
            cost_usd=round(sum(d.cost_usd for d in decisions.values()), 8) if decisions else 0.0,
        ))
        prev_hedge = act.target_hedge

    log_df = pd.DataFrame([l.__dict__ for l in logs])
    return {
        "universe": universe,
        "timeframe": timeframe,
        "spec": spec.as_dict(),
        "log": log_df,
        "n_bars": len(log_df),
        "decision_calls": n_calls,
        "decision_cost_usd": round(total_cost, 6),
        "backend": getattr(backend, "name", "none"),
        "veto_ratio": round(float(log_df["vetoed"].mean()), 4) if len(log_df) else 0.0,
        "zero_hedge_ratio": round(float((log_df["decision_hedge"] == 0).mean()), 4) if len(log_df) else 0.0,
        "escalation_ratio": round(float(log_df["escal"].mean()), 4) if len(log_df) else 0.0,
        "avg_hedge": round(float(log_df["decision_hedge"].mean()), 4) if len(log_df) else 0.0,
    }


# ------------------------------------------------------------------- report
def _diag(message: str = "") -> None:
    """Print a progress or diagnostic message to stderr.

    Diagnostics go to stderr so that stdout carries ONLY the result: the
    JSON payload in --json mode, and nothing else. A program that wants to
    consume this tool should not have to strip banners off stdout first.

    This was a real bug: in --json mode the sweep's progress lines went to
    stdout, so `json.loads(stdout)` raised "Expecting value: line 1
    column 2". The flag existed and did not work.
    """
    print(message, file=sys.stderr)


def print_report(res: SweepResult | dict, stream=None) -> None:
    """Print the human-readable research report.

    ``stream`` defaults to stdout, which is right when a human is reading
    the terminal. In --json mode the caller passes stderr, so stdout stays
    pure JSON.
    """
    out = stream if stream is not None else sys.stdout
    d = res.as_dict() if isinstance(res, SweepResult) else res
    print("\n" + "=" * 74, file=out)
    print("HEDGE AGENT - RESEARCH REPORT", file=out)
    print("=" * 74, file=out)
    print(f"universe           {d['universe']}   timeframe {d['timeframe']}", file=out)
    h = d.get("hedge", {})
    if h:
        print(f"hedge instrument   {h.get('hedge_proxy')}  side {h.get('hedge_side')} "
              f"({h.get('hedge_side_label')})", file=out)
        print(f"hedge corr         {h.get('hedge_corr_to_portfolio')}  to the portfolio", file=out)
        if h.get("correlation_warning"):
            print(f"  !! {h['correlation_warning']}", file=out)
    print(f"trials run         {d['n_trials']}   <-- every number below is deflated by this", file=out)
    print(f"winner             {d['best_spec']['name']}", file=out)
    m = d["best_metrics"]
    print(f"sharpe hedged      {m['sharpe']:>8}   unhedged {m['sharpe_unhedged']:>8}", file=out)
    print(f"max DD hedged      {m['max_dd_hedged']:>8}   unhedged {m['max_dd_unhedged']:>8}", file=out)
    print(f"avg hedge notional {m['avg_hedge']:>8}   turnover {m['annualised_turnover']}x", file=out)
    print(f"zero-hedge share   {d['zero_hedge_ratio']:>8}   veto share {d['veto_ratio']:>8}", file=out)
    print("-" * 74, file=out)
    print("MULTIPLE-TESTING CORRECTION", file=out)
    print(f"  observed Sharpe  {d['dsr']['observed_sharpe']}", file=out)
    print(f"  expected max     {d['dsr']['expected_max_sharpe']}  (the null benchmark at n_trials={d['n_trials']})", file=out)
    print(f"  deflated Sharpe  {d['dsr']['dsr']}   <- P(real edge)", file=out)
    print(f"  PBO              {d['pbo']['pbo']}  over {d['pbo'].get('n_combinations')} splits "
          f"x {d['pbo'].get('n_configs')} configs", file=out)
    print(f"  IS->OOS decay    {d['walk_forward']['max_is_oos_decay']}", file=out)
    print(f"  positive folds   {d['walk_forward']['positive_fold_ratio']} over "
          f"{d['walk_forward']['n_folds']} folds", file=out)
    print("-" * 74, file=out)
    print("REGIME BREAKDOWN (a single number across regimes is not a result)", file=out)
    for k, v in d["regime_stats"].items():
        if v.get("significant"):
            print(f"  {k:<14} sharpe {v['sharpe']:>7}  ret {v['total_return']:>9}  "
                  f"maxDD {v['max_dd']:>7}  n {v['n_bars']}", file=out)
        else:
            # say WHY there is no number. Printing a bare "n 2" invites the
            # reader to compute the Sharpe themselves, which is how a
            # two-bar regime was read as scoring 417.
            print(f"  {k:<14} sharpe    n/a   ret {v['total_return']:>9}  "
                  f"maxDD {v['max_dd']:>7}  n {v['n_bars']}"
                  f"  (too few bars to estimate: SE > 2.0)", file=out)
    print("-" * 74, file=out)
    print("PROMOTION GATE", file=out)
    for c in d["gate"]["checks"]:
        mark = "PASS" if c["pass"] else "FAIL"
        print(f"  [{mark}] {c['check']:<26} {str(c['value']):>10}  (need {c['threshold']})", file=out)
    print(f"\nVERDICT: {d['gate']['verdict']}", file=out)
    if d["gate"]["failed_checks"]:
        print(f"failed: {', '.join(d['gate']['failed_checks'])}", file=out)
    print("=" * 74, file=out)