#!/usr/bin/env python3
"""Factor scorer: hand weights against weights fitted walk-forward, with an untouched final holdout
(PROJECT_PLAN Q.14 items 2 and 3, 2026-11-02).

The live daily signal (research/signals/score_daily_signals.py) ranks the universe by a weighted sum of
rank-normalised features, with a hand-set weight dict per market regime. backtest_portfolio.py, which
gave the reference 0.58 net Sharpe / -39.6% drawdown (20d, top 5), never switched regime: it loads the
weight features but not the macro columns, so classify_regime() always answered NEUTRAL.

Variants, each scored on the same rebalance dates (every 20 sessions, top 5, the backtest's costs):
  hand_neutral   the reference backtest as it ran (NEUTRAL weights every day)
  hand_regime    the live scorer (regime weights, regime from VIX percentile and risk-on)
  fit_pooled     ridge of the next-20-session return rank on the same rank-normalised features,
                 refitted at each rebalance on every session whose label was already known
  fit_regime     the same, fitted on past sessions of the current regime (pooled if < 120 sessions)
  universe       every liquid stock, equal weight: the random-pick control. The universe is today's
                 102 large caps; their own outperformance of SPY is survivorship, not skill

Periods: selection on 2019-01 .. 2025-10; the final holdout 2025-11 .. 2026-10 is reported once and not
used to choose anything. The deflated Sharpe ratio counts the four scored variants as trials.

Also: within the user's holdings (INTC, TSM, STX, GOOGL, SKHY), does the higher-scored one beat the
lower-scored one over the next 20 sessions (which to trim first)?

Finally the pooled fit on every labelled session is written to research/signals/fitted_weights.json,
which the live scorer reads. A feature is weighted only when filled on >= 30% of the training rows and
on half its sessions (usable()); otherwise it is 0.

Result 2026-11-02 (selection 2019-2025): hand IC -0.0005 / -0.007; fitted IC 0.049 (t 2.62), +7.8%/yr
over the universe (t 1.52); holdout 12 rebalances, no variant beat the universe.

Usage: .venv311/bin/python research/backtest/factor_lab.py [refresh]
"""

from __future__ import annotations

import json
import math
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from research.backtest.backtest_portfolio import _round_trip_cost_bps, MIN_DOLLAR_VOL  # noqa: E402
from research.signals.score_daily_signals import _WEIGHTS_BY_REGIME, classify_regime  # noqa: E402
from research.lab.evaluate import deflated_sharpe  # noqa: E402

OUT = Path(os.getenv("FACTOR_LAB_DIR", "/Volumes/Data4T/factor_lab"))
HOLD = 20
TOP_N = 5
EVAL_START = "2019-01-01"
HOLDOUT_START = "2025-11-01"
MIN_REGIME_SESSIONS = 120
RIDGE_ALPHA = 10.0
RF_ANNUAL = 0.04
HOLDINGS = ("INTC", "TSM", "STX", "GOOGL", "SKHY")
FEATS = list(_WEIGHTS_BY_REGIME["NEUTRAL"])
FUT, BENCH = f"future_ret_{HOLD}d", f"spy_ret_{HOLD}d"
LIVE_WEIGHTS = ROOT / "research" / "signals" / "fitted_weights.json"     # read by score_daily_signals


def refresh() -> None:
    from dotenv import load_dotenv
    from pymongo import MongoClient
    load_dotenv(ROOT / ".env")
    uri = (os.getenv("HOST_LOCAL_MONGO_URI") or os.getenv("LOCAL_MONGO_URI") or "mongodb://root:root@127.0.0.1:37018/")
    col = MongoClient(uri.replace("host.docker.internal", "127.0.0.1"))["quant_data"]["daily_symbol_features"]
    df = pd.DataFrame(list(col.find({}, {"_id": 0})))
    df["trade_date"] = df["trade_date"].astype(str).str[:10]
    for c in df.columns:
        if df[c].dtype == object and c not in ("trade_date", "symbol"):
            df[c] = pd.to_numeric(df[c], errors="coerce")
    OUT.mkdir(parents=True, exist_ok=True)
    df.to_parquet(OUT / "features.parquet")


def load() -> pd.DataFrame:
    df = pd.read_parquet(OUT / "features.parquet")
    df = df[df.trade_date.str.match(r"^\d{4}-\d{2}-\d{2}$", na=False) & ~df.symbol.isin(["SPY", "QQQ"])].copy()
    df["dollar_vol_20d"] = df["volume_20d_avg"] * df["close"]
    df = df[df["dollar_vol_20d"].fillna(0) >= MIN_DOLLAR_VOL]
    # the scorer's transform: missing -> 0, then the cross-sectional rank in [-0.5, 0.5]
    for f in FEATS:
        df[f"r_{f}"] = pd.to_numeric(df.get(f), errors="coerce").fillna(0.0).groupby(df.trade_date).rank(pct=True) - 0.5
    df["y"] = df.groupby("trade_date")[FUT].rank(pct=True) - 0.5
    return df.sort_values(["trade_date", "symbol"]).reset_index(drop=True)


def score(g: pd.DataFrame, w: dict) -> pd.Series:
    return sum(w.get(f, 0.0) * g[f"r_{f}"] for f in FEATS)


MIN_COVERAGE = 0.30        # a feature is fitted only with this share of rows filled ...
MIN_HISTORY_SHARE = 0.50    # ... on at least half the training sessions; otherwise its weight is 0


def usable(rows: pd.DataFrame) -> list:
    """Features with enough history to be weighted. retail_sent_score (from 2025-12, 2% of rows in
    2026), analyst_buy_ratio (2026-03) and ah_gap (2026-05) are not: fitted on a few months they
    took the largest weights."""
    out = []
    for f in FEATS:
        filled = rows[f].notna() if f in rows else pd.Series(False, index=rows.index)
        if filled.mean() >= MIN_COVERAGE and rows.loc[filled, "trade_date"].nunique() >= MIN_HISTORY_SHARE * rows.trade_date.nunique():
            out.append(f)
    return out


def fit(rows: pd.DataFrame) -> dict:
    from sklearn.linear_model import Ridge
    rows = rows.dropna(subset=["y"])
    feats = usable(rows)
    if not feats:
        return {f: 0.0 for f in FEATS}
    m = Ridge(alpha=RIDGE_ALPHA).fit(rows[[f"r_{f}" for f in feats]], rows["y"])
    return {f: float(dict(zip(feats, m.coef_)).get(f, 0.0)) for f in FEATS}


def walk(df: pd.DataFrame):
    dates = sorted(df.trade_date.unique())
    regimes = {d: classify_regime(g) for d, g in df.groupby("trade_date")}
    reg_series = pd.Series(regimes)
    by_date = {d: g for d, g in df.groupby("trade_date")}
    start = next(i for i, d in enumerate(dates) if d >= EVAL_START)
    rows, holds, weights = [], [], []
    for i in range(start, len(dates), HOLD):
        d = dates[i]
        g = by_date[d].dropna(subset=[FUT])
        if len(g) < 10:
            continue
        known = df[df.trade_date <= dates[i - HOLD]]                  # labels ending by today
        reg = regimes[d]
        same = known[known.trade_date.map(reg_series) == reg]
        w = {"hand_neutral": _WEIGHTS_BY_REGIME["NEUTRAL"], "hand_regime": _WEIGHTS_BY_REGIME[reg],
             "fit_pooled": fit(known),
             "fit_regime": fit(same) if same.trade_date.nunique() >= MIN_REGIME_SESSIONS else None}
        w["fit_regime"] = w["fit_regime"] or w["fit_pooled"]
        weights.append({"date": d, "regime": reg, **{f"{k}:{f}": v for k in ("fit_pooled", "fit_regime") for f, v in w[k].items()}})
        bench = float(g[BENCH].dropna().iloc[0]) if g[BENCH].notna().any() else 0.0
        row = {"date": d, "regime": reg, "spy": bench, "universe": float(g[FUT].mean())}
        for name, wt in w.items():
            s = score(g, wt)
            top = g.loc[s.nlargest(TOP_N).index]
            cost = float(top["dollar_vol_20d"].apply(_round_trip_cost_bps).mean())
            row[name] = float(top[FUT].mean()) - cost
            row[f"ic_{name}"] = float(s.corr(g[FUT], method="spearman"))
            h = g[g.symbol.isin(HOLDINGS)]
            if len(h) >= 3:
                hs = score(h, wt)
                holds.append({"date": d, "variant": name, "n": len(h),
                              "spread": float(h.loc[hs.idxmax(), FUT] - h.loc[hs.idxmin(), FUT]),
                              "top": h.loc[hs.idxmax(), "symbol"], "bottom": h.loc[hs.idxmin(), "symbol"]})
        rows.append(row)
    return pd.DataFrame(rows), pd.DataFrame(holds), pd.DataFrame(weights)


def stats(r: pd.Series) -> dict:
    r = r.dropna()
    if len(r) < 3:
        return {"periods": int(len(r))}
    ppy = 252 / HOLD
    eq = (1 + r).cumprod()
    vol = r.std(ddof=1) * math.sqrt(ppy)
    return {"periods": int(len(r)), "ann_return": round(float(eq.iloc[-1] ** (ppy / len(r)) - 1), 4),
            "sharpe": round(float((r.mean() * ppy - RF_ANNUAL) / vol), 3) if vol > 0 else None,
            "max_drawdown": round(float((eq / eq.cummax() - 1).min()), 4), "win_rate": round(float((r > 0).mean()), 3),
            "mean_period": round(float(r.mean()), 4), "t_mean": round(float(r.mean() / (r.std(ddof=1) / math.sqrt(len(r)))), 2)}


def ic_stats(x: pd.Series) -> dict:
    x = x.dropna()
    return {"mean_ic": round(float(x.mean()), 4), "t": round(float(x.mean() / (x.std(ddof=1) / math.sqrt(len(x)))), 2), "n": int(len(x))}


def main() -> None:
    if len(sys.argv) > 1 and sys.argv[1] == "refresh":
        refresh()
    df = load()
    res, holds, weights = walk(df)
    res.to_parquet(OUT / "walk.parquet")
    holds.to_parquet(OUT / "holdings_spread.parquet")
    weights.to_parquet(OUT / "fitted_weights.parquet")
    variants = ["hand_neutral", "hand_regime", "fit_pooled", "fit_regime"]
    sel, hold = res[res.date < HOLDOUT_START], res[res.date >= HOLDOUT_START]
    rep = {"rebalances": {"selection": len(sel), "holdout": len(hold)}, "regimes": res.regime.value_counts().to_dict()}
    for name, part in (("selection_2019_2025", sel), ("holdout_2025_11_on", hold)):
        rep[name] = {v: {**stats(part[v]), **{"ic": ic_stats(part[f"ic_{v}"])}} for v in variants}
        rep[name]["universe_equal_weight"] = stats(part["universe"])
        rep[name]["spy"] = stats(part["spy"])
        rep[name]["excess_over_universe"] = {v: stats(part[v] - part["universe"]) for v in variants}
    trial = [float(sel[v].mean() / sel[v].std(ddof=1)) for v in variants]
    best = variants[int(np.argmax(trial))]
    rep["dsr_selection"] = {"best": best, **deflated_sharpe(sel[best].values, trial),
                            "excess_over_universe": deflated_sharpe((sel[best] - sel["universe"]).values,
                                                                    [float((sel[v] - sel.universe).mean() / (sel[v] - sel.universe).std(ddof=1)) for v in variants])}
    hs = holds[holds.date < HOLDOUT_START]
    rep["holdings_top_minus_bottom_20d"] = {v: {"mean": round(float(g.spread.mean()), 4), "t": round(float(g.spread.mean() / (g.spread.std(ddof=1) / math.sqrt(len(g)))), 2),
                                                "n": int(len(g)), "hit": round(float((g.spread > 0).mean()), 3)}
                                            for v, g in hs.groupby("variant")}
    last = weights.iloc[-1]
    rep["latest_fitted_weights"] = {"date": last["date"], "regime": last["regime"],
                                    "fit_regime": {f: round(float(last[f"fit_regime:{f}"]), 4) for f in FEATS},
                                    "hand": _WEIGHTS_BY_REGIME[last["regime"]]}
    (OUT / "report.json").write_text(json.dumps(rep, indent=1, default=str))
    # The live weights: the pooled fit (best in selection) on every session whose label is known now.
    dates = sorted(df.trade_date.unique())
    live = fit(df[df.trade_date <= dates[-1 - HOLD]])
    LIVE_WEIGHTS.write_text(json.dumps({
        "fitted_through": dates[-1 - HOLD], "method": f"ridge(alpha={RIDGE_ALPHA}) of the {HOLD}-session return rank on rank-normalised features, pooled",
        "evidence": {"selection_ic": rep["selection_2019_2025"]["fit_pooled"]["ic"],
                     "hand_regime_ic": rep["selection_2019_2025"]["hand_regime"]["ic"],
                     "excess_over_universe": rep["selection_2019_2025"]["excess_over_universe"]["fit_pooled"]},
        "weights": {f: round(float(v), 6) for f, v in live.items()}}, indent=1, default=str) + "\n")
    print(json.dumps(rep, indent=1, default=str))


if __name__ == "__main__":
    main()
