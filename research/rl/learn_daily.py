#!/usr/bin/env python3
"""The daily policy for the holdings advisor (PROJECT_PLAN Q.7 phase 2, 2026-10-26).

    python research/rl/learn_daily.py walkforward   # ensemble Q (mean, spread) and P10, by year
    python research/rl/learn_daily.py judge         # policy variants, MLflow experiment action_rl_daily

As learn.py, on the daily decision points (build_daily.py): Q(state, action) by regression on the
exact outcomes, an ensemble for the mean and its spread, a quantile head for P10. The policy at a
decision: within each action family (side, exit kind and distances, horizon) the entry levels
whose conservative score (mean - lcb x spread) clears the margin and whose P10 clears the floor,
at most four, weighted by score over downside; the best family is the plan. One plan per stock at
a time: a plan with a 10-day horizon blocks that stock's next decisions for ten trading days.
Variants are chosen on 2019-2022 among those with at least MIN_SELECT trades there, P10 above
P10_FLOOR and CVaR5 above CVAR5_FLOOR, by per-trade Sharpe, and reported on 2023-2026.
"""
from __future__ import annotations

import itertools
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "models"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "lab"))
import earnings_pulse as ep  # noqa: E402

D = ep.D / "rl"
ACT = ["side_code", "entry_k", "exit_code", "y", "z", "s", "horizon_code"]
HZ_DAYS = {0: 1, 1: 5, 2: 10, 3: 20}
MIN_SELECT = 50
# Multi-day holds have fat tails (MRNA +177% in one day on 2026-08-19): one worst trade is no test of
# a policy, the mean of its worst 5% (CVaR) is.
CVAR5_FLOOR = -8.0
P10_FLOOR = -2.0


def passes(sel):
    return sel.get("acted", 0) >= MIN_SELECT and sel["p10"] > P10_FLOOR and sel["cvar5"] > CVAR5_FLOOR


def frame():
    o = pd.read_parquet(D / "daily_outcomes.parquet")
    o = o[o["gain"].notna()]
    acts = pd.read_parquet(D / "daily_actions.parquet")
    dec = pd.read_parquet(D / "daily_decisions.parquet")
    dec["sector_code"] = dec["sector"].astype("category").cat.codes.astype(float)
    state = [c for c in dec.columns if c not in ("eid", "symbol", "date", "sector", "p0", "close") and pd.api.types.is_numeric_dtype(dec[c])]
    df = o.merge(acts[["action_id"] + ACT], on="action_id").merge(dec[["eid", "symbol", "date"] + state], on="eid")
    df["year"] = df["date"].str[:4].astype(int)
    df = df.sort_values(["date", "eid", "action_id"]).reset_index(drop=True)
    # the stock's own earlier outcomes on this action and on all actions (one decision per stock and date)
    g = df.groupby(["symbol", "action_id"], sort=False)["gain"]
    df["own_act"] = (g.cumsum() - df["gain"]) / g.cumcount().replace(0, np.nan)
    per = df.groupby(["symbol", "eid"], sort=False)["gain"].agg(["sum", "count"]).reset_index()
    per["date"] = per["eid"].str.split("|").str[1]
    per = per.sort_values("date")
    gs = per.groupby("symbol")
    per["own_sym"] = (gs["sum"].cumsum() - per["sum"]) / (gs["count"].cumsum() - per["count"]).replace(0, np.nan)
    df = df.merge(per[["eid", "own_sym"]], on="eid", how="left")
    feats = state + ACT + ["own_act", "own_sym"]
    for c in feats:
        df[c] = df[c].astype(np.float32)
    return df, feats


def walkforward():
    import lightgbm as lgb
    df, feats = frame()
    P = dict(objective="huber", alpha=2.0, learning_rate=0.05, num_leaves=63, min_data_in_leaf=2000, feature_fraction=0.7,
             bagging_fraction=0.5, bagging_freq=1, lambda_l2=10.0, verbose=-1, num_threads=8)
    out = D / "daily_preds"
    out.mkdir(exist_ok=True)
    years = [y for y in sorted(df["year"].unique()) if y >= 2019]
    for n, y in enumerate(years, 1):
        if (out / f"{y}.parquet").exists():
            continue
        tr, te = df[df["year"] < y], df[df["year"] == y].copy()
        X, yv, Xt = tr[feats], tr["gain"], te[feats]
        M = np.vstack([lgb.train({**P, "seed": s}, lgb.Dataset(X, yv), 300).predict(Xt) for s in (11, 23, 37)])
        te["pred"], te["pred_std"] = M.mean(0), M.std(0)
        te["q10"] = lgb.train({**P, "seed": 11, "objective": "quantile", "alpha": 0.1}, lgb.Dataset(X, yv), 300).predict(Xt)
        te[["eid", "symbol", "date", "year", "action_id", "side_code", "horizon_code", "gain", "pred", "pred_std", "q10"]].to_parquet(out / f"{y}.parquet")
        ep.progress("rl-daily-wf", n, len(years), f"{y} train={len(tr):,}")
    print("DONE daily walk-forward", flush=True)


def plans(preds, acts, margin, q10_min, cash, lcb):
    """The best plan per decision (or none), before the one-plan-per-stock rule."""
    k = preds if cash else preds[preds["side_code"] == 0]
    k = k.assign(score=k["pred"] - lcb * k["pred_std"])
    if q10_min is not None:
        k = k[k["q10"] >= q10_min]
    k = k[k["score"] >= margin]
    if not len(k):
        return k.assign(family=[], legs=[], realized=[], hz=[])
    fam = ["side", "exit", "y", "z", "s", "horizon"]
    k = k.join(acts.set_index("action_id")[fam], on="action_id")
    k["family"] = k[fam].astype(str).agg("|".join, axis=1)
    k["rk"] = k.groupby(["eid", "family"])["score"].rank(ascending=False, method="first")
    k = k[k["rk"] <= 4]
    k["w"] = k["score"] / np.maximum(k["pred"] - k["q10"], 0.5)
    k["ws"], k["wg"] = k["w"] * k["score"], k["w"] * k["gain"]
    g = k.groupby(["eid", "family"]).agg(w=("w", "sum"), ws=("ws", "sum"), wg=("wg", "sum"), legs=("action_id", "count"),
                                         hz=("horizon_code", "first")).reset_index()
    g["value"], g["realized"] = g["ws"] / g["w"], g["wg"] / g["w"]
    return g.loc[g.groupby("eid")["value"].idxmax()]


def policy(preds, acts, margin, q10_min, cash, lcb):
    best = plans(preds, acts, margin, q10_min, cash, lcb).set_index("eid")
    dec = preds[["eid", "symbol", "date"]].drop_duplicates("eid").sort_values(["symbol", "date"])
    out = []
    for sym, g in dec.groupby("symbol", sort=False):
        days = pd.to_datetime(g["date"]).values
        blocked_until = np.datetime64("1900-01-01")
        for eid, day in zip(g["eid"].values, days):
            if day < blocked_until or eid not in best.index:
                out.append((eid, False, 0.0, None))
                continue
            b = best.loc[eid]
            out.append((eid, True, float(b["realized"]), b["family"]))
            blocked_until = day + np.timedelta64(int(HZ_DAYS[int(b["hz"])] * 7 / 5) + 1, "D")   # trading days to calendar days
    r = pd.DataFrame(out, columns=["eid", "acted", "gain", "family"])
    r["year"] = r["eid"].str.split("|").str[1].str[:4].astype(int)
    return r


def judge():
    import evaluate as ev
    os.environ.setdefault("LAB_EXPERIMENT", "action_rl_daily")
    import tracking as tr
    import mlflow
    preds = pd.concat(pd.read_parquet(p) for p in sorted((D / "daily_preds").glob("*.parquet")))
    acts = pd.read_parquet(D / "daily_actions.parquet")
    pid = tr.parent("lgbm3_daily", {"actions": len(acts), "rows": len(preds)}, tags={"kind": "model", "model": "lgbm3_daily"})
    best, n = {}, 0
    grid = list(itertools.product((0.3, 0.5, 0.7, 1.0), (None, -2.0, -1.5, -1.0), (False, True), (0.0, 1.0, 2.0)))
    for margin, q, cash, lcb in grid:
        r = policy(preds, acts, margin, q, cash, lcb)
        vd = ev.verdict(r, {})
        sel = vd["select"]
        ok = passes(sel)
        v = {"margin": margin, "q10_min": q, "cash": cash, "lcb": lcb}
        tr.log_run(f"lgbm3_daily {v}", {**v, "model": "lgbm3_daily"}, vd, tags={"kind": "variant", "model": "lgbm3_daily",
                   "passes_risk": str(ok)}, parent_id=pid)
        key = "cash" if cash else "no_cash"
        if ok and (key not in best or sel["sharpe_trade"] > best[key][1]["select"]["sharpe_trade"]):
            best[key] = (v, vd, r)
        n += 1
        ep.progress("rl-daily-judge", n, len(grid))
    runs = mlflow.search_runs(experiment_names=[tr.EXPERIMENT], filter_string="tags.kind = 'variant'")
    trials = runs.get("metrics.all_sharpe_trade", pd.Series(dtype=float)).dropna().tolist()
    for key, (v, vd, r) in best.items():
        dsr = ev.deflated_sharpe(r.loc[r["acted"], "gain"].values, trials)
        a = r[r["acted"]]
        print(f"\n{key}: chosen {v} | 2019-2022 {vd['select']['acted']} acted {vd['select']['mean']:+.2f}% SR {vd['select']['sharpe_trade']:.2f} | "
              f"2023-2026 {vd['report']['acted']} acted {vd['report']['mean']:+.2f}% (se {vd['report']['se']:.2f}) win {vd['report']['win']:.0%} "
              f"p10 {vd['report']['p10']:+.2f} CVaR5 {vd['report']['cvar5']:+.1f} worst {vd['report']['worst']:+.1f} sum {vd['report']['sum']:+.0f}% | DSR {dsr['dsr']:.3f} "
              f"({dsr['trials']} trials) | years " + " ".join(f"{y}:{x:+.2f}" for y, x in vd["years"].items()))
        print("   families:", a["family"].str.replace("|", " ").value_counts().head(6).to_dict())
        for s_ in ("INTC", "STX", "TSM", "SKHY", "MU", "GOOGL"):
            g = r[r["eid"].str.startswith(s_ + "|")]; ga = g[g["acted"]]
            print(f"   {s_}: acted {len(ga)}/{len(g)} mean {ga['gain'].mean() if len(ga) else 0:+.2f}%")
    if not best:
        print("no variant passes the 2019-2022 limits")


if __name__ == "__main__":
    {"walkforward": walkforward, "judge": judge}[sys.argv[1]]()
