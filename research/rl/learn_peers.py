#!/usr/bin/env python3
"""The peer spillover policy (PROJECT_PLAN Q.7 phase 4, 2026-10-27).

    python research/rl/learn_peers.py walkforward   # ensemble Q (mean, spread) and P10, by year -> EP_DIR/rl/peer_preds/
    python research/rl/learn_peers.py judge         # policy variants, MLflow experiment action_rl_peers

As learn.py, on the peer decision points (build_peers.py): Q(state, action) by regression on the
exact outcomes, CatBoost and three LightGBMs for the mean and their spread, quantile heads for P10.
State: the reporter's and the peer's moves at the decision, the pair (correlation, score, link,
sector), the peer's volatility and run-up, the reporter's usual report move; plus the pair's own
earlier outcomes on this action and on all actions (earlier reports only). The policy at each
(reporter report, peer): in time order (r+5, r+20, o+5) the first decision point with a plan that
clears the margin (learn.compose_policy). Variants are chosen on 2019-2022 by per-trade Sharpe within
the risk limits and reported on 2023-2026, with the deflated Sharpe over every variant logged.
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
from rl import learn  # noqa: E402

D = ep.D / "rl"
ORDER = ("r+5", "r+20", "o+5")
STATE = ["bmo", "rep_move", "rep_max", "rep_min", "peer_move", "peer_max", "peer_min", "rel_move", "peer_run5", "peer_run20",
         "sigma", "rep_vol20", "rep_past_move_med", "corr", "score", "link", "same_sector", "minutes_after_release", "place_code"]
ACT = learn.ACT
FEATS = STATE + ACT + ["own_act", "own_sym"]
MIN_SELECT = 50
CVAR5_FLOOR = -8.0
P10_FLOOR = -2.0


def frame():
    o = pd.read_parquet(D / "peer_outcomes.parquet")
    o = o[o["gain"].notna()]
    o["eid"], o["place"] = o["eid"].astype(str), o["place"].astype(str)
    acts = pd.read_parquet(D / "actions.parquet")
    dec = pd.read_parquet(D / "peer_decisions.parquet")
    dec["place_code"] = dec["place"].map({p: i for i, p in enumerate(ORDER)}).astype(float)
    df = o.merge(acts[["action_id"] + ACT], on="action_id").merge(dec[["eid", "place", "symbol", "reporter", "date"] + [c for c in STATE if c != "place_code"]
                                                                       + ["place_code"]], on=["eid", "place"])
    df["year"] = df["date"].str[:4].astype(int)
    df["pair"] = df["symbol"] + "|" + df["reporter"]
    df = df.sort_values(["date", "eid", "place"]).reset_index(drop=True)
    # the pair's earlier outcomes on this action and on all actions (earlier reports only)
    first = df.groupby(["pair", "action_id", "eid"], sort=False)["gain"].mean().rename("g").reset_index()
    first["date"] = first["eid"].str.split("|").str[1]
    first = first.sort_values("date")
    g = first.groupby(["pair", "action_id"], sort=False)["g"]
    first["own_act"] = (g.cumsum() - first["g"]) / g.cumcount().replace(0, np.nan)
    df = df.merge(first[["pair", "action_id", "eid", "own_act"]], on=["pair", "action_id", "eid"], how="left")
    per = df.groupby(["pair", "eid"], sort=False)["gain"].agg(["sum", "count"]).reset_index()
    per["date"] = per["eid"].str.split("|").str[1]
    per = per.sort_values("date")
    gs = per.groupby("pair")
    per["own_sym"] = (gs["sum"].cumsum() - per["sum"]) / (gs["count"].cumsum() - per["count"]).replace(0, np.nan)
    df = df.merge(per[["eid", "own_sym"]], on="eid", how="left")
    for c in FEATS:
        df[c] = df[c].astype(np.float32)
    return df


def walkforward():
    import lightgbm as lgb
    from catboost import CatBoostRegressor
    df = frame()
    P = dict(objective="huber", alpha=2.0, learning_rate=0.05, num_leaves=63, min_data_in_leaf=2000, feature_fraction=0.7,
             bagging_fraction=0.5, bagging_freq=1, lambda_l2=10.0, verbose=-1, num_threads=8)
    out = D / "peer_preds"
    out.mkdir(exist_ok=True)
    years = [y for y in sorted(df["year"].unique()) if y >= 2019]
    for n, y in enumerate(years, 1):
        if (out / f"{y}.parquet").exists():
            continue
        tr, te = df[df["year"] < y], df[df["year"] == y].copy()
        X, yv, Xt = tr[FEATS], tr["gain"], te[FEATS]
        kw = dict(iterations=300, depth=6, learning_rate=0.1, thread_count=8, verbose=False, random_seed=7)
        means = [CatBoostRegressor(loss_function="Huber:delta=2.0", **kw).fit(X, yv).predict(Xt)]
        for seed in (11, 23, 37):
            means.append(lgb.train({**P, "seed": seed}, lgb.Dataset(X, yv), 300).predict(Xt))
        q_cb = CatBoostRegressor(loss_function="Quantile:alpha=0.1", **kw).fit(X, yv).predict(Xt)
        q_lg = lgb.train({**P, "seed": 11, "objective": "quantile", "alpha": 0.1}, lgb.Dataset(X, yv), 300).predict(Xt)
        M = np.vstack(means)
        te["pred"], te["pred_std"], te["q10"] = M.mean(0), M.std(0), (q_cb + q_lg) / 2
        te[["eid", "symbol", "reporter", "year", "place", "action_id", "side_code", "horizon_code", "gain", "pred", "pred_std", "q10"]].to_parquet(
            out / f"{y}.parquet")
        ep.progress("rl-peer-wf", n, len(years), f"{y} train={len(tr):,} test={len(te):,}")
    print("DONE peer walk-forward", flush=True)


def passes(sel):
    return sel.get("acted", 0) >= MIN_SELECT and sel["p10"] > P10_FLOOR and sel["cvar5"] > CVAR5_FLOOR


def judge():
    import evaluate as ev
    os.environ.setdefault("LAB_EXPERIMENT", "action_rl_peers")
    import tracking as tr
    import mlflow
    preds = pd.concat(pd.read_parquet(p) for p in sorted((D / "peer_preds").glob("*.parquet")))
    acts = pd.read_parquet(D / "actions.parquet").set_index("action_id")
    learn.ORDER = ORDER                                   # compose_policy walks the decision points in this order
    pid = tr.parent("ensemble_peers", {"actions": len(acts), "rows": len(preds)}, tags={"kind": "model", "model": "ensemble_peers"})
    best, n = {}, 0
    grid = list(itertools.product((0.3, 0.5, 0.7, 1.0), (None, -2.0, -1.5, -1.0), (False, True), (0.0, 1.0, 2.0)))
    for margin, q, cash, lcb in grid:
        r = learn.compose_policy(preds, acts, margin, q, cash, lcb)
        vd = ev.verdict(r, {})
        ok = passes(vd["select"])
        v = {"margin": margin, "q10_min": q, "cash": cash, "lcb": lcb}
        tr.log_run(f"ensemble_peers {v}", {**v, "model": "ensemble_peers"}, vd,
                   tags={"kind": "variant", "model": "ensemble_peers", "passes_risk": str(ok)}, parent_id=pid)
        key = "cash" if cash else "no_cash"
        if ok and (key not in best or vd["select"]["sharpe_trade"] > best[key][1]["select"]["sharpe_trade"]):
            best[key] = (v, vd, r)
        n += 1
        ep.progress("rl-peer-judge", n, len(grid))
    runs = mlflow.search_runs(experiment_names=[tr.EXPERIMENT], filter_string="tags.kind = 'variant'")
    trials = runs.get("metrics.all_sharpe_trade", pd.Series(dtype=float)).dropna().tolist()
    for key, (v, vd, r) in best.items():
        dsr = ev.deflated_sharpe(r.loc[r["acted"], "gain"].values, trials)
        a = r[r["acted"]]
        rep = vd["report"]
        print(f"\n{key}: chosen {v} | 2019-2022 {vd['select']['acted']} acted {vd['select']['mean']:+.2f}% SR {vd['select']['sharpe_trade']:.2f} | "
              f"2023-2026 {rep['acted']} acted {rep['mean']:+.2f}% (se {rep['se']:.2f}) win {rep['win']:.0%} p10 {rep['p10']:+.2f} "
              f"CVaR5 {rep['cvar5']:+.1f} worst {rep['worst']:+.1f} sum {rep['sum']:+.0f}% | DSR {dsr['dsr']:.3f} ({dsr['trials']} trials) | years "
              + " ".join(f"{y}:{x:+.2f}" for y, x in vd["years"].items()))
        print("   families:", a["family"].str.replace("|", " ").value_counts().head(6).to_dict(), "| places", a["place"].value_counts().to_dict())
        for s_ in ("INTC", "STX", "TSM", "SKHY", "MU", "GOOGL"):
            g = r[r["eid"].str.startswith(s_ + "|")]; ga = g[g["acted"]]
            print(f"   {s_} as peer: acted {len(ga)}/{len(g)} mean {ga['gain'].mean() if len(ga) else 0:+.2f}%")
    if not best:
        print("no variant passes the 2019-2022 limits")


if __name__ == "__main__":
    {"walkforward": walkforward, "judge": judge}[sys.argv[1]]()
