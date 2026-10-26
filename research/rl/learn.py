#!/usr/bin/env python3
"""First policy on the action grammar (PROJECT_PLAN Q.7, 2026-10-26).

    python research/rl/learn.py walkforward      # Q (mean) and P10 per action, by year -> EP_DIR/rl/preds/<year>.parquet
    python research/rl/learn.py judge            # policy variants: chosen on 2019-2022, reported on 2023-2026, MLflow

Q(state, action) is learned by regression on the exact outcomes (build_events.py): with every
action's result known on every past path, the one-step problem is a full-information contextual
bandit and needs no exploration. State: the move since the close at the decision and its range,
the release minute, session, run-up, volatility, typical report move, news; plus the stock's own
earlier outcomes on the same action. Action: side, entry distance, exit kind and distances, horizon.

Policy (as live): the decision points in time order; at each, the action with the highest predicted
gain among those whose predicted P10 clears the floor; act at the first that clears the margin.
Variants: margin x P10 floor x {reduce only (no cash), with cash}.
"""
from __future__ import annotations

import itertools
import math
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
ORDER = ("r+1", "r+5", "r+20", "c-5")
STATE = ["move_at", "maxmove_at", "minmove_at", "release_min", "bmo", "run5", "run20", "vol20", "past_move_med", "news_pre_mean",
         "place_code"]
ACT = ["side_code", "entry_k", "exit_code", "y", "z", "s", "horizon_code"]
FEATS = STATE + ACT + ["own_act", "own_sym"]


def frame():
    o = pd.read_parquet(D / "outcomes.parquet")
    o = o[o["gain"].notna()]
    acts = pd.read_parquet(D / "actions.parquet")
    dec = pd.read_parquet(D / "decisions.parquet")
    pe = pd.read_parquet(ep.D / "pulse_events.parquet")
    daily = pd.read_parquet(ep.D / "events.parquet")[["symbol", "date", "run5", "run20", "vol20", "past_move_med", "news_pre_mean"]]
    daily = daily.drop_duplicates(["symbol", "date"])
    st = []
    for p in ORDER:
        x = pe[["symbol", "date", "release_min", "bmo", f"move_{p}", f"maxmove_{p}", f"minmove_{p}"]].copy()
        x.columns = ["symbol", "date", "release_min", "bmo", "move_at", "maxmove_at", "minmove_at"]
        x["place"] = p
        st.append(x)
    st = pd.concat(st).merge(daily, on=["symbol", "date"], how="left")
    st["eid"] = st["symbol"] + "|" + st["date"]
    st["place_code"] = st["place"].map({p: i for i, p in enumerate(ORDER)}).astype(float)
    st = st.drop_duplicates(["eid", "place"])
    o["place"] = o["place"].astype(str)
    df = o.merge(acts[["action_id"] + ACT], on="action_id").merge(st.drop(columns=["symbol", "date"]), on=["eid", "place"], how="inner")
    df["symbol"] = df["eid"].str.split("|").str[0]
    df["date"] = df["eid"].str.split("|").str[1]
    df["year"] = df["date"].str[:4].astype(int)
    df = df.sort_values(["date", "eid", "place"]).reset_index(drop=True)
    # the stock's own earlier outcomes on this action, and on all actions (earlier reports only)
    first = df.groupby(["symbol", "action_id", "eid"], sort=False)["gain"].mean().rename("g").reset_index()
    first["date"] = first["eid"].str.split("|").str[1]
    first = first.sort_values("date")
    first["own_act"] = first.groupby(["symbol", "action_id"])["g"].transform(lambda s: s.shift().expanding().mean())
    df = df.merge(first[["symbol", "action_id", "eid", "own_act"]], on=["symbol", "action_id", "eid"], how="left")
    per = df.groupby(["symbol", "eid"], sort=False)["gain"].mean().rename("g").reset_index()
    per["date"] = per["eid"].str.split("|").str[1]
    per = per.sort_values("date")
    per["own_sym"] = per.groupby("symbol")["g"].transform(lambda s: s.shift().expanding().mean())
    df = df.merge(per[["symbol", "eid", "own_sym"]], on=["symbol", "eid"], how="left")
    for c in FEATS:
        df[c] = df[c].astype(np.float32)
    return df


def walkforward():
    import lightgbm as lgb
    df = frame()
    P = dict(objective="huber", alpha=2.0, learning_rate=0.05, num_leaves=63, min_data_in_leaf=1000, feature_fraction=0.8,
             bagging_fraction=0.5, bagging_freq=1, lambda_l2=10.0, verbose=-1, seed=7, num_threads=8)
    (D / "preds").mkdir(exist_ok=True)
    years = [y for y in sorted(df["year"].unique()) if y >= 2019]
    for n, y in enumerate(years, 1):
        tr, te = df[df["year"] < y], df[df["year"] == y].copy()
        m = lgb.train(P, lgb.Dataset(tr[FEATS], tr["gain"]), 300)
        q = lgb.train({**P, "objective": "quantile", "alpha": 0.1}, lgb.Dataset(tr[FEATS], tr["gain"]), 300)
        te["pred"], te["q10"] = m.predict(te[FEATS]), q.predict(te[FEATS])
        te[["eid", "symbol", "year", "place", "action_id", "side_code", "horizon_code", "gain", "pred", "q10"]].to_parquet(D / "preds" / f"{y}.parquet")
        ep.progress("rl-wf", n, len(years), f"{y} train={len(tr):,} test={len(te):,}")
    print("DONE rl walk-forward", flush=True)


def policy(preds, margin, q10_min, cash):
    k = preds if cash else preds[preds["side_code"] == 0]
    if q10_min is not None:
        k = k[k["q10"] >= q10_min]
    k = k[k["pred"] >= margin]
    best = k.loc[k.groupby(["eid", "place"])["pred"].idxmax()] if len(k) else k
    best = best.assign(pc=best["place"].map({p: i for i, p in enumerate(ORDER)}))
    first = best.sort_values("pc").drop_duplicates("eid")
    out = pd.DataFrame({"eid": preds["eid"].unique()})
    out = out.merge(first[["eid", "place", "action_id", "gain"]], on="eid", how="left")
    out["acted"] = out["action_id"].notna()
    out["gain"] = out["gain"].fillna(0.0)
    out["year"] = out["eid"].str.split("|").str[1].str[:4].astype(int)
    return out


def judge():
    import evaluate as ev  # research/lab
    os.environ.setdefault("LAB_EXPERIMENT", "action_rl")
    import tracking as tr
    preds = pd.concat(pd.read_parquet(p) for p in sorted((D / "preds").glob("*.parquet")))
    sess = ev.sessions()
    acts = pd.read_parquet(D / "actions.parquet").set_index("action_id")
    pid = tr.parent("lgbm_grammar", {"actions": len(acts), "rows": len(preds)}, tags={"kind": "model", "model": "lgbm_grammar"})
    best, rows = {}, []
    for margin, q, cash in itertools.product((0.3, 0.5, 0.7, 1.0, 1.5), (None, -3.0, -2.0, -1.5, -1.0), (False, True)):
        r = policy(preds, margin, q, cash)
        vd = ev.verdict(r, sess)
        ok = ev.passes(vd["select"])
        v = {"margin": margin, "q10_min": q, "cash": cash}
        rid = tr.log_run(f"lgbm_grammar {v}", {**v, "model": "lgbm_grammar"}, vd, tags={"kind": "variant", "model": "lgbm_grammar",
                         "passes_risk": str(ok)}, parent_id=pid)
        rows.append({**v, "ok": ok, **{f"s_{k}": x for k, x in vd["select"].items()}, **{f"r_{k}": x for k, x in vd["report"].items()}})
        key = "cash" if cash else "no_cash"
        if ok and (key not in best or vd["select"]["sum"] > best[key][1]["select"]["sum"]):
            best[key] = (v, vd, rid, r)
    import mlflow
    runs = mlflow.search_runs(experiment_names=[tr.EXPERIMENT], filter_string="tags.kind = 'variant'")
    trials = runs.get("metrics.all_sharpe_trade", pd.Series(dtype=float)).dropna().tolist()
    for key, (v, vd, rid, r) in best.items():
        dsr = ev.deflated_sharpe(r.loc[r["acted"], "gain"].values, trials)
        a = r[r["acted"]].merge(acts[["name", "side", "exit", "horizon"]], left_on="action_id", right_index=True)
        print(f"\n{key}: chosen {v} | 2019-2022 {vd['select']['acted']} acted {vd['select']['mean']:+.2f}% | 2023-2026 "
              f"{vd['report']['acted']} acted {vd['report']['mean']:+.2f}% (se {vd['report']['se']:.2f}) p10 {vd['report']['p10']:+.2f} "
              f"worst {vd['report']['worst']:+.1f} sum {vd['report']['sum']:+.0f}% | DSR {dsr['dsr']:.3f} ({dsr['trials']} trials)")
        print("   exits:", a["exit"].value_counts().to_dict(), "| horizons:", a["horizon"].value_counts().to_dict(), "| sides:", a["side"].value_counts().to_dict())
        print("   most chosen:", a["name"].value_counts().head(5).to_dict())
        for s in ("INTC", "STX", "TSM", "MU"):
            g = r[r["eid"].str.startswith(s + "|")]; ga = g[g["acted"]]
            print(f"   {s}: acted {len(ga)}/{len(g)} mean {ga['gain'].mean() if len(ga) else 0:+.2f}%")
    if not best:
        print("no variant passes the 2019-2022 risk limits")
    pd.DataFrame(rows).to_parquet(D / "variants.parquet")


if __name__ == "__main__":
    {"walkforward": walkforward, "judge": judge}[sys.argv[1]]()
