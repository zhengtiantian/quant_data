#!/usr/bin/env python3
"""Algorithm zoo for the earnings-pulse lab (PROJECT_PLAN Q.6, 2026-10-26).

Each model is trained walk-forward by year on the same single-step rows as the LightGBM bake-off
(earnings_pulse.leg_frame: state, step, the stock's own history; gain net of 0.25% a side), with a
mean head (Huber-type loss) and a 10th-percentile head, and writes its out-of-sample predictions in
the bake-off's format to EP_DIR/lab_preds/<model>/<year>.parquet, for zoo.py judge.

    python research/lab/models.py catboost
    python research/lab/models.py xgboost
    python research/lab/models.py lgbm_seeds        # five LightGBMs, other seeds and feature subsets, averaged
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "models"))
import earnings_pulse as ep  # noqa: E402

OUT = ep.D / "lab_preds"
F = ep.LEG_FEATS
COLS = ["eid", "symbol", "year", "place", "level", "buyback", "gain", "pred", "q10"]


def fit_predict(name, tr, te):
    X, y, Xt = tr[F].astype(float), tr["gain"].astype(float), te[F].astype(float)
    if name == "catboost":
        from catboost import CatBoostRegressor
        kw = dict(iterations=400, depth=6, learning_rate=0.08, thread_count=8, verbose=False, random_seed=7)
        m = CatBoostRegressor(loss_function="Huber:delta=2.0", **kw).fit(X, y)
        q = CatBoostRegressor(loss_function="Quantile:alpha=0.1", **kw).fit(X, y)
        return m.predict(Xt), q.predict(Xt)
    if name == "xgboost":
        import xgboost as xgb
        kw = dict(n_estimators=300, max_depth=6, learning_rate=0.05, subsample=0.7, colsample_bytree=0.8, tree_method="hist",
                  n_jobs=8, random_state=7)
        m = xgb.XGBRegressor(objective="reg:pseudohubererror", huber_slope=2.0, **kw).fit(X, y)
        q = xgb.XGBRegressor(objective="reg:quantileerror", quantile_alpha=0.1, **kw).fit(X, y)
        return m.predict(Xt), q.predict(Xt)
    if name == "lgbm_seeds":
        import lightgbm as lgb
        ps, qs = [], []
        for seed in (11, 23, 37, 51, 73):
            P = {**ep.P_LEG, "seed": seed, "feature_fraction": 0.7, "bagging_fraction": 0.7}
            ps.append(lgb.train(P, lgb.Dataset(X, y), 250).predict(Xt))
            qs.append(lgb.train({**P, "objective": "quantile", "alpha": 0.1}, lgb.Dataset(X, y), 250).predict(Xt))
        return np.mean(ps, 0), np.mean(qs, 0)
    if name in ("hurdle_catboost", "hurdle_lgbm"):
        m = ep._Model(name).fit(tr)
        return m.predict(te)
    raise SystemExit(f"unknown model {name}")


def walkforward(name: str) -> None:
    df = ep.leg_frame()
    out = OUT / name
    out.mkdir(parents=True, exist_ok=True)
    years = [y for y in sorted(df["year"].unique()) if y >= 2019]
    for n, y in enumerate(years, 1):
        tr, te = df[df["year"] < y], df[df["year"] == y].copy()
        te["pred"], te["q10"] = fit_predict(name, tr, te)
        te[COLS].to_parquet(out / f"{y}.parquet")
        ep.progress(name, n, len(years), f"{y}")
    print(f"DONE {name}", flush=True)


if __name__ == "__main__":
    walkforward(sys.argv[1])
