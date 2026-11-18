"""Q.17.4's pre-registered fallback: the empirical-Bayes shrunk table of what follows a >= 2 x ATR drop
(PROJECT_PLAN Q.17.4 and its round 4 addendum, 2026-11-18).

The meta-labelling models did not beat the grouped rates (q17_drops.py), so, as pre-registered, the
grouped rates are the answer, shrunk: a group's rate is pulled toward its parent's in proportion to how
little evidence it has. Each drop is weighted 1 / n_d (the core drops that day), so a market-wide day with
40 drops counts as one day, not 40. Levels: all -> report or not -> x market-wide (SPY <= -1.5%) or not
-> x bad news on the drop or not. Prior strength per level by the method of moments on the beta-binomial
(the between-group variance in excess of the binomial noise), floored at 1.

  fit     2018-2023 -> scored on 2024-2026 against the raw table and the pooled rate (weighted Brier,
          20-session block bootstrap over dates); then the table on 2018-2026 for use
          -> Q17_DIR/4/shrunk_table.json; also the Kish effective number of independent dates

Usage: .venv311/bin/python research/models/q17_shrunk_table.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import q17_ablation as qa  # noqa: E402

STEP = qa.Q17_DIR / "4"
TARGETS = ("lower5", "bad20")
LEVELS = (("earnings",), ("earnings", "market_wide"), ("earnings", "market_wide", "bad_news"))


def _kish(dates: pd.Series) -> float:
    n = dates.value_counts().values.astype(float)
    return float(n.sum() ** 2 / (n ** 2).sum())


def _kappa(k: np.ndarray, n: np.ndarray, p0: np.ndarray) -> float:
    """Beta-binomial prior strength from the spread of the groups' rates around their parents'."""
    ok = n > 0
    r = k[ok] / n[ok]
    excess = np.average((r - p0[ok]) ** 2, weights=n[ok]) - np.average(p0[ok] * (1 - p0[ok]) / n[ok], weights=n[ok])
    if excess <= 0:
        return 1e6                                   # no spread beyond noise: everything shrinks to the parent
    pq = np.average(p0[ok] * (1 - p0[ok]), weights=n[ok])
    return float(max(1.0, pq / excess - 1))


def fit(df: pd.DataFrame, y: str) -> dict:
    """{level key: {"rate", "n_eff_weight", "kappa"}} with every level shrunk toward the one above."""
    w = 1.0 / df.groupby("date").date.transform("size")
    d = df.assign(w=w, wy=w * df[y])
    table = {"all": {"rate": float(d.wy.sum() / d.w.sum()), "weight": float(d.w.sum()), "drops": int(len(d))}}
    parent_rate = {(): table["all"]["rate"]}
    for lv in LEVELS:
        g = d.groupby(list(lv)).agg(k=("wy", "sum"), n=("w", "sum"), drops=("w", "size"))
        keys = [tuple(int(v) for v in (i if isinstance(i, tuple) else (i,))) for i in g.index]
        p0 = np.array([parent_rate[key[:-1]] for key in keys])
        kap = _kappa(g.k.values, g.n.values, p0)
        for key, k, n, dr, p in zip(keys, g.k.values, g.n.values, g.drops.values, p0):
            rate = (k + kap * p) / (n + kap)
            parent_rate[key] = rate
            table["|".join(f"{c}={v}" for c, v in zip(lv, key))] = {"rate": round(float(rate), 4), "raw": round(float(k / n), 4),
                                                                   "weight": round(float(n), 1), "drops": int(dr), "kappa": round(kap, 1)}
    table["_rates"] = {k: float(v) for k, v in parent_rate.items() if len(k) == len(LEVELS[-1])}
    return table


def predict(table: dict, df: pd.DataFrame) -> np.ndarray:
    rates = table["_rates"]
    out = []
    for e, m, b in zip(df.earnings.astype(int), df.market_wide.astype(int), df.bad_news.astype(int)):
        out.append(rates.get((e, m, b), table["all"]["rate"]))
    return np.array(out)


def _block_boot(diff: np.ndarray, w: np.ndarray, dates: np.ndarray, block: int = 20, n_boot: int = 2000, seed: int = 7) -> list:
    """Weighted mean difference, resampling blocks of `block` consecutive drop dates."""
    ud, inv = np.unique(dates, return_inverse=True)
    num, den = np.bincount(inv, weights=w * diff), np.bincount(inv, weights=w)
    rng, nb, out = np.random.default_rng(seed), int(np.ceil(len(ud) / block)), []
    for _ in range(n_boot):
        starts = rng.integers(0, len(ud) - block + 1, nb)
        idx = np.concatenate([np.arange(s, s + block) for s in starts])[:len(ud)]
        out.append(num[idx].sum() / den[idx].sum())
    return [round(float(v), 5) for v in np.percentile(out, [2.5, 97.5])]


def main() -> dict:
    df = pd.read_parquet(STEP / "drops.parquet")
    tr, te = df[df.year < 2024], df[df.year >= 2024]
    rep = {"kish_dates": {"train": round(_kish(tr.date), 1), "test": round(_kish(te.date), 1),
                          "train_dates": int(tr.date.nunique()), "test_dates": int(te.date.nunique())}, "test": {}, "tables": {}}
    for y in TARGETS:
        t = fit(tr, y)
        raw_tr = tr.groupby(["earnings", "market_wide", "bad_news"])[y].mean()
        p_raw = np.array([raw_tr.get((e, m, b), tr[y].mean()) for e, m, b in zip(te.earnings, te.market_wide, te.bad_news)])
        p_eb = predict(t, te)
        p_pool = np.full(len(te), tr[y].mean())
        yy = te[y].values
        w = (1.0 / te.groupby("date").date.transform("size")).values
        br = {k: (p - yy) ** 2 for k, p in {"shrunk": p_eb, "raw_8_groups": p_raw, "pooled": p_pool}.items()}
        res = {k: round(float(np.average(v, weights=w)), 4) for k, v in br.items()}
        res["shrunk_minus_raw_ci95"] = _block_boot(br["shrunk"] - br["raw_8_groups"], w, te.date.values)
        res["shrunk_minus_pooled_ci95"] = _block_boot(br["shrunk"] - br["pooled"], w, te.date.values)
        rep["test"][y] = res
        full = fit(df, y)
        full.pop("_rates")
        rep["tables"][y] = full
    (STEP / "shrunk_table.json").write_text(json.dumps(rep, indent=1))
    print(json.dumps(rep, indent=1))
    return rep


if __name__ == "__main__":
    main()
