"""The rest of the Q.17.2 statistics base (round 4 addendum, 2026-11-18) for the 5-session barrier model:

  1. purge by the global trading-day index and uniqueness weights. Live, a test year's training set ends
     7 calendar days before 1 January. Here a training row must end before the test year starts, counted
     in sessions (pos + label length < first session of the year), and each row is weighted by its average
     uniqueness (Lopez de Prado: the mean over its label window of 1 / the number of the same stock's
     labels running that session) divided by the number of stocks that day, so every date weighs the same
  2. split volatility from direction: V = 1 - P(none) (will it move?), S = P(lower) / (P(lower) +
     P(upper)) (which way?). Within each fifth of V, does the top fifth of S do worse over the 5 sessions
     than the rest of that V fifth? A model whose S only restates V would fail this.

  build   label lengths, positions and weights onto 2a's v2 barrier dataset -> Q17_DIR/purge/barrier_5w.parquet
          (Colab fits both variants: live folds unweighted / purged and weighted)
  score   Brier and BSS of both against the pooled base rate, 20-session block bootstrap of the difference;
          then the double sort on the better one, Dirichlet-calibrated as live (fitted on earlier years only)

Usage: .venv311/bin/python research/models/q17_purge.py build|score
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

STEP = qa.Q17_DIR / "purge"
H, UP, DOWN = 5, 1.0, 1.5
CLS = ("lower", "none", "upper")


def _lengths(d: pd.DataFrame) -> np.ndarray:
    """Sessions each 5-session barrier label runs (to the touch, else 5), as barrier_model.barrier_labels."""
    o, hi, lo, c = d.Open.values, d.High.values, d.Low.values, d.Close.values
    pc = np.r_[np.nan, c[:-1]]
    tr = np.nanmax(np.c_[hi - lo, np.abs(hi - pc), np.abs(lo - pc)], axis=1)
    atr = pd.Series(tr).rolling(14).mean().values
    n = len(d)
    out = np.full(n, np.nan)
    for t in range(n - H - 1):
        if not np.isfinite(atr[t]):
            continue
        e = o[t + 1]
        u, l = e + UP * atr[t], e - DOWN * atr[t]
        out[t] = H
        for j in range(t + 1, t + 1 + H):
            if lo[j] <= l or hi[j] >= u:
                out[t] = j - t
                break
    return out


def build() -> Path:
    import barrier_model as bm
    import news_events as ne
    STEP.mkdir(parents=True, exist_ok=True)
    df = pd.read_parquet(qa.STEP / "v2" / "barrier_dataset.parquet")
    df = df[df.b5.notna()].copy()
    cal = ne._sessions("SPY").index
    gpos = {d: i for i, d in enumerate(cal)}
    parts = []
    for s, g in df.groupby("symbol"):
        d = bm.da._daily(s)
        L = pd.Series(_lengths(d), index=d.index).reindex(g.date).values
        pos = np.array([gpos.get(x, -1) for x in g.date])
        # concurrency of this stock's labels per session: label t covers sessions t+1 .. t+L
        span_end = pos + np.nan_to_num(L, nan=H).astype(int)
        lo_, hi_ = pos.min(), span_end.max() + 1
        conc = np.zeros(hi_ - lo_ + 1)
        for a, b in zip(pos, span_end):
            conc[a + 1 - lo_:b + 1 - lo_] += 1
        uniq = np.array([np.mean(1.0 / conc[a + 1 - lo_:b + 1 - lo_]) for a, b in zip(pos, span_end)])
        parts.append(pd.DataFrame({"symbol": s, "date": g.date.values, "label_len": L, "gpos": pos, "uniq": uniq}))
    x = df.merge(pd.concat(parts), on=["symbol", "date"], how="left", validate="one_to_one")
    x["n_t"] = x.groupby("date").symbol.transform("size")
    x["w"] = x.uniq / x.n_t
    x["w"] = x.w / x.w.mean()
    keep = list(dict.fromkeys(["symbol", "date", "b5", "label_len", "gpos", "uniq", "n_t", "w", "dd_252", "atr_pct", *bm.FEATS]))
    x = x[keep]
    num = x.select_dtypes("float64").columns
    x[num] = x[num].astype("float32")
    out = STEP / "barrier_5w.parquet"
    x.to_parquet(out)
    first = {int(y): int(gpos[cal[cal >= pd.Timestamp(f"{y}-01-01")][0]]) for y in range(2019, 2027)}
    (STEP / "meta.json").write_text(json.dumps({"feats": list(bm.FEATS), "first_session_pos": first, "h": H}, indent=1))
    print(f"{out}: {len(x):,} rows; uniqueness mean {x.uniq.mean():.3f}, label length mean {x.label_len.mean():.2f}")
    return out


def _brier(y, P):
    return np.sum((P - np.eye(3)[y]) ** 2, axis=1)


def _block(diff, dates, block=20, n_boot=2000, seed=7):
    ud, inv = np.unique(dates, return_inverse=True)
    num, cnt = np.bincount(inv, weights=diff), np.bincount(inv)
    rng, nb, out = np.random.default_rng(seed), int(np.ceil(len(ud) / block)), []
    for _ in range(n_boot):
        st = rng.integers(0, len(ud) - block + 1, nb)
        idx = np.concatenate([np.arange(s, s + block) for s in st])[:len(ud)]
        out.append(num[idx].sum() / cnt[idx].sum())
    return [round(float(v), 5) for v in np.percentile(out, [2.5, 97.5])]


def score() -> dict:
    import barrier_model as bm
    import news_events as ne
    pr = pd.read_parquet(STEP / "colab" / "preds.parquet")
    pr["date"] = pd.to_datetime(pr.date)
    y = pr.b5.astype(int).values
    base = _brier(y, pr[[f"base_{c}" for c in CLS]].values)
    rep = {"days": int(len(pr)), "models": {}}
    for k in ("live", "purged_weighted"):
        b = _brier(y, pr[[f"p_{c}__{k}" for c in CLS]].values)
        rep["models"][k] = {"brier": round(float(b.mean()), 5), "bss_vs_base_pct": round(float((1 - b.mean() / base.mean()) * 100), 2)}
    d = _brier(y, pr[[f"p_{c}__purged_weighted" for c in CLS]].values) - _brier(y, pr[[f"p_{c}__live" for c in CLS]].values)
    rep["purged_minus_live"] = {"diff": round(float(d.mean()), 5), "ci95_block20": _block(d, pr.date.values)}
    best = "purged_weighted" if rep["purged_minus_live"]["diff"] < 0 else "live"
    # Dirichlet-calibrated as live, each year fitted on the years before it (7-day purge)
    P = pr[[f"p_{c}__{best}" for c in CLS]].values
    Q = np.full_like(P, np.nan)
    years = pr.date.dt.year.values
    for yr in range(2020, int(years.max()) + 1):
        calm = (pr.date < pd.Timestamp(f"{yr}-01-01") - pd.Timedelta(days=7)).values
        te = years == yr
        if calm.sum() < 10000:
            continue
        Q[te] = bm.dirichlet_apply(bm.dirichlet_fit(P[calm], y[calm]), P[te])
    ok = np.isfinite(Q[:, 0])
    pr = pr[ok].copy()
    Q = Q[ok]
    pr["V"] = 1 - Q[:, 1]
    pr["S"] = Q[:, 0] / (Q[:, 0] + Q[:, 2])
    # the 5 sessions from the next open, minus SPY (news_events' n5)
    fw = ne._forward()[["symbol", "session", "n5"]].rename(columns={"session": "date"})
    pr = pr.merge(fw, on=["symbol", "date"], how="left").dropna(subset=["n5"])
    pr["v5"] = pr.groupby("date").V.transform(lambda s: pd.qcut(s.rank(method="first"), 5, labels=False))
    pr["s5"] = pr.groupby(["date", "v5"]).S.transform(lambda s: pd.qcut(s.rank(method="first"), 5, labels=False) if len(s) >= 5 else np.nan)
    pr = pr.dropna(subset=["s5"])
    ds = {}
    for v in range(5):
        g = pr[pr.v5 == v]
        per = g.groupby("date").apply(lambda x: x.loc[x.s5 == 4, "n5"].mean() - x.loc[x.s5 < 4, "n5"].mean(), include_groups=False).dropna()
        late = per[per.index >= "2024-01-01"]
        ds[f"V fifth {v + 1}"] = {"top_S_minus_rest_pct": round(float(per.mean()), 3),
                                  "ci95_block20": _block(per.values, per.index.values),
                                  "2024_2026_pct": round(float(late.mean()), 3), "dates": int(len(per))}
    rep["calibrated_with"] = best
    rep["double_sort_5_sessions_minus_spy"] = ds
    rep["double_sort_passes"] = all(v["ci95_block20"][1] < 0 for v in ds.values())
    (STEP / "score.json").write_text(json.dumps(rep, indent=1, default=str))
    print(json.dumps(rep, indent=1, default=str))
    return rep


if __name__ == "__main__":
    {"build": build, "score": score}[sys.argv[1] if len(sys.argv) > 1 else "build"]()
