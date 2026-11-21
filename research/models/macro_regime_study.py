"""Can macro state tell a panic that rebounds from one that keeps falling? (2026-11-20, the user's request)

After the 11-19 / 11-20 selloff (SPY -7%, chips -18%, VIX 17 -> 30 in seven sessions) neither daily model
turned bearish: in their 2016-2026 data a panic day (VIX >= 25 and SPY down >= 3% over 5 sessions) is
followed by +4.9% over 20 sessions on average against +1.8% on other days. But 7 of the 18 panic episodes kept
falling (2016-01, 2018-10, 2020-02, three in 2022, 2025-03), and the models see SPY, VIX and the chip basket
only: no rates, dollar or credit. This study adds them at the market level (the 102 stocks' equal-weight
average), where they act:

  macro    10-year yield (level, 20 / 60-session change), 2-year change, the 10y-2y curve (FRED, one-day
           publication lag), HYG / IEF (high-yield against Treasuries, adjusted closes: a credit proxy; FRED's
           ICE spread starts in 2023), the dollar index DX-Y.NYB, the Chicago Fed NFCI (weekly, six days lag)
  episodes the 18 first panic days, macro state of the bad (next 20 sessions < 0) against the good ones
  walk     walk-forward by year 2019-2026 (train on the earlier years, 20-session embargo): ridge and a small
           LightGBM on the market features with and without macro, predicting the next 20 sessions' average
           return. Out-of-sample rank IC, and the policy "half exposure when the prediction is < 0" against
           always holding, with block-bootstrap (20-session blocks) intervals

Usage: .venv311/bin/python research/models/macro_regime_study.py
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

OUT = Path("/Volumes/Data4T/macro_regime_2026-11-20")
RAW = OUT / "raw"
DATASET = Path("/Volumes/Data4T/daily_advice/dataset.parquet")
BASE = ["vix", "vix_pct", "spy_r5", "spy_r20", "semi_r5", "semi_r20", "breadth_dd252"]
MACRO = ["y10", "y10_chg20", "y10_chg60", "y2_chg60", "curve", "curve_chg60", "credit_chg20", "credit_chg60",
         "dollar_chg20", "dollar_chg60", "nfci"]
H = 20
BLOCK = 20
N_BOOT = 2000


def fred(series: str, lag_days: int) -> pd.Series:
    """A FRED series as known at each date: the value published `lag_days` calendar days after its date."""
    d = pd.read_csv(RAW / f"{series}.csv", na_values=".")
    s = pd.Series(pd.to_numeric(d.iloc[:, 1], errors="coerce").values, index=pd.to_datetime(d.iloc[:, 0])).dropna()
    s.index = s.index + pd.Timedelta(days=lag_days)
    return s


def yahoo() -> pd.DataFrame:
    p = RAW / "yahoo.parquet"
    if not p.exists():
        import yfinance as yf
        d = yf.download(["HYG", "IEF", "DX-Y.NYB"], start="2014-01-01", auto_adjust=True, progress=False)["Close"]
        d.to_parquet(p)
    return pd.read_parquet(p)


def market_frame() -> pd.DataFrame:
    d = pd.read_parquet(DATASET)
    g = d.groupby("date").agg(vix=("vix", "first"), vix_pct=("vix_pct", "first"), spy_r5=("spy_r5", "first"),
                              spy_r20=("spy_r20", "first"), semi_r5=("semi_r5", "first"),
                              semi_r20=("semi_r20", "first"), breadth_dd252=("dd_252", "mean"),
                              r1=("r1", "mean"), y5=("y5", "mean"), y20=("y20", "mean"), y60=("y60", "mean"))
    g.index = pd.to_datetime(g.index)
    days = g.index
    asof = lambda s: s.reindex(s.index.union(days)).sort_index().ffill().reindex(days)  # noqa: E731
    y10, y2, curve = asof(fred("DGS10", 1)), asof(fred("DGS2", 1)), asof(fred("T10Y2Y", 1))
    nfci = asof(fred("NFCI", 6))
    yh = yahoo()
    yh.index = pd.to_datetime(yh.index)
    credit = asof(np.log(yh["HYG"] / yh["IEF"]).dropna())
    dollar = asof(np.log(yh["DX-Y.NYB"]).dropna())
    g["y10"], g["y10_chg20"], g["y10_chg60"] = y10, y10 - y10.shift(20), y10 - y10.shift(60)
    g["y2_chg60"] = y2 - y2.shift(60)
    g["curve"], g["curve_chg60"] = curve, curve - curve.shift(60)
    g["credit_chg20"], g["credit_chg60"] = (credit - credit.shift(20)) * 100, (credit - credit.shift(60)) * 100
    g["dollar_chg20"], g["dollar_chg60"] = (dollar - dollar.shift(20)) * 100, (dollar - dollar.shift(60)) * 100
    g["nfci"] = nfci
    g["ret_next"] = g["r1"].shift(-1) / 100    # the next session's equal-weight return (r1: the session's own, in %)
    return g


def episodes(g: pd.DataFrame) -> dict:
    panic = g[(g.vix >= 25) & (g.spy_r5 <= -3)].dropna(subset=["y20"])
    first = [t for i, t in enumerate(panic.index) if i == 0 or (t - panic.index[i - 1]).days > 30]
    e = g.loc[first, ["vix", "spy_r5", "y20", "y60"] + MACRO].copy()
    e["bad"] = e["y20"] < 0
    from scipy.stats import mannwhitneyu
    cmp = {}
    for c in MACRO:
        b, ok = e.loc[e.bad, c].dropna(), e.loc[~e.bad, c].dropna()
        p = mannwhitneyu(b, ok).pvalue if len(b) and len(ok) else None
        cmp[c] = {"bad_mean": round(float(b.mean()), 3), "good_mean": round(float(ok.mean()), 3),
                  "p": None if p is None else round(float(p), 3)}
    return {"table": e.round(3).reset_index().astype({"date": str}).to_dict("records"), "compare": cmp,
            "n_bad": int(e.bad.sum()), "n_good": int((~e.bad).sum())}


def _models():
    from lightgbm import LGBMRegressor
    from sklearn.linear_model import RidgeCV
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    return {"ridge": lambda: make_pipeline(StandardScaler(), RidgeCV(alphas=np.logspace(-1, 4, 20))),
            "lgbm": lambda: LGBMRegressor(n_estimators=200, learning_rate=0.03, max_depth=3, num_leaves=7,
                                          min_child_samples=40, subsample=0.8, subsample_freq=1,
                                          colsample_bytree=0.8, verbose=-1, random_state=0)}


def walk_forward(g: pd.DataFrame) -> pd.DataFrame:
    rows = []
    data = g.dropna(subset=BASE + MACRO + ["y20"])
    for year in range(2019, 2027):
        test = data[data.index.year == year]
        if test.empty:
            continue
        cut = data.index[data.index < pd.Timestamp(f"{year}-01-01")]
        if len(cut) <= H:
            continue
        train = data.loc[: cut[-H - 1]]               # labels look 20 sessions ahead: embargo
        out = pd.DataFrame(index=test.index)
        for name, make in _models().items():
            for fs, feats in (("base", BASE), ("macro", BASE + MACRO)):
                m = make()
                m.fit(train[feats], train["y20"])
                out[f"{name}_{fs}"] = m.predict(test[feats])
        rows.append(out)
    return pd.concat(rows)


def _blocks(n: int, rng) -> np.ndarray:
    starts = rng.integers(0, n - BLOCK, size=n // BLOCK + 1)
    return np.concatenate([np.arange(s, s + BLOCK) for s in starts])[:n]


def evaluate(g: pd.DataFrame, pred: pd.DataFrame) -> dict:
    from scipy.stats import spearmanr
    d = pred.join(g[["y20", "ret_next", "vix", "spy_r5"]]).dropna(subset=["ret_next"])
    rng = np.random.default_rng(0)
    res = {}
    hold = d["ret_next"].values

    def stats(r):
        ann = r.mean() * 252
        vol = r.std() * np.sqrt(252)
        eq = np.cumprod(1 + r)
        return {"ann_ret_pct": round(ann * 100, 2), "ann_vol_pct": round(vol * 100, 2),
                "sharpe": round(ann / vol, 3), "max_dd_pct": round(float((eq / np.maximum.accumulate(eq) - 1).min()) * 100, 1)}

    res["hold"] = stats(hold)
    panic = (d.vix >= 25) & (d.spy_r5 <= -3)
    for col in pred.columns:
        ic = spearmanr(d[col], d["y20"], nan_policy="omit").correlation
        ic_p = spearmanr(d.loc[panic, col], d.loc[panic, "y20"], nan_policy="omit").correlation if panic.sum() > 5 else None
        expo = np.where(d[col].values < 0, 0.5, 1.0)
        pol = expo * hold
        diff = []
        for _ in range(N_BOOT):
            i = _blocks(len(d), rng)
            a, b = pol[i], hold[i]
            diff.append(a.mean() / a.std() - b.mean() / b.std())
        diff = np.array(diff) * np.sqrt(252)
        res[col] = {"ic_all": round(float(ic), 4), "ic_panic_days": None if ic_p is None else round(float(ic_p), 4),
                    "share_half_exposure_pct": round(float((expo < 1).mean()) * 100, 1),
                    "policy": stats(pol),
                    "sharpe_minus_hold_ci90": [round(float(np.percentile(diff, 5)), 3), round(float(np.percentile(diff, 95)), 3)]}
    # macro against base, same model: block bootstrap of the IC difference
    for name in ("ridge", "lgbm"):
        a, b, y = d[f"{name}_macro"].values, d[f"{name}_base"].values, d["y20"].values
        diffs = []
        for _ in range(N_BOOT):
            i = _blocks(len(d), rng)
            diffs.append(spearmanr(a[i], y[i]).correlation - spearmanr(b[i], y[i]).correlation)
        res[f"{name}_ic_macro_minus_base_ci90"] = [round(float(np.percentile(diffs, 5)), 4), round(float(np.percentile(diffs, 95)), 4)]
    res["test_days"] = int(len(d))
    res["panic_test_days"] = int(panic.sum())
    return res


def main() -> None:
    g = market_frame()
    g.to_parquet(OUT / "market_frame.parquet")
    ep = episodes(g)
    pred = walk_forward(g)
    pred.to_parquet(OUT / "walkforward.parquet")
    ev = evaluate(g, pred)
    report = {"episodes": ep, "walk_forward": ev, "base_features": BASE, "macro_features": MACRO}
    (OUT / "report.json").write_text(json.dumps(report, indent=1, default=str))
    print(json.dumps({"compare": ep["compare"], "n_bad": ep["n_bad"], "n_good": ep["n_good"]}, indent=1))
    print(json.dumps(ev, indent=1))


if __name__ == "__main__":
    main()
