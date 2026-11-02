"""Scheduled future events as predictors (PROJECT_PLAN Q.16, 2026-11-02).

The advice models had price, news and the earnings cycle, but not the calendar of events already known
to be coming. The literature finds some scheduled events move prices in a predictable direction:
the pre-FOMC drift (Lucca & Moench 2015: most of the equity premium since 1994 earned in the 24 hours
before FOMC announcements), macro announcement days (Savor & Wilson 2013: CPI / jobs / FOMC days earn
far more than other days), the turn of the month, the pre-holiday effect, the week before options
expiration, the earnings announcement premium (Frazzini & Lamont 2007).

  study    on SPY and on the 102 stocks (2016-2026): the return of each kind of session against all
           others, t-stats by date; and the 5-session return when such an event lies ahead
  test     the advice models (5-day P(up), 5-day barrier) walk-forward with and without the calendar
           features, same seeds, same years

Events: FOMC decision days from the Fed's calendars (fomc_dates.json, 2016-2027); jobs reports on the
first Friday of the month (the usual rule; a few months differ); options expiration on the third
Friday (Thursday when Friday is a holiday); the turn of the month (last session and first three);
the session before a market holiday (from the gaps in the trading calendar); the stock's own reports.

Usage: .venv311/bin/python research/models/calendar_events.py study|test
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import daily_advice as da  # noqa: E402

OUT = da.OUT
FOMC = sorted(pd.Timestamp(d) for d in json.loads((OUT / "fomc_dates.json").read_text()))
CAL_FEATS = ["to_fomc", "fomc_eve", "fomc_day", "to_nfp", "nfp_day", "to_opex", "opex_week", "tom", "pre_holiday", "report_in_5"]


def sessions() -> pd.DatetimeIndex:
    return da._daily("SPY").index


def calendar(idx: pd.DatetimeIndex, future_extra: list | None = None) -> pd.DataFrame:
    """Per session: sessions to the next FOMC decision / jobs report / options expiration, flags for
    the event day and its eve, the turn of the month and the eve of a holiday. Future dates beyond the
    index are reached through business days, so the latest session sees what is ahead."""
    bdays = pd.bdate_range(idx[0], idx[-1] + pd.Timedelta(days=120))
    trading = idx.union(bdays[bdays > idx[-1]])                     # beyond the data: assume weekdays trade
    f = pd.DataFrame(index=idx)
    pos = {d: i for i, d in enumerate(trading)}

    def sessions_to(dates):
        ev = sorted(pos[d] for d in dates if d in pos)
        out = np.full(len(idx), np.nan)
        j = 0
        for i, d in enumerate(idx):
            p = pos[d]
            while j < len(ev) and ev[j] < p:
                j += 1
            if j < len(ev):
                out[i] = ev[j] - p
        return out

    fomc = [d for d in FOMC if d in pos]
    f["to_fomc"] = np.minimum(sessions_to(fomc), 40)
    f["fomc_day"] = (f.to_fomc == 0).astype(float)
    f["fomc_eve"] = (f.to_fomc == 1).astype(float)
    months = pd.period_range(idx[0], trading[-1], freq="M")
    nfp = []
    for m in months:
        days = trading[(trading.year == m.year) & (trading.month == m.month)]
        fri = days[days.weekday == 4]
        if len(fri):
            nfp.append(fri[0])
    f["to_nfp"] = np.minimum(sessions_to(nfp), 25)
    f["nfp_day"] = (f.to_nfp == 0).astype(float)
    opex = []
    for m in months:
        third = pd.Timestamp(m.year, m.month, 1) + pd.offsets.WeekOfMonth(week=2, weekday=4)
        if third in pos:
            opex.append(third)
        elif third - pd.Timedelta(days=1) in pos:                    # Friday a holiday: Thursday
            opex.append(third - pd.Timedelta(days=1))
    f["to_opex"] = np.minimum(sessions_to(opex), 25)
    f["opex_week"] = (f.to_opex <= 4).astype(float)
    ym = trading.to_period("M")
    first = pd.Series(trading).groupby(ym).cumcount().values
    last = pd.Series(trading[::-1]).groupby(ym[::-1]).cumcount().values[::-1]
    tom = pd.Series((first <= 2) | (last == 0), index=trading)
    f["tom"] = tom.reindex(idx).astype(float).values
    nxt = pd.Series(trading[1:].append(pd.DatetimeIndex([trading[-1] + pd.Timedelta(days=1)])), index=trading)
    gap = (nxt - pd.Series(trading, index=trading)).dt.days
    pre_hol = ((gap > 1) & (pd.Series(trading.weekday, index=trading) < 4)) | (gap > 3)
    f["pre_holiday"] = pre_hol.reindex(idx).astype(float).values
    return f


def _t(x: pd.Series) -> dict:
    x = x.dropna()
    return {"n": int(len(x)), "mean": round(float(x.mean()), 3), "t": round(float(x.mean() / (x.std() / math.sqrt(len(x)))), 2) if len(x) > 2 else None}


def study() -> dict:
    spy = da._daily("SPY")
    spy = spy[spy.index >= "2016-01-01"]
    r = spy.Close.pct_change() * 100
    r5 = (spy.Close.shift(-5) / spy.Close - 1) * 100
    cal = calendar(spy.index)
    rep = {"spy_session_return": {}, "spy_next5_when_ahead": {}}
    base = r.mean()
    for name, m in {"FOMC day": cal.fomc_day == 1, "eve of FOMC": cal.fomc_eve == 1, "jobs report day": cal.nfp_day == 1,
                    "options-expiration week": cal.opex_week == 1, "turn of the month": cal.tom == 1, "eve of a holiday": cal.pre_holiday == 1}.items():
        rep["spy_session_return"][name] = {**_t(r[m]), "vs_other_days": _t(r[m] - r[~m].mean())}
    rep["spy_session_return"]["all days"] = _t(r)
    for name, m in {"FOMC within 5 sessions": cal.to_fomc.between(1, 5), "no FOMC within 5": ~cal.to_fomc.between(1, 5),
                    "options expiration within 5": cal.to_opex.between(1, 5)}.items():
        rep["spy_next5_when_ahead"][name] = _t(r5[m])
    # stocks: excess over SPY on their report day and the sessions just before it (the announcement premium)
    df = pd.read_parquet(OUT / "dataset.parquet")
    df = df.merge(spy.assign(spy_r=r)[["spy_r"]], left_on="date", right_index=True, how="left")
    df["xr"] = df.r1 - df.spy_r
    by = df.groupby("date")
    rep["stock_excess_by_days_to_report"] = {}
    for k in (0, 1, 2, 5):
        m = df.to_report == k
        rep["stock_excess_by_days_to_report"][f"{k} days before the report"] = _t(df[m].groupby("date").xr.mean())
    rep["stock_excess_by_days_to_report"]["other days"] = _t(df[(df.to_report > 10)].groupby("date").xr.mean())
    (OUT / "calendar_study.json").write_text(json.dumps(rep, indent=1))
    for sec, v in rep.items():
        print(sec)
        for k, x in v.items():
            print(f"   {k:34s} {x}")
    return rep


def test() -> dict:
    """The advice models with and without the calendar features, walk-forward, same seeds."""
    import lightgbm as lgb
    from scipy.stats import spearmanr
    from sklearn.metrics import brier_score_loss, log_loss
    df = pd.read_parquet(OUT / "dataset.parquet")
    cal = calendar(sessions())
    df = df.join(cal, on="date")
    df["report_in_5"] = df.to_report.between(0, 7).astype(float)
    bar = pd.read_parquet(OUT / "barrier_dataset.parquet", columns=["symbol", "date", "b5"] + __import__("barrier_model").NEWS_FEATS)
    df = df.merge(bar, on=["symbol", "date"], how="left")
    df["year"] = df.date.dt.year
    base_feats = da.FEATS + __import__("barrier_model").NEWS_FEATS
    out = {}
    for target in ("up5", "barrier5"):
        res = {}
        for name, feats in (("without calendar", base_feats), ("with calendar", base_feats + CAL_FEATS)):
            preds = []
            for y in range(da.FIRST_TEST_YEAR, int(df.year.max()) + 1):
                tr = df[(df.date < pd.Timestamp(f"{y}-01-01") - pd.Timedelta(days=10))]
                te = df[df.year == y].copy()
                if target == "up5":
                    tr, te = tr[tr.y5.notna()], te[te.y5.notna()]
                    m = lgb.LGBMClassifier(n_estimators=300, learning_rate=0.03, num_leaves=15, min_child_samples=400, subsample=0.8,
                                           subsample_freq=1, colsample_bytree=0.8, reg_lambda=10, random_state=7, verbose=-1)
                    m.fit(tr[feats], (tr.y5 > 0).astype(int))
                    te["p"] = m.predict_proba(te[feats])[:, 1]
                else:
                    tr, te = tr[tr.b5.notna()], te[te.b5.notna()]
                    m = lgb.LGBMClassifier(objective="multiclass", n_estimators=300, learning_rate=0.03, num_leaves=15, min_child_samples=400,
                                           subsample=0.8, subsample_freq=1, colsample_bytree=0.8, reg_lambda=10, random_state=7, verbose=-1)
                    m.fit(tr[feats], tr.b5.astype(int))
                    p = m.predict_proba(te[feats])
                    te["p0"], te["p1"], te["p2"] = p[:, 0], p[:, 1], p[:, 2]
                preds.append(te)
            pr = pd.concat(preds)
            if target == "up5":
                y = (pr.y5 > 0).astype(int)
                ic = pr.groupby("date").apply(lambda g: spearmanr(g.p, g.y5)[0]).dropna().iloc[::5]
                # the market-level part: does the average P(up) across stocks on a date track SPY's next 5 days?
                mk = pr.groupby("date").agg(p=("p", "mean"), y=("y5", "mean")).iloc[::5]
                res[name] = {"brier": round(brier_score_loss(y, pr.p), 5), "logloss": round(log_loss(y, pr.p), 5),
                             "ic": round(float(ic.mean()), 4), "ic_t": round(float(ic.mean() / (ic.std() / math.sqrt(len(ic)))), 2),
                             "timing_corr": round(float(mk.p.corr(mk.y)), 3), "timing_t": round(float(mk.p.corr(mk.y) * math.sqrt(len(mk) - 2) /
                                                                                                      math.sqrt(1 - mk.p.corr(mk.y) ** 2)), 2)}
            else:
                yy = pr.b5.astype(int).values
                P = pr[["p0", "p1", "p2"]].values
                res[name] = {"brier": round(float(np.mean(np.sum((P - np.eye(3)[yy]) ** 2, axis=1))), 5),
                             "logloss": round(log_loss(yy, P, labels=[0, 1, 2]), 5)}
            if name == "with calendar":
                imp = pd.Series(m.booster_.feature_importance("gain"), index=feats)
                res["calendar_share_of_gain_pct"] = round(float(imp[CAL_FEATS].sum() / imp.sum() * 100), 1)
            print(target, name, res[name], flush=True)
        out[target] = res
    (OUT / "calendar_test.json").write_text(json.dumps(out, indent=1))
    return out


if __name__ == "__main__":
    {"study": study, "test": test}[sys.argv[1] if len(sys.argv) > 1 else "study"]()
