"""Round 4 hygiene for the loss budget (PROJECT_PLAN Q.17 round 4 addendum, 2026-11-18): DAY_K and EVENT_K
as lagged expanding-window out-of-sample quantiles, and their breaches tested.

risk_budget.py set DAY_K = 1.671 and EVENT_K = 2.635 as the 95th percentile of the whole 2017-2026
sample, so the 5% it quotes was fitted on the days it is checked on. Here each day's (each report's)
multiplier is the 95th percentile of everything before it (t-1), after two years of burn-in, and the
breaches are tested:
  Kupiec POF               is the breach rate 5%?
  Christoffersen (1998)    are breaches independent (no clustering in volatile spells)?
Both should keep p > 0.05.

  day     300 random 5-stock equal-weight portfolios and the chip basket (as risk_budget.var_check);
          the multiplier pools every portfolio's history before the day
  event   report nights (as risk_budget.event_check), ordered by date

Usage: .venv311/bin/python research/models/q17_risk_budget.py
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import q17_ablation as qa  # noqa: E402
import risk_budget as rb  # noqa: E402

STEP = qa.Q17_DIR / "risk_budget"
P = 0.05
BURN_IN_YEARS = 2


def kupiec(hits: np.ndarray, p: float = P) -> dict:
    from scipy.stats import chi2
    n, x = len(hits), int(hits.sum())
    if n == 0:
        return {}
    ph = x / n
    ll0 = (n - x) * math.log(1 - p) + x * math.log(p)
    ll1 = ((n - x) * math.log(1 - ph) if ph < 1 else 0) + (x * math.log(ph) if ph > 0 else 0)
    lr = -2 * (ll0 - ll1)
    return {"n": n, "breaches": x, "rate_pct": round(ph * 100, 2), "p": round(float(1 - chi2.cdf(lr, 1)), 4)}


def christoffersen(hits: np.ndarray) -> dict:
    from scipy.stats import chi2
    h = hits.astype(int)
    a, b = h[:-1], h[1:]
    n00, n01 = int(((a == 0) & (b == 0)).sum()), int(((a == 0) & (b == 1)).sum())
    n10, n11 = int(((a == 1) & (b == 0)).sum()), int(((a == 1) & (b == 1)).sum())
    p01 = n01 / max(1, n00 + n01)
    p11 = n11 / max(1, n10 + n11)
    pi = (n01 + n11) / max(1, n00 + n01 + n10 + n11)

    def ll(q, k0, k1):
        return (k0 * math.log(1 - q) if q < 1 else 0) + (k1 * math.log(q) if q > 0 else 0)

    lr = -2 * (ll(pi, n00 + n10, n01 + n11) - (ll(p01, n00, n01) + ll(p11, n10, n11)))
    return {"p_breach_after_breach_pct": round(p11 * 100, 1), "p_breach_after_quiet_pct": round(p01 * 100, 1),
            "p": round(float(1 - chi2.cdf(max(lr, 0.0), 1)), 4)}


def _expanding_q(values: pd.Series, dates: pd.Series, start: pd.Timestamp) -> pd.Series:
    """For each date from `start`, the 95th percentile of all values dated before it."""
    s = pd.DataFrame({"d": dates.values, "v": values.values}).dropna().sort_values("d")
    ud = np.sort(s.d.unique())
    out, hist = {}, []
    by_day = s.groupby("d").v.apply(np.array)
    for d in ud:
        if d >= np.datetime64(start) and hist:
            out[d] = float(np.quantile(np.concatenate(hist), 1 - P))
        hist.append(by_day[d])
    return pd.Series(out)


def day() -> dict:
    px = rb.closes()
    rets = px.pct_change()
    rng = np.random.default_rng(rb.SEED)
    names = [c for c in px.columns if c != "SPY" and px[c].loc[rb.START:].notna().mean() > 0.95]
    series = {}
    for i in range(300):
        pick = list(rng.choice(names, 5, replace=False))
        pr = rets[pick].loc[rb.START:].mean(axis=1).dropna()
        sig = rb.ewma_sigma(pr).dropna()
        series[f"p{i}"] = (-pr.loc[sig.index] / sig)
    chips = rets[rb.CHIPS].loc[rb.START:].mean(axis=1).dropna()
    cs = rb.ewma_sigma(chips).dropna()
    series["chips"] = -chips.loc[cs.index] / cs
    long = pd.concat([s.rename("z").to_frame().assign(pf=k) for k, s in series.items() if k != "chips"])
    start = pd.Timestamp(rb.START) + pd.DateOffset(years=BURN_IN_YEARS)
    kq = _expanding_q(long.z, pd.Series(long.index), start)
    res, pass_both = {}, 0
    for k, z in series.items():
        kk = kq.reindex(z.index)
        m = kk.notna()
        hits = (z[m] > kk[m]).values
        ku, ch = kupiec(hits), christoffersen(hits)
        if k == "chips":
            res["chips"] = {"kupiec": ku, "christoffersen": ch}
        elif ku.get("p", 0) > 0.05 and ch.get("p", 0) > 0.05:
            pass_both += 1
    pooled = long[long.index >= start]
    hit_all = pooled.z.values > kq.reindex(pooled.index).values
    res["portfolios"] = {"count": 300, "pass_both_pct": round(pass_both / 300 * 100, 1),
                         "pooled_breach_rate_pct": round(float(np.nanmean(hit_all) * 100), 2)}
    res["k_now"] = round(float(kq.iloc[-1]), 3)
    res["k_first"] = round(float(kq.iloc[0]), 3)
    res["k_live"] = 1.671
    return res


def event() -> dict:
    px = rb.closes()
    ev, _ = rb.event_check(px)
    ev = ev.sort_values("date").reset_index(drop=True)
    ratio = -ev.move / ev.med_prev
    start = pd.Timestamp("2016-01-01") + pd.DateOffset(years=BURN_IN_YEARS)
    kq = _expanding_q(ratio, ev.date, start)
    kk = ev.date.map(kq)
    m = kk.notna()
    hits = (ratio[m] > kk[m]).values
    return {"reports_tested": int(m.sum()), "kupiec": kupiec(hits), "christoffersen_date_order": christoffersen(hits),
            "k_now": round(float(kq.iloc[-1]), 3), "k_first": round(float(kq.iloc[0]), 3), "k_live": 2.635}


def event_scaled() -> dict:
    """The fallback's typical move scaled by the volatility regime: x (EWMA vol before this report / median
    EWMA vol before the previous 8), clipped to [0.5, 3]."""
    px = rb.closes()
    rows = []
    for s in px.columns:
        if s == "SPY":
            continue
        c = px[s].dropna()
        ret = c.pct_change()
        vol = rb.ewma_sigma(ret)                       # known at the previous close
        sess = [d for d in rb.report_sessions(s, c.index) if d >= pd.Timestamp("2016-01-01")]
        moves = [abs(ret.get(d, np.nan)) for d in sess]
        vols = [vol.get(d, np.nan) for d in sess]
        for i, d in enumerate(sess):
            prev = [m for m in moves[max(0, i - 8):i] if np.isfinite(m)]
            pv = [v for v in vols[max(0, i - 8):i] if np.isfinite(v)]
            if len(prev) >= 4 and len(pv) >= 4 and np.isfinite(ret.get(d, np.nan)) and np.isfinite(vols[i]):
                scale = float(np.clip(vols[i] / np.median(pv), 0.5, 3.0))
                rows.append({"date": d, "move": ret[d], "typ": float(np.median(prev)) * scale, "scale": scale})
    ev = pd.DataFrame(rows).sort_values("date").reset_index(drop=True)
    ratio = -ev.move / ev.typ
    start = pd.Timestamp("2016-01-01") + pd.DateOffset(years=BURN_IN_YEARS)
    kq = _expanding_q(ratio, ev.date, start)
    kk = ev.date.map(kq)
    m = kk.notna()
    hits = (ratio[m] > kk[m]).values
    yr = ev.date[m].dt.year.values
    return {"reports_tested": int(m.sum()), "kupiec": kupiec(hits), "christoffersen_date_order": christoffersen(hits),
            "k_now": round(float(kq.iloc[-1]), 3), "by_year_pct": {int(y): round(float(hits[yr == y].mean() * 100), 1) for y in np.unique(yr)},
            "scale_median": round(float(ev.scale.median()), 3)}


def event_atr(per_atr: float = 1.6) -> dict:
    """The live fallback exactly (markets/risk_budget.report_move): typical = 1.6 x ATR14% at the close
    before the report; the multiplier out of sample as above."""
    rows = []
    for p in rb.EP_DIR.glob("*.daily.parquet"):
        s = p.name.split(".")[0]
        if s == "SPY":
            continue
        d = pd.read_parquet(p)[["High", "Low", "Close"]]
        d.index = pd.to_datetime(d.index).tz_localize(None).normalize()
        d = d[~d.index.duplicated()].sort_index().dropna()
        pc = d.Close.shift()
        tr = pd.concat([d.High - d.Low, (d.High - pc).abs(), (d.Low - pc).abs()], axis=1).max(axis=1)
        atr_pct = (tr.rolling(14).mean() / d.Close).shift()           # known at the close before
        ret = d.Close.pct_change()
        for day in rb.report_sessions(s, d.index):
            if day >= pd.Timestamp("2016-01-01") and np.isfinite(ret.get(day, np.nan)) and np.isfinite(atr_pct.get(day, np.nan)):
                rows.append({"date": day, "move": ret[day], "typ": per_atr * atr_pct[day]})
    ev = pd.DataFrame(rows).sort_values("date").reset_index(drop=True)
    ratio = -ev.move / ev.typ
    start = pd.Timestamp("2016-01-01") + pd.DateOffset(years=BURN_IN_YEARS)
    kq = _expanding_q(ratio, ev.date, start)
    kk = ev.date.map(kq)
    m = kk.notna()
    hits = (ratio[m] > kk[m]).values
    live_hits = (ratio[m] > 2.635).values
    yr = ev.date[m].dt.year.values
    return {"reports_tested": int(m.sum()), "kupiec_expanding": kupiec(hits), "christoffersen_expanding": christoffersen(hits),
            "kupiec_live_k_2.635": kupiec(live_hits), "k_now": round(float(kq.iloc[-1]), 3),
            "by_year_pct_expanding": {int(y): round(float(hits[yr == y].mean() * 100), 1) for y in np.unique(yr)}}


def main() -> dict:
    STEP.mkdir(parents=True, exist_ok=True)
    rep = {"day": day(), "event": event(), "event_vol_scaled": event_scaled(), "event_atr_live_fallback": event_atr()}
    (STEP / "oos_tests.json").write_text(json.dumps(rep, indent=1))
    print(json.dumps(rep, indent=1))
    return rep


if __name__ == "__main__":
    main()
