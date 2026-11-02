"""A loss budget for the holdings: is "a 1-in-20 bad day loses $X" measurable, and does keeping it
under 3% of the portfolio pay (PROJECT_PLAN Q.13 / Q.14, 2026-11-02).

The user's 10-28 portfolio was 98.7% chip names with no cash; INTC (37%) reported the next evening and
fell 29%. No model forecasts the direction of such a night; its size can be bounded. Three checks on
the 102-stock daily bars, 2017-2026, before the daily action quotes any of it:

  var      EWMA volatility (half-life 20 sessions) of 5-stock equal-weight portfolios: how often the
           next day's loss exceeds k x sigma. The multiplier that gives 5% (fat tails make it larger
           than the normal 1.645) is what the action uses for "a 1-in-20 bad day"
  event    report nights: the move against the median absolute move of the stock's previous 8
           reports (known before the report). The multiplier whose loss is exceeded 5% of the time
           sizes "this report night could lose $X"
  budget   a chip portfolio (INTC, TSM, STX, MU, AMD, equal weight, the user's kind): fully invested
           against kept to a 3% one-day loss budget (exposure scaled down to budget / bad-day loss,
           and a stock reporting next session capped at budget / its bad-night loss), rest in cash

Usage: .venv311/bin/python research/models/risk_budget.py
"""

import json
import os
from pathlib import Path

import numpy as np
import pandas as pd

EP_DIR = Path(os.getenv("EP_DIR", "/Volumes/Data4T/earnings_policy"))
OUT = EP_DIR / "risk_budget"
HALFLIFE = 20
BUDGET = 0.03
CHIPS = ["INTC", "TSM", "STX", "MU", "AMD"]
START = "2017-01-01"
SEED = 7


def closes():
    out = {}
    for p in EP_DIR.glob("*.daily.parquet"):
        s = p.name.split(".")[0]
        d = pd.read_parquet(p)["Close"]
        d.index = pd.to_datetime(d.index).tz_localize(None).normalize()
        out[s] = d[~d.index.duplicated()]
    return pd.DataFrame(out).sort_index()


def report_sessions(sym, index):
    """Sessions whose close-to-close move holds a report (the next one after an after-the-close report)."""
    try:
        rows = json.loads((EP_DIR / f"{sym}.earnings.json").read_text())
    except FileNotFoundError:
        return []
    out = []
    for r in rows:
        if r.get("eps") is None:
            continue
        t = pd.Timestamp(r["time"])
        if t.tzinfo is None:
            continue
        day = t.tz_convert("America/New_York").tz_localize(None).normalize()
        pos = index.searchsorted(day)
        if t.hour >= 16 and pos < len(index) and index[pos] == day:
            pos += 1
        if pos < len(index):
            out.append(index[pos])
    return sorted(set(out))


def ewma_sigma(r):
    return r.ewm(halflife=HALFLIFE).std().shift()          # known at the previous close


def var_check(px):
    rets = px.pct_change()
    rng = np.random.default_rng(SEED)
    names = [c for c in px.columns if c not in ("SPY",) and px[c].loc[START:].notna().mean() > 0.95]
    z = []
    for _ in range(300):
        pick = list(rng.choice(names, 5, replace=False))
        pr = rets[pick].loc[START:].mean(axis=1).dropna()
        sig = ewma_sigma(pr).dropna()
        pr = pr.loc[sig.index]
        z.append((-pr / sig).values)
    z = np.concatenate(z)
    k95 = float(np.quantile(z, 0.95))
    chips = rets[CHIPS].loc[START:].mean(axis=1).dropna()
    cs = ewma_sigma(chips).dropna()
    cz = (-chips.loc[cs.index] / cs)
    return {"portfolios": 300, "normal_1.645_exceeded_pct": round(float((z > 1.645).mean() * 100), 2),
            "k_for_5pct": round(k95, 3), "k_for_1pct": round(float(np.quantile(z, 0.99)), 3),
            "chips_exceeded_at_k_pct": round(float((cz > k95).mean() * 100), 2),
            "mean_loss_beyond_k_in_sigma": round(float(z[z > k95].mean()), 2)}


def event_check(px):
    rows = []
    for s in px.columns:
        if s == "SPY":
            continue
        c = px[s].dropna()
        ret = c.pct_change()
        sess = [d for d in report_sessions(s, c.index) if d >= pd.Timestamp("2016-01-01")]
        moves = [abs(ret.get(d, np.nan)) for d in sess]
        for i, d in enumerate(sess):
            prev = [m for m in moves[max(0, i - 8):i] if np.isfinite(m)]
            if len(prev) >= 4 and np.isfinite(ret.get(d, np.nan)):
                rows.append({"symbol": s, "date": d, "move": ret[d], "med_prev": float(np.median(prev))})
    ev = pd.DataFrame(rows)
    ratio = -ev["move"] / ev["med_prev"]
    k95 = float(np.quantile(ratio, 0.95))
    intc = ev[ev.symbol == "INTC"].sort_values("date").tail(3)
    return ev, {"reports": len(ev), "median_abs_move_pct": round(float(ev.move.abs().median() * 100), 2),
                "k_for_5pct_loss": round(k95, 3), "mean_move_pct": round(float(ev.move.mean() * 100), 3),
                "intc_last": [{"date": str(r.date.date()), "move_pct": round(r.move * 100, 1), "med_prev_pct": round(r.med_prev * 100, 1),
                               "bad_night_pct": round(r.med_prev * k95 * 100, 1)} for r in intc.itertuples()]}


def budget_check(px, ev, k_day, k_event):
    rets = px[CHIPS].pct_change().loc[START:]
    w = pd.DataFrame(1 / len(CHIPS), index=rets.index, columns=CHIPS)
    full = (rets * w).sum(axis=1)
    sig = ewma_sigma(full)
    scale = (BUDGET / (k_day * sig)).clip(upper=1.0).fillna(1.0)
    # a stock reporting in the next session is capped at budget / its bad-night loss (portfolio weight)
    medp = ev.set_index(["symbol", "date"])["med_prev"]
    wb = w.mul(scale, axis=0)
    for (s, d), m in medp.items():
        if s in CHIPS and d in wb.index:
            cap = BUDGET / (k_event * m)
            wb.loc[d, s] = min(wb.loc[d, s], cap)
    budget = (rets * wb).sum(axis=1)

    def stats(r):
        r = r.dropna()
        eq = (1 + r).cumprod()
        return {"cagr_pct": round((eq.iloc[-1] ** (252 / len(r)) - 1) * 100, 1), "vol_pct": round(r.std() * np.sqrt(252) * 100, 1),
                "sharpe": round(r.mean() / r.std() * np.sqrt(252), 2), "max_dd_pct": round((eq / eq.cummax() - 1).min() * 100, 1),
                "worst_day_pct": round(r.min() * 100, 1), "days_lost_over_3pct": int((r < -0.03).sum()),
                "avg_invested_pct": None}
    a, b = stats(full), stats(budget)
    b["avg_invested_pct"] = round(float(wb.sum(axis=1).mean() * 100), 1)
    a["avg_invested_pct"] = 100.0
    by_year = pd.DataFrame({"full": full, "budget": budget}).groupby(full.index.year).apply(lambda g: ((1 + g).prod() - 1) * 100).round(1)
    return {"basket": CHIPS, "full": a, "budget_3pct": b, "by_year_pct": by_year.to_dict(orient="index")}


def main():
    OUT.mkdir(exist_ok=True)
    px = closes()
    rep = {"var": var_check(px)}
    ev, rep["event"] = event_check(px)
    rep["budget"] = budget_check(px, ev, rep["var"]["k_for_5pct"], rep["event"]["k_for_5pct_loss"])
    (OUT / "report.json").write_text(json.dumps(rep, indent=1, default=str))
    print(json.dumps(rep, indent=1, default=str))


if __name__ == "__main__":
    main()
