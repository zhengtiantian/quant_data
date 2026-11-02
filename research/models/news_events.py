"""What stocks did after news, article by article (PROJECT_PLAN Q.16.8, 2026-11-02).

The advice models used the news store only as daily aggregates two days late. The literature on news
and returns (Lopez-Lira & Tang; Chen, Kelly & Xiu) scores each article and finds prices take days to
absorb it; after bad earnings and guidance the drift lasts weeks (PEAD). This asks it of our store:

  build    every article of news_articles_company_matched_v2 for the 102 stocks, placed on the first
           session it could be traded on (published before the open: that session; during or after
           hours, or with a date only: the next one), with its model sentiment, event type and signal
           strength, and flags read from the title: guidance cut / raise, downgrade / upgrade. Rolled up
           per stock and session into news_days.parquet (also features for the barrier model)
  study    forward returns from that session's open over 1 / 5 / 20 / 60 sessions, minus SPY's, by kind
           of news day, against all stock-days; and the same split by whether the stock had already
           fallen 5%+ on the day (the "already priced?" question for INTC). t-stats are clustered by date

Usage: .venv311/bin/python research/models/news_events.py build|study
"""

from __future__ import annotations

import json
import math
import os
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

EP_DIR = Path(os.getenv("EP_DIR", "/Volumes/Data4T/earnings_policy"))
OUT = Path(os.getenv("ADVICE_DIR", "/Volumes/Data4T/daily_advice"))
HORIZONS = (1, 5, 20, 60)

GUIDE = r"guidance|outlook|forecast|full[- ]year|projection"
CUT = r"\b(?:cut|cuts|lower|lowers|lowered|weak|weaker|below|disappoint\w*|miss\w*|slash\w*|trim\w*|reduce\w*|soft)\b"
RAISE = r"\b(?:rais\w*|lift\w*|boost\w*|above|beat\w*|strong|stronger|tops?)\b"
DOWNGRADE = r"\bdowngrad\w*|cut to (?:sell|underperform|hold|neutral)|price target (?:cut|lowered)|lowers? (?:price )?target"
UPGRADE = r"\bupgrad\w*|raised? to (?:buy|outperform)|price target (?:raised|hiked)|raises? (?:price )?target"


def _sessions(sym: str) -> pd.DataFrame:
    d = pd.read_parquet(EP_DIR / f"{sym}.daily.parquet")[["Open", "Close"]]
    d.index = pd.to_datetime(d.index).tz_localize(None).normalize()
    return d[~d.index.duplicated()].sort_index().dropna()


PROJ = {"_id": 0, "symbol": 1, "date": 1, "title": 1, "llm_sentiment_final": 1, "llm_event_type_a": 1, "llm_signal_strength_a": 1}


def _days_for(s: str, arts: pd.DataFrame, idx: pd.DatetimeIndex):
    """One symbol's articles placed on the first session they could be traded on, rolled up per session."""
    if arts.empty:
        return None
    arts = arts.copy()
    arts["sent"] = pd.to_numeric(arts.get("llm_sentiment_final"), errors="coerce")
    ds = arts["date"].astype(str)
    stamp = pd.to_datetime(ds.where(ds.str.len() == 14), format="%Y%m%d%H%M%S", errors="coerce", utc=True).dt.tz_convert("America/New_York")
    day = pd.to_datetime(ds.str[:8], format="%Y%m%d", errors="coerce")
    pre_open = stamp.notna() & ((stamp.dt.hour * 60 + stamp.dt.minute) < 9 * 60 + 30)
    local_day = stamp.dt.tz_localize(None).dt.normalize().where(stamp.notna(), day)
    pos = np.searchsorted(idx.values, local_day.values.astype("datetime64[ns]"))
    same = (pos < len(idx)) & (idx.values[np.minimum(pos, len(idx) - 1)] == local_day.values.astype("datetime64[ns]"))
    pos = np.where(same & ~pre_open.values, pos + 1, pos)            # not before the open: next session
    ok = (pos < len(idx)) & local_day.notna().values
    arts = arts[ok].copy()
    if arts.empty:
        return None
    arts["session"] = idx[pos[ok]]
    t = arts["title"].fillna("").str.lower()
    neg, high = arts["sent"] <= -0.3, arts["llm_signal_strength_a"].eq("high")
    arts["guide_cut"] = t.str.contains(GUIDE) & t.str.contains(CUT)
    arts["guide_raise"] = t.str.contains(GUIDE) & t.str.contains(RAISE) & ~arts["guide_cut"]
    arts["downgrade"] = t.str.contains(DOWNGRADE)
    arts["upgrade"] = t.str.contains(UPGRADE)
    arts["neg_high"] = neg & high
    arts["pos_high"] = (arts["sent"] >= 0.3) & high
    arts["neg_earn"] = neg & arts["llm_event_type_a"].eq("earnings")
    arts["pos_earn"] = (arts["sent"] >= 0.3) & arts["llm_event_type_a"].eq("earnings")
    g = arts.groupby("session")
    out = pd.DataFrame({"n": g.size(), "sent": g["sent"].mean(),
                        **{c: g[c].sum() for c in ("neg_high", "pos_high", "neg_earn", "pos_earn", "guide_cut", "guide_raise", "downgrade", "upgrade")}})
    out["symbol"] = s
    return out.reset_index()


def symbol_news_days(sym: str, idx: pd.DatetimeIndex) -> pd.DataFrame:
    """The live version for one symbol over the sessions in idx (the store as it is now)."""
    from dotenv import load_dotenv
    from pymongo import MongoClient
    load_dotenv(Path(__file__).resolve().parents[2] / ".env")
    uri = (os.getenv("HOST_LOCAL_MONGO_URI") or os.getenv("LOCAL_MONGO_URI") or "mongodb://root:root@127.0.0.1:37018/")
    col = MongoClient(uri.replace("host.docker.internal", "127.0.0.1"), serverSelectionTimeoutMS=4000)["quant_data"]["news_articles_company_matched_v2"]
    since = (idx[0] - pd.Timedelta(days=3)).strftime("%Y%m%d")
    r = _days_for(sym, pd.DataFrame(list(col.find({"symbol": sym, "date": {"$gte": since}}, PROJ))), idx)
    return r if r is not None else pd.DataFrame(columns=["session", "symbol", "n", "sent", "neg_high", "pos_high", "neg_earn", "pos_earn",
                                                         "guide_cut", "guide_raise", "downgrade", "upgrade"])


def build() -> pd.DataFrame:
    from dotenv import load_dotenv
    from pymongo import MongoClient
    load_dotenv(Path(__file__).resolve().parents[2] / ".env")
    uri = (os.getenv("HOST_LOCAL_MONGO_URI") or os.getenv("LOCAL_MONGO_URI") or "mongodb://root:root@127.0.0.1:37018/")
    col = MongoClient(uri.replace("host.docker.internal", "127.0.0.1"))["quant_data"]["news_articles_company_matched_v2"]
    syms = sorted(p.name.split(".")[0] for p in EP_DIR.glob("*.daily.parquet") if not p.name.startswith("SPY."))
    proj = PROJ
    rows = []
    for s in syms:
        r = _days_for(s, pd.DataFrame(list(col.find({"symbol": s}, proj))), _sessions(s).index)
        if r is not None:
            rows.append(r)
            print(f"  {s}: {int(r.n.sum()):,} articles on {len(r):,} sessions", flush=True)
    nd = pd.concat(rows, ignore_index=True)
    OUT.mkdir(parents=True, exist_ok=True)
    nd.to_parquet(OUT / "news_days.parquet")
    print(f"news days: {len(nd):,} stock-sessions")
    return nd


def _forward() -> pd.DataFrame:
    spy = _sessions("SPY")
    parts = []
    for p in EP_DIR.glob("*.daily.parquet"):
        s = p.name.split(".")[0]
        if s == "SPY":
            continue
        d = _sessions(s)
        f = pd.DataFrame(index=d.index)
        f["day_ret"] = (d.Close / d.Close.shift() - 1) * 100            # the session the news is traded on, close to close
        so, sc = spy.Open.reindex(d.index), spy.Close.reindex(d.index)
        for h in HORIZONS:
            r = d.Close.shift(-(h - 1)) / d.Open - 1                     # from this session's open (news known before it)
            m = sc.shift(-(h - 1)) / so - 1
            f[f"x{h}"] = (r - m) * 100
            r = d.Close.shift(-h) / d.Open.shift(-1) - 1                 # from the next open (after seeing the day's close)
            m = sc.shift(-h) / so.shift(-1) - 1
            f[f"n{h}"] = (r - m) * 100
        f["symbol"], f["session"] = s, d.index
        parts.append(f[f.index >= "2016-01-01"])
    return pd.concat(parts, ignore_index=True)


def _mean_t(x: pd.Series, dates: pd.Series) -> dict:
    g = pd.DataFrame({"x": x.values, "d": dates.values}).dropna().groupby("d")["x"].mean()   # one value per date
    if len(g) < 20:
        return {"n_dates": int(len(g))}
    return {"mean": round(float(g.mean()), 2), "t": round(float(g.mean() / (g.std() / math.sqrt(len(g)))), 2), "n_dates": int(len(g))}


def study() -> dict:
    nd = pd.read_parquet(OUT / "news_days.parquet")
    fw = _forward()
    df = fw.merge(nd, on=["symbol", "session"], how="left")
    for c in ("n", "neg_high", "pos_high", "neg_earn", "pos_earn", "guide_cut", "guide_raise", "downgrade", "upgrade"):
        df[c] = df[c].fillna(0)
    kinds = {
        "all stock-days": df.index == df.index,
        "no news": df.n == 0,
        "guidance cut": df.guide_cut > 0,
        "guidance raise": df.guide_raise > 0,
        "downgrade": df.downgrade > 0,
        "upgrade": df.upgrade > 0,
        "bad earnings news": df.neg_earn > 0,
        "good earnings news": df.pos_earn > 0,
        "high-impact bad news": df.neg_high > 0,
        "high-impact good news": df.pos_high > 0,
    }
    rep = {}
    for name, m in kinds.items():
        sub = df[m]
        rep[name] = {"days": int(m.sum()), "from": "the session's open",
                     **{f"{h}d": _mean_t(sub[f"x{h}"], sub.session) for h in HORIZONS},
                     **{f"{h}d_vs_all": _mean_t(sub[f"x{h}"] - df[f"x{h}"].mean(), sub.session) for h in (5, 20, 60)}}
    # already priced? bad news on a day the stock fell 5%+ vs bad news without the fall
    bad = (df.guide_cut > 0) | (df.neg_earn > 0) | (df.downgrade > 0)
    for name, m in {"bad news + fell 5%+ that day": bad & (df.day_ret <= -5), "bad news, fell less": bad & (df.day_ret > -5),
                    "fell 5%+ without bad news": ~bad & (df.day_ret <= -5)}.items():
        sub = df[m]
        rep[name] = {"days": int(m.sum()), "from": "the next open",
                     **{f"{h}d": _mean_t(sub[f"n{h}"], sub.session) for h in HORIZONS},
                     **{f"{h}d_vs_all": _mean_t(sub[f"n{h}"] - df[f"n{h}"].mean(), sub.session) for h in (5, 20, 60)}}
    (OUT / "news_study.json").write_text(json.dumps(rep, indent=1))
    for k, v in rep.items():
        print(f"{k:30s} {v['days']:7d} | " + " | ".join(f"{h}: {v[h].get('mean', '-'):>6} (t {v[h].get('t', '-')})" for h in ("5d_vs_all", "20d_vs_all", "60d_vs_all")))
    return rep


if __name__ == "__main__":
    {"build": build, "study": study}[sys.argv[1] if len(sys.argv) > 1 else "study"]()
