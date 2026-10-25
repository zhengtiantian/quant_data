#!/usr/bin/env python3
"""An earnings-day trading policy learned from the whole universe's past reports (PROJECT_PLAN Q.5 v3).

The question, for a holding whose report just came out, at the open of the first session that
trades on it: sell part into the move and buy it back lower, hold, or (with cash) buy the dip?
The user wants the system to learn when each pays, not a rule written for one stock.

    python research/models/earnings_policy.py fetch    # daily bars, report dates and EPS (yfinance)
    python research/models/earnings_policy.py minutes  # 5-minute bars incl. pre/after-hours around each report (Alpaca)
    python research/models/earnings_policy.py build    # one row per (report, action): features + outcome
    python research/models/earnings_policy.py train    # walk-forward LightGBM, the policy's out-of-sample record

Actions, enumerated (none chosen by hand): sell at the open and buy back at -b% (or at the close
of day H if it never gets there); sell at +a% if reached and buy back b% below the sale; buy at
the open or at -b% and sell at the close of day H (needs cash). Outcome = what the action made per
share against holding, in % of the open. Hourly bars give the order of highs and lows: a buy-back
counts only in a bar after the sale's (or, for a sale at the open, from the first bar on, since the
open comes first). Features are only what is known at the open: the gap, the pre-market path, the
EPS surprise, run-up and volatility, the gap against the stock's usual reaction, how the stock's
earlier reports faded or ran (earlier reports only), the market's own gap, report timing.

Data: yfinance, hourly back to ~2023-11 (its 730-day limit), so ~11 reports per stock. Cached in
EP_DIR (default /Volumes/Data4T/earnings_policy). Progress: "PROGRESS <stage> <done> <total>".
"""
from __future__ import annotations

import json
import math
import os
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "research" / "features"))
EP_DIR = Path(os.getenv("EP_DIR", "/Volumes/Data4T/earnings_policy"))
MARKET = "SPY"
EARNINGS_LIMIT = 48                       # ~12 years of report dates: the older ones feed the stock's history features
INTRADAY_FROM = "2016-01-01"              # Alpaca's historical minute data starts in 2016
ALPACA_BARS = "https://data.alpaca.markets/v2/stocks/{sym}/bars"
SELL_B = (1, 2, 3, 4, 6, None)            # None: buy back at the close of day H
SELL_A = (1, 2, 4)
BUY_B = (0, 2, 4)
HORIZONS = (1, 3)
MARGIN = float(os.getenv("EP_MARGIN", "0.5"))     # % per share the policy needs to see before it acts


def progress(stage, done, total, extra=""):
    print(f"PROGRESS {stage} {done} {total} {extra}".rstrip(), flush=True)


def mongo():
    from dotenv import load_dotenv
    from pymongo import MongoClient
    load_dotenv(ROOT / ".env", override=False)
    return MongoClient(os.getenv("HOST_LOCAL_MONGO_URI") or os.getenv("MONGO_URI"))["quant_data"]


# ---------------------------------------------------------------- fetch

def fetch(symbols):
    import yfinance as yf
    EP_DIR.mkdir(parents=True, exist_ok=True)
    for i, sym in enumerate(symbols, 1):
        out = EP_DIR / f"{sym}.daily.parquet"
        if out.exists() and time.time() - out.stat().st_mtime < 20 * 3600 and not os.getenv("EP_REFRESH"):
            progress("fetch", i, len(symbols), f"{sym} cached")
            continue
        try:
            t = yf.Ticker(sym)
            d = t.history(start="2015-06-01", interval="1d", auto_adjust=False)
            e = None if sym == MARKET else t.get_earnings_dates(limit=EARNINGS_LIMIT)
            if d.empty:
                raise ValueError("no bars")
            d.to_parquet(out)
            if e is not None and not e.empty:
                rows = [{"time": ts.isoformat(), "eps_est": _f(r["EPS Estimate"]), "eps": _f(r["Reported EPS"]),
                         "surprise": _f(r["Surprise(%)"])} for ts, r in e.iterrows()]
                (EP_DIR / f"{sym}.earnings.json").write_text(json.dumps(rows))
            progress("fetch", i, len(symbols), f"{sym} daily={len(d)} reports={0 if e is None else len(e)}")
        except Exception as ex:  # noqa: BLE001 - one symbol failing must not stop the rest
            progress("fetch", i, len(symbols), f"{sym} failed {type(ex).__name__}: {ex}")
        time.sleep(0.4)


def report_days(sym):
    """[(report time, reaction date, previous session, session two after)] since INTRADAY_FROM."""
    dly = pd.read_parquet(EP_DIR / f"{sym}.daily.parquet")
    days = [d.date() for d in _ny(dly.index)]
    out = []
    for e in json.loads((EP_DIR / f"{sym}.earnings.json").read_text()):
        rep = pd.Timestamp(e["time"])
        rd = reaction_date(rep, days) if rep < pd.Timestamp.now(tz=rep.tz) else None
        if rd is None or str(rd) < INTRADAY_FROM:
            continue
        k = days.index(rd)
        if k < 1 or k + 2 >= len(days):
            continue
        out.append((rep, rd, days[k - 1], days[k + 2]))
    return out


def minutes(symbols):
    """5-minute bars incl. pre- and after-hours (SIP, unadjusted, like the daily bars) from the
    session before each report to two sessions after it, from Alpaca. One file per symbol."""
    import requests
    from concurrent.futures import ThreadPoolExecutor
    from dotenv import load_dotenv
    load_dotenv(ROOT / ".env", override=False)
    hdr = {"APCA-API-KEY-ID": os.environ["ALPACA_API_KEY_ID"], "APCA-API-SECRET-KEY": os.environ["ALPACA_API_SECRET_KEY"]}
    ny = "America/New_York"

    def one(sym, prev_day, last_day):
        start = pd.Timestamp(f"{prev_day} 04:00", tz=ny).tz_convert("UTC").isoformat()
        end = pd.Timestamp(f"{last_day} 20:00", tz=ny).tz_convert("UTC").isoformat()
        bars, token = [], None
        for attempt in range(8):
            params = {"timeframe": "5Min", "start": start, "end": end, "limit": 10000, "feed": "sip", "adjustment": "raw"}
            if token:
                params["page_token"] = token
            r = requests.get(ALPACA_BARS.format(sym=sym.replace(".", "/")), headers=hdr, params=params, timeout=30)
            if r.status_code == 429:
                time.sleep(3 * (attempt + 1))
                continue
            r.raise_for_status()
            j = r.json()
            bars += j.get("bars") or []
            token = j.get("next_page_token")
            if not token:
                break
        return bars

    for n, sym in enumerate(symbols, 1):
        out = EP_DIR / f"{sym}.m5.parquet"
        if out.exists():
            progress("minutes", n, len(symbols), f"{sym} cached")
            continue
        try:
            evs = report_days(sym)
        except FileNotFoundError:
            progress("minutes", n, len(symbols), f"{sym} no daily/earnings")
            continue
        rows = []
        with ThreadPoolExecutor(3) as ex:
            for bars in ex.map(lambda e: one(sym, e[2], e[3]), evs):
                rows += bars
                time.sleep(0.35)
        if rows:
            df = pd.DataFrame(rows).drop_duplicates("t")
            df.index = pd.to_datetime(df.pop("t"), utc=True)
            df = df.rename(columns={"o": "Open", "h": "High", "l": "Low", "c": "Close", "v": "Volume"})[["Open", "High", "Low", "Close", "Volume"]]
            df.sort_index().to_parquet(out)
        progress("minutes", n, len(symbols), f"{sym} reports={len(evs)} bars={len(rows)}")


def _f(x):
    return None if x is None or (isinstance(x, float) and math.isnan(x)) else float(x)


# ---------------------------------------------------------------- build

def _ny(idx):
    return idx.tz_convert("America/New_York") if idx.tz is not None else idx.tz_localize("America/New_York")


def regular_sessions(h):
    """{date: DataFrame of that day's regular-session hourly bars (09:30 .. 15:30 starts)}."""
    h = h.copy()
    h.index = _ny(h.index)
    mins = h.index.hour * 60 + h.index.minute
    reg = h[(mins >= 570) & (mins < 960)]
    return {d: g for d, g in reg.groupby(reg.index.date)}, h


def reaction_date(report: pd.Timestamp, days):
    r = report.tz_convert("America/New_York")
    d = r.date()
    if r.hour >= 16:
        later = [x for x in days if x > d]
    elif r.hour < 9 or (r.hour == 9 and r.minute < 30):
        later = [x for x in days if x >= d]
    else:                                   # during the session: the next session is the first full reaction
        later = [x for x in days if x > d]
    return later[0] if later else None


def simulate(path, open_, kind, a=None, b=None, H=1):
    """path: list of (session_index, o, h, l, c) hourly bars from the reaction-day open on."""
    bars = [p for p in path if p[0] < H]
    if not bars:
        return None
    close_h = bars[-1][4]
    if kind == "sell_open":
        lim = open_ * (1 - b / 100) if b is not None else None
        buy = next((min(lim, o) for _, o, hi, lo, c in bars if lim is not None and lo <= lim), close_h)
        return (open_ - buy) / open_ * 100
    if kind == "sell_up":
        lim = open_ * (1 + a / 100)
        i = next((k for k, (_, o, hi, lo, c) in enumerate(bars) if hi >= lim), None)
        if i is None:
            return 0.0
        sell = max(lim, bars[i][1])
        back = sell * (1 - b / 100)
        buy = next((min(back, o) for _, o, hi, lo, c in bars[i + 1:] if lo <= back), close_h)
        return (sell - buy) / open_ * 100
    if kind == "buy":
        lim = open_ * (1 - b / 100)
        buy = next((min(lim, o) for _, o, hi, lo, c in bars if lo <= lim), None)
        return 0.0 if buy is None else (close_h - buy) / open_ * 100
    raise ValueError(kind)


ACTIONS = ([("sell_open", None, b, H) for b in SELL_B for H in HORIZONS]
           + [("sell_up", a, b, H) for a in SELL_A for b in (2, 4) for H in HORIZONS]
           + [("buy", None, b, H) for b in BUY_B for H in HORIZONS])


def action_id(kind, a, b, H):
    return f"{kind}|a={a}|b={b}|H={H}"


def _num(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return None


def news_features(col, sym, rep, open_ts):
    """The platform's own news on the stock (LLM sentiment per article, quant_data enrichment):
    what was published between the report and this open, and in the three days before the report.
    Only articles stamped before the open: an 11:00 "shares slump" headline would leak the answer."""
    lo = (rep - pd.Timedelta(days=3)).tz_convert("UTC")
    hi = open_ts.tz_convert("UTC")
    docs = col.find({"symbol": sym, "date": {"$gte": lo.strftime("%Y%m%d"), "$lte": hi.strftime("%Y%m%d")}},
                    {"_id": 0, "timestamp": 1, "publishedAt": 1, "llm_sentiment_final": 1, "llm_signal_strength_a": 1,
                     "llm_event_type_b": 1})
    post, pre = [], []
    r_utc = rep.tz_convert("UTC")
    for d in docs:
        ts = d.get("timestamp") or ""
        try:
            t = pd.Timestamp(datetime.strptime(str(ts)[:14], "%Y%m%d%H%M%S")).tz_localize("UTC") if ts else pd.Timestamp(d["publishedAt"])
        except Exception:  # noqa: BLE001
            continue
        sent = _num(d.get("llm_sentiment_final"))
        if sent is None or t >= hi or t < lo:
            continue
        row = (sent, d.get("llm_signal_strength_a") == "high", d.get("llm_event_type_b") == "earnings")
        (post if t >= r_utc - pd.Timedelta(hours=1) else pre).append(row)
    f = {"news_post_n": float(len(post)), "news_pre_n": float(len(pre))}
    ps = [x[0] for x in post]
    f["news_post_mean"] = float(np.mean(ps)) if ps else np.nan
    f["news_post_min"] = float(np.min(ps)) if ps else np.nan
    f["news_post_max"] = float(np.max(ps)) if ps else np.nan
    f["news_post_neg"] = float(np.mean([x < -0.2 for x in ps])) if ps else np.nan
    strong = [x[0] for x in post if x[1]]
    f["news_post_strong"] = float(np.mean(strong)) if strong else np.nan
    earn = [x[0] for x in post if x[2]]
    f["news_post_earnings"] = float(np.mean(earn)) if earn else np.nan
    f["news_pre_mean"] = float(np.mean([x[0] for x in pre])) if pre else np.nan
    f["news_shift"] = f["news_post_mean"] - f["news_pre_mean"] if ps and pre else np.nan
    return f


NEWS_FEATS = ["news_post_n", "news_pre_n", "news_post_mean", "news_post_min", "news_post_max", "news_post_neg",
              "news_post_strong", "news_post_earnings", "news_pre_mean", "news_shift"]


def build(symbols, sectors):
    news_col = mongo()["news_articles_company_matched_v2"]
    mk_daily = pd.read_parquet(EP_DIR / f"{MARKET}.daily.parquet")
    mk_daily.index = _ny(mk_daily.index)
    mk_close = mk_daily["Close"]
    mk_open = mk_daily["Open"]
    mk_by_date = {d.date(): i for i, d in enumerate(mk_daily.index)}
    events, rows = [], []
    for n, sym in enumerate(symbols, 1):
        try:
            m5 = EP_DIR / f"{sym}.m5.parquet"            # Alpaca 5-minute bars; the yfinance hourly ones before them
            h = pd.read_parquet(m5 if m5.exists() else EP_DIR / f"{sym}.hourly.parquet")
            dly = pd.read_parquet(EP_DIR / f"{sym}.daily.parquet")
            earn = json.loads((EP_DIR / f"{sym}.earnings.json").read_text())
        except FileNotFoundError:
            continue
        dly.index = _ny(dly.index)
        ddates = [d.date() for d in dly.index]
        sess, hall = regular_sessions(h)
        sdays = sorted(sess)
        past = []                                        # this stock's earlier reactions (daily), oldest first
        for e in sorted(earn, key=lambda r: r["time"]):
            rep = pd.Timestamp(e["time"])
            if rep > pd.Timestamp.now(tz=rep.tz):
                continue
            rd = reaction_date(rep, ddates)
            if rd is None:
                continue
            k = ddates.index(rd)
            if k < 25:
                continue
            prev_close, o, c = float(dly["Close"].iloc[k - 1]), float(dly["Open"].iloc[k]), float(dly["Close"].iloc[k])
            hist = {"gap": (o / prev_close - 1) * 100, "move": (c / prev_close - 1) * 100, "fade": (c / o - 1) * 100}
            feats = None
            if rd in sess and sdays.index(rd) + 3 <= len(sdays):
                closes = dly["Close"].iloc[:k]
                rets = np.log(closes).diff().dropna()
                prev_moves = [abs(p["move"]) for p in past]
                prev_fades = [p["fade"] for p in past]
                # pre-market / after-hours path between the previous close and this open
                pd_day = ddates[k - 1]
                t0 = pd.Timestamp(datetime.combine(pd_day, datetime.min.time())).tz_localize("America/New_York") + pd.Timedelta(hours=16)
                t0 = max(t0, rep.tz_convert("America/New_York").floor("h"))     # from the report on, not the afternoon before
                t1 = pd.Timestamp(datetime.combine(rd, datetime.min.time())).tz_localize("America/New_York") + pd.Timedelta(hours=9, minutes=30)
                ext = hall[(hall.index >= t0) & (hall.index < t1)]
                mi = mk_by_date.get(rd)
                vol20 = float(rets.iloc[-20:].std() * 100)
                feats = {
                    "gap": hist["gap"],
                    "ext_high": (float(ext["High"].max()) / o - 1) * 100 if len(ext) else np.nan,
                    "ext_low": (float(ext["Low"].min()) / o - 1) * 100 if len(ext) else np.nan,
                    "ext_first": (float(ext["Close"].iloc[0]) / prev_close - 1) * 100 if len(ext) else np.nan,
                    "ext_last": (float(ext["Close"].iloc[-1]) / o - 1) * 100 if len(ext) else np.nan,
                    "surprise": np.clip(e["surprise"], -200, 200) if e["surprise"] is not None else np.nan,
                    "beat": (1.0 if e["eps"] > e["eps_est"] else 0.0) if e["eps"] is not None and e["eps_est"] is not None else np.nan,
                    "run5": (float(closes.iloc[-1]) / float(closes.iloc[-6]) - 1) * 100,
                    "run20": (float(closes.iloc[-1]) / float(closes.iloc[-21]) - 1) * 100,
                    "vol20": vol20,
                    "gap_vs_vol": hist["gap"] / vol20 if vol20 else np.nan,
                    "gap_vs_typical": hist["gap"] / np.median(prev_moves) if len(prev_moves) >= 3 and np.median(prev_moves) else np.nan,
                    "past_fade_mean": float(np.mean(prev_fades)) if len(prev_fades) >= 3 else np.nan,
                    "past_fade_same_sign": float(np.mean([f * np.sign(p["gap"]) for f, p in zip(prev_fades, past)])) if len(past) >= 3 else np.nan,
                    "past_move_med": float(np.median(prev_moves)) if len(prev_moves) >= 3 else np.nan,
                    "past_n": len(past),
                    "mkt_gap": (float(mk_open.iloc[mi]) / float(mk_close.iloc[mi - 1]) - 1) * 100 if mi else np.nan,
                    "mkt_run5": (float(mk_close.iloc[mi - 1]) / float(mk_close.iloc[mi - 6]) - 1) * 100 if mi and mi > 6 else np.nan,
                    "after_close": 1.0 if rep.tz_convert("America/New_York").hour >= 16 else 0.0,
                    "sector": sectors.get(sym, "Unknown"),
                    **news_features(news_col, sym, rep, t1),
                }
            past.append(hist)
            if feats is None:
                continue
            j = sdays.index(rd)
            path = [(s, float(r.Open), float(r.High), float(r.Low), float(r.Close))
                    for s, d in enumerate(sdays[j:j + 3]) for _, r in sess[d].iterrows()]
            open_ = path[0][1]
            ev = {"symbol": sym, "report": rep.isoformat(), "date": rd.isoformat(), **feats,
                  "day_open_close": ([p for p in path if p[0] == 0][-1][4] / open_ - 1) * 100}
            events.append(ev)
            for kind, a, b, H in ACTIONS:
                g = simulate(path, open_, kind, a, b, H)
                if g is not None:
                    rows.append({"symbol": sym, "date": rd.isoformat(), "action": action_id(kind, a, b, H), "kind": kind,
                                 "a": a if a is not None else np.nan, "b": b if b is not None else 99.0, "H": H, "gain": g})
        progress("build", n, len(symbols), f"{sym} events={sum(1 for x in events if x['symbol'] == sym)}")
    ev = pd.DataFrame(events)
    lab = pd.DataFrame(rows)
    ev.to_parquet(EP_DIR / "events.parquet")
    lab.to_parquet(EP_DIR / "labels.parquet")
    print(f"DONE build: {len(ev)} reports, {len(lab)} (report, action) rows", flush=True)


# ---------------------------------------------------------------- train

RELEASE_FEATS = ["rel_found", "rev_vs_guide", "guide_seq", "guide_width", "fy_outlook", "margin_outlook", "tone",
                 "capital_return", "restructuring"]
_CODES = {"fy_outlook": {"raised": 1.0, "initiated": 0.5, "maintained": 0.0, "lowered": -1.0, "withdrawn": -1.0},
          "margin_outlook": {"up": 1.0, "flat": 0.0, "down": -1.0},
          "capital_return": {"new_or_raised": 1.0, "none": 0.0, "cut_or_suspended": -1.0}}


def release_features(ev):
    """Join what the LLM read in each release (earnings_releases.py) onto the events; numbers from it
    become ratios here: revenue against the outlook given for the quarter, next quarter's outlook
    against this quarter's revenue."""
    path = EP_DIR / "release_features.jsonl"
    rows = []
    if path.exists():
        for r in map(json.loads, path.open()):
            if r.get("error"):
                continue
            num = lambda k: r.get(k) if isinstance(r.get(k), (int, float)) and r.get(k) > 0 else None
            rev, pl, ph, nl, nh = (num(k) for k in ("revenue", "prior_guide_low", "prior_guide_high", "next_guide_low", "next_guide_high"))
            pm = (pl + ph) / 2 if pl and ph else None
            nm = (nl + nh) / 2 if nl and nh else None
            f = {"symbol": r["symbol"], "date": r["date"], "rel_found": 1.0,
                 "rev_vs_guide": (rev / pm - 1) * 100 if rev and pm and 0.5 < rev / pm < 2 else np.nan,
                 "guide_seq": (nm / rev - 1) * 100 if rev and nm and 0.5 < nm / rev < 2 else np.nan,
                 "guide_width": (nh - nl) / nm * 100 if nm else np.nan,
                 "tone": float(r["tone"]) if isinstance(r.get("tone"), (int, float)) else np.nan,
                 "restructuring": 1.0 if r.get("restructuring") is True else 0.0}
            for k, m in _CODES.items():
                f[k] = m.get(str(r.get("full_year_outlook" if k == "fy_outlook" else k)), np.nan)
            rows.append(f)
    if not rows:
        return ev.assign(**{k: np.nan for k in RELEASE_FEATS})
    return ev.merge(pd.DataFrame(rows).drop_duplicates(["symbol", "date"]), on=["symbol", "date"], how="left").fillna({"rel_found": 0.0})


FEATS = ["gap", "ext_high", "ext_low", "ext_first", "ext_last", "surprise", "beat", "run5", "run20", "vol20", "gap_vs_vol",
         "gap_vs_typical", "past_fade_mean", "past_fade_same_sign", "past_move_med", "past_n", "mkt_gap", "mkt_run5",
         "after_close", "sector_code"] + NEWS_FEATS + RELEASE_FEATS
ACT_FEATS = ["kind_code", "a", "b", "H"]


def train(holdings):
    import lightgbm as lgb
    ev = release_features(pd.read_parquet(EP_DIR / "events.parquet"))
    lab = pd.read_parquet(EP_DIR / "labels.parquet")
    ev["sector_code"] = ev["sector"].astype("category").cat.codes
    df = lab.merge(ev, on=["symbol", "date"])
    df["kind_code"] = df["kind"].map({"sell_open": 0, "sell_up": 1, "buy": 2})
    df["q"] = pd.PeriodIndex(pd.to_datetime(df["date"]), freq="Q").astype(str)
    quarters = sorted(df["q"].unique())
    print(f"{ev.shape[0]} reports, {df['action'].nunique()} actions, quarters {quarters[0]}..{quarters[-1]}", flush=True)
    params = dict(objective="huber", alpha=3.0, learning_rate=0.03, num_leaves=15, min_data_in_leaf=60,
                  feature_fraction=0.8, bagging_fraction=0.8, bagging_freq=1, lambda_l2=5.0, verbose=-1, seed=7)
    picks = []
    sides = {"sell": ["sell_open", "sell_up"], "all": ["sell_open", "sell_up", "buy"]}
    test_q = [q for q in quarters if (df["q"] < q).sum() >= 800 * len(ACTIONS)]
    for i, q in enumerate(test_q, 1):
        tr, te = df[df["q"] < q], df[df["q"] == q]
        model = lgb.train(params, lgb.Dataset(tr[FEATS + ACT_FEATS], tr["gain"]), num_boost_round=300)
        te = te.assign(pred=model.predict(te[FEATS + ACT_FEATS]))
        static = {s: tr[tr["kind"].isin(k)].groupby("action")["gain"].mean().idxmax() for s, k in sides.items()}
        for (sym, d), g in te.groupby(["symbol", "date"]):
            for side, kinds in sides.items():
                gs = g[g["kind"].isin(kinds)]
                best = gs.loc[gs["pred"].idxmax()]
                act = best["pred"] >= MARGIN
                st_row = gs[gs["action"] == static[side]]
                picks.append({"q": q, "symbol": sym, "date": d, "side": side, "acted": bool(act),
                              "action": best["action"] if act else "hold", "pred": float(best["pred"]),
                              "gain": float(best["gain"]) if act else 0.0,
                              "static_action": static[side], "static_gain": float(st_row["gain"].iloc[0]) if len(st_row) else 0.0,
                              "oracle": float(max(gs["gain"].max(), 0.0)), "gap": float(g["gap"].iloc[0])})
        progress("train", i, len(test_q), f"{q} test reports={te[['symbol', 'date']].drop_duplicates().shape[0]}")
    pk = pd.DataFrame(picks)
    pk.to_parquet(EP_DIR / "walkforward_picks.parquet")
    report(pk, holdings)
    # final model on everything, with feature importance
    model = lgb.train(params, lgb.Dataset(df[FEATS + ACT_FEATS], df["gain"]), num_boost_round=300)
    model.save_model(str(EP_DIR / "model.txt"))
    imp = sorted(zip(FEATS + ACT_FEATS, model.feature_importance("gain")), key=lambda x: -x[1])
    print("\nfeature importance (gain):", ", ".join(f"{f} {v:.0f}" for f, v in imp[:12]))


def report(pk, holdings):
    for side in ("sell", "all"):
        p = pk[pk["side"] == side]
        acted = p[p["acted"]]
        print(f"\n== {side}-side actions ({'existing shares only' if side == 'sell' else 'with cash too'}), "
              f"out of sample, {len(p)} reports, margin {MARGIN}%")
        print(f"  policy acted on {len(acted)} ({len(acted) / max(len(p), 1):.0%}); "
              f"mean gain when acting {acted['gain'].mean():+.2f}%/share, win {(acted['gain'] > 0).mean():.0%}, "
              f"p10 {acted['gain'].quantile(0.1):+.2f}%; per report incl. holds {p['gain'].mean():+.2f}%")
        print(f"  static rule (best on past data each quarter): {p['static_gain'].mean():+.2f}%/share per report, "
              f"win {(p['static_gain'] > 0).mean():.0%}")
        print(f"  oracle (best action in hindsight): {p['oracle'].mean():+.2f}%")
        print("  by quarter:", " ".join(f"{q}:{g['gain'].mean():+.2f}({int(g['acted'].sum())})" for q, g in p.groupby("q")))
        print("  most chosen:", acted["action"].value_counts().head(5).to_dict())
        hp = p[p["symbol"].isin(holdings)]
        if len(hp):
            print("  holdings:", "; ".join(f"{r.symbol} {r.date} gap {r.gap:+.1f}% -> {r.action} {r.gain:+.2f}%" for r in hp.itertuples()))


def main():
    stage = sys.argv[1] if len(sys.argv) > 1 else "all"
    db = mongo()
    uni = {d["symbol"]: d.get("sector") or "Unknown" for d in db["stock_universe"].find({}, {"symbol": 1, "sector": 1, "_id": 0})}
    only = [s.strip().upper() for s in os.getenv("EP_SYMBOLS", "").split(",") if s.strip()]
    symbols = only or sorted(uni)
    holdings = ["INTC", "STX", "TSM", "SKHY", "WDC"]
    if stage in ("fetch", "all"):
        fetch(symbols + [MARKET])
    if stage in ("minutes", "all"):
        minutes(symbols)
    if stage in ("build", "all"):
        build(symbols, uni)
    if stage in ("train", "all"):
        train(holdings)
    return 0


if __name__ == "__main__":
    sys.exit(main())
