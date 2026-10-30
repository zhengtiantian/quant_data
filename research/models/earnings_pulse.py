#!/usr/bin/env python3
"""The after-the-close earnings pulse, learned from 1-minute bars (PROJECT_PLAN Q.5 v4).

The user's observation (2026-10-25): around an after-the-close report the stock spikes within
minutes of the release and often fades before or during the call; selling a ladder into the spike
and buying the shares back lower cuts the cost of a losing position. The user wants the system to
find when, at what levels and when to buy back by itself, per stock, and to know when not to.

    python research/models/earnings_pulse.py fetch     # 1-minute bars, report day 15:30 .. next day 10:30 ET (Alpaca)
    python research/models/earnings_pulse.py build     # every (report, plan) outcome
    python research/models/earnings_pulse.py learn     # walk-forward: per-stock choice and a conditional model

A plan, enumerated by code: when the sell ladder goes in (before the release, or k minutes after
the SEC accepted it, or 5 minutes before the call), its three levels above the price then, and when
the sold shares come back (at a time, or at a limit below each sale, else at the next open).
Outcome = what the plan made against holding, % of the price when placed, after a cost per side.
"""
from __future__ import annotations

import json
import os
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
D = Path(os.getenv("EP_DIR", "/Volumes/Data4T/earnings_policy"))
M1 = D / "m1"
NY = "America/New_York"
COST = float(os.getenv("PULSE_COST_PCT", "0.10"))      # % per side: after-hours spread and slippage
THROUGH = float(os.getenv("PULSE_FILL_THROUGH_PCT", "0.0"))   # a limit fills only when traded through by this much
PLACE = ("pre", "r+1", "r+3", "r+5", "r+10", "r+20", "r+30", "c-5")
LADDERS = ((1, 2, 3), (1.5, 3, 5), (3, 5, 8), (5, 8, 12))
BUYBACK = ("c+5", "c+30", "ah_end", "next_open", "lim1", "lim2", "lim3")


def progress(stage, done, total, extra=""):
    print(f"PROGRESS {stage} {done} {total} {extra}".rstrip(), flush=True)


# ---------------------------------------------------------------- events: release and call times

TZ = {"et": 0, "edt": 0, "est": 0, "eastern": 0, "ct": 1, "cdt": 1, "cst": 1, "central": 1,
      "mt": 2, "mdt": 2, "mst": 2, "pt": 3, "pdt": 3, "pst": 3, "pacific": 3}
TIME = re.compile(r"(\d{1,2})(?::(\d{2}))?\s*(a\.?m\.?|p\.?m\.?)\s*\(?\s*(eastern|pacific|central|e[sd]?t|p[sd]?t|c[sd]?t|m[sd]?t)\b", re.I)


def call_minutes(sym, date, session="amc"):
    """Minutes after midnight ET of the conference call, from the release text; None if not found."""
    p = D / "releases" / f"{sym}_{date}.txt"
    if not p.exists():
        return None
    t = p.read_text(errors="ignore")
    for m in re.finditer(r"conference call|earnings call|webcast|call to discuss", t, re.I):
        mm = TIME.search(t[m.start():m.start() + 400])
        if mm:
            h, mi = int(mm.group(1)), int(mm.group(2) or 0)
            h = h % 12 + (12 if mm.group(3).lower().startswith("p") else 0)
            v = h * 60 + mi + 60 * TZ.get(mm.group(4).lower().replace(".", ""), 0)
            lo, hi = (16 * 60, 19 * 60) if session == "amc" else (6 * 60, 11 * 60)
            return v if lo <= v <= hi else None
    return None


def events():
    """After-the-close reports ("amc": the evening of the report day) and, since 2026-10-26,
    before-the-open ones ("bmo": released in the reaction day's pre-market). Both windows run from
    the previous close (day 16:00) to the reaction day's 10:30; `day` is the session of that close."""
    ev = pd.read_parquet(D / "events.parquet")
    rel = {(r["symbol"], r["date"]): r for r in map(json.loads, open(D / "releases.jsonl")) if r.get("found")}
    sessions = {}
    out = []
    for r in ev.itertuples():
        rep = pd.Timestamp(r.report).tz_convert(NY)
        mins = rep.hour * 60 + rep.minute
        session = "amc" if r.after_close == 1.0 else "bmo" if mins < 570 else None
        if session is None:
            continue
        if session == "amc":
            day, react = rep.date(), pd.Timestamp(r.date).date()
            lo, hi, dflt, call_d = 15 * 60 + 55, 18 * 60, "16:02", 17 * 60
        else:
            if r.symbol not in sessions:
                sessions[r.symbol] = sorted(d.date() for d in pd.read_parquet(D / f"{r.symbol}.daily.parquet").index)
            react = pd.Timestamp(r.date).date()
            prev = [d for d in sessions[r.symbol] if d < react]
            if not prev:
                continue
            day = prev[-1]
            lo, hi, dflt, call_d = 4 * 60, 9 * 60 + 25, "07:00", 8 * 60 + 30
        rr = rel.get((r.symbol, r.date))
        acc = pd.Timestamp(rr["accepted"]).tz_localize(NY) if rr and rr.get("accepted") else None
        anchor = day if session == "amc" else react
        if acc is None or acc.date() != anchor or not (lo <= acc.hour * 60 + acc.minute <= hi):
            acc, src = pd.Timestamp(f"{anchor} {dflt}", tz=NY), "default"       # no usable SEC time: the usual minute
        else:
            src = "sec"
        call = call_minutes(r.symbol, r.date, session) or call_d
        out.append({"symbol": r.symbol, "date": r.date, "day": str(day), "session": session, "release": acc, "release_src": src,
                    "call": pd.Timestamp(f"{anchor} 00:00", tz=NY) + pd.Timedelta(minutes=call)})
    return pd.DataFrame(out)


# ---------------------------------------------------------------- fetch

def fetch():
    import requests
    from dotenv import load_dotenv
    load_dotenv(ROOT / ".env", override=False)
    hdr = {"APCA-API-KEY-ID": os.environ["ALPACA_API_KEY_ID"], "APCA-API-SECRET-KEY": os.environ["ALPACA_API_SECRET_KEY"]}
    ev = events()
    M1.mkdir(parents=True, exist_ok=True)
    todo = [r for r in ev.itertuples() if not (M1 / f"{r.symbol}_{r.day}.parquet").exists()]
    progress("fetch", 0, len(todo), f"reports={len(ev)}")

    def one(r):
        start = pd.Timestamp(f"{r.day} 15:30", tz=NY)
        nxt = pd.Timestamp(r.date)                                  # the reaction session
        end = pd.Timestamp(f"{nxt.date()} 10:30", tz=NY)
        bars, token = [], None
        for attempt in range(8):
            params = {"timeframe": "1Min", "start": start.tz_convert("UTC").isoformat(), "end": end.tz_convert("UTC").isoformat(),
                      "limit": 10000, "feed": "sip", "adjustment": "raw"}
            if token:
                params["page_token"] = token
            resp = requests.get(f"https://data.alpaca.markets/v2/stocks/{r.symbol.replace('.', '/')}/bars", headers=hdr,
                                params=params, timeout=30)
            if resp.status_code == 429:
                time.sleep(3 * (attempt + 1))
                continue
            resp.raise_for_status()
            j = resp.json()
            bars += j.get("bars") or []
            token = j.get("next_page_token")
            if not token:
                break
        if bars:
            df = pd.DataFrame(bars)
            df.index = pd.to_datetime(df.pop("t"), utc=True)
            df.rename(columns={"o": "open", "h": "high", "l": "low", "c": "close", "v": "volume"})[
                ["open", "high", "low", "close", "volume"]].to_parquet(M1 / f"{r.symbol}_{r.day}.parquet")
        time.sleep(0.9)                                             # 3 threads: under Alpaca's 200 calls a minute
        return len(bars)

    done = 0
    with ThreadPoolExecutor(3) as ex:
        for n in ex.map(one, todo):
            done += 1
            if done % 25 == 0 or done == len(todo):
                progress("fetch", done, len(todo))


# ---------------------------------------------------------------- build

def scrub_extended_bars(m: pd.DataFrame, close_i: int) -> tuple[pd.DataFrame, int]:
    """Scrub off-exchange TRF / late-closing-cross phantom needles in extended-hours 1-min SIP bars.

    Outside regular trading (09:30-16:00 ET) and past the 16:00-16:01 closing-cross transition,
    late benchmark/Form-T prints periodically report at `close16` (e.g. INTC 2026-10-29 16:15/16:16
    printing h=92.07 while trading at 75, or AAPL 2024-05-02 16:53 printing l=173.03 while at 185)
    or as isolated single-bar off-market needles without moving open/close of surrounding bars.
    """
    n = len(m)
    if n == 0:
        return m, 0
    close16 = float(m["close"].values[close_i - 1]) if 0 < close_i <= n else 0.0
    o = m["open"].values.astype(float).copy()
    h = m["high"].values.astype(float).copy()
    l = m["low"].values.astype(float).copy()
    c = m["close"].values.astype(float).copy()
    idx_ns = m.index.asi8
    mins = (m.index.hour * 60 + m.index.minute).values
    is_report_day = (m.index.date == m.index[min(max(close_i - 1, 0), n - 1)].date()) if close_i > 0 else np.zeros(n, dtype=bool)
    body_hi = np.maximum(o, c)
    body_lo = np.minimum(o, c)
    scrubbed = 0
    five_min_ns = 300 * 1_000_000_000

    for i in range(n):
        if 570 <= mins[i] < 960:
            continue
        if is_report_day[i] and mins[i] in (960, 961):
            continue
        loc_hi = body_hi[i]
        loc_lo = body_lo[i]
        if i > 0 and not (570 <= mins[i - 1] < 960) and (idx_ns[i] - idx_ns[i - 1]) <= five_min_ns:
            if body_hi[i - 1] > loc_hi:
                loc_hi = body_hi[i - 1]
            if body_lo[i - 1] < loc_lo:
                loc_lo = body_lo[i - 1]
        if i + 1 < n and not (570 <= mins[i + 1] < 960) and (idx_ns[i + 1] - idx_ns[i]) <= five_min_ns:
            if body_hi[i + 1] > loc_hi:
                loc_hi = body_hi[i + 1]
            if body_lo[i + 1] < loc_lo:
                loc_lo = body_lo[i + 1]
        loc_span_pct = (loc_hi - loc_lo) / max(loc_lo, 1e-6) * 100.0
        changed = False

        up_wick_pct = (h[i] / max(loc_hi, 1e-6) - 1.0) * 100.0
        near_c16_hi = close16 > 0 and loc_hi < close16 * 0.98 and abs(h[i] - close16) / close16 <= 0.008 and up_wick_pct >= 1.5
        isolated_hi = up_wick_pct >= 2.5 and loc_span_pct < 3.0 and up_wick_pct >= 5.0 * max(loc_span_pct, 0.20)
        if near_c16_hi or isolated_hi:
            h[i] = body_hi[i]
            changed = True

        dn_wick_pct = (1.0 - l[i] / max(loc_lo, 1e-6)) * 100.0
        near_c16_lo = close16 > 0 and loc_lo > close16 * 1.02 and abs(l[i] - close16) / close16 <= 0.008 and dn_wick_pct >= 1.5
        isolated_lo = dn_wick_pct >= 2.5 and loc_span_pct < 3.0 and dn_wick_pct >= 5.0 * max(loc_span_pct, 0.20)
        if near_c16_lo or isolated_lo:
            l[i] = body_lo[i]
            changed = True

        if changed:
            scrubbed += 1

    if scrubbed == 0:
        return m, 0
    out = m.copy()
    out["high"] = h
    out["low"] = l
    return out, scrubbed


def _release(m, close_i, r, pos):
    """The release minute: the first bar after the close (amc) or after 04:00 on the reaction day
    (bmo) with 3x the last half hour's volume and a 0.5% move from the close, else the SEC time.
    The wire beats the SEC acceptance by minutes (INTC moved at 16:00-16:01, accepted 16:09)."""
    vol, cl = m["volume"].values, m["close"].values
    if getattr(r, "session", "amc") == "amc":
        pre_med = float(np.median(vol[max(0, close_i - 30):close_i])) or 1.0
        start, span = close_i, 90
    else:
        # Pre-market volume is thin: compare with the pre-market's own bars before the release, and
        # look only around the SEC time, where the wire story lands (20 minutes before to 10 after).
        p4, sec = pos(pd.Timestamp(f"{pd.Timestamp(r.date).date()} 04:00", tz=NY)), pos(r.release)
        start = max(p4, pos(r.release - pd.Timedelta(minutes=20)))
        pre = vol[p4:start]
        pre_med = float(np.median(pre)) if len(pre) >= 5 else 1.0
        pre_med = pre_med or 1.0
        span = max(1, pos(r.release + pd.Timedelta(minutes=10)) - start)
    jump = [i for i in range(start, min(start + span, len(m))) if vol[i] >= 3 * pre_med and abs(cl[i] / cl[close_i - 1] - 1) >= 0.005]
    rel_i = min(jump[0] if jump else len(m), pos(r.release))
    return rel_i, (m.index[rel_i] if rel_i < len(m) else r.release)


def _session_end(r):
    """End of the extended session the pulse trades in: after-hours (19:55) or the pre-market (09:25)."""
    if getattr(r, "session", "amc") == "amc":
        return pd.Timestamp(f"{r.day} 19:55", tz=NY)
    return pd.Timestamp(f"{pd.Timestamp(r.date).date()} 09:25", tz=NY)


def _plan_ids():
    return [(p, lad, b) for p in PLACE for lad in LADDERS for b in BUYBACK]


def simulate(bars, t0_idx, ladder, buyback, idx):
    """bars: arrays (open, high, low, close) of 1-minute bars; t0_idx: first bar after placing the
    orders; returns the plan's gain in % of the price when placed (mean over the three parts)."""
    o, h, l, c = bars
    p0 = c[t0_idx - 1]
    end = idx["next_open"]
    g = []
    for lev in ladder:
        lim = p0 * (1 + lev / 100)
        hit = np.nonzero(h[t0_idx:end] >= lim * (1 + THROUGH / 100))[0]
        if not len(hit):
            g.append(0.0)
            continue
        i = t0_idx + hit[0]
        sell = max(lim, o[i])
        if buyback.startswith("lim"):
            back = sell * (1 - int(buyback[3:]) / 100)
            j = np.nonzero(l[i + 1:end + 1] <= back * (1 - THROUGH / 100))[0]
            buy = min(back, o[i + 1 + j[0]]) if len(j) else o[end]
        else:
            k = max(idx[buyback], i + 1)
            buy = o[min(k, end)]
        g.append((sell - buy) / p0 * 100 - 2 * COST)
    return float(np.mean(g)), int(sum(x != 0.0 for x in g))


def build():
    ev = events()
    rows, feats = [], []
    plans = _plan_ids()
    for n, r in enumerate(ev.itertuples(), 1):
        p = M1 / f"{r.symbol}_{r.day}.parquet"
        if not p.exists():
            continue
        m = pd.read_parquet(p)
        m.index = m.index.tz_convert(NY)
        day = m[m.index.date == pd.Timestamp(r.day).date()]
        nxt = m[m.index.date > pd.Timestamp(r.day).date()]
        reg_next = nxt[(nxt.index.hour * 60 + nxt.index.minute) >= 570]
        if len(day) < 30 or reg_next.empty:
            continue
        t = m.index
        pos = lambda ts: int(np.searchsorted(t.values, ts.to_datetime64()))
        close_i = pos(pd.Timestamp(f"{r.day} 16:00", tz=NY))
        if close_i == 0 or close_i >= len(m):
            continue
        if os.getenv("PULSE_NOSCRUB") != "1":
            m, _ = scrub_extended_bars(m, close_i)
        # The release reaches the wires minutes before the SEC accepts the 8-K (INTC: price jumped at
        # 16:00-16:01, SEC 16:09): the release minute is the first after the close where volume is 3x
        # the last half hour's and the price has moved 0.5%, else the SEC time.
        rel_i, release_ts = _release(m, close_i, r, pos)
        idx = {"c+5": pos(r.call + pd.Timedelta(minutes=5)), "c+30": pos(r.call + pd.Timedelta(minutes=30)),
               "ah_end": pos(_session_end(r)), "next_open": pos(reg_next.index[0])}
        place = {"pre": close_i, "c-5": pos(r.call - pd.Timedelta(minutes=5))}
        for k in (1, 3, 5, 10, 20, 30):
            place[f"r+{k}"] = pos(release_ts + pd.Timedelta(minutes=k))
        arr = (m["open"].values, m["high"].values, m["low"].values, m["close"].values)
        close16 = arr[3][close_i - 1]
        vol = m["volume"].values
        pre_vol = vol[max(0, close_i - 30):close_i].mean() or 1.0
        f = {"symbol": r.symbol, "date": r.date, "day": r.day, "bmo": 1.0 if getattr(r, "session", "amc") == "bmo" else 0.0,
             "release_min": release_ts.hour * 60 + release_ts.minute,
             "release_from_price": rel_i < pos(r.release)}
        for name, i in place.items():
            if i <= 0 or i >= idx["next_open"]:
                continue
            f[f"move_{name}"] = (arr[3][i - 1] / close16 - 1) * 100
            f[f"maxmove_{name}"] = (arr[1][close_i:i].max() / close16 - 1) * 100 if i > close_i else 0.0
            f[f"minmove_{name}"] = (arr[2][close_i:i].min() / close16 - 1) * 100 if i > close_i else 0.0
            f[f"vol_{name}"] = vol[close_i:i].sum() / pre_vol if i > close_i else 0.0
        feats.append(f)
        for pl, lad, bb in plans:
            i = place.get(pl)
            if i is None or i <= 0 or i >= idx["next_open"] or (not bb.startswith("lim") and idx[bb] <= i):
                continue
            g, nf = simulate(arr, i, lad, bb, idx)
            rows.append({"symbol": r.symbol, "date": r.date, "place": pl, "ladder": "/".join(map(str, lad)), "buyback": bb,
                         "gain": g, "fills": nf})
        if n % 100 == 0 or n == len(ev):
            progress("build", n, len(ev), f"rows={len(rows)}")
    pd.DataFrame(rows).to_parquet(D / f"pulse_labels{os.getenv('PULSE_TAG', '')}.parquet")
    pd.DataFrame(feats).to_parquet(D / "pulse_events.parquet")
    print(f"DONE build: {len(feats)} reports, {len(rows)} (report, plan) rows, {len(plans)} plans", flush=True)


# ---------------------------------------------------------------- learn

def learn():
    lab = pd.read_parquet(D / "pulse_labels.parquet")
    lab["plan"] = lab["place"] + " | " + lab["ladder"] + " | " + lab["buyback"]
    lab["year"] = lab["date"].str[:4].astype(int)
    years = sorted(lab["year"].unique())
    test_years = [y for y in years if y >= 2019]
    picks = []
    SHRINK = 8.0                        # a stock's own mean counts like this many reports of the pooled mean
    for y in test_years:
        past, now = lab[lab["year"] < y], lab[lab["year"] == y]
        pooled = past.groupby("plan")["gain"].mean()
        per = past.groupby(["symbol", "plan"])["gain"].agg(["mean", "count"]).reset_index()
        per["score"] = (per["mean"] * per["count"] + pooled.reindex(per["plan"]).values * SHRINK) / (per["count"] + SHRINK)
        best = per.loc[per.groupby("symbol")["score"].idxmax()].set_index("symbol")
        for (sym, d), g in now.groupby(["symbol", "date"]):
            if sym not in best.index:
                continue
            b = best.loc[sym]
            act = b["score"] > 0.15
            row = g[g["plan"] == b["plan"]]
            picks.append({"year": y, "symbol": sym, "date": d, "plan": b["plan"] if act else "hold",
                          "score": b["score"], "gain": float(row["gain"].iloc[0]) if act and len(row) else 0.0})
    pk = pd.DataFrame(picks)
    pk.to_parquet(D / "pulse_walkforward.parquet")
    acted = pk[pk["plan"] != "hold"]
    print(f"\nPer-stock plan chosen on earlier years only, tested on the next year ({test_years[0]}-{test_years[-1]}), cost {COST}% a side:")
    print(f"  {len(pk)} reports; acted on {len(acted)} ({len(acted) / max(len(pk), 1):.0%}); "
          f"when acting mean {acted['gain'].mean():+.2f}% win {(acted['gain'] > 0).mean():.0%} p10 {acted['gain'].quantile(.1):+.2f}% "
          f"(se {acted['gain'].std() / np.sqrt(max(len(acted), 1)):.2f})")
    print("  by year:", " ".join(f"{y}:{g['gain'].mean():+.2f}({int((g['plan'] != 'hold').sum())})" for y, g in pk.groupby("year")))
    for s in ("INTC", "STX", "AMD", "MU", "WDC", "TSM"):
        g = pk[pk["symbol"] == s]
        a = g[g["plan"] != "hold"]
        if len(g):
            print(f"  {s}: acted {len(a)}/{len(g)} mean {a['gain'].mean() if len(a) else 0:+.2f}% win {(a['gain'] > 0).mean() if len(a) else 0:.0%}; "
                  f"plan now: {pk[pk['symbol'] == s].iloc[-1]['plan']}")
    # best plans on all data, pooled and for INTC (descriptive, in-sample)
    print("\nBest plans in-sample, all stocks:")
    print(lab.groupby("plan")["gain"].agg(["mean", "count"]).sort_values("mean", ascending=False).head(8).round(3).to_string())
    print("\nBest plans in-sample, INTC:")
    print(lab[lab["symbol"] == "INTC"].groupby("plan")["gain"].agg(["mean", "count"]).sort_values("mean", ascending=False).head(8).round(3).to_string())


def model():
    """A conditional policy: LightGBM over (report, plan) rows with what is known when the orders go
    in (the move since the close and its range, the volume, EPS surprise, the outlook and tone the
    LLM read in the release, the stock's earlier reports), trained on earlier years only."""
    import lightgbm as lgb
    sys.path.insert(0, str(ROOT / "research" / "models"))
    import earnings_policy as ep
    lab = pd.read_parquet(D / f"pulse_labels{os.getenv('PULSE_TAG', '')}.parquet")
    pe = pd.read_parquet(D / "pulse_events.parquet")
    base = ep.release_features(pd.read_parquet(D / "events.parquet"))[
        ["symbol", "date", "surprise", "beat", "run5", "run20", "vol20", "past_move_med", "fy_outlook", "margin_outlook",
         "tone", "guide_seq", "rev_vs_guide", "news_pre_mean"]]
    df = lab.merge(pe, on=["symbol", "date"]).merge(base, on=["symbol", "date"], how="left")
    cols = {k: [f"{k}_{p}" for p in PLACE] for k in ("move", "maxmove", "minmove", "vol")}
    pidx = df["place"].map({p: i for i, p in enumerate(PLACE)}).values
    for k, cs in cols.items():
        mat = df[[c if c in df else "surprise" for c in cs]].values.astype(float)
        df[f"{k}_at"] = mat[np.arange(len(df)), pidx]
    df["place_code"] = pidx
    # Orders placed before the release cannot know what the release says or when it came.
    pre = df["place"] == "pre"
    for c in ("surprise", "beat", "fy_outlook", "margin_outlook", "tone", "guide_seq", "rev_vs_guide", "release_min"):
        df.loc[pre, c] = np.nan
    df["lad_lo"] = df["ladder"].str.split("/").str[0].astype(float)
    df["bb_code"] = df["buyback"].map({b: i for i, b in enumerate(BUYBACK)})
    df["year"] = df["date"].str[:4].astype(int)
    df["plan"] = df["place"] + " | " + df["ladder"] + " | " + df["buyback"]
    # the stock's own mean gain on this plan in earlier years (computed per fold below)
    F = ["move_at", "maxmove_at", "minmove_at", "vol_at", "release_min", "surprise", "beat", "run5", "run20", "vol20",
         "past_move_med", "fy_outlook", "margin_outlook", "tone", "guide_seq", "rev_vs_guide", "news_pre_mean",
         "place_code", "lad_lo", "bb_code", "own_past"]
    F = [f for f in F if f not in set(filter(None, os.getenv("PULSE_DROP", "").split(",")))]
    params = dict(objective="huber", alpha=2.0, learning_rate=0.03, num_leaves=15, min_data_in_leaf=200,
                  feature_fraction=0.8, bagging_fraction=0.8, bagging_freq=1, lambda_l2=10.0, verbose=-1, seed=7)
    margin = float(os.getenv("PULSE_MARGIN", "0.3"))
    picks = []
    years = [y for y in sorted(df["year"].unique()) if y >= 2019]
    for n, y in enumerate(years, 1):
        tr, te = df[df["year"] < y].copy(), df[df["year"] == y].copy()
        own = tr.groupby(["symbol", "plan"])["gain"].mean().rename("own_past")
        # inside training, a stock's own past must also come from before each row: use the year-wise expanding mean
        tr = tr.sort_values("date")
        tr["own_past"] = tr.groupby(["symbol", "plan"])["gain"].transform(lambda x: x.shift().expanding().mean())
        te = te.join(own, on=["symbol", "plan"])
        m = lgb.train(params, lgb.Dataset(tr[F], tr["gain"]), num_boost_round=400)
        te["pred"] = m.predict(te[F])
        for (sym, d), g in te.groupby(["symbol", "date"]):
            b = g.loc[g["pred"].idxmax()]
            act = b["pred"] >= margin
            picks.append({"year": y, "symbol": sym, "date": d, "plan": b["plan"] if act else "hold", "pred": b["pred"],
                          "gain": float(b["gain"]) if act else 0.0, "top_gain": float(b["gain"]), "top_plan": b["plan"],
                          "best_possible": float(max(g["gain"].max(), 0))})
        progress("model", n, len(years), f"{y}")
    pk = pd.DataFrame(picks)
    pk.to_parquet(D / f"pulse_model_walkforward{os.getenv('PULSE_TAG', '')}.parquet")
    a = pk[pk["plan"] != "hold"]
    print(f"\nConditional model, trained on earlier years only, margin {margin}%: {len(pk)} reports, acted on {len(a)} ({len(a) / len(pk):.0%})")
    print(f"  when acting: mean {a['gain'].mean():+.2f}% (se {a['gain'].std() / np.sqrt(max(len(a), 1)):.2f}) win {(a['gain'] > 0).mean():.0%} "
          f"p10 {a['gain'].quantile(.1):+.2f}% worst {a['gain'].min():+.1f}%  | hindsight best {pk['best_possible'].mean():+.2f}%")
    print("  by year:", " ".join(f"{y}:{g['gain'].mean():+.2f}({int((g['plan'] != 'hold').sum())})" for y, g in pk.groupby("year")))
    for s in ("INTC", "STX", "AMD", "MU", "TSM"):
        g = pk[pk["symbol"] == s]; ga = g[g["plan"] != "hold"]
        if len(g):
            print(f"  {s}: acted {len(ga)}/{len(g)} mean {ga['gain'].mean() if len(ga) else 0:+.2f}% win {(ga['gain'] > 0).mean() if len(ga) else 0:.0%}")
    imp = sorted(zip(F, m.feature_importance("gain")), key=lambda x: -x[1])
    print("  importance:", ", ".join(f"{f} {v:.0f}" for f, v in imp[:10]))


# ---------------------------------------------------------------- v2: single legs, composed by the algorithm

LEVELS = (0.5, 1, 1.5, 2, 2.5, 3, 4, 5, 6, 8, 10, 12)
MAX_LEGS = 4


def leg_gains_with_fill(bars, t0_idx, idx, level):
    """One step at `level` % above the price when placed: returns (dict of gain per buy-back rule, filled 0/1)."""
    o, h, l, c = bars
    p0, end = c[t0_idx - 1], idx["next_open"]
    lim = p0 * (1 + level / 100)
    hit = np.nonzero(h[t0_idx:end] >= lim * (1 + THROUGH / 100))[0]
    if not len(hit):
        return {bb: 0.0 for bb in BUYBACK}, 0
    i = t0_idx + hit[0]
    sell = max(lim, o[i])
    out = {}
    for bb in BUYBACK:
        if bb.startswith("lim"):
            back = sell * (1 - int(bb[3:]) / 100)
            j = np.nonzero(l[i + 1:end + 1] <= back * (1 - THROUGH / 100))[0]
            buy = min(back, o[i + 1 + j[0]]) if len(j) else o[end]
        else:
            buy = o[min(max(idx[bb], i + 1), end)]
        out[bb] = (sell - buy) / p0 * 100 - 2 * COST
    return out, 1


def leg_gains(bars, t0_idx, idx, level):
    """Backward-compatible wrapper returning only the gain map per buy-back rule."""
    return leg_gains_with_fill(bars, t0_idx, idx, level)[0]


def build_legs():
    """One row per (report, placement, level, buy-back): what that single step made and whether it filled."""
    ev = events()
    rows = []
    scrubbed_reports, scrubbed_bars = 0, 0
    for n, r in enumerate(ev.itertuples(), 1):
        p = M1 / f"{r.symbol}_{r.day}.parquet"
        if not p.exists():
            continue
        m = pd.read_parquet(p)
        m.index = m.index.tz_convert(NY)
        nxt = m[m.index.date > pd.Timestamp(r.day).date()]
        reg_next = nxt[(nxt.index.hour * 60 + nxt.index.minute) >= 570]
        if len(m[m.index.date == pd.Timestamp(r.day).date()]) < 30 or reg_next.empty:
            continue
        t = m.index
        pos = lambda ts: int(np.searchsorted(t.values, ts.to_datetime64()))
        close_i = pos(pd.Timestamp(f"{r.day} 16:00", tz=NY))
        if close_i == 0 or close_i >= len(m):
            continue
        if os.getenv("PULSE_NOSCRUB") != "1":
            m, nb = scrub_extended_bars(m, close_i)
            if nb:
                scrubbed_reports += 1
                scrubbed_bars += nb
        cl = m["close"].values
        rel_i, release_ts = _release(m, close_i, r, pos)
        idx = {"c+5": pos(r.call + pd.Timedelta(minutes=5)), "c+30": pos(r.call + pd.Timedelta(minutes=30)),
               "ah_end": pos(_session_end(r)), "next_open": pos(reg_next.index[0])}
        place = {"pre": close_i, "c-5": pos(r.call - pd.Timedelta(minutes=5))}
        for k in (1, 3, 5, 10, 20, 30):
            place[f"r+{k}"] = pos(release_ts + pd.Timedelta(minutes=k))
        arr = (m["open"].values, m["high"].values, m["low"].values, cl)
        for pl, i in place.items():
            if i <= 0 or i >= idx["next_open"]:
                continue
            for lev in LEVELS:
                gains_map, filled = leg_gains_with_fill(arr, i, idx, lev)
                for bb, g in gains_map.items():
                    if not bb.startswith("lim") and idx[bb] <= i:
                        continue
                    rows.append((r.symbol, r.date, pl, lev, bb, g, filled))
        if n % 200 == 0 or n == len(ev):
            progress("legs", n, len(ev), f"rows={len(rows)} scrubbed={scrubbed_reports}/{scrubbed_bars}")
    df = pd.DataFrame(rows, columns=["symbol", "date", "place", "level", "buyback", "gain", "filled"])
    df["gain"] = df["gain"].astype("float32")
    df["filled"] = df["filled"].astype("int8")
    df.to_parquet(D / f"pulse_legs{os.getenv('PULSE_TAG', '')}.parquet")
    print(f"DONE legs: {df[['symbol', 'date']].drop_duplicates().shape[0]} reports, {len(df)} rows "
          f"(scrubbed {scrubbed_bars} phantom bars across {scrubbed_reports} reports)", flush=True)


# ---------------------------------------------------------------- v2: the bake-off

ORDER = ("pre", "r+1", "r+3", "r+5", "r+10", "r+20", "r+30", "c-5")        # the placements in time order
STATE = ["move_at", "maxmove_at", "minmove_at", "release_min", "run5", "run20", "vol20", "past_move_med",
         "news_pre_mean", "place_code", "bmo"]
LEG_FEATS = STATE + ["level", "bb_code", "own_leg", "own_sym"]
FILL_FEATS = STATE + ["level", "own_fill"]


def leg_frame():
    tag = os.getenv("PULSE_TAG", "_strict")
    legs = pd.read_parquet(D / f"pulse_legs{tag}.parquet")
    pe = pd.read_parquet(D / "pulse_events.parquet")
    daily = pd.read_parquet(D / "events.parquet")[["symbol", "date", "run5", "run20", "vol20", "past_move_med", "news_pre_mean"]]
    st = []
    for p in ORDER:
        x = pe[["symbol", "date", "release_min", "bmo", f"move_{p}", f"maxmove_{p}", f"minmove_{p}"]].copy()
        x.columns = ["symbol", "date", "release_min", "bmo", "move_at", "maxmove_at", "minmove_at"]
        x["place"] = p
        if p == "pre":
            x["release_min"] = np.nan
        st.append(x)
    st = pd.concat(st).merge(daily, on=["symbol", "date"], how="left")
    st["place_code"] = st["place"].map({p: i for i, p in enumerate(ORDER)})
    df = legs.merge(st, on=["symbol", "date", "place"], how="inner")
    df["bb_code"] = df["buyback"].map({b: i for i, b in enumerate(BUYBACK)}).astype(float)
    df = df.sort_values("date").reset_index(drop=True)
    # what this stock's earlier reports made on the same step, and on all steps (earlier dates only)
    if "filled" not in df.columns:
        df["filled"] = (df["gain"] != 0.0).astype("int8")
    df["own_leg"] = df.groupby(["symbol", "place", "level", "buyback"])["gain"].transform(lambda x: x.shift().expanding().mean())
    f_step = df.drop_duplicates(["symbol", "date", "place", "level"]).sort_values("date").copy()
    f_step["own_fill"] = f_step.groupby(["symbol", "place", "level"])["filled"].transform(lambda x: x.shift().expanding().mean())
    df = df.merge(f_step[["symbol", "date", "place", "level", "own_fill"]], on=["symbol", "date", "place", "level"], how="left")
    per = df.groupby(["symbol", "date"])["gain"].mean().rename("g").reset_index().sort_values("date")
    per["own_sym"] = per.groupby("symbol")["g"].transform(lambda x: x.shift().expanding().mean())
    df = df.merge(per[["symbol", "date", "own_sym"]], on=["symbol", "date"], how="left")
    df["year"] = df["date"].str[:4].astype(int)
    df["eid"] = df["symbol"] + "|" + df["date"]
    return df


def compose(d, margin, weight="pred"):
    """d: rows of (eid, place, level, buyback, pred[, q10], gain). For each (eid, place), the plan the
    algorithm would give: per buy-back rule, the steps predicted above `margin` (at most MAX_LEGS, the
    best), sized by prediction (or by prediction over downside with weight='risk'); the rule whose
    plan has the highest predicted value wins. Returns one row per (eid, place)."""
    c = d[d["pred"] >= margin].copy()
    if c.empty:
        return pd.DataFrame(columns=["eid", "place", "value", "realized", "legs", "buyback"])
    c["rk"] = c.groupby(["eid", "place", "buyback"])["pred"].rank(ascending=False, method="first")
    c = c[c["rk"] <= MAX_LEGS]
    c["w"] = c["pred"] if weight == "pred" else c["pred"] / np.maximum(c["pred"] - c["q10"], 0.5)
    c["wp"], c["wg"] = c["w"] * c["pred"], c["w"] * c["gain"]
    g = c.groupby(["eid", "place", "buyback"]).agg(w=("w", "sum"), wp=("wp", "sum"), wg=("wg", "sum"), legs=("level", "count")).reset_index()
    g["value"], g["realized"] = g["wp"] / g["w"], g["wg"] / g["w"]
    return g.loc[g.groupby(["eid", "place"])["value"].idxmax(), ["eid", "place", "value", "realized", "legs", "buyback"]]


def run_policy(plans, events_, margin, cont=None):
    """Walk each evening's placements in time order. Greedy: act at the first plan worth `margin`.
    With cont (place -> predicted value of waiting, per eid): act only when the plan beats waiting."""
    out = []
    by = {k: g.set_index("place") for k, g in plans.groupby("eid")}
    for eid in events_:
        g = by.get(eid)
        res = {"eid": eid, "acted": False, "gain": 0.0, "place": None, "legs": 0}
        if g is not None:
            for p in ORDER:
                if p not in g.index:
                    continue
                r = g.loc[p]
                wait = cont.get((eid, p), -np.inf) if cont is not None else -np.inf
                if r["value"] >= margin and r["value"] >= wait:
                    res.update(acted=True, gain=float(r["realized"]), place=p, legs=int(r["legs"]), buyback=r["buyback"])
                    break
        out.append(res)
    return pd.DataFrame(out)


def bakeoff():
    """Walk-forward by year (2019-2026) of the three policies on single-leg predictions."""
    import lightgbm as lgb
    df = leg_frame()
    margin = float(os.getenv("PULSE_MARGIN", "0.3"))
    P = dict(objective="huber", alpha=2.0, learning_rate=0.05, num_leaves=31, min_data_in_leaf=500, feature_fraction=0.8,
             bagging_fraction=0.7, bagging_freq=1, lambda_l2=10.0, verbose=-1, seed=7, num_threads=8)
    Q = {**P, "objective": "quantile", "alpha": 0.1}
    R = 250
    results = {k: [] for k in ("A_greedy", "B_stopping", "C_stopping_risk")}
    years = [y for y in sorted(df["year"].unique()) if y >= 2019]
    (D / "bakeoff_preds").mkdir(exist_ok=True)
    for n, y in enumerate(years, 1):
        tr, te = df[df["year"] < y], df[df["year"] == y].copy()
        # out-of-fold predictions on the training years, for the value of waiting
        half = (pd.util.hash_pandas_object(tr["eid"], index=False) % 2).values
        oof = np.zeros(len(tr))
        for k in (0, 1):
            m = lgb.train(P, lgb.Dataset(tr.loc[half != k, LEG_FEATS], tr.loc[half != k, "gain"]), R)
            oof[half == k] = m.predict(tr.loc[half == k, LEG_FEATS])
        trp = tr.assign(pred=oof)
        m = lgb.train(P, lgb.Dataset(tr[LEG_FEATS], tr["gain"]), R)
        te["pred"] = m.predict(te[LEG_FEATS])
        mq = lgb.train(Q, lgb.Dataset(tr[LEG_FEATS], tr["gain"]), R)
        te["q10"] = mq.predict(te[LEG_FEATS])
        # B: fitted value iteration over the placements, backwards from the last one
        tplans = compose(trp, margin)
        sfeat = tr.drop_duplicates(["eid", "place"]).set_index(["eid", "place"])[STATE]
        tfeat = te.drop_duplicates(["eid", "place"]).set_index(["eid", "place"])[STATE]
        tp = tplans.set_index(["eid", "place"])
        follow = pd.Series(0.0, index=tr["eid"].unique())          # what following the policy from here on made
        cont_te = {}
        for i in range(len(ORDER) - 1, -1, -1):
            p = ORDER[i]
            ids = follow.index
            if i < len(ORDER) - 1:
                X = sfeat.reindex(pd.MultiIndex.from_product([ids, [p]])).values
                ok = ~np.isnan(X[:, 0])
                if ok.sum() > 200:
                    cm = lgb.train({**P, "min_data_in_leaf": 100}, lgb.Dataset(X[ok], follow.values[ok]), 150)
                    cw = pd.Series(cm.predict(np.nan_to_num(X, nan=np.nan)), index=ids)
                    Xt = tfeat.reindex(pd.MultiIndex.from_product([te["eid"].unique(), [p]]))
                    okt = ~Xt.iloc[:, 0].isna()
                    cont_te.update({(e, p): float(v) for (e, _), v in zip(Xt[okt].index, cm.predict(Xt[okt].values))})
                else:
                    cw = pd.Series(0.0, index=ids)
            else:
                cw = pd.Series(0.0, index=ids)
            here = tp.xs(p, level="place") if p in tp.index.get_level_values("place") else pd.DataFrame(columns=["value", "realized"])
            v = here["value"].reindex(ids)
            act = (v >= margin) & (v >= cw)
            follow = pd.Series(np.where(act, here["realized"].reindex(ids).fillna(0.0), follow.values), index=ids)
        eids = te["eid"].unique()
        te[["eid", "symbol", "year", "place", "level", "buyback", "gain", "pred", "q10"]].to_parquet(D / "bakeoff_preds" / f"{y}.parquet")
        (D / "bakeoff_preds" / f"{y}_cont.json").write_text(json.dumps({f"{e}||{p}": v for (e, p), v in cont_te.items()}))
        plans_te = compose(te, margin)
        results["A_greedy"].append(run_policy(plans_te, eids, margin).assign(year=y))
        results["B_stopping"].append(run_policy(plans_te, eids, margin, cont_te).assign(year=y))
        results["C_stopping_risk"].append(run_policy(compose(te, margin, "risk"), eids, margin, cont_te).assign(year=y))
        progress("bakeoff", n, len(years), f"{y}")
    print(f"\nWalk-forward {years[0]}-{years[-1]}, single legs composed by the policy, margin {margin}%, {COST}% a side "
          f"(labels: {os.getenv('PULSE_TAG', '_strict')}):")
    for k, parts in results.items():
        r = pd.concat(parts)
        r.to_parquet(D / f"pulse_bakeoff_{k}.parquet")
        a = r[r["acted"]]
        yr = r.groupby("year")["gain"].mean()
        intc = a[a["eid"].str.startswith("INTC|")]
        print(f"  {k:16s} acted {len(a):4d}/{len(r)} ({len(a) / len(r):.0%})  mean {a['gain'].mean():+.2f}% (se {a['gain'].std() / np.sqrt(len(a)):.2f})"
              f"  win {(a['gain'] > 0).mean():.0%}  p10 {a['gain'].quantile(.1):+.2f}  worst {a['gain'].min():+.1f}  sum {a['gain'].sum():+.0f}%"
              f"  worst year {yr.min():+.2f}  legs {a['legs'].mean():.1f}  INTC {len(intc)}x {intc['gain'].mean():+.2f}%")
        print("     placements:", a["place"].value_counts().to_dict())


# ---------------------------------------------------------------- the live policy (chosen 2026-10-26)

# Chosen on the 2019-2022 walk-forward (highest total with worst > -10% and P10 > -2%) among 160
# variants, then checked on 2023-2026 untouched. After-the-close reports only (2026-10-26 01:00):
# margin 0.3 / P10 -1.5, 395 acted, +0.86%. With the before-the-open reports added (2026-10-26
# 06:00): margin 0.5 / P10 -2.0, 407 acted, +1.40% per acted report, P10 -0.09%, every year
# positive; the 1D-CNN challenger passed no risk limit and made +0.40% under the same policy.
POLICY = {"margin": 0.5, "no_pre": True, "q10_min": -2.0, "weight": "risk", "stopping": False}
P_LEG = dict(objective="huber", alpha=2.0, learning_rate=0.05, num_leaves=31, min_data_in_leaf=500, feature_fraction=0.8,
             bagging_fraction=0.7, bagging_freq=1, lambda_l2=10.0, verbose=-1, seed=7, num_threads=8)


# The live models (2026-10-26, research/lab): the champion acts, the shadow is logged and scored
# beside it. Chosen by the lab's rule: deflated Sharpe ratio >= 0.95 over every trial so far
# (640), highest wins; each with the policy its own 2019-2022 walk-forward chose. Both chose
# margin 0.7 with pre-release orders allowed, which they used once in 371 reports: the live watch
# places nothing before the release (no_pre), at no cost (+665% vs +667%).
# Updated 2026-10-30 after scrubbing 1,319 late-reported SIP regular-close phantom bars across 844 reports:
# CatBoost failed 2019-2022 risk limits (-12.0% worst) once phantom fills were removed; 5-seed LightGBM
# passed 26/80 variants (OOS 2023-2026: 116 acted, +1.40% mean, P10 0.00%, worst -3.18%, SR/trade 0.454)
# and promotes to champion, with two-stage Hurdle LightGBM (P(fill)*E[gain|fill], conditional Q10|fill gate;
# OOS 163 acted, +1.06% mean, worst -5.08%) in shadow.
LIVE_MODELS = {
    "lgbm_seeds": {"role": "champion", "policy": {"margin": 0.5, "no_pre": True, "q10_min": -1.5, "weight": "risk"},
                   "dsr": 0.712, "preds": "lab_preds/lgbm_seeds"},
    "hurdle_lgbm": {"role": "shadow", "policy": {"margin": 0.5, "no_pre": True, "q10_min": -1.0, "weight": "risk"},
                    "dsr": 0.961, "preds": "lab_preds/hurdle_lgbm"},
}


def policy_walkforward(preds_dir: str = "bakeoff_preds", pol: dict = POLICY) -> pd.DataFrame:
    """A live policy replayed on a model's saved out-of-sample predictions."""
    out = []
    for p in sorted((D / preds_dir).glob("*.parquet")):
        d = pd.read_parquet(p)
        k = d[(d["q10"] >= pol["q10_min"]) & ((d["place"] != "pre") if pol["no_pre"] else True)]
        r = run_policy(compose(k, pol["margin"], pol["weight"]), d["eid"].unique(), pol["margin"])
        out.append(r.assign(year=int(p.stem)))
    return pd.concat(out)


class _Model:
    """The mean and the 10th percentile of a step's gain (direct or two-stage P(fill) * E[gain|fill])."""

    def __init__(self, name):
        self.name = name
        self.f = []

    def fit(self, df_or_X, y=None):
        if self.name.startswith("hurdle_"):
            df = df_or_X
            dedup = df.drop_duplicates(["eid", "place", "level"])
            Xf, yf = dedup[FILL_FEATS].astype(float), dedup["filled"].astype(int)
            hit = df[df["filled"] == 1]
            Xc, yc = hit[LEG_FEATS].astype(float), hit["gain"].astype(float)
            if self.name == "hurdle_catboost":
                from catboost import CatBoostClassifier, CatBoostRegressor
                self.f = [CatBoostClassifier(iterations=350, depth=6, learning_rate=0.08, thread_count=8, verbose=False, random_seed=7).fit(Xf, yf)]
                kw = dict(iterations=400, depth=6, learning_rate=0.08, thread_count=8, verbose=False, random_seed=7)
                self.m = [CatBoostRegressor(loss_function="Huber:delta=2.0", **kw).fit(Xc, yc)]
                self.q = [CatBoostRegressor(loss_function="Quantile:alpha=0.1", **kw).fit(Xc, yc)]
            else:
                import lightgbm as lgb
                self.f, self.m, self.q = [], [], []
                for seed in (11, 23, 37):
                    Pf = {**P_LEG, "objective": "binary", "seed": seed, "feature_fraction": 0.8, "min_data_in_leaf": 200}
                    Pm = {**P_LEG, "seed": seed, "feature_fraction": 0.7, "min_data_in_leaf": 150}
                    self.f.append(lgb.train(Pf, lgb.Dataset(Xf, yf), 250))
                    self.m.append(lgb.train(Pm, lgb.Dataset(Xc, yc), 250))
                    self.q.append(lgb.train({**Pm, "objective": "quantile", "alpha": 0.1}, lgb.Dataset(Xc, yc), 250))
            return self
        X = df_or_X[LEG_FEATS].astype(float) if y is None else df_or_X
        y = df_or_X["gain"].astype(float) if y is None else y
        if self.name == "catboost":
            from catboost import CatBoostRegressor
            kw = dict(iterations=400, depth=6, learning_rate=0.08, thread_count=8, verbose=False, random_seed=7)
            self.m = [CatBoostRegressor(loss_function="Huber:delta=2.0", **kw).fit(X, y)]
            self.q = [CatBoostRegressor(loss_function="Quantile:alpha=0.1", **kw).fit(X, y)]
        else:
            import lightgbm as lgb
            self.m, self.q = [], []
            for seed in (11, 23, 37, 51, 73):
                P = {**P_LEG, "seed": seed, "feature_fraction": 0.7, "bagging_fraction": 0.7}
                self.m.append(lgb.train(P, lgb.Dataset(X, y), 250))
                self.q.append(lgb.train({**P, "objective": "quantile", "alpha": 0.1}, lgb.Dataset(X, y), 250))
        return self

    def save(self, d: Path):
        d.mkdir(parents=True, exist_ok=True)
        for old in d.glob("*"):
            if old.is_file():
                old.unlink()
        is_cb = "catboost" in self.name
        ext = "cbm" if is_cb else "txt"
        for kind, ms in (("f", self.f), ("m", self.m), ("q", self.q)):
            for i, m in enumerate(ms):
                m.save_model(str(d / f"{kind}{i}.{ext}"))

    @classmethod
    def load(cls, name, d: Path):
        self = cls(name)
        if "catboost" in name:
            from catboost import CatBoostClassifier, CatBoostRegressor
            self.f = [CatBoostClassifier().load_model(str(p)) for p in sorted(d.glob("f*.cbm"))]
            self.m = [CatBoostRegressor().load_model(str(p)) for p in sorted(d.glob("m*.cbm"))]
            self.q = [CatBoostRegressor().load_model(str(p)) for p in sorted(d.glob("q*.cbm"))]
        else:
            import lightgbm as lgb
            self.f = [lgb.Booster(model_file=str(p)) for p in sorted(d.glob("f*.txt"))]
            self.m = [lgb.Booster(model_file=str(p)) for p in sorted(d.glob("m*.txt"))]
            self.q = [lgb.Booster(model_file=str(p)) for p in sorted(d.glob("q*.txt"))]
        return self

    def predict(self, X):
        if self.name.startswith("hurdle_"):
            Xf = X[FILL_FEATS].astype(float)
            Xc = X[LEG_FEATS].astype(float)
            if "catboost" in self.name:
                pf = np.mean([f.predict_proba(Xf)[:, 1] for f in self.f], 0)
            else:
                pf = np.mean([f.predict(Xf) for f in self.f], 0)
            mc = np.mean([m.predict(Xc) for m in self.m], 0)
            qc = np.mean([q.predict(Xc) for q in self.q], 0)
            return pf * mc, qc
        Xc = X[LEG_FEATS].astype(float) if isinstance(X, pd.DataFrame) and set(LEG_FEATS).issubset(X.columns) else X
        return np.mean([m.predict(Xc) for m in self.m], 0), np.mean([q.predict(Xc) for q in self.q], 0)


def final():
    """Each live model trained on every report and saved under pulse_model/<name>/, with each
    stock's step history and the model's out-of-sample record (overall and per stock) under its
    policy. Nothing is replaced unless every model's record is still positive."""
    df = leg_frame()
    meta = {"trained_at": pd.Timestamp.now(tz="UTC").isoformat(), "version": 3, "levels": list(LEVELS), "buyback": list(BUYBACK),
            "features": LEG_FEATS, "reports": int(df["eid"].nunique()), "models": {}}
    fitted = {}
    for n, (name, spec) in enumerate(LIVE_MODELS.items(), 1):
        wf = policy_walkforward(spec["preds"], spec["policy"])
        acted = wf[wf["acted"]]
        if len(acted) < 100 or acted["gain"].mean() <= 0:
            raise SystemExit(f"{name}: walk-forward no longer positive ({len(acted)} acted): live models kept")
        wf["symbol"] = wf["eid"].str.split("|").str[0]
        rec = {}
        for s_, g in wf.groupby("symbol"):
            a = g[g["acted"]]
            rec[s_] = {"reports": int(len(g)), "acted": int(len(a)), "mean_when_acting": round(float(a["gain"].mean()), 3) if len(a) else None,
                       "win": round(float((a["gain"] > 0).mean()), 3) if len(a) else None}
        fitted[name] = _Model(name).fit(df) if name.startswith("hurdle_") else _Model(name).fit(df[LEG_FEATS].astype(float), df["gain"].astype(float))
        meta["models"][name] = {**spec, "walkforward": {"years": f"{wf['year'].min()}-{wf['year'].max()}", "acted": int(len(acted)),
                                                        "mean_when_acting": round(float(acted["gain"].mean()), 3),
                                                        "p10": round(float(acted["gain"].quantile(.1)), 3),
                                                        "worst": round(float(acted["gain"].min()), 2)},
                                "per_symbol": rec}
        progress("final", n, len(LIVE_MODELS), name)
    for name, model in fitted.items():
        model.save(MODEL_DIR / name)
    last = df.sort_values("date").groupby(["symbol", "place", "level", "buyback"])["gain"].mean().rename("own_leg").reset_index()
    last.to_parquet(MODEL_DIR / "own_leg.parquet")
    df.drop_duplicates(["symbol", "date", "place", "level"]).groupby(["symbol", "place", "level"])["filled"].mean().rename("own_fill").reset_index().to_parquet(MODEL_DIR / "own_fill.parquet")
    df.groupby("symbol")["gain"].mean().rename("own_sym").reset_index().to_parquet(MODEL_DIR / "own_sym.parquet")
    (MODEL_DIR / "meta.json").write_text(json.dumps(meta, indent=1))
    print(f"DONE final: {meta['reports']} reports; " + "; ".join(f"{n} ({m['role']}) {m['walkforward']}" for n, m in meta["models"].items()),
          flush=True)


def _plan_for(name, spec, X, req):
    """One model's plan at this placement: its steps scored for gain and downside, composed under
    its policy, and whether it acts (the margin, and the stock's out-of-sample record)."""
    pol = spec["policy"]
    rec = spec["per_symbol"].get(req["symbol"])
    gate = bool(rec and rec.get("acted", 0) >= 5 and (rec.get("mean_when_acting") or 0) > 0)
    base = {"model": name, "role": spec["role"], "stock_record": rec, "stock_gate": gate, "margin": pol["margin"],
            "model_record": spec["walkforward"], "dsr": spec.get("dsr")}
    if pol["no_pre"] and req["place"] == "pre":
        return {**base, "act": False, "best": None}
    model = _Model.load(name, MODEL_DIR / name)
    X = X.copy()
    X["pred"], X["q10"] = model.predict(X if name.startswith("hurdle_") else X[LEG_FEATS].astype(float))
    K = X[X["q10"] >= pol["q10_min"]]
    plan = compose(K, pol["margin"], pol["weight"])
    if plan.empty:
        return {**base, "act": False, "best": None}
    bb = plan.iloc[0]["buyback"]
    legs = K[(K["buyback"] == bb) & (K["pred"] >= pol["margin"])].sort_values("pred", ascending=False).head(MAX_LEGS)
    w = legs["pred"] / np.maximum(legs["pred"] - legs["q10"], 0.5)
    legs = legs.assign(weight=w / w.sum()).sort_values("level")
    value = float(plan.iloc[0]["value"])
    return {**base, "act": bool(value >= pol["margin"] and gate), "clears_margin": value >= pol["margin"],
            "best": {"plan": f"{req['place']} | {'/'.join(f'{v:g}' for v in legs['level'])} | {bb}", "ladder": [float(v) for v in legs["level"]],
                     "weights": [round(float(v), 3) for v in legs["weight"]], "buyback": bb, "pred_pct": round(value, 3),
                     "steps": [{"level": float(r.level), "pred": round(float(r.pred), 3), "q10": round(float(r.q10), 3)} for r in legs.itertuples()]}}


def predict(req):
    """req: {symbol, place, move_at, maxmove_at, minmove_at, release_min, bmo, run5, run20, vol20,
    past_move_med, news_pre_mean}. The champion's plan at the top level (what the live watch acts
    on) and the shadow's under "shadow"."""
    meta = json.loads((MODEL_DIR / "meta.json").read_text())
    own = pd.read_parquet(MODEL_DIR / "own_leg.parquet")
    own = own[(own["symbol"] == req["symbol"]) & (own["place"] == req["place"])].set_index(["level", "buyback"])["own_leg"]
    ofill_p = MODEL_DIR / "own_fill.parquet"
    ofill = pd.Series(dtype=float)
    if ofill_p.exists():
        of_df = pd.read_parquet(ofill_p)
        ofill = of_df[(of_df["symbol"] == req["symbol"]) & (of_df["place"] == req["place"])].set_index("level")["own_fill"]
    osym = pd.read_parquet(MODEL_DIR / "own_sym.parquet").set_index("symbol")["own_sym"].get(req["symbol"], np.nan)
    rows = [{**{k: req.get(k) for k in STATE if k in req}, "place_code": ORDER.index(req["place"]), "level": float(lev),
             "bb_code": float(BUYBACK.index(bb)), "own_leg": own.get((lev, bb), np.nan), "own_fill": ofill.get(lev, np.nan), "own_sym": osym,
             "eid": "live", "place": req["place"], "buyback": bb, "gain": 0.0} for lev in LEVELS for bb in BUYBACK]
    X = pd.DataFrame(rows)
    for k in LEG_FEATS:
        if k not in X:
            X[k] = np.nan
    plans = {name: _plan_for(name, spec, X, req) for name, spec in meta["models"].items()}
    champ = next(p for p in plans.values() if p["role"] == "champion")
    shadow = next((p for p in plans.values() if p["role"] == "shadow"), None)
    return {"symbol": req["symbol"], "place": req["place"], **champ, "shadow": shadow}


# ---------------------------------------------------------------- v1 (fixed three-step ladders), kept for the research record

LIVE_FEATS = ["move_at", "maxmove_at", "minmove_at", "release_min", "run5", "run20", "vol20", "past_move_med",
              "news_pre_mean", "place_code", "lad_lo", "bb_code", "own_past"]
MODEL_DIR = D / "pulse_model"
LIVE_MARGIN = float(os.getenv("PULSE_MARGIN", "0.3"))
LADDER_OF = {"/".join(map(str, lad)): lad for lad in LADDERS}


def final_v1():
    """The v1 live model: price-only features (the walk-forward showed the release and volume features
    add nothing: +0.64% vs +0.66% per acted report), trained on every report, on the strict labels
    (0.25% a side, limits traded through). Plus each stock's own mean per plan and its out-of-sample
    record from the walk-forward, which gates the live calls."""
    import lightgbm as lgb
    tag = os.getenv("PULSE_TAG", "_strict")
    lab = pd.read_parquet(D / f"pulse_labels{tag}.parquet")
    pe = pd.read_parquet(D / "pulse_events.parquet")
    sys.path.insert(0, str(ROOT / "research" / "models"))
    import earnings_policy as ep
    base = pd.read_parquet(D / "events.parquet")[["symbol", "date", "run5", "run20", "vol20", "past_move_med", "news_pre_mean"]]
    df = lab.merge(pe, on=["symbol", "date"]).merge(base, on=["symbol", "date"], how="left")
    pidx = df["place"].map({p: i for i, p in enumerate(PLACE)}).values
    for k in ("move", "maxmove", "minmove"):
        mat = df[[f"{k}_{p}" for p in PLACE]].values.astype(float)
        df[f"{k}_at"] = mat[np.arange(len(df)), pidx]
    df["place_code"], df["lad_lo"] = pidx, df["ladder"].str.split("/").str[0].astype(float)
    df["bb_code"] = df["buyback"].map({b: i for i, b in enumerate(BUYBACK)})
    df.loc[df["place"] == "pre", "release_min"] = np.nan
    df["plan"] = df["place"] + " | " + df["ladder"] + " | " + df["buyback"]
    df = df.sort_values("date")
    df["own_past"] = df.groupby(["symbol", "plan"])["gain"].transform(lambda x: x.shift().expanding().mean())
    params = dict(objective="huber", alpha=2.0, learning_rate=0.03, num_leaves=15, min_data_in_leaf=200,
                  feature_fraction=0.8, bagging_fraction=0.8, bagging_freq=1, lambda_l2=10.0, verbose=-1, seed=7)
    m = lgb.train(params, lgb.Dataset(df[LIVE_FEATS], df["gain"]), num_boost_round=400)
    wf = pd.read_parquet(D / f"pulse_model_walkforward{tag}.parquet")
    acted = wf[wf["plan"] != "hold"]
    if len(acted) < 100 or acted["gain"].mean() <= 0:
        # The live model is only replaced by one whose own walk-forward still makes money.
        raise SystemExit(f"walk-forward no longer positive ({len(acted)} acted, {acted['gain'].mean():+.3f}%): live model kept")
    rec = {s: {"reports": int(len(g)), "acted": int((g["plan"] != "hold").sum()),
               "mean_when_acting": round(float(g.loc[g["plan"] != "hold", "gain"].mean()), 3) if (g["plan"] != "hold").any() else None,
               "win": round(float((g.loc[g["plan"] != "hold", "gain"] > 0).mean()), 3) if (g["plan"] != "hold").any() else None}
           for s, g in wf.groupby("symbol")}
    a = wf[wf["plan"] != "hold"]
    meta = {"trained_at": pd.Timestamp.now(tz="UTC").isoformat(), "reports": int(df[["symbol", "date"]].drop_duplicates().shape[0]),
            "features": LIVE_FEATS, "place": list(PLACE), "ladders": list(LADDER_OF), "buyback": list(BUYBACK), "margin": LIVE_MARGIN,
            "cost_pct_per_side": 0.25, "walkforward": {"years": f"{wf['year'].min()}-{wf['year'].max()}", "acted": int(len(a)),
                                                      "mean_when_acting": round(float(a["gain"].mean()), 3),
                                                      "win": round(float((a["gain"] > 0).mean()), 3)},
            "per_symbol": rec}
    MODEL_DIR.mkdir(exist_ok=True)
    m.save_model(str(MODEL_DIR / "model.txt"))
    df.groupby(["symbol", "plan"])["gain"].mean().rename("own_past").reset_index().to_parquet(MODEL_DIR / "own_past.parquet")
    (MODEL_DIR / "meta.json").write_text(json.dumps(meta, indent=1))
    print(f"DONE final: {meta['reports']} reports; walk-forward {meta['walkforward']}", flush=True)


def daily_context(symbol):
    """The pre-release features: run-up and volatility from daily bars, the stock's typical report
    move, and the news sentiment of the last three days (all known before the close)."""
    import yfinance as yf
    d = yf.Ticker(symbol).history(period="3mo", interval="1d", auto_adjust=False)
    c = d["Close"].dropna()
    rets = np.log(c).diff().dropna()
    out = {"run5": float((c.iloc[-1] / c.iloc[-6] - 1) * 100), "run20": float((c.iloc[-1] / c.iloc[-21] - 1) * 100),
           "vol20": float(rets.iloc[-20:].std() * 100), "close": float(c.iloc[-1])}
    ev = pd.read_parquet(D / "events.parquet")
    past = ev[ev["symbol"] == symbol].sort_values("date")
    out["past_move_med"] = float(past["past_move_med"].dropna().iloc[-1]) if past["past_move_med"].notna().any() else None
    try:
        from dotenv import load_dotenv
        from pymongo import MongoClient
        load_dotenv(ROOT / ".env", override=False)
        col = MongoClient(os.getenv("HOST_LOCAL_MONGO_URI"))["quant_data"]["news_articles_company_matched_v2"]
        since = (pd.Timestamp.now(tz="UTC") - pd.Timedelta(days=3)).strftime("%Y%m%d")
        s = [float(x["llm_sentiment_final"]) for x in col.find({"symbol": symbol, "date": {"$gte": since}}, {"llm_sentiment_final": 1})
             if x.get("llm_sentiment_final") not in (None, "")]
        out["news_pre_mean"] = float(np.mean(s)) if s else None
    except Exception:  # noqa: BLE001 - no news store: the model treats it as missing
        out["news_pre_mean"] = None
    return out


def predict_v1(req):
    """req: {symbol, place, move_at, maxmove_at, minmove_at, release_min, run5, run20, vol20,
    past_move_med, news_pre_mean}. Returns the plans placeable now ranked by predicted gain, the
    best, whether it clears the margin, and the stock's out-of-sample record."""
    import lightgbm as lgb
    m = lgb.Booster(model_file=str(MODEL_DIR / "model.txt"))
    meta = json.loads((MODEL_DIR / "meta.json").read_text())
    own = pd.read_parquet(MODEL_DIR / "own_past.parquet")
    own = own[own["symbol"] == req["symbol"]].set_index("plan")["own_past"]
    rows = []
    for lad in LADDERS:
        for bb in BUYBACK:
            plan = f"{req['place']} | {'/'.join(map(str, lad))} | {bb}"
            rows.append({**{k: req.get(k) for k in LIVE_FEATS if k in req}, "place_code": PLACE.index(req["place"]),
                         "lad_lo": float(lad[0]), "bb_code": BUYBACK.index(bb), "own_past": own.get(plan, np.nan),
                         "plan": plan, "ladder": list(lad), "buyback": bb})
    X = pd.DataFrame(rows)
    for k in LIVE_FEATS:
        if k not in X:
            X[k] = np.nan
    X["pred"] = m.predict(X[LIVE_FEATS].astype(float))
    X = X.sort_values("pred", ascending=False)
    best = X.iloc[0]
    rec = meta["per_symbol"].get(req["symbol"])
    gate = bool(rec and rec.get("acted", 0) >= 5 and (rec.get("mean_when_acting") or 0) > 0)
    return {"symbol": req["symbol"], "place": req["place"], "best": {"plan": best["plan"], "ladder": best["ladder"],
            "buyback": best["buyback"], "pred_pct": round(float(best["pred"]), 3)},
            "act": bool(best["pred"] >= meta["margin"] and gate), "clears_margin": bool(best["pred"] >= meta["margin"]),
            "stock_gate": gate, "stock_record": rec, "margin": meta["margin"], "model_record": meta["walkforward"],
            "top": [{"plan": r.plan, "pred_pct": round(float(r.pred), 3)} for r in X.head(5).itertuples()]}


def main():
    stage = sys.argv[1] if len(sys.argv) > 1 else ""
    if stage == "predict":                # JSON request on stdin, JSON answer on stdout (quant_ai's live watcher)
        req = json.load(sys.stdin)
        out = daily_context(req["symbol"]) if req.get("context_only") else predict(req)
        print(json.dumps(out, default=float))
        return 0
    {"fetch": fetch, "build": build, "learn": learn, "model": model, "final": final, "legs": build_legs, "bakeoff": bakeoff}.get(
        stage, lambda: sys.exit(__doc__))()
    return 0


if __name__ == "__main__":
    sys.exit(main())
