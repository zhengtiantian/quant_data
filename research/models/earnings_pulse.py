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


def call_minutes(sym, date):
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
            return v if 16 * 60 <= v <= 19 * 60 else None
    return None


def events():
    ev = pd.read_parquet(D / "events.parquet")
    ev = ev[ev["after_close"] == 1.0]
    rel = {(r["symbol"], r["date"]): r for r in map(json.loads, open(D / "releases.jsonl")) if r.get("found")}
    out = []
    for r in ev.itertuples():
        rr = rel.get((r.symbol, r.date))
        rep = pd.Timestamp(r.report).tz_convert(NY)
        acc = pd.Timestamp(rr["accepted"]).tz_localize(NY) if rr and rr.get("accepted") else None
        if acc is None or acc.date() != rep.date() or not (15 * 60 + 55 <= acc.hour * 60 + acc.minute <= 18 * 60):
            acc = pd.Timestamp(f"{rep.date()} 16:02", tz=NY)          # no usable SEC time: the usual minute
            src = "default"
        else:
            src = "sec"
        call = call_minutes(r.symbol, r.date) or 17 * 60
        out.append({"symbol": r.symbol, "date": r.date, "day": str(rep.date()), "release": acc, "release_src": src,
                    "call": pd.Timestamp(f"{rep.date()} 00:00", tz=NY) + pd.Timedelta(minutes=call)})
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
        # The release reaches the wires minutes before the SEC accepts the 8-K (INTC: price jumped at
        # 16:00-16:01, SEC 16:09): the release minute is the first after the close where volume is 3x
        # the last half hour's and the price has moved 0.5%, else the SEC time.
        pre_med = float(np.median(m["volume"].values[max(0, close_i - 30):close_i])) or 1.0
        jump = [i for i in range(close_i, min(close_i + 90, len(m)))
                if m["volume"].values[i] >= 3 * pre_med and abs(m["close"].values[i] / m["close"].values[close_i - 1] - 1) >= 0.005]
        rel_i = min(jump[0] if jump else len(m), pos(r.release))
        release_ts = m.index[rel_i] if rel_i < len(m) else r.release
        idx = {"c+5": pos(r.call + pd.Timedelta(minutes=5)), "c+30": pos(r.call + pd.Timedelta(minutes=30)),
               "ah_end": pos(pd.Timestamp(f"{r.day} 19:55", tz=NY)), "next_open": pos(reg_next.index[0])}
        place = {"pre": close_i, "c-5": pos(r.call - pd.Timedelta(minutes=5))}
        for k in (1, 3, 5, 10, 20, 30):
            place[f"r+{k}"] = pos(release_ts + pd.Timedelta(minutes=k))
        arr = (m["open"].values, m["high"].values, m["low"].values, m["close"].values)
        close16 = arr[3][close_i - 1]
        vol = m["volume"].values
        pre_vol = vol[max(0, close_i - 30):close_i].mean() or 1.0
        f = {"symbol": r.symbol, "date": r.date, "day": r.day, "release_min": release_ts.hour * 60 + release_ts.minute,
             "release_from_price": bool(jump) and jump[0] < pos(r.release)}
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


def leg_gains(bars, t0_idx, idx, level):
    """One step at `level` % above the price when placed: the gain against holding for each buy-back
    rule, or 0 for all when the step never trades (net of COST a side when it does)."""
    o, h, l, c = bars
    p0, end = c[t0_idx - 1], idx["next_open"]
    lim = p0 * (1 + level / 100)
    hit = np.nonzero(h[t0_idx:end] >= lim * (1 + THROUGH / 100))[0]
    if not len(hit):
        return {bb: 0.0 for bb in BUYBACK}
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
    return out


def build_legs():
    """One row per (report, placement, level, buy-back): what that single step made. Any ladder is a
    weighted sum of these, so the number of steps, their levels and sizes are left to the policy."""
    ev = events()
    rows = []
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
        vol = m["volume"].values
        pre_med = float(np.median(vol[max(0, close_i - 30):close_i])) or 1.0
        cl = m["close"].values
        jump = [i for i in range(close_i, min(close_i + 90, len(m))) if vol[i] >= 3 * pre_med and abs(cl[i] / cl[close_i - 1] - 1) >= 0.005]
        rel_i = min(jump[0] if jump else len(m), pos(r.release))
        release_ts = m.index[rel_i] if rel_i < len(m) else r.release
        idx = {"c+5": pos(r.call + pd.Timedelta(minutes=5)), "c+30": pos(r.call + pd.Timedelta(minutes=30)),
               "ah_end": pos(pd.Timestamp(f"{r.day} 19:55", tz=NY)), "next_open": pos(reg_next.index[0])}
        place = {"pre": close_i, "c-5": pos(r.call - pd.Timedelta(minutes=5))}
        for k in (1, 3, 5, 10, 20, 30):
            place[f"r+{k}"] = pos(release_ts + pd.Timedelta(minutes=k))
        arr = (m["open"].values, m["high"].values, m["low"].values, cl)
        for pl, i in place.items():
            if i <= 0 or i >= idx["next_open"]:
                continue
            for lev in LEVELS:
                for bb, g in leg_gains(arr, i, idx, lev).items():
                    if not bb.startswith("lim") and idx[bb] <= i:
                        continue
                    rows.append((r.symbol, r.date, pl, lev, bb, g))
        if n % 200 == 0 or n == len(ev):
            progress("legs", n, len(ev), f"rows={len(rows)}")
    df = pd.DataFrame(rows, columns=["symbol", "date", "place", "level", "buyback", "gain"])
    df["gain"] = df["gain"].astype("float32")
    df.to_parquet(D / f"pulse_legs{os.getenv('PULSE_TAG', '')}.parquet")
    print(f"DONE legs: {df[['symbol', 'date']].drop_duplicates().shape[0]} reports, {len(df)} rows", flush=True)


# ---------------------------------------------------------------- v2: the bake-off

ORDER = ("pre", "r+1", "r+3", "r+5", "r+10", "r+20", "r+30", "c-5")        # the placements in time order
STATE = ["move_at", "maxmove_at", "minmove_at", "release_min", "run5", "run20", "vol20", "past_move_med",
         "news_pre_mean", "place_code"]
LEG_FEATS = STATE + ["level", "bb_code", "own_leg", "own_sym"]


def leg_frame():
    tag = os.getenv("PULSE_TAG", "_strict")
    legs = pd.read_parquet(D / f"pulse_legs{tag}.parquet")
    pe = pd.read_parquet(D / "pulse_events.parquet")
    daily = pd.read_parquet(D / "events.parquet")[["symbol", "date", "run5", "run20", "vol20", "past_move_med", "news_pre_mean"]]
    st = []
    for p in ORDER:
        x = pe[["symbol", "date", "release_min", f"move_{p}", f"maxmove_{p}", f"minmove_{p}"]].copy()
        x.columns = ["symbol", "date", "release_min", "move_at", "maxmove_at", "minmove_at"]
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
    df["own_leg"] = df.groupby(["symbol", "place", "level", "buyback"])["gain"].transform(lambda x: x.shift().expanding().mean())
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

# Chosen on the 2019-2022 walk-forward among 160 variants (highest total with P10 above -1%), then
# checked on 2023-2026 untouched: 395 acted, +0.86% per acted report, P10 -0.16%.
POLICY = {"margin": 0.3, "no_pre": True, "q10_min": -1.5, "weight": "risk", "stopping": False}
P_LEG = dict(objective="huber", alpha=2.0, learning_rate=0.05, num_leaves=31, min_data_in_leaf=500, feature_fraction=0.8,
             bagging_fraction=0.7, bagging_freq=1, lambda_l2=10.0, verbose=-1, seed=7, num_threads=8)


def policy_walkforward():
    """The live policy replayed on the bake-off's saved out-of-sample predictions."""
    out = []
    for p in sorted((D / "bakeoff_preds").glob("*.parquet")):
        d = pd.read_parquet(p)
        k = d[(d["q10"] >= POLICY["q10_min"]) & ((d["place"] != "pre") if POLICY["no_pre"] else True)]
        r = run_policy(compose(k, POLICY["margin"], POLICY["weight"]), d["eid"].unique(), POLICY["margin"])
        out.append(r.assign(year=int(p.stem)))
    return pd.concat(out)


def final():
    """The live models on every report: per-step gain (huber) and its 10th percentile (quantile);
    each stock's step history; the policy's out-of-sample record, overall and per stock, which
    gates the live calls. The live models are replaced only while that record stays positive."""
    import lightgbm as lgb
    df = leg_frame()
    wf = policy_walkforward()
    acted = wf[wf["acted"]]
    if len(acted) < 100 or acted["gain"].mean() <= 0:
        raise SystemExit(f"walk-forward no longer positive ({len(acted)} acted, {acted['gain'].mean():+.3f}%): live model kept")
    m = lgb.train(P_LEG, lgb.Dataset(df[LEG_FEATS], df["gain"]), 250)
    mq = lgb.train({**P_LEG, "objective": "quantile", "alpha": 0.1}, lgb.Dataset(df[LEG_FEATS], df["gain"]), 250)
    wf["symbol"] = wf["eid"].str.split("|").str[0]
    rec = {}
    for s_, g in wf.groupby("symbol"):
        a = g[g["acted"]]
        rec[s_] = {"reports": int(len(g)), "acted": int(len(a)), "mean_when_acting": round(float(a["gain"].mean()), 3) if len(a) else None,
                   "win": round(float((a["gain"] > 0).mean()), 3) if len(a) else None, "worst": round(float(a["gain"].min()), 2) if len(a) else None}
    meta = {"trained_at": pd.Timestamp.now(tz="UTC").isoformat(), "version": 2, "policy": POLICY, "levels": list(LEVELS),
            "buyback": list(BUYBACK), "features": LEG_FEATS, "reports": int(df["eid"].nunique()),
            "walkforward": {"years": f"{wf['year'].min()}-{wf['year'].max()}", "acted": int(len(acted)),
                            "mean_when_acting": round(float(acted["gain"].mean()), 3), "p10": round(float(acted["gain"].quantile(.1)), 3),
                            "worst": round(float(acted["gain"].min()), 2)},
            "per_symbol": rec}
    MODEL_DIR.mkdir(exist_ok=True)
    m.save_model(str(MODEL_DIR / "leg_model.txt"))
    mq.save_model(str(MODEL_DIR / "leg_q10.txt"))
    last = df.sort_values("date").groupby(["symbol", "place", "level", "buyback"])["gain"].mean().rename("own_leg").reset_index()
    last.to_parquet(MODEL_DIR / "own_leg.parquet")
    df.groupby("symbol")["gain"].mean().rename("own_sym").reset_index().to_parquet(MODEL_DIR / "own_sym.parquet")
    (MODEL_DIR / "meta.json").write_text(json.dumps(meta, indent=1))
    print(f"DONE final: {meta['reports']} reports; policy walk-forward {meta['walkforward']}", flush=True)


def predict(req):
    """req: {symbol, place, move_at, maxmove_at, minmove_at, release_min, run5, run20, vol20,
    past_move_med, news_pre_mean}. Every step (level x buy-back) the policy could put in now, scored
    for gain and downside; the plan composed from them (steps, levels, weights); whether to act."""
    import lightgbm as lgb
    meta = json.loads((MODEL_DIR / "meta.json").read_text())
    pol = meta["policy"]
    rec = meta["per_symbol"].get(req["symbol"])
    gate = bool(rec and rec.get("acted", 0) >= 5 and (rec.get("mean_when_acting") or 0) > 0)
    base = {"symbol": req["symbol"], "place": req["place"], "stock_record": rec, "stock_gate": gate,
            "margin": pol["margin"], "model_record": meta["walkforward"]}
    if pol["no_pre"] and req["place"] == "pre":
        return {**base, "act": False, "best": None, "reason": "the policy does not place orders before the release"}
    m, mq = lgb.Booster(model_file=str(MODEL_DIR / "leg_model.txt")), lgb.Booster(model_file=str(MODEL_DIR / "leg_q10.txt"))
    own = pd.read_parquet(MODEL_DIR / "own_leg.parquet")
    own = own[(own["symbol"] == req["symbol"]) & (own["place"] == req["place"])].set_index(["level", "buyback"])["own_leg"]
    osym = pd.read_parquet(MODEL_DIR / "own_sym.parquet").set_index("symbol")["own_sym"].get(req["symbol"], np.nan)
    rows = [{**{k: req.get(k) for k in STATE if k in req}, "place_code": ORDER.index(req["place"]), "level": float(lev),
             "bb_code": float(BUYBACK.index(bb)), "own_leg": own.get((lev, bb), np.nan), "own_sym": osym,
             "eid": "live", "place": req["place"], "buyback": bb, "gain": 0.0} for lev in LEVELS for bb in BUYBACK]
    X = pd.DataFrame(rows)
    for k in LEG_FEATS:
        if k not in X:
            X[k] = np.nan
    X["pred"] = m.predict(X[LEG_FEATS].astype(float))
    X["q10"] = mq.predict(X[LEG_FEATS].astype(float))
    K = X[X["q10"] >= pol["q10_min"]]
    plan = compose(K, pol["margin"], pol["weight"])
    if plan.empty:
        top = X.sort_values("pred", ascending=False).head(3)
        return {**base, "act": False, "best": None, "best_step": [{"level": r.level, "buyback": r.buyback, "pred": round(r.pred, 3),
                                                                     "q10": round(r.q10, 3)} for r in top.itertuples()]}
    bb = plan.iloc[0]["buyback"]
    legs = K[(K["buyback"] == bb) & (K["pred"] >= pol["margin"])].sort_values("pred", ascending=False).head(MAX_LEGS)
    w = legs["pred"] / np.maximum(legs["pred"] - legs["q10"], 0.5)
    legs = legs.assign(weight=w / w.sum()).sort_values("level")
    value = float(plan.iloc[0]["value"])
    return {**base, "act": bool(value >= pol["margin"] and gate), "clears_margin": value >= pol["margin"],
            "best": {"plan": f"{req['place']} | {'/'.join(f'{v:g}' for v in legs['level'])} | {bb}", "ladder": [float(v) for v in legs["level"]],
                     "weights": [round(float(v), 3) for v in legs["weight"]], "buyback": bb, "pred_pct": round(value, 3),
                     "steps": [{"level": float(r.level), "pred": round(float(r.pred), 3), "q10": round(float(r.q10), 3)} for r in legs.itertuples()]}}


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
