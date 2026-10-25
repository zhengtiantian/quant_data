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
                          "gain": float(b["gain"]) if act else 0.0, "best_possible": float(max(g["gain"].max(), 0))})
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


def main():
    stage = sys.argv[1] if len(sys.argv) > 1 else ""
    {"fetch": fetch, "build": build, "learn": learn, "model": model}.get(stage, lambda: sys.exit(__doc__))()
    return 0


if __name__ == "__main__":
    sys.exit(main())
