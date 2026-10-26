#!/usr/bin/env python3
"""Every grammar action at every decision point of every past report (PROJECT_PLAN Q.7, 2026-10-26).

    python research/rl/build_events.py          # -> EP_DIR/rl/outcomes.parquet, decisions.parquet, actions.parquet

A report's decision points are the earnings-pulse placements after the release (1, 5 and 20
minutes after it, and 5 minutes before the call). Each gets one price path, stitched from the
report's 1-minute bars (to the reaction day's 10:30), the 5-minute bars (the rest of that day and
the two sessions after) and the daily bars (to trading day 20), and every action of the grammar is
simulated on it (simulator.py). sigma is the stock's vol20 (daily % volatility, known before the
report). Outcomes: one row per (report, placement, action): gain %, entry filled, exit code.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "models"))
import earnings_pulse as ep  # noqa: E402
from rl import grammar, simulator  # noqa: E402

OUT = ep.D / "rl"
PLACES = ("r+1", "r+5", "r+20", "c-5")
NY = ep.NY


def _bars(df):
    return df["open"].values, df["high"].values, df["low"].values, df["close"].values


def stitched(r, m1, m5, daily):
    """One frame (t, open, high, low, close, kind) from the start of the 1-minute window to day 20,
    and the trading days from the reaction day on."""
    rd = pd.Timestamp(r.date).date()
    days = [d for d in daily.index.date if d >= rd]
    if len(days) < 3:
        return None, days
    a = m1[["open", "high", "low", "close"]].assign(kind=1)
    if len(m5):
        b = m5[(m5.index > a.index[-1]) & (m5.index.date <= days[2])]
        b = b.rename(columns=str.lower)[["open", "high", "low", "close"]].assign(kind=5)
    else:                                                # no 5-minute file: the daily bars carry on from here
        b = a.iloc[0:0]
    last_day = (b.index[-1] if len(b) else a.index[-1]).date()
    c = daily[(daily.index.date > last_day)].rename(columns=str.lower)[["open", "high", "low", "close"]].head(20).assign(kind=0)
    # yfinance's daily bars are split-adjusted, the Alpaca minute bars are raw (AAPL 2016: ~$24 against
    # ~$99 before the 2020 4:1 split). Put the daily bars on the minute bars' scale through the last
    # day both have: its regular-session close, else the reaction day's open.
    ratio = None
    if len(b):
        reg = b[(b.index.date == last_day) & ((b.index.hour * 60 + b.index.minute) < 960) & ((b.index.hour * 60 + b.index.minute) >= 570)]
        drow = daily[daily.index.date == last_day]
        if len(reg) and len(drow):
            ratio = float(reg["close"].iloc[-1]) / float(drow["Close"].iloc[0])
    if ratio is None:
        op = a[(a.index.date == rd) & ((a.index.hour * 60 + a.index.minute) >= 570)]
        drow = daily[daily.index.date == rd]
        if len(op) and len(drow):
            ratio = float(op["open"].iloc[0]) / float(drow["Open"].iloc[0])
    if ratio is None or not (0.05 < ratio < 20):
        c = c.iloc[0:0]                                   # cannot be put on one scale: no daily part
    else:
        c[["open", "high", "low", "close"]] = c[["open", "high", "low", "close"]] * ratio
    return pd.concat([a, b, c]), days


def horizons(frame, days):
    """Bar index of each horizon end in the stitched frame (-1 when beyond the data)."""
    t = frame.index
    mins = t.hour * 60 + t.minute
    regular = (frame["kind"].values == 0) | ((mins >= 570) & (mins < 960))
    def last_regular_on(d):
        idx = np.nonzero((t.date == d) & regular)[0]
        return int(idx[-1]) if len(idx) else -1
    first_open = np.nonzero((t.date == days[0]) & (mins >= 570) & (frame["kind"].values != 0))[0]
    out = {"next_open": int(first_open[0]) if len(first_open) else -1, "d1": last_regular_on(days[0]),
           "d3": last_regular_on(days[2]) if len(days) > 2 else -1}
    for name, k in (("d5", 4), ("d10", 9), ("d20", 19)):
        out[name] = last_regular_on(days[k]) if len(days) > k else -1
    return np.array([out[h] for h in grammar.HORIZONS], np.int64)


def build():
    acts = grammar.actions()
    A = {k: acts[k].values for k in ("side_code", "entry_k", "exit_code", "y", "z", "s", "horizon_code")}
    ev = ep.events()
    if os.getenv("RL_LIMIT"):
        ev = ev.head(int(os.environ["RL_LIMIT"]))
    vol = pd.read_parquet(ep.D / "events.parquet").set_index(["symbol", "date"])["vol20"]
    vol = vol[~vol.index.duplicated()]                   # a report date listed twice (yfinance) counts once
    daily_cache, m5_cache = {}, {}
    rows, dec = [], []
    for n, r in enumerate(ev.itertuples(), 1):
        p1 = ep.M1 / f"{r.symbol}_{r.day}.parquet"
        sigma = vol.get((r.symbol, r.date))
        if not p1.exists() or sigma is None or not np.isfinite(sigma) or sigma <= 0:
            continue
        if r.symbol not in daily_cache:
            d = pd.read_parquet(ep.D / f"{r.symbol}.daily.parquet")
            d.index = d.index.tz_convert(NY) if d.index.tz is not None else d.index.tz_localize(NY)
            daily_cache[r.symbol] = d
            m5p = ep.D / f"{r.symbol}.m5.parquet"
            m = pd.read_parquet(m5p) if m5p.exists() else pd.DataFrame(columns=["Open", "High", "Low", "Close"])
            if len(m):
                m.index = m.index.tz_convert(NY)
            m5_cache[r.symbol] = m
        m1 = pd.read_parquet(p1)
        m1.index = m1.index.tz_convert(NY)
        frame, days = stitched(r, m1, m5_cache[r.symbol], daily_cache[r.symbol])
        if frame is None:
            continue
        hz = horizons(frame, days)
        t = frame.index
        pos = lambda ts: int(np.searchsorted(t.values, ts.to_datetime64()))
        close_i = pos(pd.Timestamp(f"{r.day} 16:00", tz=NY))
        if close_i == 0 or close_i >= len(frame):
            continue
        sub = m1
        _, release_ts = ep._release(sub, int(np.searchsorted(sub.index.values, pd.Timestamp(f"{r.day} 16:00", tz=NY).to_datetime64())),
                                    r, lambda ts: int(np.searchsorted(sub.index.values, ts.to_datetime64())))
        place = {"c-5": pos(r.call - pd.Timedelta(minutes=5))}
        for k in (1, 5, 20):
            place[f"r+{k}"] = pos(release_ts + pd.Timedelta(minutes=k))
        o, h, l, c = _bars(frame)
        for pl in PLACES:
            i = place.get(pl)
            if i is None or i <= 0 or hz[1] < 0 or i >= hz[1]:
                continue
            p0 = float(c[i - 1])
            hz_rel = np.where(hz > i, hz - i, -1).astype(np.int64)
            g, f, cd = simulator.simulate(o[i:], h[i:], l[i:], c[i:], p0, p0, float(sigma), int(hz_rel[1]), hz_rel,
                                          A["side_code"], A["entry_k"], A["exit_code"], A["y"], A["z"], A["s"], A["horizon_code"])
            eid = f"{r.symbol}|{r.date}"
            rows.append(pd.DataFrame({"eid": eid, "place": pl, "action_id": acts["action_id"].values, "gain": g, "filled": f, "code": cd}))
            dec.append({"eid": eid, "symbol": r.symbol, "date": r.date, "place": pl, "p0": p0, "sigma": float(sigma),
                        "session": getattr(r, "session", "amc"), "bars": len(frame) - i})
        if n % 250 == 0 or n == len(ev):
            ep.progress("rl-build", n, len(ev), f"decisions={len(dec)}")
    OUT.mkdir(exist_ok=True)
    out = pd.concat(rows, ignore_index=True)
    out["place"] = out["place"].astype("category")
    out.to_parquet(OUT / "outcomes.parquet")
    pd.DataFrame(dec).to_parquet(OUT / "decisions.parquet")
    acts.to_parquet(OUT / "actions.parquet")
    print(f"DONE rl-build: {len(dec)} decisions x {len(acts)} actions = {len(out):,} outcomes", flush=True)


if __name__ == "__main__":
    build()
