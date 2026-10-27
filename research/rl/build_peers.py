#!/usr/bin/env python3
"""Peer spillover decision points (PROJECT_PLAN Q.7 phase 4, 2026-10-27).

    python research/rl/build_peers.py     # -> EP_DIR/rl/peer_outcomes.parquet, peer_decisions.parquet

When a reporter releases, each of its peers as known before the report (peers.py pit) gets decision points on its own price path:
5 and 20 minutes after the reporter's release (the report evening or pre-market, extended-hours
cost) and 5 minutes into the reaction day's regular session (regular cost). The path is the peer's
1-minute bars over the reporter's window (peers.py fetch), then its 5-minute bars of the same window
(m5peer/, through the third session) and its daily bars, stitched as for a report (build_events.py);
a path without 5-minute bars on the reaction day is left out (it would jump from 10:30 to the next
day's daily bar: the 2026-10-27 bug); every action of the report grammar is simulated on it. A peer
that reports itself within a day of the reporter is left out (that is its own report, not spillover).

State: the reporter's move since its close at the decision and its range, the peer's, the peer's
volatility and run-up, the pair's return correlation, peer score, link and sector, the reporter's
session and its usual report move. eid = "<peer>|<reaction date>|<reporter>".
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "models"))
import earnings_pulse as ep  # noqa: E402
from rl import grammar, simulator  # noqa: E402
from rl.build_events import horizons, stitched  # noqa: E402

OUT = ep.D / "rl"
NY = ep.NY
PLACES = ("r+5", "r+20", "o+5")


def _m1(path):
    m = pd.read_parquet(path)
    m.index = m.index.tz_convert(NY)
    return m


def _moves(m, close_px, close_i, ts):
    """Move since the close, and its high and low since the close, up to (not including) ts, in %."""
    i = int(np.searchsorted(m.index.values, ts.to_datetime64()))
    if i <= close_i:
        return 0.0, 0.0, 0.0
    c = m["close"].values
    return ((c[i - 1] / close_px - 1) * 100, (m["high"].values[close_i:i].max() / close_px - 1) * 100,
            (m["low"].values[close_i:i].min() / close_px - 1) * 100)


def _daily(sym, cache):
    if sym not in cache:
        d = pd.read_parquet(ep.D / f"{sym}.daily.parquet")
        d.index = d.index.tz_convert(NY) if d.index.tz is not None else d.index.tz_localize(NY)
        m5p = ep.D / f"{sym}.m5.parquet"
        m = pd.read_parquet(m5p) if m5p.exists() else pd.DataFrame(columns=["Open", "High", "Low", "Close"])
        if len(m):
            m.index = m.index.tz_convert(NY)
        cache[sym] = (d, m)
    return cache[sym]


def build():
    acts = grammar.actions()
    A = {k: acts[k].values for k in ("side_code", "entry_k", "exit_code", "y", "z", "s", "horizon_code")}
    ev = ep.events()
    pe = pd.read_parquet(OUT / "peer_events.parquet").set_index(["reporter", "date"]).sort_index()
    sector = pd.read_parquet(OUT / "daily_store.parquet")[["symbol", "sector"]].drop_duplicates("symbol").set_index("symbol")["sector"]
    evd = pd.read_parquet(ep.D / "events.parquet").drop_duplicates(["symbol", "date"]).set_index(["symbol", "date"])
    reports = {s: pd.to_datetime(g["date"]).values for s, g in ev.groupby("symbol")}
    cache, rows, dec, seen, no_m5 = {}, [], [], set(), 0
    for n, r in enumerate(ev.itertuples(), 1):
        rp = ep.M1 / f"{r.symbol}_{r.day}.parquet"
        if (r.symbol, r.date) in seen:                   # a report date listed twice counts once
            continue
        seen.add((r.symbol, r.date))
        peers = (pe.loc[[(r.symbol, r.date)]] if (r.symbol, r.date) in pe.index else pe.iloc[0:0]).drop_duplicates("peer")
        if not rp.exists() or not len(peers):
            continue
        rm = _m1(rp)
        close_ts = pd.Timestamp(f"{r.day} 16:00", tz=NY)
        rpos = lambda ts: int(np.searchsorted(rm.index.values, ts.to_datetime64()))
        r_close_i = rpos(close_ts)
        if r_close_i == 0:
            continue
        _, release_ts = ep._release(rm, r_close_i, r, rpos)
        r_close = float(rm["close"].values[r_close_i - 1])
        at = {f"r+{k}": release_ts + pd.Timedelta(minutes=k) for k in (5, 20)}
        at["o+5"] = pd.Timestamp(f"{pd.Timestamp(r.date).date()} 09:35", tz=NY)
        rx = evd.loc[(r.symbol, r.date)] if (r.symbol, r.date) in evd.index else None
        for q in peers.itertuples():
            p = q.peer
            near = reports.get(p)
            if near is not None and (np.abs(near - np.datetime64(r.date)) <= np.timedelta64(1, "D")).any():
                continue
            pp = ep.M1 / f"{p}_{r.day}.parquet"
            if p == r.symbol or not pp.exists():
                continue
            daily, _ = _daily(p, cache)
            p5 = ep.D / "m5peer" / f"{p}_{r.day}.parquet"
            if not p5.exists():
                no_m5 += 1
                continue
            m5 = pd.read_parquet(p5)
            if len(m5):
                m5.index = m5.index.tz_convert(NY)
            hist = daily[daily.index.date <= pd.Timestamp(r.day).date()]["Close"]
            if len(hist) < 25:
                continue
            lr = np.log(hist).diff().iloc[-20:]
            sigma = float(lr.std() * 100)
            if not np.isfinite(sigma) or sigma <= 0:
                continue
            pm = _m1(pp)
            frame, days = stitched(r, pm, m5, daily)
            if frame is None:
                continue
            k = frame["kind"].values
            if not ((k == 5) & (frame.index.date == days[0])).any():
                no_m5 += 1
                continue
            hz = horizons(frame, days)
            t = frame.index
            pos = lambda ts: int(np.searchsorted(t.values, ts.to_datetime64()))
            p_close_i = int(np.searchsorted(pm.index.values, close_ts.to_datetime64()))
            if p_close_i == 0 or p_close_i >= len(pm):
                continue
            p_close = float(pm["close"].values[p_close_i - 1])
            o, h, l, c = frame["open"].values, frame["high"].values, frame["low"].values, frame["close"].values
            eid = f"{p}|{r.date}|{r.symbol}"
            for pl in PLACES:
                ts = at[pl]
                i = pos(ts)
                if i <= 0 or hz[1] < 0 or i >= hz[1]:
                    continue
                p0 = float(c[i - 1])
                hz_rel = np.where(hz > i, hz - i, -1).astype(np.int64)
                cost = simulator.REGULAR_COST if pl == "o+5" else simulator.COST
                g, f, cd = simulator.simulate(o[i:], h[i:], l[i:], c[i:], p0, p0, sigma, int(hz_rel[1]), hz_rel,
                                              A["side_code"], A["entry_k"], A["exit_code"], A["y"], A["z"], A["s"], A["horizon_code"], cost)
                rows.append(pd.DataFrame({"eid": eid, "place": pl, "action_id": acts["action_id"].values.astype(np.int16),
                                          "gain": g, "filled": f, "code": cd}))
                rmv, rmax, rmin = _moves(rm, r_close, r_close_i, ts)
                pmv, pmax, pmin = _moves(pm, p_close, p_close_i, ts)
                dec.append({"eid": eid, "symbol": p, "reporter": r.symbol, "date": r.date, "place": pl, "p0": p0, "sigma": sigma,
                            "bmo": float(getattr(r, "session", "amc") == "bmo"),
                            "rep_move": rmv, "rep_max": rmax, "rep_min": rmin, "peer_move": pmv, "peer_max": pmax, "peer_min": pmin,
                            "rel_move": pmv - rmv,
                            "peer_run5": float((hist.iloc[-1] / hist.iloc[-6] - 1) * 100),
                            "peer_run20": float((hist.iloc[-1] / hist.iloc[-21] - 1) * 100),
                            "rep_vol20": float(rx["vol20"]) if rx is not None else np.nan,
                            "rep_past_move_med": float(rx["past_move_med"]) if rx is not None else np.nan,
                            "corr": q.corr, "score": q.score, "link": float(q.link),
                            "same_sector": float(sector.get(p) == sector.get(r.symbol)),
                            "minutes_after_release": (ts - release_ts).total_seconds() / 60})
        if n % 250 == 0 or n == len(ev):
            ep.progress("rl-peers", n, len(ev), f"decisions={len(dec)} no_m5={no_m5}")
    out = pd.concat(rows, ignore_index=True)
    out["place"] = out["place"].astype("category")
    out["eid"] = out["eid"].astype("category")
    out.to_parquet(OUT / "peer_outcomes.parquet")
    pd.DataFrame(dec).to_parquet(OUT / "peer_decisions.parquet")
    print(f"DONE rl-peers: {len(dec)} decisions x {len(acts)} actions = {len(out):,} outcomes", flush=True)


if __name__ == "__main__":
    build()
