#!/usr/bin/env python3
"""Peer graph for earnings spillover (PROJECT_PLAN Q.7 phase 4, 2026-10-27).

    python research/rl/peers.py        # -> EP_DIR/rl/peers.parquet (reporter, peer, score, corr, link): today's graph
    python research/rl/peers.py pit    # -> EP_DIR/rl/peer_events.parquet: each report's peers as known before it
    python research/rl/peers.py fetch  # 1-minute bars of those peers over each report's window (Alpaca)

When a bellwether reports, its peers move the same night or the next morning (Lam Research's
guidance on 2026-10-21 took Tokyo Electron -7% and SK hynix -5% the next day). Peers here are
found in the data, not listed by hand: the correlation of daily returns over the two years before
(market-neutral: each stock's return less the universe mean that day), raised for the same sector,
plus a short list of supply-chain links the returns cannot show for young listings (SKHY trades in
the US only since 2026-07; its history is the Seoul listing's). Each reporter keeps its TOP peers.

For the study each report's peers come from the two years before it (pit), never from today's
graph: the pairs that moved together on the 2024-2026 reports are the ones today's correlations
pick, and the study's report years are 2023-2026.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "models"))
import earnings_pulse as ep  # noqa: E402

OUT = ep.D / "rl"
TOP = 5
LOOKBACK = 504                      # two years of sessions
SECTOR_BONUS = 0.05
# Supply-chain and product links (reporter -> peers), kept short: where the returns are too young or
# too noisy to show them.
LINKS = {
    "MU": ["SKHY", "WDC", "STX"], "SKHY": ["MU", "WDC", "STX"], "WDC": ["STX", "MU", "SKHY"], "STX": ["WDC", "MU", "SKHY"],
    "TSM": ["NVDA", "AMD", "AVGO", "AAPL", "INTC", "AMAT", "LRCX", "KLAC", "ASML"],
    "ASML": ["AMAT", "LRCX", "KLAC", "TSM", "INTC"], "LRCX": ["AMAT", "KLAC", "ASML", "MU", "SKHY"],
    "AMAT": ["LRCX", "KLAC", "ASML"], "KLAC": ["LRCX", "AMAT", "ASML"],
    "NVDA": ["AMD", "AVGO", "TSM", "SMCI", "MU", "SKHY"], "AMD": ["NVDA", "INTC", "TSM"], "INTC": ["AMD", "TSM", "NVDA"],
}


def returns() -> pd.DataFrame:
    cols = {}
    for p in sorted(ep.D.glob("*.daily.parquet")):
        sym = p.name.split(".")[0]
        if sym == "SPY":
            continue
        d = pd.read_parquet(p)
        d.index = d.index.tz_convert("America/New_York").normalize().tz_localize(None) if d.index.tz is not None else d.index
        cols[sym] = np.log(d["Close"]).diff()
    r = pd.DataFrame(cols).sort_index()
    return r.sub(r.mean(axis=1), axis=0)                   # market-neutral


def build():
    r = returns().iloc[-LOOKBACK:]
    corr = r.corr(min_periods=120)
    uni = pd.read_parquet(OUT / "daily_store.parquet")[["symbol", "sector"]].drop_duplicates("symbol").set_index("symbol")["sector"]
    rows = []
    for rep in corr.index:
        c = corr[rep].drop(rep).dropna()
        score = c + np.where(uni.reindex(c.index).values == uni.get(rep), SECTOR_BONUS, 0.0)
        link = set(LINKS.get(rep, []))
        top = list(score.sort_values(ascending=False).head(TOP).index)
        for p in list(dict.fromkeys(top + [x for x in link if x in corr.index or x == "SKHY"])):
            rows.append({"reporter": rep, "peer": p, "score": float(score.get(p, np.nan)), "corr": float(c.get(p, np.nan)),
                         "link": p in link})
    out = pd.DataFrame(rows)
    out.to_parquet(OUT / "peers.parquet")
    print(f"DONE peers: {len(out)} pairs for {out['reporter'].nunique()} reporters", flush=True)
    return out


def build_pit():
    r = returns()
    uni = pd.read_parquet(OUT / "daily_store.parquet")[["symbol", "sector"]].drop_duplicates("symbol").set_index("symbol")["sector"]
    ev = ep.events()
    rows = []
    for n, e in enumerate(ev.itertuples(), 1):
        if e.symbol not in r.columns:
            continue
        w = r[r.index < pd.Timestamp(e.day)].iloc[-LOOKBACK:]
        ok = w.columns[w.notna().sum() >= 120]
        c = w[ok].corrwith(w[e.symbol]).drop(e.symbol, errors="ignore").dropna() if e.symbol in ok else pd.Series(dtype=float)
        score = c + np.where(uni.reindex(c.index).values == uni.get(e.symbol), SECTOR_BONUS, 0.0)
        link = set(LINKS.get(e.symbol, []))
        top = list(score.sort_values(ascending=False).head(TOP).index)
        for p in dict.fromkeys(top + [x for x in link if x in r.columns]):
            rows.append({"reporter": e.symbol, "date": e.date, "day": e.day, "peer": p, "score": float(score.get(p, np.nan)),
                         "corr": float(c.get(p, np.nan)), "link": p in link})
        if n % 500 == 0 or n == len(ev):
            ep.progress("peers-pit", n, len(ev))
    out = pd.DataFrame(rows)
    out.to_parquet(OUT / "peer_events.parquet")
    print(f"DONE peer events: {len(out)} (report, peer) pairs for {out[['reporter', 'date']].drop_duplicates().shape[0]} reports", flush=True)
    return out


M5 = ep.D / "m5peer"


def _alpaca(sym, start, end, timeframe):
    """Raw SIP bars from Alpaca between two times, all pages; [] when there are none."""
    import os
    import time
    import requests
    from dotenv import load_dotenv
    load_dotenv(ep.ROOT / ".env", override=False)
    hdr = {"APCA-API-KEY-ID": os.environ["ALPACA_API_KEY_ID"], "APCA-API-SECRET-KEY": os.environ["ALPACA_API_SECRET_KEY"]}
    bars, token = [], None
    for attempt in range(8):
        params = {"timeframe": timeframe, "start": start.tz_convert("UTC").isoformat(), "end": end.tz_convert("UTC").isoformat(),
                  "limit": 10000, "feed": "sip", "adjustment": "raw"}
        if token:
            params["page_token"] = token
        try:
            resp = requests.get(f"https://data.alpaca.markets/v2/stocks/{sym.replace('.', '/')}/bars", headers=hdr, params=params, timeout=30)
        except requests.RequestException:                # the network drops now and then (VPN, Wi-Fi): wait, retry
            time.sleep(10 * (attempt + 1))
            continue
        if resp.status_code == 429:
            time.sleep(3 * (attempt + 1))
            continue
        if resp.status_code >= 400:
            break
        j = resp.json()
        bars += j.get("bars") or []
        token = j.get("next_page_token")
        if not token:
            break
    time.sleep(0.4)
    return bars


def _frame(bars, cols):
    df = pd.DataFrame(bars)
    df.index = pd.to_datetime(df.pop("t"), utc=True)
    return df.rename(columns=dict(zip("ohlcv", cols)))[list(cols)]


def _fetch_all(stage, todo, one):
    from concurrent.futures import ThreadPoolExecutor
    ep.progress(stage, 0, len(todo))
    done = 0
    with ThreadPoolExecutor(4) as ex:
        for _ in ex.map(one, todo.items()):
            done += 1
            if done % 250 == 0 or done == len(todo):
                ep.progress(stage, done, len(todo))
    print(f"DONE {stage}: {len(todo)}", flush=True)


def fetch_peer_windows():
    """1-minute bars of each reporter's peers over the reporter's window (its report evening or
    pre-market to the reaction day's 10:30), cached as m1/<PEER>_<day>.parquet: the same window
    as the peer's own report would use when it reports that day too. Then the 5-minute bars from
    there through the third session (m5peer/<PEER>_<day>.parquet): a stock's own <SYM>.m5.parquet
    covers its own reports only, and without these the peer's path jumped from the reaction day's
    10:30 to the next day's daily bar (the d1 close was the 10:30 price; found 2026-10-27)."""
    todo, todo5 = {}, {}
    for r in pd.read_parquet(OUT / "peer_events.parquet").itertuples():
        if r.peer == r.reporter:
            continue
        if not (ep.M1 / f"{r.peer}_{r.day}.parquet").exists():
            todo[(r.peer, r.day)] = r.date
        if not (M5 / f"{r.peer}_{r.day}.parquet").exists():
            todo5[(r.peer, r.day)] = r.date

    def one(item):
        (p, day), react = item
        bars = _alpaca(p, pd.Timestamp(f"{day} 15:30", tz=ep.NY), pd.Timestamp(f"{pd.Timestamp(react).date()} 10:30", tz=ep.NY), "1Min")
        if bars:
            _frame(bars, ("open", "high", "low", "close", "volume")).to_parquet(ep.M1 / f"{p}_{day}.parquet")
        return len(bars)

    def one5(item):
        (p, day), react = item
        rd = pd.Timestamp(react).date()
        # the reaction day from 10:25 through the third session: a week of calendar days covers holidays
        bars = _alpaca(p, pd.Timestamp(f"{rd} 10:25", tz=ep.NY), pd.Timestamp(f"{rd + pd.Timedelta(days=7)} 20:00", tz=ep.NY), "5Min")
        out = _frame(bars, ("Open", "High", "Low", "Close", "Volume")) if bars else pd.DataFrame(columns=["Open", "High", "Low", "Close", "Volume"])
        out.to_parquet(M5 / f"{p}_{day}.parquet")       # empty when Alpaca has none: not asked again
        return len(bars)

    M5.mkdir(exist_ok=True)
    _fetch_all("peer-fetch", todo, one)
    _fetch_all("peer-fetch-m5", todo5, one5)


if __name__ == "__main__" and len(sys.argv) > 1:
    {"pit": build_pit, "fetch": fetch_peer_windows}[sys.argv[1]]()
elif __name__ == "__main__":
    pe = build()
    held = ["INTC", "SKHY", "STX", "TSM", "GOOGL"]
    print("\nwho moves the holdings (reporters listing a holding as a peer):")
    for h in held:
        g = pe[pe["peer"] == h].sort_values("score", ascending=False)
        print(f"  {h}: " + ", ".join(f"{r.reporter}({r.corr:+.2f}{', link' if r.link else ''})" for r in g.head(8).itertuples()))
