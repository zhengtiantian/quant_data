"""A wider, point-in-time universe (PROJECT_PLAN Q.14 item 3 / Q.16.10, 2026-11-02).

The research so far ran on 102 hand-picked large caps chosen today, so every base rate ("73% recover
within a year", "holding beat managing") was measured on survivors. This builds the universe the user
can actually face: every company that was in the S&P 500 at any time since 2016 (747, of which 244 have
left), plus today's Nasdaq-100 and the original 102 (TSM, ASML, SKHY ... are not in the S&P 500), with
the dates each was a member, so a study on day D uses only the companies in the index on day D.

  members   S&P 500 snapshots (github.com/fja05680/sp500, "Historical Components & Changes") ->
            membership.parquet: one row per ticker and interval of membership
  fetch     daily bars and report dates per ticker from yfinance, by earnings_policy.fetch, into
            /Volumes/Data4T/universe/bars (delisted tickers often have no bars there: counted, not hidden)

Usage: .venv311/bin/python research/universe/build_universe.py members|fetch|status
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
UNI = Path(os.getenv("UNIVERSE_DIR", "/Volumes/Data4T/universe"))
BARS = UNI / "bars"
SINCE = "2016-01-01"
OLD_EP = Path("/Volumes/Data4T/earnings_policy")
# Tickers whose yfinance bars belong to another listing: kept out of the studies (moved to bars/excluded),
# counted as members without bars like the delisted ones. Found 2026-11-03 by the >50% daily-move scan.
BAD_BARS = {"PARA": "yfinance's PARA is Parabolic (prices ~100,000 from 2021-02), not Paramount; Paramount's "
                    "history (CBS / VIAC / PARA, merged into PSKY 2025-08) is not available there"}


def members() -> pd.DataFrame:
    c = pd.read_csv(UNI / "sp500_components.csv")
    c["date"] = pd.to_datetime(c["date"])
    c = c.sort_values("date")
    rows, open_ = [], {}
    for d, tick in zip(c.date, c.tickers):
        now = set(tick.split(","))
        for t in now - set(open_):
            open_[t] = d
        for t in set(open_) - now:
            rows.append({"ticker": t, "start": open_.pop(t), "end": d})
    for t, s in open_.items():
        rows.append({"ticker": t, "start": s, "end": pd.NaT})
    m = pd.DataFrame(rows)
    m = m[m.end.isna() | (m.end >= SINCE)].copy()
    m["index"] = "SP500"
    extra = sorted({p.name.split(".")[0] for p in OLD_EP.glob("*.daily.parquet")} - set(m.ticker) - {"SPY"})
    try:                                           # today's Nasdaq-100 members not in the S&P 500 (Nasdaq's own list)
        req = __import__("urllib.request").request.Request("https://api.nasdaq.com/api/quote/list-type/nasdaq100", headers={
            "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/126.0 Safari/537.36",
            "Accept": "application/json"})
        rows = ((json.load(__import__("urllib.request").request.urlopen(req, timeout=20)).get("data") or {}).get("data") or {}).get("rows") or []
        extra = sorted(set(extra) | ({r["symbol"] for r in rows} - set(m.ticker)))
    except Exception as e:  # noqa: BLE001
        print(f"Nasdaq-100 list unavailable ({type(e).__name__}); using the original universe's extras only")
    # names outside the S&P 500 count as members from their first bar: a known bias, kept small
    m = pd.concat([m, pd.DataFrame({"ticker": extra, "start": pd.Timestamp(SINCE), "end": pd.NaT, "index": "extra"})], ignore_index=True)
    UNI.mkdir(parents=True, exist_ok=True)
    m.to_parquet(UNI / "membership.parquet")
    left = m[(m["index"] == "SP500") & m.end.notna()].ticker.nunique()
    print(f"membership: {m.ticker.nunique()} tickers ({left} left the S&P 500 since {SINCE}; {len(extra)} outside it)")
    return m


def yf_symbol(t: str) -> str:
    return t.replace(".", "-")


def fetch() -> None:
    os.environ["EP_DIR"] = str(BARS)
    sys.path.insert(0, str(ROOT / "research" / "models"))
    import earnings_policy as ep
    ep.EP_DIR = BARS
    m = pd.read_parquet(UNI / "membership.parquet")
    syms = sorted(({yf_symbol(t) for t in m.ticker} | {"SPY"}) - set(BAD_BARS))
    ep.fetch(syms)
    exclude()
    status()


def exclude() -> None:
    """Move the bars of BAD_BARS out of the folder the studies read (they glob bars/*.daily.parquet)."""
    (BARS / "excluded").mkdir(exist_ok=True)
    for sym, why in BAD_BARS.items():
        for p in BARS.glob(f"{sym}.*"):
            p.rename(BARS / "excluded" / p.name)
            print(f"excluded {p.name}: {why}")


def status() -> dict:
    m = pd.read_parquet(UNI / "membership.parquet")
    have = {p.name.split(".")[0] for p in BARS.glob("*.daily.parquet")}
    m["yf"] = m.ticker.map(yf_symbol)
    left = m[(m["index"] == "SP500") & m.end.notna()]
    out = {"tickers": int(m.ticker.nunique()), "with_bars": int(m.yf.isin(have).groupby(m.ticker).any().sum()),
           "left_with_bars": int(left.yf.isin(have).groupby(left.ticker).any().sum()), "left": int(left.ticker.nunique())}
    print(json.dumps(out))
    return out


if __name__ == "__main__":
    {"members": members, "fetch": fetch, "exclude": exclude, "status": status}[sys.argv[1] if len(sys.argv) > 1 else "status"]()
