"""Daily snapshot of the option market and analyst ratings (PROJECT_PLAN Q.16.9, 2026-11-02).

The research found the market's own view of what is coming in two places our data lacked:
option prices (the implied-volatility skew, Xing-Zhang-Zhao 2010; put-call volume, Pan-Poteshman 2006)
and analysts' ratings. Free history exists for neither, so this keeps one snapshot a day from now on,
for the 102 research stocks and the holdings:

  options   Nasdaq's option chain for the first expiry at least 5 days out and the first at least 25:
            implied volatility (Black-Scholes, from the bid/ask mid) at the money and for the puts and
            calls nearest 10% out of the money, the skew (put IV - call IV), the at-the-money straddle as
            a % of the price, put / call volume and open interest
  analysts  Finnhub's monthly rating counts (strong buy .. strong sell; the free tier gives the history)
            and the last reported quarters' surprises

Written to /Volumes/Data4T/market_snapshots/{options,analysts}/YYYY-MM-DD.parquet.

Usage: .venv311/bin/python collectors/market/options_analyst_snapshot.py [SYMBOLS...]
"""

from __future__ import annotations

import json
import math
import os
import sys
import time
import urllib.request
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(os.getenv("SNAPSHOT_DIR", "/Volumes/Data4T/market_snapshots"))
EP_DIR = Path("/Volumes/Data4T/earnings_policy")
EXTRA = ["SKHY", "QQQ", "SPY"]
UA = {"User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/126.0 Safari/537.36",
      "Accept": "application/json"}
CHAIN = ("https://api.nasdaq.com/api/quote/{sym}/option-chain?assetclass={cls}&limit=1000&fromdate={start}&todate={end}"
         "&excode=oprac&callput=callput&money=all&type=all")
QUOTE = "https://api.nasdaq.com/api/quote/{sym}/info?assetclass={cls}"
RATE = 0.04


def _get(url: str) -> dict:
    req = urllib.request.Request(url, headers=UA)
    with urllib.request.urlopen(req, timeout=20) as r:
        return json.load(r)


def _num(x) -> float | None:
    try:
        v = float(str(x).replace(",", "").replace("$", ""))
        return v if math.isfinite(v) else None
    except (TypeError, ValueError):
        return None


def _bs(s: float, k: float, t: float, sigma: float, call: bool) -> float:
    if sigma <= 0 or t <= 0:
        return max(0.0, (s - k) if call else (k - s))
    d1 = (math.log(s / k) + (RATE + sigma * sigma / 2) * t) / (sigma * math.sqrt(t))
    d2 = d1 - sigma * math.sqrt(t)
    n = lambda x: 0.5 * (1 + math.erf(x / math.sqrt(2)))
    return s * n(d1) - k * math.exp(-RATE * t) * n(d2) if call else k * math.exp(-RATE * t) * n(-d2) - s * n(-d1)


def implied_vol(price: float, s: float, k: float, t: float, call: bool) -> float | None:
    if not price or price <= 0 or t <= 0:
        return None
    lo, hi = 0.01, 5.0
    if _bs(s, k, t, hi, call) < price or _bs(s, k, t, lo, call) > price:
        return None
    for _ in range(60):
        mid = (lo + hi) / 2
        if _bs(s, k, t, mid, call) > price:
            hi = mid
        else:
            lo = mid
    return round((lo + hi) / 2, 4)


def _cls(sym: str) -> str:
    return "etf" if sym in ("SPY", "QQQ") else "stocks"


def options(sym: str, today: date) -> list:
    q = _get(QUOTE.format(sym=sym, cls=_cls(sym)))
    s = _num((((q.get("data") or {}).get("primaryData") or {}).get("lastSalePrice")))
    if not s:
        return []
    d = _get(CHAIN.format(sym=sym, cls=_cls(sym), start=(today + timedelta(days=5)).isoformat(), end=(today + timedelta(days=60)).isoformat()))
    rows = ((d.get("data") or {}).get("table") or {}).get("rows") or []
    by_exp: dict = {}
    for r in rows:
        if not r.get("strike") or not r.get("expiryDate"):
            continue
        by_exp.setdefault(r["expiryDate"], []).append(r)
    out = []
    for label in sorted(by_exp, key=lambda e: datetime.strptime(f"{e} {today.year}", "%b %d %Y")):
        exp = datetime.strptime(f"{label} {today.year}", "%b %d %Y").date()
        if exp < today:
            exp = exp.replace(year=today.year + 1)
        days = (exp - today).days
        if not out and days < 5 or out and days < 25:
            continue
        t = days / 365
        strikes = by_exp[label]
        mid = lambda r, p: (lambda b, a: (b + a) / 2 if b and a and a >= b else _num(r.get(f"{p}_Last")))(_num(r.get(f"{p}_Bid")), _num(r.get(f"{p}_Ask")))
        k = lambda r: _num(r["strike"])
        atm = min(strikes, key=lambda r: abs(k(r) - s))
        put10 = min(strikes, key=lambda r: abs(k(r) - 0.9 * s))
        call10 = min(strikes, key=lambda r: abs(k(r) - 1.1 * s))
        c_atm, p_atm = mid(atm, "c"), mid(atm, "p")
        iv_c = implied_vol(c_atm, s, k(atm), t, True) if c_atm else None
        iv_p = implied_vol(p_atm, s, k(atm), t, False) if p_atm else None
        iv_put10 = implied_vol(mid(put10, "p"), s, k(put10), t, False) if mid(put10, "p") else None
        iv_call10 = implied_vol(mid(call10, "c"), s, k(call10), t, True) if mid(call10, "c") else None
        vol = lambda p: sum(_num(r.get(f"{p}_Volume")) or 0 for r in strikes)
        oi = lambda p: sum(_num(r.get(f"{p}_Openinterest")) or 0 for r in strikes)
        out.append({"date": today.isoformat(), "symbol": sym, "price": s, "expiry": exp.isoformat(), "days": days,
                    "iv_atm": round((iv_c + iv_p) / 2, 4) if iv_c and iv_p else (iv_c or iv_p),
                    "iv_put10": iv_put10, "iv_call10": iv_call10,
                    "skew": round(iv_put10 - iv_call10, 4) if iv_put10 and iv_call10 else None,
                    "straddle_pct": round((c_atm + p_atm) / s * 100, 2) if c_atm and p_atm else None,
                    "put_vol": vol("p"), "call_vol": vol("c"), "put_oi": oi("p"), "call_oi": oi("c"),
                    "pc_vol": round(vol("p") / vol("c"), 3) if vol("c") else None,
                    "pc_oi": round(oi("p") / oi("c"), 3) if oi("c") else None})
        if len(out) == 2:
            break
    return out


def analysts(sym: str, key: str, today: date) -> dict:
    rec = json.load(urllib.request.urlopen(f"https://finnhub.io/api/v1/stock/recommendation?symbol={sym}&token={key}", timeout=20))
    surp = json.load(urllib.request.urlopen(f"https://finnhub.io/api/v1/stock/earnings?symbol={sym}&token={key}", timeout=20))
    return {"date": today.isoformat(), "symbol": sym, "recommendation": json.dumps(rec), "surprises": json.dumps(surp)}


def main(syms: list) -> None:
    from dotenv import dotenv_values
    today = datetime.now(timezone.utc).date()
    key = dotenv_values(ROOT / "collectors/news/finnhub/.env").get("FINNHUB_API_KEY") or os.getenv("FINNHUB_API_KEY")
    universe = syms or sorted({p.name.split(".")[0] for p in EP_DIR.glob("*.daily.parquet")} | set(EXTRA))
    opt, ana, failed = [], [], []
    for s in universe:
        try:
            opt += options(s, today)
        except Exception as e:  # noqa: BLE001 - one symbol's chain does not stop the rest
            failed.append(f"{s} options: {type(e).__name__}")
        if key and s not in ("SPY", "QQQ"):
            try:
                ana.append(analysts(s, key, today))
            except Exception as e:  # noqa: BLE001
                failed.append(f"{s} analysts: {type(e).__name__}")
            time.sleep(1.1)                     # Finnhub free tier: 60 calls a minute, two per symbol
        time.sleep(0.4)
    for name, rows in (("options", opt), ("analysts", ana)):
        if rows:
            (OUT / name).mkdir(parents=True, exist_ok=True)
            pd.DataFrame(rows).to_parquet(OUT / name / f"{today.isoformat()}.parquet")
    print(f"{today}: options rows {len(opt)} ({len({r['symbol'] for r in opt})} symbols), analysts {len(ana)}; failed {len(failed)}: {failed[:8]}")


if __name__ == "__main__":
    main(sys.argv[1:])
