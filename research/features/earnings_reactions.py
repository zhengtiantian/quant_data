#!/usr/bin/env python3
"""How each stock actually moved around its past earnings, for the holdings advisor's strategy tests.

For every symbol of the stock universe and every holding in quant_ai: the last ~16 reported earnings dates (yfinance), the daily
bars around each (5 trading days before the reaction day to 10 after), and the reaction itself:
the gap at the open and the move to the close of the first session that could react. One document
per (symbol, report date) in `earnings_reactions`; quant_ai's strategy engine replays its limit-order
ladders on these real paths (PROJECT_PLAN Q.5, 2026-10-25).

Reaction day: a report after the close (yfinance time >= 16:00 New York) is first traded the next
session; one before the open is traded the same day. SK hynix's ADR (SKHY) only listed on
2026-07-10, so its history comes from the Seoul listing, 000660.KS, whose reports land before the
Korean open (a gap that day, in won; the percentages are what matters).

    EARNINGS_SYMBOLS=STX,SKHY python research/features/earnings_reactions.py
"""
from __future__ import annotations

import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import requests
import yfinance as yf
from dotenv import load_dotenv
from pymongo import MongoClient, UpdateOne

ROOT = Path(__file__).resolve().parents[2]
load_dotenv(ROOT / ".env", override=False)
MONGO_URI = os.getenv("MONGO_URI") or os.getenv("HOST_LOCAL_MONGO_URI")
COLLECTION = "earnings_reactions"
HISTORY_SOURCE = {"SKHY": "000660.KS"}       # where a young listing's earnings history lives
QUANT_AI_SYMBOLS = os.getenv("QUANT_AI_SYMBOLS_URL", "http://127.0.0.1:18000/api/markets/portfolio/symbols")
EVENTS = int(os.getenv("EARNINGS_REACTION_EVENTS", "16"))
BEFORE, AFTER = 5, 10


def reaction_index(bars: pd.DataFrame, report: pd.Timestamp, korean: bool) -> int | None:
    """Position in `bars` of the first session that could react to the report."""
    day = report.tz_convert("Asia/Seoul" if korean else "America/New_York")
    d = day.normalize().tz_localize(None)
    idx = bars.index.tz_localize(None) if bars.index.tz is not None else bars.index
    same = (idx.normalize() == d)
    if korean:
        after = idx.normalize() >= d                         # announced before the Seoul open
    elif day.hour >= 16:
        after = idx.normalize() > d                          # after the close: next session
    elif day.hour < 9 or (day.hour == 9 and day.minute < 30):
        after = idx.normalize() >= d                         # before the open: same session
    else:
        after = idx.normalize() > d if not same.any() else idx.normalize() >= d
    pos = [i for i, a in enumerate(after) if a]
    return pos[0] if pos else None


def events_for(symbol: str) -> list[dict]:
    source = HISTORY_SOURCE.get(symbol, symbol)
    korean = source.endswith(".KS")
    t = yf.Ticker(source)
    dates = t.get_earnings_dates(limit=EVENTS + 8)
    if dates is None or dates.empty:
        return []
    now = pd.Timestamp.now(tz="UTC")
    past = sorted([d for d in dates.index if d < now])[-EVENTS:]
    bars = t.history(period="5y", interval="1d", auto_adjust=False)
    if bars.empty:
        return []
    out = []
    for report in past:
        i = reaction_index(bars, report, korean)
        if i is None or i < BEFORE or i + AFTER >= len(bars):
            continue
        prev_close = float(bars["Close"].iloc[i - 1])
        r = bars.iloc[i]
        window = bars.iloc[i - BEFORE:i + AFTER + 1]
        out.append({
            "symbol": symbol, "history_source": source, "report_time": report.to_pydatetime(),
            "reaction_date": bars.index[i].strftime("%Y-%m-%d"),
            "prev_close": round(prev_close, 4),
            "gap_pct": round((float(r["Open"]) / prev_close - 1) * 100, 2),
            "move_pct": round((float(r["Close"]) / prev_close - 1) * 100, 2),
            "run_up_5d_pct": round((prev_close / float(bars["Close"].iloc[i - BEFORE]) - 1) * 100, 2),
            "drift_10d_pct": round((float(bars["Close"].iloc[i + AFTER]) / float(r["Close"]) - 1) * 100, 2),
            "bars": [{"date": d.strftime("%Y-%m-%d"), "open": round(float(b["Open"]), 4), "high": round(float(b["High"]), 4),
                      "low": round(float(b["Low"]), 4), "close": round(float(b["Close"]), 4)}
                     for d, b in window.iterrows()],
            "reaction_offset": BEFORE,                      # index of the reaction day in `bars`
            "currency": "KRW" if korean else "USD",
            "computed_at": datetime.now(timezone.utc),
        })
    return out


def main() -> int:
    db = MongoClient(MONGO_URI)["quant_data"]
    only = [s.strip().upper() for s in os.getenv("EARNINGS_SYMBOLS", "").split(",") if s.strip()]
    symbols = only or [d["symbol"] for d in db["stock_universe"].find({}, {"symbol": 1, "_id": 0})]
    if not only:
        try:                                     # a holding outside the universe still needs its history
            held = requests.get(QUANT_AI_SYMBOLS, timeout=5).json().get("held") or []
            symbols += [s for s in held if s not in symbols]
        except Exception as e:  # noqa: BLE001 - quant_ai not running: the universe is still refreshed
            print(f"quant_ai holdings unavailable: {e}", flush=True)
    db[COLLECTION].create_index([("symbol", 1), ("report_time", 1)], unique=True)
    total = 0
    for n, sym in enumerate(symbols, 1):
        try:
            evs = events_for(sym)
        except Exception as e:  # noqa: BLE001 - one symbol failing must not stop the rest
            print(f"  {sym}: {type(e).__name__}: {e}", flush=True)
            evs = []
        if evs:
            db[COLLECTION].bulk_write([UpdateOne({"symbol": e["symbol"], "report_time": e["report_time"]}, {"$set": e}, upsert=True)
                                       for e in evs], ordered=False)
        total += len(evs)
        print(f"PROGRESS reactions {n} {len(symbols)} {sym} events={len(evs)}", flush=True)
        time.sleep(0.3)
    print(f"DONE {total} events for {len(symbols)} symbols", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
