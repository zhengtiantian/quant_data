#!/usr/bin/env python3
"""Macro indicator collector (D.1 regime features).

Fetches daily history for VIX, the US 10y yield, the dollar index and SPY,
and upserts one doc per date into `macro_indicators`. These market-level
series drive the regime features attached to daily_symbol_features.

Run (on host — yfinance needs egress):
  LOCAL_MONGO_URI="mongodb://root:root@127.0.0.1:37018/" \
  .venv311/bin/python collectors/macro/collector.py
"""

from __future__ import annotations

import os
import time
from datetime import UTC, datetime

import yfinance as yf
from pymongo import MongoClient, UpdateOne

MONGO_URI = os.getenv("MONGO_URI") or os.getenv("LOCAL_MONGO_URI", "mongodb://root:root@127.0.0.1:37018/")
DB_NAME = "quant_data"
COLLECTION = "macro_indicators"
PERIOD = os.getenv("MACRO_PERIOD", "5y")  # first fill long; daily job can use "1mo"

# yfinance ticker -> field name in macro_indicators
TICKERS = {
    "^VIX": "vix",
    "^TNX": "tnx",          # 10y treasury yield
    "DX-Y.NYB": "dxy",      # US dollar index
    "SPY": "spy_close",
}


def fetch_series(ticker: str) -> dict[str, float]:
    try:
        df = yf.download(ticker, period=PERIOD, interval="1d",
                         progress=False, auto_adjust=False, threads=False)
    except Exception as e:
        print(f"  {ticker} download failed: {e}")
        return {}
    if df is None or len(df) == 0:
        print(f"  {ticker} empty")
        return {}
    out = {}
    for ts, row in df.iterrows():
        try:
            out[ts.date().isoformat()] = float(row["Close"].item())
        except Exception:
            continue
    return out


def main():
    print(f"=== Macro collector (period={PERIOD}) ===")
    per_date: dict[str, dict] = {}
    for ticker, field in TICKERS.items():
        series = fetch_series(ticker)
        print(f"  {ticker:10} -> {field:10} {len(series)} days")
        for d, v in series.items():
            per_date.setdefault(d, {})[field] = v
        time.sleep(0.5)

    if not per_date:
        print("No macro data fetched.")
        return

    now = datetime.now(UTC)
    ops = []
    for d, fields in per_date.items():
        fields["date"] = d
        fields["collectedAt"] = now
        ops.append(UpdateOne({"date": d}, {"$set": fields}, upsert=True))

    client = MongoClient(MONGO_URI)
    col = client[DB_NAME][COLLECTION]
    col.create_index("date", unique=True, background=True)
    res = col.bulk_write(ops)
    print(f"Upserted {res.upserted_count} / modified {res.modified_count} "
          f"macro rows ({len(per_date)} dates)")


if __name__ == "__main__":
    main()
