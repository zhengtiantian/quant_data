#!/usr/bin/env python3
"""Load historical and upcoming earnings dates into earnings_events."""

from __future__ import annotations

import os
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd
import yfinance as yf
from dotenv import load_dotenv
from pymongo import MongoClient, UpdateOne


CURRENT = Path(__file__).resolve()
ROOT = CURRENT.parents[2]
GLOBAL_ENV = ROOT / ".env"
load_dotenv(GLOBAL_ENV, override=False)

MONGO_URI = os.getenv("MONGO_URI")
if not MONGO_URI:
    raise RuntimeError("Missing MONGO_URI")

DB_NAME = os.getenv("FEATURE_DB_NAME", "quant_data")
UNIVERSE_COLLECTION = os.getenv("FEATURE_UNIVERSE_COLLECTION", "stock_universe")
EARNINGS_COLLECTION = os.getenv("FEATURE_EARNINGS_COLLECTION", "earnings_events")
EARNINGS_LIMIT = min(int(os.getenv("EARNINGS_LIMIT", "100")), 100)


def _coerce_value(value):
    if value is None or pd.isna(value):
        return None
    if isinstance(value, pd.Timestamp):
        return value.to_pydatetime()
    if isinstance(value, (int, float)):
        return float(value)
    return value


def load_symbols(client: MongoClient) -> list[str]:
    col = client[DB_NAME][UNIVERSE_COLLECTION]
    return sorted(
        {
            str(doc["symbol"]).strip().upper()
            for doc in col.find({"symbol": {"$exists": True, "$ne": None}}, {"symbol": 1})
            if str(doc.get("symbol", "")).strip()
        }
    )


def fetch_symbol_earnings(symbol: str, limit: int) -> list[dict]:
    ticker = yf.Ticker(symbol)
    try:
        df = ticker.get_earnings_dates(limit=limit)
    except Exception as exc:
        print(f"⚠️ {symbol} earnings fetch failed: {exc}")
        return []

    if df is None or df.empty:
        print(f"⚠️ No earnings dates returned for {symbol}")
        return []

    df = df.reset_index()
    date_column = next((col for col in df.columns if "date" in str(col).lower()), None)
    if date_column is None:
        print(f"⚠️ Could not find earnings date column for {symbol}")
        return []

    records = []
    fetched_at = datetime.now(UTC).isoformat()
    for _, row in df.iterrows():
        raw_date = row.get(date_column)
        if pd.isna(raw_date):
            continue
        ts = pd.Timestamp(raw_date)
        if ts.tzinfo is None:
            ts = ts.tz_localize(UTC)
        else:
            ts = ts.tz_convert(UTC)
        event_date = ts.normalize()
        record = {
            "symbol": symbol,
            "earnings_date": ts.isoformat(),
            "event_date": event_date.date().isoformat(),
            "fetchedAt": fetched_at,
            "source": "yfinance",
        }
        for column in ("EPS Estimate", "Reported EPS", "Surprise(%)"):
            if column in row.index:
                key = column.lower().replace(" ", "_").replace("(%)", "_pct").replace("__", "_")
                value = _coerce_value(row.get(column))
                if isinstance(value, datetime):
                    value = value.isoformat()
                record[key] = value
        records.append(record)

    return records


def save_records(client: MongoClient, records: list[dict]) -> int:
    if not records:
        return 0

    col = client[DB_NAME][EARNINGS_COLLECTION]
    col.create_index([("symbol", 1), ("event_date", 1)], unique=True)

    ops = [
        UpdateOne(
            {"symbol": rec["symbol"], "event_date": rec["event_date"]},
            {"$set": rec},
            upsert=True,
        )
        for rec in records
    ]
    result = col.bulk_write(ops, ordered=False)
    return result.upserted_count + result.modified_count


def main() -> None:
    client = MongoClient(MONGO_URI)
    symbols = load_symbols(client)
    print(f"=== Loading earnings events for {len(symbols)} symbols ===")

    total_saved = 0
    for symbol in symbols:
        print(f"Fetching earnings dates for {symbol} ...")
        records = fetch_symbol_earnings(symbol, EARNINGS_LIMIT)
        saved = save_records(client, records)
        total_saved += saved
        print(f"Saved {saved} earnings rows for {symbol}")

    print(f"=== DONE: saved {total_saved} earnings rows ===")


if __name__ == "__main__":
    main()
