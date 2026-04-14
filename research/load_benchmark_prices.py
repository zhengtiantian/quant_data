#!/usr/bin/env python3
"""Load benchmark ETF daily prices into stock_prices_history."""

from __future__ import annotations

import os
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd
import yfinance as yf
from dotenv import load_dotenv
from pymongo import MongoClient, errors


CURRENT = Path(__file__).resolve()
ROOT = CURRENT.parents[1]
GLOBAL_ENV = ROOT / ".env"
load_dotenv(GLOBAL_ENV, override=False)

MONGO_URI = os.getenv("MONGO_URI")
if not MONGO_URI:
    raise RuntimeError("Missing MONGO_URI")

BENCHMARK_SYMBOLS = [
    symbol.strip().upper()
    for symbol in os.getenv("FEATURE_BENCHMARK_SYMBOLS", "SPY,QQQ").split(",")
    if symbol.strip()
]
BENCHMARK_PERIOD = os.getenv("BENCHMARK_PRICE_PERIOD", "10y")
BENCHMARK_INTERVAL = os.getenv("BENCHMARK_PRICE_INTERVAL", "1d")


def val(x):
    if isinstance(x, pd.Series):
        return float(x.values[0])
    return float(x)


def fetch_daily_history(symbol: str, period: str, interval: str) -> list[dict]:
    try:
        df = yf.download(
            symbol,
            period=period,
            interval=interval,
            auto_adjust=False,
            progress=False,
            threads=True,
        )
    except Exception as exc:
        print(f"❌ {symbol} download failed: {exc}")
        return []

    if df.empty:
        print(f"⚠️ No data for benchmark {symbol}")
        return []

    df = df.dropna()
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    records = []
    for ts, row in df.iterrows():
        try:
            records.append(
                {
                    "symbol": symbol,
                    "timestamp": ts.to_pydatetime().replace(tzinfo=UTC).isoformat(),
                    "open": val(row["Open"]),
                    "high": val(row["High"]),
                    "low": val(row["Low"]),
                    "close": val(row["Close"]),
                    "volume": int(row["Volume"]),
                    "interval": interval,
                    "source": "yahoo",
                    "collectedAt": datetime.now(UTC).isoformat(),
                }
            )
        except Exception as exc:
            print(f"⚠️ Row parse error for {symbol}: {exc}")
    return records


def save_history(records: list[dict]) -> int:
    if not records:
        return 0

    client = MongoClient(MONGO_URI)
    col = client["quant_data"]["stock_prices_history"]
    try:
        col.create_index([("symbol", 1), ("timestamp", 1)], unique=True, sparse=True)
    except Exception:
        pass

    inserted = 0
    for rec in records:
        try:
            col.insert_one(rec)
            inserted += 1
        except errors.DuplicateKeyError:
            continue
    return inserted


def main() -> None:
    print(f"=== Loading benchmark prices: {', '.join(BENCHMARK_SYMBOLS)} ===")
    total = 0
    for symbol in BENCHMARK_SYMBOLS:
        print(f"\nFetching benchmark {symbol} ...")
        records = fetch_daily_history(symbol, BENCHMARK_PERIOD, BENCHMARK_INTERVAL)
        count = save_history(records)
        total += count
        print(f"Inserted {count} records for benchmark {symbol}")
    print(f"\n=== DONE: Inserted {total} benchmark records ===")


if __name__ == "__main__":
    main()
