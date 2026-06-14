#!/usr/bin/env python3
"""Pre/after-market price signal collector (D.8).

Fetches 1-minute bars (extended hours) from yfinance for each symbol in the universe.
Extracts:
  pm_gap           : (premarket last price / previous close) - 1
  pm_volume_ratio  : premarket volume / same-day regular volume
  ah_gap           : (after-hours last price / same-day close) - 1
  ah_volume_ratio  : after-hours volume / same-day regular volume

Upserts per (symbol, trade_date) to `premarket_signals` collection.

Limitation: yfinance only provides 1m data for the last ~30 days.
Historical rows in daily_symbol_features will have NULL for these fields.

Run daily at 07:45 (after price quotes at 07:30, before feature build at 08:00).
"""

from __future__ import annotations

import os
import time
import zoneinfo
from datetime import UTC, datetime

import yfinance as yf
from pymongo import MongoClient, UpdateOne

MONGO_URI = os.getenv("MONGO_URI") or os.getenv("LOCAL_MONGO_URI", "mongodb://root:root@127.0.0.1:37018/")
DB_NAME = "quant_data"
COLLECTION = "premarket_signals"
UNIVERSE_COLLECTION = "stock_universe"
PERIOD = os.getenv("PREMARKET_PERIOD", "5d")
SLEEP_BETWEEN = float(os.getenv("PREMARKET_SLEEP", "0.6"))

ET = zoneinfo.ZoneInfo("America/New_York")

# Extended hours boundaries (ET hour, inclusive/exclusive)
_PM_START_H = 4   # 04:00 ET — pre-market open
_PM_END_H = 9     # 09:30 ET — use hour<9 or (hour==9 and minute<30)
_AH_START_H = 16  # 16:00 ET — after-hours open
_AH_END_H = 20    # 20:00 ET — after-hours close


def load_universe() -> list[str]:
    client = MongoClient(MONGO_URI)
    docs = list(client[DB_NAME][UNIVERSE_COLLECTION].find({}, {"symbol": 1, "_id": 0}))
    return sorted(d["symbol"] for d in docs)


def fetch_extended(symbol: str) -> dict[str, dict]:
    """Return {date_str: {pm_*, ah_*, reg_*}} for last PERIOD days."""
    try:
        tk = yf.Ticker(symbol)
        df = tk.history(period=PERIOD, interval="1m", prepost=True, auto_adjust=False)
    except Exception as e:
        print(f"  {symbol}: download error — {e}")
        return {}

    if df is None or df.empty:
        return {}

    # Normalise timezone to ET
    if df.index.tz is None:
        df.index = df.index.tz_localize("UTC")
    df.index = df.index.tz_convert(ET)

    per_date: dict[str, dict] = {}

    # ── Regular session (09:30–16:00 ET) ──────────────────────────
    reg = df[
        ((df.index.hour == 9) & (df.index.minute >= 30) | (df.index.hour > 9))
        & (df.index.hour < 16)
    ]
    for dt, grp in reg.groupby(reg.index.date):
        if grp.empty:
            continue
        d = dt.isoformat()
        per_date.setdefault(d, {})
        per_date[d]["reg_open"] = float(grp["Open"].iloc[0])
        per_date[d]["reg_close"] = float(grp["Close"].iloc[-1])
        per_date[d]["reg_volume"] = float(grp["Volume"].sum())

    # ── Pre-market (04:00–09:29 ET) ───────────────────────────────
    pm = df[
        (df.index.hour >= _PM_START_H)
        & ((df.index.hour < _PM_END_H) | ((df.index.hour == 9) & (df.index.minute < 30)))
    ]
    for dt, grp in pm.groupby(pm.index.date):
        if grp.empty:
            continue
        d = dt.isoformat()
        per_date.setdefault(d, {})
        per_date[d]["pm_first"] = float(grp["Close"].iloc[0])
        per_date[d]["pm_last"] = float(grp["Close"].iloc[-1])
        per_date[d]["pm_volume"] = float(grp["Volume"].sum())

    # ── After-hours (16:00–20:00 ET) ──────────────────────────────
    ah = df[(df.index.hour >= _AH_START_H) & (df.index.hour < _AH_END_H)]
    for dt, grp in ah.groupby(ah.index.date):
        if grp.empty:
            continue
        d = dt.isoformat()
        per_date.setdefault(d, {})
        per_date[d]["ah_first"] = float(grp["Close"].iloc[0])
        per_date[d]["ah_last"] = float(grp["Close"].iloc[-1])
        per_date[d]["ah_volume"] = float(grp["Volume"].sum())

    return per_date


def compute_signals(symbol: str, per_date: dict[str, dict]) -> list[dict]:
    """Derive gap / volume-ratio metrics from raw per-date buckets."""
    dates = sorted(per_date.keys())
    now = datetime.now(UTC)
    records: list[dict] = []

    for i, d in enumerate(dates):
        row = per_date[d]
        reg_close = row.get("reg_close")
        reg_volume = row.get("reg_volume") or None

        if reg_close is None:
            continue  # no regular session data → skip

        prev_close = per_date[dates[i - 1]].get("reg_close") if i > 0 else None

        # Pre-market gap: pm_last vs previous day's regular close
        pm_gap: float | None = None
        pm_volume_ratio: float | None = None
        if prev_close and prev_close > 0 and row.get("pm_last"):
            pm_gap = (row["pm_last"] - prev_close) / prev_close
        if row.get("pm_volume") is not None and reg_volume and reg_volume > 0:
            pm_volume_ratio = row["pm_volume"] / reg_volume

        # After-hours gap: ah_last vs same-day regular close
        ah_gap: float | None = None
        ah_volume_ratio: float | None = None
        if reg_close > 0 and row.get("ah_last"):
            ah_gap = (row["ah_last"] - reg_close) / reg_close
        if row.get("ah_volume") is not None and reg_volume and reg_volume > 0:
            ah_volume_ratio = row["ah_volume"] / reg_volume

        records.append({
            "symbol": symbol,
            "trade_date": d,
            "pm_gap": pm_gap,
            "pm_volume_ratio": pm_volume_ratio,
            "ah_gap": ah_gap,
            "ah_volume_ratio": ah_volume_ratio,
            "reg_close": reg_close,
            "reg_volume": reg_volume,
            "collectedAt": now,
        })

    return records


def upsert_records(records: list[dict]) -> None:
    if not records:
        return
    client = MongoClient(MONGO_URI)
    col = client[DB_NAME][COLLECTION]
    col.create_index([("symbol", 1), ("trade_date", 1)], unique=True, background=True)
    ops = [
        UpdateOne(
            {"symbol": r["symbol"], "trade_date": r["trade_date"]},
            {"$set": r},
            upsert=True,
        )
        for r in records
    ]
    res = col.bulk_write(ops)
    print(f"  upserted={res.upserted_count} modified={res.modified_count}")


def main() -> None:
    print(f"=== Pre/after-market collector (period={PERIOD}) ===")
    symbols = load_universe()
    print(f"Universe: {len(symbols)} symbols")

    total_up = total_mod = 0
    for sym in symbols:
        per_date = fetch_extended(sym)
        records = compute_signals(sym, per_date)
        if records:
            client = MongoClient(MONGO_URI)
            col = client[DB_NAME][COLLECTION]
            col.create_index([("symbol", 1), ("trade_date", 1)], unique=True, background=True)
            ops = [
                UpdateOne(
                    {"symbol": r["symbol"], "trade_date": r["trade_date"]},
                    {"$set": r},
                    upsert=True,
                )
                for r in records
            ]
            res = col.bulk_write(ops)
            total_up += res.upserted_count
            total_mod += res.modified_count
            print(f"  {sym:6} {len(records)} dates  up={res.upserted_count} mod={res.modified_count}")
        else:
            print(f"  {sym:6} no data")
        time.sleep(SLEEP_BETWEEN)

    print(f"\nDone. total upserted={total_up} modified={total_mod}")


if __name__ == "__main__":
    main()
