#!/usr/bin/env python3
"""Pre/after-market price signal collector (D.8).

Fetches 1-minute bars (extended hours) from yfinance for each symbol in the universe.
Extracts:
  pm_gap           : (premarket last price / previous close) - 1
  pm_volume_ratio  : premarket volume / same-day regular volume
  ah_gap           : (after-hours last price / same-day close) - 1
  ah_volume_ratio  : after-hours volume / same-day regular volume

Upserts per (symbol, trade_date) to `premarket_signals` collection.

Yahoo Finance 1m limit: max 8 days per request, up to 30 days look-back.
We chunk into 7-day windows to stay within the limit.

Daily job (PREMARKET_BACKFILL_DAYS=7): 1 request per symbol at 07:45.
Initial backfill (PREMARKET_BACKFILL_DAYS=30): ~5 requests per symbol.
"""

from __future__ import annotations

import os
import time
import zoneinfo
from datetime import UTC, date, datetime, timedelta

import yfinance as yf
from pymongo import MongoClient, UpdateOne

MONGO_URI = os.getenv("MONGO_URI") or os.getenv("LOCAL_MONGO_URI", "mongodb://root:root@127.0.0.1:37018/")
DB_NAME = "quant_data"
COLLECTION = "premarket_signals"
UNIVERSE_COLLECTION = "stock_universe"

# How many calendar days to look back (chunked into 7-day windows)
BACKFILL_DAYS = int(os.getenv("PREMARKET_BACKFILL_DAYS", "7"))
CHUNK_DAYS = 7          # Yahoo 1m limit is 8 days; use 7 to be safe
SLEEP_BETWEEN = float(os.getenv("PREMARKET_SLEEP", "0.8"))

ET = zoneinfo.ZoneInfo("America/New_York")

# Extended hours boundaries (ET hour)
_PM_START_H = 4    # 04:00 ET — pre-market open
_AH_START_H = 16   # 16:00 ET — after-hours open
_AH_END_H = 20     # 20:00 ET — after-hours close


def load_universe() -> list[str]:
    client = MongoClient(MONGO_URI)
    docs = list(client[DB_NAME][UNIVERSE_COLLECTION].find({}, {"symbol": 1, "_id": 0}))
    return sorted(d["symbol"] for d in docs)


def _date_windows(total_days: int) -> list[tuple[date, date]]:
    """Split [today-total_days, today] into CHUNK_DAYS-day windows (newest first)."""
    end = datetime.now(ET).date()
    start = end - timedelta(days=total_days)
    windows: list[tuple[date, date]] = []
    cur_end = end
    while cur_end > start:
        cur_start = max(cur_end - timedelta(days=CHUNK_DAYS - 1), start)
        windows.append((cur_start, cur_end + timedelta(days=1)))  # yf end is exclusive
        cur_end = cur_start - timedelta(days=1)
    return windows


def _parse_df(df) -> dict[str, dict]:
    """Extract {date_str: {pm_*, ah_*, reg_*}} from a 1m OHLCV DataFrame."""
    if df is None or df.empty:
        return {}

    if df.index.tz is None:
        df.index = df.index.tz_localize("UTC")
    df.index = df.index.tz_convert(ET)

    per_date: dict[str, dict] = {}

    # Regular session: 09:30–16:00 ET
    reg = df[
        (((df.index.hour == 9) & (df.index.minute >= 30)) | (df.index.hour > 9))
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

    # Pre-market: 04:00–09:29 ET
    pm = df[
        (df.index.hour >= _PM_START_H)
        & ((df.index.hour < 9) | ((df.index.hour == 9) & (df.index.minute < 30)))
    ]
    for dt, grp in pm.groupby(pm.index.date):
        if grp.empty:
            continue
        d = dt.isoformat()
        per_date.setdefault(d, {})
        per_date[d]["pm_last"] = float(grp["Close"].iloc[-1])
        per_date[d]["pm_volume"] = float(grp["Volume"].sum())

    # After-hours: 16:00–20:00 ET
    ah = df[(df.index.hour >= _AH_START_H) & (df.index.hour < _AH_END_H)]
    for dt, grp in ah.groupby(ah.index.date):
        if grp.empty:
            continue
        d = dt.isoformat()
        per_date.setdefault(d, {})
        per_date[d]["ah_last"] = float(grp["Close"].iloc[-1])
        per_date[d]["ah_volume"] = float(grp["Volume"].sum())

    return per_date


def fetch_symbol(symbol: str, windows: list[tuple[date, date]]) -> dict[str, dict]:
    """Fetch all windows for one symbol, merge into a single per_date dict."""
    merged: dict[str, dict] = {}
    tk = yf.Ticker(symbol)
    for win_start, win_end in windows:
        try:
            df = tk.history(
                start=win_start.isoformat(),
                end=win_end.isoformat(),
                interval="1m",
                prepost=True,
                auto_adjust=False,
            )
        except Exception as e:
            print(f"    {symbol} [{win_start}→{win_end}] error: {e}")
            continue
        chunk = _parse_df(df)
        for d, vals in chunk.items():
            merged.setdefault(d, {}).update(vals)
        time.sleep(0.3)  # small pause between chunks for same symbol
    return merged


def compute_signals(symbol: str, per_date: dict[str, dict]) -> list[dict]:
    """Derive gap / volume-ratio metrics. Needs chronological per_date."""
    dates = sorted(per_date.keys())
    now = datetime.now(UTC)
    records: list[dict] = []

    for i, d in enumerate(dates):
        row = per_date[d]
        reg_close = row.get("reg_close")
        if reg_close is None:
            continue

        reg_volume = row.get("reg_volume") or None
        prev_close = per_date[dates[i - 1]].get("reg_close") if i > 0 else None

        # Pre-market gap: pm_last vs previous day's close
        pm_gap: float | None = None
        pm_volume_ratio: float | None = None
        if prev_close and prev_close > 0 and row.get("pm_last"):
            pm_gap = (row["pm_last"] - prev_close) / prev_close
        # yfinance returns 0 volume for pre-market bars on most symbols;
        # only compute ratio when pm_volume > 0 to avoid a meaningless 0 feature.
        pm_vol = row.get("pm_volume")
        if pm_vol is not None and pm_vol > 0 and reg_volume and reg_volume > 0:
            pm_volume_ratio = pm_vol / reg_volume

        # After-hours gap: ah_last vs same-day close
        ah_gap: float | None = None
        ah_volume_ratio: float | None = None
        if reg_close > 0 and row.get("ah_last"):
            ah_gap = (row["ah_last"] - reg_close) / reg_close
        ah_vol = row.get("ah_volume")
        if ah_vol is not None and ah_vol > 0 and reg_volume and reg_volume > 0:
            ah_volume_ratio = ah_vol / reg_volume

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


def upsert_batch(records: list[dict], col) -> tuple[int, int]:
    if not records:
        return 0, 0
    ops = [
        UpdateOne(
            {"symbol": r["symbol"], "trade_date": r["trade_date"]},
            {"$set": r},
            upsert=True,
        )
        for r in records
    ]
    res = col.bulk_write(ops)
    return res.upserted_count, res.modified_count


def main() -> None:
    windows = _date_windows(BACKFILL_DAYS)
    mode = "backfill" if BACKFILL_DAYS > 7 else "daily"
    print(f"=== Pre/after-market collector ({mode}, {BACKFILL_DAYS}d, {len(windows)} chunks) ===")

    symbols = load_universe()
    print(f"Universe: {len(symbols)} symbols")

    client = MongoClient(MONGO_URI)
    col = client[DB_NAME][COLLECTION]
    col.create_index([("symbol", 1), ("trade_date", 1)], unique=True, background=True)

    total_up = total_mod = 0
    for sym in symbols:
        per_date = fetch_symbol(sym, windows)
        records = compute_signals(sym, per_date)
        up, mod = upsert_batch(records, col)
        total_up += up
        total_mod += mod
        if records:
            print(f"  {sym:6} {len(records)} dates  up={up} mod={mod}")
        else:
            print(f"  {sym:6} no data")
        time.sleep(SLEEP_BETWEEN)

    print(f"\nDone. total upserted={total_up} modified={total_mod}")


if __name__ == "__main__":
    main()
