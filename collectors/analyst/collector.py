#!/usr/bin/env python3
"""Analyst consensus collector (D.4).

Fetches monthly consensus recommendation counts from Finnhub
(/api/v1/stock/recommendation) via urllib (works under LightningX VPN).

Upserts per (symbol, period) to `analyst_consensus` collection.
Finnhub free tier returns ~4 months of history per symbol.

Run daily at 07:48 (after premarket 07:45, before macro 07:50).
"""

from __future__ import annotations

import json
import os
import ssl
import time
import urllib.request
from datetime import UTC, datetime
from pathlib import Path

from dotenv import load_dotenv
from pymongo import MongoClient, UpdateOne

ROOT = Path(__file__).resolve().parents[2]
load_dotenv(ROOT / ".env", override=False)
load_dotenv(ROOT / "news_collectors" / "finnhub" / ".env", override=False)

FINNHUB_KEY = (
    os.getenv("FINNHUB_API_KEY")
    or os.getenv("FINNHUB_TOKEN")
    or os.getenv("FINNHUB_KEY")
)
MONGO_URI = os.getenv("MONGO_URI") or os.getenv("LOCAL_MONGO_URI", "mongodb://root:root@127.0.0.1:37018/")
DB_NAME = "quant_data"
COLLECTION = "analyst_consensus"
UNIVERSE_COLLECTION = "stock_universe"
SLEEP_BETWEEN = float(os.getenv("ANALYST_SLEEP", "0.4"))

_SSL_CTX = ssl.create_default_context()


def _get(url: str) -> list | dict | None:
    try:
        with urllib.request.urlopen(url, context=_SSL_CTX, timeout=10) as r:
            return json.loads(r.read())
    except Exception as e:
        print(f"  GET failed: {e}")
        return None


def load_universe() -> list[str]:
    client = MongoClient(MONGO_URI)
    docs = list(client[DB_NAME][UNIVERSE_COLLECTION].find({}, {"symbol": 1, "_id": 0}))
    return sorted(d["symbol"] for d in docs)


def fetch_consensus(symbol: str) -> list[dict]:
    """Return list of monthly consensus dicts from Finnhub."""
    if not FINNHUB_KEY:
        raise RuntimeError("FINNHUB_API_KEY missing")
    url = f"https://finnhub.io/api/v1/stock/recommendation?symbol={symbol}&token={FINNHUB_KEY}"
    data = _get(url)
    if not isinstance(data, list):
        return []
    return data


def compute_features(rows: list[dict]) -> list[dict]:
    """Compute analyst factors from Finnhub consensus rows (newest first)."""
    if not rows:
        return []

    # Sort newest-first so index 0 = latest month
    rows = sorted(rows, key=lambda r: r.get("period", ""), reverse=True)
    now = datetime.now(UTC)
    out: list[dict] = []

    for i, row in enumerate(rows):
        period = row.get("period", "")[:7]  # "2026-06"
        symbol = row.get("symbol", "")
        sb = int(row.get("strongBuy", 0) or 0)
        b = int(row.get("buy", 0) or 0)
        h = int(row.get("hold", 0) or 0)
        s = int(row.get("sell", 0) or 0)
        ss = int(row.get("strongSell", 0) or 0)
        total = sb + b + h + s + ss

        buy_ratio = (sb + b) / total if total > 0 else None
        sell_ratio = (ss + s) / total if total > 0 else None
        # weighted consensus: +2 strong buy → +1 buy → 0 hold → -1 sell → -2 strong sell
        consensus_score = (2 * sb + b - s - 2 * ss) / total if total > 0 else None

        # 1-month change: compare to next entry (one month older)
        buy_ratio_chg_1m = None
        if i + 1 < len(rows) and buy_ratio is not None:
            prev = rows[i + 1]
            pt = sum(int(prev.get(k, 0) or 0) for k in ("strongBuy", "buy", "hold", "sell", "strongSell"))
            if pt > 0:
                prev_buy = (int(prev.get("strongBuy", 0) or 0) + int(prev.get("buy", 0) or 0)) / pt
                buy_ratio_chg_1m = buy_ratio - prev_buy

        out.append({
            "symbol": symbol,
            "period": period,
            "strongBuy": sb, "buy": b, "hold": h, "sell": s, "strongSell": ss,
            "total_analysts": total,
            "analyst_buy_ratio": buy_ratio,
            "analyst_sell_ratio": sell_ratio,
            "analyst_consensus_score": consensus_score,
            "analyst_buy_ratio_chg_1m": buy_ratio_chg_1m,
            "collectedAt": now,
        })

    return out


def upsert_batch(records: list[dict], col) -> tuple[int, int]:
    if not records:
        return 0, 0
    ops = [
        UpdateOne(
            {"symbol": r["symbol"], "period": r["period"]},
            {"$set": r},
            upsert=True,
        )
        for r in records
    ]
    res = col.bulk_write(ops)
    return res.upserted_count, res.modified_count


def main() -> None:
    print("=== Analyst consensus collector (Finnhub) ===")
    symbols = load_universe()
    print(f"Universe: {len(symbols)} symbols")

    client = MongoClient(MONGO_URI)
    col = client[DB_NAME][COLLECTION]
    col.create_index([("symbol", 1), ("period", 1)], unique=True, background=True)

    total_up = total_mod = 0
    for sym in symbols:
        rows = fetch_consensus(sym)
        records = compute_features(rows)
        up, mod = upsert_batch(records, col)
        total_up += up
        total_mod += mod
        if records:
            periods = [r["period"] for r in records]
            print(f"  {sym:6} {len(records)} periods {periods[0]}→{periods[-1]}  up={up} mod={mod}")
        else:
            print(f"  {sym:6} no data")
        time.sleep(SLEEP_BETWEEN)

    print(f"\nDone. total upserted={total_up} modified={total_mod}")


if __name__ == "__main__":
    main()
