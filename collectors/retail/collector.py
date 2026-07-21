#!/usr/bin/env python3
"""Retail sentiment collector (D.2) — StockTwits public stream.

Fetches the latest messages per symbol from StockTwits, aggregates
per-day retail sentiment metrics, and upserts to `retail_sentiment`.

Sentiment source (priority):
  1. Explicit user tags: entities.sentiment.basic → "Bullish" / "Bearish"
  2. Keyword fallback for untagged messages (lower confidence, stored separately)

Features produced per (symbol, date):
  retail_msg_count        : total messages on that date
  retail_tagged_count     : messages with explicit sentiment tag
  retail_bull_ratio       : Bullish / (Bullish + Bearish) tagged msgs
  retail_sent_score       : (bull − bear) / (bull + bear), range [−1, +1]

Rate limit: StockTwits public allows ~200 req/hr unauthenticated.
Fetching 1 page (30 msgs) per symbol × 100 symbols = 100 req (~50s with sleep).

Run daily at 20:30 (after US market close, after-hours messages accumulate).
"""

from __future__ import annotations

import os
import re
import time
import urllib.request
import json
import ssl
from datetime import UTC, datetime
from pathlib import Path

from dotenv import load_dotenv
from pymongo import MongoClient, UpdateOne

ROOT = Path(__file__).resolve().parents[2]
load_dotenv(ROOT / ".env", override=False)

MONGO_URI = os.getenv("MONGO_URI") or os.getenv("LOCAL_MONGO_URI", "mongodb://root:root@127.0.0.1:37018/")
DB_NAME = "quant_data"
COLLECTION = "retail_sentiment"
UNIVERSE_COLLECTION = "stock_universe"

PAGES_PER_SYMBOL = int(os.getenv("RETAIL_PAGES", "2"))   # 2 pages = 60 msgs max
SLEEP_BETWEEN = float(os.getenv("RETAIL_SLEEP", "0.6"))  # stay well within rate limit

_SSL_CTX = ssl.create_default_context()
_BASE_URL = "https://api.stocktwits.com/api/2/streams/symbol"

# Keyword fallback (only used when no explicit tag)
_BULL_KW = {
    "bull", "bullish", "buy", "long", "calls", "moon", "squeeze", "breakout",
    "hold", "gain", "beat", "strong", "green", "load", "support", "entry",
    "uptrend", "rally", "bounce", "oversold", "accumulate", "dip",
}
_BEAR_KW = {
    "bear", "bearish", "sell", "short", "puts", "crash", "dump", "drop",
    "weak", "miss", "red", "breakdown", "cut", "resist", "overbought",
    "downtrend", "decline", "correction", "fail", "loss",
}


def _kw_sentiment(text: str) -> str | None:
    words = set(re.findall(r"[a-zA-Z]+", text.lower()))
    b = len(words & _BULL_KW)
    br = len(words & _BEAR_KW)
    if b > br:
        return "Bullish"
    if br > b:
        return "Bearish"
    return None  # truly neutral / unknown → skip


def _fetch_page(symbol: str, max_id: int | None = None) -> dict:
    url = f"{_BASE_URL}/{symbol}.json?limit=30"
    if max_id:
        url += f"&max={max_id}"
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    try:
        with urllib.request.urlopen(req, context=_SSL_CTX, timeout=10) as r:
            return json.loads(r.read())
    except Exception as e:
        print(f"    {symbol} fetch error: {e}")
        return {}


def fetch_messages(symbol: str) -> list[dict]:
    """Fetch up to PAGES_PER_SYMBOL pages of messages."""
    messages: list[dict] = []
    data = _fetch_page(symbol)
    if not data:
        return []
    messages.extend(data.get("messages") or [])
    cursor = data.get("cursor") or {}

    for _ in range(PAGES_PER_SYMBOL - 1):
        if not cursor.get("more"):
            break
        time.sleep(0.3)
        data = _fetch_page(symbol, max_id=cursor["max"])
        if not data:
            break
        messages.extend(data.get("messages") or [])
        cursor = data.get("cursor") or {}

    return messages


def aggregate_by_date(messages: list[dict]) -> dict[str, dict]:
    """
    Aggregate messages into per-date buckets.
    Returns {date_str: {bull, bear, total, tagged}} dict.
    """
    buckets: dict[str, dict] = {}

    for m in messages:
        ts = m.get("created_at", "")
        date_str = ts[:10]
        if not date_str:
            continue

        ent = m.get("entities") or {}
        explicit = (ent.get("sentiment") or {}).get("basic")  # "Bullish"/"Bearish"/None

        if explicit in ("Bullish", "Bearish"):
            sentiment = explicit
            tagged = 1
        else:
            sentiment = _kw_sentiment(m.get("body") or "")
            tagged = 0

        b = buckets.setdefault(date_str, {"bull": 0, "bear": 0, "total": 0, "tagged": 0})
        b["total"] += 1
        b["tagged"] += tagged
        if sentiment == "Bullish":
            b["bull"] += 1
        elif sentiment == "Bearish":
            b["bear"] += 1

    return buckets


def compute_features(symbol: str, buckets: dict[str, dict]) -> list[dict]:
    now = datetime.now(UTC)
    records: list[dict] = []
    for date_str, b in buckets.items():
        bull, bear, total, tagged = b["bull"], b["bear"], b["total"], b["tagged"]
        tagged_sum = bull + bear  # actual bull+bear count (explicit + kw)
        bull_ratio = bull / tagged_sum if tagged_sum > 0 else None
        sent_score = (bull - bear) / tagged_sum if tagged_sum > 0 else None
        records.append({
            "symbol": symbol,
            "date": date_str,
            "retail_msg_count": total,
            "retail_tagged_count": tagged,
            "retail_bull_count": bull,
            "retail_bear_count": bear,
            "retail_bull_ratio": bull_ratio,
            "retail_sent_score": sent_score,
            "collectedAt": now,
        })
    return records


def load_universe() -> list[str]:
    client = MongoClient(MONGO_URI)
    docs = list(client[DB_NAME][UNIVERSE_COLLECTION].find({}, {"symbol": 1, "_id": 0}))
    return sorted(d["symbol"] for d in docs)


def main() -> None:
    print("=== Retail sentiment collector (StockTwits) ===")
    symbols = load_universe()
    print(f"Universe: {len(symbols)} symbols  pages/symbol={PAGES_PER_SYMBOL}")

    client = MongoClient(MONGO_URI)
    col = client[DB_NAME][COLLECTION]
    col.create_index([("symbol", 1), ("date", 1)], unique=True, background=True)

    total_up = total_mod = 0
    for sym in symbols:
        msgs = fetch_messages(sym)
        if not msgs:
            print(f"  {sym:6} no messages")
            time.sleep(SLEEP_BETWEEN)
            continue

        buckets = aggregate_by_date(msgs)
        records = compute_features(sym, buckets)

        ops = [
            UpdateOne(
                {"symbol": r["symbol"], "date": r["date"]},
                {"$set": r},
                upsert=True,
            )
            for r in records
        ]
        res = col.bulk_write(ops)
        total_up += res.upserted_count
        total_mod += res.modified_count

        dates = sorted(buckets.keys(), reverse=True)
        summary = f"up={res.upserted_count} mod={res.modified_count}"
        if dates:
            latest = buckets[dates[0]]
            pct = latest["bull"] / (latest["bull"] + latest["bear"]) if (latest["bull"] + latest["bear"]) > 0 else None
            pct_s = f"{pct:.0%}" if pct is not None else "N/A"
            summary += f" | {dates[0]} msgs={latest['total']} bull={pct_s}"
        print(f"  {sym:6} {len(msgs)} msgs → {len(records)} dates  {summary}")

        time.sleep(SLEEP_BETWEEN)

    print(f"\nDone. total upserted={total_up} modified={total_mod}")


if __name__ == "__main__":
    main()
