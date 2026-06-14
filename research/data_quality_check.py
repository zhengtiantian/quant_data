#!/usr/bin/env python3
"""Daily ETL data-quality checks (C.7).

Runs after the pipeline and records the health of each key collection into
`data_quality_checks` (one doc per run). Each check has a status of
ok / warn / fail plus a metric and message, so failures are visible instead
of silently corrupting downstream features.

Checks: news volume, price freshness, feature freshness, feature NULL rate,
feature coverage, signal freshness.

Run:
  LOCAL_MONGO_URI="mongodb://root:root@127.0.0.1:37018/" \
  .venv311/bin/python research/data_quality_check.py
"""

from __future__ import annotations

import os
import sys
from datetime import datetime, timedelta, timezone

from pymongo import MongoClient

MONGO_URI = os.getenv("LOCAL_MONGO_URI", "mongodb://root:root@127.0.0.1:37018/")
DB_NAME = "quant_data"

PRICE_STALE_DAYS = int(os.getenv("DQ_PRICE_STALE_DAYS", "4"))
FEATURE_STALE_DAYS = int(os.getenv("DQ_FEATURE_STALE_DAYS", "4"))
MIN_FEATURE_SYMBOLS = int(os.getenv("DQ_MIN_FEATURE_SYMBOLS", "20"))
MAX_NULL_RATE = float(os.getenv("DQ_MAX_NULL_RATE", "0.5"))
MIN_NEWS_2D = int(os.getenv("DQ_MIN_NEWS_2D", "5"))

NULL_KEY_FIELDS = ["quality_score", "full_ratio", "close", "past_ret_20d"]


def _check(name, status, metric, message):
    return {"name": name, "status": status, "metric": metric, "message": message}


def _days_since(date_str: str) -> int | None:
    try:
        d = datetime.fromisoformat(str(date_str)[:10]).date()
        return (datetime.now(timezone.utc).date() - d).days
    except Exception:
        return None


def check_price_freshness(db):
    doc = db.stock_prices_history.find_one(sort=[("timestamp", -1)])
    if not doc:
        return _check("price_freshness", "fail", None, "stock_prices_history is empty")
    latest = str(doc["timestamp"])[:10]
    age = _days_since(latest)
    status = "ok" if (age is not None and age <= PRICE_STALE_DAYS) else "warn"
    return _check("price_freshness", status, latest, f"latest price bar {latest} ({age}d old)")


def check_feature_freshness(db):
    doc = db.daily_symbol_features.find_one(sort=[("trade_date", -1)])
    if not doc:
        return _check("feature_freshness", "fail", None, "daily_symbol_features is empty")
    latest = str(doc["trade_date"])[:10]
    age = _days_since(latest)
    status = "ok" if (age is not None and age <= FEATURE_STALE_DAYS) else "warn"
    return _check("feature_freshness", status, latest, f"latest feature date {latest} ({age}d old)")


def check_feature_coverage(db):
    doc = db.daily_symbol_features.find_one(sort=[("trade_date", -1)])
    if not doc:
        return _check("feature_coverage", "fail", 0, "no features")
    latest = doc["trade_date"]
    n = db.daily_symbol_features.count_documents({"trade_date": latest})
    status = "ok" if n >= MIN_FEATURE_SYMBOLS else "warn"
    return _check("feature_coverage", status, n, f"{n} symbols on {str(latest)[:10]} (min {MIN_FEATURE_SYMBOLS})")


def check_feature_null_rate(db):
    doc = db.daily_symbol_features.find_one(sort=[("trade_date", -1)])
    if not doc:
        return _check("feature_null_rate", "fail", None, "no features")
    latest = doc["trade_date"]
    rows = list(db.daily_symbol_features.find({"trade_date": latest},
                                              {f: 1 for f in NULL_KEY_FIELDS}))
    total = len(rows) or 1
    worst_field, worst_rate = None, 0.0
    for f in NULL_KEY_FIELDS:
        nulls = sum(1 for r in rows if r.get(f) is None)
        rate = nulls / total
        if rate > worst_rate:
            worst_field, worst_rate = f, rate
    status = "ok" if worst_rate <= MAX_NULL_RATE else "warn"
    detail = "no NULLs in key fields" if worst_field is None \
        else f"worst NULL rate {worst_rate:.0%} on '{worst_field}' (max {MAX_NULL_RATE:.0%})"
    return _check("feature_null_rate", status, round(worst_rate, 3), detail)


def check_signal_freshness(db):
    doc = db.daily_signals.find_one(sort=[("trade_date", -1)])
    if not doc:
        return _check("signal_freshness", "fail", None, "daily_signals is empty")
    latest = str(doc["trade_date"])[:10]
    longs = db.daily_signals.count_documents({"trade_date": doc["trade_date"], "signal_type": "LONG"})
    status = "ok" if longs > 0 else "warn"
    return _check("signal_freshness", status, longs, f"{longs} LONG signals on {latest}")


def check_news_volume(db):
    now = datetime.now(timezone.utc)
    lo = (now - timedelta(days=2)).strftime("%Y%m%d000000")
    hi = now.strftime("%Y%m%d235959")  # exclude corrupt future dates
    n = db.news_articles.count_documents({"date": {"$gte": lo, "$lte": hi}})
    status = "ok" if n >= MIN_NEWS_2D else "warn"
    return _check("news_volume", status, n, f"{n} articles in last 2d (min {MIN_NEWS_2D})")


def main():
    client = MongoClient(MONGO_URI)
    db = client[DB_NAME]

    checks = [
        check_news_volume(db),
        check_price_freshness(db),
        check_feature_freshness(db),
        check_feature_coverage(db),
        check_feature_null_rate(db),
        check_signal_freshness(db),
    ]
    overall = "fail" if any(c["status"] == "fail" for c in checks) else \
              "warn" if any(c["status"] == "warn" for c in checks) else "ok"

    doc = {
        "run_at": datetime.now(timezone.utc),
        "overall": overall,
        "checks": checks,
    }
    db.data_quality_checks.create_index([("run_at", -1)], background=True)
    db.data_quality_checks.insert_one(doc)

    icon = {"ok": "✅", "warn": "⚠️", "fail": "❌"}
    print(f"Data quality: {icon[overall]} {overall.upper()}")
    for c in checks:
        print(f"  {icon[c['status']]} {c['name']:<20} {c['message']}")

    sys.exit(1 if overall == "fail" else 0)


if __name__ == "__main__":
    main()
