#!/usr/bin/env python3
"""Run company-match cleaning on Mongo news articles using the collector RuleManager path."""

from __future__ import annotations

import os
import sys
import threading
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from datetime import UTC, datetime
from pathlib import Path
from typing import Iterable

from bson import ObjectId
from dotenv import load_dotenv
from pymongo import MongoClient, UpdateOne


CURRENT = Path(__file__).resolve()
ROOT = CURRENT.parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
GLOBAL_ENV = ROOT / ".env"
load_dotenv(GLOBAL_ENV, override=False)

from news_collectors.gdelt.special_rules import RuleManager

MONGO_URI = os.getenv("LOCAL_MONGO_URI", "mongodb://root:root@127.0.0.1:37018/")
DB_NAME = os.getenv("FEATURE_DB_NAME", "quant_data")
COLLECTION = os.getenv("FEATURE_NEWS_COLLECTION", "news_articles")

MATCH_FIELD = os.getenv("COMPANY_MATCH_FIELD", "company_match_v1")
MATCH_VERSION = os.getenv("COMPANY_MATCH_VERSION", "v1")
MATCH_BATCH_SIZE = int(os.getenv("COMPANY_MATCH_BATCH_SIZE", "1000"))
MATCH_WORKERS = int(os.getenv("COMPANY_MATCH_WORKERS", "4"))
MATCH_PROGRESS_EVERY = int(os.getenv("COMPANY_MATCH_PROGRESS_EVERY", "1000"))
MATCH_LIMIT = int(os.getenv("COMPANY_MATCH_LIMIT", "0"))
MATCH_FORCE = os.getenv("COMPANY_MATCH_FORCE", "false").lower() == "true"
MATCH_DATE_PREFIXES = [s.strip() for s in os.getenv("COMPANY_MATCH_DATE_PREFIXES", "").split(",") if s.strip()]

_thread_local = threading.local()


def get_client() -> MongoClient:
    return MongoClient(MONGO_URI)


def get_rule_manager() -> RuleManager:
    manager = getattr(_thread_local, "rule_manager", None)
    if manager is None:
        manager = RuleManager()
        _thread_local.rule_manager = manager
    return manager


def build_query(last_id: ObjectId | None = None) -> dict:
    query: dict = {
        "symbol": {"$exists": True, "$ne": None},
        "title": {"$exists": True, "$ne": None},
    }
    if not MATCH_FORCE:
        query[MATCH_FIELD] = {"$exists": False}
    if last_id is not None:
        query["_id"] = {"$gt": last_id}
    if MATCH_DATE_PREFIXES:
        query["$or"] = [{"date": {"$regex": f"^{prefix}"}} for prefix in MATCH_DATE_PREFIXES]
    return query


def fetch_batch(col, last_id: ObjectId | None) -> list[dict]:
    cursor = (
        col.find(
            build_query(last_id),
            {
                "_id": 1,
                "symbol": 1,
                "name": 1,
                "title": 1,
                "content": 1,
                "url": 1,
                "date": 1,
                "data_quality": 1,
                "note": 1,
            },
        )
        .sort("_id", 1)
        .limit(MATCH_BATCH_SIZE)
    )
    return list(cursor)


def rescore_doc(doc: dict) -> dict:
    manager = get_rule_manager()
    symbol = doc.get("symbol") or ""
    article = {
        "title": doc.get("title") or "",
        "content": doc.get("content") or "",
        "url": doc.get("url") or "",
        "date": doc.get("date") or "",
        "data_quality": doc.get("data_quality") or "unknown",
        "note": doc.get("note") or "",
        "name": doc.get("name") or symbol,
        "symbol": symbol,
    }
    matched = manager.should_include(symbol, article) if symbol else False
    return {
        "_id": doc["_id"],
        "symbol": symbol,
        "matched": bool(matched),
    }


def save_results(col, results: Iterable[dict]) -> int:
    now_iso = datetime.now(UTC).isoformat()
    ops = []
    for row in results:
        ops.append(
            UpdateOne(
                {"_id": row["_id"]},
                {
                    "$set": {
                        MATCH_FIELD: {
                            "matched": row["matched"],
                            "version": MATCH_VERSION,
                            "scoredAt": now_iso,
                            "engine": "rule_manager",
                        }
                    }
                },
            )
        )
    if not ops:
        return 0
    result = col.bulk_write(ops, ordered=False)
    return result.modified_count + result.upserted_count


def run() -> None:
    client = get_client()
    col = client[DB_NAME][COLLECTION]

    total_to_process = col.count_documents(build_query())
    print("=== Company match rescore start ===")
    print(f"collection={COLLECTION} field={MATCH_FIELD} version={MATCH_VERSION}")
    print(f"workers={MATCH_WORKERS} batch_size={MATCH_BATCH_SIZE} limit={MATCH_LIMIT or 'ALL'}")
    print(f"date_prefixes={MATCH_DATE_PREFIXES or 'ALL'}")
    print(f"pending_docs={total_to_process:,}")

    processed = 0
    matched_yes = 0
    matched_no = 0
    last_id: ObjectId | None = None

    with ThreadPoolExecutor(max_workers=MATCH_WORKERS, thread_name_prefix="company-match") as executor:
        while True:
            batch = fetch_batch(col, last_id)
            if not batch:
                break

            futures = {executor.submit(rescore_doc, doc): doc["_id"] for doc in batch}
            batch_results = []
            while futures:
                done, _ = wait(futures.keys(), return_when=FIRST_COMPLETED)
                for fut in done:
                    futures.pop(fut, None)
                    row = fut.result()
                    batch_results.append(row)
                    processed += 1
                    matched_yes += int(row["matched"])
                    matched_no += int(not row["matched"])
                    if processed % MATCH_PROGRESS_EVERY == 0:
                        print(
                            f"processed={processed:,} matched={matched_yes:,} rejected={matched_no:,} "
                            f"match_rate={(matched_yes / processed * 100):.1f}%"
                        )
                    if MATCH_LIMIT and processed >= MATCH_LIMIT:
                        break
                if MATCH_LIMIT and processed >= MATCH_LIMIT:
                    break

            save_results(col, batch_results)
            last_id = batch[-1]["_id"]

            if MATCH_LIMIT and processed >= MATCH_LIMIT:
                break

    print("=== Company match rescore done ===")
    print(f"processed={processed:,} matched={matched_yes:,} rejected={matched_no:,}")
    if processed:
        print(f"match_rate={matched_yes / processed * 100:.1f}%")


if __name__ == "__main__":
    run()
