#!/usr/bin/env python3
"""Build a matched-news collection from raw news_articles using RuleManager + SLM."""

from __future__ import annotations

import os
import sys
import threading
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from bson import ObjectId
from dotenv import load_dotenv
from pymongo import MongoClient, ReplaceOne, UpdateOne


CURRENT = Path(__file__).resolve()
ROOT = CURRENT.parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
GLOBAL_ENV = ROOT / ".env"
load_dotenv(GLOBAL_ENV, override=False)

from news_collectors.gdelt.special_rules import RuleManager

MONGO_URI = os.getenv("LOCAL_MONGO_URI", "mongodb://root:root@127.0.0.1:37018/")
DB_NAME = os.getenv("FEATURE_DB_NAME", "quant_data")
SOURCE_COLLECTION = os.getenv("FEATURE_NEWS_COLLECTION", "news_articles")
TARGET_COLLECTION = os.getenv("COMPANY_MATCH_TARGET_COLLECTION", "news_articles_company_matched_v1")
PROGRESS_COLLECTION = os.getenv("COMPANY_MATCH_PROGRESS_COLLECTION", "company_match_jobs")

MATCH_FIELD = os.getenv("COMPANY_MATCH_FIELD", "company_match_v1")
MATCH_VERSION = os.getenv("COMPANY_MATCH_VERSION", "v1")
MATCH_JOB_NAME = os.getenv("COMPANY_MATCH_JOB_NAME", f"{TARGET_COLLECTION}:{MATCH_VERSION}")
MATCH_BATCH_SIZE = int(os.getenv("COMPANY_MATCH_BATCH_SIZE", "1000"))
MATCH_WORKERS = int(os.getenv("COMPANY_MATCH_WORKERS", "4"))
MATCH_PROGRESS_EVERY = int(os.getenv("COMPANY_MATCH_PROGRESS_EVERY", "1000"))
MATCH_LIMIT = int(os.getenv("COMPANY_MATCH_LIMIT", "0"))
MATCH_FORCE = os.getenv("COMPANY_MATCH_FORCE", "false").lower() == "true"
MATCH_RESUME = os.getenv("COMPANY_MATCH_RESUME", "true").lower() == "true"
MATCH_DATE_PREFIXES = [s.strip() for s in os.getenv("COMPANY_MATCH_DATE_PREFIXES", "").split(",") if s.strip()]

_thread_local = threading.local()


def now_iso() -> str:
    return datetime.now(UTC).isoformat()


def get_client() -> MongoClient:
    return MongoClient(MONGO_URI)


def get_rule_manager() -> RuleManager:
    manager = getattr(_thread_local, "rule_manager", None)
    if manager is None:
        manager = RuleManager()
        _thread_local.rule_manager = manager
    return manager


def load_progress(progress_col) -> dict[str, Any] | None:
    return progress_col.find_one({"_id": MATCH_JOB_NAME})


def save_progress(
    progress_col,
    *,
    status: str,
    processed: int,
    matched_yes: int,
    matched_no: int,
    last_id: ObjectId | None,
    extra: dict[str, Any] | None = None,
) -> None:
    payload: dict[str, Any] = {
        "_id": MATCH_JOB_NAME,
        "jobName": MATCH_JOB_NAME,
        "sourceCollection": SOURCE_COLLECTION,
        "targetCollection": TARGET_COLLECTION,
        "matchField": MATCH_FIELD,
        "version": MATCH_VERSION,
        "status": status,
        "processed": processed,
        "matched": matched_yes,
        "rejected": matched_no,
        "matchRate": round((matched_yes / processed * 100), 2) if processed else 0.0,
        "lastId": last_id,
        "updatedAt": now_iso(),
    }
    if extra:
        payload.update(extra)
    progress_col.update_one({"_id": MATCH_JOB_NAME}, {"$set": payload}, upsert=True)


def build_query(last_id: ObjectId | None = None) -> dict[str, Any]:
    query: dict[str, Any] = {
        "symbol": {"$exists": True, "$ne": None},
        "title": {"$exists": True, "$ne": None},
    }
    if last_id is not None:
        query["_id"] = {"$gt": last_id}
    if MATCH_DATE_PREFIXES:
        query["$or"] = [{"date": {"$regex": f"^{prefix}"}} for prefix in MATCH_DATE_PREFIXES]
    return query


def fetch_batch(source_col, last_id: ObjectId | None) -> list[dict[str, Any]]:
    cursor = (
        source_col.find(
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
                "source": 1,
                "platform": 1,
                "author": 1,
                "published_at": 1,
            },
        )
        .sort("_id", 1)
        .limit(MATCH_BATCH_SIZE)
    )
    return list(cursor)


def rescore_doc(doc: dict[str, Any]) -> dict[str, Any]:
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
        "doc": doc,
    }


def build_target_doc(row: dict[str, Any]) -> dict[str, Any]:
    src = dict(row["doc"])
    src["_id"] = row["_id"]
    src["raw_id"] = row["_id"]
    src[MATCH_FIELD] = {
        "matched": row["matched"],
        "version": MATCH_VERSION,
        "scoredAt": now_iso(),
        "engine": "rule_manager",
    }
    return src


def save_results(source_col, target_col, results: list[dict[str, Any]]) -> tuple[int, int]:
    if not results:
        return 0, 0

    scored_at = now_iso()
    source_ops = []
    target_ops = []

    for row in results:
        source_ops.append(
            UpdateOne(
                {"_id": row["_id"]},
                {
                    "$set": {
                        MATCH_FIELD: {
                            "matched": row["matched"],
                            "version": MATCH_VERSION,
                            "scoredAt": scored_at,
                            "engine": "rule_manager",
                            "targetCollection": TARGET_COLLECTION,
                        }
                    }
                },
            )
        )
        if row["matched"]:
            target_doc = build_target_doc(row)
            target_doc[MATCH_FIELD]["scoredAt"] = scored_at
            target_ops.append(ReplaceOne({"_id": row["_id"]}, target_doc, upsert=True))

    source_col.bulk_write(source_ops, ordered=False)
    matched_written = 0
    if target_ops:
        result = target_col.bulk_write(target_ops, ordered=False)
        matched_written = result.upserted_count + result.modified_count

    return len(source_ops), matched_written


def run() -> None:
    client = get_client()
    source_col = client[DB_NAME][SOURCE_COLLECTION]
    target_col = client[DB_NAME][TARGET_COLLECTION]
    progress_col = client[DB_NAME][PROGRESS_COLLECTION]

    progress_doc = load_progress(progress_col) if MATCH_RESUME and not MATCH_FORCE else None
    last_id = progress_doc.get("lastId") if progress_doc else None
    processed = int(progress_doc.get("processed", 0)) if progress_doc else 0
    matched_yes = int(progress_doc.get("matched", 0)) if progress_doc else 0
    matched_no = int(progress_doc.get("rejected", 0)) if progress_doc else 0

    total_to_process = source_col.count_documents(build_query(last_id))
    print("=== Company match build start ===")
    print(f"source={SOURCE_COLLECTION} target={TARGET_COLLECTION} field={MATCH_FIELD} version={MATCH_VERSION}")
    print(f"job={MATCH_JOB_NAME}")
    print(f"workers={MATCH_WORKERS} batch_size={MATCH_BATCH_SIZE} limit={MATCH_LIMIT or 'ALL'} resume={MATCH_RESUME} force={MATCH_FORCE}")
    print(f"date_prefixes={MATCH_DATE_PREFIXES or 'ALL'}")
    print(f"pending_docs={total_to_process:,}")
    if progress_doc:
        print(
            f"resume_from={last_id} prior_processed={processed:,} "
            f"prior_matched={matched_yes:,} prior_rejected={matched_no:,}"
        )

    save_progress(
        progress_col,
        status="running",
        processed=processed,
        matched_yes=matched_yes,
        matched_no=matched_no,
        last_id=last_id,
        extra={"startedAt": progress_doc.get("startedAt", now_iso()) if progress_doc else now_iso()},
    )

    with ThreadPoolExecutor(max_workers=MATCH_WORKERS, thread_name_prefix="company-match") as executor:
        while True:
            batch = fetch_batch(source_col, last_id)
            if not batch:
                break

            futures = {executor.submit(rescore_doc, doc): doc["_id"] for doc in batch}
            batch_results: list[dict[str, Any]] = []

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

            save_results(source_col, target_col, batch_results)
            last_id = batch[-1]["_id"]
            save_progress(
                progress_col,
                status="running",
                processed=processed,
                matched_yes=matched_yes,
                matched_no=matched_no,
                last_id=last_id,
            )

            if MATCH_LIMIT and processed >= MATCH_LIMIT:
                break

    save_progress(
        progress_col,
        status="completed",
        processed=processed,
        matched_yes=matched_yes,
        matched_no=matched_no,
        last_id=last_id,
        extra={"completedAt": now_iso()},
    )

    print("=== Company match build done ===")
    print(f"processed={processed:,} matched={matched_yes:,} rejected={matched_no:,}")
    if processed:
        print(f"match_rate={matched_yes / processed * 100:.1f}%")


if __name__ == "__main__":
    run()
