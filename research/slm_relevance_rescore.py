#!/usr/bin/env python3
"""Rescore all news articles with the local small model for company relevance."""

from __future__ import annotations

import os
import sys
import threading
import time
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

from news_collectors.gdelt.special_rules.slm_filter import SLMFilter

MONGO_URI = os.getenv("LOCAL_MONGO_URI", "mongodb://root:root@127.0.0.1:37018/")
DB_NAME = os.getenv("FEATURE_DB_NAME", "quant_data")
COLLECTION = os.getenv("FEATURE_NEWS_COLLECTION", "news_articles")

RESCORE_FIELD = os.getenv("SLM_RESCORE_FIELD", "slm_relevance_v1")
RESCORE_VERSION = os.getenv("SLM_RESCORE_VERSION", "v1")
RESCORE_BATCH_SIZE = int(os.getenv("SLM_RESCORE_BATCH_SIZE", "200"))
RESCORE_WORKERS = int(os.getenv("SLM_RESCORE_WORKERS", os.getenv("SLM_MAX_CONCURRENCY", "4")))
RESCORE_PROGRESS_EVERY = int(os.getenv("SLM_RESCORE_PROGRESS_EVERY", "100"))
RESCORE_LIMIT = int(os.getenv("SLM_RESCORE_LIMIT", "0"))
RESCORE_SYMBOLS = [s.strip().upper() for s in os.getenv("SLM_RESCORE_SYMBOLS", "").split(",") if s.strip()]
RESCORE_DATA_QUALITY = [q.strip() for q in os.getenv("SLM_RESCORE_DATA_QUALITY", "").split(",") if q.strip()]
RESCORE_FORCE = os.getenv("SLM_RESCORE_FORCE", "false").lower() == "true"
RESCORE_CONTENT_CHARS = int(os.getenv("SLM_RESCORE_CONTENT_CHARS", "3000"))
RESCORE_SLEEP_MS = int(os.getenv("SLM_RESCORE_SLEEP_MS", "0"))

_thread_local = threading.local()


def get_client() -> MongoClient:
    return MongoClient(MONGO_URI)


def get_slm() -> SLMFilter:
    slm = getattr(_thread_local, "slm", None)
    if slm is None:
        slm = SLMFilter(enabled=True)
        _thread_local.slm = slm
    return slm


def build_query(last_id: ObjectId | None = None) -> dict:
    query: dict = {"symbol": {"$exists": True, "$ne": None}, "title": {"$exists": True, "$ne": None, "$ne": ""}}
    if not RESCORE_FORCE:
        query[RESCORE_FIELD] = {"$exists": False}
    if last_id is not None:
        query["_id"] = {"$gt": last_id}
    if RESCORE_SYMBOLS:
        query["symbol"] = {"$in": RESCORE_SYMBOLS}
    if RESCORE_DATA_QUALITY:
        query["data_quality"] = {"$in": RESCORE_DATA_QUALITY}
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
            },
        )
        .sort("_id", 1)
        .limit(RESCORE_BATCH_SIZE)
    )
    return list(cursor)


def rescore_doc(doc: dict) -> dict:
    title = (doc.get("title") or "").strip()
    content = (doc.get("content") or "")[:RESCORE_CONTENT_CHARS]
    quality = doc.get("data_quality") or "unknown"
    basis = "title+content" if content else "title_only"
    slm = get_slm()
    relevant = slm.is_relevant(
        symbol=doc.get("symbol") or "",
        company_name=doc.get("name") or doc.get("symbol") or "",
        title=title,
        content=content,
    )
    if RESCORE_SLEEP_MS > 0:
        time.sleep(RESCORE_SLEEP_MS / 1000.0)
    return {
        "_id": doc["_id"],
        "symbol": doc.get("symbol"),
        "data_quality": quality,
        "basis": basis,
        "relevant": bool(relevant),
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
                        RESCORE_FIELD: {
                            "relevant": row["relevant"],
                            "basis": row["basis"],
                            "version": RESCORE_VERSION,
                            "model": os.getenv("SLM_MODEL", "qwen3.5-4b"),
                            "provider": os.getenv("SLM_PROVIDER", "lmstudio"),
                            "scoredAt": now_iso,
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

    total_query = build_query()
    total_to_process = col.count_documents(total_query)
    print(f"=== SLM relevance rescore start ===")
    print(f"collection={COLLECTION} field={RESCORE_FIELD} version={RESCORE_VERSION}")
    print(f"workers={RESCORE_WORKERS} batch_size={RESCORE_BATCH_SIZE} limit={RESCORE_LIMIT or 'ALL'}")
    print(f"sleep_ms={RESCORE_SLEEP_MS}")
    print(f"pending_docs={total_to_process:,}")

    processed = 0
    relevant_yes = 0
    relevant_no = 0
    last_id: ObjectId | None = None

    with ThreadPoolExecutor(max_workers=RESCORE_WORKERS, thread_name_prefix="slm-rescore") as executor:
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
                    relevant_yes += int(row["relevant"])
                    relevant_no += int(not row["relevant"])
                    if processed % RESCORE_PROGRESS_EVERY == 0:
                        print(
                            f"processed={processed:,} yes={relevant_yes:,} no={relevant_no:,} "
                            f"yes_rate={(relevant_yes / processed * 100):.1f}%"
                        )
                    if RESCORE_LIMIT and processed >= RESCORE_LIMIT:
                        break
                if RESCORE_LIMIT and processed >= RESCORE_LIMIT:
                    break

            save_results(col, batch_results)
            last_id = batch[-1]["_id"]

            if RESCORE_LIMIT and processed >= RESCORE_LIMIT:
                break

    print("=== SLM relevance rescore done ===")
    print(f"processed={processed:,} yes={relevant_yes:,} no={relevant_no:,}")
    if processed:
        print(f"yes_rate={relevant_yes / processed * 100:.1f}%")


if __name__ == "__main__":
    run()
