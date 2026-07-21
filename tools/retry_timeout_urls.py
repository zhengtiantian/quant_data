"""
Retry article URLs that failed to fetch due to network timeouts.
Reads pending retry records from MongoDB url_retry_queue collection; on successful fetch, writes to the target collection.

Usage:
  python tools/retry_timeout_urls.py
  python tools/retry_timeout_urls.py --dst_collection news_articles_backfill_26 --batch 50
"""
import os
import sys
import argparse
import time
from datetime import datetime, timezone
from concurrent.futures import ThreadPoolExecutor, as_completed

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pymongo import MongoClient, UpdateOne
from collectors.news.gdelt.historical_collector import (
    _extract_article_payload,
    _build_article_base_data,
    MONGO_URI,
    DB_NAME,
)

RETRY_COLLECTION = "url_retry_queue"
MAX_RETRIES = 3
FETCH_TIMEOUT = int(os.getenv("ARTICLE_REQUEST_TIMEOUT", "10"))
WORKERS = int(os.getenv("RETRY_WORKERS", "5"))


def get_db():
    return MongoClient(MONGO_URI)[DB_NAME]


def fetch_one(doc):
    url = doc["url"]
    company = {
        "symbol": doc.get("symbol", ""),
        "name": doc.get("company_name", ""),
    }
    row = {
        "URL": url,
        "Title": doc.get("title", ""),
        "Date": doc.get("date", ""),
        "Timestamp": doc.get("timestamp", ""),
        "PublishedAt": doc.get("publishedAt"),
    }
    try:
        result = _extract_article_payload(row, company)
        return doc["_id"], url, result, None
    except Exception as e:
        return doc["_id"], url, None, str(e)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dst_collection", type=str, default=None)
    parser.add_argument("--batch", type=int, default=100)
    args = parser.parse_args()

    db = get_db()
    queue_col = db[RETRY_COLLECTION]

    query = {"status": "pending", "retry_count": {"$lt": MAX_RETRIES}}
    total = queue_col.count_documents(query)
    print(f"📋 Found {total} URLs pending retry")
    if total == 0:
        return

    docs = list(queue_col.find(query).sort("added_at", 1).limit(args.batch))
    print(f"🔄 Retrying {len(docs)} URLs with {WORKERS} workers...\n")

    success = fail = 0
    with ThreadPoolExecutor(max_workers=WORKERS) as executor:
        futures = {executor.submit(fetch_one, doc): doc for doc in docs}
        for future in as_completed(futures):
            doc_id, url, result, err = future.result()
            now = datetime.now(timezone.utc).isoformat()
            doc = futures[future]
            dst_col_name = args.dst_collection or doc.get("dst_collection", "news_articles")

            if result:
                # Write to target collection
                dst_col = db[dst_col_name]
                dst_col.update_one(
                    {"url": result["url"]},
                    {"$setOnInsert": result},
                    upsert=True,
                )
                # Mark as done
                queue_col.update_one(
                    {"_id": doc_id},
                    {"$set": {"status": "done", "last_retry_at": now}},
                )
                success += 1
                print(f"✅ {url[:80]}")
            else:
                new_count = doc.get("retry_count", 0) + 1
                new_status = "failed" if new_count >= MAX_RETRIES else "pending"
                queue_col.update_one(
                    {"_id": doc_id},
                    {"$set": {
                        "status": new_status,
                        "retry_count": new_count,
                        "last_retry_at": now,
                        "last_error": str(err)[:200],
                    }},
                )
                fail += 1
                print(f"❌ [{new_count}/{MAX_RETRIES}] {url[:80]} — {err}")

    print(f"\n✅ Done: success={success}, failed={fail}, remaining={total - len(docs)}")


if __name__ == "__main__":
    main()
