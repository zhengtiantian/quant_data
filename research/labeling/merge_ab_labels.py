#!/usr/bin/env python3
"""Merge enrich Pass-A / Pass-B sentiment into llm_sentiment_final + llm_disagreement.

Lightweight replacement for the never-implemented snorkel step. With only two
label sources a probabilistic label model (snorkel) buys nothing, so we combine
the two LLM sentiments directly:

    llm_sentiment_final = W_A * llm_sentiment_a + W_B * llm_sentiment_b   (default 0.5 / 0.5)
    llm_disagreement    = abs(llm_sentiment_a - llm_sentiment_b)

feature rebuild (daily_symbol_features.py) reads exactly these two fields plus
llm_signal_strength_a / llm_event_type_a (taken straight from pass A).

Incremental: only touches docs that have BOTH sentiment_a and sentiment_b but
are missing final OR disagreement — so new articles get final+disagreement, and
any historical docs that have final but never got disagreement are backfilled.
No model / LM Studio needed; this is pure arithmetic and runs in minutes.

Env:
  LOCAL_MONGO_URI / MONGO_URI   mongo connection (default local :37018)
  FEATURE_DB_NAME               db (default quant_data)
  MERGE_COLLECTION              collection (default news_articles_company_matched_v2)
  MERGE_WEIGHT_A / MERGE_WEIGHT_B   blend weights (default 0.5 / 0.5)
  MERGE_BATCH_SIZE              bulk batch size (default 2000)
"""

from __future__ import annotations

import os

from pymongo import MongoClient, UpdateOne

MONGO_URI = os.getenv("LOCAL_MONGO_URI", os.getenv("MONGO_URI", "mongodb://root:root@127.0.0.1:37018/"))
DB_NAME = os.getenv("FEATURE_DB_NAME", "quant_data")
COLLECTION = os.getenv("MERGE_COLLECTION", "news_articles_company_matched_v2")
W_A = float(os.getenv("MERGE_WEIGHT_A", "0.5"))
W_B = float(os.getenv("MERGE_WEIGHT_B", "0.5"))
BATCH = int(os.getenv("MERGE_BATCH_SIZE", "2000"))


def main() -> None:
    col = MongoClient(MONGO_URI)[DB_NAME][COLLECTION]
    query = {
        "llm_sentiment_a": {"$exists": True, "$ne": None},
        "llm_sentiment_b": {"$exists": True, "$ne": None},
        "$or": [
            {"llm_sentiment_final": {"$exists": False}},
            {"llm_disagreement": {"$exists": False}},
        ],
    }
    total = col.count_documents(query)
    print(f"merge_ab_labels: {total:,} docs to merge "
          f"(collection={COLLECTION}, weights a={W_A} b={W_B})")
    if total == 0:
        print("nothing to merge — all docs already have final + disagreement.")
        return

    cursor = col.find(
        query, {"_id": 1, "llm_sentiment_a": 1, "llm_sentiment_b": 1}
    ).batch_size(BATCH)

    ops: list[UpdateOne] = []
    processed = 0
    for doc in cursor:
        a = float(doc["llm_sentiment_a"])
        b = float(doc["llm_sentiment_b"])
        ops.append(UpdateOne(
            {"_id": doc["_id"]},
            {"$set": {
                "llm_sentiment_final": round(W_A * a + W_B * b, 4),
                "llm_disagreement": round(abs(a - b), 4),
            }},
        ))
        if len(ops) >= BATCH:
            col.bulk_write(ops, ordered=False)
            processed += len(ops)
            ops = []
            print(f"  merged {processed:,}/{total:,}")

    if ops:
        col.bulk_write(ops, ordered=False)
        processed += len(ops)

    print(f"done: merged {processed:,} docs")


if __name__ == "__main__":
    main()
