#!/usr/bin/env python3
"""Stage 3.5.2 — Snorkel Label Model aggregation + disagreement feature.

Reads llm_sentiment_a / llm_sentiment_b from MongoDB, runs Snorkel Label Model
to produce a single consensus float, and stores:
  - llm_sentiment_final   : float, P(positive) - P(negative)
  - llm_disagreement      : float, |sentiment_a - sentiment_b|  (方案三 feature)
  - llm_label_model_probs : [P(neg), P(neutral), P(pos)]

Usage:
  LOCAL_MONGO_URI="mongodb://root:root@127.0.0.1:37018/" \
  .venv311/bin/python research/snorkel_label_merge.py
"""

from __future__ import annotations

import logging
import os
import time
from pathlib import Path

import numpy as np
from dotenv import load_dotenv
from pymongo import MongoClient, UpdateOne

ROOT = Path(__file__).resolve().parents[1]
load_dotenv(ROOT / ".env", override=False)

MONGO_URI   = os.getenv("LOCAL_MONGO_URI", "mongodb://root:root@127.0.0.1:37018/")
DB_NAME     = "quant_data"
COLLECTION  = "news_articles_company_matched_v2"
BATCH_SIZE  = int(os.getenv("MERGE_BATCH_SIZE", "5000"))

# 离散化阈值：|sentiment| <= NEUTRAL_THRESH 归为 neutral
NEUTRAL_THRESH = float(os.getenv("MERGE_NEUTRAL_THRESH", "0.05"))

# Snorkel class indices
NEG, NEU, POS = 0, 1, 2

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
log = logging.getLogger(__name__)


def discretize(score: float) -> int:
    if score > NEUTRAL_THRESH:
        return POS
    if score < -NEUTRAL_THRESH:
        return NEG
    return NEU


def main() -> None:
    client = MongoClient(MONGO_URI, serverSelectionTimeoutMS=5000)
    client.admin.command("ping")
    coll = client[DB_NAME][COLLECTION]

    # ── 1. Load all scored articles ───────────────────────────────────────────
    log.info("Loading articles from MongoDB …")
    t0 = time.time()
    docs = list(coll.find(
        {"llm_sentiment_a": {"$exists": True}, "llm_sentiment_b": {"$exists": True}},
        {"_id": 1, "llm_sentiment_a": 1, "llm_sentiment_b": 1},
    ))
    log.info("Loaded %d docs in %.1fs", len(docs), time.time() - t0)

    ids  = [d["_id"]           for d in docs]
    sa   = np.array([d["llm_sentiment_a"] for d in docs], dtype=np.float32)
    sb   = np.array([d["llm_sentiment_b"] for d in docs], dtype=np.float32)

    # ── 2. Build Snorkel label matrix  (n_docs × n_labelers) ─────────────────
    L = np.column_stack([
        [discretize(s) for s in sa],
        [discretize(s) for s in sb],
    ]).astype(np.int32)

    log.info("Label matrix shape: %s", L.shape)
    for cls, name in [(NEG, "negative"), (NEU, "neutral"), (POS, "positive")]:
        log.info("  A → %s: %d  |  B → %s: %d",
                 name, (L[:, 0] == cls).sum(),
                 name, (L[:, 1] == cls).sum())

    # ── 3. Fit Dawid-Skene (crowd-kit) — works with 2 annotators ─────────────
    import pandas as pd

    log.info("Fitting Dawid-Skene Label Model …")
    t1 = time.time()

    try:
        from crowdkit.aggregation import DawidSkene

        # crowd-kit expects a DataFrame with columns: task, worker, label
        n = len(ids)
        tasks   = list(range(n)) * 2
        workers = ["model_a"] * n + ["model_b"] * n
        labels  = list(L[:, 0]) + list(L[:, 1])
        df = pd.DataFrame({"task": tasks, "worker": workers, "label": labels})

        ds = DawidSkene(n_iter=100)
        ds.fit(df)

        # probas_: index=task, columns=[0,1,2] → P(neg), P(neu), P(pos)
        probs_df = ds.probas_
        probs = probs_df[[NEG, NEU, POS]].values.astype(np.float32)  # (n, 3)

    except ImportError:
        # Fallback: simple majority vote when crowd-kit is not installed
        log.warning("crowd-kit not installed — falling back to simple average aggregation")
        n = len(ids)
        probs = np.zeros((n, 3), dtype=np.float32)
        for i in range(n):
            for cls in [NEG, NEU, POS]:
                col = [NEG, NEU, POS].index(cls)
                votes = np.sum(L[i] == cls)
                probs[i, col] = votes / 2.0  # 2 annotators

    log.info("Dawid-Skene fitted in %.1fs", time.time() - t1)
    if hasattr(ds, 'errors_'):
        log.info("Worker error matrix:\n%s", ds.errors_.round(3))

    # ── 4. Infer probabilistic labels ─────────────────────────────────────────
    log.info("Inferring final sentiment …")
    final_sentiment = (probs[:, POS] - probs[:, NEG]).astype(np.float32)
    disagreement    = np.abs(sa - sb).astype(np.float32)

    log.info("final_sentiment: mean=%.3f  std=%.3f  min=%.3f  max=%.3f",
             final_sentiment.mean(), final_sentiment.std(),
             final_sentiment.min(), final_sentiment.max())
    log.info("disagreement:    mean=%.3f  std=%.3f  max=%.3f",
             disagreement.mean(), disagreement.std(), disagreement.max())

    # Agreement breakdown
    agree_mask = disagreement < 0.3
    log.info("Agreement (<0.3): %d (%.1f%%) | Disagreement: %d (%.1f%%)",
             agree_mask.sum(), agree_mask.mean()*100,
             (~agree_mask).sum(), (~agree_mask).mean()*100)

    # ── 5. Write back to MongoDB in batches ───────────────────────────────────
    log.info("Writing results to MongoDB …")
    t2 = time.time()
    ops: list[UpdateOne] = []
    written = 0

    for i, _id in enumerate(ids):
        ops.append(UpdateOne(
            {"_id": _id},
            {"$set": {
                "llm_sentiment_final":    round(float(final_sentiment[i]), 4),
                "llm_disagreement":       round(float(disagreement[i]), 4),
                "llm_label_model_probs":  [round(float(p), 4) for p in probs[i]],
            }},
        ))
        if len(ops) >= BATCH_SIZE:
            coll.bulk_write(ops, ordered=False)
            written += len(ops)
            ops.clear()
            log.info("  written %d / %d", written, len(ids))

    if ops:
        coll.bulk_write(ops, ordered=False)
        written += len(ops)

    log.info("Done. %d docs updated in %.1fs", written, time.time() - t2)

    # ── 6. Quick sanity check ─────────────────────────────────────────────────
    sample = list(coll.find(
        {"llm_disagreement": {"$exists": True}},
        {"llm_sentiment_a": 1, "llm_sentiment_b": 1,
         "llm_sentiment_final": 1, "llm_disagreement": 1},
    ).limit(5))
    log.info("Sample docs:")
    for d in sample:
        log.info("  a=%.3f b=%.3f → final=%.3f  diff=%.3f",
                 d["llm_sentiment_a"], d["llm_sentiment_b"],
                 d["llm_sentiment_final"], d["llm_disagreement"])


if __name__ == "__main__":
    main()
