#!/usr/bin/env python3
"""Shadow run of the GD1 students (N.9, 2026-11-13): relevance and labels side by side with production.

Writes one new field on news_articles and nothing else:
  student_gd1 = {relevant, student_relevant, rel_rule, sentiment, event_type, signal_strength, rel_model, label_model, at}
relevant is the student's answer unless the key-person rule (key_people.py) overrides it to True;
student_relevant keeps the student's own answer and rel_rule says what matched.
Every article in the window gets both answers (labels too where the student says not relevant), so
the trial can be compared with production: kept in news_articles_company_matched_v2 or moved to
_rejected, and the A/B labels there. Nothing reads student_gd1 yet; features and retrieval are unchanged.
The raw student verdict only: production's prefilter and direct-mention gate are not applied here.

Resumable: rows that already have student_gd1 are skipped. An unparsed answer stores None for that part.
Run under the GPU lock with the Python that has mlx_lm:
  lm_session.py local --name gd1-shadow -- $STUDENT_PYTHON research/labeling/student_gd1_shadow.py
Env: GD1_SINCE (publishedAt ISO prefix, default 7 days ago), GD1_LIMIT, GD1_INCLUDE_UNIVERSE=1,
GD1_FETCH_BATCH, GD1_DRY_RUN=1 (answer and print, write nothing).
"""
from __future__ import annotations

import logging
import os
import sys
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

from pymongo import MongoClient, UpdateOne

sys.path.insert(0, str(Path(__file__).resolve().parent))
import llm_enrich_articles as le  # noqa: E402 - label vocabulary and normalisation
import key_people  # noqa: E402
import mlx_students as ms  # noqa: E402
from student_enrich_articles import to_label  # noqa: E402

SINCE = os.getenv("GD1_SINCE") or (datetime.now(timezone.utc) - timedelta(days=7)).strftime("%Y-%m-%dT%H:%M")
LIMIT = int(os.getenv("GD1_LIMIT", "0"))
FETCH_BATCH = int(os.getenv("GD1_FETCH_BATCH", "512"))
DRY_RUN = os.getenv("GD1_DRY_RUN", "0") == "1"
INCLUDE_UNIVERSE = os.getenv("GD1_INCLUDE_UNIVERSE", "0") == "1"
# GD1_REAPPLY_RULES=1: no model; re-apply the key-person rule to rows that already have student_gd1
REAPPLY = os.getenv("GD1_REAPPLY_RULES", "0") == "1"
FIELD = "student_gd1"

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
log = logging.getLogger(__name__)


def rule_verdict(student_says: bool | None, symbol: str, title: str) -> tuple[bool | None, str | None]:
    """The student's answer, overridden to relevant when the title is about one of the company's leaders."""
    why = key_people.hit(symbol, title)
    return (True, f"key_person:{why}") if why and student_says is not True else (student_says, None)


def reapply(coll) -> None:
    q = {f"{FIELD}.student_relevant": {"$ne": True}, FIELD: {"$exists": True}}
    ops = []
    for d in coll.find(q, {"symbol": 1, "title": 1, FIELD: 1}):
        g = d[FIELD]
        said = g.get("student_relevant", g.get("relevant"))
        relevant, rule = rule_verdict(said, d.get("symbol"), d.get("title"))
        if (relevant, rule) != (g.get("relevant"), g.get("rel_rule")) or "student_relevant" not in g:
            ops.append(UpdateOne({"_id": d["_id"]}, {"$set": {f"{FIELD}.relevant": relevant, f"{FIELD}.rel_rule": rule,
                                                             f"{FIELD}.student_relevant": said}}))
    log.info("reapply: %d rows to update%s", len(ops), " (dry run)" if DRY_RUN else "")
    if ops and not DRY_RUN:
        coll.bulk_write(ops, ordered=False)


def main() -> None:
    coll = MongoClient(le.MONGO_URI, serverSelectionTimeoutMS=5000)[le.DB_NAME]["news_articles"]
    if REAPPLY:
        return reapply(coll)
    # upper bound: some old articles carry a wrong publishedAt years in the future
    until = (datetime.now(timezone.utc) + timedelta(days=1)).strftime("%Y-%m-%dT%H:%M")
    # rows without a company (symbol None or "") are not judged: the student would see an empty company
    query: dict = {FIELD: {"$exists": False}, "publishedAt": {"$gte": SINCE, "$lte": until},
                   "symbol": {"$nin": [None, ""]}}
    if not INCLUDE_UNIVERSE:
        query["origin"] = {"$ne": "universe_backfill"}
    total = coll.count_documents(query)
    total = min(total, LIMIT) if LIMIT else total
    log.info("since %s | articles: %d%s", SINCE, total, " (dry run)" if DRY_RUN else "")
    if not total:
        return
    rel, lab = ms.Student("gd1_relevance"), ms.Student("gd1_label")
    log.info("students ready: %s, %s", rel.id, lab.id)
    n = unparsed = kept = 0
    last_id, start = None, time.time()
    while n < total:
        page = dict(query)
        if last_id is not None:
            page["_id"] = {"$gt": last_id}
        docs = list(coll.find(page, {"title": 1, "content": 1, "symbol": 1, "name": 1})
                    .sort("_id", 1).limit(min(FETCH_BATCH, total - n)))
        if not docs:
            break
        last_id = docs[-1]["_id"]
        args = [(d.get("symbol") or "", d.get("name") or d.get("symbol") or "", d.get("title") or "", d.get("content") or "")
                for d in docs]
        rel_txt = rel.generate([ms.relevance_input(*a) for a in args])
        lab_txt = lab.generate([ms.label_input(*a) for a in args])
        now, ops = datetime.now(timezone.utc), []
        for d, rt, lt in zip(docs, rel_txt, lab_txt):
            said, label = ms.parse_relevance(rt), to_label(lt, d.get("title") or "")
            relevant, rule = rule_verdict(said, d.get("symbol"), d.get("title"))
            unparsed += said is None or label is None
            kept += relevant is True
            value = {"relevant": relevant, "student_relevant": said, "rel_rule": rule,
                     "rel_model": rel.id, "label_model": lab.id, "at": now,
                     "sentiment": label and label["sentiment"], "event_type": label and label["event_type"],
                     "signal_strength": label and label["signal_strength"]}
            if DRY_RUN:
                if len(ops) < 15:
                    print(f"  {d.get('symbol') or '':6} {str(relevant):5} {label} {(d.get('title') or '')[:70]}")
                ops.append(None)
                continue
            ops.append(UpdateOne({"_id": d["_id"]}, {"$set": {FIELD: value}}))
        if not DRY_RUN:
            coll.bulk_write(ops, ordered=False)
        n += len(docs)
        rate = n / (time.time() - start)
        log.info("[%d/%d] relevant=%d unparsed=%d | %.1f art/s | ETA %.0f min", n, total, kept, unparsed, rate,
                 (total - n) / rate / 60)
    log.info("finished: %d articles, %d relevant (%.1f%%), %d unparsed", n, kept, 100 * kept / max(n, 1), unparsed)


if __name__ == "__main__":
    main()
