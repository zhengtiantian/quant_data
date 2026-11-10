#!/usr/bin/env python3
"""Article labels (pass A / pass B) from the distilled students instead of the LM Studio teachers.

Same job and same fields as llm_enrich_articles.py (llm_sentiment_x, llm_event_type_x,
llm_signal_strength_x, llm_enriched_x_at), so merge_ab_labels.py and the features read them as
before; llm_model_x records which student wrote the row. Pass A = Gemma-4-E2B (teacher E4B),
pass B = Qwen3-1.7B (teacher qwen3.5-9b), each ~20 articles/s in batches (mlx_students.py).

Resumable like the teacher pass: rows that already have llm_sentiment_x are skipped. Rows the
relevance pass rejected (company_match_v2.matched = false) are not labelled. An answer that does
not parse is left without labels and tried again on the next run.

Run under `tools/lm_session.py local` with the Python that has mlx_lm:
  ENRICH_PASS=A lm_session.py local --name label-a -- $STUDENT_PYTHON research/labeling/student_enrich_articles.py
Env: ENRICH_PASS (A|B), ENRICH_COLLECTION, ENRICH_LIMIT, ENRICH_INCLUDE_UNIVERSE=1 (also the 490
universe companies' rows), ENRICH_FETCH_BATCH, ENRICH_DRY_RUN=1 (label and print, write nothing).
"""
from __future__ import annotations

import logging
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

from pymongo import MongoClient, UpdateOne

sys.path.insert(0, str(Path(__file__).resolve().parent))
import llm_enrich_articles as le  # noqa: E402 - the teacher script: label vocabulary and normalisation
import mlx_students as ms  # noqa: E402

PASS = os.getenv("ENRICH_PASS", "A").upper()
LIMIT = int(os.getenv("ENRICH_LIMIT", "0"))
FETCH_BATCH = int(os.getenv("ENRICH_FETCH_BATCH", "512"))
DRY_RUN = os.getenv("ENRICH_DRY_RUN", "0") == "1"
INCLUDE_UNIVERSE = os.getenv("ENRICH_INCLUDE_UNIVERSE", "0") == "1"
LABEL_INPUT = "title+content500"

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
log = logging.getLogger(__name__)


def to_label(text: str, title: str) -> dict | None:
    parsed = ms.parse_label(text)
    if parsed is None:
        return None
    event, sentiment, strength = parsed
    event = le._title_override(title, le._normalize_event_type(event))
    return {"sentiment": round(max(-1.0, min(1.0, sentiment)), 4), "event_type": event,
            "signal_strength": strength if strength in le.VALID_STRENGTHS else "low"}


def main() -> None:
    if PASS not in ("A", "B"):
        sys.exit(f"Unknown ENRICH_PASS={PASS!r}. Use A or B.")
    suffix = PASS.lower()
    client = MongoClient(le.MONGO_URI, serverSelectionTimeoutMS=5000)
    client.admin.command("ping")
    coll = client[le.DB_NAME][le.COLLECTION]

    query: dict = {f"llm_sentiment_{suffix}": {"$exists": False}, "company_match_v2.matched": {"$ne": False}}
    if not INCLUDE_UNIVERSE:
        query["origin"] = {"$ne": "universe_backfill"}
    else:
        # a universe row is labelled only once the relevance student has judged it (rows promoted on
        # rules alone carry engine backfill_rules until then)
        query["$or"] = [{"origin": {"$ne": "universe_backfill"}}, {"company_match_v2.engine": "student"}]
    total = coll.count_documents(query)
    if LIMIT:
        total = min(total, LIMIT)
    log.info("PASS=%s | articles to label: %d", PASS, total)
    if total == 0:
        return

    student = ms.Student(f"label_{suffix}")
    log.info("student %s ready (batch %d)", student.id, student.batch)
    done = errors = 0
    last_id = None
    start = time.time()
    while done + errors < total:
        page = dict(query)
        if last_id is not None:
            page["_id"] = {"$gt": last_id}
        docs = list(coll.find(page, {"_id": 1, "title": 1, "content": 1, "symbol": 1, "name": 1})
                    .sort("_id", 1).limit(min(FETCH_BATCH, total - done - errors)))
        if not docs:
            break
        last_id = docs[-1]["_id"]
        texts = student.generate([
            ms.label_input(d.get("symbol") or "", d.get("name") or d.get("symbol") or "", d.get("title") or "",
                           "" if le.TITLE_ONLY else d.get("content") or "") for d in docs])
        now, ops = datetime.now(timezone.utc), []
        for d, text in zip(docs, texts):
            label = to_label(text, d.get("title") or "")
            if label is None:
                errors += 1
                continue
            done += 1
            if DRY_RUN:
                if done <= 20:
                    print(f"  {d.get('symbol'):6} {label['event_type']:11} {label['sentiment']:+.2f} {label['signal_strength']:6} {(d.get('title') or '')[:80]}")
                continue
            ops.append(UpdateOne({"_id": d["_id"]}, {"$set": {
                f"llm_sentiment_{suffix}": label["sentiment"],
                f"llm_event_type_{suffix}": label["event_type"],
                f"llm_signal_strength_{suffix}": label["signal_strength"],
                f"llm_enriched_{suffix}_at": now,
                f"llm_model_{suffix}": student.id,
                "llm_label_input": LABEL_INPUT,
            }}))
        if ops:
            coll.bulk_write(ops, ordered=False)
        n, elapsed = done + errors, time.time() - start
        rate = n / elapsed if elapsed else 0
        log.info("[%d/%d] done=%d errors=%d | %.1f art/s | ETA %.0f min", n, total, done, errors, rate,
                 (total - n) / rate / 60 if rate else 0)
    log.info("Finished%s. %d labelled, %d unparsed, %.1f art/s", " (dry run, nothing written)" if DRY_RUN else "",
             done, errors, (done + errors) / max(time.time() - start, 1e-9))


if __name__ == "__main__":
    main()
