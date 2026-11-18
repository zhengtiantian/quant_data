#!/usr/bin/env python3
"""One-off (2026-11-18): re-judge what the v2 direct-mention gate rejected because a keyword list left
the bare company name out ("Intel", "Oracle", "KLA"; see _NAME_PATTERNS in slm_company_match_v2.py).

Picks news_articles rows inserted since the gate went live (cc5108c, 2026-11-04 16:05 UTC) that are not
in the target collection, fail the keyword-only test and name the company as written, and runs them
through the production student path (process_batch_student, prefilter included). Matches are saved with
save_batch: upserts of rows the target does not have, so nothing there is replaced. The label passes
pick the new rows up on their next run (they select rows without labels, whatever the _id).

  --count   only count the candidates per symbol, write nothing
Env: REMATCH_INCLUDE_UNIVERSE=1 also takes origin=universe_backfill rows.
Run under the GPU lock with the Python that has mlx_lm:
  tools/lm_session.py local --name rematch-gate -- $STUDENT_PYTHON research/labeling/rematch_mention_gate.py
"""
from __future__ import annotations

import os
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

from bson import ObjectId

sys.path.insert(0, str(Path(__file__).resolve().parent))
import slm_company_match_v2 as m2  # noqa: E402

GATE_LIVE = datetime(2026, 11, 4, 16, 5, tzinfo=timezone.utc)
INCLUDE_UNIVERSE = os.getenv("REMATCH_INCLUDE_UNIVERSE", "0") == "1"


def _body(doc: dict) -> str:
    content = (doc.get("content") or "")[:m2.CONTENT_CHARS]
    thin = doc.get("data_quality") == "title_only" or m2._is_thin_body(content)
    return "" if thin else content


def candidates(src, tgt) -> list[ObjectId]:
    query = {"_id": {"$gte": ObjectId.from_datetime(GATE_LIVE)}, "symbol": {"$in": sorted(m2._NAME_PATTERNS)},
             "title": {"$nin": [None, ""]}}
    if not INCLUDE_UNIVERSE:
        query["origin"] = {"$ne": m2.UNIVERSE_ORIGIN}
    proj = {"symbol": 1, "title": 1, "data_quality": 1, "content": {"$substrCP": [{"$ifNull": ["$content", ""]}, 0, m2.CONTENT_CHARS]}}
    picked = []
    for d in src.aggregate([{"$match": query}, {"$project": proj}], allowDiskUse=True, batchSize=5000):
        sym, title, body = d["symbol"], d.get("title") or "", _body(d)
        kw = m2._SCAN_PATTERNS.get(sym)
        if kw and (kw.search(title) or kw.search(body)):
            continue                                   # passed the gate already: judged then
        name = m2._NAME_PATTERNS[sym]
        if name.search(title) or name.search(body):
            picked.append(d["_id"])
    have = set()
    for i in range(0, len(picked), 5000):
        have.update(x["_id"] for x in tgt.find({"_id": {"$in": picked[i:i + 5000]}}, {"_id": 1}))
    return [i for i in picked if i not in have]


def main() -> None:
    client = m2.get_client()
    src, tgt = client[m2.DB_NAME][m2.SOURCE_COLLECTION], client[m2.DB_NAME][m2.TARGET_COLLECTION]
    ids = candidates(src, tgt)
    by_sym = Counter(d["symbol"] for d in src.find({"_id": {"$in": ids}}, {"symbol": 1})) if ids else Counter()
    print(f"candidates since {GATE_LIVE:%Y-%m-%d %H:%M} UTC (universe {'in' if INCLUDE_UNIVERSE else 'out'}): {len(ids):,}")
    print("  " + ", ".join(f"{s} {n}" for s, n in by_sym.most_common(30)))
    if "--count" in sys.argv or not ids:
        return
    matched, rejected = Counter(), Counter()
    for i in range(0, len(ids), m2.BATCH_SIZE):
        docs = list(src.find({"_id": {"$in": ids[i:i + m2.BATCH_SIZE]}}))
        rows = m2.process_batch_student(docs)
        m2.save_batch(tgt, rows)
        for r in rows:
            (matched if r["status"] in ("match", "reassigned") else rejected)[r["symbol"]] += 1
        print(f"  {i + len(docs):,}/{len(ids):,}: matched {sum(matched.values()):,}, rejected {sum(rejected.values()):,}", flush=True)
    print("matched by symbol: " + ", ".join(f"{s} {n}" for s, n in matched.most_common(30)))


if __name__ == "__main__":
    main()
