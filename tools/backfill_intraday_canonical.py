#!/usr/bin/env python3
"""Repair the intraday articles that were stored without `date` or `symbol`.

The finnhub / newsapi / yahoo collectors wrote a document shape that
`slm_company_match_v2.build_query()` cannot select — it requires a `symbol` — so
everything they collected from 2026-04-13 onward sat in `news_articles` and went no
further: not matched, not labeled, not featurised, not in any signal. The collectors are
fixed; this repairs what they already wrote.

Articles that match no company keep their `date` and stay without a symbol. That is the
correct outcome, not a failure: general market and macro news is exactly the material the
theme-propagation work will need, and it should not be forced under a ticker it is not
about — a wrong symbol pollutes that ticker's sentiment aggregate, while no symbol simply
leaves it out.

    python tools/backfill_intraday_canonical.py --dry-run
    python tools/backfill_intraday_canonical.py
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

from pymongo import MongoClient, UpdateOne

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "collectors" / "news"))

from _canonical import canonical_date, match_symbol  # noqa: E402

MONGO_URI = os.getenv("LOCAL_MONGO_URI") or os.getenv("MONGO_URI") \
    or "mongodb://root:root@127.0.0.1:37018/?authSource=admin"
BATCH = 500


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dry-run", action="store_true",
                    help="report what would change without writing")
    ap.add_argument("--limit", type=int, default=0, help="stop after N documents")
    args = ap.parse_args()

    col = MongoClient(MONGO_URI)["quant_data"]["news_articles"]
    query = {"date": {"$exists": False}}
    total = col.count_documents(query)
    print(f"articles missing `date`: {total:,}")
    if total == 0:
        return 0

    ops: list[UpdateOne] = []
    seen = dated = symbolled = 0
    by_symbol: dict[str, int] = {}

    cursor = col.find(query, {"title": 1, "description": 1, "content": 1, "publishedAt": 1})
    for doc in cursor:
        seen += 1
        update = {"date": canonical_date(doc.get("publishedAt"))}
        dated += 1
        matched = match_symbol(doc.get("title"), doc.get("description"), doc.get("content"))
        if matched:
            update["symbol"], update["name"] = matched
            symbolled += 1
            by_symbol[matched[0]] = by_symbol.get(matched[0], 0) + 1
        ops.append(UpdateOne({"_id": doc["_id"]}, {"$set": update}))

        if len(ops) >= BATCH:
            if not args.dry_run:
                col.bulk_write(ops, ordered=False)
            ops.clear()
            print(f"  {seen:,}/{total:,} …", flush=True)
        if args.limit and seen >= args.limit:
            break

    if ops and not args.dry_run:
        col.bulk_write(ops, ordered=False)

    print()
    print(f"{'would set' if args.dry_run else 'set'} date on   {dated:,}")
    print(f"{'would set' if args.dry_run else 'set'} symbol on {symbolled:,} "
          f"({symbolled / seen * 100:.1f}% of those examined)")
    print(f"left without a symbol   {seen - symbolled:,} "
          f"— general/macro news that names no covered company")
    if by_symbol:
        top = sorted(by_symbol.items(), key=lambda kv: -kv[1])[:10]
        print("\ntop matched symbols:")
        for sym, n in top:
            print(f"  {sym:<6} {n:,}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
