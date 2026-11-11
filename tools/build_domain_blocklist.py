#!/usr/bin/env python3
"""Sites whose articles the relevance filter almost never keeps -> prefilter.BLOCKLIST_FILE (outside the
repository; research/labeling/prefilter.py reads it and rejects their rows before a model is asked).
Runs every Saturday in keyword_refresh_weekly.

Counts every judged row per site: core rows are judged when they lie behind the v2 job's cursor and
kept when they are in the matched collection; universe rows carry the student's verdict on the
matched row. A site is blocked with at least MIN_ROWS judged rows and a keep rate under MAX_KEEP.
On 1.86M verdicts (2026-11-10) that is 31 sites, 0.6% of the rows, 88 kept rows given up; at 5% the
list reaches working business outlets (chicagobusiness.com, kyivindependent.com), so the bar stays at 2%.

    build_domain_blocklist.py            print the list, write nothing
    build_domain_blocklist.py --write    write the list
Env: LOCAL_MONGO_URI, BLOCK_MIN_ROWS (100), BLOCK_MAX_KEEP (0.02).
"""
from __future__ import annotations

import collections
import json
import os
import sys
from datetime import date
from pathlib import Path

from pymongo import MongoClient

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "research" / "labeling"))
import prefilter  # noqa: E402

MONGO = os.getenv("LOCAL_MONGO_URI", "mongodb://root:root@127.0.0.1:37018/")
MIN_ROWS = int(os.getenv("BLOCK_MIN_ROWS", "100"))
MAX_KEEP = float(os.getenv("BLOCK_MAX_KEEP", "0.02"))
MAIN_JOB = "news_articles_company_matched_v2:v2"
UNIVERSE = "universe_backfill"


def domain_stats(db) -> dict[str, list[int]]:
    """site -> [judged rows, kept rows]"""
    stats: dict[str, list[int]] = collections.defaultdict(lambda: [0, 0])
    v2 = db.news_articles_company_matched_v2
    kept = {d["_id"] for d in v2.find({"origin": {"$ne": UNIVERSE}}, {"_id": 1})}
    query: dict = {"origin": {"$ne": UNIVERSE}, "symbol": {"$ne": None}}
    cursor = (db.company_match_jobs.find_one({"_id": MAIN_JOB}) or {}).get("lastId")
    if cursor:
        query["_id"] = {"$lte": cursor}
    for d in db.news_articles.find(query, {"url": 1}):
        s = stats[prefilter.domain_of(d.get("url") or "")]
        s[0] += 1
        s[1] += d["_id"] in kept
    for d in v2.find({"origin": UNIVERSE, "company_match_v2.engine": "student"},
                     {"url": 1, "company_match_v2.matched": 1}):
        s = stats[prefilter.domain_of(d.get("url") or "")]
        s[0] += 1
        s[1] += bool(d["company_match_v2"].get("matched"))
    stats.pop("", None)
    return stats


def main() -> None:
    stats = domain_stats(MongoClient(MONGO)["quant_data"])
    total = sum(v[0] for v in stats.values())
    blocked = sorted((d for d, (n, k) in stats.items() if n >= MIN_ROWS and k / n < MAX_KEEP),
                     key=lambda d: -stats[d][0])
    # a site already blocked stays at keep 0 from then on, so the earlier list is carried over
    earlier = sorted(prefilter.blocked_domains())
    domains = blocked + [d for d in earlier if d not in blocked]
    rows = sum(stats[d][0] for d in blocked)
    print(f"{total:,} judged rows on {len(stats):,} sites; blocked {len(blocked)} sites with {rows:,} rows "
          f"({rows / total:.1%}), kept rows given up {sum(stats[d][1] for d in blocked):,}; "
          f"{len(domains) - len(blocked)} carried over")
    for d in blocked:
        print(f"  {d:40} {stats[d][0]:7,} rows, kept {stats[d][1]:5,}")
    if "--write" in sys.argv:
        prefilter.BLOCKLIST_FILE.parent.mkdir(parents=True, exist_ok=True)
        prefilter.BLOCKLIST_FILE.write_text(json.dumps({
            "generated": date.today().isoformat(), "min_rows": MIN_ROWS, "max_keep": MAX_KEEP,
            "domains": domains, "stats": {d: stats[d] for d in blocked}}, indent=1) + "\n")
        print(f"wrote {prefilter.BLOCKLIST_FILE}")


if __name__ == "__main__":
    main()
