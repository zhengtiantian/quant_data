#!/usr/bin/env python3
"""Backfill one company's GDELT news history straight from the GKG Parquet index.

    tools/backfill_symbol_news.py --symbol STX --since 2018-01-01
    tools/backfill_symbol_news.py --symbol SKHY --since 2026-07-10 --stage fetch

Why not the batch queue (2026-10-25): historical_collector.py walks all ~300,000 GKG files in batches
and matches every company in each; it suits the nightly increment, where all companies move forward
together. A company added later needs its whole history for one name, and the Parquet copy of
gkg_index (tools/gkg_parquet.py) answers that with one keyword scan.

Stages, each resumable (state under --work-dir/<SYMBOL>/):
  scan    the company's search keywords (its rule file, as the collector uses them) over the url and
          GKG text of every month since --since -> candidates.jsonl
  filter  the collector's rule engine on each candidate's GKG text (kill lists, strong-evidence
          bypass, the SLM for single weak hits), one row per URL, URLs already stored skipped
          -> kept.jsonl
  fetch   the collector's own fetch_article(): download, extract, rules again on the full text,
          then upsert into news_articles exactly as the collector writes them, so company match,
          enrichment and the Qdrant index pick them up as usual.
Progress: one "PROGRESS <stage> <done> <total> <extra>" line per step on stdout.

Run it under tools/lm_session.py (the filter and fetch stages may call the SLM).
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "collectors" / "news" / "gdelt"))

PARQUET_DIR = os.getenv("GKG_PARQUET_DIR", "/Volumes/Data4T/gkg_parquet")
WORK_DIR = os.getenv("BACKFILL_WORK_DIR", "/Volumes/Data4T/news_backfill")


def progress(stage: str, done: int, total: int, extra: str = "") -> None:
    print(f"PROGRESS {stage} {done} {total} {extra}".rstrip(), flush=True)


def scan(symbol: str, since: str, keywords: list[str], out: Path) -> None:
    import pyarrow.compute as pc
    import pyarrow.dataset as ds
    months = [m for m in sorted(glob.glob(f"{PARQUET_DIR}/month=*")) if m.split("=")[1] >= since[:7]]
    done_file = out.with_suffix(".months")
    done = set(done_file.read_text().split()) if done_file.exists() else set()
    rx = "(?i)\\b(" + "|".join(re.escape(k) for k in keywords if len(k) > 1) + ")\\b"
    since_ts = since.replace("-", "") + "000000"
    total = 0
    for i, m in enumerate(months, 1):
        month = m.split("=")[1]
        if month in done:
            continue
        t = ds.dataset(m, format="parquet").to_table(columns=["ts", "url", "raw"])
        # The collector matches the GKG text and strips URLs first ("platform domain names" would
        # trigger keywords); do the same.
        t = t.filter(pc.match_substring_regex(t["raw"], rx))
        rows = [r for r in t.to_pylist() if str(r["ts"]) >= since_ts]
        with out.open("a") as f:
            for r in rows:
                f.write(json.dumps(r) + "\n")
        total += len(rows)
        with done_file.open("a") as f:
            f.write(month + "\n")
        progress("scan", i, len(months), f"{month} candidates+{len(rows)}")


def existing_urls(urls: list[str]) -> set[str]:
    import pymongo
    col = pymongo.MongoClient(os.getenv("MONGO_URI"))["quant_data"]["news_articles"]
    have = set()
    for i in range(0, len(urls), 1000):
        have.update(d["url"] for d in col.find({"url": {"$in": urls[i:i + 1000]}}, {"url": 1, "_id": 0}))
    return have


def filter_stage(symbol: str, cand: Path, out: Path, workers: int) -> None:
    from special_rules import RuleManager
    rm = RuleManager()
    rows: dict[str, dict] = {}
    for line in cand.open():
        r = json.loads(line)
        if r["url"] not in rows or str(r["ts"]) < str(rows[r["url"]]["ts"]):
            rows[r["url"]] = r                     # one row per URL, the earliest sighting
    seen = set()
    if out.exists():
        seen = {json.loads(l)["url"] for l in out.open()}
    decided_file = out.with_suffix(".decided")
    if decided_file.exists():
        seen |= set(decided_file.read_text().split("\n"))
    stored = existing_urls([u for u in rows if u not in seen])
    todo = [r for u, r in rows.items() if u not in seen and u not in stored]
    progress("filter", 0, len(todo), f"unique={len(rows)} already_stored={len(stored)}")

    def judge(r):
        ok = rm.should_include(symbol, {"title": "", "content": r["raw"], "url": r["url"], "date": str(r["ts"])[:8]})
        return r, ok
    kept = 0
    with ThreadPoolExecutor(workers) as ex, out.open("a") as fo, decided_file.open("a") as fd:
        futures = [ex.submit(judge, r) for r in todo]
        for i, f in enumerate(as_completed(futures), 1):
            r, ok = f.result()
            if ok:
                fo.write(json.dumps(r) + "\n"); kept += 1
            fd.write(r["url"] + "\n")
            if i % 25 == 0 or i == len(todo):
                fo.flush(); fd.flush()
                progress("filter", i, len(todo), f"kept={kept}")


# Big News Network syndicates one story to dozens of city-named sites with the same path
# (/news/279179603/sk-hynix-jumps-...). Most of those mirrors no longer load (SKHY, 2026-10-25: 10 of
# 14 sampled failures), but usually one does: they are one story, tried mirror by mirror.
MIRROR_KEY = re.compile(r"^https?://[^/]+/news/(\d{6,})/")
MAX_MIRRORS = 3


def story_key(url: str) -> str:
    m = MIRROR_KEY.match(url)
    return f"bnn:{m.group(1)}" if m else url


def fetch_stage(symbol: str, name: str, kept: Path, workers: int) -> None:
    import historical_collector as hc
    import pymongo
    from pymongo import UpdateOne
    col = pymongo.MongoClient(os.getenv("MONGO_URI"))["quant_data"]["news_articles"]
    done_file = kept.with_suffix(".fetched")
    done = set(done_file.read_text().split("\n")) if done_file.exists() else set()
    rows = [json.loads(l) for l in kept.open()]
    stored = existing_urls([r["url"] for r in rows if r["url"] not in done])
    groups: dict[str, list[dict]] = {}
    for r in rows:
        groups.setdefault(story_key(r["url"]), []).append(r)
    # A story is done when any of its URLs was stored or tried.
    todo = [g for g in groups.values() if not any(r["url"] in done or r["url"] in stored for r in g)]
    company = {"symbol": symbol, "name": name}
    progress("fetch", 0, len(todo), f"stories={len(groups)} urls={len(rows)} already_stored={len(stored)}")

    def one(group):
        art = None
        for r in group[:MAX_MIRRORS]:
            ts = str(r["ts"])
            row = {"URL": r["url"], "Date": ts[:8], "Timestamp": ts, "Title": "", "PublishedAt": None}
            try:
                art = hc.fetch_article(row, company)
            except Exception:  # noqa: BLE001 - a page that will not load: try the next mirror
                art = None
            if art:
                break
        return group[0]["url"], art
    inserted, pending, stats = 0, [], {"full": 0, "title_only": 0, "rejected_or_dead": 0}
    with ThreadPoolExecutor(workers) as ex, done_file.open("a") as fd:
        futures = [ex.submit(one, g) for g in todo]
        for i, f in enumerate(as_completed(futures), 1):
            url, art = f.result()
            fd.write(url + "\n")
            if art:
                pending.append(UpdateOne({"url": art["url"]}, {"$setOnInsert": art}, upsert=True))
                stats[art.get("data_quality", "full") if art.get("data_quality") in stats else "full"] += 1
            else:
                stats["rejected_or_dead"] += 1
            if len(pending) >= 100 or (i == len(todo) and pending):
                res = col.bulk_write(pending, ordered=False)
                inserted += int(res.upserted_count or 0)
                pending = []
            if i % 25 == 0 or i == len(todo):
                fd.flush()
                progress("fetch", i, len(todo), f"inserted={inserted} full={stats['full']} title_only={stats['title_only']} "
                                                 f"rejected_or_dead={stats['rejected_or_dead']}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--symbol", required=True)
    ap.add_argument("--since", required=True, help="YYYY-MM-DD")
    ap.add_argument("--stage", choices=("scan", "filter", "fetch", "all"), default="all")
    ap.add_argument("--filter-workers", type=int, default=int(os.getenv("LM_WORKERS", "8")))
    ap.add_argument("--fetch-workers", type=int, default=12)
    a = ap.parse_args()
    sym = a.symbol.upper()
    from special_rules import RuleManager
    rm = RuleManager()
    if sym not in rm.company_configs:
        raise SystemExit(f"no rule file for {sym} in company_rules/")
    cfg = rm.company_configs[sym]
    keywords = rm.get_keywords(sym, datetime.strptime(a.since, "%Y-%m-%d"))
    work = Path(WORK_DIR) / sym
    work.mkdir(parents=True, exist_ok=True)
    cand, kept = work / "candidates.jsonl", work / "kept.jsonl"
    print(f"{sym}: since {a.since}, keywords {sorted(keywords)}, work dir {work}", flush=True)
    t0 = time.time()
    if a.stage in ("scan", "all"):
        scan(sym, a.since, keywords, cand)
    if a.stage in ("filter", "all"):
        filter_stage(sym, cand, kept, a.filter_workers)
    if a.stage in ("fetch", "all"):
        fetch_stage(sym, cfg.get("name") or sym, kept, a.fetch_workers)
    print(f"DONE {sym} in {time.time() - t0:.0f}s", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
