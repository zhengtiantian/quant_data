#!/usr/bin/env python3
"""GDELT GKG bulk import to MongoDB gkg_index.

Reads .gkg.csv files from Data24T and Data6T, extracts the columns used
by the news collector for keyword matching, and inserts into gkg_index.

MongoDB document structure:
  { "ts": "20200314120000", "url": "https://...", "raw": "Apple iPhone ..." }

Indexes created automatically:
  url  — unique (dedup)
  ts   — for date-range queries in the collector
  raw  — $text for keyword search

Usage:
  LOCAL_MONGO_URI="mongodb://root:root@127.0.0.1:37018/" \\
  python tools/gdelt_import_to_mongo.py

  # Limit to a specific drive (for testing)
  GDELT_IMPORT_DIRS="/Volumes/Data24T/docker-volumes/gdelt_cache/files" \\
  GDELT_IMPORT_WORKERS=8 \\
  python tools/gdelt_import_to_mongo.py
"""

from __future__ import annotations

import csv
import logging
import os
import re
import sys
import time
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed

import pymongo
from pymongo import MongoClient, UpdateOne

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)
log = logging.getLogger(__name__)

MONGO_URI   = os.getenv("LOCAL_MONGO_URI", "mongodb://root:root@127.0.0.1:37018/")
DB_NAME     = "quant_data"
GKG_COL     = "gkg_index"
PROG_COL    = "gkg_import_progress"

# Dirs to scan — space-separated list via env, or default both drives
_DEFAULT_DIRS = " ".join([
    "/Volumes/Data24T/docker-volumes/gdelt_cache/files",
    "/Volumes/Data6T/gdelt_cache/files",
])
IMPORT_DIRS  = os.getenv("GDELT_IMPORT_DIRS", _DEFAULT_DIRS).split()
WORKERS      = int(os.getenv("GDELT_IMPORT_WORKERS", "6"))
BATCH_SIZE   = int(os.getenv("GDELT_IMPORT_BATCH", "2000"))   # docs per bulk_write
PROGRESS_N   = int(os.getenv("GDELT_IMPORT_PROGRESS", "500")) # files between log lines

# Columns used by the news collector for keyword matching
ENTITY_COLS  = [7, 9, 11, 23, 24, 26, 28]
URL_COL      = 4


_XML_RE      = re.compile(r"(?is)<[^>]+>")
_URL_RE      = re.compile(r"https?://\S+|\b(?:www\.)?[a-z0-9.-]+\.[a-z]{2,}(?:/\S*)?\b", re.I)
_WS_RE       = re.compile(r"\s+")


def _strip(text: str) -> str:
    """Remove XML tags and bare URLs, collapse whitespace."""
    text = re.sub(r"(?is)<PAGE_LINKS>.*?</PAGE_LINKS>", " ", text)
    text = _XML_RE.sub(" ", text)
    text = _URL_RE.sub(" ", text)
    return _WS_RE.sub(" ", text).strip()


def _ts_from_name(name: str) -> str:
    """Extract 14-char timestamp from GKG filename."""
    base = os.path.basename(name)
    return base[:14] if len(base) >= 14 and base[:14].isdigit() else base[:8]


def _parse_gkg(path: str) -> list[dict]:
    """Parse one GKG CSV file; return list of {ts, url, raw} dicts."""
    ts = _ts_from_name(path)
    docs: list[dict] = []
    try:
        with open(path, encoding="ISO-8859-1", errors="replace", newline="") as fh:
            reader = csv.reader(fh, delimiter="\t", quoting=csv.QUOTE_NONE)
            for row in reader:
                if len(row) <= URL_COL:
                    continue

                # Detect XML format (col 26 contains <PAGE_LINKS>)
                col26 = row[26] if len(row) > 26 else ""
                if "<PAGE_LINKS>" in col26:
                    m_url = re.search(r"<PAGE_LINKS>(.*?)</PAGE_LINKS>", col26, re.S)
                    url = (m_url.group(1).split(";")[0] if m_url else "").strip()
                    raw = _strip(col26)
                else:
                    url = row[URL_COL].strip()
                    parts = []
                    for c in ENTITY_COLS:
                        if c < len(row) and row[c]:
                            parts.append(row[c])
                    raw = _strip(" ".join(parts))

                if not url.startswith("http"):
                    continue
                docs.append({"ts": ts, "url": url, "raw": raw})
    except Exception as exc:
        log.warning("Parse error %s: %s", path, exc)
    return docs


def _collect_csv_files(dirs: list[str]) -> list[str]:
    """Recursively collect all .gkg.csv paths from given directories."""
    paths: list[str] = []
    for d in dirs:
        if not os.path.isdir(d):
            log.warning("Directory not found: %s", d)
            continue
        for root, _, files in os.walk(d):
            for f in files:
                if f.endswith(".gkg.csv"):
                    paths.append(os.path.join(root, f))
    return paths


def _already_imported(paths: list[str], prog_col) -> set[str]:
    """Return set of basenames already recorded in gkg_import_progress."""
    existing = prog_col.find({"_id": {"$in": [os.path.basename(p) for p in paths]}}, {"_id": 1})
    return {doc["_id"] for doc in existing}


def _process_file(path: str) -> tuple[str, list[dict], str]:
    """Worker function: parse a CSV file, return (basename, docs, error)."""
    try:
        docs = _parse_gkg(path)
        return os.path.basename(path), docs, ""
    except Exception as exc:
        return os.path.basename(path), [], str(exc)


def _ensure_indexes(gkg_col) -> None:
    gkg_col.create_index("url", unique=True, background=True)
    gkg_col.create_index("ts", background=True)
    try:
        gkg_col.create_index([("raw", pymongo.TEXT)], name="gkg_raw_text", background=True)
        log.info("$text index on raw created (may take time to build)")
    except pymongo.errors.OperationFailure as e:
        if "already exists" in str(e):
            log.info("$text index already exists")
        else:
            raise


def main() -> None:
    client = MongoClient(MONGO_URI, serverSelectionTimeoutMS=5000)
    client.admin.command("ping")
    db = client[DB_NAME]
    gkg_col  = db[GKG_COL]
    prog_col = db[PROG_COL]

    log.info("Ensuring indexes on %s.%s ...", DB_NAME, GKG_COL)
    _ensure_indexes(gkg_col)

    log.info("Scanning directories: %s", IMPORT_DIRS)
    all_paths = _collect_csv_files(IMPORT_DIRS)
    log.info("Found %d .gkg.csv files total", len(all_paths))

    done_names = _already_imported(all_paths, prog_col)
    todo = [p for p in all_paths if os.path.basename(p) not in done_names]
    log.info("Already imported: %d | Remaining: %d", len(done_names), len(todo))

    if not todo:
        log.info("Nothing to import. Done.")
        return

    start     = time.time()
    n_done    = 0
    n_errors  = 0
    n_records = 0
    pending_ops: list[UpdateOne]      = []
    pending_prog: list[pymongo.InsertOne] = []

    def _flush(force: bool = False) -> None:
        nonlocal pending_ops, pending_prog
        if pending_ops and (force or len(pending_ops) >= BATCH_SIZE):
            try:
                gkg_col.bulk_write(pending_ops, ordered=False)
            except pymongo.errors.BulkWriteError:
                pass  # duplicate URLs are expected and ignored
            pending_ops = []
        if pending_prog and (force or len(pending_prog) >= 200):
            try:
                prog_col.bulk_write(pending_prog, ordered=False)
            except pymongo.errors.BulkWriteError:
                pass
            pending_prog = []

    with ProcessPoolExecutor(max_workers=WORKERS) as pool:
        futures = {pool.submit(_process_file, p): p for p in todo}
        for future in as_completed(futures):
            name, docs, err = future.result()
            if err:
                log.warning("ERROR %s: %s", name, err)
                n_errors += 1
            else:
                n_done    += 1
                n_records += len(docs)
                for d in docs:
                    pending_ops.append(UpdateOne(
                        {"url": d["url"]},
                        {"$setOnInsert": d},
                        upsert=True,
                    ))
                pending_prog.append(
                    pymongo.InsertOne({"_id": name, "ts": time.time()})
                )
                _flush()

            total = n_done + n_errors
            if total % PROGRESS_N == 0 or total == len(todo):
                elapsed = time.time() - start
                rate    = n_done / elapsed if elapsed else 0
                eta_h   = (len(todo) - total) / rate / 3600 if rate else 0
                log.info(
                    "[%d/%d] done=%d errors=%d records=%d | %.1f files/s | ETA %.1fh",
                    total, len(todo), n_done, n_errors, n_records, rate, eta_h,
                )

    _flush(force=True)
    elapsed = time.time() - start
    log.info(
        "Import complete. %d files | %d records | %d errors | %.0fs",
        n_done, n_records, n_errors, elapsed,
    )


if __name__ == "__main__":
    main()
