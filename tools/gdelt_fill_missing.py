#!/usr/bin/env python3
"""Validate and backfill missing GDELT GKG files (MongoDB edition).

Logic:
  1. Read masterfilelist.txt (stored on Data4T; auto-downloaded if absent)
  2. Diff against MongoDB gkg_import_progress collection to find missing files
  3. Download missing zips → extract csv → import into gkg_index → delete csv/zip

Usage:
  # Validate only, no download
  DRY_RUN=true python tools/gdelt_fill_missing.py

  # Validate + download + import
  caffeinate -i env FILL_WORKERS=10 python tools/gdelt_fill_missing.py
"""

from __future__ import annotations

import csv
import logging
import os
import re
import time
import zipfile
from concurrent.futures import ThreadPoolExecutor, as_completed

import requests
import pymongo
from pymongo import MongoClient

csv.field_size_limit(10 * 1024 * 1024)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)
log = logging.getLogger(__name__)

# ── Config ────────────────────────────────────────────────────────────────────
DATA4T       = os.getenv("GDELT_CACHE_DIR", "/Volumes/Data4T/docker-volumes/gdelt_cache")
TMP_DIR      = os.getenv("GKG_TMP_DIR",    "/Volumes/Data4T/gdelt_tmp")
MASTER       = os.path.join(DATA4T, "masterfilelist.txt")
MASTER_URL   = "http://data.gdeltproject.org/gdeltv2/masterfilelist.txt"
BASE_URL     = "http://data.gdeltproject.org/gdeltv2"

MONGO_URI    = os.getenv("LOCAL_MONGO_URI", "mongodb://root:root@127.0.0.1:37018/")
DB_NAME      = "quant_data"
GKG_COL      = "gkg_index"
PROG_COL     = "gkg_import_progress"

START_DATE   = os.getenv("GDELT_START_DATE", "20160101")
WORKERS      = int(os.getenv("FILL_WORKERS", "8"))
DRY_RUN      = os.getenv("DRY_RUN", "false").lower() == "true"

ENTITY_COLS  = [7, 9, 11, 23, 24, 26, 28]
URL_COL      = 4
_XML_RE      = re.compile(r"(?is)<[^>]+>")
_URL_RE      = re.compile(r"https?://\S+|\b(?:www\.)?[a-z0-9.-]+\.[a-z]{2,}(?:/\S*)?\b", re.I)
_WS_RE       = re.compile(r"\s+")


# ── masterfilelist ────────────────────────────────────────────────────────────

def ensure_masterfilelist() -> str:
    os.makedirs(DATA4T, exist_ok=True)
    if not os.path.exists(MASTER):
        log.info("masterfilelist not found, downloading from GDELT...")
        r = requests.get(MASTER_URL, timeout=120, stream=True)
        r.raise_for_status()
        with open(MASTER, "wb") as f:
            for chunk in r.iter_content(65536):
                f.write(chunk)
        log.info("Downloaded masterfilelist → %s", MASTER)
    else:
        age_h = (time.time() - os.path.getmtime(MASTER)) / 3600
        log.info("masterfilelist found (age=%.1fh): %s", age_h, MASTER)
    return MASTER


def load_gkg_entries(master_path: str) -> list[tuple[str, str]]:
    """Read masterfilelist → list of (csv_name, zip_url) from START_DATE onwards."""
    entries: list[tuple[str, str]] = []
    with open(master_path, encoding="utf-8", errors="replace") as fh:
        for line in fh:
            parts = line.strip().split()
            if not parts:
                continue
            url = parts[-1]
            if ".gkg.csv.zip" not in url:
                continue
            ts = os.path.basename(url)[:8]
            if ts < START_DATE:
                continue
            csv_name = os.path.basename(url)[:-4]  # drop .zip
            entries.append((csv_name, url))
    return entries


# ── MongoDB diff ──────────────────────────────────────────────────────────────

def find_missing(entries: list[tuple[str, str]], prog_col) -> list[tuple[str, str]]:
    """Return entries whose csv_name is NOT in gkg_import_progress."""
    all_names = [name for name, _ in entries]
    log.info("Checking %d entries against MongoDB...", len(all_names))

    existing: set[str] = set()
    CHUNK = 10000
    for i in range(0, len(all_names), CHUNK):
        chunk = all_names[i:i + CHUNK]
        docs = prog_col.find({"_id": {"$in": chunk}}, {"_id": 1})
        existing.update(doc["_id"] for doc in docs)
        if (i // CHUNK) % 5 == 0:
            log.info("  checked %d/%d, found %d existing", min(i + CHUNK, len(all_names)), len(all_names), len(existing))

    missing = [(name, url) for name, url in entries if name not in existing]
    return missing


# ── Parse + import ────────────────────────────────────────────────────────────

def _strip(text: str) -> str:
    text = re.sub(r"(?is)<PAGE_LINKS>.*?</PAGE_LINKS>", " ", text)
    text = _XML_RE.sub(" ", text)
    text = _URL_RE.sub(" ", text)
    return _WS_RE.sub(" ", text).strip()


def _parse_and_import(csv_path: str, gkg_col, prog_col) -> int:
    ts = os.path.basename(csv_path)[:14]
    docs: list[dict] = []
    try:
        with open(csv_path, encoding="ISO-8859-1", errors="replace", newline="") as fh:
            reader = csv.reader(fh, delimiter="\t", quoting=csv.QUOTE_NONE)
            for row in reader:
                if len(row) <= URL_COL:
                    continue
                col26 = row[26] if len(row) > 26 else ""
                if "<PAGE_LINKS>" in col26:
                    m = re.search(r"<PAGE_LINKS>(.*?)</PAGE_LINKS>", col26, re.S)
                    url = (m.group(1).split(";")[0] if m else "").strip()
                    raw = _strip(col26)
                else:
                    url = row[URL_COL].strip()
                    parts = [row[c] for c in ENTITY_COLS if c < len(row) and row[c]]
                    raw = _strip(" ".join(parts))
                if not url.startswith("http"):
                    continue
                docs.append({"ts": ts, "url": url, "raw": raw})
    except Exception as e:
        log.warning("Parse error %s: %s", csv_path, e)
        return 0

    if docs:
        try:
            gkg_col.insert_many(docs, ordered=False)
        except pymongo.errors.BulkWriteError:
            pass

    csv_name = os.path.basename(csv_path)
    try:
        prog_col.insert_one({"_id": csv_name, "ts": time.time()})
    except pymongo.errors.DuplicateKeyError:
        pass

    return len(docs)


# ── Download ──────────────────────────────────────────────────────────────────

def _download_one(name: str, url: str, gkg_col, prog_col) -> tuple[str, bool, int]:
    os.makedirs(TMP_DIR, exist_ok=True)
    zip_path = os.path.join(TMP_DIR, name + ".zip")
    csv_path = os.path.join(TMP_DIR, name)

    for attempt in range(3):
        try:
            r = requests.get(url, timeout=90)
            if r.status_code == 404:
                log.warning("404 %s", name)
                return name, False, 0
            r.raise_for_status()
            with open(zip_path, "wb") as f:
                f.write(r.content)
            with zipfile.ZipFile(zip_path) as z:
                inner = z.namelist()[0]
                with z.open(inner) as src, open(csv_path, "wb") as dst:
                    dst.write(src.read())
            os.remove(zip_path)

            n = _parse_and_import(csv_path, gkg_col, prog_col)
            os.remove(csv_path)
            return name, True, n

        except Exception as e:
            if attempt == 2:
                log.warning("FAIL %s: %s", name, e)
                for p in (zip_path, csv_path):
                    try:
                        os.remove(p)
                    except OSError:
                        pass
                return name, False, 0
            time.sleep(3 * (attempt + 1))

    return name, False, 0


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    client = MongoClient(MONGO_URI, serverSelectionTimeoutMS=5000)
    client.admin.command("ping")
    db = client[DB_NAME]
    gkg_col  = db[GKG_COL]
    prog_col = db[PROG_COL]

    master_path = ensure_masterfilelist()
    entries = load_gkg_entries(master_path)
    log.info("masterfilelist GKG entries (from %s): %d", START_DATE, len(entries))

    missing = find_missing(entries, prog_col)
    log.info("Missing from MongoDB: %d / %d", len(missing), len(entries))

    if not missing:
        log.info("All files present in MongoDB. Nothing to do.")
        return

    if DRY_RUN:
        log.info("DRY_RUN=true — stopping here. Sample missing:")
        for name, url in missing[:10]:
            log.info("  %s", name)
        return

    log.info("Downloading %d missing files (workers=%d) → %s", len(missing), WORKERS, TMP_DIR)
    start   = time.time()
    ok = fail = records = 0

    with ThreadPoolExecutor(max_workers=WORKERS) as pool:
        futures = {pool.submit(_download_one, name, url, gkg_col, prog_col): name
                   for name, url in missing}
        for future in as_completed(futures):
            name, success, n = future.result()
            if success:
                ok      += 1
                records += n
            else:
                fail += 1
            total = ok + fail
            log.info("[%d/%d] %s  records=%d  %s",
                     total, len(missing), "OK  " if success else "FAIL", n, name)

    elapsed = time.time() - start
    log.info("Done. ok=%d fail=%d records=%d | %.0fs", ok, fail, records, elapsed)


if __name__ == "__main__":
    main()
