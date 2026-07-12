"""
Move CSV files under files/ into batch_XXXX subdirectories, 100 files per batch,
following masterfilelist order. batch_id maps 1-to-1 with the MySQL gdelt_batch_tasks table.

Usage:
  python tools/organize_files_by_batch.py --dry-run   # preview only
  python tools/organize_files_by_batch.py             # execute
"""
import os
import argparse
from datetime import datetime

CACHE_DIR = os.getenv("GDELT_CACHE_DIR", "/Volumes/Data24T/docker-volumes/gdelt_cache")
FILES_DIR = os.path.join(CACHE_DIR, "files")
MASTER_FILE = os.path.join(CACHE_DIR, "masterfilelist.txt")
BATCH_SIZE = 100
START_DATE = datetime(2016, 1, 1)


def get_ordered_urls():
    """Read .gkg.csv.zip URLs from masterfilelist, filtered to >= 2016, sorted ascending."""
    urls = []
    with open(MASTER_FILE, "r") as f:
        for line in f:
            if ".gkg.csv.zip" not in line:
                continue
            url = line.split()[-1]
            ts_str = os.path.basename(url).split(".")[0]
            try:
                ts = datetime.strptime(ts_str, "%Y%m%d%H%M%S")
                if ts >= START_DATE:
                    urls.append(url)
            except Exception:
                continue
    urls.sort()
    return urls


def csv_name(filename):
    return filename[:-4] if filename.endswith(".zip") else filename


def find_file(filename):
    """Locate a file in the year subdirectory or root; return path or None."""
    csv = csv_name(filename)
    year = filename[:4]
    candidates = [
        os.path.join(FILES_DIR, year, csv),
        os.path.join(FILES_DIR, year, filename),
        os.path.join(FILES_DIR, csv),
        os.path.join(FILES_DIR, filename),
    ]
    for p in candidates:
        if os.path.exists(p):
            return p
    return None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    print(f"📖 Reading masterfilelist: {MASTER_FILE}")
    urls = get_ordered_urls()
    total_batches = (len(urls) + BATCH_SIZE - 1) // BATCH_SIZE
    print(f"📊 Total files: {len(urls):,}, batches: {total_batches:,}")

    found = missing = moved = errors = 0

    for i in range(0, len(urls), BATCH_SIZE):
        batch_id = i // BATCH_SIZE + 1
        batch_dir = os.path.join(FILES_DIR, str(batch_id))
        batch_urls = urls[i:i + BATCH_SIZE]

        for url in batch_urls:
            filename = os.path.basename(url)
            src = find_file(filename)

            if src is None:
                missing += 1
                continue

            found += 1
            dst = os.path.join(batch_dir, os.path.basename(src))

            if src == dst:
                continue

            if args.dry_run:
                moved += 1
                continue

            os.makedirs(batch_dir, exist_ok=True)
            try:
                os.rename(src, dst)
                moved += 1
            except Exception as e:
                print(f"  ERROR batch_{batch_id:04d}/{os.path.basename(src)}: {e}")
                errors += 1

        if batch_id % 500 == 0 or batch_id == total_batches:
            print(f"  [{batch_id}/{total_batches}] found={found:,} moved={moved:,} missing={missing:,} errors={errors}")

    tag = "[dry-run] " if args.dry_run else ""
    print(f"\n{tag}Done: found={found:,}, moved={moved:,}, missing={missing:,}, errors={errors}")


if __name__ == "__main__":
    main()
