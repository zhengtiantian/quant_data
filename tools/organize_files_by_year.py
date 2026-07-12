"""
Move CSV files under files/ into per-year subdirectories.
Example: files/20160101000000.gkg.csv → files/2016/20160101000000.gkg.csv

Usage:
  python tools/organize_files_by_year.py
  python tools/organize_files_by_year.py --dry-run   # preview only, no moves
"""
import os
import argparse
from collections import defaultdict

FILES_DIR = "/Volumes/Data24T/docker-volumes/gdelt_cache/files"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true", help="count only, no moves")
    args = parser.parse_args()

    print(f"Scanning: {FILES_DIR}")
    entries = os.listdir(FILES_DIR)
    # process only root-level files (skip year subdirectories already created)
    files = [f for f in entries if f.endswith(".gkg.csv") and os.path.isfile(os.path.join(FILES_DIR, f))]
    print(f"Files to move: {len(files):,}")

    year_counts = defaultdict(int)
    for f in files:
        year = f[:4]
        year_counts[year] += 1

    print("\nFiles per year:")
    for y in sorted(year_counts):
        print(f"  {y}: {year_counts[y]:,}")

    if args.dry_run:
        print("\n[dry-run] no files moved")
        return

    moved = 0
    errors = 0
    for f in files:
        year = f[:4]
        year_dir = os.path.join(FILES_DIR, year)
        if not os.path.exists(year_dir):
            os.makedirs(year_dir, exist_ok=True)
        src = os.path.join(FILES_DIR, f)
        dst = os.path.join(year_dir, f)
        try:
            os.rename(src, dst)
            moved += 1
            if moved % 10000 == 0:
                print(f"  moved {moved:,} / {len(files):,}")
        except Exception as e:
            print(f"  ERROR {f}: {e}")
            errors += 1

    print(f"\nDone: moved {moved:,} files, {errors} errors")


if __name__ == "__main__":
    main()
