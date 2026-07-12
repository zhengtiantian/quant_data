"""
Move odd-numbered batches (batch_id % 2 == 1) from Data24T to Data6T.
Even batches stay on Data24T. Copies file-by-file so it can be interrupted and resumed.

Usage:
  python tools/split_batches_to_drives.py --dry-run
  python tools/split_batches_to_drives.py
"""
import os
import shutil
import argparse

DATA24T_FILES = "/Volumes/Data24T/docker-volumes/gdelt_cache/files"
DATA6T_FILES  = "/Volumes/Data6T/gdelt_cache/files"


def check_drives():
    if not os.path.isdir(DATA24T_FILES):
        raise RuntimeError(f"Data24T not mounted: {DATA24T_FILES}")
    if not os.path.isdir(DATA6T_FILES):
        raise RuntimeError(f"Data6T not mounted: {DATA6T_FILES}")


def move_batch(name):
    """Copy a batch directory file-by-file, then delete the source once all files succeed."""
    src_dir = os.path.join(DATA24T_FILES, name)
    dst_dir = os.path.join(DATA6T_FILES, name)
    os.makedirs(dst_dir, exist_ok=True)

    files = os.listdir(src_dir)
    for f in files:
        src_f = os.path.join(src_dir, f)
        dst_f = os.path.join(dst_dir, f)
        if os.path.exists(dst_f):
            continue
        shutil.copy2(src_f, dst_f)

    # delete source only after all files copied successfully
    shutil.rmtree(src_dir)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    check_drives()

    entries = os.listdir(DATA24T_FILES)
    batch_dirs = sorted(
        [e for e in entries if e.isdigit()],
        key=lambda x: int(x)
    )
    odd_dirs = [e for e in batch_dirs if int(e) % 2 == 1]
    print(f"Data24T: {len(batch_dirs)} batch dirs found, {len(odd_dirs)} odd batches to move to Data6T")

    if args.dry_run:
        print("[dry-run] no moves executed")
        return

    moved = skipped = errors = 0
    for name in odd_dirs:
        dst_dir = os.path.join(DATA6T_FILES, name)
        src_dir = os.path.join(DATA24T_FILES, name)

        # already fully migrated (source gone or destination exists and source is empty)
        if not os.path.exists(src_dir):
            skipped += 1
            continue
        if os.path.exists(dst_dir) and not os.listdir(src_dir):
            os.rmdir(src_dir)
            skipped += 1
            continue

        # abort if either drive became unavailable
        if not os.path.isdir(DATA24T_FILES) or not os.path.isdir(DATA6T_FILES):
            print("Drive disconnected — stopping; re-run the script to resume")
            break

        try:
            move_batch(name)
            moved += 1
            if moved % 10 == 0:
                print(f"  ✅ moved {moved}/{len(odd_dirs)} (batch {name})")
        except Exception as e:
            print(f"  ❌ ERROR batch {name}: {type(e).__name__}: {e}")
            errors += 1

    print(f"\nDone: moved={moved}, skipped={skipped}, errors={errors}")
    if errors == 0 and moved > 0:
        print("All batches moved successfully")


if __name__ == "__main__":
    main()
