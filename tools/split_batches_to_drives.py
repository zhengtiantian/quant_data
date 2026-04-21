"""
将奇数批次 (batch_id % 2 == 1) 从 Data24T 移动到 Data6T。
偶数批次留在 Data24T 不动。

用法:
  python tools/split_batches_to_drives.py --dry-run
  python tools/split_batches_to_drives.py
"""
import os
import shutil
import argparse

DATA24T_FILES = "/Volumes/Data24T/docker-volumes/gdelt_cache/files"
DATA6T_FILES  = "/Volumes/Data6T/gdelt_cache/files"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    if not os.path.exists(DATA6T_FILES):
        os.makedirs(DATA6T_FILES, exist_ok=True)
        print(f"📁 Created {DATA6T_FILES}")

    # 找出所有数字目录（批次目录）
    entries = os.listdir(DATA24T_FILES)
    batch_dirs = sorted(
        [e for e in entries if e.isdigit()],
        key=lambda x: int(x)
    )
    print(f"📊 Found {len(batch_dirs)} batch dirs in Data24T")

    odd_total = sum(1 for e in batch_dirs if int(e) % 2 == 1)
    print(f"   偶数(留Data24T): {len(batch_dirs) - odd_total}  奇数(移Data6T): {odd_total}")
    if args.dry_run:
        print("[dry-run] 不执行移动")
        return

    moved = skipped = errors = 0
    for name in batch_dirs:
        batch_id = int(name)
        if batch_id % 2 == 0:
            continue  # 偶数留原处

        src = os.path.join(DATA24T_FILES, name)
        dst = os.path.join(DATA6T_FILES, name)

        if os.path.exists(dst):
            skipped += 1
            continue

        try:
            shutil.move(src, dst)
            moved += 1
            if moved % 200 == 0:
                print(f"  moved {moved}/{odd_total}...")
        except Exception as e:
            print(f"  ERROR {name}: {e}")
            errors += 1

    print(f"\n完成: moved={moved}, skipped={skipped}, errors={errors}")


if __name__ == "__main__":
    main()
