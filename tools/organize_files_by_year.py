"""
把 files/ 目录下的 CSV 文件按年份移入子目录。
例如: files/20160101000000.gkg.csv → files/2016/20160101000000.gkg.csv

用法:
  python tools/organize_files_by_year.py
  python tools/organize_files_by_year.py --dry-run   # 只预览不移动
"""
import os
import sys
import argparse
from collections import defaultdict

FILES_DIR = "/Volumes/Data24T/docker-volumes/gdelt_cache/files"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true", help="只统计不移动")
    args = parser.parse_args()

    print(f"扫描目录: {FILES_DIR}")
    entries = os.listdir(FILES_DIR)
    # 只处理根目录下的文件（跳过已经是目录的年份文件夹）
    files = [f for f in entries if f.endswith(".gkg.csv") and os.path.isfile(os.path.join(FILES_DIR, f))]
    print(f"待移动文件: {len(files):,}")

    year_counts = defaultdict(int)
    for f in files:
        year = f[:4]
        year_counts[year] += 1

    print("\n各年份文件数:")
    for y in sorted(year_counts):
        print(f"  {y}: {year_counts[y]:,}")

    if args.dry_run:
        print("\n[dry-run] 未移动任何文件")
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
                print(f"  已移动 {moved:,} / {len(files):,}")
        except Exception as e:
            print(f"  ERROR {f}: {e}")
            errors += 1

    print(f"\n完成: 移动 {moved:,} 个文件, 错误 {errors} 个")


if __name__ == "__main__":
    main()
