"""
将奇数批次 (batch_id % 2 == 1) 从 Data24T 移动到 Data6T。
偶数批次留在 Data24T 不动。逐文件复制，可中断重启。

用法:
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
        raise RuntimeError(f"Data24T 未挂载: {DATA24T_FILES}")
    if not os.path.isdir(DATA6T_FILES):
        raise RuntimeError(f"Data6T 未挂载: {DATA6T_FILES}")


def move_batch(name):
    """逐文件复制批次目录，全部成功后删除源目录。"""
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

    # 全部复制成功后删除源
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
    print(f"📊 Data24T 找到 {len(batch_dirs)} 个批次目录，其中奇数 {len(odd_dirs)} 个需移到 Data6T")

    if args.dry_run:
        print("[dry-run] 不执行移动")
        return

    moved = skipped = errors = 0
    for name in odd_dirs:
        dst_dir = os.path.join(DATA6T_FILES, name)
        src_dir = os.path.join(DATA24T_FILES, name)

        # 已完全迁移（源不存在或目标已有且源为空）
        if not os.path.exists(src_dir):
            skipped += 1
            continue
        if os.path.exists(dst_dir) and not os.listdir(src_dir):
            os.rmdir(src_dir)
            skipped += 1
            continue

        # 检查盘是否仍然可用
        if not os.path.isdir(DATA24T_FILES) or not os.path.isdir(DATA6T_FILES):
            print("⚠️  硬盘断开，停止迁移，可重新运行脚本继续")
            break

        try:
            move_batch(name)
            moved += 1
            if moved % 10 == 0:
                print(f"  ✅ moved {moved}/{len(odd_dirs)} (batch {name})")
        except Exception as e:
            print(f"  ❌ ERROR batch {name}: {type(e).__name__}: {e}")
            errors += 1

    print(f"\n完成: moved={moved}, skipped={skipped}, errors={errors}")
    if errors == 0 and moved > 0:
        print("✅ 全部成功，可重启收集脚本")


if __name__ == "__main__":
    main()
