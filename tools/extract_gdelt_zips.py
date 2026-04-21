"""
预解压 GDELT zip 文件 → .gkg.csv，解压完删除原 zip。
支持多线程并行，可中断重启（已解压的自动跳过）。
"""
import os
import zipfile
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from glob import glob

FILES_DIR = os.getenv(
    "GDELT_FILES_DIR",
    "/Volumes/Data24T/docker-volumes/gdelt_cache/files"
)
WORKERS = int(os.getenv("EXTRACT_WORKERS", "4"))


def extract_one(zip_path):
    csv_path = zip_path[:-4]  # strip .zip
    if os.path.exists(csv_path):
        return "skip", zip_path
    try:
        with zipfile.ZipFile(zip_path) as z:
            inner = z.namelist()[0]
            with z.open(inner) as src, open(csv_path, "wb") as dst:
                dst.write(src.read())
        os.remove(zip_path)
        return "ok", zip_path
    except Exception as e:
        return "err", f"{zip_path}: {e}"


def main():
    # 扫描根目录和所有年份子目录
    zips = sorted(
        glob(os.path.join(FILES_DIR, "*.gkg.csv.zip")) +
        glob(os.path.join(FILES_DIR, "*", "*.gkg.csv.zip"))
    )
    total = len(zips)
    print(f"📦 Found {total} zip files in {FILES_DIR} (incl. year subdirs)")
    print(f"🔧 Workers: {WORKERS}\n")

    if total == 0:
        print("✅ Nothing to extract.")
        return

    done = skip = err = 0
    with ThreadPoolExecutor(max_workers=WORKERS) as ex:
        futures = {ex.submit(extract_one, z): z for z in zips}
        for i, fut in enumerate(as_completed(futures), 1):
            status, info = fut.result()
            if status == "ok":
                done += 1
            elif status == "skip":
                skip += 1
            else:
                err += 1
                print(f"❌ {info}")
            if i % 1000 == 0 or i == total:
                print(f"  [{i}/{total}] extracted={done} skipped={skip} errors={err}")

    print(f"\n✅ Done: extracted={done}, skipped={skip}, errors={err}")


if __name__ == "__main__":
    main()
