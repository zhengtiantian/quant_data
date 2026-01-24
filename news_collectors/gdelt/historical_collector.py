import os
import requests
import zipfile
import io
import pandas as pd
from datetime import datetime, timedelta
from newspaper import Article, Config
import concurrent.futures
import re
import random
from pymongo import MongoClient, errors
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

# ============================
# 全局配置
# ============================
MONGO_URI = "mongodb://root:root@192.168.1.26:37018/"
DB_NAME = "quant_data"
SRC_COLLECTION = "stock_universe"
DST_COLLECTION = "news_articles"

YEARS_BACK = 10
MAX_FILES = None  # 全量
CACHE_DIR = "/mnt/data24t/docker-volumes/gdelt_cache"  # 存储到 22TB 硬盘
FILES_DIR = os.path.join(CACHE_DIR, "files")
os.makedirs(CACHE_DIR, exist_ok=True)
os.makedirs(FILES_DIR, exist_ok=True)

BASE_URL = "http://data.gdeltproject.org/gdeltv2"
MASTER_FILE = os.path.join(CACHE_DIR, "masterfilelist.txt")

# User-Agent 配置
user_agent = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 14_0) "
    "AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.0 Safari/605.1.15"
)
config = Config()
config.browser_user_agent = user_agent
config.request_timeout = 10


# ============================
# MongoDB 操作
# ============================
def get_db():
    client = MongoClient(MONGO_URI)
    return client[DB_NAME]


def load_companies():
    db = get_db()
    col = db[SRC_COLLECTION]
    data = list(col.find({}, {"_id": 0, "symbol": 1, "name": 1, "related_keywords": 1}))
    print(f"✅ Loaded {len(data)} companies from {SRC_COLLECTION}")
    return data


def ensure_index():
    """为 URL 建立唯一索引"""
    db = get_db()
    col = db[DST_COLLECTION]
    try:
        col.create_index("url", unique=True)
        print("✅ MongoDB index created on 'url'")
    except errors.OperationFailure:
        print("ℹ️ Index already exists on 'url'")


# ============================
# masterfilelist 缓存
# ============================
def load_masterfilelist():
    if os.path.exists(MASTER_FILE):
        print(f"📂 Using cached masterfilelist.txt ({os.path.getsize(MASTER_FILE)/1e6:.1f} MB)")
        with open(MASTER_FILE, "r") as f:
            return f.readlines()
    print("⬇️ Downloading masterfilelist.txt (≈450MB, first time only)...")
    with requests.get(f"{BASE_URL}/masterfilelist.txt", stream=True, timeout=1800) as r:
        r.raise_for_status()
        with open(MASTER_FILE, "wb") as f:
            for chunk in r.iter_content(8192):
                f.write(chunk)
    with open(MASTER_FILE, "r") as f:
        return f.readlines()


# ============================
# 并行批量下载（自动跳过已缓存）
# ============================
def download_with_retry(url, retries=3, backoff=3):
    """下载单个文件（带重试机制）"""
    for attempt in range(retries):
        try:
            r = requests.get(url, timeout=90)
            r.raise_for_status()
            size_mb = len(r.content) / 1e6
            print(f"✅ Downloaded {os.path.basename(url)} ({size_mb:.2f} MB)")
            return r.content
        except Exception as e:
            wait = backoff * (attempt + 1)
            print(f"⚠️ Download failed {attempt+1}/{retries} for {url}: {e} -> retrying in {wait}s")
            time.sleep(wait)
    print(f"❌ Permanent failure downloading {url}")
    return None


def batch_download_files(urls, batch_size=10):
    """并行批量下载 GDELT 文件，已缓存文件自动跳过"""
    to_download = [
        url for url in urls
        if not os.path.exists(os.path.join(FILES_DIR, os.path.basename(url)))
    ]
    skipped = len(urls) - len(to_download)
    print(f"📂 Cached files skipped: {skipped}")
    print(f"⬇️ Need to download: {len(to_download)} files\n")

    for i in range(0, len(to_download), batch_size):
        batch = to_download[i:i + batch_size]
        print(f"🚀 Batch {i//batch_size + 1}: downloading {len(batch)} files...")

        with ThreadPoolExecutor(max_workers=5) as executor:
            futures = {executor.submit(download_with_retry, url): url for url in batch}
            for future in as_completed(futures):
                url = futures[future]
                try:
                    content = future.result()
                    if content:
                        filename = os.path.basename(url)
                        path = os.path.join(FILES_DIR, filename)
                        with open(path, "wb") as f:
                            f.write(content)
                except Exception as e:
                    print(f"❌ Unexpected error for {url}: {e}")

        print(f"✅ Batch {i//batch_size + 1} complete. Sleeping 1s before next batch.\n")
        time.sleep(1)

    print("🎯 All downloads finished (cached files skipped automatically).")


# ============================
# 获取时间范围内文件 URL
# ============================
def get_gkg_file_urls(years_back, max_files=None):
    lines = load_masterfilelist()
    end_dt = datetime.utcnow()
    start_dt = end_dt - timedelta(days=years_back * 365)
    urls = []
    for line in lines:
        if ".gkg.csv.zip" not in line:
            continue
        url = line.split()[-1]
        ts_str = os.path.basename(url).split(".")[0]
        try:
            ts = datetime.strptime(ts_str, "%Y%m%d%H%M%S")
        except:
            continue
        if start_dt <= ts <= end_dt:
            urls.append(url)
        if max_files and len(urls) >= max_files:
            break
    print(f"✅ Found {len(urls)} GKG files from {start_dt.date()} → {end_dt.date()}")
    return urls


# ============================
# 清洗公司名
# ============================
def clean_company_name(name):
    if not name:
        return ""
    name = re.sub(r"[,\.\n\r\t]", " ", name)
    name = re.sub(r"\s+", " ", name)
    name = re.sub(r"\b(Inc|Corporation|Corp|Ltd|LLC|PLC|Group|Co|Company)\b", "", name, flags=re.I)
    return name.strip()


# ============================
# 解析 GDELT 文件（从缓存读取）
# ============================
def parse_gdelt_files(urls, company_name, keywords):
    all_rows = []
    pattern = "|".join(re.escape(k) for k in [company_name] + keywords if k)
    print(f"🔍 Searching for: {company_name} ({len(keywords)} extra keywords)")

    for idx, url in enumerate(urls, 1):
        filename = os.path.basename(url)
        cache_path = os.path.join(FILES_DIR, filename)
        if not os.path.exists(cache_path):
            continue

        try:
            z = zipfile.ZipFile(cache_path)
            with z.open(z.namelist()[0]) as f:
                df = pd.read_csv(f, sep="\t", header=None, encoding="ISO-8859-1", on_bad_lines="skip")

            if 26 not in df.columns:
                continue

            df = df[[26]].copy()
            df.columns = ["Raw"]
            df["Raw"] = df["Raw"].astype(str)

            df["Title"] = df["Raw"].str.extract(r"<PAGE_TITLE>(.*?)</PAGE_TITLE>", expand=False).fillna("")
            df["URL"] = df["Raw"].str.extract(r"<PAGE_LINKS>(.*?)</PAGE_LINKS>", expand=False).fillna("")
            df["URL"] = df["URL"].astype(str).str.split(";").str[0]
            df = df[df["URL"].str.startswith(("http", "https"), na=False)]

            mask = df["Title"].astype(str).str.contains(pattern, case=False, na=False)
            df = df[mask]

            if df.empty:
                continue

            df["Date"] = filename[:8]
            df = df[["Date", "Title", "URL"]]
            all_rows.append(df)

            if idx % 50 == 0:
                print(f"📦 Processed {idx} cached GKG files...")

        except Exception as e:
            print(f"⚠️ Error reading {filename}: {e}")

    if not all_rows:
        print("⚠️ No matches found across all cached files.")
        return pd.DataFrame()
    combined = pd.concat(all_rows, ignore_index=True).drop_duplicates(subset=["URL"])
    print(f"📊 Total unique URLs found: {len(combined)}")
    return combined


# ============================
# 抓取新闻正文
# ============================
def fetch_article(row, company):
    url = str(row["URL"])
    try:
        art = Article(url, config=config)
        art.download()
        art.parse()
        if len(art.text.strip()) < 100:
            return None
        return {
            "symbol": company["symbol"],
            "name": company["name"],
            "date": row.get("Date", ""),
            "title": art.title.strip(),
            "url": url,
            "content": art.text.strip(),
            "source": {"platform": "gdelt"},
            "collectedAt": datetime.utcnow().isoformat()
        }
    except Exception as e:
        print(f"❌ [Error] Failed to parse {url}: {e}")
        return None


def crawl_articles(df, company):
    print(f"\n📰 Fetching {len(df)} articles for {company['symbol']} ({company['name']}) ...")
    results = []
    with ThreadPoolExecutor(max_workers=5) as executor:
        futures = [executor.submit(fetch_article, row, company) for _, row in df.iterrows()]
        for i, future in enumerate(as_completed(futures), 1):
            res = future.result()
            if res:
                results.append(res)
                print(f"[{i}] ✅ {res['title'][:70]}")
            if i % 10 == 0:
                print(f"📊 Progress: {i}/{len(df)} articles processed...")
    print(f"✅ Completed {len(results)} valid articles.\n")
    return results


# ============================
# 主逻辑
# ============================
if __name__ == "__main__":
    print(f"🕒 Running full GDELT extraction for last {YEARS_BACK} years (parallel download + cache skip)...\n")

    ensure_index()
    urls = get_gkg_file_urls(YEARS_BACK, MAX_FILES)
    batch_download_files(urls)  # ✅ 并行下载 + 缓存跳过

    companies = load_companies()
    db = get_db()
    dst_col = db[DST_COLLECTION]

    for idx, company in enumerate(companies, 1):
        symbol = company.get("symbol", "")
        name = company.get("name", "")

        print(f"\n============================")
        print(f"🏢 [{idx}/{len(companies)}] Processing {symbol} - {name}")
        print("============================")

        cleaned_name = clean_company_name(name)
        keywords = company.get("related_keywords", []) or []

        df = parse_gdelt_files(urls, cleaned_name, keywords)
        if df.empty:
            print("⚠️ No matches found.")
            continue

        articles = crawl_articles(df, company)
        inserted = 0

        for art in articles:
            try:
                dst_col.update_one(
                    {"url": art["url"]},
                    {"$setOnInsert": art},
                    upsert=True
                )
                inserted += 1
            except errors.DuplicateKeyError:
                pass

        print(f"💾 Inserted or confirmed {inserted} unique docs for {symbol}.")
        print("=" * 120)
        time.sleep(2)

    print("🏁 All companies processed successfully!")