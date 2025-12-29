import os
import io
import socket
import zipfile
import requests
import pandas as pd
from datetime import datetime, timedelta
from pymongo import MongoClient
from newspaper import Article, Config
import concurrent.futures
import time

# =====================================================
# 0. 环境配置
# =====================================================
def get_mongo_uri():
    try:
        socket.gethostbyname("mongo6")
        print("🔗 Running inside Docker network, using internal Mongo URI")
        return "mongodb://root:root@mongo6:27017/quant_data?authSource=admin"
    except socket.gaierror:
        print("💻 Running on external machine, using host Mongo URI")
        return "mongodb://root:root@192.168.1.26:37018/quant_data?authSource=admin"

MONGO_URI = os.getenv("MONGO_URI", get_mongo_uri())
DB_NAME = "quant_data"
COLLECTION_STOCKS = "stock_universe"
COLLECTION_NEWS = "news_articles"

CACHE_DIR = os.getenv("GDELT_CACHE", "./cache_gdelt")
FILES_DIR = os.path.join(CACHE_DIR, "files")
os.makedirs(FILES_DIR, exist_ok=True)

BASE_URL = "http://data.gdeltproject.org/gdeltv2"
MASTER_FILE = os.path.join(CACHE_DIR, "masterfilelist.txt")

# 时间范围：过去 5 年
END_DATE = datetime.utcnow()
START_DATE = END_DATE - timedelta(days=5 * 365)

# newspaper3k 配置
user_agent = "Mozilla/5.0 (Macintosh; Intel Mac OS X 14_0) AppleWebKit/605.1.15 (KHTML, like Gecko)"
config = Config()
config.browser_user_agent = user_agent
config.request_timeout = 10

# =====================================================
# 1. MongoDB 连接
# =====================================================
def get_mongo_client():
    client = MongoClient(MONGO_URI)
    return client[DB_NAME]

# =====================================================
# 2. 加载 masterfilelist.txt（带缓存）
# =====================================================
def load_masterfilelist():
    if os.path.exists(MASTER_FILE) and os.path.getsize(MASTER_FILE) > 100_000_000:
        print(f"📂 Using cached masterfilelist.txt ({os.path.getsize(MASTER_FILE)/1e6:.1f} MB)")
        with open(MASTER_FILE, "r") as f:
            return f.readlines()
    print("⬇️ Downloading full masterfilelist.txt (≈450MB, once only)...")
    with requests.get(f"{BASE_URL}/masterfilelist.txt", stream=True, verify=False, timeout=1800) as r:
        r.raise_for_status()
        with open(MASTER_FILE, "wb") as f:
            for chunk in r.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)
    print(f"✅ Saved to {MASTER_FILE}")
    with open(MASTER_FILE, "r") as f:
        return f.readlines()

# =====================================================
# 3. 按时间分批生成 GKG 文件（2天一批）
# =====================================================
def generate_batches(start, end, days_per_batch=2):
    batches = []
    cursor = start
    while cursor < end:
        batch_end = min(cursor + timedelta(days=days_per_batch), end)
        batches.append((cursor, batch_end))
        cursor = batch_end
    return batches

def get_gkg_file_urls(start, end):
    lines = load_masterfilelist()
    urls = []
    for line in lines:
        if ".gkg.csv.zip" not in line:
            continue
        parts = line.split()
        if len(parts) < 3:
            continue
        url = parts[2].strip()
        ts_str = os.path.basename(url).split(".")[0]
        try:
            ts = datetime.strptime(ts_str, "%Y%m%d%H%M%S")
        except:
            continue
        if start <= ts <= end:
            urls.append(url)
    print(f"✅ Found {len(urls)} GKG files between {start.date()} → {end.date()}")
    return urls

# =====================================================
# 4. 解析 GDELT 文件（带详细下载日志）
# =====================================================
def parse_gdelt_files(urls, company):
    all_rows = []
    cache_hits = downloads = 0

    print(f"\n🔍 Matching company: {company}")
    print(f"📦 Total GDELT files this batch: {len(urls)}")

    for idx, url in enumerate(urls, 1):
        filename = os.path.basename(url)
        cache_path = os.path.join(FILES_DIR, filename)

        if os.path.exists(cache_path):
            cache_hits += 1
            size_mb = os.path.getsize(cache_path) / 1e6
            print(f"[{idx}/{len(urls)}] 📂 Cached {filename} ({size_mb:.2f} MB)")
            with open(cache_path, "rb") as f:
                data = io.BytesIO(f.read())
        else:
            downloads += 1
            print(f"[{idx}/{len(urls)}] ⬇️ Downloading {filename} ...")
            try:
                r = requests.get(url, timeout=90)
                r.raise_for_status()
                with open(cache_path, "wb") as f:
                    f.write(r.content)
                size_mb = len(r.content) / 1e6
                print(f"✅ Downloaded {filename} ({size_mb:.2f} MB)")
                data = io.BytesIO(r.content)
            except Exception as e:
                print(f"❌ Download failed {filename}: {e}")
                continue

        try:
            z = zipfile.ZipFile(data)
            with z.open(z.namelist()[0]) as f:
                df = pd.read_csv(f, sep="\t", header=None, encoding="ISO-8859-1", on_bad_lines="skip")

            if 26 not in df.columns:
                continue
            df = df[[26]]
            df.columns = ["Raw"]

            df["Title"] = df["Raw"].str.extract(r"<PAGE_TITLE>(.*?)</PAGE_TITLE>", expand=False)
            df["AltURL"] = df["Raw"].str.extract(r"<PAGE_ALTURL_AMP>(.*?)</PAGE_ALTURL_AMP>", expand=False)
            df["Links"] = df["Raw"].str.extract(r"<PAGE_LINKS>(.*?)</PAGE_LINKS>", expand=False)
            df["URL"] = df["AltURL"].combine_first(df["Links"]).astype(str).str.split(";").str[0]

            mask = df["Title"].astype(str).str.contains(company, case=False, na=False)
            df = df[mask & df["URL"].str.startswith(("http://", "https://"))]

            if not df.empty:
                df["Date"] = filename[:8]
                df = df[["Date", "Title", "URL"]]
                all_rows.append(df)

        except Exception as e:
            print(f"⚠️ Error parsing {filename}: {e}")

    print(f"\n📦 Cache hits: {cache_hits} | Downloads: {downloads}")
    if not all_rows:
        return pd.DataFrame()
    combined = pd.concat(all_rows, ignore_index=True).drop_duplicates(subset=["URL", "Title"])
    print(f"📰 Found {len(combined)} articles for {company}")
    return combined

# =====================================================
# 5. 抓取正文（多线程）
# =====================================================
def fetch_article(row, company):
    url = str(row["URL"])
    try:
        art = Article(url, config=config)
        art.download()
        art.parse()
        title = art.title.strip()
        content = art.text.strip().replace("\n", " ")
        if len(content) < 100:
            return None
        return {
            "company": company,
            "title": title,
            "url": url,
            "content": content,
            "date": row.get("Date", ""),
            "source": "GDELT",
            "collectedAt": datetime.utcnow(),
        }
    except Exception:
        return None

def crawl_articles(df, company):
    results = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=6) as executor:
        futures = [executor.submit(fetch_article, row, company) for _, row in df.iterrows()]
        for future in concurrent.futures.as_completed(futures):
            res = future.result()
            if res:
                results.append(res)
    print(f"✅ Extracted {len(results)} valid articles for {company}")
    return results

# =====================================================
# 6. 主逻辑
# =====================================================
def main():
    db = get_mongo_client()
    companies = [x["name"] for x in db[COLLECTION_STOCKS].find({}, {"name": 1})]
    print(f"📊 Loaded {len(companies)} companies from MongoDB.")

    batches = generate_batches(START_DATE, END_DATE, days_per_batch=2)
    print(f"🗓️ Total batches: {len(batches)} (2 days each)")

    for company in companies:
        print(f"\n==============================")
        print(f"🏢 Processing company: {company}")

        for batch_start, batch_end in batches:
            print(f"\n📆 {batch_start.date()} → {batch_end.date()}")
            urls = get_gkg_file_urls(batch_start, batch_end)
            if not urls:
                continue
            df = parse_gdelt_files(urls, company)
            if df.empty:
                continue
            articles = crawl_articles(df, company)
            if articles:
                db[COLLECTION_NEWS].insert_many(articles, ordered=False)
                print(f"💾 Saved {len(articles)} articles for {company} ({batch_start.date()}→{batch_end.date()})")

            # 每批暂停几秒
            time.sleep(2)

if __name__ == "__main__":
    main()