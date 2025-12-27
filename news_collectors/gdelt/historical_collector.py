import os
import requests
import zipfile
import io
import pandas as pd
from datetime import datetime, timedelta
from newspaper import Article, Config
import time
import re
import concurrent.futures

# ============================
# 配置参数
# ============================
COMPANY = "Apple"
START_DATE = "2020-03-01"
END_DATE = "2020-03-03"
MAX_FILES = 50
CACHE_DIR = "./cache_gdelt"
FILES_DIR = os.path.join(CACHE_DIR, "files")
os.makedirs(CACHE_DIR, exist_ok=True)
os.makedirs(FILES_DIR, exist_ok=True)

BASE_URL = "http://data.gdeltproject.org/gdeltv2"
MASTER_FILE = os.path.join(CACHE_DIR, "masterfilelist.txt")

# 自定义 User-Agent
user_agent = "Mozilla/5.0 (Macintosh; Intel Mac OS X 14_0) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.0 Safari/605.1.15"
config = Config()
config.browser_user_agent = user_agent
config.request_timeout = 10

# ============================
# masterfilelist 缓存
# ============================
def load_masterfilelist():
    if os.path.exists(MASTER_FILE):
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

# ============================
# 从 masterfilelist 过滤日期范围
# ============================
def get_gkg_file_urls(start, end, max_files):
    lines = load_masterfilelist()
    start_dt = datetime.strptime(start, "%Y-%m-%d")
    end_dt = datetime.strptime(end, "%Y-%m-%d") + timedelta(days=1)
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
        if start_dt <= ts <= end_dt:
            urls.append(url)
        if len(urls) >= max_files:
            break
    print(f"✅ Found {len(urls)} candidate files in range {start} → {end}")
    return urls

# ============================
# 下载并解析 GKG 文件
# ============================
def parse_gdelt_files(urls, company):
    all_rows = []
    print(f"🔍 Filtering for company: {company}")

    for url in urls:
        filename = os.path.basename(url)
        cache_path = os.path.join(FILES_DIR, filename)
        try:
            # 缓存
            if os.path.exists(cache_path):
                print(f"📦 Using cached {filename}")
                with open(cache_path, "rb") as f:
                    data = io.BytesIO(f.read())
            else:
                print(f"⬇️ Downloading {filename} ...")
                r = requests.get(url, timeout=60)
                r.raise_for_status()
                with open(cache_path, "wb") as f:
                    f.write(r.content)
                data = io.BytesIO(r.content)

            # 读取文件
            z = zipfile.ZipFile(data)
            with z.open(z.namelist()[0]) as f:
                df = pd.read_csv(f, sep="\t", header=None, encoding="ISO-8859-1", on_bad_lines="skip")

            print(f"\n📄 {filename} shape={df.shape}")

            # 第26列包含 <PAGE_TITLE> 等
            if 26 not in df.columns:
                print(f"⚠️ Column 26 missing in {filename}")
                continue

            df = df[[26]]
            df.columns = ["Raw"]

            # 提取 PAGE_TITLE 与 URL
            df["Title"] = df["Raw"].str.extract(r"<PAGE_TITLE>(.*?)</PAGE_TITLE>", expand=False)
            df["AltURL"] = df["Raw"].str.extract(r"<PAGE_ALTURL_AMP>(.*?)</PAGE_ALTURL_AMP>", expand=False)
            df["Links"] = df["Raw"].str.extract(r"<PAGE_LINKS>(.*?)</PAGE_LINKS>", expand=False)

            # 优先用 AltURL，没有则用 Links
            df["URL"] = df["AltURL"].combine_first(df["Links"])

            # 清洗 URL 与标题
            df["URL"] = df["URL"].astype(str).str.split(";").str[0]
            df["Title"] = df["Title"].astype(str).str.replace("&amp;", "&")

            # 匹配公司名
            mask = df["Title"].astype(str).str.contains(company, case=False, na=False)
            df = df[mask & df["URL"].astype(str).str.startswith(("http://", "https://"))]

            if df.empty:
                continue

            df["Date"] = filename[:8]
            df = df[["Date", "Title", "URL"]]
            print(f"✅ {filename}: {len(df)} matches (e.g. {df['Title'].iloc[0][:60]})")
            all_rows.append(df)

        except Exception as e:
            print(f"⚠️ Error parsing {url}: {e}")

    if not all_rows:
        return pd.DataFrame()

    combined = pd.concat(all_rows, ignore_index=True)
    combined = combined.drop_duplicates(subset=["URL", "Title"])
    return combined

# ============================
# 并行抓取新闻正文
# ============================
def fetch_article(row, company):
    url = str(row["URL"])
    try:
        art = Article(url, config=config)
        art.download()
        art.parse()
        title = art.title.strip()
        content = art.text.strip().replace("\n", " ")
        if len(content) < 80:
            return None
        return {
            "date": row.get("Date", ""),
            "company": company,
            "title": title,
            "url": url,
            "content": content
        }
    except Exception:
        return None

def crawl_articles(df, company):
    print(f"\n📑 Extracting article content for {len(df)} URLs...\n")
    results = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
        futures = [executor.submit(fetch_article, row, company) for _, row in df.iterrows()]
        for future in concurrent.futures.as_completed(futures):
            res = future.result()
            if res:
                results.append(res)
                print(f"[{len(results)}] ✅ {res['title'][:80]}...")
    return results

# ============================
# 主逻辑
# ============================
if __name__ == "__main__":
    print(f"\n🕒 Running fixed range {START_DATE} → {END_DATE} ...")

    urls = get_gkg_file_urls(START_DATE, END_DATE, MAX_FILES)
    if not urls:
        print("❌ No GKG files found in range.")
        exit(0)

    df = parse_gdelt_files(urls, COMPANY)
    if df.empty:
        print("❌ No matching records found for this company.")
        exit(0)

    news = crawl_articles(df, COMPANY)

    output_file = f"{CACHE_DIR}/news_{COMPANY}_{START_DATE}_{END_DATE}.csv"
    pd.DataFrame(news).to_csv(output_file, index=False, encoding="utf-8")
    print(f"\n✅ Saved {len(news)} articles to {output_file}")

    print("\n========================= SAMPLE NEWS =========================")
    for n in news[:3]:
        print(f"🗞 Title: {n['title']}")
        print(f"📅 Date: {n['date']}")
        print(f"🔗 URL: {n['url']}")
        print(f"📝 Content:\n{n['content'][:800]}")
        print("=" * 90)