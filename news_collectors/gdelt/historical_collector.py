import os
import requests
import zipfile
import io
import pandas as pd
from datetime import datetime, timedelta, timezone
from newspaper import Article, Config
import concurrent.futures
import re
import random
from pymongo import MongoClient, errors
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

# ============================
# 启用 SLM 过滤器
# ============================
os.environ["USE_SLM_FILTER"] = "true"  # 启用 SLM 智能过滤

from special_rules import RuleManager

# ============================
# 全局配置
# ============================
rule_manager = RuleManager()
MONGO_URI = "mongodb://root:root@localhost:37018/"
DB_NAME = "quant_data"
SRC_COLLECTION = "stock_universe"
DST_COLLECTION = "news_articles"  # 正式数据集合

YEARS_BACK = 10  # 10年历史数据
MAX_FILES = None  # 全量
TEST_MODE = False  # 正式模式
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
        except requests.exceptions.HTTPError as e:
            if e.response.status_code == 404:
                print(f"⚠️ File missing on server (404): {url} -> Skipping")
                return None
            wait = backoff * (attempt + 1)
            print(f"⚠️ Download failed {attempt+1}/{retries} for {url}: {e} -> retrying in {wait}s")
            time.sleep(wait)
        except Exception as e:
            wait = backoff * (attempt + 1)
            print(f"⚠️ Download failed {attempt+1}/{retries} for {url}: {e} -> retrying in {wait}s")
            time.sleep(wait)
    print(f"❌ Permanent failure downloading {url}")
    return None


def batch_download_files(urls, batch_size=20):
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

        with ThreadPoolExecutor(max_workers=10) as executor:
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

        print(f"✅ Batch {i//batch_size + 1} complete. Sleeping 0.5s before next batch.\n")
        time.sleep(0.5)

    print("🎯 All downloads finished (cached files skipped automatically).")


# ============================
# 获取时间范围内文件 UR
# ============================
def get_gkg_file_urls(years_back, max_files=None):
    lines = load_masterfilelist()
    urls = []
    
    if years_back is None:
        # 测试模式：获取2025年开始的最早文件
        print("🧪 TEST MODE: Getting earliest GKG files from 2025...")
        for line in lines:
            if ".gkg.csv.zip" not in line:
                continue
            url = line.split()[-1]
            ts_str = os.path.basename(url).split(".")[0]
            
            # 跳过2025年1月1日之前的文件
            try:
                ts = datetime.strptime(ts_str, "%Y%m%d%H%M%S")
                if ts < datetime(2025, 1, 1):
                    continue
            except:
                continue
            
            urls.append(url)
            if max_files and len(urls) >= max_files:
                break
        
        # 2. TEST模式：不再在这里强制硬编码 slice，由外部传入的 max_files 控制
        if TEST_MODE and not max_files:
            urls = urls[:100] 

        if urls:
            # 提取日期范围
            first_file = os.path.basename(urls[0]).split(".")[0]
            last_file = os.path.basename(urls[-1]).split(".")[0]
            try:
                first_date = datetime.strptime(first_file, "%Y%m%d%H%M%S").date()
                last_date = datetime.strptime(last_file, "%Y%m%d%H%M%S").date()
                print(f"✅ Found {len(urls)} GKG files from {first_date} → {last_date}")
            except:
                print(f"✅ Found {len(urls)} GKG files")
    else:
        # 正式模式：按时间范围获取
        end_dt = datetime.utcnow()
        start_dt = end_dt - timedelta(days=years_back * 365)
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


# 全局去重，防止同一个 URL 在一次运行中被多次分析
processed_urls = set()

# ============================
# 解析 GDELT 文件（核心优化版：文件主导循环）
# ============================
def process_batch_files(batch_urls, companies):
    """
    读取一批 GKG 文件，并在内存中匹配所有公司
    返回匹配到的任务列表：[{'row': row, 'company': company_info}, ...]
    """
    matches = []
    
    # 引用全局变量
    global processed_urls
    
    # 1. 并行读取所有 DataFrame（IO 密集型）
    # 但由于 zip 解压也是 CPU 密集的，这里串行处理或小规模并行即可，避免内存爆炸
    # 对于 50 个文件，串行读取通常也很快
    
    print(f"📂 Processing batch of {len(batch_urls)} files against {len(companies)} companies...")
    
    cached_dfs = []
    total_raw_rows = 0
    
    for url in batch_urls:
        filename = os.path.basename(url)
        cache_path = os.path.join(FILES_DIR, filename)
        if not os.path.exists(cache_path):
            continue
            
        try:
            with zipfile.ZipFile(cache_path) as z:
                with z.open(z.namelist()[0]) as f:
                    # 先读取所有列以判断格式
                    df = pd.read_csv(f, sep="\t", header=None, encoding="ISO-8859-1", on_bad_lines="skip")
                    
            if df.empty:
                continue

            total_raw_rows += len(df)
            
            # 判断格式：检查第27列（索引26）是否有数据
            has_xml_column = (26 in df.columns) and (df[26].notna().any())
            
            if has_xml_column:
                # 新格式 (2021+): 使用第27列的XML数据
                df["Raw"] = df[26].astype(str)
                df["Title"] = df["Raw"].str.extract(r"(?s)<PAGE_TITLE>(.*?)</PAGE_TITLE>", expand=False).fillna("")
                df["URL"] = df["Raw"].str.extract(r"(?s)<PAGE_LINKS>(.*?)</PAGE_LINKS>", expand=False).fillna("")
                df["URL"] = df["URL"].astype(str).str.split(";").str[0]
            else:
                # 旧格式 (2016-2020): 第5列（索引4）是URL，没有标题
                # 但我们可以利用第8, 10, 12列 (V2: 23, 27, 29) 的 Themes, Persons, Organizations
                df["URL"] = df[4].astype(str) if 4 in df.columns else ""
                df["Title"] = ""
                
                # 聚合这些列作为搜索文本，提供比单纯 URL 更多的上下文
                context_cols = [22, 26, 28] # V2Themes, V2Persons, V2Organizations
                found_cols = [c for c in context_cols if c in df.columns]
                
                if found_cols:
                    df["Raw"] = df["URL"] + " " + df[found_cols].fillna("").agg(" ".join, axis=1)
                else:
                    df["Raw"] = df["URL"]
            
            df["Date"] = filename[:8]
            # 只保留有效的 HTTP URL
            df = df[df["URL"].str.startswith(("http"), na=False)]
            
            if not df.empty:
                cached_dfs.append(df[["Raw", "Title", "URL", "Date"]])
            
        except Exception as e:
            # 文件损坏或读取失败，跳过
            pass

    if not cached_dfs:
        print(f"⚠️ No data extracted from {len(batch_urls)} files.")
        return []

    try:
        combined_df = pd.concat(cached_dfs, ignore_index=True)
    except ValueError:
        return []

    # 提取这批文件的时间范围
    batch_dates = []
    for url in batch_urls:
        filename = os.path.basename(url)
        date_str = filename[:8]  # YYYYMMDD
        try:
            batch_dates.append(datetime.strptime(date_str, "%Y%m%d"))
        except:
            pass
    
    if batch_dates:
        min_date = min(batch_dates).strftime("%Y-%m-%d")
        max_date = max(batch_dates).strftime("%Y-%m-%d")
        print(f"📅 Batch date range: {min_date} → {max_date}")

    print(f"🔍 Scanning {len(combined_df)} rows... (Raw rows: {total_raw_rows})")
    if not combined_df.empty:
        # 确保 sample 是字符串，防止 NaN (float) 导致报错
        raw_sample = str(combined_df["Raw"].iloc[0])
        print(f"📄 Raw Snippet (Col 26): {raw_sample[:200]}...")

    # 3. Match against all companies (Unified Multi-Company Match)
    # 🚀 优化：一次性构建全量关键词正则，显著降低对 DataFrame 的扫描次数 (O(N) vs O(N*C))
    company_match_counts = {}  
    
    all_keywords_map = {} # {keyword_lower: [symbol1, symbol2, ...]}
    case_sensitive_set = set()
    case_insensitive_set = set()
    
    # 计算平均日期用于动态关键词获取
    avg_date = datetime.utcnow()
    if batch_dates:
        avg_date = min(batch_dates) + (max(batch_dates) - min(batch_dates)) / 2
    
    # 确保 avg_date 是 naive 的，以便与规则库中的 naive 日期比较
    if hasattr(avg_date, 'tzinfo') and avg_date.tzinfo is not None:
        avg_date = avg_date.replace(tzinfo=None)

    for company in companies:
        name = company.get('cleaned_name')
        symbol = company.get('symbol', 'UNKNOWN')
        if not name: continue
        
        keywords = rule_manager.get_keywords(symbol, avg_date)
        if not keywords:
            keywords = [name, symbol]
            
        for k in keywords:
            if not isinstance(k, str) or len(k) <= 1: continue
            k_lower = k.lower()
            if k_lower not in all_keywords_map:
                all_keywords_map[k_lower] = []
            all_keywords_map[k_lower].append(symbol)
            
            if len(k) <= 4 and k.isupper():
                case_sensitive_set.add(k)
            else:
                case_insensitive_set.add(k_lower)

    # 4. 构建合并正则表达式并进行单次筛选
    mask = pd.Series([False] * len(combined_df))
    
    if case_sensitive_set:
        pattern_sensitive_regex = "|".join(r"\b" + re.escape(k) + r"\b" for k in case_sensitive_set)
        pattern_sensitive_simple = "|".join(re.escape(k) for k in case_sensitive_set)
        
        mask |= (combined_df["Title"].str.contains(pattern_sensitive_regex, case=True, na=False, regex=True)) | \
                (combined_df["Raw"].str.contains(pattern_sensitive_regex, case=True, na=False, regex=True)) | \
                (combined_df["URL"].str.contains(pattern_sensitive_simple, case=False, na=False)) # URL match 总是忽略大小写

    if case_insensitive_set:
        # 使用词边界 (\b) 防止正文误命中
        pattern_insensitive_regex = "|".join(r"\b" + re.escape(k) + r"\b" for k in case_insensitive_set)
        pattern_insensitive_simple = "|".join(re.escape(k) for k in case_insensitive_set)
        
        mask |= (combined_df["Title"].str.contains(pattern_insensitive_regex, case=False, na=False, regex=True)) | \
                (combined_df["Raw"].str.contains(pattern_insensitive_regex, case=False, na=False, regex=True)) | \
                (combined_df["URL"].str.contains(pattern_insensitive_simple, case=False, na=False))

    matched_rows = combined_df[mask]
    
    if not matched_rows.empty:
        # 立即去重
        matched_rows = matched_rows.drop_duplicates(subset=["URL"])
        print(f"🎯 Unified matching found {len(matched_rows)} candidate rows. Applying precise rules...")
        
        symbol_to_company = {c['symbol']: c for c in companies if 'symbol' in c}
        
        for _, row in matched_rows.iterrows():
            final_url = row["URL"]
            if not final_url or not final_url.startswith("http") or final_url in processed_urls:
                continue
                
            full_text_lower = f"{(row['Title'] or '')} {(row['Raw'] or '')} {final_url}".lower()
            
            # 找出该行命中的所有公司
            hit_symbols = set()
            for kw, symbols in all_keywords_map.items():
                if kw in full_text_lower:
                    hit_symbols.update(symbols)
            
            # 逐个公司验证 RuleManager
            for symbol in hit_symbols:
                article_data = {
                    "title": row["Title"],
                    "content": row["Raw"] or "", 
                    "date": row.get("Date", "")
                }
                if rule_manager.should_include(symbol, article_data):
                    processed_urls.add(final_url)
                    matches.append({
                        "row": {
                            "URL": final_url,
                            "Title": row["Title"] or f"News about {symbol}",
                            "Date": row.get("Date", "")
                        },
                        "company": symbol_to_company.get(symbol, {"symbol": symbol, "name": symbol})
                    })
                    company_match_counts[symbol] = company_match_counts.get(symbol, 0) + 1
    
    print(f"✅ Fast match complete: {len(matches)} articles identified after filtering.")
    
    # 打印匹配统计
    if company_match_counts:
        print(f"\n📊 Match Summary (Full List):")
        sorted_counts = sorted(company_match_counts.items(), key=lambda x: x[1], reverse=True)
        for symbol, count in sorted_counts:
            print(f"  • {symbol}: {count} articles")
    
    if not matches:
        sample_co = companies[0]
        print(f"ℹ️ No matches in this batch. Example search: '{sample_co.get('cleaned_name')}' across {len(combined_df)} rows.")

    print(f"✅ Found {len(matches)} matches in this batch.")
    return matches


# ============================
# 抓取新闻正文
# ============================
def fetch_article(row, company):
    """
    抓取新闻正文，采用分级保存策略：
    - 优先级1：完整正文 (data_quality: "full")
    - 优先级2：低价值 (data_quality: "low_value") - 有标题没正文
    - 优先级3：仅URL (data_quality: "url_only")
    """
    url = str(row["URL"])
    title = row.get("Title", "")
    date = row.get("Date", "")
    symbol = company["symbol"]
    company_name = company["name"]    
    
    # 基础数据结构
    base_data = {
        "symbol": symbol,
        "name": company_name,
        "date": date,
        "url": url,
        "source": {"platform": "gdelt"},
        "collectedAt": datetime.now(timezone.utc).isoformat(),
        "missing_fields": [],
        "data_quality": "url_only"
    }
    
    try:
        art = Article(url, config=config)
        art.download()
        art.parse()
        
        fetched_title = art.title.strip()
        final_title = fetched_title or title or "No Title"
        
        # 精准识别 GDELT 占位符标题 (如 "News about Adobe")
        is_placeholder = final_title.lower().startswith("news about ")
        
        # 垃圾页面检测 (检测常用的 Bot/Cookie 屏蔽页)
        content = art.text.strip()
        junk_indicators = [
            "necessary cookies", "functional cookies", "analytical cookies",
            "confirm you are a human", "not a bot", "captcha test",
            "page was not found", "404 not found", "access denied",
            "pilihan situs bandar", "togel online", "slot gacor", "toto macau",
            "before you continue to youtube", "before you continue to google"
        ]
        content_lower = content.lower()
        title_lower = final_title.lower()
        if any(indicator in content_lower for indicator in junk_indicators) or \
           any(indicator in title_lower for indicator in ["before you continue to youtube", "before you continue to google"]):
            return None

        
        if len(content) >= 60:
            # 优先级 1: 完整正文 (质量: full)
            test_article_data = {"title": final_title, "content": content, "date": date}
            if not rule_manager.should_include(symbol, test_article_data):
                return None # 被规则引擎拦截的噪音
            
            base_data.update({
                "title": final_title,
                "content": content,
                "data_quality": "full",
                "content_length": len(content)
            })
        elif final_title and final_title != "No Title":
            # 优先级 2: 只有标题
            # 如果是占位符且没正文 -> 视为垃圾，丢弃
            if is_placeholder:
                return None 
            
            # 只有标题没有正文的文章，也需要通过规则检查
            test_article_data = {"title": final_title, "content": "", "date": date}
            if not rule_manager.should_include(symbol, test_article_data):
                return None  # 被规则引擎拦截（如 "Beauty intel" 没有 Intel 关键词）
            
            # ⚠️ 策略调整：只有标题没有正文的文章价值太低，直接丢弃
            # 这样可以避免 "PAGE NOT FOUND"、"Latest News" 等垃圾数据
            return None
        else:
            # 连标题都没有 -> 仅存 URL (url_only)
            base_data.update({"title": "No Title", "data_quality": "url_only"})
            
        return base_data
        
    except Exception as e:
        # 抓取彻底失败时的回退逻辑
        if title:
            is_placeholder = title.lower().startswith("news about ")
            if is_placeholder:
                return None # 抓取失败且标题是占位符，直接放弃
            
            # 即使抓取失败，也要通过规则检查
            test_article_data = {"title": title, "content": "", "date": date}
            if not rule_manager.should_include(symbol, test_article_data):
                return None  # 被规则引擎拦截
            
            # 抓取失败且只有标题，价值太低，直接丢弃
            return None
        return None


# ============================
# 主逻辑
# ============================
if __name__ == "__main__":
    # 自动调整时间范围：测试模式使用最早的2天数据，正式模式收集10年
    if TEST_MODE:
        # 测试模式：获取约 1 天的数据
        actual_years_back = None # 触发最早文件逻辑
        actual_max_files = 100    # 约 1.05 天的数据 (96 files/day)
        mode_str = "TEST (approx 1 day, 2 batches)"
    else:
        actual_years_back = YEARS_BACK
        actual_max_files = MAX_FILES
        mode_str = f"PRODUCTION ({YEARS_BACK} years, all batches)"
    
    print(f"🕒 Running OPTIMIZED GDELT extraction - Mode: {mode_str}\n")

    ensure_index()
    urls = get_gkg_file_urls(actual_years_back, actual_max_files)
    
    # 1. 预下载逻辑改为在批次循环中进行，避免启动时长时间阻塞
    # batch_download_files(urls, batch_size=50) 
    
    # 2. 准备公司数据
    companies = load_companies()
    
    # 预处理公司名，避免重复计算
    valid_companies = []
    for c in companies:
        c['cleaned_name'] = clean_company_name(c.get('name', ''))
        if c['cleaned_name']:
            valid_companies.append(c)
    
    print(f"🏢 Prepared {len(valid_companies)} companies for matching (all stocks).")
    
    # 3. 按批次处理文件 (File-First Loop)
    BATCH_SIZE = 100  # 每次处理100个zip文件
    db = get_db()
    dst_col = db[DST_COLLECTION]
    
    total_inserted = 0
    
    # 断点续传：读取上次处理到的批次
    PROGRESS_FILE = os.path.join(CACHE_DIR, "progress.txt")
    start_batch = 0
    
    if os.path.exists(PROGRESS_FILE):
        try:
            with open(PROGRESS_FILE, "r") as f:
                content = f.read().strip()
                if content:
                    start_batch = int(content)
                    print(f"🔄 Resuming from batch {start_batch + 1}")
        except Exception as e:
            print(f"⚠️ Error reading progress file: {e}. Starting from batch 1.")
    
    if TEST_MODE:
        # 测试模式：强行从头开始
        start_batch = 0
        print(f"🧪 TEST MODE: Starting from batch 1 (progress not saved)")
    
    # 分批遍历 URL
    total_batches = (len(urls) + BATCH_SIZE - 1) // BATCH_SIZE
    
    for i in range(0, len(urls), BATCH_SIZE):
        batch_idx = (i // BATCH_SIZE) + 1
        
        # 跳过已处理的批次
        if batch_idx <= start_batch:
            continue
        
        batch_urls = urls[i : i + BATCH_SIZE]
        print(f"\n🚀 Processing File Batch {batch_idx}/{total_batches} ({i}/{len(urls)})...")
        
        # 实时下载当前批次的文件
        batch_download_files(batch_urls, batch_size=20)
        
        # A. 内存匹配
        matches = process_batch_files(batch_urls, valid_companies)
        
        if not matches:
            continue
            
        # C. 数据库级云端去重：在抓取前先检查库里是否已存在
        all_batch_urls = list(set(m['row']['URL'] for m in matches))
        existing_urls_cursor = dst_col.find({"url": {"$in": all_batch_urls}}, {"url": 1})
        existing_urls = set(doc['url'] for doc in existing_urls_cursor)
        
        filtered_matches = [m for m in matches if m['row']['URL'] not in existing_urls]
        skipped_db = len(matches) - len(filtered_matches)
        
        if skipped_db > 0:
            print(f"⏭️ Database Deduplication: Skipped {skipped_db} articles already in {DST_COLLECTION}")
            
        matches = filtered_matches
        if not matches:
            # 如果本批次所有 URL 都已在库中，更新进度并跳过
            if not TEST_MODE:
                with open(PROGRESS_FILE, "w") as f:
                    f.write(str(batch_idx))
            continue

        # D. 按公司分组爬取全文
        print(f"\n📰 Crawling {len(matches)} articles found in batch...")
        
        # 按公司分组
        company_groups = {}
        for m in matches:
            symbol = m['company'].get('symbol', 'UNKNOWN')
            if symbol not in company_groups:
                company_groups[symbol] = {
                    'company': m['company'],
                    'articles': []
                }
            company_groups[symbol]['articles'].append(m['row'])
        
        print(f"📊 Articles grouped by {len(company_groups)} companies\n")
        
        # 逐个公司处理
        results = []
        for idx, (symbol, group) in enumerate(company_groups.items(), 1):
            company = group['company']
            articles = group['articles']
            company_name = company.get('name', symbol)
            
            print(f"  [{idx}/{len(company_groups)}] 🔄 Crawling {len(articles)} articles for {symbol} ({company_name})...")
            
            # 并行抓取该公司的所有文章 (降低线程数以防止 double free or corruption)
            company_results = []
            with ThreadPoolExecutor(max_workers=10) as executor:
                futures = [executor.submit(fetch_article, row, company) for row in articles]
                
                completed = 0
                for future in as_completed(futures):
                    try:
                        res = future.result()
                        if res:
                            company_results.append(res)
                            # 调试：打印前3个结果
                            if len(company_results) <= 3:
                                print(f"    🐛 DEBUG: Got result with quality={res.get('data_quality')}, title={res.get('title', '')[:50]}")
                        completed += 1
                        
                        # 每10篇打印一次进度
                        if completed % 10 == 0 or completed == len(articles):
                            print(f"    ⏳ {completed}/{len(articles)} articles processed...")
                    except Exception as e:
                        if completed < 3:
                            print(f"    🐛 DEBUG: Exception in future: {type(e).__name__}: {str(e)[:100]}")
                        completed += 1
            
            results.extend(company_results)
            print(f"    ✅ {len(company_results)}/{len(articles)} articles successfully crawled\n")

        # D. 入库
        if results:
            inserted_count = 0
            company_insert_counts = {}  # 统计每个公司实际插入的数量
            quality_stats = {"full": 0, "low_value": 0, "url_only": 0}
            
            for art in results:
                try:
                    quality = art.get("data_quality", "unknown")
                    quality_stats[quality] = quality_stats.get(quality, 0) + 1

                    result = dst_col.update_one(
                        {"url": art["url"]},
                        {"$setOnInsert": art},
                        upsert=True
                    )
                    if result.upserted_id or result.modified_count > 0:
                        inserted_count += 1
                        symbol = art.get("symbol", "UNKNOWN")
                        company_insert_counts[symbol] = company_insert_counts.get(symbol, 0) + 1
                except Exception:
                    pass
            
            total_inserted += inserted_count
            print(f"\n💾 Batch saved: {inserted_count}/{len(results)} new articles (Total: {total_inserted})")
            
            print(f"📊 Data Quality Distribution:")
            total_res = len(results)
            print(f"  • Full (Content matched): {quality_stats.get('full', 0)} ({quality_stats.get('full', 0)/total_res*100:.1f}%)")
            print(f"  • Low Value (Title only): {quality_stats.get('low_value', 0)} ({quality_stats.get('low_value', 0)/total_res*100:.1f}%)")
            print(f"  • URL Only/Unknown: {quality_stats.get('url_only', 0)} ({quality_stats.get('url_only', 0)/total_res*100:.1f}%)")
            # 打印每个公司的插入统计
            if company_insert_counts:
                print(f"\n� Inserted by company:")
                sorted_inserts = sorted(company_insert_counts.items(), key=lambda x: x[1], reverse=True)
                for symbol, count in sorted_inserts:
                    print(f"  • {symbol}: {count} articles")
        
        # 保存进度（每批完成后）- 仅在正式模式下
        if not TEST_MODE:
            with open(PROGRESS_FILE, "w") as f:
                f.write(str(batch_idx))
        
        # 测试模式：处理完所有请求的文件后停止（不保存进度）
        if TEST_MODE and batch_idx * BATCH_SIZE >= len(urls):
            print(f"\n🧪 TEST MODE complete. Results saved in '{DST_COLLECTION}'.")
            break
        
        # 稍微休息释放内存
        import gc
        gc.collect()

    print(f"\n🏁 All done! Processed {len(urls)} files. Total articles inserted: {total_inserted}")