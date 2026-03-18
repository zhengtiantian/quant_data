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
import threading
import mysql.connector
import builtins
import signal
import traceback

# ============================
# 启用 SLM 过滤器
# ============================
os.environ["USE_SLM_FILTER"] = "true"  # 启用 SLM 智能过滤
# 默认关闭规则细粒度日志，避免刷屏淹没进度
RULE_EVENT_LOGS = os.getenv("RULE_EVENT_LOGS", "true").lower() == "true"
if RULE_EVENT_LOGS:
    os.environ["RULE_VERBOSE"] = "true"
    os.environ["SLM_LOG_INTERCEPTIONS"] = "true"
else:
    os.environ["RULE_VERBOSE"] = "false"
    os.environ["SLM_LOG_INTERCEPTIONS"] = "false"

from special_rules import RuleManager

# 统一日志前缀：所有 print 自动附带线程名
_ORIGINAL_PRINT = builtins.print

def _thread_print(*args, **kwargs):
    if args and isinstance(args[0], str) and args[0].startswith("["):
        _ORIGINAL_PRINT(*args, **kwargs)
        return
    thread_name = threading.current_thread().name
    _ORIGINAL_PRINT(f"[{thread_name}]", *args, **kwargs)

builtins.print = _thread_print

# ============================
# 全局配置
# ============================
rule_manager = RuleManager()
MONGO_URI = os.getenv("MONGO_URI", "mongodb://root:root@127.0.0.1:37018/")
DB_NAME = "quant_data"
SRC_COLLECTION = "stock_universe"
DST_COLLECTION = "news_articles"  # 正式数据集合

YEARS_BACK = 10  # 10年历史数据
MAX_FILES = None  # 全量
TEST_MODE = False  # 正式模式
CACHE_DIR = os.getenv("GDELT_CACHE_DIR", "/Volumes/data24T/docker-volumes/gdelt_cache")
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
ARTICLE_REQUEST_TIMEOUT = int(os.getenv("ARTICLE_REQUEST_TIMEOUT", "6"))
config.request_timeout = ARTICLE_REQUEST_TIMEOUT

# 并行抓取阶段总超时（秒）：防止少数 parse 卡死拖住整批
FETCH_BATCH_TIMEOUT = int(os.getenv("FETCH_BATCH_TIMEOUT", "900"))
FETCH_WORKERS = int(os.getenv("FETCH_WORKERS", "3"))
FETCH_TASK_TIMEOUT = int(os.getenv("FETCH_TASK_TIMEOUT", "45"))
SCAN_WORKERS = int(os.getenv("SCAN_WORKERS", "2"))
SCAN_PROGRESS_EVERY_FILES = int(os.getenv("SCAN_PROGRESS_EVERY_FILES", "100"))
SCAN_RULE_PROGRESS_EVERY_RECORDS = int(os.getenv("SCAN_RULE_PROGRESS_EVERY_RECORDS", "500"))
STUCK_URLS_FILE = os.path.join(CACHE_DIR, "stuck_urls.txt")

# 调试开关：只控制日志，不影响业务逻辑
DEBUG_TASK_TRACE = os.getenv("DEBUG_TASK_TRACE", "false").lower() == "true"
USE_MYSQL_BATCH_QUEUE = os.getenv("USE_MYSQL_BATCH_QUEUE", "false").lower() == "true"
BATCH_WORKERS = int(os.getenv("BATCH_WORKERS", "3"))
RUNNING_RECLAIM_MINUTES = int(os.getenv("RUNNING_RECLAIM_MINUTES", "720"))
RESET_ALL_RUNNING_ON_START = os.getenv("RESET_ALL_RUNNING_ON_START", "true").lower() == "true"
STARTUP_PREDOWNLOAD_ENABLED = os.getenv("STARTUP_PREDOWNLOAD_ENABLED", "true").lower() == "true"
STARTUP_PREDOWNLOAD_RECENT_FILES = int(os.getenv("STARTUP_PREDOWNLOAD_RECENT_FILES", "288"))
HOST_ID = os.getenv("HOST_ID", "mac")
QUEUE_INSTANCE_ID = os.getenv("QUEUE_INSTANCE_ID", HOST_ID)
QUEUE_HEARTBEAT_SECONDS = int(os.getenv("QUEUE_HEARTBEAT_SECONDS", "30"))

MYSQL_HOST = os.getenv("MYSQL_HOST", "127.0.0.1")
MYSQL_PORT = int(os.getenv("MYSQL_PORT", "23306"))
MYSQL_USER = os.getenv("MYSQL_USER", "root")
MYSQL_PASSWORD = os.getenv("MYSQL_PASSWORD", "root")
MYSQL_DATABASE = os.getenv("MYSQL_DATABASE", "workflow")
MYSQL_TASK_TABLE = os.getenv("MYSQL_TASK_TABLE", "gdelt_batch_tasks")
SHUTDOWN_EVENT = threading.Event()
SHUTDOWN_REASON = {"reason": None}


def _mark_shutdown(reason):
    if not SHUTDOWN_EVENT.is_set():
        SHUTDOWN_REASON["reason"] = reason
        SHUTDOWN_EVENT.set()
        print(f"⚠️ Shutdown requested: {reason}")


def _signal_handler(sig, _frame):
    _mark_shutdown(f"signal={sig}")


signal.signal(signal.SIGINT, _signal_handler)
signal.signal(signal.SIGTERM, _signal_handler)


def load_stuck_urls():
    """加载历史超时 URL 黑名单，避免重复卡死。"""
    if not os.path.exists(STUCK_URLS_FILE):
        return set()
    try:
        with open(STUCK_URLS_FILE, "r", encoding="utf-8") as f:
            return {line.strip() for line in f if line.strip()}
    except Exception:
        return set()


def append_stuck_urls(urls):
    """追加记录本批卡住 URL。"""
    if not urls:
        return
    try:
        with open(STUCK_URLS_FILE, "a", encoding="utf-8") as f:
            for u in sorted(set(urls)):
                f.write(u + "\n")
    except Exception:
        pass


def get_mysql_conn():
    return mysql.connector.connect(
        host=MYSQL_HOST,
        port=MYSQL_PORT,
        user=MYSQL_USER,
        password=MYSQL_PASSWORD,
        database=MYSQL_DATABASE,
        autocommit=False,
    )


def ensure_task_table():
    conn = get_mysql_conn()
    try:
        cur = conn.cursor()
        cur.execute(
            f"""
            CREATE TABLE IF NOT EXISTS {MYSQL_TASK_TABLE} (
              batch_id INT PRIMARY KEY,
              status ENUM('pending','running','done','failed') NOT NULL DEFAULT 'pending',
              owner VARCHAR(64) DEFAULT NULL,
              owner_host VARCHAR(128) DEFAULT NULL,
              retries INT NOT NULL DEFAULT 0,
              last_error TEXT,
              started_at DATETIME NULL,
              finished_at DATETIME NULL,
              updated_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
              INDEX idx_status_batch (status, batch_id)
            )
            """
        )
        cur.execute(f"SHOW COLUMNS FROM {MYSQL_TASK_TABLE} LIKE 'owner_host'")
        if cur.fetchone() is None:
            cur.execute(f"ALTER TABLE {MYSQL_TASK_TABLE} ADD COLUMN owner_host VARCHAR(128) DEFAULT NULL AFTER owner")
        conn.commit()
    finally:
        conn.close()


def seed_tasks(total_batches, resume_batch):
    conn = get_mysql_conn()
    try:
        cur = conn.cursor()
        # 确保所有批次都在任务表里
        values = [(i,) for i in range(1, total_batches + 1)]
        if values:
            cur.executemany(
                f"INSERT IGNORE INTO {MYSQL_TASK_TABLE} (batch_id, status) VALUES (%s, 'pending')",
                values,
            )
        # 进度语义：progress=N -> 1..N-1 已完成，N..end 待执行
        if resume_batch > 1:
            cur.execute(
                f"""
                UPDATE {MYSQL_TASK_TABLE}
                SET status='done', owner=NULL, finished_at=IFNULL(finished_at, NOW()), updated_at=NOW()
                WHERE batch_id < %s AND status <> 'done'
                """,
                (resume_batch,),
            )
        if RESET_ALL_RUNNING_ON_START:
            # 仅回收当前主机遗留的 running，避免多主机互相打断
            cur.execute(
                f"""
                UPDATE {MYSQL_TASK_TABLE}
                SET status='pending', owner=NULL, owner_host=NULL, updated_at=NOW()
                WHERE status='running'
                  AND owner_host=%s
                """
                ,
                (HOST_ID,),
            )
        else:
            cur.execute(
                f"""
                UPDATE {MYSQL_TASK_TABLE}
                SET status='pending', owner=NULL, owner_host=NULL, updated_at=NOW()
                WHERE status='running'
                  AND updated_at < (NOW() - INTERVAL %s MINUTE)
                """,
                (RUNNING_RECLAIM_MINUTES,),
            )
        conn.commit()
    finally:
        conn.close()


def claim_next_batch(owner):
    conn = get_mysql_conn()
    try:
        cur = conn.cursor(dictionary=True)
        cur.execute("START TRANSACTION")
        cur.execute(
            f"""
            SELECT batch_id
            FROM {MYSQL_TASK_TABLE}
            WHERE status IN ('pending', 'failed')
               OR (status='running' AND updated_at < (NOW() - INTERVAL %s MINUTE))
            ORDER BY batch_id ASC
            LIMIT 1
            FOR UPDATE SKIP LOCKED
            """,
            (RUNNING_RECLAIM_MINUTES,),
        )
        row = cur.fetchone()
        if not row:
            conn.commit()
            return None
        batch_id = int(row["batch_id"])
        cur.execute(
            f"""
            UPDATE {MYSQL_TASK_TABLE}
            SET status='running', owner=%s, owner_host=%s, started_at=NOW(), updated_at=NOW(), last_error=NULL
            WHERE batch_id=%s
            """,
            (owner, HOST_ID, batch_id),
        )
        conn.commit()
        return batch_id
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def mark_batch_done(batch_id):
    conn = get_mysql_conn()
    try:
        cur = conn.cursor()
        cur.execute(
            f"""
            UPDATE {MYSQL_TASK_TABLE}
            SET status='done', owner=NULL, owner_host=NULL, finished_at=NOW(), updated_at=NOW(), last_error=NULL
            WHERE batch_id=%s
            """,
            (batch_id,),
        )
        conn.commit()
    finally:
        conn.close()


def mark_batch_failed(batch_id, err_msg):
    conn = get_mysql_conn()
    try:
        cur = conn.cursor()
        cur.execute(
            f"""
            UPDATE {MYSQL_TASK_TABLE}
            SET retries=retries+1,
                status=CASE WHEN retries+1 >= 3 THEN 'failed' ELSE 'pending' END,
                owner=NULL,
                owner_host=NULL,
                updated_at=NOW(),
                last_error=%s
            WHERE batch_id=%s
            """,
            (str(err_msg)[:1000], batch_id),
        )
        conn.commit()
    finally:
        conn.close()


def requeue_batch(batch_id, reason):
    """把当前批次放回 pending，避免异常退出时任务丢失。"""
    conn = get_mysql_conn()
    try:
        cur = conn.cursor()
        cur.execute(
            f"""
            UPDATE {MYSQL_TASK_TABLE}
            SET status='pending',
                owner=NULL,
                owner_host=NULL,
                updated_at=NOW(),
                last_error=%s
            WHERE batch_id=%s
            """,
            (str(reason)[:1000], batch_id),
        )
        conn.commit()
    finally:
        conn.close()


def heartbeat_batch(batch_id, owner):
    conn = get_mysql_conn()
    try:
        cur = conn.cursor()
        cur.execute(
            f"""
            UPDATE {MYSQL_TASK_TABLE}
            SET updated_at=NOW()
            WHERE batch_id=%s
              AND status='running'
              AND owner=%s
              AND owner_host=%s
            """,
            (batch_id, owner, HOST_ID),
        )
        conn.commit()
    finally:
        conn.close()


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
# 时间戳解析工具
# ============================
def parse_gdelt_timestamp(filename_or_timestamp):
    """
    解析 GDELT 时间戳
    输入: 文件名 (如 '20210125184500.gkg.csv.zip') 或时间戳字符串 (如 '20210125184500')
    输出: {
        'timestamp_str': '20210125184500',  # 原始字符串
        'date_str': '20210125',              # 日期字符串 (YYYYMMDD)
        'datetime': datetime(2021, 1, 25, 18, 45, 0),  # Python datetime 对象
        'iso': '2021-01-25T18:45:00Z'        # ISO 8601 格式
    }
    """
    # 从文件名中提取时间戳
    if '.' in filename_or_timestamp:
        timestamp_str = os.path.basename(filename_or_timestamp).split('.')[0]
    else:
        timestamp_str = filename_or_timestamp
    
    # 处理不同长度的时间戳
    if len(timestamp_str) >= 14:
        # 完整时间戳: YYYYMMDDHHMMSS
        dt = datetime.strptime(timestamp_str[:14], "%Y%m%d%H%M%S")
        date_str = timestamp_str[:8]
    elif len(timestamp_str) >= 8:
        # 仅日期: YYYYMMDD
        dt = datetime.strptime(timestamp_str[:8], "%Y%m%d")
        date_str = timestamp_str[:8]
        timestamp_str = date_str + "000000"  # 补充为完整格式
    else:
        # 无效格式，返回 None
        return None
    
    return {
        'timestamp_str': timestamp_str[:14],
        'date_str': date_str,
        'datetime': dt,
        'iso': dt.strftime("%Y-%m-%dT%H:%M:%SZ")
    }

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


def batch_download_files(urls, batch_size=20, worker_name=None):
    """并行批量下载 GDELT 文件，已缓存文件自动跳过"""
    to_download = [
        url for url in urls
        if not os.path.exists(os.path.join(FILES_DIR, os.path.basename(url)))
    ]
    skipped = len(urls) - len(to_download)
    print(f"📂 Cached files skipped: {skipped}")
    print(f"⬇️ Need to download: {len(to_download)} files\n")

    for i in range(0, len(to_download), 60):
        if SHUTDOWN_EVENT.is_set():
            print("⚠️ Skip download batch because shutdown is in progress")
            return
        batch = to_download[i:i + 60]
        print(f"🚀 Batch {i//batch_size + 1}: downloading {len(batch)} files...")

        prefix = f"{worker_name}-download" if worker_name else f"{threading.current_thread().name}-download"
        with ThreadPoolExecutor(max_workers=10, thread_name_prefix=prefix) as executor:
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


def predownload_recent_missing_files(urls):
    """启动时优先预下载最新一段缺失的 GKG 文件，避免新一天的数据在首批任务里集中阻塞。"""
    if not STARTUP_PREDOWNLOAD_ENABLED or not urls:
        return
    lookback = max(1, STARTUP_PREDOWNLOAD_RECENT_FILES)
    recent_urls = urls[-lookback:]
    missing = [
        url for url in recent_urls
        if not os.path.exists(os.path.join(FILES_DIR, os.path.basename(url)))
    ]
    if not missing:
        print(f"📦 Startup predownload: latest {len(recent_urls)} files already cached")
        return
    print(
        f"📦 Startup predownload: caching {len(missing)}/{len(recent_urls)} recent files "
        f"(window={lookback}) before workers start"
    )
    batch_download_files(recent_urls, batch_size=50, worker_name="startup-cache")


# ============================
# 获取时间范围内文件 UR
# ============================
def get_gkg_file_urls(years_back, max_files=None):
    lines = load_masterfilelist()
    urls = []
    
    # 强制从 2016年1月1日 开始收集
    start_dt = datetime(2016, 1, 1)
    end_dt = datetime.utcnow()
    
    print(f"📅 Collecting GKG files from 2016-01-01 to {end_dt.date()}...")
    
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


def strip_urls_and_xml(text):
    """移除 XML 标签、PAGE_LINKS 和裸 URL，避免平台域名误触发关键词匹配。"""
    if not isinstance(text, str) or not text:
        return ""
    text = re.sub(r"(?is)<PAGE_LINKS>.*?</PAGE_LINKS>", " ", text)
    text = re.sub(r"(?is)<[^>]+>", " ", text)
    text = re.sub(r"https?://\S+", " ", text)
    text = re.sub(r"\b(?:www\.)?[a-z0-9.-]+\.[a-z]{2,}(?:/\S*)?\b", " ", text, flags=re.I)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


# 全局去重，防止同一个 URL 在一次运行中被多次分析
processed_urls = set()

# ============================
# 解析 GDELT 文件（核心优化版：文件主导循环）
# ============================
def process_single_file(url, companies, rule_manager, avg_date):
    """处理单个 GKG 文件的协程/线程函数"""
    filename = os.path.basename(url)
    cache_path = os.path.join(FILES_DIR, filename)
    if not os.path.exists(cache_path):
        return []

    matches = []
    try:
        with zipfile.ZipFile(cache_path) as z:
            with z.open(z.namelist()[0]) as f:
                df = pd.read_csv(f, sep="\t", header=None, encoding="ISO-8859-1", on_bad_lines="skip")
        
        if df.empty:
            return []

        # 判断并提取格式
        has_xml_column = (26 in df.columns) and (df[26].notna().any())
        if has_xml_column:
            df["URL"] = df[26].astype(str).str.extract(r"(?s)<PAGE_LINKS>(.*?)</PAGE_LINKS>", expand=False).fillna("").str.split(";").str[0]
            df["Title"] = df[26].astype(str).str.extract(r"(?s)<PAGE_TITLE>(.*?)</PAGE_TITLE>", expand=False).fillna("")
            df["Raw"] = df[26].astype(str).map(strip_urls_and_xml)
        else:
            df["URL"] = df[4].astype(str) if 4 in df.columns else ""
            df["Title"] = ""
            context_cols = [22, 26, 28] if any(c in df.columns for c in [22, 26, 28]) else []
            raw_text = df[context_cols].fillna("").agg(" ".join, axis=1) if context_cols else ""
            df["Raw"] = raw_text.map(strip_urls_and_xml) if hasattr(raw_text, "map") else ""

        df["Date"] = filename[:8]
        df = df[df["URL"].str.startswith(("http"), na=False)]
        
        if df.empty:
            return []

        # 预构建关键词库
        for company in companies:
            symbol = company['symbol']
            keywords = rule_manager.get_keywords(symbol, avg_date)
            # 简单匹配：只要正文含有关键词
            pattern = "|".join(r"\b" + re.escape(str(k)) + r"\b" for k in keywords if len(str(k)) > 1)
            
            # 使用更快的搜索方式
            mask = df["Title"].str.contains(pattern, case=False, na=False, regex=True) | \
                   df["Raw"].str.contains(pattern, case=False, na=False, regex=True)
            
            matched_df = df[mask]
            for _, row in matched_df.iterrows():
                final_url = str(row["URL"])
                # 再次校验
                article_data = {"title": row["Title"], "content": row["Raw"], "date": row["Date"]}
                if rule_manager.should_include(symbol, article_data):
                    matches.append({
                        "row": {"URL": final_url, "Title": row["Title"] or f"News about {symbol}", "Date": row["Date"]},
                        "company": company
                    })
    except:
        pass
    return matches

def process_batch_files(batch_urls, companies, worker_name=None):
    """并行处理一批 GKG 文件并高效匹配所有公司"""
    import numpy as np
    
    # 1. 计算平均日期用于获取动态关键词
    batch_dates = []
    for url in batch_urls:
        try: batch_dates.append(datetime.strptime(os.path.basename(url)[:8], "%Y%m%d"))
        except: pass
    avg_date = datetime.now(timezone.utc).replace(tzinfo=None)
    if batch_dates:
        avg_date = min(batch_dates) + (max(batch_dates) - min(batch_dates)) / 2
    avg_date = avg_date.replace(tzinfo=None)

    # 2. 预构建全局关键词映射表（超级正则）
    all_keywords_map = {}
    case_insensitive_keywords = []
    
    for company in companies:
        symbol = company['symbol']
        keywords = rule_manager.get_keywords(symbol, avg_date)
        if not keywords:
            keywords = [company.get('cleaned_name', symbol)]
        
        for k in keywords:
            if not isinstance(k, str) or len(k) <= 1: continue
            k_lower = k.lower()
            if k_lower not in all_keywords_map:
                all_keywords_map[k_lower] = []
            all_keywords_map[k_lower].append(symbol)
            case_insensitive_keywords.append(re.escape(k_lower))

    # 构建统一匹配正则（移除 \b 以兼容带点的公司名，改为之后精确校验）
    # 使用简单包含初筛，后续用 RuleManager 精筛
    global_pattern = "|".join(case_insensitive_keywords)
    if not global_pattern:
        return []

    if SHUTDOWN_EVENT.is_set():
        print("⚠️ Skip scanning because shutdown is in progress")
        return []
    print(f"📂 Scanning {len(batch_urls)} files in parallel (Unified Regex)...")
    
    all_combined_matches = []
    total_files = len(batch_urls)
    progress_lock = threading.Lock()
    progress = {
        "files_done": 0,
        "rows_scanned": 0,
        "candidates": 0,
        "accepted": 0,
        "rules_total_discovered": 0,
        "rules_processed": 0,
    }
    symbol_to_company = {c['symbol']: c for c in companies}

    def process_file_task(url):
        filename = os.path.basename(url)
        cache_path = os.path.join(FILES_DIR, filename)
        if not os.path.exists(cache_path):
            return {
                "matches": [],
                "rows": 0,
                "candidates": 0,
                "sample": "",
            }

        try:
            t_start = time.time()
            with zipfile.ZipFile(cache_path) as z:
                with z.open(z.namelist()[0]) as f:
                    df = pd.read_csv(f, sep="\t", header=None, encoding="ISO-8859-1", on_bad_lines="skip")
            t_parse = time.time()

            if df.empty:
                return {
                    "matches": [],
                    "rows": 0,
                    "candidates": 0,
                    "sample": "",
                }

            # 统一列名并增强字段覆盖
            if (26 in df.columns) and (df[26].notna().any()):
                # V2 XML 格式 (2021+)
                df["URL"] = df[26].astype(str).str.extract(r"(?s)<PAGE_LINKS>(.*?)</PAGE_LINKS>", expand=False).fillna("").str.split(";").str[0]
                df["Title"] = df[26].astype(str).str.extract(r"(?s)<PAGE_TITLE>(.*?)</PAGE_TITLE>", expand=False).fillna("")
                df["Raw"] = df[26].astype(str).map(strip_urls_and_xml)
            else:
                # 传统格式 (2016-2020): 同时扫描 V1 和 V2 字段
                df["URL"] = df[4].astype(str) if 4 in df.columns else ""
                df["Title"] = ""
                # GDELT GKG 字段索引: 7,9,11 是 V1 实体; 22,26,28 是 V2 实体
                entity_cols = [7, 9, 11, 22, 26, 28]
                found_cols = [c for c in entity_cols if c in df.columns]
                raw_text = df[found_cols].fillna("").agg(" ".join, axis=1) if found_cols else ""
                df["Raw"] = raw_text.map(strip_urls_and_xml) if hasattr(raw_text, "map") else ""

            df = df[df["URL"].str.startswith("http", na=False)]
            # 过滤社交媒体主页链接（不是新闻）
            social_domains = r"(?:instagram\.com|facebook\.com|twitter\.com|x\.com|tiktok\.com|linkedin\.com)/(?!.*\b(?:news|blog|article|press)\b)"
            df = df[~df["URL"].str.contains(social_domains, case=False, na=False, regex=True)]
            if df.empty:
                return {
                    "matches": [],
                    "rows": 0,
                    "candidates": 0,
                    "sample": "",
                }

            # 核心匹配逻辑：大小写不敏感初筛（仅 Title + Raw 实体字段，不含 URL）
            combined_text = (df["Title"].fillna("") + " " + df["Raw"].fillna("")).str.lower()
            t_combine = time.time()
            mask = combined_text.str.contains(global_pattern, case=False, na=False, regex=True)

            matched_df = df[mask]
            t_regex = time.time()
            file_matches = []
            sample_text = ""
            if not matched_df.empty:
                srow = matched_df.iloc[0]
                sample_text = f"title='{str(srow.get('Title', ''))[:50]}' url='{str(srow.get('URL', ''))[:80]}'"

            # 记录已发现的候选记录总数（分母会逐步增长，最终收敛）
            with progress_lock:
                progress["rules_total_discovered"] += len(matched_df)

            for idx, row in matched_df.iterrows():
                row_text = combined_text.loc[idx]
                # 找出究竟是哪个 symbol 被命中
                hit_symbols = set()
                for kw, symbols in all_keywords_map.items():
                    if kw in row_text:
                        hit_symbols.update(symbols)
                
                for sym in hit_symbols:
                    # 解析完整时间戳
                    time_info = parse_gdelt_timestamp(filename)
                    article_data = {
                        "title": row["Title"],
                        "content": row["Raw"],
                        "date": time_info['date_str'] if time_info else filename[:8],
                        "source_file": filename,
                        "url": row["URL"],
                    }
                    if rule_manager.should_include(sym, article_data):
                        file_matches.append({
                            "row": {
                                "URL": row["URL"], 
                                "Title": row["Title"] or f"News about {sym}", 
                                "Date": time_info['date_str'] if time_info else filename[:8],
                                "Timestamp": time_info['timestamp_str'] if time_info else filename[:14],
                                "PublishedAt": time_info['iso'] if time_info else None
                            },
                            "company": symbol_to_company.get(sym)
                        })

                # 按“候选记录”粒度输出规则进度（每100条）
                with progress_lock:
                    progress["rules_processed"] += 1
                    rp = progress["rules_processed"]
                    rt = progress["rules_total_discovered"]
                    if rp % SCAN_RULE_PROGRESS_EVERY_RECORDS == 0:
                        worker = worker_name or threading.current_thread().name
                        print(f"[{worker}] 🧪 Rule check progress: {rp} records processed")
            t_rules = time.time()
            total_time = t_rules - t_start
            # 慢文件 (>2s) 打印详细耗时分解
            if total_time > 2.0:
                print(
                    f"🐢 Slow file {filename}: {total_time:.1f}s total "
                    f"(parse={t_parse-t_start:.2f}s, combine={t_combine-t_parse:.2f}s, "
                    f"regex={t_regex-t_combine:.2f}s, rules={t_rules-t_regex:.2f}s) "
                    f"rows={len(df)}, candidates={len(matched_df)}, accepted={len(file_matches)}"
                )
            return {
                "matches": file_matches,
                "rows": len(df),
                "candidates": len(matched_df),
                "sample": sample_text,
            }
        except Exception as e:
            print(f"❌ Error processing {filename}: {e}")
            return {
                "matches": [],
                "rows": 0,
                "candidates": 0,
                "sample": "",
                "error": str(e),
            }

    prefix = f"{worker_name}-scan" if worker_name else f"{threading.current_thread().name}-scan"
    scan_start_time = time.time()
    # 每10个文件打印一次进度，方便排查卡住
    SCAN_FILE_PROGRESS_INTERVAL = int(os.getenv("SCAN_FILE_PROGRESS_INTERVAL", "10"))
    with ThreadPoolExecutor(max_workers=SCAN_WORKERS, thread_name_prefix=prefix) as executor:
        futures = [executor.submit(process_file_task, url) for url in batch_urls]
        for future in as_completed(futures):
            res = future.result()
            all_combined_matches.extend(res["matches"])
            with progress_lock:
                progress["files_done"] += 1
                progress["rows_scanned"] += res["rows"]
                progress["candidates"] += res["candidates"]
                progress["accepted"] += len(res["matches"])
                done = progress["files_done"]
                worker = worker_name or threading.current_thread().name
                if done == 1 or done % SCAN_FILE_PROGRESS_INTERVAL == 0 or done == total_files:
                    elapsed = time.time() - scan_start_time
                    rate = done / elapsed if elapsed > 0 else 0
                    print(
                        f"[{worker}] 🔍 Scan progress: {done}/{total_files} files "
                        f"({elapsed:.1f}s, {rate:.1f} files/s), "
                        f"rows={progress['rows_scanned']}, candidates={progress['candidates']}, accepted={progress['accepted']}"
                    )

    # 去重
    seen = set()
    final_output = []
    for m in all_combined_matches:
        key = (m['row']['URL'], m['company']['symbol'])
        if key not in seen:
            seen.add(key)
            final_output.append(m)

    print(f"✅ Parallel scan complete: {len(final_output)} matched tasks identified.")
    if progress["rules_total_discovered"] > 0:
        print(
            f"🧪 Rule check done: {progress['rules_processed']}/{progress['rules_total_discovered']} records"
        )

    # 输出每步过滤汇总
    from special_rules.ambiguous_names import print_filter_summary
    w = worker_name or threading.current_thread().name
    print_filter_summary(w)

    return final_output



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
    date = row.get("Date", "")  # YYYYMMDD 格式
    timestamp = row.get("Timestamp", "")  # YYYYMMDDHHMMSS 格式
    published_at = row.get("PublishedAt", None)  # ISO 8601 格式
    symbol = company["symbol"]
    company_name = company["name"]
    task_id = f"{symbol}|{url}"
    t0 = time.time()
    if DEBUG_TASK_TRACE:
        print(f"🧭 TASK START {task_id}")
    
    # 基础数据结构
    base_data = {
        "symbol": symbol,
        "name": company_name,
        "date": date,  # YYYYMMDD (保留用于兼容)
        "timestamp": timestamp,  # YYYYMMDDHHMMSS (完整时间戳)
        "publishedAt": published_at,  # ISO 8601 格式 (便于查询)
        "url": url,
        "source": {"platform": "gdelt"},
        "collectedAt": datetime.now(timezone.utc).isoformat(),
        "missing_fields": [],
        "data_quality": "url_only"
    }
    
    try:
        if DEBUG_TASK_TRACE:
            t_dl = time.time()
            print(f"🧭 TASK {task_id} -> download start")
        art = Article(url, config=config)
        art.download()
        if DEBUG_TASK_TRACE:
            print(f"🧭 TASK {task_id} -> download done {time.time()-t_dl:.2f}s")
        if DEBUG_TASK_TRACE:
            t_ps = time.time()
            print(f"🧭 TASK {task_id} -> parse start")
        art.parse()
        if DEBUG_TASK_TRACE:
            print(f"🧭 TASK {task_id} -> parse done {time.time()-t_ps:.2f}s")
        
        # 尝试从文章中提取真实发布时间
        if art.publish_date:
            try:
                # newspaper3k 提取的发布时间（datetime 对象）
                article_publish_date = art.publish_date.strftime("%Y%m%d%H%M%S")
                # 确保是 14 位且时分秒不是 000000
                if len(article_publish_date) == 14 and article_publish_date.isdigit():
                    # 检查时分秒是否为 000000（只有日期没有时间）
                    time_part = article_publish_date[8:]  # 提取 HHMMSS
                    if time_part != "000000":
                        # 只有当有具体时间时才更新
                        base_data["date"] = article_publish_date
                        base_data["timestamp"] = article_publish_date
                        base_data["publishedAt"] = art.publish_date.isoformat()
                    # 否则保留 GDELT 文件名的精确时间戳
            except:
                pass  # 转换失败则保持使用 GDELT 时间（已经是14位）
        
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
            test_article_data = {
                "title": final_title,
                "content": content,
                "date": date,
                "url": url,
            }
            if DEBUG_TASK_TRACE:
                t_rule = time.time()
                print(f"🧭 TASK {task_id} -> rule_check(full) start")
            if not rule_manager.should_include(symbol, test_article_data):
                return None # 被规则引擎拦截的噪音
            if DEBUG_TASK_TRACE:
                print(f"🧭 TASK {task_id} -> rule_check(full) done {time.time()-t_rule:.2f}s")
            
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
            test_article_data = {
                "title": final_title,
                "content": "",
                "date": date,
                "url": url,
            }
            if DEBUG_TASK_TRACE:
                t_rule = time.time()
                print(f"🧭 TASK {task_id} -> rule_check(title_only) start")
            if not rule_manager.should_include(symbol, test_article_data):
                return None  # 被规则引擎拦截（如 "Beauty intel" 没有 Intel 关键词）
            if DEBUG_TASK_TRACE:
                print(f"🧭 TASK {task_id} -> rule_check(title_only) done {time.time()-t_rule:.2f}s")
            
            # ✅ 保留只有标题的文章
            base_data.update({
                "title": final_title,
                "content": "",
                "data_quality": "title_only",
                "content_length": 0,
                "note": "Title extracted, content unavailable"
            })
        else:
            # 连标题都没有 -> 仅存 URL (url_only)
            base_data.update({"title": "No Title", "data_quality": "url_only"})
        if DEBUG_TASK_TRACE:
            print(f"🧭 TASK END {task_id} total={time.time()-t0:.2f}s quality={base_data.get('data_quality')}")
        return base_data
        
    except Exception as e:
        # 抓取彻底失败时的回退逻辑
        if title:
            is_placeholder = title.lower().startswith("news about ")
            if is_placeholder:
                return None # 抓取失败且标题是占位符，直接放弃
            
            # 即使抓取失败，也要通过规则检查
            test_article_data = {
                "title": title,
                "content": "",
                "date": date,
                "url": url,
            }
            if not rule_manager.should_include(symbol, test_article_data):
                return None  # 被规则引擎拦截
            
            # ✅ 抓取失败但有标题，保留为 title_only
            base_data.update({
                "title": title,
                "content": "",
                "data_quality": "title_only",
                "content_length": 0,
                "note": "Content extraction failed"
            })
            if DEBUG_TASK_TRACE:
                print(f"🧭 TASK END {task_id} total={time.time()-t0:.2f}s quality=title_only(fallback)")
            return base_data
        if DEBUG_TASK_TRACE:
            print(f"🧭 TASK END {task_id} total={time.time()-t0:.2f}s quality=None err={type(e).__name__}")
        return None


def process_one_batch(
    batch_idx,
    total_batches,
    batch_urls,
    valid_companies,
    dst_col,
    stuck_urls,
    stuck_urls_lock,
):
    worker = threading.current_thread().name
    def wlog(msg):
        print(f"[{worker}] {msg}")

    wlog(f"🚀 Processing File Batch {batch_idx}/{total_batches}...")

    # 实时下载当前批次的文件
    batch_download_files(batch_urls, batch_size=20, worker_name=worker)

    # A. 内存匹配
    matches = process_batch_files(batch_urls, valid_companies, worker_name=worker)
    if not matches:
        return 0

    # C. 数据库级去重：在抓取前先检查库里是否已存在
    all_batch_urls = list(set(m["row"]["URL"] for m in matches))
    existing_urls_cursor = dst_col.find({"url": {"$in": all_batch_urls}}, {"url": 1})
    existing_urls = set(doc["url"] for doc in existing_urls_cursor)

    filtered_matches = [m for m in matches if m["row"]["URL"] not in existing_urls]
    skipped_db = len(matches) - len(filtered_matches)
    if skipped_db > 0:
        wlog(f"⏭️ Database Deduplication: Skipped {skipped_db} articles already in {DST_COLLECTION}")

    matches = filtered_matches
    if not matches:
        return 0

    wlog(f"📰 Globally crawling {len(matches)} identified articles in parallel...")
    with stuck_urls_lock:
        before_stuck_filter = len(matches)
        matches = [m for m in matches if m["row"]["URL"] not in stuck_urls]
    if before_stuck_filter != len(matches):
        wlog(f"⏭️ Stuck URL skip: {before_stuck_filter - len(matches)}")
    if not matches:
        return 0
    if SHUTDOWN_EVENT.is_set():
        wlog("⚠️ Skip fetch stage because shutdown is in progress")
        return 0

    # 并行抓取正文
    final_results = []
    executor = ThreadPoolExecutor(max_workers=FETCH_WORKERS, thread_name_prefix=f"{worker}-fetch")
    timed_out = False
    try:
        matches_iter = iter(matches)
        future_to_meta = {}
        future_start_ts = {}
        pending = set()
        timed_out_urls = []

        def submit_next_task():
            try:
                m = next(matches_iter)
            except StopIteration:
                return False
            fut = executor.submit(fetch_article, m["row"], m["company"])
            future_to_meta[fut] = f"{m['company']['symbol']}|{m['row']['URL']}"
            future_start_ts[fut] = time.time()
            pending.add(fut)
            return True

        init_n = min(FETCH_WORKERS, len(matches))
        for _ in range(init_n):
            submit_next_task()

        completed = 0
        last_heartbeat = time.time()
        batch_start_ts = time.time()
        while pending:
            done_now, _ = concurrent.futures.wait(
                pending, timeout=1, return_when=concurrent.futures.FIRST_COMPLETED
            )

            for future in done_now:
                pending.discard(future)
                try:
                    res = future.result()
                    if res:
                        final_results.append(res)
                        if len(final_results) <= 3:
                            wlog(
                                f"    🐛 DEBUG: Got result with quality={res.get('data_quality')}, title={res.get('title', '')[:50]}"
                            )
                except Exception as e:
                    if DEBUG_TASK_TRACE:
                        meta = future_to_meta.get(future, "unknown")
                        wlog(f"⚠️ FUTURE ERROR {meta}: {type(e).__name__}: {e}")
                finally:
                    future_to_meta.pop(future, None)
                    future_start_ts.pop(future, None)
                    completed += 1
                    if completed % 200 == 0 or completed == len(matches):
                        wlog(f"⏳ {completed}/{len(matches)} articles processed in parallel pool...")
                    if not SHUTDOWN_EVENT.is_set():
                        submit_next_task()

            # 单 URL 超时：只跳过卡住任务，不终止整批
            now = time.time()
            stale = [
                f for f in list(pending)
                if now - future_start_ts.get(f, now) > FETCH_TASK_TIMEOUT
            ]
            if stale:
                timed_out = True
                stale_metas = [future_to_meta.get(sf, "unknown") for sf in stale]
                for sf in stale:
                    pending.discard(sf)
                    sf.cancel()
                    meta = future_to_meta.get(sf, "")
                    if "|" in meta:
                        timed_out_urls.append(meta.split("|", 1)[1])
                    future_to_meta.pop(sf, None)
                    future_start_ts.pop(sf, None)
                for meta in stale_metas[:10]:
                    wlog(f"   • stuck: {meta}")
                wlog(f"⚠️ FETCH TASK TIMEOUT ({FETCH_TASK_TIMEOUT}s): skipped {len(stale)} stuck URLs in this round.")
                completed += len(stale)
                if completed % 200 == 0 or completed == len(matches):
                    wlog(f"⏳ {completed}/{len(matches)} articles processed in parallel pool...")
                if not SHUTDOWN_EVENT.is_set():
                    for _ in stale:
                        if not submit_next_task():
                            break

            # 可选硬上限：防止极端情况下循环无限挂起
            if FETCH_BATCH_TIMEOUT > 0 and (now - batch_start_ts) > FETCH_BATCH_TIMEOUT:
                timed_out = True
                if pending:
                    wlog(f"⚠️ FETCH BATCH HARD TIMEOUT ({FETCH_BATCH_TIMEOUT}s): force skipping remaining {len(pending)} URLs.")
                    for pf in list(pending)[:10]:
                        wlog(f"   • stuck: {future_to_meta.get(pf, 'unknown')}")
                    for pf in list(pending):
                        pf.cancel()
                        meta = future_to_meta.get(pf, "")
                        if "|" in meta:
                            timed_out_urls.append(meta.split("|", 1)[1])
                        future_to_meta.pop(pf, None)
                        future_start_ts.pop(pf, None)
                    completed += len(pending)
                    pending.clear()

            if DEBUG_TASK_TRACE and time.time() - last_heartbeat >= 30:
                wlog(f"💓 HEARTBEAT done={completed}/{len(matches)} pending={len(pending)}")
                for pf in list(pending)[:5]:
                    wlog(f"   • pending: {future_to_meta.get(pf, 'unknown')}")
                last_heartbeat = time.time()

        if timed_out_urls:
            append_stuck_urls(timed_out_urls)
            with stuck_urls_lock:
                stuck_urls.update(timed_out_urls)
            wlog(f"⛔ Added {len(set(timed_out_urls))} URLs to stuck blacklist")
    finally:
        if timed_out:
            executor.shutdown(wait=False, cancel_futures=True)
        else:
            executor.shutdown(wait=True)

    results = final_results
    wlog(f"✅ {len(results)}/{len(matches)} articles successfully crawled in global pool")

    # D. 入库
    inserted_count = 0
    if results:
        company_insert_counts = {}
        quality_stats = {"full": 0, "low_value": 0, "url_only": 0}
        for art in results:
            try:
                quality = art.get("data_quality", "unknown")
                quality_stats[quality] = quality_stats.get(quality, 0) + 1

                result = dst_col.update_one(
                    {"url": art["url"]},
                    {"$setOnInsert": art},
                    upsert=True,
                )
                if result.upserted_id or result.modified_count > 0:
                    inserted_count += 1
                    symbol = art.get("symbol", "UNKNOWN")
                    company_insert_counts[symbol] = company_insert_counts.get(symbol, 0) + 1
            except Exception:
                pass

        wlog(f"💾 Batch saved: {inserted_count}/{len(results)} new articles")
        wlog("📊 Data Quality Distribution:")
        total_res = len(results)
        wlog(f"  • Full (Content matched): {quality_stats.get('full', 0)} ({quality_stats.get('full', 0)/total_res*100:.1f}%)")
        wlog(f"  • Low Value (Title only): {quality_stats.get('low_value', 0)} ({quality_stats.get('low_value', 0)/total_res*100:.1f}%)")
        wlog(f"  • URL Only/Unknown: {quality_stats.get('url_only', 0)} ({quality_stats.get('url_only', 0)/total_res*100:.1f}%)")
        if company_insert_counts:
            wlog("📌 Inserted by company:")
            sorted_inserts = sorted(company_insert_counts.items(), key=lambda x: x[1], reverse=True)
            for symbol, count in sorted_inserts:
                wlog(f"  • {symbol}: {count} articles")

    return inserted_count


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
    print(
        f"⚙️ Runtime config: scan_workers={SCAN_WORKERS}, fetch_workers={FETCH_WORKERS}, "
        f"article_timeout={ARTICLE_REQUEST_TIMEOUT}s, batch_timeout={FETCH_BATCH_TIMEOUT}s"
    )

    ensure_index()
    urls = get_gkg_file_urls(actual_years_back, actual_max_files)
    predownload_recent_missing_files(urls)
    
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
    stuck_urls = load_stuck_urls()
    print(f"⛔ Loaded {len(stuck_urls)} historical stuck URLs")
    
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
    total_files = len(urls)
    print(f"📌 Global scan target: files={total_files}, batches={total_batches}, batch_size={BATCH_SIZE}")
    stuck_urls_lock = threading.Lock()
    progress_lock = threading.Lock()
    progress_state = {
        "scanned_files": 0,
        "done_batches": 0,
    }

    if USE_MYSQL_BATCH_QUEUE:
        print(f"🗂️ MySQL queue mode enabled: workers={BATCH_WORKERS}")
        ensure_task_table()
        queue_resume_batch = max(1, start_batch)
        print(f"🧭 Queue resume batch: {queue_resume_batch} (mark 1..{max(0, queue_resume_batch-1)} as done)")
        seed_tasks(total_batches, queue_resume_batch)

        total_insert_lock = threading.Lock()

        def worker_loop(worker_id):
            global total_inserted
            owner = QUEUE_INSTANCE_ID
            threading.current_thread().name = owner
            print(f"[{owner}] started")
            while not SHUTDOWN_EVENT.is_set():
                batch_idx = claim_next_batch(owner)
                if batch_idx is None:
                    break
                i = (batch_idx - 1) * BATCH_SIZE
                batch_urls = urls[i : i + BATCH_SIZE]
                heartbeat_stop = threading.Event()

                def heartbeat_loop():
                    while not heartbeat_stop.wait(QUEUE_HEARTBEAT_SECONDS):
                        try:
                            heartbeat_batch(batch_idx, owner)
                        except Exception as hb_err:
                            print(f"[{owner}] ⚠️ heartbeat failed for batch {batch_idx}: {hb_err}")

                heartbeat_thread = threading.Thread(
                    target=heartbeat_loop,
                    name=f"{owner}-heartbeat",
                    daemon=True,
                )
                heartbeat_thread.start()
                try:
                    inserted = process_one_batch(
                        batch_idx=batch_idx,
                        total_batches=total_batches,
                        batch_urls=batch_urls,
                        valid_companies=valid_companies,
                        dst_col=dst_col,
                        stuck_urls=stuck_urls,
                        stuck_urls_lock=stuck_urls_lock,
                    )
                    with total_insert_lock:
                        total_inserted += inserted
                    mark_batch_done(batch_idx)
                    with progress_lock:
                        progress_state["scanned_files"] += len(batch_urls)
                        progress_state["done_batches"] += 1
                        pct = (progress_state["scanned_files"] / total_files * 100) if total_files else 0
                        print(
                            f"[{owner}] 📈 Global progress: "
                            f"batches={progress_state['done_batches']}/{total_batches}, "
                            f"files={progress_state['scanned_files']}/{total_files} ({pct:.2f}%)"
                        )
                except Exception as e:
                    print(f"[{owner}] ❌ failed on batch {batch_idx}: {e}")
                    # 解释器退出阶段会触发该错误，避免误标失败并继续调度
                    if "interpreter shutdown" in str(e).lower():
                        _mark_shutdown(f"{owner} got interpreter shutdown")
                        try:
                            requeue_batch(batch_idx, f"shutdown_requeue: {e}")
                            print(f"[{owner}] 🔁 requeued batch {batch_idx} to pending")
                        except Exception as requeue_err:
                            print(f"[{owner}] ⚠️ requeue failed for batch {batch_idx}: {requeue_err}")
                        print(f"[{owner}] ⚠️ interpreter is shutting down, stop worker loop")
                        break
                    mark_batch_failed(batch_idx, e)
                finally:
                    heartbeat_stop.set()
                    heartbeat_thread.join(timeout=1)
            print(f"[{owner}] ✅ finished.")

        workers = []
        for wid in range(1, BATCH_WORKERS + 1):
            t = threading.Thread(target=worker_loop, args=(wid,))
            workers.append(t)
            t.start()
        try:
            for t in workers:
                t.join()
        except KeyboardInterrupt:
            _mark_shutdown("KeyboardInterrupt in main join")
            for t in workers:
                t.join(timeout=5)
    else:
        for i in range(0, len(urls), BATCH_SIZE):
            batch_idx = (i // BATCH_SIZE) + 1
            if batch_idx <= start_batch:
                continue

            batch_urls = urls[i : i + BATCH_SIZE]
            inserted = process_one_batch(
                batch_idx=batch_idx,
                total_batches=total_batches,
                batch_urls=batch_urls,
                valid_companies=valid_companies,
                dst_col=dst_col,
                stuck_urls=stuck_urls,
                stuck_urls_lock=stuck_urls_lock,
            )
            total_inserted += inserted
            with progress_lock:
                progress_state["scanned_files"] += len(batch_urls)
                progress_state["done_batches"] += 1
                pct = (progress_state["scanned_files"] / total_files * 100) if total_files else 0
                print(
                    f"[MainThread] 📈 Global progress: "
                    f"batches={progress_state['done_batches']}/{total_batches}, "
                    f"files={progress_state['scanned_files']}/{total_files} ({pct:.2f}%)"
                )

            if not TEST_MODE:
                with open(PROGRESS_FILE, "w") as f:
                    f.write(str(batch_idx))

            if TEST_MODE and batch_idx * BATCH_SIZE >= len(urls):
                print(f"\n🧪 TEST MODE complete. Results saved in '{DST_COLLECTION}'.")
                break

            import gc
            gc.collect()

    if SHUTDOWN_EVENT.is_set():
        print(f"\n⚠️ Exiting with shutdown flag. reason={SHUTDOWN_REASON.get('reason')}")
    print(f"\n🏁 All done! Processed {len(urls)} files. Total articles inserted: {total_inserted}")
