import os
import requests
import zipfile
import io
import csv
import pandas as pd
from datetime import datetime, timedelta, timezone
from newspaper import Article, Config
import concurrent.futures
import re
import random
from pymongo import MongoClient, UpdateOne, errors
import time
import multiprocessing
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from concurrent.futures.process import BrokenProcessPool
import threading
import mysql.connector
import builtins
import signal
import traceback
import sys
from bs4 import BeautifulSoup

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
WORKER_BATCH_CONTEXT = {}


def _get_slm_stats_suffix(thread_name):
    return ""


def _thread_print(*args, **kwargs):
    thread_name = threading.current_thread().name
    suffix = _get_slm_stats_suffix(thread_name)
    if args and isinstance(args[0], str) and args[0].startswith("["):
        rendered = list(args)
        rendered[0] = f"{rendered[0]}{suffix}"
        _ORIGINAL_PRINT(*rendered, **kwargs)
        return
    _ORIGINAL_PRINT(f"[{thread_name}]{suffix}", *args, **kwargs)

builtins.print = _thread_print

# ============================
# 全局配置
# ============================
_rule_manager = None
_rule_manager_lock = threading.Lock()


def get_rule_manager():
    global _rule_manager
    if _rule_manager is None:
        with _rule_manager_lock:
            if _rule_manager is None:
                _rule_manager = RuleManager()
    return _rule_manager


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
config.language = "en"

FETCH_WORKERS = int(os.getenv("FETCH_WORKERS", "3"))
PARSE_WORKERS = int(os.getenv("PARSE_WORKERS", "2"))
FETCH_TASK_TIMEOUT = int(os.getenv("FETCH_TASK_TIMEOUT", "20"))
PARSE_POOL_MAX_TASKS_PER_CHILD = int(os.getenv("PARSE_POOL_MAX_TASKS_PER_CHILD", "100"))
SCAN_WORKERS = int(os.getenv("SCAN_WORKERS", "2"))
SCAN_PROGRESS_EVERY_FILES = int(os.getenv("SCAN_PROGRESS_EVERY_FILES", "100"))
SCAN_RULE_PROGRESS_EVERY_RECORDS = int(os.getenv("SCAN_RULE_PROGRESS_EVERY_RECORDS", "0"))
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


def _build_article_base_data(row, company):
    return {
        "symbol": company["symbol"],
        "name": company["name"],
        "date": row.get("Date", ""),
        "timestamp": row.get("Timestamp", ""),
        "publishedAt": row.get("PublishedAt", None),
        "url": str(row["URL"]),
        "source": {"platform": "gdelt"},
        "collectedAt": datetime.now(timezone.utc).isoformat(),
        "missing_fields": [],
        "data_quality": "url_only",
    }


def _normalize_publish_date(base_data, publish_date):
    if not publish_date:
        return
    try:
        article_publish_date = publish_date.strftime("%Y%m%d%H%M%S")
        if len(article_publish_date) == 14 and article_publish_date.isdigit():
            time_part = article_publish_date[8:]
            if time_part != "000000":
                base_data["date"] = article_publish_date
                base_data["timestamp"] = article_publish_date
                base_data["publishedAt"] = publish_date.isoformat()
    except Exception:
        pass


def _is_junk_page(title, content):
    junk_indicators = [
        "necessary cookies", "functional cookies", "analytical cookies",
        "confirm you are a human", "not a bot", "captcha test",
        "page was not found", "404 not found", "access denied",
        "pilihan situs bandar", "togel online", "slot gacor", "toto macau",
        "before you continue to youtube", "before you continue to google",
    ]
    content_lower = (content or "").lower()
    title_lower = (title or "").lower()
    return any(indicator in content_lower for indicator in junk_indicators) or any(
        indicator in title_lower
        for indicator in ["before you continue to youtube", "before you continue to google"]
    )


def _finalize_article_result(base_data, title_candidate, content, note=None):
    final_title = (title_candidate or "").strip() or "No Title"
    is_placeholder = final_title.lower().startswith("news about ")
    content = (content or "").strip()
    symbol = base_data["symbol"]
    article_for_rules = {
        "title": final_title,
        "content": content,
        "date": base_data.get("date", ""),
        "url": base_data["url"],
    }

    if _is_junk_page(final_title, content):
        return None

    rule_manager = get_rule_manager()

    if len(content) >= 60:
        if not rule_manager.should_include(symbol, article_for_rules):
            return None
        base_data.update({
            "title": final_title,
            "content": content,
            "data_quality": "full",
            "content_length": len(content),
        })
        if note:
            base_data["note"] = note
        return base_data

    if final_title != "No Title":
        if is_placeholder:
            return None
        article_for_rules["content"] = ""
        if not rule_manager.should_include(symbol, article_for_rules):
            return None
        base_data.update({
            "title": final_title,
            "content": "",
            "data_quality": "title_only",
            "content_length": 0,
            "note": note or "Title extracted, content unavailable",
        })
        return base_data

    base_data.update({
        "title": "No Title",
        "content": "",
        "content_length": 0,
        "data_quality": "url_only",
    })
    if note:
        base_data["note"] = note
    return base_data


def _extract_text_with_bs4(html):
    if not html:
        return "", ""
    soup = BeautifulSoup(html, "html.parser")
    for tag in soup(["script", "style", "noscript", "svg", "iframe", "header", "footer", "nav", "form"]):
        tag.decompose()
    title = ""
    if soup.title and soup.title.string:
        title = soup.title.string.strip()
    body = soup.body or soup
    text = body.get_text("\n", strip=True)
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    return title, "\n".join(lines)


def _normalize_lang_hint(value):
    if not value:
        return ""
    value = str(value).strip().lower().replace("_", "-")
    if "," in value:
        value = value.split(",", 1)[0].strip()
    if ";" in value:
        value = value.split(";", 1)[0].strip()
    return value


def _extract_html_language_hints(html):
    if not html:
        return []
    hints = []
    try:
        soup = BeautifulSoup(html, "html.parser")
        html_tag = soup.find("html")
        if html_tag:
            hints.extend(
                [
                    html_tag.get("lang"),
                    html_tag.get("xml:lang"),
                ]
            )
        for meta in soup.find_all("meta"):
            hints.extend(
                [
                    meta.get("lang"),
                    meta.get("content-language"),
                ]
            )
            http_equiv = (meta.get("http-equiv") or "").strip().lower()
            if http_equiv == "content-language":
                hints.append(meta.get("content"))
    except Exception:
        return []
    normalized = []
    for hint in hints:
        norm = _normalize_lang_hint(hint)
        if norm:
            normalized.append(norm)
    return normalized


def _is_english_hint(lang_hint):
    return bool(lang_hint) and lang_hint.startswith("en")


def _looks_like_english_text(text):
    if not text:
        return False
    sample = text[:4000]
    letters = [ch for ch in sample if ch.isalpha()]
    if letters:
        ascii_ratio = sum(("a" <= ch.lower() <= "z") for ch in letters) / len(letters)
        if ascii_ratio < 0.85:
            return False
    words = re.findall(r"[a-z']+", sample.lower())
    if len(words) < 20:
        return True
    english_markers = {
        "the", "and", "for", "with", "from", "that", "this", "will", "said",
        "have", "has", "was", "were", "are", "is", "be", "as", "at", "by",
        "on", "of", "to", "in", "after", "before", "about", "into", "over",
    }
    marker_hits = sum(word in english_markers for word in words[:200])
    return (marker_hits / min(len(words), 200)) >= 0.04


def _should_keep_english_only(html="", title="", content=""):
    lang_hints = _extract_html_language_hints(html) if html else []
    if lang_hints and not any(_is_english_hint(hint) for hint in lang_hints):
        return False
    text_sample = "\n".join(part for part in [title, content] if part).strip()
    if not text_sample:
        return not lang_hints or any(_is_english_hint(hint) for hint in lang_hints)
    return _looks_like_english_text(text_sample)


def _fetch_html_with_requests(url):
    resp = requests.get(
        url,
        headers={"User-Agent": user_agent},
        timeout=ARTICLE_REQUEST_TIMEOUT,
    )
    resp.raise_for_status()
    return resp.text


def _extract_article_payload(row, company, debug_trace=False):
    url = str(row["URL"])
    title = row.get("Title", "")
    base_data = _build_article_base_data(row, company)
    task_id = f"{company['symbol']}|{url}"
    t0 = time.time()

    try:
        if debug_trace:
            print(f"🧭 TASK START {task_id}")
        art = Article(url, config=config)
        html = _fetch_html_with_requests(url)
        if not _should_keep_english_only(html=html, title=title, content=""):
            return None
        art.set_html(html)
        if debug_trace:
            print(f"🧭 TASK {task_id} -> download done {time.time() - t0:.2f}s")
        art.parse()
        _normalize_publish_date(base_data, art.publish_date)
        final_title = (art.title or "").strip() or title
        if not _should_keep_english_only(html=html, title=final_title, content=art.text):
            return None
        result = _finalize_article_result(
            base_data,
            final_title,
            art.text,
            note=None,
        )
        if result:
            return result

        # 规则或垃圾页过滤掉的内容直接丢弃，不再 fallback
        return None
    except Exception as err:
        try:
            html = ""
            if "art" in locals() and getattr(art, "html", None):
                html = art.html
            if not html:
                html = _fetch_html_with_requests(url)
            fallback_title, fallback_content = _extract_text_with_bs4(html)
            if not _should_keep_english_only(html=html, title=fallback_title or title, content=fallback_content):
                return None
            return _finalize_article_result(
                base_data,
                fallback_title or title,
                fallback_content,
                note=f"bs4_fallback_after_{type(err).__name__}",
            )
        except Exception:
            if title:
                return _finalize_article_result(
                    base_data,
                    title,
                    "",
                    note=f"Content extraction failed after {type(err).__name__}",
                )
            return None
        finally:
            if debug_trace:
                print(
                    f"🧭 TASK END {task_id} total={time.time() - t0:.2f}s "
                    f"fallback_from={type(err).__name__}"
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
    print("⬇️ Refreshing masterfilelist.txt from GDELT...")
    try:
        with requests.get(f"{BASE_URL}/masterfilelist.txt", stream=True, timeout=1800) as r:
            r.raise_for_status()
            with open(MASTER_FILE, "wb") as f:
                for chunk in r.iter_content(8192):
                    f.write(chunk)
    except Exception as e:
        if os.path.exists(MASTER_FILE):
            print(
                f"⚠️ Refresh masterfilelist failed: {e}. "
                f"Falling back to cached file ({os.path.getsize(MASTER_FILE)/1e6:.1f} MB)"
            )
        else:
            raise
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


def get_latest_cached_gkg_timestamp():
    """返回本地缓存中最新 GKG 文件的时间戳；没有缓存则返回 None。"""
    latest_dt = None
    try:
        for name in os.listdir(FILES_DIR):
            if not name.endswith(".gkg.csv.zip"):
                continue
            ts = parse_gdelt_timestamp(name)
            if not ts:
                continue
            dt = ts["datetime"]
            if latest_dt is None or dt > latest_dt:
                latest_dt = dt
    except FileNotFoundError:
        return None
    return latest_dt


def predownload_recent_missing_files(urls):
    """启动时从本地最新缓存补齐到今天，避免新时间段的 GKG 文件缺口。"""
    if not STARTUP_PREDOWNLOAD_ENABLED or not urls:
        return

    latest_cached_dt = get_latest_cached_gkg_timestamp()
    if latest_cached_dt is None:
        lookback = max(1, STARTUP_PREDOWNLOAD_RECENT_FILES)
        candidate_urls = urls[-lookback:]
        mode_desc = f"no local cache, fallback to latest {lookback} files"
    else:
        candidate_urls = []
        for url in urls:
            ts = parse_gdelt_timestamp(os.path.basename(url))
            if not ts:
                continue
            if ts["datetime"] > latest_cached_dt:
                candidate_urls.append(url)
        mode_desc = f"latest cached file at {latest_cached_dt.strftime('%Y-%m-%d %H:%M:%S')}"

    missing = [
        url for url in candidate_urls
        if not os.path.exists(os.path.join(FILES_DIR, os.path.basename(url)))
    ]
    if not missing:
        print(f"📦 Startup predownload: nothing missing to catch up ({mode_desc})")
        return
    print(
        f"📦 Startup predownload: caching {len(missing)}/{len(candidate_urls)} files "
        f"before workers start ({mode_desc})"
    )
    batch_download_files(candidate_urls, batch_size=50, worker_name="startup-cache")


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


def read_gkg_tsv(file_obj):
    """使用 pandas Python engine 读取 GKG，绕开多线程下不稳定的 C parser。"""
    return pd.read_csv(
        file_obj,
        sep="\t",
        header=None,
        encoding="ISO-8859-1",
        on_bad_lines="skip",
        engine="python",
        dtype=str,
        keep_default_na=False,
        quoting=csv.QUOTE_NONE,
    )


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
                df = read_gkg_tsv(f)
        
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
    rule_manager = get_rule_manager()
    
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
                    df = read_gkg_tsv(f)
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
                    if SCAN_RULE_PROGRESS_EVERY_RECORDS > 0 and rp % SCAN_RULE_PROGRESS_EVERY_RECORDS == 0:
                        worker = worker_name or threading.current_thread().name
                        print(f"[{worker}] 🧪 Rule check progress: {rp} records processed")
            t_rules = time.time()
            total_time = t_rules - t_start
            return {
                "matches": file_matches,
                "rows": len(df),
                "candidates": len(matched_df),
                "sample": sample_text,
                "filename": filename,
                "timings": {
                    "total": total_time,
                    "parse": t_parse - t_start,
                    "combine": t_combine - t_parse,
                    "regex": t_regex - t_combine,
                    "rules": t_rules - t_regex,
                },
            }
        except Exception as e:
            print(f"❌ Error processing {filename}: {e}")
            return {
                "matches": [],
                "rows": 0,
                "candidates": 0,
                "sample": "",
                "error": str(e),
                "filename": filename,
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
                    timings = res.get("timings", {})
                    file_name = res.get("filename", "?")
                    print(
                        f"[{worker}] 🔍 Scan progress: {done}/{total_files} files "
                        f"({elapsed:.1f}s, {rate:.1f} files/s), "
                        f"rows={progress['rows_scanned']}, candidates={progress['candidates']}, accepted={progress['accepted']} | "
                        f"last={file_name} total={timings.get('total', 0):.1f}s "
                        f"(parse={timings.get('parse', 0):.2f}s, combine={timings.get('combine', 0):.2f}s, "
                        f"regex={timings.get('regex', 0):.2f}s, rules={timings.get('rules', 0):.2f}s)"
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
    return _extract_article_payload(row, company, debug_trace=DEBUG_TASK_TRACE)


def build_title_only_fallback(row, company, note):
    base_data = _build_article_base_data(row, company)
    title = row.get("Title", "")
    return _finalize_article_result(base_data, title, "", note=note)


def build_parse_executor():
    kwargs = {
        "max_workers": PARSE_WORKERS,
        "mp_context": multiprocessing.get_context("spawn"),
    }
    if sys.version_info >= (3, 11):
        kwargs["max_tasks_per_child"] = PARSE_POOL_MAX_TASKS_PER_CHILD
    return ProcessPoolExecutor(**kwargs)


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
    log_prefix = f"[{worker}][batch-{batch_idx}]"
    def wlog(msg):
        print(f"{log_prefix} {msg}")

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

    # 正文抓取放到子进程池，避免 lxml/newspaper3k native crash 直接打死主进程。
    final_results = []
    executor = build_parse_executor()
    timed_out = False
    pool_broken = False
    fetch_stage_start = time.time()
    try:
        matches_iter = iter(matches)
        future_to_meta = {}
        future_to_match = {}
        future_start_ts = {}
        pending = set()
        timed_out_urls = []
        total_task_elapsed = 0.0
        total_task_count = 0
        total_timeout_count = 0

        def log_fetch_progress():
            elapsed = time.time() - fetch_stage_start
            rate = completed / elapsed if elapsed > 0 else 0
            avg_task = (total_task_elapsed / total_task_count) if total_task_count > 0 else 0
            remaining = max(len(matches) - completed, 0)
            eta_seconds = (remaining / rate) if rate > 0 else 0
            eta_minutes = eta_seconds / 60 if eta_seconds > 0 else 0
            wlog(
                f"⏳ {completed}/{len(matches)} articles processed in parallel pool... "
                f"(elapsed={elapsed:.1f}s, rate={rate:.2f}/s, avg_task={avg_task:.2f}s, "
                f"pending={len(pending)}, timeouts={total_timeout_count}, eta={eta_minutes:.1f}m)"
            )

        def submit_next_task():
            try:
                m = next(matches_iter)
            except StopIteration:
                return False
            fut = executor.submit(fetch_article, m["row"], m["company"])
            future_to_meta[fut] = f"{m['company']['symbol']}|{m['row']['URL']}"
            future_to_match[fut] = m
            future_start_ts[fut] = time.time()
            pending.add(fut)
            return True

        init_n = min(FETCH_WORKERS, len(matches))
        for _ in range(init_n):
            submit_next_task()

        completed = 0
        last_heartbeat = time.time()
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
                except BrokenProcessPool as e:
                    pool_broken = True
                    timed_out = True
                    meta = future_to_meta.get(future, "unknown")
                    wlog(f"⚠️ Parse pool crashed on {meta}: {e}")
                    fallback_match = future_to_match.get(future)
                    if fallback_match:
                        fallback = build_title_only_fallback(
                            fallback_match["row"],
                            fallback_match["company"],
                            note="process_pool_crash_fallback",
                        )
                        if fallback:
                            final_results.append(fallback)
                    remaining_matches = [future_to_match.get(pf) for pf in list(pending)]
                    for pending_match in remaining_matches:
                        if not pending_match:
                            continue
                        fallback = build_title_only_fallback(
                            pending_match["row"],
                            pending_match["company"],
                            note="process_pool_crash_fallback",
                        )
                        if fallback:
                            final_results.append(fallback)
                    pending.clear()
                except Exception as e:
                    if DEBUG_TASK_TRACE:
                        meta = future_to_meta.get(future, "unknown")
                        wlog(f"⚠️ FUTURE ERROR {meta}: {type(e).__name__}: {e}")
                finally:
                    task_started_at = future_start_ts.get(future)
                    if task_started_at is not None:
                        total_task_elapsed += max(time.time() - task_started_at, 0)
                        total_task_count += 1
                    future_to_meta.pop(future, None)
                    future_to_match.pop(future, None)
                    future_start_ts.pop(future, None)
                    completed += 1
                    if completed % 10 == 0 or completed == len(matches):
                        log_fetch_progress()
                    if not SHUTDOWN_EVENT.is_set() and not pool_broken:
                        submit_next_task()

            if pool_broken:
                completed = len(matches)
                break

            # 单 URL 超时：只跳过卡住任务，不终止整批
            now = time.time()
            stale = [
                f for f in list(pending)
                if now - future_start_ts.get(f, now) > FETCH_TASK_TIMEOUT
            ]
            if stale:
                timed_out = True
                total_timeout_count += len(stale)
                stale_metas = [future_to_meta.get(sf, "unknown") for sf in stale]
                for sf in stale:
                    task_started_at = future_start_ts.get(sf)
                    if task_started_at is not None:
                        total_task_elapsed += max(now - task_started_at, 0)
                        total_task_count += 1
                    pending.discard(sf)
                    sf.cancel()
                    meta = future_to_meta.get(sf, "")
                    if "|" in meta:
                        timed_out_urls.append(meta.split("|", 1)[1])
                    future_to_meta.pop(sf, None)
                    pending_match = future_to_match.pop(sf, None)
                    if pending_match:
                        fallback = build_title_only_fallback(
                            pending_match["row"],
                            pending_match["company"],
                            note=f"fetch_timeout_{FETCH_TASK_TIMEOUT}s_fallback",
                        )
                        if fallback:
                            final_results.append(fallback)
                    future_start_ts.pop(sf, None)
                for meta in stale_metas[:10]:
                    wlog(f"   • stuck: {meta}")
                wlog(f"⚠️ FETCH TASK TIMEOUT ({FETCH_TASK_TIMEOUT}s): skipped {len(stale)} stuck URLs in this round.")
                completed += len(stale)
                if completed % 10 == 0 or completed == len(matches):
                    log_fetch_progress()
                if not SHUTDOWN_EVENT.is_set():
                    for _ in stale:
                        if not submit_next_task():
                            break

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
        quality_rank = {"full": 3, "title_only": 2, "url_only": 1}
        quality_stats = {"full": 0, "title_only": 0, "url_only": 0}
        deduped_results = {}
        for art in results:
            quality = art.get("data_quality", "unknown")
            quality_stats[quality] = quality_stats.get(quality, 0) + 1
            url = art.get("url")
            if not url:
                continue
            existing = deduped_results.get(url)
            if existing is None:
                deduped_results[url] = art
                continue
            existing_rank = quality_rank.get(existing.get("data_quality", "url_only"), 0)
            current_rank = quality_rank.get(art.get("data_quality", "url_only"), 0)
            if current_rank > existing_rank:
                deduped_results[url] = art

        bulk_start = time.time()
        ops = [
            UpdateOne(
                {"url": art["url"]},
                {"$setOnInsert": art},
                upsert=True,
            )
            for art in deduped_results.values()
        ]
        if ops:
            try:
                bulk_result = dst_col.bulk_write(ops, ordered=False)
                inserted_count = int(getattr(bulk_result, "upserted_count", 0) or 0)
            except errors.BulkWriteError as e:
                details = getattr(e, "details", {}) or {}
                inserted_count = int(details.get("nUpserted", 0) or 0)
                write_errors = details.get("writeErrors", [])
                non_dup_errors = [we for we in write_errors if we.get("code") != 11000]
                if non_dup_errors:
                    wlog(f"⚠️ Bulk write had {len(non_dup_errors)} non-duplicate errors")
            except Exception as e:
                wlog(f"⚠️ Bulk write failed: {type(e).__name__}: {e}")
        bulk_elapsed = time.time() - bulk_start

        wlog(
            f"💾 Batch saved: {inserted_count}/{len(deduped_results)} new articles "
            f"(bulk_upsert_time={bulk_elapsed:.2f}s, raw_results={len(results)})"
        )
        wlog("📊 Data Quality Distribution:")
        total_res = len(results)
        wlog(f"  • Full (Content matched): {quality_stats.get('full', 0)} ({quality_stats.get('full', 0)/total_res*100:.1f}%)")
        wlog(f"  • Title Only: {quality_stats.get('title_only', 0)} ({quality_stats.get('title_only', 0)/total_res*100:.1f}%)")
        wlog(f"  • URL Only/Unknown: {quality_stats.get('url_only', 0)} ({quality_stats.get('url_only', 0)/total_res*100:.1f}%)")

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
        f"parse_workers={PARSE_WORKERS}, article_timeout={ARTICLE_REQUEST_TIMEOUT}s, "
        f"fetch_task_timeout={FETCH_TASK_TIMEOUT}s"
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
    
    if TEST_MODE:
        print(f"🧪 TEST MODE: Starting from batch 1")
    
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
        queue_resume_batch = 1
        print("🧭 Queue resume source: MySQL task table only")
        seed_tasks(total_batches, queue_resume_batch)

        total_insert_lock = threading.Lock()

        def worker_loop(worker_id):
            global total_inserted
            owner = QUEUE_INSTANCE_ID
            worker_name = f"{HOST_ID}-worker{worker_id}"
            threading.current_thread().name = worker_name
            print("started")
            while not SHUTDOWN_EVENT.is_set():
                batch_idx = claim_next_batch(owner)
                if batch_idx is None:
                    break
                WORKER_BATCH_CONTEXT[worker_name] = batch_idx
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
                    name=f"{worker_name}-heartbeat",
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
                    print(f"❌ failed on batch {batch_idx}: {e}")
                    # 解释器退出阶段会触发该错误，避免误标失败并继续调度
                    if "interpreter shutdown" in str(e).lower():
                        _mark_shutdown(f"{owner} got interpreter shutdown")
                        try:
                            requeue_batch(batch_idx, f"shutdown_requeue: {e}")
                            print(f"🔁 requeued batch {batch_idx} to pending")
                        except Exception as requeue_err:
                            print(f"⚠️ requeue failed for batch {batch_idx}: {requeue_err}")
                        print("⚠️ interpreter is shutting down, stop worker loop")
                        break
                    mark_batch_failed(batch_idx, e)
                finally:
                    heartbeat_stop.set()
                    heartbeat_thread.join(timeout=1)
                    WORKER_BATCH_CONTEXT.pop(worker_name, None)
            print("✅ finished.")

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

            if TEST_MODE and batch_idx * BATCH_SIZE >= len(urls):
                print(f"\n🧪 TEST MODE complete. Results saved in '{DST_COLLECTION}'.")
                break

            import gc
            gc.collect()

    if SHUTDOWN_EVENT.is_set():
        print(f"\n⚠️ Exiting with shutdown flag. reason={SHUTDOWN_REASON.get('reason')}")
    print(f"\n🏁 All done! Processed {len(urls)} files. Total articles inserted: {total_inserted}")
