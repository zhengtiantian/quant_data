# quant_data/scheduler/tasks.py

import os
import time
import logging
from pathlib import Path

import schedule
from dotenv import load_dotenv
from pymongo import MongoClient

# 加载环境变量
ROOT = Path(__file__).resolve().parents[1]
ENV_FILE = ROOT / ".env"
load_dotenv(ENV_FILE)
MONGO_URI = os.getenv("MONGO_URI")
if not MONGO_URI:
    raise RuntimeError("MONGO_URI not set")

# 导入你的 collectors
from news_collectors.finnhub.collector import fetch_finnhub
from news_collectors.newsapi.collector import fetch_news
from news_collectors.yahoo.collector import fetch_yahoo_news

# 设置日志
LOG_FILE = ROOT / "news_collect_scheduler.log"
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    handlers=[
        logging.FileHandler(LOG_FILE),
        logging.StreamHandler()
    ]
)

# deduper
class Deduper:
    def __init__(self, mongo_uri):
        client = MongoClient(mongo_uri)
        self.col = client["quant_data"]["news_articles"]
        self.col.create_index({"url": 1}, unique=True, sparse=True)

    def save_articles(self, articles):
        inserted = 0
        for a in articles:
            try:
                self.col.insert_one(a)
                inserted += 1
            except Exception:
                pass
        return inserted

deduper = Deduper(MONGO_URI)

def job_finnhub():
    logging.info("Start Finnhub job")
    try:
        data = fetch_finnhub()
        n = deduper.save_articles(data)
        logging.info(f"Finnhub → inserted {n} articles")
    except Exception as e:
        logging.exception("Finnhub job failed")

def job_newsapi():
    logging.info("Start NewsAPI job")
    try:
        data = fetch_news()
        n = deduper.save_articles(data)
        logging.info(f"NewsAPI → inserted {n} articles")
    except Exception as e:
        logging.exception("NewsAPI job failed")

def job_yahoo():
    logging.info("Start Yahoo job")
    try:
        data = fetch_yahoo_news()
        n = deduper.save_articles(data)
        logging.info(f"Yahoo → inserted {n} articles")
    except Exception as e:
        logging.exception("Yahoo job failed")

def main_loop():
    logging.info("=== Scheduler started ===")
    # 设定频率：可根据实际需求调整
    schedule.every(10).minutes.do(job_finnhub)
    schedule.every(30).minutes.do(job_newsapi)
    schedule.every(60).minutes.do(job_yahoo)

    while True:
        schedule.run_pending()
        time.sleep(5)

if __name__ == "__main__":
    main_loop()