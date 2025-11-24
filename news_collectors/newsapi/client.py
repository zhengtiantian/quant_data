import os
import requests
import json
from datetime import datetime
from pathlib import Path
from dotenv import load_dotenv
from pymongo import MongoClient

CURRENT_FILE = Path(__file__).resolve()
PROJECT_ROOT = CURRENT_FILE.parents[2]
GLOBAL_ENV = PROJECT_ROOT / ".env"
MODULE_ENV = CURRENT_FILE.parent / ".env"

if GLOBAL_ENV.exists():
    load_dotenv(GLOBAL_ENV, override=False)
if MODULE_ENV.exists():
    load_dotenv(MODULE_ENV, override=True)

BASE_URL = os.getenv("NEWS_API_BASE_URL", "https://newsapi.org/v2/everything")
NEWSAPI_KEY = os.getenv("NEWS_API_KEY")
QUERY = os.getenv("NEWS_DEFAULT_QUERY", "stock OR market OR finance")
LANGUAGE = os.getenv("NEWS_LANGUAGE", "en")
PAGE_SIZE = int(os.getenv("NEWS_PAGE_SIZE", 50))
MONGO_URI = os.getenv("MONGO_URI")

if not NEWSAPI_KEY:
    raise RuntimeError("Missing NEWS_API_KEY")

def fetch_news(query=None):
    q = query or QUERY
    params = {
        "q": q,
        "language": LANGUAGE,
        "pageSize": PAGE_SIZE,
        "apiKey": NEWSAPI_KEY,
        "sortBy": "publishedAt",
    }
    try:
        resp = requests.get(BASE_URL, params=params)
        resp.raise_for_status()
        return resp.json().get("articles", [])
    except:
        return []

# 美股过滤器
US_STOCK_KEYWORDS = [
    "nasdaq", "dow jones", "s&p", "s&p500", "sp500",
    "federal reserve", "fed", "wall street",
    "us stocks", "u.s. stocks",
    "stock market crash", "rate cut", "inflation", "rate hike",
    "treasury yield", "bond yield",
    "microsoft", "google", "alphabet", "amazon",
    "meta", "facebook", "apple", "tesla", "nvidia", "amd", "intel"
]

def is_us_stock_news(article):
    text = " ".join([
        article.get("title") or "",
        article.get("description") or "",
        article.get("content") or ""
    ]).lower()
    return any(k in text for k in US_STOCK_KEYWORDS)

if __name__ == "__main__":
    raw = fetch_news()
    articles = [a for a in raw if is_us_stock_news(a)]

    print("Total US stock articles:", len(articles))
    for i, a in enumerate(articles[:10], 1):
        print(f"{i}. {a.get('title')}")