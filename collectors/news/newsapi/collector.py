import os
import requests
from datetime import datetime, UTC
from pathlib import Path
from dotenv import load_dotenv
from pymongo import MongoClient, errors

CURRENT_FILE = Path(__file__).resolve()
PROJECT_ROOT = CURRENT_FILE.parents[3]
GLOBAL_ENV = PROJECT_ROOT / ".env"
MODULE_ENV = CURRENT_FILE.parent / ".env"

# Load env
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
if not MONGO_URI:
    raise RuntimeError("Missing MONGO_URI")


# =========================================================
# 1. Fetch NewsAPI Data
# =========================================================
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


# =========================================================
# 2. Filters
# =========================================================

US_STOCK_KEYWORDS = [
    "nasdaq", "dow jones", "s&p", "s&p500", "sp500",
    "federal reserve", "fed", "wall street",
    "us stocks", "u.s. stocks", "stock market",
    "rate cut", "inflation", "rate hike",
    "treasury yield", "bond yield",
    "microsoft", "google", "alphabet", "amazon",
    "meta", "facebook", "apple", "tesla", "nvidia", "amd", "intel",
    "goldman", "jpmorgan", "bank of america", "citigroup",
]

A_LEVEL_KEYWORDS = [
    "federal reserve", "fed", "rate hike", "rate cut",
    "inflation", "cpi", "ppi", "treasury yield",
    "market crash", "selloff", "plunge", "rally",
    "earnings", "forecast", "guidance", "revenue",
    "sec investigation", "lawsuit", "antitrust",
    "nvidia", "apple", "google", "tesla",
    "trade war", "tariff", "sanction",
]


def is_us_stock_news(article):
    text = " ".join([
        article.get("title") or "",
        article.get("description") or "",
        article.get("content") or ""
    ]).lower()
    return any(k in text for k in US_STOCK_KEYWORDS)


def classify_impact(article):
    text = " ".join([
        article.get("title") or "",
        article.get("description") or "",
        article.get("content") or ""
    ]).lower()

    for k in A_LEVEL_KEYWORDS:
        if k in text:
            return "A"
    return "B"


# =========================================================
# 3. Save to MongoDB
# =========================================================
def save_to_mongo(articles):
    client = MongoClient(MONGO_URI)
    col = client["quant_data"]["news_articles"]

    inserted = 0

    for a in articles:
        doc = {
            "source": {
                "platform": "newsapi",
                "name": a.get("source", {}).get("name", ""),
            },
            "title": a.get("title"),
            "description": a.get("description"),
            "content": a.get("content"),
            "url": a.get("url"),
            "publishedAt": a.get("publishedAt"),
            "collectedAt": datetime.now(UTC).isoformat(),
            "language": LANGUAGE,
            "impact": a.get("impact", "B"),
            "meta": {"collector": "newsapi.collector", "version": "1.0"},
        }

        try:
            col.insert_one(doc)
            inserted += 1
        except errors.DuplicateKeyError:
            continue

    print(f"Inserted {inserted} NewsAPI articles into MongoDB")


# =========================================================
# 4. Ensure unique index on URL
# =========================================================
def ensure_index():
    client = MongoClient(MONGO_URI)
    col = client["quant_data"]["news_articles"]
    try:
        col.create_index({"url": 1}, unique=True, sparse=True)
        print("Index created on url")
    except:
        print("Index already exists")


# =========================================================
# 5. Main
# =========================================================
if __name__ == "__main__":
    ensure_index()

    raw = fetch_news()

    filtered = []
    for a in raw:
        if is_us_stock_news(a):
            a["impact"] = classify_impact(a)
            filtered.append(a)

    print("Total US stock articles:", len(filtered))
    for i, a in enumerate(filtered[:10], 1):
        print(f"{i}. ({a['impact']}) {a.get('title')}")

    save_to_mongo(filtered)