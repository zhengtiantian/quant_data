import os
import requests
from datetime import datetime, UTC, timedelta
from pathlib import Path
from dotenv import load_dotenv
from pymongo import MongoClient


# =========================================================
# 1. Load environment variables
# =========================================================

CURRENT = Path(__file__).resolve()

# Global .env  (project root)
ROOT = CURRENT.parents[2]
GLOBAL_ENV = ROOT / ".env"
if GLOBAL_ENV.exists():
    load_dotenv(GLOBAL_ENV, override=False)
    print(f"Loaded GLOBAL .env: {GLOBAL_ENV}")
else:
    print(f"Global .env NOT FOUND at: {GLOBAL_ENV}")

# Module-level .env  (finnhub/.env)
MODULE_ENV = CURRENT.parent / ".env"
if MODULE_ENV.exists():
    load_dotenv(MODULE_ENV, override=True)
    print(f"Loaded MODULE .env: {MODULE_ENV}")
else:
    print(f"Module .env NOT FOUND at: {MODULE_ENV}")

# =========================================================
# 2. Read configuration
# =========================================================

FINNHUB_API_KEY = os.getenv("FINNHUB_API_KEY")
LIMIT = int(os.getenv("FINNHUB_NEWS_LIMIT", 50))
LANGUAGE = os.getenv("NEWS_LANGUAGE", "en")
MONGO_URI = os.getenv("MONGO_URI")

if not FINNHUB_API_KEY:
    raise RuntimeError("❌ FINNHUB_API_KEY missing in .env")


# =========================================================
# 3. Fetch Finnhub news
# =========================================================

def fetch_finnhub_news():
    """
    Finnhub 官方新闻接口:
    https://finnhub.io/docs/api/news
    免费计划可用，一分钟最多 60 请求
    """

    BASE_URL = "https://finnhub.io/api/v1/news"

    # Date range: last 3 days (free API requires date range)
    today = datetime.now(UTC).date()
    from_date = (today - timedelta(days=3)).isoformat()
    to_date = today.isoformat()

    params = {
        "category": "general",   # 市场新闻
        "minId": 0,
        "token": FINNHUB_API_KEY
    }

    print(f"\nRequest URL: {BASE_URL}")
    print("Params:", params)

    try:
        resp = requests.get(BASE_URL, params=params, headers={"User-Agent": "QuantNewsCollector"})
        resp.raise_for_status()
        data = resp.json()

        # 限制条数
        return data[:LIMIT]

    except Exception as e:
        print("❌ Finnhub request failed:", e)
        print("Response:", getattr(resp, "text", "N/A"))
        return []


# =========================================================
# 4. Save to MongoDB
# =========================================================

def save_to_mongo(articles):
    if not MONGO_URI:
        print("⚠️ No MONGO_URI set, skipping save.")
        return

    try:
        client = MongoClient(MONGO_URI)
        col = client["quant_data"]["news_finnhub"]

        docs = []
        for a in articles:
            docs.append({
                "source": {
                    "platform": "finnhub",
                    "name": a.get("source"),
                },
                "category": a.get("category"),
                "headline": a.get("headline"),
                "summary": a.get("summary"),
                "url": a.get("url"),
                "image": a.get("image"),
                "datetime": a.get("datetime"),
                "collectedAt": datetime.now(UTC).date().isoformat(),
                "language": LANGUAGE,
                "meta": {
                    "collector": "finnhub.collector",
                    "version": "1.0.0"
                }
            })

        if docs:
            col.insert_many(docs)
            print(f"Inserted {len(docs)} Finnhub news articles into MongoDB")

    except Exception as e:
        print("❌ MongoDB save error:", e)


# =========================================================
# 5. Main
# =========================================================

if __name__ == "__main__":
    articles = fetch_finnhub_news()

    print(f"\nFetched: {len(articles)}")
    for i, a in enumerate(articles[:5], 1):
        print(f"{i}. {a.get('headline')}")

    save_to_mongo(articles)