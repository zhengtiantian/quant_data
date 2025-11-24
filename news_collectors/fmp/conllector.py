import os
import requests
from datetime import datetime
from pathlib import Path
from dotenv import load_dotenv
from pymongo import MongoClient

# =========================================================
# 1. Load environment variables
# =========================================================

CURRENT = Path(__file__).resolve()

# Global .env (project root)
ROOT = CURRENT.parents[2]
GLOBAL_ENV = ROOT / ".env"
load_dotenv(GLOBAL_ENV, override=False)
print(f"Loaded GLOBAL .env: {GLOBAL_ENV}")

# Module .env (FMP overrides)
MODULE_ENV = CURRENT.parent / ".env"
if MODULE_ENV.exists():
    load_dotenv(MODULE_ENV, override=True)
    print(f"Loaded MODULE .env: {MODULE_ENV}")

# Read env vars
FMP_API_KEY = os.getenv("FMP_API_KEY")
LIMIT = int(os.getenv("FMP_NEWS_LIMIT", 50))
MONGO_URI = os.getenv("MONGO_URI")

if not FMP_API_KEY:
    raise RuntimeError("❌ FMP_API_KEY missing in fmp/.env")

BASE_URL = "https://financialmodelingprep.com/stable/stock_news"


# =========================================================
# 2. Fetch FMP news
# =========================================================
def fetch_fmp_news():
    url = f"{BASE_URL}?limit={LIMIT}&apikey={FMP_API_KEY}"
    print("\nRequest URL:", url)

    try:
        resp = requests.get(url, headers={"User-Agent": "QuantNewsCollector"})
        resp.raise_for_status()
        data = resp.json()

        if isinstance(data, dict) and "error" in data:
            print("❌ API Error:", data["error"])
            return []

        return data

    except requests.exceptions.HTTPError as e:
        print("❌ FMP request failed:", e)
        print("Response:", resp.text)
        return []
    except Exception as e:
        print("❌ Other request error:", e)
        return []


# =========================================================
# 3. Save to MongoDB
# =========================================================
def save_to_mongo(articles):
    if not MONGO_URI:
        print("⚠️ No MONGO_URI set — skipping save.")
        return

    try:
        client = MongoClient(MONGO_URI)
        col = client["quant_data"]["news_fmp"]

        docs = []
        for a in articles:
            docs.append({
                "symbol": a.get("symbol"),
                "title": a.get("title"),
                "text": a.get("text"),
                "site": a.get("site"),
                "url": a.get("url"),
                "image": a.get("image"),
                "publishedDate": a.get("publishedDate"),
                "collectedAt": datetime.utcnow().isoformat(),
                "platform": "fmp"
            })

        if docs:
            col.insert_many(docs)
            print(f"Inserted {len(docs)} FMP articles into MongoDB")

    except Exception as e:
        print("❌ MongoDB save error:", e)


# =========================================================
# 4. Main
# =========================================================
if __name__ == "__main__":
    articles = fetch_fmp_news()

    print("\nFetched:", len(articles))
    for i, a in enumerate(articles[:5], 1):
        print(f"{i}. {a.get('title')}")

    save_to_mongo(articles)