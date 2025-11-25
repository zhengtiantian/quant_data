import os
import requests
from datetime import datetime, UTC
from pathlib import Path
from dotenv import load_dotenv
from pymongo import MongoClient

# Load .env
CURRENT = Path(__file__).resolve()
ROOT = CURRENT.parents[2]
GLOBAL_ENV = ROOT / ".env"
load_dotenv(GLOBAL_ENV, override=False)

MODULE_ENV = CURRENT.parent / ".env"
if MODULE_ENV.exists():
    load_dotenv(MODULE_ENV, override=True)

API_KEY = os.getenv("MARKETAUX_API_KEY")
MONGO_URI = os.getenv("MONGO_URI")
LANGUAGE = "en"
LIMIT = 50

if not API_KEY:
    raise RuntimeError("MARKETAUX_API_KEY missing in .env")

BASE_URL = "https://api.marketaux.com/v1/news/all"


# --- Fetch Free Plan News ---
def fetch_marketaux_news():
    params = {
        "api_token": API_KEY,
        "language": LANGUAGE,
        "limit": LIMIT
        # ⛔ Free plan CANNOT use: countries, industries, markets
    }

    print("Requesting MarketAux...")
    print("URL:", BASE_URL)
    print("Params:", params)

    try:
        resp = requests.get(BASE_URL, params=params)
        resp.raise_for_status()

        data = resp.json()
        return data.get("data", [])

    except Exception as e:
        print("❌ MarketAux request failed:", e)
        print("Response:", getattr(resp, "text", "N/A"))
        return []


# --- Save ---
def save_to_mongo(articles):
    if not MONGO_URI:
        print("Skipping MongoDB save (no URI)")
        return

    client = MongoClient(MONGO_URI)
    col = client["quant_data"]["news_marketaux"]

    docs = []
    for a in articles:
        docs.append({
            "source": a.get("source"),
            "title": a.get("title"),
            "description": a.get("description"),
            "url": a.get("url"),
            "published_at": a.get("published_at"),
            "image_url": a.get("image_url"),
            "collectedAt": datetime.now(UTC).isoformat(),
            "platform": "marketaux"
        })

    if docs:
        col.insert_many(docs)
        print(f"Inserted {len(docs)} MarketAux articles")


# --- Main ---
if __name__ == "__main__":
    articles = fetch_marketaux_news()
    print("Fetched:", len(articles))

    for i, a in enumerate(articles[:5], 1):
        print(f"{i}. {a.get('title')}")

    save_to_mongo(articles)