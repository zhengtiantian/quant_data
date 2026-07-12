import os
import requests
from datetime import datetime
from pathlib import Path
from dotenv import load_dotenv
from pymongo import MongoClient


# =========================================================
# 1. Load environment variables (correctly load quant_data/.env)
# =========================================================

CURRENT_FILE = Path(__file__).resolve()
PROJECT_ROOT = CURRENT_FILE.parents[2]              # .../quant_data/
GLOBAL_ENV = PROJECT_ROOT / ".env"                 # .../quant_data/.env

if GLOBAL_ENV.exists():
    load_dotenv(GLOBAL_ENV, override=False)
    print("Loaded GLOBAL env:", GLOBAL_ENV)
else:
    raise RuntimeError(f"Global .env NOT FOUND at: {GLOBAL_ENV}")


# =========================================================
# 2. Load current module .env (optional override)
# =========================================================

MODULE_ENV = CURRENT_FILE.parent / ".env"
if MODULE_ENV.exists():
    load_dotenv(MODULE_ENV, override=True)
    print("Loaded MODULE env:", MODULE_ENV)
else:
    print("No module env found:", MODULE_ENV)


# =========================================================
# 3. Read environment variables
# =========================================================

BASE_URL = os.getenv("GDELT_BASE_URL", "https://api.gdeltproject.org/api/v2/doc/doc")
QUERY = os.getenv("GDELT_QUERY", "(finance OR economy OR stock OR market) AND sourcecountry:US")
PAGE_SIZE = int(os.getenv("GDELT_PAGE_SIZE", 50))
MONGO_URI = os.getenv("MONGO_URI")

print("DEBUG MONGO_URI:", MONGO_URI)

if not MONGO_URI:
    raise RuntimeError("❌ ERROR: MONGO_URI missing from quant_data/.env")


# =========================================================
# 4. News fetch function
# =========================================================

def fetch_news(query=None):
    q = query or QUERY

    # Auto-wrap OR query in parentheses
    if " OR " in q and not (q.startswith("(") and q.endswith(")")):
        q = f"({q})"
    q = f"{q} AND sourcecountry:US AND -sourcecountry:VN"
    params = {
        "query": q,
        "mode": "ArtList",
        "maxrecords": PAGE_SIZE,
        "format": "JSON"
    }

    print("\n=== Fetching GDELT ===")
    print("URL:", BASE_URL)
    print("Params:", params)

    resp = requests.get(
        BASE_URL,
        params=params,
        headers={"User-Agent": "Mozilla/5.0"}
    )

    print("Status:", resp.status_code)
    print("Raw response (first 500 chars):\n", resp.text[:500])

    # Prevent JSON crash
    try:
        data = resp.json()
    except Exception as e:
        print("\n❌ JSON parse failed:", e)
        print("Response content (reprinted):\n", resp.text[:500])
        return []

    return data.get("articles", [])


# =========================================================
# 5. Save to MongoDB (optional)
# =========================================================

def save_to_mongo(articles):
    try:
        client = MongoClient(MONGO_URI)
        col = client["quant_data"]["news_articles"]

        docs = []
        for a in articles:
            docs.append({
                "source": {
                    "platform": "gdelt",
                    "name": a.get("domain"),
                },
                "title": a.get("title"),
                "description": a.get("sourcecountry"),
                "content": None,
                "url": a.get("url"),
                "publishedAt": a.get("seendate"),
                "collectedAt": datetime.utcnow().isoformat(),
                "language": a.get("language", "unknown"),
                "meta": {
                    "collector": "gdelt.collector",
                    "version": "1.0.0"
                }
            })

        if docs:
            col.insert_many(docs)
            print(f"Inserted {len(docs)} docs into MongoDB")

    except Exception as e:
        print("❌ MongoDB insert failed:", e)


# =========================================================
# 6. Main program
# =========================================================

if __name__ == "__main__":
    articles = fetch_news()

    print("\nTotal articles:", len(articles))

    if articles:
        print("\nSample articles:")
        for i, a in enumerate(articles[:5], 1):
            print(f"{i}. {a.get('title')}")