import os
import requests
from datetime import datetime, UTC
from pathlib import Path
from dotenv import load_dotenv
from pymongo import MongoClient

CURRENT = Path(__file__).resolve()
ROOT = CURRENT.parents[2]
GLOBAL_ENV = ROOT / ".env"

if GLOBAL_ENV.exists():
    load_dotenv(GLOBAL_ENV, override=False)
    print(f"Loaded GLOBAL .env: {GLOBAL_ENV}")

MODULE_ENV = CURRENT.parent / ".env"
if MODULE_ENV.exists():
    load_dotenv(MODULE_ENV, override=True)
    print(f"Loaded MODULE .env: {MODULE_ENV}")

FINNHUB_API_KEY = os.getenv("FINNHUB_API_KEY")
LIMIT = int(os.getenv("FINNHUB_NEWS_LIMIT", 50))
LANGUAGE = os.getenv("NEWS_LANGUAGE", "en")
MONGO_URI = os.getenv("MONGO_URI")

if not FINNHUB_API_KEY:
    raise RuntimeError("FINNHUB_API_KEY missing")

# ====== A/B 分类关键词 ======

STRONG = [
    "earnings","revenue","guidance","profit","loss",
    "acquisition","merger","price target","downgrade","upgrade",
    "sec","investigation","lawsuit","hack",
    "inflation","cpi","ppi","rate hike","rate cut",
    "federal reserve","treasury yield",
    "nvidia","apple","tesla","google"
]

WEAK = [
    "economy","macro","trend","industry",
    "housing","energy","crypto","cloud","ev"
]

def classify(text):
    t = text.lower()
    if any(x in t for x in STRONG):
        return "A"
    if any(x in t for x in WEAK):
        return "B"
    return None

# ====== Fetch ======

def fetch_finnhub():
    url = "https://finnhub.io/api/v1/news"
    params = {"category":"general", "token": FINNHUB_API_KEY}
    print("Request:", url)
    r = requests.get(url, params=params)
    r.raise_for_status()

    return r.json()[:LIMIT]

# ====== Save (统一 schema!) ======

def save_to_mongo(articles):
    client = MongoClient(MONGO_URI)
    col = client["quant_data"]["news_articles"]

    docs = []
    for a in articles:
        impact = a.get("impact")
        published = (
            datetime.fromtimestamp(a["datetime"], UTC).isoformat()
            if a.get("datetime") else None
        )

        docs.append({
            "source": {"platform": "finnhub"},
            "title": a.get("headline"),
            "description": a.get("summary"),
            "content": None,
            "url": a.get("url"),
            "impact": impact,
            "publishedAt": published,
            "collectedAt": datetime.now(UTC).isoformat(),
            "language": LANGUAGE,
            "meta": {"collector": "finnhub", "version": "1.2"}
        })

    if docs:
        col.insert_many(docs)
        print("Inserted:", len(docs))

# ====== Main ======

if __name__ == "__main__":
    raw = fetch_finnhub()

    filtered = []
    for a in raw:
        text = (a.get("headline") or "") + " " + (a.get("summary") or "")
        c = classify(text)
        if c:
            a["impact"] = c
            filtered.append(a)

    print("Fetched AB:", len(filtered))
    for i, a in enumerate(filtered[:5]):
        print(f"{i+1}. ({a['impact']}) {a['headline']}")

    save_to_mongo(filtered)