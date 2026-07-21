import os
import requests
from datetime import datetime, UTC
from pathlib import Path
from dotenv import load_dotenv
from pymongo import MongoClient, errors

CURRENT = Path(__file__).resolve()
ROOT = CURRENT.parents[3]
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

# ====== A/B classification keywords ======

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

# ====== Save (unified schema!) ======

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
        try:
            result = col.insert_many(docs, ordered=False)
            print("Inserted:", len(result.inserted_ids))
        except errors.BulkWriteError as exc:
            details = getattr(exc, "details", {}) or {}
            inserted = int(details.get("nInserted", 0) or 0)
            write_errors = details.get("writeErrors", [])
            non_dup_errors = [err for err in write_errors if err.get("code") != 11000]
            duplicate_count = len(write_errors) - len(non_dup_errors)
            print(f"Inserted: {inserted} (duplicates skipped: {duplicate_count})")
            if non_dup_errors:
                raise

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
