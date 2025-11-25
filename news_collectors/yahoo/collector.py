import os
import requests
from datetime import datetime, UTC
from pathlib import Path
from bs4 import BeautifulSoup
from dotenv import load_dotenv
from pymongo import MongoClient

CURRENT = Path(__file__).resolve()
ROOT = CURRENT.parents[2]
GLOBAL_ENV = ROOT / ".env"

load_dotenv(GLOBAL_ENV, override=False)
print(f"Loaded GLOBAL .env: {GLOBAL_ENV}")

MONGO_URI = os.getenv("MONGO_URI")

TARGET_URL = "https://finance.yahoo.com/topic/stock-market-news/"


def fetch_yahoo_news():
    print(f"\nRequesting Yahoo Finance: {TARGET_URL}")

    headers = {
        "User-Agent": "Mozilla/5.0",
        "Accept-Language": "en-US,en;q=0.9"
    }

    resp = requests.get(TARGET_URL, headers=headers)
    if resp.status_code != 200:
        print(f"❌ HTTP Error {resp.status_code}")
        return []

    soup = BeautifulSoup(resp.text, "html.parser")

    articles = []

    # Yahoo uses many <a> tags for articles
    for a in soup.find_all("a"):
        h3 = a.find("h3")
        if not h3:
            continue

        title = h3.get_text(strip=True)
        href = a.get("href")

        if not href or not title:
            continue

        if href.startswith("/"):
            full_url = "https://finance.yahoo.com" + href
        else:
            full_url = href

        articles.append({
            "title": title,
            "url": full_url,
            "publishedAt": None,  # Yahoo HTML does not contain time
            "source": "Yahoo Finance"
        })

    return articles


def save_to_mongo(articles):
    if not MONGO_URI:
        print("⚠️ No MONGO_URI. Skipping save.")
        return

    client = MongoClient(MONGO_URI)
    col = client["quant_data"]["news_yahoo"]

    docs = []
    for a in articles:
        docs.append({
            "source": "yahoo",
            "title": a["title"],
            "url": a["url"],
            "publishedAt": a["publishedAt"],
            "collectedAt": datetime.now(UTC).isoformat(),
            "language": "en",
            "meta": {"collector": "yahoo.html", "version": "2.0"}
        })

    if docs:
        col.insert_many(docs)
        print(f"Inserted {len(docs)} Yahoo news into MongoDB")


if __name__ == "__main__":
    articles = fetch_yahoo_news()

    print(f"\nFetched: {len(articles)}")
    for i, a in enumerate(articles[:10], 1):
        print(f"{i}. {a['title']}")

    save_to_mongo(articles)