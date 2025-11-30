import os
import requests
from datetime import datetime, UTC
from pathlib import Path
from bs4 import BeautifulSoup
from dotenv import load_dotenv
from pymongo import MongoClient


# =========================================================
# 1. Load env
# =========================================================

CURRENT = Path(__file__).resolve()
ROOT = CURRENT.parents[2]
GLOBAL_ENV = ROOT / ".env"

load_dotenv(GLOBAL_ENV, override=False)
print(f"Loaded GLOBAL .env: {GLOBAL_ENV}")

MONGO_URI = os.getenv("MONGO_URI")

TARGET_URL = "https://finance.yahoo.com/topic/stock-market-news/"


# =========================================================
# 2. 美股关键词 + A/B 级分类器
# =========================================================

US_STOCK_KEYWORDS = [
    "nasdaq", "dow jones", "s&p", "s&p500", "sp500",
    "federal reserve", "fed", "wall street",
    "us stocks", "u.s. stocks", "stock market",
    "rate cut", "rate hike", "inflation", "treasury yield",
    "bond yield",
    "microsoft", "google", "alphabet", "amazon",
    "meta", "facebook", "apple", "tesla", "nvidia", "amd", "intel",
    "goldman", "jpmorgan", "citigroup"
]

A_LEVEL_KEYWORDS = [
    "federal reserve", "fed", "rate hike", "rate cut",
    "inflation", "cpi", "ppi", "treasury yield",
    "market crash", "selloff", "plunge", "rally",
    "earnings", "forecast", "guidance", "revenue",
    "lawsuit", "sec investigation", "antitrust",
    "nvidia", "apple", "google", "tesla"
]


def is_us_stock_news(text: str):
    text = text.lower()
    return any(k in text for k in US_STOCK_KEYWORDS)


def classify_impact(text: str):
    text = text.lower()
    for k in A_LEVEL_KEYWORDS:
        if k in text:
            return "A"
    return "B"


# =========================================================
# 3. Yahoo 爬虫（HTML fallback 版）
# =========================================================

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

    raw_articles = []

    for a in soup.find_all("a"):
        h3 = a.find("h3")
        if not h3:
            continue

        title = h3.get_text(strip=True)
        href = a.get("href")

        if not title or not href:
            continue

        if href.startswith("/"):
            full_url = "https://finance.yahoo.com" + href
        else:
            full_url = href

        raw_articles.append({
            "title": title,
            "url": full_url,
            "publishedAt": None,
            "source": "Yahoo Finance"
        })

    # =========================================================
    # 过滤美股新闻 + A/B 分类
    # =========================================================
    filtered = []
    for art in raw_articles:
        text = art["title"]

        if not is_us_stock_news(text):
            continue

        art["impact"] = classify_impact(text)
        filtered.append(art)

    return filtered


# =========================================================
# 4. Save to MongoDB
# =========================================================

def save_to_mongo(articles):
    if not MONGO_URI:
        print("⚠️ No MONGO_URI. Skipping save.")
        return

    client = MongoClient(MONGO_URI)
    col = client["quant_data"]["news_articles"]

    docs = []
    for a in articles:
        docs.append({
            "source": "yahoo",
            "title": a["title"],
            "url": a["url"],
            "impact": a["impact"],        # ★ 加入 A/B 分类
            "publishedAt": a["publishedAt"],
            "collectedAt": datetime.now(UTC).isoformat(),
            "language": "en",
            "meta": {"collector": "yahoo.html", "version": "2.0"}
        })

    if docs:
        col.insert_many(docs)
        print(f"Inserted {len(docs)} Yahoo news into MongoDB")


# =========================================================
# 5. Main
# =========================================================

if __name__ == "__main__":
    articles = fetch_yahoo_news()

    print(f"\nFetched (US stock filtered): {len(articles)}")
    for i, a in enumerate(articles[:10], 1):
        print(f"{i}. ({a['impact']}) {a['title']}")

    save_to_mongo(articles)