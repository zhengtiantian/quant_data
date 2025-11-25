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

ROOT = CURRENT.parents[2]
GLOBAL_ENV = ROOT / ".env"
if GLOBAL_ENV.exists():
    load_dotenv(GLOBAL_ENV, override=False)
    print(f"Loaded GLOBAL .env: {GLOBAL_ENV}")

MODULE_ENV = CURRENT.parent / ".env"
if MODULE_ENV.exists():
    load_dotenv(MODULE_ENV, override=True)
    print(f"Loaded MODULE .env: {MODULE_ENV}")

# =========================================================
# 2. Config
# =========================================================

FINNHUB_API_KEY = os.getenv("FINNHUB_API_KEY")
LIMIT = int(os.getenv("FINNHUB_NEWS_LIMIT", 50))
LANGUAGE = os.getenv("NEWS_LANGUAGE", "en")
MONGO_URI = os.getenv("MONGO_URI")

if not FINNHUB_API_KEY:
    raise RuntimeError("❌ FINNHUB_API_KEY missing in .env")

# —— 预留未来公司级别新闻功能（暂不启用） ——
# 例如 ["AAPL", "MSFT", "NVDA", "AMD", "INTC", "TSLA"]
SYMBOLS = []   # 未来你要用时直接改这里即可


# =========================================================
# 3. AB 分类过滤器
# =========================================================

STRONG_KEYWORDS = [  # A 类：强影响股价
    "earnings", "EPS", "revenue", "guidance", "profit",
    "loss", "acquisition", "merger",
    "price target", "downgrade", "upgrade",
    "sec", "investigation", "lawsuit", "hack",
    "restructuring", "layoff",
    "CEO", "CFO", "leadership",
    "FOMC", "Federal Reserve", "inflation", "CPI", "PPI",
    "Treasury", "yields", "interest rate",
    "AI", "data center", "chip", "semiconductor",
]

MID_KEYWORDS = [  # B 类：次强影响
    "economy", "macro", "consumer",
    "tech", "trend", "industry",
    "EV", "energy", "housing",
    "cloud", "crypto", "forex"
]


def classify_news(article):
    """Return 'A' for strong-related, 'B' for mid-related, else None."""

    text = (article.get("headline") or "").lower() + " " + (article.get("summary") or "").lower()

    # Strong A
    if any(k.lower() in text for k in STRONG_KEYWORDS):
        return "A"

    # Mid B
    if any(k.lower() in text for k in MID_KEYWORDS):
        return "B"

    return None


# =========================================================
# 4. Fetch Finnhub AB news
# =========================================================

def fetch_finnhub_news():
    """Fetch AB-classified market news from Finnhub."""

    BASE_URL = "https://finnhub.io/api/v1/news"

    params = {
        "category": "general",   # 覆盖范围最大的一类
        "minId": 0,
        "token": FINNHUB_API_KEY
    }

    print(f"\nRequest URL: {BASE_URL}")
    print("Params:", params)

    try:
        resp = requests.get(BASE_URL, params=params, headers={"User-Agent": "QuantNewsCollector"})
        resp.raise_for_status()
        data = resp.json()
    except Exception as e:
        print("❌ Finnhub request failed:", e)
        print("Response:", getattr(resp, "text", "N/A"))
        return []

    # 限制数量
    data = data[:LIMIT]

    # 过滤出 AB 类新闻
    ab_articles = []
    for a in data:
        cls = classify_news(a)
        if cls:
            a["class"] = cls
            ab_articles.append(a)

    return ab_articles


# =========================================================
# 5. Save to MongoDB
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
                "class": a.get("class"),        # A or B
                "source": {"platform": "finnhub"},
                "headline": a.get("headline"),
                "summary": a.get("summary"),
                "url": a.get("url"),
                "image": a.get("image"),
                "datetime": a.get("datetime"),
                "publishedAt": datetime.fromtimestamp(a.get("datetime"), UTC).isoformat()
                if a.get("datetime") else None,
                "collectedAt": datetime.now(UTC).isoformat(),
                "language": LANGUAGE,
                "meta": {
                    "collector": "finnhub.collector",
                    "version": "1.1.0"
                }
            })

        if docs:
            col.insert_many(docs)
            print(f"Inserted {len(docs)} Finnhub AB-classified articles into MongoDB")

    except Exception as e:
        print("❌ MongoDB save error:", e)


# =========================================================
# 6. Main
# =========================================================

if __name__ == "__main__":
    articles = fetch_finnhub_news()

    print(f"\nFetched AB articles: {len(articles)}")
    for i, a in enumerate(articles[:5], 1):
        print(f"{i}. ({a.get('class')}) {a.get('headline')}")

    save_to_mongo(articles)