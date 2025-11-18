import os
import requests
from datetime import datetime
from pathlib import Path
from dotenv import load_dotenv
from pymongo import MongoClient

# =========================================================
# 1. 加载环境变量（强制加载项目根目录 + 覆盖模块 .env）
# =========================================================

# 项目根目录：quant_data （父级的父级）
PROJECT_ROOT = Path(__file__).resolve().parents[2]
GLOBAL_ENV = PROJECT_ROOT / ".env"

# 加载项目根目录的 .env（不能 override）
load_dotenv(GLOBAL_ENV, override=False)
print(f"Loaded GLOBAL .env: {GLOBAL_ENV}")

# 加载当前模块下的 .env（能 override）
MODULE_ENV = Path(__file__).parent / ".env"
if MODULE_ENV.exists():
    load_dotenv(MODULE_ENV, override=True)
    print(f"Loaded MODULE .env: {MODULE_ENV}")
else:
    print(f"No module .env found at: {MODULE_ENV}")

# 调试输出
print("DEBUG MONGO_URI:", os.getenv("MONGO_URI"))
print("DEBUG GDELT_BASE_URL:", os.getenv("GDELT_BASE_URL"))

# =========================================================
# 2. 环境变量
# =========================================================
BASE_URL = os.getenv("GDELT_BASE_URL", "https://api.gdeltproject.org/api/v2/doc/doc")
QUERY = os.getenv("NEWS_DEFAULT_QUERY", "Wall Street")
PAGE_SIZE = int(os.getenv("GDELT_PAGE_SIZE", 50))
MONGO_URI = os.getenv("MONGO_URI")

if not MONGO_URI:
    raise RuntimeError("❌ ERROR: 未读取到 MONGO_URI，请检查 quant_data/.env 文件")

# =========================================================
# 3. 请求 GDELT
# =========================================================
def fetch_news(query: str = None):
    q = query or QUERY

    if " OR " in q and not (q.startswith("(") and q.endswith(")")):
        q = f"({q})"

    params = {
        "query": q,
        "mode": "ArtList",
        "maxrecords": PAGE_SIZE,
        "format": "JSON"
    }

    resp = requests.get(BASE_URL, params=params, headers={"User-Agent": "Mozilla/5.0"})
    try:
        data = resp.json()
    except:
        return []

    print("RAW RESPONSE:", resp.text)
    articles = data.get("articles", [])

    # 关键过滤，只保留美国新闻
    us_articles = [a for a in articles if a.get("sourcecountry") == "US"]

    return us_articles

# =========================================================
# 4. 保存到 MongoDB
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
# 5. 主程序
# =========================================================
if __name__ == "__main__":
    articles = fetch_news()
    if not articles:
        print("No articles fetched.")
    else:
        print("\nSample articles:")
        for i, a in enumerate(articles[:5], 1):
            print(f"{i}. {a.get('title')}")

        # save_to_mongo(articles)  # 需要时开启