import os
import requests
from datetime import datetime
from pathlib import Path
from dotenv import load_dotenv
from pymongo import MongoClient

# =========================================================
# 1. 加载环境变量（全局 + 模块专属）
# =========================================================

# 先加载根目录下的 .env（全局配置）
root_env = Path(__file__).resolve().parents[2] / ".env"
if root_env.exists():
    load_dotenv(dotenv_path=root_env, override=False)
    print(f"✅ 已加载全局配置: {root_env}")
else:
    print(f"⚠️ 未找到全局 .env 文件: {root_env}")

# 再加载当前模块专属 .env（覆盖同名变量）
local_env = Path(__file__).parent / ".env"
if local_env.exists():
    load_dotenv(dotenv_path=local_env, override=True)
    print(f"✅ 已加载模块配置: {local_env}")
else:
    print(f"⚠️ 未找到模块 .env 文件: {local_env}")

# 打印调试信息
print("MONGO_URI =", os.getenv("MONGO_URI"))
print("NEWS_API_KEY =", os.getenv("NEWS_API_KEY"))

# =========================================================
# 2. 环境变量配置
# =========================================================
API_KEY = os.getenv("NEWS_API_KEY")
BASE_URL = os.getenv("NEWS_API_BASE_URL", "https://newsapi.org/v2/everything")
DEFAULT_QUERY = os.getenv("NEWS_DEFAULT_QUERY", "finance")
LANGUAGE = os.getenv("NEWS_LANGUAGE", "en")
PAGE_SIZE = int(os.getenv("NEWS_PAGE_SIZE", 50))
MONGO_URI = os.getenv("MONGO_URI")

if not MONGO_URI:
    raise RuntimeError("❌ 未能从全局 .env 读取 MONGO_URI，请检查 quant_data/.env 文件")

# =========================================================
# 3. 获取新闻函数
# =========================================================
def fetch_news(query: str = None):
    """从 NewsAPI 获取新闻"""
    params = {
        "q": query or DEFAULT_QUERY,
        "language": LANGUAGE,
        "pageSize": PAGE_SIZE,
        "apiKey": API_KEY
    }

    response = requests.get(BASE_URL, params=params)
    if response.status_code != 200:
        print(f"❌ 请求失败: {response.status_code} - {response.text}")
        return []

    data = response.json()
    articles = data.get("articles", [])
    print(f"✅ 获取到 {len(articles)} 篇新闻")
    return articles

# =========================================================
# 4. 保存到 MongoDB
# =========================================================
def save_to_mongo(articles):
    """将新闻存入 MongoDB"""
    try:
        client = MongoClient(MONGO_URI)
        db = client["quant_data"]
        collection = db["news_articles"]

        docs = []
        for a in articles:
            docs.append({
                "source": {
                    "platform": "newsapi",
                    "name": a.get("source", {}).get("name")
                },
                "title": a.get("title"),
                "description": a.get("description"),
                "content": a.get("content"),
                "url": a.get("url"),
                "publishedAt": a.get("publishedAt"),
                "collectedAt": datetime.utcnow().isoformat(),
                "language": a.get("language", LANGUAGE),
                "meta": {
                    "collector": "newsapi.client",
                    "version": "1.0.0"
                }
            })

        if docs:
            collection.insert_many(docs)
            print(f"✅ 已写入 {len(docs)} 条新闻到 MongoDB")

    except Exception as e:
        print("❌ 写入 MongoDB 失败:", e)

# =========================================================
# 5. 主程序入口
# =========================================================
if __name__ == "__main__":
    news = fetch_news()
    if news:
        save_to_mongo(news)
        for i, article in enumerate(news[:5], 1):
            print(f"{i}. {article['title']}")