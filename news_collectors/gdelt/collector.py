import os
import requests
from datetime import datetime
from pathlib import Path
from dotenv import load_dotenv, find_dotenv
from pymongo import MongoClient

# 全局 env
ENV_PATH = find_dotenv()
load_dotenv(ENV_PATH, override=False)
print(f"Loaded global env: {ENV_PATH}")

# 模块 env
local_env = Path(__file__).parent / ".env"
if local_env.exists():
    load_dotenv(local_env, override=True)
    print("Loaded module env:", local_env)

BASE_URL = os.getenv("GDELT_BASE_URL", "https://api.gdeltproject.org/api/v2/doc/doc")
QUERY = os.getenv("GDELT_QUERY", "finance OR economy OR stock OR market")
MONGO_URI = os.getenv("MONGO_URI")

if not MONGO_URI:
    raise RuntimeError("MONGO_URI missing from quant_data/.env")

def fetch_news(query=None):
    params = {
        "query": query or QUERY,
        "mode": "ArtList",
        "maxrecords": 50,
        "format": "JSON"
    }

    print("\n=== Fetching GDELT ===")
    print("URL:", BASE_URL)
    print("Params:", params)

    response = requests.get(
        BASE_URL,
        params=params,
        headers={"User-Agent": "Mozilla/5.0"}
    )

    print("Status:", response.status_code)
    print("Raw response (前 500 字符):\n", response.text[:500])

    # 防止 JSON 崩溃
    try:
        data = response.json()
    except Exception as e:
        print("\n❌ JSON 解析失败:", e)
        print("返回内容（再次打印）:\n", response.text[:500])
        return []

    return data.get("articles", [])

if __name__ == "__main__":
    articles = fetch_news()
    print("\n总文章数:", len(articles))