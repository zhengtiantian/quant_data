import os
import requests
from pymongo import MongoClient, UpdateOne
import json
import sys

# =====================================================
# 配置区域
# =====================================================
MONGO_URI = "mongodb://root:root@127.0.0.1:37018/"
DB_NAME = "quant_data"
COLLECTION = "stock_universe"
LANGCHAIN_API = "http://127.0.0.1:18000/api/ask"
TIMEOUT = 120
FIELD_NAME = "related_keywords"  # 新增字段名
BATCH_SIZE = 10  # 每10个写入一次数据库
# =====================================================


def check_mongo_connection(uri):
    """检查 MongoDB 连接"""
    try:
        client = MongoClient(uri, serverSelectionTimeoutMS=3000)
        client.admin.command("ping")
        print(f"✅ [MongoDB] Connected to {uri}")
        return client
    except Exception as e:
        print(f"❌ [MongoDB] Connection failed: {e}")
        sys.exit(1)


def generate_related_keywords(company_name):
    """调用 LangChain API，让模型生成严格 JSON 数组格式的关键词字符串"""
    prompt = f"""
    You are a financial market intelligence AI.
    Given the company name "{company_name}", list **15 to 20** related products, services, brands, technologies, or economic factors that can significantly affect its stock price.

    Output a **pure JSON array of strings only**, no explanations, no objects, no key-value pairs, no Markdown formatting.

    Example:
    Input: Apple
    Output: ["iPhone", "iPad", "MacBook", "App Store", "Apple Watch", "iCloud", "iOS", "MacOS", "Vision Pro", "Apple TV+", "M3 Chip"]

    Now generate for: {company_name}
    """

    try:
        resp = requests.post(
            LANGCHAIN_API,
            headers={"Content-Type": "application/json"},
            json={"question": prompt},
            timeout=TIMEOUT,
        )
        if resp.status_code == 200:
            answer = resp.json().get("answer", "").strip()
            # 尝试解析成 JSON
            try:
                arr = json.loads(answer)
                if isinstance(arr, list):
                    return arr
            except json.JSONDecodeError:
                cleaned = answer.replace("```json", "").replace("```", "").strip()
                try:
                    arr = json.loads(cleaned)
                    if isinstance(arr, list):
                        return arr
                except Exception:
                    pass
            return [answer]
        else:
            return [f"[Error] HTTP {resp.status_code}"]
    except Exception as e:
        return [f"[Exception] {e}"]


def main():
    print("🚀 Starting keyword generation via LangChain local model...\n")

    client = check_mongo_connection(MONGO_URI)
    db = client[DB_NAME]
    col = db[COLLECTION]

    companies = list(col.find({}, {"_id": 1, "name": 1}))
    print(f"📊 Found {len(companies)} companies in {DB_NAME}.{COLLECTION}\n")

    ops = []
    for i, doc in enumerate(companies, 1):
        name = doc.get("name")
        if not name:
            continue

        print(f"🧠 [{i}/{len(companies)}] Generating keywords for: {name}")
        keywords = generate_related_keywords(name)

        print(f"🔹 {name} related keywords:")
        print(json.dumps(keywords, indent=2, ensure_ascii=False))
        print("-" * 80)

        # 更新操作
        ops.append(
            UpdateOne(
                {"_id": doc["_id"]},
                {"$set": {FIELD_NAME: keywords}},
            )
        )

        if len(ops) >= BATCH_SIZE:
            result = col.bulk_write(ops)
            print(f"💾 Wrote {len(ops)} updates to MongoDB.\n")
            ops = []

    # 写入剩余部分
    if ops:
        result = col.bulk_write(ops)
        print(f"💾 Wrote final {len(ops)} updates to MongoDB.\n")

    client.close()
    print("✅ Finished.")


if __name__ == "__main__":
    main()
