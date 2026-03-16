import os
import re
import json
import requests
from pymongo import MongoClient, UpdateOne

# =====================================================
# 配置区域
# =====================================================
MONGO_URI = "mongodb://root:root@127.0.0.1:37018/"
DB_NAME = "quant_data"
COLLECTION = "stock_universe"
FIELD_NAME = "related_keywords"
LANGCHAIN_API = "http://127.0.0.1:18000/api/ask"
TIMEOUT = 60
BATCH_SIZE = 10
# =====================================================


def check_mongo_connection(uri):
    """验证 MongoDB 是否可连接"""
    try:
        client = MongoClient(uri, serverSelectionTimeoutMS=3000)
        client.admin.command("ping")
        print(f"✅ [MongoDB] Connected to {uri}")
        return client
    except Exception as e:
        print(f"❌ [MongoDB] Connection failed: {e}")
        exit(1)


def contains_chinese(text):
    """检测字符串中是否包含中文"""
    return bool(re.search(r"[\u4e00-\u9fa5]", text))


def translate_keywords(keywords):
    """调用 LangChain 翻译关键词为英文"""
    joined = ", ".join(keywords)
    prompt = f"""
    You are a bilingual translator.
    Translate the following keywords into English, keep them concise.
    Output a **pure JSON array of English strings only**, no extra text.

    Input: {joined}
    Output:
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
            print(f"⚠️ [Warning] Model returned non-JSON, raw: {answer[:100]}")
            return keywords  # 保留原文
        else:
            print(f"❌ [HTTP {resp.status_code}] Translation failed.")
            return keywords
    except Exception as e:
        print(f"❌ [Error] translate_keywords: {e}")
        return keywords


def main():
    client = check_mongo_connection(MONGO_URI)
    db = client[DB_NAME]
    col = db[COLLECTION]

    print("🔍 Checking for Chinese keywords...\n")

    docs = list(col.find({}, {"_id": 1, "name": 1, FIELD_NAME: 1}))
    ops = []
    count_total, count_translated = 0, 0

    for doc in docs:
        name = doc.get("name")
        keywords = doc.get(FIELD_NAME, [])
        if not keywords or not isinstance(keywords, list):
            continue

        count_total += 1
        has_chinese = any(contains_chinese(k) for k in keywords)

        if has_chinese:
            print(f"🌏 [{name}] contains Chinese → Translating...")
            new_keywords = translate_keywords(keywords)
            print(f"✅ Translated: {new_keywords}\n")

            ops.append(UpdateOne({"_id": doc["_id"]}, {"$set": {FIELD_NAME: new_keywords}}))
            count_translated += 1

            if len(ops) >= BATCH_SIZE:
                col.bulk_write(ops)
                print(f"💾 Wrote {len(ops)} translations to MongoDB.\n")
                ops = []
        else:
            print(f"✅ [{name}] All keywords already English.\n")

    if ops:
        col.bulk_write(ops)
        print(f"💾 Wrote final {len(ops)} translations to MongoDB.\n")

    print(f"✨ Done. Checked {count_total} docs, translated {count_translated}.\n")
    client.close()


if __name__ == "__main__":
    main()
