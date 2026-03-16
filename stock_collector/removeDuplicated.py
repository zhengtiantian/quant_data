import re
from pymongo import MongoClient

# =====================================================
# MongoDB 配置
# =====================================================
MONGO_URI = "mongodb://root:root@127.0.0.1:37018/"
DB_NAME = "quant_data"
COLLECTION = "stock_universe"
# =====================================================


def connect_mongo():
    client = MongoClient(MONGO_URI)
    db = client[DB_NAME]
    return db[COLLECTION]


def clean_stock_universe():
    col = connect_mongo()

    print("🚀 Starting stock_universe cleanup...\n")

    # 1️⃣ 检查 name 中是否包含股票代码（如含大写字母组合 2~5 位）
    pattern = re.compile(r"\b[A-Z]{2,5}\b")
    count_removed_name_symbol = 0

    for doc in col.find({}, {"_id": 1, "name": 1, "symbol": 1}):
        name = doc.get("name", "")
        symbol = doc.get("symbol", "")
        if pattern.search(name) and symbol in name:
            col.delete_one({"_id": doc["_id"]})
            count_removed_name_symbol += 1
            print(f"❌ Deleted record with code in name: {name}")

    print(f"✅ Removed {count_removed_name_symbol} records with code in name.\n")

    # 2️⃣ 检查重复 symbol
    symbols = {}
    duplicate_symbol_count = 0
    for doc in col.find({}, {"_id": 1, "symbol": 1}):
        sym = doc.get("symbol")
        if sym:
            if sym in symbols:
                col.delete_one({"_id": doc["_id"]})
                duplicate_symbol_count += 1
                print(f"❌ Deleted duplicate symbol: {sym}")
            else:
                symbols[sym] = doc["_id"]

    print(f"✅ Removed {duplicate_symbol_count} duplicate symbol records.\n")

    # 3️⃣ 检查重复 name
    names = {}
    duplicate_name_count = 0
    for doc in col.find({}, {"_id": 1, "name": 1}):
        name = doc.get("name")
        if name:
            if name in names:
                col.delete_one({"_id": doc["_id"]})
                duplicate_name_count += 1
                print(f"❌ Deleted duplicate name: {name}")
            else:
                names[name] = doc["_id"]

    print(f"✅ Removed {duplicate_name_count} duplicate name records.\n")

    print("🎯 Cleanup finished.")


if __name__ == "__main__":
    clean_stock_universe()
