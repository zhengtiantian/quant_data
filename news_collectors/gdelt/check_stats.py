from pymongo import MongoClient
import pandas as pd
from datetime import datetime

MONGO_URI = "mongodb://root:root@localhost:37018/"
DB_NAME = "quant_data"
COLLECTION_NAME = "news_articles"

def check_stats():
    try:
        client = MongoClient(MONGO_URI, serverSelectionTimeoutMS=5000)
        client.server_info() # verify connection
        db = client[DB_NAME]
        col = db[COLLECTION_NAME]

        total_docs = col.count_documents({})
        print(f"📊 Total Documents: {total_docs}")

        # Data Quality Stats
        pipeline_quality = [
            {"$group": {"_id": "$data_quality", "count": {"$sum": 1}}}
        ]
        quality_results = list(col.aggregate(pipeline_quality))
        print("\n📈 Data Quality Distribution:")
        for res in quality_results:
            q = res['_id'] or "unknown"
            count = res['count']
            percentage = (count / total_docs * 100) if total_docs > 0 else 0
            print(f"  • {q}: {count} ({percentage:.2f}%)")

        # Symbol Stats (Top 10)
        pipeline_symbol = [
            {"$group": {"_id": "$symbol", "count": {"$sum": 1}}},
            {"$sort": {"count": -1}},
            {"$limit": 10}
        ]
        symbol_results = list(col.aggregate(pipeline_symbol))
        print("\n🏢 Top 10 Companies (Article Count):")
        for res in symbol_results:
            print(f"  • {res['_id']}: {res['count']} articles")

        # Sample check for quality
        print("\n🔍 Sampling 'full' data content...")
        samples = col.find({"data_quality": "full"}).limit(2)
        for i, s in enumerate(samples):
            print(f"  [{i+1}] Symbol: {s['symbol']} | Title: {s['title'][:70]}...")
            content_len = len(s.get('content', ''))
            print(f"      Content Length: {content_len} chars")

        print("\n🔍 Sampling 'title_only' data...")
        samples = col.find({"data_quality": {"$in": ["title_only", "url_only", "low_value"]}}).limit(2)
        for i, s in enumerate(samples):
            print(f"  [{i+1}] Symbol: {s['symbol']} | Title: {s['title'][:70]}...")
            note = s.get('note', 'N/A')
            print(f"      Note: {note}")
    except Exception as e:
        print(f"❌ Error connecting to MongoDB or executing query: {e}")

if __name__ == "__main__":
    check_stats()
