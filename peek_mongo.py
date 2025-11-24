from pymongo import MongoClient
import os
from dotenv import load_dotenv
from pathlib import Path
import json

# Load global .env
ROOT = Path(__file__).resolve().parents[0]
ENV = ROOT / ".env"
load_dotenv(ENV, override=False)

MONGO_URI = os.getenv("MONGO_URI")
client = MongoClient(MONGO_URI)

col = client["quant_data"]["news_articles"]

print("Total documents:", col.count_documents({}))

print("\n=== Sample documents ===")
for doc in col.find().limit(10):
    print(json.dumps(doc, indent=2, default=str))
    print("-" * 80)