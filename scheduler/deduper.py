from pymongo import MongoClient

class Deduper:
    def __init__(self, mongo_uri):
        self.client = MongoClient(mongo_uri)
        self.col = self.client["quant_data"]["news_articles"]
        self.col.create_index({"url": 1}, unique=True, sparse=True)

    def save_articles(self, articles):
        inserted = 0
        for a in articles:
            try:
                self.col.insert_one(a)   # Duplicates are automatically skipped
                inserted += 1
            except:
                pass
        return inserted