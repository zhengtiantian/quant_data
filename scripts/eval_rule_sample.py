from pymongo import MongoClient

from collectors.news.gdelt.special_rules import RuleManager
from collectors.news.gdelt.special_rules.slm_filter import get_slm_stats


def main() -> None:
    client = MongoClient("mongodb://root:root@mongo6:27017/", serverSelectionTimeoutMS=5000)
    client.admin.command("ping")
    col = client["quant_data"]["news_articles"]

    docs = list(
        col.aggregate(
            [
                {"$match": {"symbol": {"$exists": True}, "title": {"$exists": True}}},
                {"$sample": {"size": 50}},
                {"$project": {"_id": 0, "symbol": 1, "name": 1, "date": 1, "title": 1, "url": 1, "content": 1}},
            ]
        )
    )

    rm = RuleManager()
    results = []
    for i, doc in enumerate(docs, 1):
        before = get_slm_stats()
        matched = rm.should_include(doc.get("symbol"), doc)
        after = get_slm_stats()
        slm_used = (after["req"] > before["req"]) or (after["cache"] > before["cache"])
        content = (doc.get("content") or "").replace("\n", " ").replace("\r", " ").strip()
        preview = content[:220] + ("..." if len(content) > 220 else "")
        results.append(
            {
                "idx": i,
                "symbol": doc.get("symbol"),
                "name": doc.get("name"),
                "date": doc.get("date"),
                "title": doc.get("title"),
                "matched": matched,
                "slm_used": slm_used,
                "url": doc.get("url"),
                "preview": preview,
            }
        )

    matched_n = sum(1 for r in results if r["matched"])
    slm_n = sum(1 for r in results if r["slm_used"])
    print(f"SUMMARY total=50 matched={matched_n} rejected={50 - matched_n} slm_used={slm_n}")
    for r in results:
        print("---")
        print(
            f"{r['idx']}. {r['symbol']} | {r['name']} | date={r['date']} | "
            f"matched={r['matched']} | slm_used={r['slm_used']}"
        )
        print(f"title: {r['title']}")
        print(f"body: {r['preview']}")
        print(f"url: {r['url']}")


if __name__ == "__main__":
    main()
