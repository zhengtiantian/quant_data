#!/usr/bin/env python3
"""Audit relevance and content quality of collected news."""

from __future__ import annotations

import os
from collections import defaultdict
from pathlib import Path
from urllib.parse import urlparse

from dotenv import load_dotenv
from pymongo import MongoClient


CURRENT = Path(__file__).resolve()
ROOT = CURRENT.parents[1]
GLOBAL_ENV = ROOT / ".env"
load_dotenv(GLOBAL_ENV, override=False)

MONGO_URI = os.getenv("LOCAL_MONGO_URI", "mongodb://root:root@127.0.0.1:37018/")
DB_NAME = os.getenv("FEATURE_DB_NAME", "quant_data")
COLLECTION = "news_articles"

AUDIT_SAMPLE_PER_SYMBOL = int(os.getenv("AUDIT_SAMPLE_PER_SYMBOL", "5"))
AMBIGUOUS_SYMBOLS = ["ARM", "META", "MU", "AMD", "DDOG", "AAPL", "MSFT", "AMZN", "GOOGL"]


def domain_from_url(url: str | None) -> str:
    if not url:
        return "unknown"
    try:
        return urlparse(url).netloc.lower() or "unknown"
    except Exception:
        return "unknown"


def print_header(title: str) -> None:
    print("\n" + "=" * 90)
    print(title)
    print("=" * 90)


def print_subheader(title: str) -> None:
    print("\n" + title)
    print("-" * 90)


def main() -> None:
    client = MongoClient(MONGO_URI)
    col = client[DB_NAME][COLLECTION]

    total = col.count_documents({})
    full_count = col.count_documents({"data_quality": "full"})
    title_only_count = col.count_documents({"data_quality": "title_only"})

    print_header("新闻有效性审计报告")
    print(f"总文章数: {total:,}")
    print(f"full: {full_count:,}")
    print(f"title_only: {title_only_count:,}")

    print_subheader("1. 结构与内容质量风险")
    short_full = col.count_documents({"data_quality": "full", "content_length": {"$lt": 200}})
    long_full = col.count_documents({"data_quality": "full", "content_length": {"$gt": 100000}})
    huge_full = col.count_documents({"data_quality": "full", "content_length": {"$gt": 1000000}})
    print(f"full 但 content_length < 200: {short_full:,}")
    print(f"full 但 content_length > 100000: {long_full:,}")
    print(f"full 但 content_length > 1000000: {huge_full:,}")

    print_subheader("2. title_only 原因分布")
    reason_stats = list(
        col.aggregate(
            [
                {"$match": {"data_quality": "title_only"}},
                {"$group": {"_id": "$note", "count": {"$sum": 1}}},
                {"$sort": {"count": -1}},
                {"$limit": 10},
            ]
        )
    )
    for row in reason_stats:
        reason = (row["_id"] or "N/A").replace("\n", " ")
        print(f"{row['count']:>8,} | {reason[:100]}")

    print_subheader("3. 重复风险")
    url_dupe = list(
        col.aggregate(
            [
                {"$match": {"url": {"$type": "string", "$ne": ""}}},
                {"$group": {"_id": "$url", "n": {"$sum": 1}}},
                {"$match": {"n": {"$gt": 1}}},
                {"$group": {"_id": None, "dup_groups": {"$sum": 1}, "dup_docs": {"$sum": "$n"}}},
            ]
        )
    )
    sym_title_dupe = list(
        col.aggregate(
            [
                {"$match": {"symbol": {"$type": "string", "$ne": ""}, "title": {"$type": "string", "$ne": ""}}},
                {"$group": {"_id": {"symbol": "$symbol", "title": "$title"}, "n": {"$sum": 1}}},
                {"$match": {"n": {"$gt": 1}}},
                {"$group": {"_id": None, "dup_groups": {"$sum": 1}, "dup_docs": {"$sum": "$n"}}},
            ]
        )
    )
    url_dupe = url_dupe[0] if url_dupe else {"dup_groups": 0, "dup_docs": 0}
    sym_title_dupe = sym_title_dupe[0] if sym_title_dupe else {"dup_groups": 0, "dup_docs": 0}
    print(f"URL 完全重复组: {url_dupe['dup_groups']:,}, 涉及文档: {url_dupe['dup_docs']:,}")
    print(f"symbol+title 重复组: {sym_title_dupe['dup_groups']:,}, 涉及文档: {sym_title_dupe['dup_docs']:,}")

    print_subheader("4. 域名质量概览（Top 15）")
    domain_rows = list(
        col.aggregate(
            [
                {
                    "$project": {
                        "domain": {
                            "$let": {
                                "vars": {"u": "$url"},
                                "in": {
                                    "$cond": [
                                        {"$and": [{"$ne": ["$$u", None]}, {"$ne": ["$$u", ""]}]},
                                        "$$u",
                                        "unknown",
                                    ]
                                },
                            }
                        },
                        "data_quality": 1,
                    }
                }
            ]
        )
    )
    domain_stats = defaultdict(lambda: {"total": 0, "full": 0, "title_only": 0})
    for row in domain_rows:
        domain = domain_from_url(row.get("domain"))
        quality = row.get("data_quality") or "unknown"
        domain_stats[domain]["total"] += 1
        if quality in ("full", "title_only"):
            domain_stats[domain][quality] += 1

    ranked_domains = sorted(domain_stats.items(), key=lambda kv: kv[1]["total"], reverse=True)[:15]
    print(f"{'domain':<40} {'total':>8} {'full%':>8} {'title%':>8}")
    for domain, stats in ranked_domains:
        total_rows = stats["total"]
        full_pct = stats["full"] / total_rows * 100 if total_rows else 0
        title_pct = stats["title_only"] / total_rows * 100 if total_rows else 0
        print(f"{domain[:40]:<40} {total_rows:>8,} {full_pct:>7.1f}% {title_pct:>7.1f}%")

    print_subheader("5. 歧义股票专项抽样")
    for symbol in AMBIGUOUS_SYMBOLS:
        print(f"\n[{symbol}]")
        samples = list(
            col.find(
                {"symbol": symbol},
                {"_id": 0, "title": 1, "url": 1, "data_quality": 1, "content_length": 1},
            )
            .sort("date", -1)
            .limit(AUDIT_SAMPLE_PER_SYMBOL)
        )
        if not samples:
            print("  无样本")
            continue
        for idx, doc in enumerate(samples, 1):
            print(
                f"  {idx}. ({doc.get('data_quality','N/A')}, len={doc.get('content_length',0)}) "
                f"{(doc.get('title') or 'N/A')[:90]}"
            )
            print(f"     {doc.get('url', 'N/A')[:120]}")

    print_subheader("6. 极端长正文样本")
    extreme_samples = list(
        col.find(
            {"content_length": {"$gt": 1000000}},
            {"_id": 0, "symbol": 1, "title": 1, "url": 1, "content_length": 1},
        ).limit(10)
    )
    for idx, doc in enumerate(extreme_samples, 1):
        print(
            f"  {idx}. [{doc.get('symbol','N/A')}] len={doc.get('content_length',0)} "
            f"{(doc.get('title') or 'N/A')[:90]}"
        )
        print(f"     {doc.get('url', 'N/A')[:120]}")

    print_subheader("7. 初步判断")
    print("结构完整性是好的，但研究有效性不是全量成立。")
    print("建议默认只把 full 且内容长度在合理范围内的新闻作为强信号。")
    print("title_only 作为弱信号单独使用，歧义股票需要继续专项抽样。")


if __name__ == "__main__":
    main()
