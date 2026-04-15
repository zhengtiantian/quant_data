#!/usr/bin/env python3
import argparse
import csv
import json
import os
import random
import time
from dataclasses import dataclass
from typing import Dict, List

import requests
from pymongo import MongoClient

from news_collectors.gdelt.special_rules.slm_filter import SLMFilter


@dataclass
class Article:
    symbol: str
    name: str
    title: str
    content: str
    url: str
    date: str


def fetch_articles(sample_size: int, seed: int) -> List[Article]:
    uri = os.getenv("MONGO_URI", "mongodb://root:root@127.0.0.1:37018/")
    if "mongo6:27017" in uri:
        uri = "mongodb://root:root@127.0.0.1:37018/"
    client = MongoClient(uri, serverSelectionTimeoutMS=5000)
    client.admin.command("ping")
    col = client["quant_data"]["news_articles"]

    docs = list(
        col.aggregate(
            [
                {"$match": {"symbol": {"$exists": True}, "title": {"$exists": True}}},
                {"$sample": {"size": sample_size}},
                {"$project": {"_id": 0, "symbol": 1, "name": 1, "title": 1, "content": 1, "url": 1, "date": 1}},
            ]
        )
    )
    # Keep fixed order so both models see the same sequence.
    rnd = random.Random(seed)
    rnd.shuffle(docs)
    return [
        Article(
            symbol=d.get("symbol") or "",
            name=d.get("name") or "",
            title=d.get("title") or "",
            content=d.get("content") or "",
            url=d.get("url") or "",
            date=d.get("date") or "",
        )
        for d in docs
    ]


def read_articles_from_csv(path: str, sample_size: int) -> List[Article]:
    articles: List[Article] = []
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            company = row.get("company", "")
            if " | " in company:
                symbol, name = company.split(" | ", 1)
            else:
                symbol, name = company, ""
            articles.append(
                Article(
                    symbol=symbol,
                    name=name,
                    title=row.get("title", "") or "",
                    content=row.get("body", "") or "",
                    url=row.get("url", "") or "",
                    date=row.get("date", "") or "",
                )
            )
    return articles[:sample_size]


def ensure_model_loaded(base_url: str, model: str) -> None:
    resp = requests.get(f"{base_url.rstrip('/')}/models", timeout=10)
    resp.raise_for_status()
    model_ids = {m["id"] for m in resp.json().get("data", [])}
    if model not in model_ids:
        raise RuntimeError(f"Model not loaded in LM Studio: {model}. Available={sorted(model_ids)}")


def benchmark_model(base_url: str, model: str, articles: List[Article], progress_every: int = 5) -> Dict:
    ensure_model_loaded(base_url, model)
    filt = SLMFilter(api_url=base_url, model=model, enabled=True)
    results = []
    started = time.perf_counter()
    print(f"[bench] model={model} total_articles={len(articles)}", flush=True)
    for idx, article in enumerate(articles, 1):
        t0 = time.perf_counter()
        matched = filt.is_relevant(article.symbol, article.name, article.title, article.content)
        elapsed = time.perf_counter() - t0
        results.append(
            {
                "idx": idx,
                "symbol": article.symbol,
                "company": article.name,
                "date": article.date,
                "title": article.title,
                "body_preview": article.content[:220].replace("\n", " ").replace("\r", " "),
                "url": article.url,
                "matched": matched,
                "latency_s": round(elapsed, 4),
            }
        )
        if idx % progress_every == 0 or idx == len(articles):
            print(
                f"[bench] model={model} progress={idx}/{len(articles)} "
                f"avg_latency={round((time.perf_counter()-started)/idx, 4)}s/article"
            , flush=True)
    total = time.perf_counter() - started
    matched = sum(1 for r in results if r["matched"])
    return {
        "model": model,
        "total_articles": len(results),
        "matched": matched,
        "rejected": len(results) - matched,
        "total_time_s": round(total, 3),
        "avg_latency_s": round(total / len(results), 4) if results else 0.0,
        "results": results,
    }


def collect_disagreements(qwen: Dict, gemma: Dict) -> List[Dict]:
    q_by_idx = {r["idx"]: r for r in qwen["results"]}
    g_by_idx = {r["idx"]: r for r in gemma["results"]}
    disagreements = []
    for idx in sorted(q_by_idx):
        if q_by_idx[idx]["matched"] != g_by_idx[idx]["matched"]:
            disagreements.append(
                {
                    "idx": idx,
                    "symbol": q_by_idx[idx]["symbol"],
                    "company": q_by_idx[idx]["company"],
                    "title": q_by_idx[idx]["title"],
                    "body_preview": q_by_idx[idx]["body_preview"],
                    "qwen_match": q_by_idx[idx]["matched"],
                    "gemma_match": g_by_idx[idx]["matched"],
                }
            )
    return disagreements


def write_csv(path: str, qwen: Dict, gemma: Dict) -> None:
    q_by_idx = {r["idx"]: r for r in qwen["results"]}
    g_by_idx = {r["idx"]: r for r in gemma["results"]}
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "idx",
                "symbol",
                "company",
                "date",
                "title",
                "body_preview",
                "qwen_match",
                "qwen_latency_s",
                "gemma_match",
                "gemma_latency_s",
                "url",
            ]
        )
        for idx in sorted(q_by_idx):
            qr = q_by_idx[idx]
            gr = g_by_idx[idx]
            writer.writerow(
                [
                    idx,
                    qr["symbol"],
                    qr["company"],
                    qr["date"],
                    qr["title"],
                    qr["body_preview"],
                    qr["matched"],
                    qr["latency_s"],
                    gr["matched"],
                    gr["latency_s"],
                    qr["url"],
                ]
            )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sample-size", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--base-url", default="http://127.0.0.1:1234/v1")
    parser.add_argument("--qwen-model", default="qwen3.5-4b")
    parser.add_argument("--gemma-model", default="google/gemma-4-e4b")
    parser.add_argument("--input-csv", default="")
    parser.add_argument("--progress-every", type=int, default=5)
    parser.add_argument("--csv", default="/Users/xiz/Quant_trade/quant_data/tmp_model_benchmark_50.csv")
    parser.add_argument("--json", default="/Users/xiz/Quant_trade/quant_data/tmp_model_benchmark_50.json")
    args = parser.parse_args()

    if args.input_csv:
        articles = read_articles_from_csv(args.input_csv, args.sample_size)
    else:
        articles = fetch_articles(args.sample_size, args.seed)
    qwen = benchmark_model(args.base_url, args.qwen_model, articles, args.progress_every)
    gemma = benchmark_model(args.base_url, args.gemma_model, articles, args.progress_every)
    disagreements = collect_disagreements(qwen, gemma)

    write_csv(args.csv, qwen, gemma)
    payload = {
        "summary": {
            "sample_size": args.sample_size,
            "seed": args.seed,
            "qwen": {k: v for k, v in qwen.items() if k != "results"},
            "gemma": {k: v for k, v in gemma.items() if k != "results"},
            "agreement": sum(
                1
                for i in range(args.sample_size)
                if qwen["results"][i]["matched"] == gemma["results"][i]["matched"]
            ),
            "disagreements": len(disagreements),
        },
        "disagreements": disagreements[:12],
    }
    with open(args.json, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

    print(json.dumps(payload, ensure_ascii=False, indent=2))
    print(f"CSV={args.csv}")
    print(f"JSON={args.json}")


if __name__ == "__main__":
    main()
