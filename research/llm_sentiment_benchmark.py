#!/usr/bin/env python3
"""Benchmark LLM sentiment enrichment across models.

First run fetches N stratified articles and saves them to a fixture file.
Subsequent runs reuse the same articles so models are compared fairly.

Usage:
  # Run current LM Studio model and record results
  SLM_API_URL=http://192.168.31.226:1234/v1 \
  SLM_MODEL=google/gemma-4-e4b \
  LOCAL_MONGO_URI="mongodb://root:root@127.0.0.1:37018/" \
  .venv311/bin/python research/llm_sentiment_benchmark.py

  # Compare all recorded results
  .venv311/bin/python research/llm_sentiment_benchmark.py --compare
"""

from __future__ import annotations

import json
import os
import re
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import requests
from dotenv import load_dotenv
from pymongo import MongoClient

CURRENT = Path(__file__).resolve()
ROOT = CURRENT.parents[1]
load_dotenv(ROOT / ".env", override=False)

MONGO_URI = os.getenv("LOCAL_MONGO_URI", "mongodb://root:root@127.0.0.1:37018/")
DB_NAME = "quant_data"
COLLECTION = "news_articles_company_matched_v2"

API_URL = os.getenv("SLM_API_URL", "http://192.168.31.226:1234/v1").rstrip("/")
MODEL_A = os.getenv("SLM_MODEL_A", "google/gemma-4-e4b")
MODEL_B = os.getenv("SLM_MODEL_B", "qwen/qwen3.5-9b")
LLM_TIMEOUT = int(os.getenv("ENRICH_LLM_TIMEOUT", "60"))
CONTENT_CHARS = int(os.getenv("ENRICH_CONTENT_CHARS", "500"))
DISAGREE_THRESHOLD = float(os.getenv("ENRICH_DISAGREE_THRESHOLD", "0.3"))

SAMPLES_PER_SYMBOL = int(os.getenv("BENCH_SAMPLES_PER_SYMBOL", "2"))
RANDOM_SEED = int(os.getenv("BENCH_SEED", "42"))

RESULTS_DIR = CURRENT.parent / "benchmark_results"
FIXTURE_FILE = RESULTS_DIR / "_fixture_articles.json"

VALID_EVENT_TYPES = frozenset(
    {"earnings", "product", "MA", "regulation", "macro", "competition", "other"}
)
VALID_STRENGTHS = frozenset({"high", "medium", "low"})

PROMPT = """\
You are analyzing financial news for an investor holding {symbol} ({name}) stock.

Title: {title}
Content: {content}

From a {symbol} stock investor's perspective, assess the direct financial impact.

Choose event_type by these definitions (pick the FIRST that matches):
- MA:          {symbol} acquires or is acquired — "acquires", "acquisition", "merger", \
"buyout", "takeover", "joint venture". IMPORTANT: if the title contains "acquires" or \
"acquisition", always use MA.
- earnings:    ANY of these → use earnings:
  * revenue / profit / EPS / quarterly results / guidance / outlook
  * analyst upgrades or downgrades (e.g. "upgrades to Buy", "downgrades to Sell", "raises price target")
  * pricing changes by {symbol} (e.g. "raises prices", "price increase", "price cut", "price hike")
  * subscriber / user count changes
  * earnings call transcript
- product:     new product launch, hardware announcement, software release, feature update \
(NOT a price change — price changes are earnings)
- regulation:  lawsuit, fine, settlement, government ban, export restriction, patent dispute, recall
- macro:       broad market trends, economic data, Fed/rates, tariffs, industry-wide reports \
NOT directly about {symbol}
- competition: competitor wins/loses, market share shift, direct rival news that impacts {symbol}
- other:       executive changes, brand/PR, or general articles with no direct financial impact

signal_strength rules:
- "high":   article is primarily about {symbol}'s own financials, strategy, or analyst rating
- "medium": article mentions {symbol} meaningfully but is not exclusively about it
- "low":    {symbol} is mentioned briefly, OR the article is primarily about another company \
or a general topic unrelated to {symbol}'s business

Sentiment calibration — use the FULL range:
- 0.0  = neutral / no clear financial impact (default for irrelevant or ambiguous articles)
- Do NOT give a positive score just because the article tone is not negative
- Negative events (bug admissions, lawsuits, fines, competitive losses) MUST score below 0
- Only give scores above +0.5 for clearly strong bullish catalysts (beat earnings, major win)
- Only give scores below -0.5 for clearly strong bearish events (major loss, serious recall)

Return exactly this JSON (no other text):
{{"sentiment": <float -1.0 very bearish to 1.0 very bullish>, "event_type": "<MA|earnings|product|regulation|macro|competition|other>", "signal_strength": "<high|medium|low>"}}"""

_EVENT_KEYWORDS: list[tuple[str, list[str]]] = [
    ("MA",          ["acqui", "merger", "takeover", "buyout", "joint venture"]),
    ("earnings",    ["earning", "eps", "revenue", "profit", "quarterly", "guidance",
                     "upgrade", "downgrade", "price target", "rating", "subscriber",
                     "price hike", "price increase", "price cut", "pricing",
                     "raises prices", "lowers prices"]),
    ("product",     ["product", "launch", "release", "iphone", "chip", "device",
                     "software", "hardware", "feature", "update", "roadmap", "beta"]),
    ("regulation",  ["regulat", "lawsuit", "settlement", "tax", "fine", "penalty",
                     "sec", "ftc", "court", "legal", "recall", "ban", "sanction",
                     "fbi", "patent", "export"]),
    ("macro",       ["fed", "interest rate", "inflation", "economy", "gdp",
                     "recession", "tariff", "market", "index", "shares open"]),
    ("competition", ["compet", "rival", "market share", "disrupt", "surpass",
                     "beats", "ahead of", "overtakes"]),
]

_MA_TITLE_RE = re.compile(
    r'\b(acqui[a-z]*|merger|takeover|buyout|acquires|acquired)\b', re.IGNORECASE
)
_EARNINGS_TITLE_RE = re.compile(
    r'\b(earnings call|earnings transcript|q[1-4] 20\d\d earnings'
    r'|upgrades? (?:from |to |\w+ to )'
    r'|downgrades? (?:from |to |\w+ to )'
    r'|raises? price target|cuts? price target'
    r'|raises? (?:its )?(?:annual |full.year )?(?:guidance|outlook|forecast)'
    r'|price (?:increase|hike|cut|raise)s?'
    r'|raising prices|lowering prices)\b',
    re.IGNORECASE
)


def _normalize_event_type(raw: str) -> str:
    t = raw.lower().strip()
    if t in VALID_EVENT_TYPES:
        return t
    for canonical, keywords in _EVENT_KEYWORDS:
        if any(k in t for k in keywords):
            return canonical
    return "other"


def _title_override(title: str, event_type: str) -> str:
    if event_type != "other":
        return event_type
    if _MA_TITLE_RE.search(title):
        return "MA"
    if _EARNINGS_TITLE_RE.search(title):
        return "earnings"
    return event_type


from concurrent.futures import ThreadPoolExecutor as _TPE

STRENGTH_RANK: dict[str, int] = {"low": 0, "medium": 1, "high": 2}


def call_llm_one(article: dict, model: str) -> tuple[dict | None, float]:
    """Call a single model. Returns (result_dict, latency_seconds)."""
    prompt = PROMPT.format(
        title=(article.get("title") or "")[:200],
        content=(article.get("content") or "")[:CONTENT_CHARS],
        symbol=article.get("symbol") or "this stock",
        name=article.get("name") or article.get("symbol") or "this company",
    )
    payload = {
        "messages": [
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": "{"},
        ],
        "max_tokens": 120,
        "temperature": 0.05,
        "stream": False,
        "model": model,
    }
    t0 = time.perf_counter()
    try:
        resp = requests.post(f"{API_URL}/chat/completions", json=payload, timeout=LLM_TIMEOUT)
        latency = time.perf_counter() - t0
        resp.raise_for_status()
        partial = resp.json()["choices"][0]["message"]["content"].strip()
        raw = "{" + partial if not partial.startswith("{") else partial
        m = re.search(r'\{[^{}]+\}', raw, re.DOTALL)
        if not m:
            return None, latency
        parsed = json.loads(m.group())
        sentiment = float(parsed.get("sentiment", 0.0))
        sentiment = round(max(-1.0, min(1.0, sentiment)), 4)
        event_type = _normalize_event_type(str(parsed.get("event_type", "other")))
        event_type = _title_override(article.get("title", ""), event_type)
        signal_strength = str(parsed.get("signal_strength", "low")).lower().strip()
        if signal_strength not in VALID_STRENGTHS:
            signal_strength = "low"
        return {"sentiment": sentiment, "event_type": event_type,
                "signal_strength": signal_strength}, latency
    except Exception:
        return None, time.perf_counter() - t0


def call_llm_both(article: dict) -> tuple[dict | None, float, dict | None, float]:
    """Call MODEL_A and MODEL_B in parallel. Returns (res_a, lat_a, res_b, lat_b)."""
    with _TPE(max_workers=2) as pool:
        fut_a = pool.submit(call_llm_one, article, MODEL_A)
        fut_b = pool.submit(call_llm_one, article, MODEL_B)
        res_a, lat_a = fut_a.result()
        res_b, lat_b = fut_b.result()
    return res_a, lat_a, res_b, lat_b


def merge_results(res_a: dict | None, res_b: dict | None,
                  title: str) -> dict | None:
    if res_a is None and res_b is None:
        return None
    if res_a is None:
        return {**res_b, "ensemble": "single_b"}
    if res_b is None:
        return {**res_a, "ensemble": "single_a"}

    def _sign(x: float) -> int:
        return 1 if x > 0 else (-1 if x < 0 else 0)

    sent_a, sent_b = res_a["sentiment"], res_b["sentiment"]
    event_match = res_a["event_type"] == res_b["event_type"]
    agreed = (_sign(sent_a) == _sign(sent_b)) and (abs(sent_a - sent_b) <= DISAGREE_THRESHOLD)

    event_type = _title_override(title, res_a["event_type"])
    strength = max(res_a["signal_strength"], res_b["signal_strength"],
                   key=lambda s: STRENGTH_RANK[s]) if event_match else res_a["signal_strength"]

    return {
        "sentiment": round((sent_a + sent_b) / 2, 4),
        "sentiment_a": sent_a,
        "sentiment_b": sent_b,
        "event_type": event_type,
        "event_type_b": res_b["event_type"] if not event_match else event_type,
        "signal_strength": strength,
        "ensemble": "agree" if agreed else "disagree",
    }


def fetch_fixture_articles() -> list[dict]:
    """Stratified sample: SAMPLES_PER_SYMBOL articles per symbol, with content."""
    import random
    client = MongoClient(MONGO_URI, serverSelectionTimeoutMS=5000)
    coll = client[DB_NAME][COLLECTION]

    symbols = coll.distinct("symbol")
    print(f"Found {len(symbols)} symbols, sampling {SAMPLES_PER_SYMBOL} each...")

    query = {
        "content": {"$exists": True, "$ne": ""},
        "$expr": {"$gt": [{"$strLenCP": {"$ifNull": ["$content", ""]}}, 100]},
    }
    projection = {"_id": 1, "title": 1, "content": 1, "symbol": 1, "name": 1}

    articles = []
    offset = RANDOM_SEED % 17  # deterministic offset for variety
    for sym in sorted(symbols):
        docs = list(
            coll.find({"symbol": sym, **query}, projection)
            .sort("_id", 1)
            .skip(offset)
            .limit(SAMPLES_PER_SYMBOL)
        )
        for doc in docs:
            articles.append({
                "id": str(doc["_id"]),
                "symbol": doc.get("symbol", ""),
                "name": doc.get("name", ""),
                "title": doc.get("title", ""),
                "content": (doc.get("content") or "")[:CONTENT_CHARS],
            })

    rng = random.Random(RANDOM_SEED)
    rng.shuffle(articles)
    return articles


def load_fixture() -> list[dict]:
    if FIXTURE_FILE.exists():
        with open(FIXTURE_FILE) as f:
            data = json.load(f)
        print(f"Loaded {len(data)} fixture articles from {FIXTURE_FILE.name}")
        return data
    print("No fixture file found — fetching from MongoDB...")
    articles = fetch_fixture_articles()
    RESULTS_DIR.mkdir(exist_ok=True)
    with open(FIXTURE_FILE, "w") as f:
        json.dump(articles, f, indent=2, ensure_ascii=False)
    print(f"Saved {len(articles)} articles to {FIXTURE_FILE.name}")
    return articles


def result_file(model_name: str) -> Path:
    safe = re.sub(r'[^a-zA-Z0-9._-]', '_', model_name)
    return RESULTS_DIR / f"llm_sentiment_{safe}.json"


def run_benchmark(articles: list[dict]) -> None:
    ensemble_name = f"ensemble_{MODEL_A.split('/')[-1]}_{MODEL_B.split('/')[-1]}"
    out_path = result_file(ensemble_name)
    results = []
    errors = 0
    lat_a_list: list[float] = []
    lat_b_list: list[float] = []

    print(f"\nEnsemble: {MODEL_A}  +  {MODEL_B}")
    print(f"Articles: {len(articles)}  |  disagree_threshold={DISAGREE_THRESHOLD}")
    print("-" * 80)

    for i, art in enumerate(articles, 1):
        res_a, lat_a, res_b, lat_b = call_llm_both(art)
        lat_a_list.append(lat_a)
        lat_b_list.append(lat_b)
        result = merge_results(res_a, res_b, art.get("title", ""))

        if result is None:
            errors += 1
            row = {"error": True, "lat_a": round(lat_a, 3), "lat_b": round(lat_b, 3)}
        else:
            row = {**result, "lat_a": round(lat_a, 3), "lat_b": round(lat_b, 3)}

        row["id"] = art["id"]
        row["symbol"] = art["symbol"]
        row["title"] = art["title"][:80]
        results.append(row)

        if result is None:
            status = "ERROR (both)"
        else:
            a_str = f"{res_a['sentiment']:+.2f}" if res_a else "ERR"
            b_str = f"{res_b['sentiment']:+.2f}" if res_b else "ERR"
            tag = "✓" if result.get("ensemble") == "agree" else "✗"
            status = (
                f"A={a_str} B={b_str} {tag}  "
                f"→ {result['sentiment']:+.2f}/{result['event_type']:<12}"
                f"/{result['signal_strength']}"
            )
        print(f"[{i:3d}/{len(articles)}] {art['symbol']:<6} {status}  "
              f"({lat_a:.2f}s+{lat_b:.2f}s)")

    ok = len(results) - errors
    agree_count = sum(1 for r in results if r.get("ensemble") == "agree")
    avg_a = sum(lat_a_list) / len(lat_a_list) if lat_a_list else 0
    avg_b = sum(lat_b_list) / len(lat_b_list) if lat_b_list else 0
    p95_a = sorted(lat_a_list)[int(len(lat_a_list) * 0.95)] if lat_a_list else 0

    summary = {
        "model": ensemble_name,
        "model_a": MODEL_A,
        "model_b": MODEL_B,
        "run_at": datetime.now(timezone.utc).isoformat(),
        "total": len(articles),
        "ok": ok,
        "errors": errors,
        "agree": agree_count,
        "disagree": ok - agree_count,
        "agree_pct": round(agree_count / ok * 100, 1) if ok else 0,
        "avg_latency_a": round(avg_a, 3),
        "avg_latency_b": round(avg_b, 3),
        "p95_latency_a": round(p95_a, 3),
        "results": results,
    }

    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print(f"\nDone: {ok}/{len(articles)} ok | agree={agree_count} ({summary['agree_pct']}%)"
          f" | avg A={avg_a:.2f}s B={avg_b:.2f}s")
    print(f"Results saved to {out_path.name}")


def compare() -> None:
    files = sorted(RESULTS_DIR.glob("llm_sentiment_*.json"))
    if not files:
        print("No benchmark results found.")
        return

    all_data = {}
    for f in files:
        with open(f) as fp:
            data = json.load(fp)
        all_data[data["model"]] = data

    models = list(all_data.keys())
    print(f"\n{'='*80}")
    print(f"SENTIMENT BENCHMARK COMPARISON  ({len(models)} models)")
    print(f"{'='*80}")

    # Summary stats
    print(f"\n{'Model':<35} {'OK':>5} {'Err':>5} {'AvgLat':>8} {'P95Lat':>8}")
    print("-" * 65)
    for m, d in all_data.items():
        print(f"{m:<35} {d['ok']:>5} {d['errors']:>5} {d['avg_latency_s']:>7.2f}s {d['p95_latency_s']:>7.2f}s")

    # Per-article comparison
    if len(models) < 2:
        return

    # Load fixture for titles
    fixture = {a["id"]: a for a in load_fixture()} if FIXTURE_FILE.exists() else {}

    # Build article-level table
    # Collect all article IDs in order
    base_results = {r["id"]: r for r in all_data[models[0]]["results"]}
    article_ids = [r["id"] for r in all_data[models[0]]["results"]]

    print(f"\n{'#':>3} {'Sym':>6}  {'Title':<40}", end="")
    for m in models:
        short = m.split("/")[-1][:16]
        print(f"  {short:<20}", end="")
    print()
    print("-" * (55 + 22 * len(models)))

    for i, aid in enumerate(article_ids, 1):
        art = fixture.get(aid, {})
        title = (art.get("title") or base_results.get(aid, {}).get("title", ""))[:38]
        sym = art.get("symbol") or base_results.get(aid, {}).get("symbol", "")
        print(f"{i:3d} {sym:>6}  {title:<40}", end="")
        for m in models:
            r_map = {r["id"]: r for r in all_data[m]["results"]}
            r = r_map.get(aid)
            if r is None or r.get("error"):
                cell = "ERROR"
            else:
                cell = f"{r['sentiment']:+.2f}/{r['event_type'][:6]}/{r['signal_strength'][0]}"
            print(f"  {cell:<20}", end="")
        print()


def main() -> None:
    if "--compare" in sys.argv:
        compare()
        return

    print(f"API: {API_URL}")
    print(f"Model A: {MODEL_A}")
    print(f"Model B: {MODEL_B}")

    articles = load_fixture()
    run_benchmark(articles)


if __name__ == "__main__":
    main()
