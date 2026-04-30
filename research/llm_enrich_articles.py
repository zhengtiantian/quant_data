#!/usr/bin/env python3
"""Stage 3.5.1 — LLM article enrichment (two-pass).

Pass A  — run MODEL_A (+ optional MODEL_A2 for round-robin) on all articles.
          Stores: llm_sentiment_a, llm_event_type_a, llm_signal_strength_a
Pass B  — run MODEL_B on all articles.
          Stores: llm_sentiment_b, llm_event_type_b, llm_signal_strength_b

Each pass is resumable: docs that already have the target field are skipped.
Passes A and B are independent and can run simultaneously.

Usage:
  # Pass A — two Gemma instances for load balancing
  SLM_MODEL_A=google/gemma-4-e4b \
  SLM_MODEL_A2=google/gemma-4-e4b:2 \
  ENRICH_PASS=A ENRICH_WORKERS=8 \
  LOCAL_MONGO_URI="mongodb://root:root@127.0.0.1:37018/" \
  .venv311/bin/python research/llm_enrich_articles.py

  # Pass B — Qwen
  SLM_MODEL_B=qwen/qwen3.5-9b \
  ENRICH_PASS=B ENRICH_WORKERS=8 \
  LOCAL_MONGO_URI="mongodb://root:root@127.0.0.1:37018/" \
  .venv311/bin/python research/llm_enrich_articles.py
"""

from __future__ import annotations

import itertools
import json
import logging
import os
import re
import threading
import time
from concurrent.futures import Future, ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path

import requests
from dotenv import load_dotenv
from pymongo import MongoClient, UpdateOne

CURRENT = Path(__file__).resolve()
ROOT = CURRENT.parents[1]
load_dotenv(ROOT / ".env", override=False)

UTC = timezone.utc

MONGO_URI = os.getenv("LOCAL_MONGO_URI", "mongodb://root:root@127.0.0.1:37018/")
DB_NAME = "quant_data"
COLLECTION = "news_articles_company_matched_v2"

API_URL = os.getenv("SLM_API_URL", "http://192.168.31.226:1234/v1").rstrip("/")
MODEL_A  = os.getenv("SLM_MODEL_A",  "google/gemma-4-e4b")
MODEL_A2 = os.getenv("SLM_MODEL_A2", "google/gemma-4-e4b:2")  # second Gemma instance
MODEL_B  = os.getenv("SLM_MODEL_B",  "qwen/qwen3.5-9b")

PASS          = os.getenv("ENRICH_PASS", "A")          # A | B | merge
WORKERS       = int(os.getenv("ENRICH_WORKERS",       "8"))
BATCH_SIZE    = int(os.getenv("ENRICH_BATCH_SIZE",    "200"))
PROGRESS_EVERY= int(os.getenv("ENRICH_PROGRESS_EVERY","500"))
CONTENT_CHARS = int(os.getenv("ENRICH_CONTENT_CHARS", "500"))
LLM_TIMEOUT   = int(os.getenv("ENRICH_LLM_TIMEOUT",  "30"))
LIMIT         = int(os.getenv("ENRICH_LIMIT",         "0"))   # 0 = all

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)
log = logging.getLogger(__name__)

VALID_EVENT_TYPES = frozenset(
    {"earnings", "product", "MA", "regulation", "macro", "competition", "other"}
)
VALID_STRENGTHS = frozenset({"high", "medium", "low"})
STRENGTH_RANK: dict[str, int] = {"low": 0, "medium": 1, "high": 2}

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


def call_llm(title: str, content: str, symbol: str = "", name: str = "",
             model: str = "") -> dict | None:
    prompt = PROMPT.format(
        title=(title or "")[:200],
        content=(content or "")[:CONTENT_CHARS],
        symbol=symbol or "this stock",
        name=name or symbol or "this company",
    )
    payload: dict = {
        "messages": [
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": "{"},
        ],
        "max_tokens": 120,
        "temperature": 0.05,
        "stream": False,
    }
    if model:
        payload["model"] = model

    try:
        resp = requests.post(f"{API_URL}/chat/completions", json=payload, timeout=LLM_TIMEOUT)
        resp.raise_for_status()
        partial = resp.json()["choices"][0]["message"]["content"].strip()
        raw = "{" + partial if not partial.startswith("{") else partial

        m = re.search(r'\{[^{}]+\}', raw, re.DOTALL)
        if not m:
            return None
        parsed = json.loads(m.group())

        sentiment = float(parsed.get("sentiment", 0.0))
        sentiment = round(max(-1.0, min(1.0, sentiment)), 4)
        event_type = _normalize_event_type(str(parsed.get("event_type", "other")))
        event_type = _title_override(title, event_type)
        signal_strength = str(parsed.get("signal_strength", "low")).lower().strip()
        if signal_strength not in VALID_STRENGTHS:
            signal_strength = "low"

        return {"sentiment": sentiment, "event_type": event_type,
                "signal_strength": signal_strength}
    except Exception:
        return None


# ── Pass A/B: single-model with round-robin across available instances ─────────

def _build_model_cycle(pass_name: str):
    if pass_name == "A":
        models = [m for m in [MODEL_A, MODEL_A2] if m]
    else:
        models = [MODEL_B]
    it = itertools.cycle(models)
    lock = threading.Lock()

    def next_model() -> str:
        with lock:
            return next(it)

    return next_model, models


def enrich_one(doc: dict, next_model_fn) -> tuple[UpdateOne | None, str, float]:
    """Single-model pass. Returns (op, status_line, wall_seconds)."""
    suffix = PASS.lower()
    title = doc.get("title", "")
    model = next_model_fn()
    t0 = time.time()
    result = call_llm(title, doc.get("content", ""),
                      doc.get("symbol", ""), doc.get("name", ""), model)
    latency = time.time() - t0

    if result is None:
        return None, f"ERROR ({model.split('/')[-1]})", latency

    status = (f"{result['sentiment']:+.2f}"
              f"/{result['event_type']}/{result['signal_strength']}"
              f" [{model.split(':')[-1] if ':' in model else '1'}]")
    op = UpdateOne({"_id": doc["_id"]}, {"$set": {
        f"llm_sentiment_{suffix}":      result["sentiment"],
        f"llm_event_type_{suffix}":     result["event_type"],
        f"llm_signal_strength_{suffix}": result["signal_strength"],
        f"llm_enriched_{suffix}_at":    datetime.now(UTC),
    }})
    return op, status, latency


# ── Shared runner ─────────────────────────────────────────────────────────────

def run_loop(coll, query, projection, submit_fn) -> None:
    total = coll.count_documents(query)
    if LIMIT:
        total = min(total, LIMIT)
    log.info("Articles to process: %d", total)
    if total == 0:
        log.info("Nothing to do.")
        return

    cursor = coll.find(query, projection)
    if LIMIT:
        cursor = cursor.limit(LIMIT)

    done = errors = 0
    pending_ops: list[UpdateOne] = []
    start = time.time()

    with ThreadPoolExecutor(max_workers=WORKERS) as pool:
        future_map: dict[Future, str] = {}  # future → symbol

        def flush(wait_all: bool = False) -> None:
            nonlocal done, errors
            ready = (list(as_completed(future_map)) if wait_all
                     else [f for f in future_map if f.done()])
            for f in ready:
                op, status, latency = f.result()
                if op:
                    pending_ops.append(op)
                    done += 1
                else:
                    errors += 1
                del future_map[f]

                n = done + errors
                if n % PROGRESS_EVERY == 0 or n == total:
                    elapsed = time.time() - start
                    rate = done / elapsed if elapsed > 0 else 0
                    eta_h = (total - n) / rate / 3600 if rate > 0 else 0
                    eta_str = f"{eta_h:.1f}h" if eta_h >= 1 else f"{eta_h*60:.0f}min"
                    elapsed_str = (f"{elapsed/3600:.1f}h" if elapsed >= 3600
                                   else f"{elapsed/60:.1f}min")
                    log.info(
                        "[%d/%d] done=%d errors=%d | elapsed=%s | %.1f art/s | ETA %s",
                        n, total, done, errors, elapsed_str, rate, eta_str,
                    )

        for doc in cursor:
            future_map[pool.submit(submit_fn, doc)] = doc.get("symbol", "")

            if len(future_map) >= WORKERS * 3:
                flush()
            if len(pending_ops) >= BATCH_SIZE:
                coll.bulk_write(pending_ops, ordered=False)
                pending_ops.clear()

        flush(wait_all=True)

    if pending_ops:
        coll.bulk_write(pending_ops, ordered=False)

    elapsed = time.time() - start
    rate = done / elapsed if elapsed > 0 else 0
    log.info(
        "Finished. %d/%d done | errors %d | %.1fs | %.1f art/s",
        done, total, errors, elapsed, rate,
    )


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    client = MongoClient(MONGO_URI, serverSelectionTimeoutMS=5000)
    client.admin.command("ping")
    coll = client[DB_NAME][COLLECTION]

    if PASS in ("A", "B"):
        suffix = PASS.lower()
        next_model_fn, models = _build_model_cycle(PASS)
        log.info("PASS=%s | models: %s | workers: %d", PASS, models, WORKERS)

        query = {f"llm_sentiment_{suffix}": {"$exists": False}}
        projection = {"_id": 1, "title": 1, "content": 1, "symbol": 1, "name": 1}

        run_loop(coll, query, projection,
                 lambda doc: enrich_one(doc, next_model_fn))
    else:
        log.error("Unknown ENRICH_PASS=%r. Use A or B.", PASS)


if __name__ == "__main__":
    main()
