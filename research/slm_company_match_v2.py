#!/usr/bin/env python3
"""
Build news_articles_company_matched_v2 using SLM verification.

Pipeline:
  1. For each article in news_articles, verify current symbol via SLM.
  2. If no match, keyword-scan for other companies in the 40-stock universe,
     then SLM-confirm each candidate.
  3. Confirmed articles (same or reassigned symbol) → news_articles_company_matched_v2.
  4. Unmatched articles → discarded.

Usage:
  SLM_MAX_CONCURRENCY=32 \
  SLM_MODELS="qwen3.5-4b,qwen3.5-4b:2" \
  SLM_API_URL=http://192.168.31.226:1234/v1 \
  V2_WORKERS=32 \
  LOCAL_MONGO_URI="mongodb://root:root@127.0.0.1:37018/" \
  .venv/bin/python research/slm_company_match_v2.py
"""

from __future__ import annotations

import os
import re
import sys
import threading
import time
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from datetime import datetime, timezone
UTC = timezone.utc
from pathlib import Path
from typing import Any

from bson import ObjectId
from dotenv import load_dotenv
from pymongo import MongoClient, ReplaceOne

CURRENT = Path(__file__).resolve()
ROOT = CURRENT.parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
load_dotenv(ROOT / ".env", override=False)

from news_collectors.gdelt.special_rules.slm_filter import SLMFilter

MONGO_URI = os.getenv("LOCAL_MONGO_URI", "mongodb://root:root@127.0.0.1:37018/")
DB_NAME = os.getenv("FEATURE_DB_NAME", "quant_data")
SOURCE_COLLECTION = os.getenv("V2_SOURCE_COLLECTION", "news_articles")
TARGET_COLLECTION = os.getenv("V2_TARGET_COLLECTION", "news_articles_company_matched_v2")
PROGRESS_COLLECTION = os.getenv("V2_PROGRESS_COLLECTION", "company_match_jobs")

MATCH_VERSION = "v2"
MATCH_FIELD = "company_match_v2"
JOB_NAME = os.getenv("V2_JOB_NAME", f"{TARGET_COLLECTION}:{MATCH_VERSION}")
BATCH_SIZE = int(os.getenv("V2_BATCH_SIZE", "500"))
WORKERS = int(os.getenv("V2_WORKERS", os.getenv("SLM_MAX_CONCURRENCY", "4")))
PROGRESS_EVERY = int(os.getenv("V2_PROGRESS_EVERY", "500"))
LIMIT = int(os.getenv("V2_LIMIT", "0"))
FORCE = os.getenv("V2_FORCE", "false").lower() == "true"
RESUME = os.getenv("V2_RESUME", "true").lower() == "true"
SYMBOLS_FILTER = [s.strip().upper() for s in os.getenv("V2_SYMBOLS", "").split(",") if s.strip()]
SLM_MODELS = [m.strip() for m in os.getenv("SLM_MODELS", "").split(",") if m.strip()]
SCAN_OTHER = os.getenv("V2_SCAN_OTHER", "true").lower() == "true"
CONTENT_CHARS = int(os.getenv("V2_CONTENT_CHARS", "1500"))

# Multi-endpoint support: SLM_ENDPOINTS="url1|model1|workers1,url2|model2|workers2"
# e.g. "http://192.168.31.226:1234/v1|qwen3.5-4b|15,http://127.0.0.1:1234/v1|qwen3-4b|5"
def _parse_endpoints() -> list[tuple[str, str]]:
    raw = os.getenv("SLM_ENDPOINTS", "").strip()
    if not raw:
        return []
    assignments: list[tuple[str, str]] = []
    for entry in raw.split(","):
        parts = entry.strip().split("|")
        if len(parts) < 2:
            continue
        url = parts[0].strip().rstrip("/")
        model = parts[1].strip()
        workers = int(parts[2].strip()) if len(parts) >= 3 else 1
        assignments.extend([(url, model)] * workers)
    return assignments

_ENDPOINT_ASSIGNMENTS = _parse_endpoints()

# 103-stock universe: symbol → (company_name, scan_keywords)
COMPANY_UNIVERSE: dict[str, tuple[str, list[str]]] = {
    # ── Original 40 tech stocks ──────────────────────────────────────────────
    "AAPL":  ("Apple Inc.",                 ["Apple", "iPhone", "iPad", "MacBook", "Tim Cook"]),
    "GOOGL": ("Alphabet Inc.",              ["Google", "Alphabet", "YouTube", "Android", "DeepMind"]),
    "MSFT":  ("Microsoft Corporation",      ["Microsoft", "Windows", "Azure", "Xbox", "Satya Nadella"]),
    "TSLA":  ("Tesla, Inc.",                ["Tesla", "Elon Musk", "Model 3", "Model S", "Model Y", "Cybertruck"]),
    "AMZN":  ("Amazon.com Inc.",            ["Amazon", "AWS", "Jeff Bezos", "Andy Jassy"]),
    "NVDA":  ("NVIDIA Corporation",         ["NVIDIA", "GeForce", "Jensen Huang", "CUDA", "H100", "A100"]),
    "META":  ("Meta Platforms",             ["Meta Platforms", "Facebook", "Instagram", "WhatsApp", "Zuckerberg"]),
    "INTC":  ("Intel Corporation",          ["Intel Corporation", "Intel Corp", "Xeon processor", "Intel chip"]),
    "QCOM":  ("Qualcomm Inc.",              ["Qualcomm", "Snapdragon"]),
    "AMD":   ("Advanced Micro Devices",     ["Advanced Micro Devices", "Ryzen", "EPYC", "Radeon GPU"]),
    "ARM":   ("ARM Holdings",               ["ARM Holdings", "Arm Ltd", "Arm architecture"]),
    "AVGO":  ("Broadcom Inc.",              ["Broadcom"]),
    "MU":    ("Micron Technology",          ["Micron Technology", "Micron Memory"]),
    "DDOG":  ("Datadog Inc.",               ["Datadog"]),
    "TSM":   ("Taiwan Semiconductor",       ["TSMC", "Taiwan Semiconductor", "Morris Chang"]),
    "ASML":  ("ASML Holding",               ["ASML", "EUV lithography"]),
    "AMAT":  ("Applied Materials",          ["Applied Materials"]),
    "LRCX":  ("Lam Research",               ["Lam Research"]),
    "KLAC":  ("KLA Corporation",            ["KLA Corporation", "KLA Corp"]),
    "TXN":   ("Texas Instruments",          ["Texas Instruments"]),
    "ADI":   ("Analog Devices",             ["Analog Devices"]),
    "MCHP":  ("Microchip Technology",       ["Microchip Technology"]),
    "CRM":   ("Salesforce",                 ["Salesforce", "Marc Benioff"]),
    "NOW":   ("ServiceNow",                 ["ServiceNow"]),
    "ADBE":  ("Adobe Inc.",                 ["Adobe", "Photoshop", "Creative Cloud"]),
    "ORCL":  ("Oracle Corporation",         ["Oracle Corporation", "Larry Ellison"]),
    "PLTR":  ("Palantir Technologies",      ["Palantir"]),
    "SNOW":  ("Snowflake Inc.",             ["Snowflake Inc", "Snowflake data"]),
    "MDB":   ("MongoDB Inc.",               ["MongoDB"]),
    "PANW":  ("Palo Alto Networks",         ["Palo Alto Networks"]),
    "FTNT":  ("Fortinet",                   ["Fortinet"]),
    "CRWD":  ("CrowdStrike",                ["CrowdStrike"]),
    "NFLX":  ("Netflix",                    ["Netflix", "Reed Hastings"]),
    "UBER":  ("Uber Technologies",          ["Uber"]),
    "ABNB":  ("Airbnb Inc.",                ["Airbnb", "Brian Chesky"]),
    "CSCO":  ("Cisco Systems",              ["Cisco Systems", "Cisco Corp"]),
    "IBM":   ("IBM",                        ["IBM", "International Business Machines"]),
    "DELL":  ("Dell Technologies",          ["Dell Technologies", "Dell Computer"]),
    "SMCI":  ("Super Micro Computer",       ["Supermicro", "Super Micro Computer"]),
    "INTU":  ("Intuit Inc.",                ["Intuit", "TurboTax", "QuickBooks"]),
    # ── Storage expansion (2026-07-15) ───────────────────────────────────────
    "STX":   ("Seagate Technology",        ["Seagate", "Seagate Technology", "IronWolf", "BarraCuda", "Nytro SSD", "Dave Mosley"]),
    "WDC":   ("Western Digital",           ["Western Digital", "WD Blue", "WD Red", "SanDisk", "HGST", "Ultrastar", "David Goeckeler"]),
    "HXSCL": ("SK Hynix",                  ["SK Hynix", "Kwak Noh-jung", "SK hynix DRAM", "HBM3E", "SK Hynix memory"]),
    # ── NEW 60: SaaS / growth tech ───────────────────────────────────────────
    "SHOP":  ("Shopify",                    ["Shopify", "Tobi Lütke", "Shop Pay", "Shopify Plus", "Shopify Payments"]),
    "NET":   ("Cloudflare",                 ["Cloudflare", "Matthew Prince", "Cloudflare Workers", "DDoS protection"]),
    "ZS":    ("Zscaler",                    ["Zscaler", "Jay Chaudhry", "Zero Trust Exchange", "Zscaler Internet Access"]),
    "HUBS":  ("HubSpot",                    ["HubSpot", "Yamini Rangan", "HubSpot CRM"]),
    "WDAY":  ("Workday",                    ["Workday", "Carl Eschenbach", "Aneel Bhusri", "Workday HCM"]),
    "VEEV":  ("Veeva Systems",              ["Veeva", "Peter Gassner", "Veeva Systems", "Veeva Vault"]),
    "TEAM":  ("Atlassian",                  ["Atlassian", "Mike Cannon-Brookes", "Jira Software", "Confluence", "Trello"]),
    "TTD":   ("The Trade Desk",             ["The Trade Desk", "Jeff Green Trade Desk", "programmatic TV", "connected TV advertising"]),
    "OKTA":  ("Okta",                       ["Okta", "Todd McKinnon", "Auth0", "Okta Identity Engine"]),
    "APP":   ("AppLovin",                   ["AppLovin", "Adam Foroughi", "AppDiscovery", "AppLovin MAX"]),
    "RBLX":  ("Roblox Corporation",         ["Roblox", "David Baszucki", "Robux", "Roblox Studio"]),
    "COIN":  ("Coinbase",                   ["Coinbase", "Brian Armstrong", "Coinbase Pro", "Coinbase Wallet", "Base blockchain"]),
    "TWLO":  ("Twilio",                     ["Twilio", "Jeff Lawson", "SendGrid", "Twilio Segment"]),
    "DUOL":  ("Duolingo",                   ["Duolingo", "Luis von Ahn", "Duolingo Max", "Super Duolingo"]),
    "CFLT":  ("Confluent",                  ["Confluent", "Jay Kreps", "Confluent Cloud", "Apache Kafka"]),
    "GTLB":  ("GitLab",                     ["GitLab", "Sid Sijbrandij", "GitLab Duo", "GitLab CI"]),
    "MNDY":  ("monday.com",                 ["monday.com", "Roy Mann", "Eran Zinman", "monday WorkOS"]),
    "S":     ("SentinelOne",                ["SentinelOne", "Tomer Weingarten", "Singularity platform", "Purple AI"]),
    # ── NEW 60: Biotech / Healthcare ─────────────────────────────────────────
    "LLY":   ("Eli Lilly",                  ["Eli Lilly", "David Ricks", "Mounjaro", "tirzepatide", "Zepbound", "Kisunla"]),
    "JNJ":   ("Johnson & Johnson",          ["Johnson & Johnson", "Joaquin Duato", "Janssen", "Darzalex", "Stelara", "Kenvue"]),
    "AMGN":  ("Amgen",                      ["Amgen", "Robert Bradway", "Repatha", "MariTide", "Enbrel", "Otezla"]),
    "GILD":  ("Gilead Sciences",            ["Gilead Sciences", "Gilead", "Daniel O'Day", "Biktarvy", "Veklury", "remdesivir"]),
    "REGN":  ("Regeneron",                  ["Regeneron", "Leonard Schleifer", "Dupixent", "dupilumab", "EYLEA"]),
    "VRTX":  ("Vertex Pharmaceuticals",     ["Vertex Pharmaceuticals", "Reshma Kewalramani", "Trikafta", "Casgevy"]),
    "ISRG":  ("Intuitive Surgical",         ["Intuitive Surgical", "Gary Guthart", "da Vinci", "da Vinci surgical"]),
    "UNH":   ("UnitedHealth Group",         ["UnitedHealth", "UnitedHealthcare", "Andrew Witty", "Optum", "Change Healthcare"]),
    "MRNA":  ("Moderna",                    ["Moderna", "Stephane Bancel", "Spikevax", "mRNA-1273"]),
    "ABBV":  ("AbbVie",                     ["AbbVie", "Richard Gonzalez", "Humira", "Skyrizi", "Rinvoq", "Botox"]),
    "PFE":   ("Pfizer",                     ["Pfizer", "Albert Bourla", "Paxlovid", "Comirnaty", "Eliquis", "Ibrance"]),
    "MDT":   ("Medtronic",                  ["Medtronic", "Geoff Martha", "MiniMed", "Hugo robotic", "Micra pacemaker"]),
    "SYK":   ("Stryker",                    ["Stryker", "Kevin Lobo", "Mako robot", "LIFEPAK", "Stryker Corporation"]),
    "DXCM":  ("Dexcom",                     ["Dexcom", "Kevin Sayer", "G7 sensor", "G6 sensor", "continuous glucose monitor"]),
    "ILMN":  ("Illumina",                   ["Illumina", "Jacob Thaysen", "NovaSeq", "NextSeq", "Grail", "Galleri test"]),
    # ── NEW 60: Financials ───────────────────────────────────────────────────
    "V":     ("Visa Inc.",                  ["Visa", "Ryan McInerney", "Visa Direct", "VisaNet", "Visa payment"]),
    "MA":    ("Mastercard",                 ["Mastercard", "Michael Miebach", "Mastercard Send", "Vocalink"]),
    "PYPL":  ("PayPal",                     ["PayPal", "Alex Chriss", "Venmo", "Braintree", "PayPal Checkout"]),
    "GS":    ("Goldman Sachs",              ["Goldman Sachs", "David Solomon", "Marcus by Goldman"]),
    "JPM":   ("JPMorgan Chase",             ["JPMorgan", "JPMorgan Chase", "Jamie Dimon", "Chase Bank", "J.P. Morgan"]),
    "MS":    ("Morgan Stanley",             ["Morgan Stanley", "Ted Pick", "James Gorman", "E*Trade"]),
    "BLK":   ("BlackRock",                  ["BlackRock", "Larry Fink", "iShares", "Aladdin platform"]),
    "SCHW":  ("Charles Schwab",             ["Charles Schwab", "Schwab", "TD Ameritrade", "Rick Wurster"]),
    "AXP":   ("American Express",           ["American Express", "AmEx", "Stephen Squeri", "Amex Platinum"]),
    "COF":   ("Capital One",                ["Capital One", "Richard Fairbank", "Discover Financial"]),
    # ── NEW 60: Consumer / Media ─────────────────────────────────────────────
    "DIS":   ("Walt Disney Company",        ["Disney", "Walt Disney", "Bob Iger", "Disney+", "Hulu", "Pixar", "ESPN"]),
    "SNAP":  ("Snap Inc.",                  ["Snapchat", "Snap Inc", "Evan Spiegel", "Spectacles"]),
    "SPOT":  ("Spotify",                    ["Spotify", "Daniel Ek", "Spotify Wrapped", "Spotify Premium"]),
    "RDDT":  ("Reddit",                     ["Reddit", "Steve Huffman", "subreddit", "Reddit API"]),
    "PINS":  ("Pinterest",                  ["Pinterest", "Bill Ready", "shoppable pins", "Pinterest Lens"]),
    "NKE":   ("Nike",                       ["Nike", "Elliott Hill", "Air Jordan", "Jordan Brand", "Swoosh"]),
    "HD":    ("Home Depot",                 ["Home Depot", "Ted Decker", "SRS Distribution"]),
    "SBUX":  ("Starbucks",                  ["Starbucks", "Brian Niccol", "Howard Schultz", "Frappuccino", "Starbucks Rewards"]),
    "MCD":   ("McDonald's",                 ["McDonald's", "McDonalds", "Chris Kempczinski", "Big Mac", "Golden Arches", "McNuggets"]),
    "TGT":   ("Target Corporation",         ["Target Corporation", "Target stores", "Brian Cornell", "Target Circle", "Shipt"]),
    # ── NEW 60: Industrials / Defense ────────────────────────────────────────
    "CAT":   ("Caterpillar",                ["Caterpillar", "Jim Umpleby", "CAT equipment"]),
    "HON":   ("Honeywell",                  ["Honeywell", "Vimal Kapur", "Quantinuum", "Honeywell spinoff"]),
    "RTX":   ("RTX Corporation",            ["Raytheon", "RTX Corp", "Chris Calio", "Pratt & Whitney", "Collins Aerospace", "F135 engine"]),
    "LMT":   ("Lockheed Martin",            ["Lockheed Martin", "Jim Taiclet", "F-35", "PAC-3", "Sikorsky", "HIMARS"]),
    "GE":    ("GE Aerospace",               ["GE Aerospace", "GE Aviation", "General Electric", "Larry Culp", "LEAP engine", "GE9X"]),
    "DE":    ("Deere & Company",            ["John Deere", "Deere", "John May", "Deere & Co"]),
    "BA":    ("Boeing",                     ["Boeing", "Kelly Ortberg", "737 MAX", "787 Dreamliner", "777X", "Starliner"]),
}

_SCAN_PATTERNS: dict[str, re.Pattern] = {
    sym: re.compile(
        r'\b(' + '|'.join(re.escape(kw) for kw in sorted(kws, key=len, reverse=True)) + r')\b',
        re.IGNORECASE,
    )
    for sym, (_, kws) in COMPANY_UNIVERSE.items()
}

_thread_local = threading.local()
_thread_counter = 0
_thread_counter_lock = threading.Lock()


def now_iso() -> str:
    return datetime.now(UTC).isoformat()


def get_client() -> MongoClient:
    return MongoClient(MONGO_URI)


def get_slm() -> SLMFilter:
    slm = getattr(_thread_local, "slm", None)
    if slm is None:
        global _thread_counter
        with _thread_counter_lock:
            idx = _thread_counter
            _thread_counter += 1
        if _ENDPOINT_ASSIGNMENTS:
            url, model = _ENDPOINT_ASSIGNMENTS[idx % len(_ENDPOINT_ASSIGNMENTS)]
            slm = SLMFilter(api_url=url, model=model, enabled=True)
        elif SLM_MODELS:
            slm = SLMFilter(enabled=True, model=SLM_MODELS[idx % len(SLM_MODELS)])
        else:
            slm = SLMFilter(enabled=True)
        _thread_local.slm = slm
    return slm


def _scan_candidates(current_symbol: str, title: str, snippet: str) -> list[str]:
    text = f"{title} {snippet}"
    return [
        sym for sym, pat in _SCAN_PATTERNS.items()
        if sym != current_symbol and pat.search(text)
    ]


def _title_candidates(current_symbol: str, title: str) -> list[str]:
    """Return other symbols whose keywords appear in the title only."""
    return [
        sym for sym, pat in _SCAN_PATTERNS.items()
        if sym != current_symbol and pat.search(title)
    ]


def _has_direct_mention(symbol: str, title: str, content: str) -> bool:
    """True when the matched symbol's keywords appear explicitly in title or body."""
    pat = _SCAN_PATTERNS.get(symbol)
    if not pat:
        return False
    return bool(pat.search(title) or pat.search(content))


_THIN_BODY_MARKERS = (
    "subscribe", "sign up", "log in", "login", "error 451",
    "recommended videos", "unavailable in your location",
    "skip to main content", "skip to content",
    "we're sorry", "this website is unavailable",
    "loading your experience", "we value your privacy",
)


def _is_thin_body(content: str) -> bool:
    """True when body is too short or clearly a paywall/navigation page."""
    stripped = content.strip()
    if len(stripped) < 120:
        return True
    lower = stripped.lower()
    return len(stripped) < 400 and any(m in lower for m in _THIN_BODY_MARKERS)


def process_doc(doc: dict[str, Any]) -> dict[str, Any]:
    symbol = (doc.get("symbol") or "").strip().upper()
    title = (doc.get("title") or "").strip()
    content = (doc.get("content") or "")[:CONTENT_CHARS]
    # Use collector-stamped data_quality when available, fall back to heuristic
    thin_body = doc.get("data_quality") == "title_only" or _is_thin_body(content)

    if not symbol or not title:
        return {"_id": doc["_id"], "status": "skip", "symbol": symbol}

    company_name = COMPANY_UNIVERSE.get(symbol, (symbol, []))[0]
    slm = get_slm()

    # Step 1: verify current symbol with full content
    if slm.is_relevant(symbol, company_name, title, content):
        # Step 1b: if another company appears in the title, it may be the primary subject.
        # SLM-confirm each title candidate; first confirmed one wins (reassign).
        if SCAN_OTHER:
            for cand in _title_candidates(symbol, title):
                cand_name = COMPANY_UNIVERSE[cand][0]
                if slm.is_relevant(cand, cand_name, title, content):
                    return {"_id": doc["_id"], "status": "reassigned", "symbol": cand,
                            "original_symbol": symbol, "reassigned": True,
                            "title_only": False,
                            "direct_mention": _has_direct_mention(cand, title, content),
                            "model": slm.model, "doc": doc}
        return {"_id": doc["_id"], "status": "match", "symbol": symbol,
                "reassigned": False, "title_only": False,
                "direct_mention": _has_direct_mention(symbol, title, content),
                "model": slm.model, "doc": doc}

    # Step 2: scan for other candidates (body + title)
    if SCAN_OTHER:
        for cand in _scan_candidates(symbol, title, content[:500]):
            cand_name = COMPANY_UNIVERSE[cand][0]
            if slm.is_relevant(cand, cand_name, title, content):
                return {"_id": doc["_id"], "status": "reassigned", "symbol": cand,
                        "original_symbol": symbol, "reassigned": True,
                        "title_only": False,
                        "direct_mention": _has_direct_mention(cand, title, content),
                        "model": slm.model, "doc": doc}

    # Step 3: body is thin/empty — retry with title only
    if thin_body:
        if slm.is_relevant(symbol, company_name, title, ""):
            return {"_id": doc["_id"], "status": "match", "symbol": symbol,
                    "reassigned": False, "title_only": True,
                    "direct_mention": _has_direct_mention(symbol, title, ""),
                    "model": slm.model, "doc": doc}
        if SCAN_OTHER:
            for cand in _title_candidates(symbol, title):
                cand_name = COMPANY_UNIVERSE[cand][0]
                if slm.is_relevant(cand, cand_name, title, ""):
                    return {"_id": doc["_id"], "status": "reassigned", "symbol": cand,
                            "original_symbol": symbol, "reassigned": True,
                            "title_only": True,
                            "direct_mention": _has_direct_mention(cand, title, ""),
                            "model": slm.model, "doc": doc}

    return {"_id": doc["_id"], "status": "reject", "symbol": symbol}


def build_target_doc(row: dict[str, Any]) -> dict[str, Any]:
    src = dict(row["doc"])
    src["_id"] = row["_id"]
    src["raw_id"] = row["_id"]
    if row.get("reassigned"):
        src["symbol"] = row["symbol"]
        src["original_symbol"] = row.get("original_symbol")
        src["name"] = COMPANY_UNIVERSE.get(row["symbol"], (row["symbol"], []))[0]
    src[MATCH_FIELD] = {
        "matched": True,
        "reassigned": row.get("reassigned", False),
        "original_symbol": row.get("original_symbol"),
        "title_only": row.get("title_only", False),
        "direct_mention": row.get("direct_mention", False),
        "version": MATCH_VERSION,
        "engine": "slm",
        "model": row.get("model", ""),
        "scoredAt": now_iso(),
    }
    return src


def build_query(last_id: ObjectId | None = None) -> dict[str, Any]:
    query: dict[str, Any] = {
        "symbol": {"$exists": True, "$ne": None},
        "title": {"$exists": True, "$ne": None, "$ne": ""},
    }
    if last_id:
        query["_id"] = {"$gt": last_id}
    if SYMBOLS_FILTER:
        query["symbol"] = {"$in": SYMBOLS_FILTER}
    return query


def fetch_batch(col, last_id: ObjectId | None) -> list[dict[str, Any]]:
    return list(
        col.find(build_query(last_id))
        .sort("_id", 1)
        .limit(BATCH_SIZE)
    )


def save_batch(target_col, results: list[dict[str, Any]]) -> int:
    ops = [
        ReplaceOne({"_id": row["_id"]}, build_target_doc(row), upsert=True)
        for row in results if row["status"] in ("match", "reassigned")
    ]
    if not ops:
        return 0
    r = target_col.bulk_write(ops, ordered=False)
    return r.upserted_count + r.modified_count


def load_progress(col) -> dict[str, Any] | None:
    return col.find_one({"_id": JOB_NAME})


def save_progress(col, *, status, processed, matched, reassigned, rejected, title_only, last_id, extra=None):
    payload: dict[str, Any] = {
        "_id": JOB_NAME,
        "sourceCollection": SOURCE_COLLECTION,
        "targetCollection": TARGET_COLLECTION,
        "status": status,
        "processed": processed,
        "matched": matched,
        "reassigned": reassigned,
        "rejected": rejected,
        "titleOnly": title_only,
        "matchRate": round(matched / processed * 100, 2) if processed else 0.0,
        "lastId": last_id,
        "updatedAt": now_iso(),
    }
    if extra:
        payload.update(extra)
    col.update_one({"_id": JOB_NAME}, {"$set": payload}, upsert=True)


def run() -> None:
    client = get_client()
    source_col = client[DB_NAME][SOURCE_COLLECTION]
    target_col = client[DB_NAME][TARGET_COLLECTION]
    progress_col = client[DB_NAME][PROGRESS_COLLECTION]

    progress_doc = load_progress(progress_col) if RESUME and not FORCE else None
    last_id = progress_doc.get("lastId") if progress_doc else None
    processed = int(progress_doc.get("processed", 0)) if progress_doc else 0
    matched = int(progress_doc.get("matched", 0)) if progress_doc else 0
    reassigned = int(progress_doc.get("reassigned", 0)) if progress_doc else 0
    rejected = int(progress_doc.get("rejected", 0)) if progress_doc else 0
    title_only = int(progress_doc.get("titleOnly", 0)) if progress_doc else 0
    prior_processed = processed  # articles done before this session

    total_pending = source_col.count_documents(build_query(last_id))
    total_all = prior_processed + total_pending  # grand total across all sessions
    if _ENDPOINT_ASSIGNMENTS:
        from collections import Counter
        ep_counts = Counter(_ENDPOINT_ASSIGNMENTS)
        models_info = "  ".join(f"{m}@{u.split('//')[1].split('/')[0]}×{n}" for (u, m), n in ep_counts.items())
    elif SLM_MODELS:
        models_info = ",".join(SLM_MODELS)
    else:
        models_info = os.getenv("SLM_MODEL", "default")

    print("=== SLM Company Match v2 ===")
    print(f"source={SOURCE_COLLECTION}  target={TARGET_COLLECTION}")
    print(f"workers={WORKERS}  batch={BATCH_SIZE}  scan_other={SCAN_OTHER}")
    print(f"models={models_info}")
    print(f"pending={total_pending:,}  prior_processed={processed:,}")
    if progress_doc:
        print(f"resuming from last_id={last_id}")

    save_progress(progress_col, status="running", processed=processed,
                  matched=matched, reassigned=reassigned, rejected=rejected, title_only=title_only,
                  last_id=last_id,
                  extra={"startedAt": progress_doc.get("startedAt", now_iso()) if progress_doc else now_iso()})

    t_start = time.time()

    with ThreadPoolExecutor(max_workers=WORKERS, thread_name_prefix="slm-v2") as executor:
        while True:
            batch = fetch_batch(source_col, last_id)
            if not batch:
                break

            futures = {executor.submit(process_doc, doc): doc["_id"] for doc in batch}
            batch_results: list[dict[str, Any]] = []

            while futures:
                done, _ = wait(futures.keys(), return_when=FIRST_COMPLETED)
                for fut in done:
                    futures.pop(fut, None)
                    row = fut.result()
                    batch_results.append(row)
                    processed += 1
                    if row["status"] == "match":
                        matched += 1
                        if row.get("title_only"):
                            title_only += 1
                    elif row["status"] == "reassigned":
                        matched += 1
                        reassigned += 1
                        if row.get("title_only"):
                            title_only += 1
                    else:
                        rejected += 1

                    if processed % PROGRESS_EVERY == 0:
                        elapsed = time.time() - t_start
                        rate = processed / elapsed if elapsed > 0 else 0
                        session_done = processed - prior_processed
                        rate = session_done / elapsed if elapsed > 0 else 0
                        remaining = total_pending - session_done
                        eta_s = int(remaining / rate) if rate > 0 else 0
                        eta_str = f"{eta_s//3600}h{eta_s%3600//60}m" if eta_s >= 60 else f"{eta_s}s"
                        el_s = int(elapsed)
                        el_str = f"{el_s//3600}h{el_s%3600//60}m{el_s%60}s" if el_s >= 3600 else f"{el_s//60}m{el_s%60}s"
                        print(
                            f"[{processed:,}/{total_all:,}] "
                            f"match={matched:,} reassign={reassigned:,} title_only={title_only:,} reject={rejected:,} "
                            f"rate={rate:.1f}/s  elapsed={el_str}  ETA={eta_str}"
                        )

                if LIMIT and processed >= LIMIT:
                    break

            save_batch(target_col, batch_results)
            last_id = batch[-1]["_id"]
            save_progress(progress_col, status="running", processed=processed,
                          matched=matched, reassigned=reassigned, rejected=rejected,
                          title_only=title_only, last_id=last_id)

            if LIMIT and processed >= LIMIT:
                break

    save_progress(progress_col, status="completed", processed=processed,
                  matched=matched, reassigned=reassigned, rejected=rejected,
                  title_only=title_only, last_id=last_id, extra={"completedAt": now_iso()})

    print("=== Done ===")
    print(f"processed={processed:,}  match={matched:,}  reassign={reassigned:,}  title_only={title_only:,}  reject={rejected:,}")
    if processed:
        print(f"match_rate={matched/processed*100:.1f}%  reassign_rate={reassigned/processed*100:.1f}%  title_only_rate={title_only/processed*100:.1f}%")


if __name__ == "__main__":
    run()
