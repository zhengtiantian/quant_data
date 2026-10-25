import json
import os
import sys
import time
import requests
from datetime import date, datetime, timedelta, UTC
from pathlib import Path
from dotenv import load_dotenv
from pymongo import MongoClient, errors

CURRENT = Path(__file__).resolve()
ROOT = CURRENT.parents[3]

# The collectors are run as scripts rather than imported as a package, so the shared
# canonicaliser has to be reachable by path.
sys.path.insert(0, str(CURRENT.parents[1]))
from _canonical import canonicalise  # noqa: E402
GLOBAL_ENV = ROOT / ".env"

if GLOBAL_ENV.exists():
    load_dotenv(GLOBAL_ENV, override=False)
    print(f"Loaded GLOBAL .env: {GLOBAL_ENV}")

MODULE_ENV = CURRENT.parent / ".env"
if MODULE_ENV.exists():
    load_dotenv(MODULE_ENV, override=True)
    print(f"Loaded MODULE .env: {MODULE_ENV}")

FINNHUB_API_KEY = os.getenv("FINNHUB_API_KEY")
LIMIT = int(os.getenv("FINNHUB_NEWS_LIMIT", 50))
LANGUAGE = os.getenv("NEWS_LANGUAGE", "en")
MONGO_URI = os.getenv("MONGO_URI")

if not FINNHUB_API_KEY:
    raise RuntimeError("FINNHUB_API_KEY missing")

# ====== A/B classification keywords ======

STRONG = [
    "earnings","revenue","guidance","profit","loss",
    "acquisition","merger","price target","downgrade","upgrade",
    "sec","investigation","lawsuit","hack",
    "inflation","cpi","ppi","rate hike","rate cut",
    "federal reserve","treasury yield",
    "nvidia","apple","tesla","google"
]

WEAK = [
    "economy","macro","trend","industry",
    "housing","energy","crypto","cloud","ev"
]

def classify(text):
    t = text.lower()
    if any(x in t for x in STRONG):
        return "A"
    if any(x in t for x in WEAK):
        return "B"
    return None

# ====== Fetch ======

def fetch_finnhub():
    url = "https://finnhub.io/api/v1/news"
    params = {"category":"general", "token": FINNHUB_API_KEY}
    print("Request:", url)
    r = requests.get(url, params=params)
    r.raise_for_status()

    return r.json()[:LIMIT]

# ====== Company news, per symbol ======
# The general feed rarely names a mid-cap such as Seagate, so every symbol of the stock universe and
# every holding in quant_ai also gets its own company-news call (2026-10-25). Finnhub's free tier
# allows 60 calls a minute: one call per symbol, 1.05 s apart, about two minutes a run.
COMPANY_NEWS_URL = "https://finnhub.io/api/v1/company-news"
CALL_GAP = 1.05
QUANT_AI_SYMBOLS = os.getenv("QUANT_AI_SYMBOLS_URL", "http://127.0.0.1:18000/api/markets/portfolio/symbols")


def tracked_symbols():
    """{symbol: company name}: the stock universe plus the user's holdings and watchlist in quant_ai
    (so a stock bought tomorrow is followed from its next run on)."""
    out = {}
    try:
        for d in MongoClient(MONGO_URI)["quant_data"]["stock_universe"].find({}, {"symbol": 1, "name": 1, "_id": 0}):
            out[d["symbol"]] = d.get("name") or d["symbol"]
    except Exception as exc:  # noqa: BLE001
        print("stock_universe unavailable:", exc)
    try:
        r = requests.get(QUANT_AI_SYMBOLS, timeout=5).json()
        for s in (r.get("held") or []) + (r.get("watchlist") or []):
            out.setdefault(s, s)
    except Exception as exc:  # noqa: BLE001 - quant_ai not running: the universe still gets its news
        print("quant_ai holdings unavailable:", exc)
    return out


def fetch_company_news(symbol, start, end):
    r = requests.get(COMPANY_NEWS_URL, params={"symbol": symbol, "from": start.isoformat(), "to": end.isoformat(),
                                               "token": FINNHUB_API_KEY}, timeout=20)
    r.raise_for_status()
    data = r.json()
    return data if isinstance(data, list) else []


def company_news_docs(symbol, name, articles):
    """Canonical documents: the keyword matcher may still file an article under another company it
    is really about; one it cannot place stays with the symbol it was asked for."""
    docs = []
    for a in articles:
        if not a.get("url") or not a.get("headline"):
            continue
        doc = canonicalise({
            "source": {"platform": "finnhub", "name": a.get("source")},
            "title": a.get("headline"),
            "description": a.get("summary"),
            "content": None,
            "url": a.get("url"),
            "impact": None,
            "publishedAt": datetime.fromtimestamp(a["datetime"], UTC).isoformat() if a.get("datetime") else None,
            "collectedAt": datetime.now(UTC).isoformat(),
            "language": LANGUAGE,
            "meta": {"collector": "finnhub_company", "requested_symbol": symbol, "version": "1.0"},
        })
        if not doc.get("symbol"):
            doc["symbol"], doc["name"] = symbol, name
        docs.append(doc)
    return docs


def collect_company_news(symbols, start, end, window_days=None):
    """One call per symbol (or per window per symbol for a backfill); returns documents."""
    docs = []
    for i, (sym, name) in enumerate(symbols.items(), 1):
        s = start
        while s <= end:
            e = min(end, s + timedelta(days=window_days)) if window_days else end
            try:
                docs += company_news_docs(sym, name, fetch_company_news(sym, s, e))
            except Exception as exc:  # noqa: BLE001 - one symbol failing must not stop the run
                print(f"  {sym} {s}..{e}: {type(exc).__name__}: {exc}")
            time.sleep(CALL_GAP)
            s = e + timedelta(days=1)
        print(f"PROGRESS company_news {i} {len(symbols)} {sym} docs={len(docs)}", flush=True)
    return docs


# ====== Save (unified schema!) ======

def insert_docs(docs):
    if not docs:
        return
    col = MongoClient(MONGO_URI)["quant_data"]["news_articles"]
    try:
        result = col.insert_many(docs, ordered=False)
        print("Inserted:", len(result.inserted_ids))
    except errors.BulkWriteError as exc:
        details = getattr(exc, "details", {}) or {}
        write_errors = details.get("writeErrors", [])
        non_dup = [err for err in write_errors if err.get("code") != 11000]
        print(f"Inserted: {int(details.get('nInserted', 0) or 0)} (duplicates skipped: {len(write_errors) - len(non_dup)})")
        if non_dup:
            raise


def save_to_mongo(articles):
    client = MongoClient(MONGO_URI)
    col = client["quant_data"]["news_articles"]

    docs = []
    for a in articles:
        impact = a.get("impact")
        published = (
            datetime.fromtimestamp(a["datetime"], UTC).isoformat()
            if a.get("datetime") else None
        )

        # canonicalise() adds `date` and, where a company can be identified, `symbol`.
        # Without those two fields slm_company_match_v2 cannot see the article at all —
        # its query requires a symbol — so everything collected here used to stop dead
        # in news_articles.
        docs.append(canonicalise({
            "source": {"platform": "finnhub"},
            "title": a.get("headline"),
            "description": a.get("summary"),
            "content": None,
            "url": a.get("url"),
            "impact": impact,
            "publishedAt": published,
            "collectedAt": datetime.now(UTC).isoformat(),
            "language": LANGUAGE,
            "meta": {"collector": "finnhub", "version": "1.3"}
        }))

    if docs:
        try:
            result = col.insert_many(docs, ordered=False)
            print("Inserted:", len(result.inserted_ids))
        except errors.BulkWriteError as exc:
            details = getattr(exc, "details", {}) or {}
            inserted = int(details.get("nInserted", 0) or 0)
            write_errors = details.get("writeErrors", [])
            non_dup_errors = [err for err in write_errors if err.get("code") != 11000]
            duplicate_count = len(write_errors) - len(non_dup_errors)
            print(f"Inserted: {inserted} (duplicates skipped: {duplicate_count})")
            if non_dup_errors:
                raise

# ====== Main ======

if __name__ == "__main__":
    raw = fetch_finnhub()

    filtered = []
    for a in raw:
        text = (a.get("headline") or "") + " " + (a.get("summary") or "")
        c = classify(text)
        if c:
            a["impact"] = c
            filtered.append(a)

    print("Fetched AB:", len(filtered))
    for i, a in enumerate(filtered[:5]):
        print(f"{i+1}. ({a['impact']}) {a['headline']}")

    save_to_mongo(filtered)

    # Per-symbol company news: yesterday and today, or a backfill when FINNHUB_BACKFILL is set to
    # "SYMBOL:YYYY-MM-DD,..." (two-week windows from that date; the free tier reaches about a year back).
    backfill = os.getenv("FINNHUB_BACKFILL", "").strip()
    if backfill:
        names = tracked_symbols()
        for item in backfill.split(","):
            sym, since = item.split(":")
            sym = sym.strip().upper()
            docs = collect_company_news({sym: names.get(sym, sym)}, date.fromisoformat(since.strip()),
                                        date.today(), window_days=13)
            insert_docs(docs)
    else:
        today = date.today()
        insert_docs(collect_company_news(tracked_symbols(), today - timedelta(days=1), today))
