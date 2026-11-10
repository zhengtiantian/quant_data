"""Asian English-language business news from RSS, every 30 minutes with the other intraday collectors.

Added 2026-11-10: Morgan Stanley's SK hynix target cut was reported by Korean outlets in the Korean
morning and reached none of the intraday sources (Finnhub / NewsAPI / Yahoo are US-centric), while
GDELT is collected once a day. These feeds keep only their newest ~50 items (Korea Herald: the last
~9 hours), so they have to be read often.

Only items that name a tracked company are stored, one copy per company named (news_articles is
unique on (url, symbol)). The article page is fetched for its text, since slm_company_match_v2 judges
title + content; when that fails the feed's summary stands in. Duplicates are skipped by the index.
"""
import html
import os
import re
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

import feedparser
import requests
from bs4 import BeautifulSoup
from dotenv import load_dotenv
from pymongo import MongoClient, errors

CURRENT = Path(__file__).resolve()
ROOT = CURRENT.parents[3]
sys.path.insert(0, str(CURRENT.parents[1]))
from _canonical import canonical_date, match_symbols  # noqa: E402

if (ROOT / ".env").exists():
    load_dotenv(ROOT / ".env", override=False)
MONGO_URI = os.getenv("MONGO_URI")       # the Airflow host task passes the host-side URI
if not MONGO_URI:
    raise RuntimeError("MONGO_URI missing")

FEEDS = {
    "Yonhap": "https://en.yna.co.kr/RSS/news.xml",
    "The Korea Herald": "https://www.koreaherald.com/rss/newsAll",
    "BusinessKorea": "https://www.businesskorea.co.kr/rss/allArticle.xml",
    "The Elec": "https://www.thelec.net/rss/allArticle.xml",
    "The Korea Times": "https://www.koreatimes.co.kr/www/rss/biz.xml",
    "Nikkei Asia": "https://asia.nikkei.com/rss/feed/nar",
    "SCMP Business": "https://www.scmp.com/rss/92/feed",
    "DigiTimes": "https://www.digitimes.com/rss/daily.xml",
    "Taipei Times": "https://www.taipeitimes.com/xml/index.rss",
}
HEADERS = {"User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 "
                         "(KHTML, like Gecko) Chrome/126.0 Safari/537.36",
           "Accept-Language": "en-US,en;q=0.9"}
TIMEOUT = 15
MIN_BODY = 300          # shorter page text is a paywall / consent page: the summary is used instead
MAX_BODY = 20000


def _clean(text: str | None) -> str:
    text = BeautifulSoup(html.unescape(text or ""), "html.parser").get_text(" ")
    return re.sub(r"\s+", " ", text).strip()


def _published(entry) -> str | None:
    t = entry.get("published_parsed") or entry.get("updated_parsed")
    return datetime(*t[:6], tzinfo=UTC).isoformat() if t else None


def _page_text(url: str) -> str:
    try:
        r = requests.get(url, headers=HEADERS, timeout=TIMEOUT)
        if r.status_code != 200:
            return ""
        soup = BeautifulSoup(r.text, "html.parser")
        for tag in soup(["script", "style", "nav", "header", "footer", "aside", "form"]):
            tag.decompose()
        body = soup.find("article") or soup
        text = " ".join(p.get_text(" ", strip=True) for p in body.find_all("p"))
        text = re.sub(r"\s+", " ", text).strip()
        return text[:MAX_BODY] if len(text) >= MIN_BODY else ""
    except Exception:  # noqa: BLE001 - the summary stands in
        return ""


def collect(existing) -> list[dict]:
    docs, now = [], datetime.now(UTC).isoformat()
    for name, url in FEEDS.items():
        try:
            feed = feedparser.parse(requests.get(url, headers=HEADERS, timeout=TIMEOUT).content)
        except Exception as exc:  # noqa: BLE001 - one feed down must not stop the others
            print(f"  {name}: {type(exc).__name__}: {exc}")
            continue
        kept = 0
        for e in feed.entries:
            link, title = (e.get("link") or "").strip(), _clean(e.get("title"))
            if not link or not title:
                continue
            summary = _clean(e.get("summary") or e.get("description"))
            companies = [c for c in match_symbols(title, summary, expansion_fallback=False) if (link, c[0]) not in existing]
            if not companies:
                continue
            body = _page_text(link) or summary
            published = _published(e)
            for symbol, company in companies:
                docs.append({
                    "source": {"platform": "asia_rss", "name": name},
                    "title": title, "description": summary or None, "content": body or None,
                    "url": link, "impact": None, "publishedAt": published, "collectedAt": now,
                    "language": "en", "date": canonical_date(published), "symbol": symbol, "name": company,
                    "data_quality": "full" if body and body != summary else "summary",
                    "meta": {"collector": "asia_rss", "feed": name, "version": "1.0"},
                })
            kept += 1
            time.sleep(0.3)
        print(f"  {name}: {len(feed.entries)} items, {kept} name a tracked company")
    return docs


def main() -> None:
    col = MongoClient(MONGO_URI)["quant_data"]["news_articles"]
    # (url, symbol) pairs already stored by this collector, so a page is not fetched again every run.
    existing = {(d["url"], d["symbol"]) for d in col.find(
        {"source.platform": "asia_rss"}, {"url": 1, "symbol": 1, "_id": 0}).sort("_id", -1).limit(5000) if d.get("symbol")}
    docs = collect(existing)
    if os.getenv("ASIA_RSS_DRY_RUN") == "1":
        for d in docs:
            print(f"  DRY {d['symbol']:5} {d['data_quality']:7} {len(d['content'] or ''):6} {d['source']['name']}: {d['title'][:90]}")
        print(f"Dry run: {len(docs)} documents, nothing written")
        return
    if not docs:
        print("Inserted: 0")
        return
    try:
        print("Inserted:", len(col.insert_many(docs, ordered=False).inserted_ids))
    except errors.BulkWriteError as exc:
        details = exc.details or {}
        write_errors = details.get("writeErrors", [])
        non_dup = [w for w in write_errors if w.get("code") != 11000]
        print(f"Inserted: {int(details.get('nInserted', 0) or 0)} (duplicates skipped: {len(write_errors) - len(non_dup)})")
        if non_dup:
            raise


if __name__ == "__main__":
    main()
