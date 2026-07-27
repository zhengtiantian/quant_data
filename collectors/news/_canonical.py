#!/usr/bin/env python3
"""Canonical document shape for the intraday news collectors.

The GDELT collector queries per company, so every article it stores already carries a
`symbol` and a `date`. The intraday collectors (finnhub / newsapi / yahoo) fetch general
market feeds instead and stored neither — which quietly stranded everything they
collected. `slm_company_match_v2.build_query()` requires

    {"symbol": {"$exists": True, "$ne": None}}

so those articles were never matched, never labeled, never turned into features and never
reached a signal. 9,636 articles accumulated that way from 2026-04-13 onward: the freshest
news in the platform was the only news nothing downstream could see.

This module gives the intraday collectors the two fields that matter:

- `date` — the same YYYYMMDDHHMMSS string GDELT writes, derived from `publishedAt`.
- `symbol` — a *candidate* assignment from the company rule files. It is deliberately a
  candidate and not a verdict: the SLM validator downstream is what decides whether the
  article is really about that company, and that is where the accuracy work already lives.

`url` is UNIQUE on `news_articles`, so an article gets exactly one symbol — the same
one-article-one-company shape GDELT produces. Articles matching no company are stored
without a symbol rather than dropped: general market and macro news is precisely the
material the theme-propagation work needs, and throwing it away now would be expensive to
undo.
"""

from __future__ import annotations

import json
import re
from datetime import datetime, UTC
from functools import lru_cache
from pathlib import Path

RULES_DIR = Path(__file__).resolve().parent / "gdelt" / "company_rules"

# Below this many characters a keyword matches too much to be evidence of anything —
# "GE" and "MA" are the obvious cases, and both are real tickers in the universe.
MIN_KEYWORD_LEN = 3


@lru_cache(maxsize=1)
def load_company_rules() -> list[dict]:
    """The 100 company rule files, read once per process.

    Each contributes its primary and expansion keywords. Primary keywords are the company
    name and its unmistakable forms; expansion keywords are products and executives, which
    are strong evidence but not proof, so they score lower.
    """
    rules = []
    for path in sorted(RULES_DIR.glob("*.json")):
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (ValueError, OSError):
            continue
        symbol = data.get("symbol")
        if not symbol:
            continue
        rules.append({
            "symbol": symbol,
            "name": data.get("name", symbol),
            "primary": _compile(data.get("primary_keywords") or []),
            "expansion": _compile(data.get("expansion_keywords") or []),
        })
    return rules


def _compile(keywords) -> list[re.Pattern]:
    """Word-boundary patterns, so "GE" does not match "general" or "Germany"."""
    out = []
    for kw in keywords:
        if not isinstance(kw, str):
            continue
        kw = kw.strip()
        if len(kw) < MIN_KEYWORD_LEN:
            continue
        out.append(re.compile(rf"\b{re.escape(kw)}\b", re.IGNORECASE))
    return out


def match_symbol(*texts: str | None) -> tuple[str, str] | None:
    """Best candidate company for an article, or None when nothing matches.

    Scores primary keyword hits above expansion hits and returns the single highest —
    `url` is unique on the collection, so an article cannot be filed under two companies.
    A tie is broken by symbol so repeated runs over the same article agree with each
    other; a collector that reassigned an article on every pass would keep churning
    downstream state.

    Returns (symbol, company_name).
    """
    haystack = " ".join(t for t in texts if t)
    if not haystack.strip():
        return None

    best: tuple[int, str, str] | None = None
    for rule in load_company_rules():
        score = 3 * sum(1 for p in rule["primary"] if p.search(haystack))
        score += sum(1 for p in rule["expansion"] if p.search(haystack))
        if score == 0:
            continue
        candidate = (score, rule["symbol"], rule["name"])
        if best is None or (score, ) > (best[0], ) or (score == best[0] and rule["symbol"] < best[1]):
            best = candidate
    return (best[1], best[2]) if best else None


def canonical_date(published_at: str | None) -> str:
    """`publishedAt` (ISO 8601) rendered as the YYYYMMDDHHMMSS string GDELT writes.

    Falls back to collection time when the feed gives no timestamp: an article with no
    date at all is invisible to every date-filtered query downstream, and "now" is a
    closer approximation for an intraday feed than nothing.
    """
    if published_at:
        text = str(published_at).strip().replace("Z", "+00:00")
        for parse in (datetime.fromisoformat,):
            try:
                return parse(text).astimezone(UTC).strftime("%Y%m%d%H%M%S")
            except (ValueError, TypeError):
                pass
    return datetime.now(UTC).strftime("%Y%m%d%H%M%S")


def canonicalise(doc: dict) -> dict:
    """Add `date`, and `symbol`/`name` when a company can be identified.

    Mutates and returns the document so a collector can wrap its existing dict without
    restructuring how it builds one.
    """
    doc["date"] = canonical_date(doc.get("publishedAt"))
    matched = match_symbol(doc.get("title"), doc.get("description"), doc.get("content"))
    if matched:
        doc["symbol"], doc["name"] = matched
    return doc
