#!/usr/bin/env python3
"""Hard rules in front of the relevance model (2026-11-10).

The model was asked to judge pages that are not news at all, and it let about half of them through:
quote pages ("NOV - NOV - Stock Price & News"), executive profile pages ("... Net Worth, Biography,
and Insider Trading"), product catalogue pages, tag archives, and rows whose title is a site name
("BizToc", "MSN") or a placeholder ("No Title", "404"). A title pattern settles these without a model.
The patterns were taken from the rows the relevance student judged and checked against the rows it
kept, so each one names a page type, not a word that also appears in headlines ("stock price falls",
"promotes ... to CEO" stay with the model).

    junk_reason(title, url, names) -> None, or the rule that rejects the row:
        "generic_title"    a number, a placeholder, a site name, or the company's own name and nothing else
                           ("Agilent Technologies Inc. (A)": a profile page; names = the company's names)
        "non_news_page"    quote / profile / catalogue / archive page
        "source_only"      a rating agency's headline about someone else's rating ("Moody's downgrades US
                           banks" for Moody's): the model answers YES to these however the prompt is put
        "blocked_domain"   a site in blocked_domains.json (tools/build_domain_blocklist.py: sites with
                           many judged rows of which almost none were kept)

The rejected row carries the rule name as its model, so a rule can be measured and undone.
"""
from __future__ import annotations

import json
import os
import re
from pathlib import Path
from urllib.parse import urlsplit

# The list is rebuilt every week (keyword_refresh_weekly) into a file outside the repository, so the
# working tree stays clean; the copy beside this module is the list of 2026-11-10, used until the
# first rebuild and on a machine that has none.
SEED_FILE = Path(__file__).with_name("blocked_domains.json")
BLOCKLIST_FILE = Path(os.getenv("QUANT_BLOCKLIST_FILE",
                                "~/Library/Application Support/quant/blocked_domains.json")).expanduser()

# Page types. Each alternative is anchored on the punctuation or position that makes it a page title.
_NON_NEWS_TITLE = re.compile(
    r"\bstock price\s*(?:[,&|]|and (?:news|quote|chart))"            # "... Stock Price & News", "Stock Price, Quote"
    r"|\bquote\s*(?:&|and)\s*(?:history|news|chart)"                  # "... News, Quote & History"
    r"|\bstock quote\b.*[|(]"                                         # "AAPL Stock Quote (U.S.: Nasdaq) | MarketWatch"
    r"|\bnet worth, biography"                                        # "<name> Net Worth, Biography, and Insider Trading"
    r"|\bread reviews\s*$"                                            # product catalogue: "Buy <product> Read Reviews"
    r"|\barchives\s*$"                                                # tag page: "<company> Archives"
    r"|\bresearch reports\s*&\s*trade ideas"
    r"|^\s*(?:page not found|access denied|just a moment|attention required|are you a robot|subscribe to read)"
    r"\s*(?:$|[|:!.…\-–—])",                                          # the whole title, not "Access denied and insults; ..."
    re.I)

# Shorter than any headline: "iPhone", "Macworld", "Daniel Ek", "Xcel Energy" (product, site, person and
# company pages). 12,098 core rows, of which the model had kept 3,679.
MIN_TITLE_CHARS = 12

_PLACEHOLDER = frozenset({
    "no title", "untitled", "home", "homepage", "home page", "news", "latest news", "log in", "login", "sign in",
    "sign up", "subscribe", "error", "not found", "404", "403", "forbidden", "redirecting", "loading", "public",
    "press release", "press releases", "newsroom", "blog", "video", "videos", "podcast", "search", "search results",
})

_blocked: frozenset[str] | None = None


def domain_of(url: str) -> str:
    try:
        host = (urlsplit(url or "").hostname or "").lower()
    except ValueError:
        return ""
    return host[4:] if host.startswith("www.") else host


def blocked_domains() -> frozenset[str]:
    global _blocked
    if _blocked is None:
        _blocked = frozenset()
        for f in (BLOCKLIST_FILE, SEED_FILE):
            try:
                _blocked = frozenset(json.loads(f.read_text())["domains"])
                break
            except (OSError, ValueError, KeyError):
                continue
    return _blocked


def _squash(text: str) -> str:
    return re.sub(r"[^a-z0-9]", "", text.lower())


def _is_site_name(title: str, host: str) -> bool:
    """"Fox News" from foxnews.com, "Amazon.com" from amazon.com: the title is the site itself."""
    t = _squash(title)
    if not t or not host:
        return False
    labels = host.split(".")
    names = {_squash(host), _squash(".".join(labels[:-1]))}
    if len(labels) >= 2:
        names.add(labels[-2])
    return t in names


_LEGAL = re.compile(r"\b(?:the|inc|incorporated|corp|corporation|co|company|ltd|limited|plc|sa|nv|ag|se|holdings?|group)\b\.?", re.I)


def _bare_name(text: str) -> str:
    """"Agilent Technologies Inc. (A)" -> "agilenttechnologies": no ticker in brackets, no legal form."""
    return _squash(_LEGAL.sub(" ", re.sub(r"[(\[][^)\]]*[)\]]", " ", text)))


# Rating agencies as the source: "<agency> <rating verb> ..." unless the object is the agency's own business.
# Only for these two companies: "TSMC cuts sales forecast" and "PepsiCo boosts outlook" have the same shape
# and are the company's own news. 914 rows of 2.6M (2026-11-10), 758 of them Moody's.
_RATERS = {"MCO": r"Moody's", "SPGI": r"S&P Global|S&P|Standard & Poor's"}
_RATER_TITLE = {sym: re.compile(
    rf"^\W*(?:{alt})(?:\s+(?:Ratings|Investors Service))?\s+(?:upgrades?|downgrades?|affirms?|reaffirms?|assigns?|"
    rf"withdraws?|maintains?|raises?|cuts?|lifts?|lowers?|revises?|changes?|places?|warns?|sees?|expects?|says?)\b"
    rf"(?!.*\b(?:its|own|jobs?|staff|workforce|dividend|buyback|guidance|profit|revenue|earnings|quarter|acquir\w+|buys?)\b)",
    re.I) for sym, alt in _RATERS.items()}


# A generic title over a full body is an article whose title was scraped wrong: 562 core rows titled
# "Deadline" are whole deadline.com articles. Those stay with the model, which reads the body.
FULL_BODY_CHARS = 500


def _generic_title(t: str, host: str, names) -> bool:
    if len(t) < MIN_TITLE_CHARS or re.fullmatch(r"[\d\W_]+", t) or t.lower() in _PLACEHOLDER or _is_site_name(t, host):
        return True
    bare = _bare_name(t) if names else ""
    return bool(bare) and any(bare == _bare_name(n) for n in names if n)


def junk_reason(title: str, url: str = "", names: tuple[str, ...] | list[str] = (), symbol: str = "",
                body: str | None = None) -> str | None:
    """`body` is the article text when the caller has it: a generic title rejects the row only when the
    body is short too (or unknown)."""
    t = " ".join((title or "").split())
    host = domain_of(url)
    if _generic_title(t, host, names) and (body is None or len(" ".join(body.split())) < FULL_BODY_CHARS):
        return "generic_title"
    if _NON_NEWS_TITLE.search(t):
        return "non_news_page"
    rater = _RATER_TITLE.get(symbol.upper())
    if rater and rater.search(t):
        return "source_only"
    if host and host in blocked_domains():
        return "blocked_domain"
    return None
