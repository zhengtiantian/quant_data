"""The headline check shared by the GDELT matchers (2026-11-04).

A GKG row lists every organisation and person an article mentions, so a company named once in a market wrap,
a stadium named after a sponsor or a research-report advert all match its keywords. title_gate() keeps a row
only when one of the company's terms is in the headline or the URL slug. Used by tools/backfill_symbol_news.py
(the universe and keyword-rescan backfills) and by historical_collector.py (the nightly core match).
"""
from __future__ import annotations

import re

SPAM_RE = re.compile(r"\bMarket\b.{0,40}\b(20\d\d|Size|Report|Analysis|Trends?|Outlook|Forecast|Research|Growth|Worth|"
                     r"Insights|Overview|Segmentation|Study|Demand|Landscape|Consumption|Business Strategy|Development Plans|"
                     r"Key Factors)\b|Key Players|Players Include|Top Companies|Major Manufacturers|\bFeaturing\b|"
                     r"Industry (to|by) 20\d\d|Forecast (to|till|by) 20\d\d|MarketsandMarkets|\bCAGR\b|SIDE EFFECTS|\bReviews? \(|【", re.I)
# a venue or event named after a sponsor ("Jones AT&T Stadium", "Globe Life Field") is not news of it: a
# name directly followed by one of these words does not count
VENUE_AFTER = (r"(?! (?:Stadium|Field|Arena|Center|Centre|Park|Dome|Superdome|Speedway|Forum|Amphitheat(?:er|re)|Pavilion|Bowl|"
               r"Championship|Classic|Invitational|Foundation|Literacy Foundation|Series|600|500|400)\b)")
_ALLNAMES_TAIL = re.compile(r"^[^,;]{1,80},\d+\s+")
_TS14 = re.compile(r"\b\d{14}\b")


def gkg_title(raw: str) -> str:
    """The headline at the end of a GKG raw row: after the last entity / amount entry (each ends with ";"),
    a last AllNames entry without one ("Synopsys Inc,329 ..."), and the author and 14-digit timestamp."""
    import html
    t = html.unescape(raw or "").rsplit(";", 1)[-1]
    t = _ALLNAMES_TAIL.sub("", t.strip())
    m = None
    for m in _TS14.finditer(t):
        pass
    return (t[m.end():] if m else t).strip()


def url_slug(url: str) -> str:
    last = (url or "").split("?")[0].rstrip("/").rsplit("/", 1)[-1]
    return re.sub(r"[-_+]+", " ", re.sub(r"\.\w{2,5}$", "", last))


def title_gate(terms: list, raw: str, url: str, title: str | None = None) -> bool:
    """`title` when the caller has the headline already (GDELT CSV's PAGE_TITLE); else it is read from `raw`."""
    title = (title or gkg_title(raw)).replace("\u2019", "'")     # "McDonald’s" -> "McDonald's"
    if SPAM_RE.search(title):
        return False
    text = title + " | " + url_slug(url)     # ("Share" left the spam words 2026-11-04: "rivals snag market share")
    return any(re.search(r"(?<!\w)" + re.escape(k) + r"(?!\w)" + VENUE_AFTER, text, re.I) for k in terms if len(k) > 1)


# Headlines use a company's short name ("Apple", "Intel", "Lilly") where the rules match the formal one
# ("Apple Inc.") to stay unambiguous in GDELT's organisation lists. The check runs only after a row matched the
# company's keywords and passed its rule, so the short name in the headline is extra evidence, not a new match.
# Without it the first version dropped 114 of 117 sampled Apple rows that were about Apple (2026-11-04).
_SUFFIX = re.compile(r"(?:,?\s+(?:and\s+Company|Inc\.?|Incorporated|Corp\.?|Corporation|Co\.?|Company|Ltd\.?|Limited|plc|PLC|"
                     r"N\.?V\.?|S\.?A\.?|AG|SE|Holdings?|Group|Platforms|Technologies|Technology|Class [A-C])|\.com)$", re.I)


def short_names(cfg: dict) -> list:
    """`headline_names` from the rule file when it has them (MNDY: "monday" alone is a weekday), else the
    company's own name without "The" and legal / generic suffixes."""
    if cfg.get("headline_names"):
        return list(cfg["headline_names"])
    out = []
    for n in (cfg.get("name"), cfg.get("company_name")):
        n = re.sub(r"^The\s+", "", (n or "").strip())
        prev = None
        while n and n != prev:
            prev, n = n, _SUFFIX.sub("", n).strip(" ,")
        if len(n) > 2 and n not in out:
            out.append(n)
    return out


def gate_terms(rm, symbol: str, date) -> list:
    """Terms title_gate() looks for: the company's match keywords plus its short names."""
    terms = [k for k in rm.get_keywords(symbol, date) if isinstance(k, str)]
    return terms + [n for n in short_names(rm.company_configs.get(symbol, {})) if n not in terms]
