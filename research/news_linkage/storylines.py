"""N.6 step 1 (2026-11-14): group a symbol's news into storylines.

A storyline is one ongoing story (the Huang affair, a Microsoft 365 outage, an earnings week): articles
filed under the same symbol, within WINDOW_DAYS of each other, that an article joins when
  - some article of the storyline is close in meaning (cosine >= MEMBER_SIM), and
  - the article is close to the storyline as a whole (cosine to its centroid >= CENTROID_SIM), and
  - its title shares a rare word with the storyline's titles (a word in under RARE_DF of a year of titles,
    not the company's own name): "Megaspeed", "outage", "Lutnick", "buyouts".
Tuned on the 2026-10-15..11-14 replay (holdings + watchlist). One company's articles are close in meaning
by default (Nvidia, AI, chips), so the first rule alone chained four weeks of unrelated Nvidia news into
one 5,900-article line; the centroid rule cut the off-topic share, and the rare-word rule split the
generic commentary buckets (Intel's earnings week, Google's cloud outage, analyst notes became their own
lines). Centring the vectors on the company mean was worse: during the affair the mean *is* the affair.
Later steps decide from a storyline's new progress when Claude should judge the combined
impact; this module only builds the storylines and marks every article as
  repeat    the same story again: same normalised title, or a near-identical text (cosine >= DUP_SIM)
  filler    the company is the subject but nothing new is reported: low signal strength or a stock-
            commentary title ("Is Nvidia a buy?", "Prediction: ..."). The noise audit of 2026-11-14 found
            34% of kept articles are of this kind; they stay in the storyline but do not count as progress
  progress  everything else
Neither label set separates filler cleanly (audit: strength >= medium keeps 80% of real news and 43% of
filler; the title pattern drops half of the filler and 12% of real news), so the marks only steer when to
call Claude, and Claude reads the articles themselves.

Sources, merged (the audit found 21% of rejected articles relevant):
  kept    news_articles_company_matched_v2 (production relevance), labels A/B
  gd1     news_articles rows production rejected that the GD1 relevance student keeps (student_gd1, nightly)
  person  news_articles rows rejected by both whose title hits the key-person rule (key_people.hit)

Embeddings: nomic-embed-text-v1.5 on title + first 1000 chars of the body (the description when the
body is empty), the same text and model as the Qdrant index, cached per article id in EMB_CACHE.
Time of an article: publishedAt, or collectedAt when the published time lies after it (date-only or
future-dated feeds), so a replay never sees an article before it was collected.
"""
from __future__ import annotations

import os
import re
import sqlite3
import sys
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from urllib.parse import urlparse

import numpy as np
import requests
from pymongo import MongoClient

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "labeling"))
import key_people  # noqa: E402

MONGO_URI = os.getenv("LOCAL_MONGO_URI", "mongodb://root:root@127.0.0.1:37018/?authSource=admin")
LM = os.getenv("LM_STUDIO_URL", "http://127.0.0.1:1234/v1").rstrip("/")
EMBED_MODEL = os.getenv("EMBED_MODEL", "text-embedding-nomic-embed-text-v1.5")
EMB_CACHE = Path(os.getenv("STORYLINE_EMB_CACHE", "/Volumes/Data4T/news_linkage/emb_cache.sqlite"))
DOC_PREFIX, LEDE_CHARS = "search_document: ", 1000

WINDOW_DAYS = 14     # an article is compared with the symbol's articles of the last 14 days
MEMBER_SIM = 0.80    # some article of the storyline at least this close
CENTROID_SIM = 0.82  # and the storyline as a whole at least this close
RARE_DF = 0.015      # a title word is rare when under 1.5% of a year's titles carry it
DUP_SIM = 0.95       # near-identical text: a repeat (joins without the rare-word rule)
TITLE_DF = Path(os.getenv("STORYLINE_TITLE_DF", "/Volumes/Data4T/news_linkage/title_df.json"))

FILLER_TITLE = re.compile(
    r"(\?\s*$|^prediction:|\bshould you\b|\bis it (too late|time)\b|\bbuy (the dip|now|right now)\b|\bto buy\b|"
    r"\bstrong buy\b|\bhere'?s (why|what)\b|\bwhat (you|investors) need to know\b|\bstocks? to watch\b|"
    r"\bstock market today\b|\bthink again\b|\bstock (could|will|is)\b|\bnarrative\b|\bvaluation\b|\bupside\b|"
    r"\bwho is\b|\bfacts about\b|\bmemes?\b|\bmy cart\b|\bi bought\b|\betfs? (are buying|to watch)\b|"
    r"\bbetter buy\b|\bthis week in\b|\bprice prediction\b)", re.I)
STRENGTH = {"low": 0, "medium": 1, "high": 2}


@dataclass
class Article:
    id: str
    symbol: str
    at: datetime
    title: str
    text: str
    url: str
    domain: str
    src: str                       # kept | gd1 | person
    sentiment: float | None = None
    strength: int | None = None    # 0 low, 1 medium, 2 high (the stronger of the label sets)
    event: str | None = None
    person: str | None = None      # key-person rule hit, if any
    mark: str = ""                 # repeat | filler | progress
    storyline: str = ""
    sim: float = 0.0               # similarity to the closest earlier article of its storyline


@dataclass
class Storyline:
    id: str
    symbol: str
    articles: list[Article] = field(default_factory=list)

    @property
    def first_at(self): return self.articles[0].at

    @property
    def last_at(self): return self.articles[-1].at

    def count(self, mark): return sum(a.mark == mark for a in self.articles)


def _norm_title(t: str) -> str:
    return re.sub(r"[^a-z0-9 ]", "", (t or "").lower()).strip()


def _at(doc) -> datetime | None:
    def parse(v):
        if isinstance(v, datetime):
            return v if v.tzinfo else v.replace(tzinfo=timezone.utc)
        if not v:
            return None
        try:
            d = datetime.fromisoformat(str(v).replace("Z", "+00:00"))
            return d if d.tzinfo else d.replace(tzinfo=timezone.utc)
        except ValueError:
            return None
    pub, col = parse(doc.get("publishedAt")), parse(doc.get("collectedAt"))
    if pub and col and pub > col:
        return col
    return pub or col


def _text(doc) -> str:
    title = (doc.get("title") or "").strip()
    body = (doc.get("content") or "").strip()
    if not body and str(doc.get("description") or "") not in ("", "None"):
        body = str(doc["description"]).strip()
    return f"{DOC_PREFIX}{title}\n\n{body[:LEDE_CHARS]}" if body else DOC_PREFIX + title


def _strength(*vals) -> int | None:
    s = [STRENGTH[v] for v in vals if v in STRENGTH]
    return max(s) if s else None


def load_articles(symbols: list[str], start: datetime, end: datetime, db=None) -> list[Article]:
    """Every article of the three sources for these symbols, published in [start, end)."""
    db = db or MongoClient(MONGO_URI)["quant_data"]
    lo, hi = start.strftime("%Y-%m-%dT%H:%M"), end.strftime("%Y-%m-%dT%H:%M")
    q = {"symbol": {"$in": symbols}, "publishedAt": {"$gte": lo, "$lt": hi}}
    proj = {"title": 1, "content": 1, "description": 1, "url": 1, "symbol": 1, "publishedAt": 1, "collectedAt": 1}
    out, kept = [], set()
    for d in db["news_articles_company_matched_v2"].find(q, {**proj, "llm_sentiment_final": 1, "llm_event_type_a": 1,
                                                             "llm_signal_strength_a": 1, "llm_signal_strength_b": 1}):
        kept.add(str(d["_id"]))
        out.append(_article(d, "kept", d.get("llm_sentiment_final"), _strength(d.get("llm_signal_strength_a"), d.get("llm_signal_strength_b")),
                            d.get("llm_event_type_a")))
    for d in db["news_articles"].find({**q, "origin": {"$ne": "universe_backfill"}}, {**proj, "student_gd1": 1}):
        if str(d["_id"]) in kept:
            continue
        g = d.get("student_gd1") or {}
        if g.get("relevant"):
            src = "gd1"          # includes the rows the key-person rule turned to relevant (rel_rule)
        elif key_people.hit(d.get("symbol"), d.get("title")):
            src = "person"
        else:
            continue
        out.append(_article(d, src, g.get("sentiment"), _strength(g.get("signal_strength")), g.get("event_type")))
    return [a for a in out if a.at and start <= a.at < end]


def _article(d, src, sentiment, strength, event) -> Article:
    url = d.get("url") or ""
    return Article(id=str(d["_id"]), symbol=d.get("symbol") or "", at=_at(d), title=(d.get("title") or "").strip(),
                   text=_text(d), url=url, domain=urlparse(url).netloc.replace("www.", ""), src=src,
                   sentiment=None if sentiment is None else float(sentiment), strength=strength, event=event,
                   person=key_people.hit(d.get("symbol"), d.get("title")))


def embed(articles: list[Article], batch: int = 32) -> np.ndarray:
    """Unit vectors in the order of `articles`; computed once per article id and cached."""
    EMB_CACHE.parent.mkdir(parents=True, exist_ok=True)
    con = sqlite3.connect(EMB_CACHE)
    con.execute("CREATE TABLE IF NOT EXISTS emb (id TEXT PRIMARY KEY, model TEXT, vec BLOB)")
    have = {}
    ids = [a.id for a in articles]
    for i in range(0, len(ids), 900):
        chunk = ids[i:i + 900]
        for k, v in con.execute(f"SELECT id, vec FROM emb WHERE model = ? AND id IN ({','.join('?' * len(chunk))})", [EMBED_MODEL, *chunk]):
            have[k] = np.frombuffer(v, dtype=np.float32)
    todo = [a for a in articles if a.id not in have]
    for i in range(0, len(todo), batch):
        part = todo[i:i + batch]
        r = requests.post(f"{LM}/embeddings", json={"model": EMBED_MODEL, "input": [a.text for a in part]}, timeout=600)
        r.raise_for_status()
        rows = []
        for a, e in zip(part, r.json()["data"]):
            v = np.asarray(e["embedding"], dtype=np.float32)
            v /= np.linalg.norm(v) or 1.0
            have[a.id] = v
            rows.append((a.id, EMBED_MODEL, v.tobytes()))
        con.executemany("INSERT OR REPLACE INTO emb VALUES (?, ?, ?)", rows)
        con.commit()
    con.close()
    return np.stack([have[a.id] for a in articles]) if articles else np.zeros((0, 768), np.float32)


TOKEN = re.compile(r"[a-z][a-z0-9'’\-]{2,}")


def _tokens(title: str) -> set[str]:
    return {w.strip("'’-").replace("’s", "").replace("'s", "") for w in TOKEN.findall((title or "").lower())}


def title_df(db=None, end: datetime | None = None, days: int = 365, refresh: bool = False) -> tuple[int, dict]:
    """Document frequency of title words over the `days` of kept titles before `end`, cached in TITLE_DF."""
    import json
    if TITLE_DF.exists() and not refresh:
        d = json.loads(TITLE_DF.read_text())
        return d["n"], d["df"]
    db = db or MongoClient(MONGO_URI)["quant_data"]
    end = end or datetime.now(timezone.utc)
    lo, hi = (end - timedelta(days=days)).strftime("%Y-%m-%d"), end.strftime("%Y-%m-%d")
    n, df = 0, {}
    for d in db["news_articles_company_matched_v2"].find({"publishedAt": {"$gte": lo, "$lt": hi}}, {"title": 1}):
        n += 1
        for w in _tokens(d.get("title")):
            df[w] = df.get(w, 0) + 1
    TITLE_DF.parent.mkdir(parents=True, exist_ok=True)
    TITLE_DF.write_text(json.dumps({"n": n, "from": lo, "to": hi, "df": df}))
    return n, df


def _own_words(symbol: str) -> set[str]:
    import json
    entry = json.loads(key_people.TABLE.read_text()).get(symbol, {})
    words = {symbol.lower()}
    for a in entry.get("aliases", []):
        words |= set(TOKEN.findall(a.lower()))
    return words


def build(articles: list[Article], vecs: np.ndarray, df: tuple[int, dict], member_sim: float = MEMBER_SIM,
          centroid_sim: float = CENTROID_SIM, rare_df: float = RARE_DF, dup_sim: float = DUP_SIM,
          window_days: int = WINDOW_DAYS) -> list[Storyline]:
    """Assign every article, in time order, to a storyline of its symbol (see the module doc)."""
    n_titles, freq = df
    own = {s: _own_words(s) for s in {a.symbol for a in articles}}
    rare = lambda a: {w for w in _tokens(a.title) if freq.get(w, 0) / n_titles < rare_df and w not in own[a.symbol]}
    order = sorted(range(len(articles)), key=lambda i: (articles[i].symbol, articles[i].at))
    window = timedelta(days=window_days)
    lines: dict[str, Storyline] = {}
    sums: dict[str, np.ndarray] = {}
    words: dict[str, set] = {}
    by_symbol: dict[str, list[int]] = {}
    for i in order:
        a = articles[i]
        a.mark = ""
        mine = rare(a)
        prior = [j for j in by_symbol.setdefault(a.symbol, []) if a.at - articles[j].at <= window]
        key = _norm_title(a.title)
        same = next((j for j in reversed(prior) if key and _norm_title(articles[j].title) == key), None)
        if same is not None:
            a.storyline, a.mark, a.sim = articles[same].storyline, "repeat", 1.0
        else:
            closest: dict[str, float] = {}
            if prior:
                for j, sim in zip(prior, vecs[prior] @ vecs[i]):
                    if sim >= member_sim:
                        lid = articles[j].storyline
                        closest[lid] = max(closest.get(lid, -1.0), float(sim))
            best, best_c, best_m = None, -1.0, 0.0
            for lid, m in closest.items():
                if m < dup_sim and not (mine & words[lid]):
                    continue
                c = float(vecs[i] @ sums[lid] / (np.linalg.norm(sums[lid]) or 1.0))
                if c > best_c:
                    best, best_c, best_m = lid, c, m
            if best is not None and (best_c >= centroid_sim or best_m >= dup_sim):
                a.storyline, a.sim = best, best_c
                if best_m >= dup_sim:
                    a.mark = "repeat"
            else:
                a.storyline = f"{a.symbol}:{a.at:%Y%m%d%H%M}:{a.id[-6:]}"
                lines[a.storyline] = Storyline(a.storyline, a.symbol)
                sums[a.storyline] = np.zeros(vecs.shape[1], np.float32)
                words[a.storyline] = set()
        if not a.mark:
            a.mark = "filler" if (a.strength == 0 or FILLER_TITLE.search(a.title)) else "progress"
        lines[a.storyline].articles.append(a)
        if a.mark != "repeat":
            sums[a.storyline] += vecs[i]
            words[a.storyline] |= mine
        by_symbol[a.symbol].append(i)
    return list(lines.values())


def headline(line: Storyline, articles: list[Article], vecs: np.ndarray) -> str:
    """The title of the article closest to the rest of the storyline (its most typical report)."""
    idx = {a.id: k for k, a in enumerate(articles)}
    members = [a for a in line.articles if a.mark != "repeat"] or line.articles
    if len(members) <= 2:
        return members[0].title
    m = vecs[[idx[a.id] for a in members]]
    return members[int(np.argmax((m @ m.T).mean(axis=1)))].title
