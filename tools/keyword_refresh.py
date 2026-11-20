"""Weekly keyword refresh for the company rules (2026-11-04, approved by the user: model check + co-mention
check, every new term reviewed in quant_ui before it is used).

Why: the rules' product and executive names were written once (the 102 hand-written rules last in July, the
490 universe rules hold names only), so news of a new product or a new CEO is missed until someone edits
the file. Each week this proposes the names that appeared since, and the user decides.

Stages:
  mine    the last 7 days of GKG headlines (/Volumes/Data4T/gkg_parquet, refreshed nightly). For each
          company, the headlines that name it with today's keywords; in them, the 1-3 word phrases that
          are not already a keyword (keyword spans are cut out first, so "Apple Watch Ultra 3" adds
          nothing), seen in >= MIN_TITLES distinct headlines from >= MIN_SOURCES sites. Co-mention: of
          all the week's headlines that contain the phrase, the share that also name the company; generic
          phrases ("Q3 results", "Wall Street") are in everyone's headlines and fall below WEAK.
  judge   the local model, one candidate per request with up to 4 of its headlines: a product, brand,
          service, subsidiary or current executive of this company, or none.
  run     mine + judge -> <OUT>/runs/<date>.json, shown on quant_ui's Keywords tab
          (quant_ai markets/keyword_review.py serves it and writes the user's decisions to
          <OUT>/decisions.json).
  apply   the decisions into the rule files: "add" -> expansion_keywords, "weak" -> weak_names_not_counted
          (kept for the SLM pass, not matched), each listed in auto_keywords with the week and the first
          headline date; a decision changed to "reject" later takes the term out again. The old file is
          copied to <OUT>/backups/<time>/ first. No model, safe to run any time.

Tiers shown to the user: "add" (model says it belongs, co-mention >= STRONG, not an ordinary word),
"weak" (model says it belongs, co-mention between WEAK and STRONG, or an ordinary word), "model_none".

  rescan  past months for every added keyword not rescanned yet (products, brands, services from
          KEYWORD_RESCAN_SINCE, 2021-01; subsidiaries and executives from the day first seen), one
          GKG read per month for all of them (<OUT>/rescan.json records each). core: the collector's rules,
          then downloaded into news_articles like backfill_symbol_news.py, for the nightly company match
          and enrichment. universe: rules + headline gate, merged into backfill_universe_news.py's kept
          rows (downloaded by the same run's fetch, then labelled by the students). Run daily since
          2026-11-20 by backfill_universe_news.py daily (airflow coverage_watch_daily, 09:20 after the
          audit; it holds LM Studio: the core filter may ask the SLM). The universe windows that ran it
          until 2026-11-07 were removed with the title-only labels.

  screen  (2026-11-19) the collateral check of the latest run's add / weak candidates (or --run DATE): past
          co-mention over 24 months of GKG headlines, the words and companies next to the term in the
          headlines that do not name the company, Claude on other meanings (`claude -p`; --no-claude skips).
  auto    what auto_decide() would add (preview; the hourly apply adds them when settings.json says so)
  audit   (daily) what each keyword in the rules brought in and the share the relevance model kept;
          auto-added terms that fail are taken out again, those that pass are confirmed

Usage: .venv311/bin/python tools/keyword_refresh.py mine|judge|run|screen|auto|apply|audit|rescan|status [--until HH:MM] [--dry-run]
       (judge / run / rescan under tools/lm_session.py run --model qwen3.5-9b-mlx --workers-env LM_WORKERS -- ...)
"""

from __future__ import annotations

import json
import os
import re
import shutil
import sys
import time
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta
from pathlib import Path
from urllib.parse import urlparse

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
sys.path.insert(0, str(ROOT / "collectors" / "news" / "gdelt"))

RULE_SETS = {"core": Path(os.getenv("CORE_RULES_DIR", str(ROOT / "collectors" / "news" / "gdelt" / "company_rules"))),
             "universe": Path(os.getenv("UNIVERSE_RULES_DIR",
                                        str(ROOT / "collectors" / "news" / "gdelt" / "universe_rules" / "company_rules")))}
PARQUET_DIR = Path(os.getenv("GKG_PARQUET_DIR", "/Volumes/Data4T/gkg_parquet"))
OUT = Path(os.getenv("KEYWORD_REFRESH_DIR", "/Volumes/Data4T/keyword_refresh"))
MODEL = os.getenv("KEYWORD_REFRESH_MODEL", "qwen3.5-9b-mlx")
API = os.getenv("SLM_API_URL", "http://localhost:1234/v1").rstrip("/")
WORKERS = int(os.getenv("LM_WORKERS", "8"))
os.environ.setdefault("MONGO_URI", "mongodb://root:root@127.0.0.1:37018/")    # as backfill_universe_news.py
DAYS, MIN_TITLES, MIN_SOURCES = 7, 3, 2
# distinct stories, not headlines: one syndicated story ("Wollongong couple takes 270km Australian road trip
# in a Tesla", 39 copies, 2026-11-04 dry run) gives every phrase in it 100% co-mention
MIN_STORIES, SAME_STORY = 3, 0.4
STRONG, WEAK = 0.6, 0.3
MAX_PER_COMPANY = 10
KINDS = {"product", "brand", "service", "subsidiary", "executive"}
DOUBT_RE = re.compile(r"\b(former|formerly|discontinued|defunct|category|generic|not a specific|being acquired|"
                      r"to acquire|acquiring|bid|stake|partner|featuring|uses|used in|not owned)\b", re.I)

STOP = set("""a an the and or but nor of to in on at by for from with without into onto over under after before
since until about as is are was were be been being has have had do does did will would can could should may
might must shall it its this that these those he she they them his her their our your my we you i me us
who whom whose which what when where why how than then so not no yes all any some more most less few many
much very just also only even still new news says said say report reports reported update updates latest
today yesterday week weeks month months year years day days amid via vs per up down out off top big
january february march april may june july august september october november december
monday tuesday wednesday thursday friday saturday sunday jan feb mar apr jun jul aug sep sept oct nov dec
inc corp co ltd plc llc company companies group stock stocks share shares price prices market markets s
""".split())
_SEG = re.compile(r"[|:;,!?()\[\]{}\"“”–—]|\s-\s|\s/\s")
_TOK = re.compile(r"[A-Za-z0-9][A-Za-z0-9&'+.-]*[A-Za-z0-9+]|[A-Za-z0-9]")


def now_iso() -> str:
    return datetime.now().isoformat(timespec="seconds")


def load_rules() -> dict:
    """{(set, symbol): (path, rule, terms the company is matched by today, every term already known, RuleManager)}."""
    from special_rules import RuleManager
    out = {}
    for name, d in RULE_SETS.items():
        if not d.exists():
            continue
        rm = RuleManager(str(d))
        for sym, cfg in rm.company_configs.items():
            match = {k for k in rm.get_keywords(sym, datetime.now()) if isinstance(k, str) and len(k.strip()) > 1}
            amb = (cfg.get("special_rules") or {}).get("ambiguous") or {}
            known = set(match) | set(amb.get("required_keywords") or []) | set(cfg.get("weak_names_not_counted") or []) \
                | set(cfg.get("dropped_expansion") or []) | {a["term"] for a in cfg.get("auto_keywords") or []} \
                | {cfg.get("name") or sym}
            out[(name, sym)] = (d / f"{sym}.json", cfg, sorted(match), sorted(k for k in known if isinstance(k, str) and k.strip()), rm)
    return out


def _no_slm(rule) -> None:
    """Mining asks the rule engine about ~50k headlines a week: rules only, no SLM call per single weak hit."""
    for r in getattr(rule, "rules", None) or [rule]:
        if hasattr(r, "use_slm"):
            r.use_slm = False


def _kw_rx(terms) -> str:
    return r"(?i)(?<![\w])(" + "|".join(re.escape(t) for t in sorted(terms, key=len, reverse=True)) + r")(?![\w])"


def _namelike(tok: str) -> bool:
    return tok[:1].isupper() or tok[:1].isdigit() or any(c.isupper() for c in tok[1:])


def _sentence_case(title: str) -> bool:
    """A headline that capitalises only names (most do), so a capitalised phrase in it is a name."""
    words = [w for w in title.split() if w[:1].isalpha()]
    return len(words) >= 4 and sum(w[:1].islower() for w in words) >= 0.3 * len(words)


def _story_words(title: str, drop: set) -> set:
    return {w for w in re.findall(r"[a-z0-9]+", title.lower()) if w not in STOP and w not in drop and len(w) > 2}


def stories(titles: list, phrase: str) -> int:
    """Distinct stories among headlines: greedy clusters of word overlap (Jaccard >= SAME_STORY)."""
    drop = set(phrase.split())
    clusters: list = []
    for t in titles:
        w = _story_words(t, drop)
        for c in clusters:
            if w and len(w & c) / len(w | c) >= SAME_STORY:
                c |= w
                break
        else:
            clusters.append(set(w))
    return len(clusters)


def domain(url: str) -> str:
    try:
        return urlparse(url or "").netloc.lower().removeprefix("www.")
    except ValueError:                              # malformed GKG URLs ("Invalid IPv6 URL")
        return (url or "").split("/")[2:3][0] if (url or "").count("/") >= 2 else ""


def week_titles(days: int = DAYS) -> list:
    """Distinct headlines of the last `days` days: [{title, key, urls, domains, ts}]."""
    import pyarrow.compute as pc
    import pyarrow.dataset as ds
    import backfill_symbol_news as bf
    since = (datetime.now() - timedelta(days=days)).strftime("%Y%m%d%H%M%S")
    months = sorted({(datetime.now() - timedelta(days=d)).strftime("%Y-%m") for d in range(days + 1)})
    by_key: dict = {}
    for m in months:
        p = PARQUET_DIR / f"month={m}"
        if not p.exists():
            continue
        t = ds.dataset(str(p), format="parquet").to_table(columns=["ts", "url", "raw"])
        t = t.filter(pc.greater_equal(t["ts"], since))
        for r in t.to_pylist():
            title = bf.gkg_title(r["raw"]).replace("’", "'").strip()
            if len(title) < 12 or bf.SPAM_RE.search(title):
                continue
            key = re.sub(r"[^a-z0-9]+", " ", title.lower()).strip()
            e = by_key.get(key)
            if e is None:
                e = by_key[key] = {"title": title, "key": key, "urls": [], "domains": set(), "ts": str(r["ts"]), "raw": r["raw"]}
            if len(e["urls"]) < 3:
                e["urls"].append(r["url"])
            e["domains"].add(domain(r["url"]))
            e["ts"] = min(e["ts"], str(r["ts"]))
    return list(by_key.values())


def _tokens(segment: str) -> list:
    out = []
    for t in _TOK.findall(segment):
        t = re.sub(r"'s$", "", t)
        if t:
            out.append(t)
    return out


def ngrams(text: str, n_max: int = 3):
    """(lower-case phrase, surface form) for each 1..n_max word phrase inside one punctuation segment."""
    for seg in _SEG.split(text):
        toks = _tokens(seg)
        for i in range(len(toks)):
            for n in range(1, n_max + 1):
                if i + n > len(toks):
                    break
                g = toks[i:i + n]
                lo = [x.lower() for x in g]
                if lo[0] in STOP or lo[-1] in STOP or all(x.isdigit() for x in lo) or len(" ".join(lo)) < 3:
                    continue
                if any(x.replace(".", "").isdigit() and len(x) > 4 for x in lo):        # years, amounts
                    continue
                yield " ".join(lo), " ".join(g)


def mine(days: int = DAYS) -> dict:
    import pyarrow as pa
    import pyarrow.compute as pc
    t0 = time.time()
    rules = load_rules()
    titles = week_titles(days)
    print(f"mine: {len(titles)} distinct headlines in {days} days, {len(rules)} companies ({time.time() - t0:.0f}s)", flush=True)
    col = pa.array([e["title"] for e in titles])
    known_lc = {k: [x.lower() for x in v[3]] for k, v in rules.items()}
    # another tracked company's own name is that company's news, not a new name of this one
    others = defaultdict(set)
    for (rs, sym), (_p, cfg, _m, _k, _rm) in rules.items():
        for k in cfg.get("primary_keywords") or []:
            if isinstance(k, str) and len(k) > 2:
                others[k.lower()].add(sym)
    per_company: dict = {}
    wanted: set = set()
    for (rs, sym), (_path, cfg, match, known, rm) in rules.items():
        if not match:
            continue
        rx = _kw_rx(match)
        idx = pc.indices_nonzero(pc.match_substring_regex(col, rx.replace("(?<![\\w])", "\\b").replace("(?![\\w])", "\\b"))).to_pylist()
        # and the collector's rule engine on the GKG text, as the backfill filter does: a company with an
        # ambiguous name needs its required words there. Without it NET's "NET" took every "net profit"
        # headline, V's "Visa" the immigration news, ABNB's "IPO" everyone's (2026-11-04 dry run).
        # GKG text has no "Apple Inc." in most Apple rows, so a headline also counts when it carries a name that is
        # proof on its own: a primary keyword that is not an ordinary word, or the rule's identity words (AAPL:
        # iPhone, MacBook).
        rule = rm.rules.get(sym)
        if idx and rule is not None and type(rule).__name__ != "NoRule":
            _no_slm(rule)
            amb = (cfg.get("special_rules") or {}).get("ambiguous") or {}
            proof = [k for k in cfg.get("primary_keywords") or [] if isinstance(k, str) and len(k) > 3 and not _weak_word(k)
                     and k.upper() != k] + [k for k in (amb.get("strict_validation_keywords") or []) if isinstance(amb, dict) and isinstance(k, str)]
            prx = re.compile(_kw_rx(proof)[4:], re.I) if proof else None
            idx = [i for i in idx if (prx is not None and prx.search(titles[i]["title"]))
                   or rm.should_include(sym, {"title": titles[i]["title"], "content": titles[i]["raw"],
                                              "url": titles[i]["urls"][0] if titles[i]["urls"] else "", "date": titles[i]["ts"][:8]})]
        if not idx:
            continue
        cut = re.compile(_kw_rx(known)[4:] + r"(?:'s\b)?", re.I)
        counts: Counter = Counter()
        doms: dict = defaultdict(set)
        surf: dict = defaultdict(Counter)
        ex: dict = defaultdict(list)
        tl: dict = defaultdict(list)
        sc: Counter = Counter()           # occurrences in sentence-case headlines
        scn: Counter = Counter()          # ... written like a name there
        first: dict = {}
        for i in idx:
            e = titles[i]
            seen = set()
            sent = _sentence_case(e["title"])
            for lo, sf in ngrams(cut.sub(" | ", e["title"])):
                if lo in seen:
                    continue
                seen.add(lo)
                counts[lo] += 1
                doms[lo] |= e["domains"]
                surf[lo][sf] += 1
                if len(tl[lo]) < 200:
                    tl[lo].append(e["title"])
                if sent:
                    sc[lo] += 1
                    scn[lo] += all(_namelike(t) for t in sf.split())
                if len(ex[lo]) < 4:
                    ex[lo].append({"title": e["title"], "url": e["urls"][0] if e["urls"] else "", "ts": e["ts"]})
                first[lo] = min(first.get(lo, e["ts"]), e["ts"])
        kl = known_lc[(rs, sym)]
        ticks = {sym.lower(), sym.lower().replace("-", "."), sym.lower().replace("-", "")}

        def ok(g: str, c: int) -> bool:
            if c < MIN_TITLES or len(doms[g]) < MIN_SOURCES or g in ticks:
                return False
            if any(g == k or re.search(r"(?<!\w)" + re.escape(g) + r"(?!\w)", k) for k in kl):
                return False
            w = g.split()
            if any(sym not in others.get(" ".join(w[i:j]), {sym}) for i in range(len(w)) for j in range(i + 1, len(w) + 1)):
                return False
            if not all(_namelike(t) for t in surf[g].most_common(1)[0][0].split()):
                return False
            if sc[g] and scn[g] < 0.6 * sc[g]:
                return False
            return stories(tl[g], g) >= MIN_STORIES

        cands = {g: c for g, c in counts.items() if ok(g, c)}
        if not cands:
            continue
        per_company[(rs, sym)] = {"n_titles": len(idx), "cands": cands, "doms": doms, "surf": surf, "ex": ex, "first": first, "tl": tl}
        wanted |= set(cands)
    print(f"mine: {len(per_company)} companies with candidates, {len(wanted)} phrases to count ({time.time() - t0:.0f}s)", flush=True)
    total: Counter = Counter()
    for e in titles:
        seen = set()
        for lo, _ in ngrams(e["title"]):
            if lo in wanted and lo not in seen:
                seen.add(lo)
                total[lo] += 1
    out = []
    for (rs, sym), d in per_company.items():
        rows = []
        for g, c in d["cands"].items():
            ratio = min(1.0, c / max(1, total[g]))
            if ratio < WEAK:
                continue
            rows.append({"g": g, "with": c, "total": total[g], "ratio": round(ratio, 3)})
        # a fragment of a longer phrase seen almost as often is the longer phrase ("17 pro" in "iphone 17 pro")
        rows.sort(key=lambda r: (-len(r["g"].split()), -r["with"]))
        keep = []
        for r in rows:
            if any(re.search(r"(?<!\w)" + re.escape(r["g"]) + r"(?!\w)", k["g"]) and k["with"] >= 0.8 * r["with"] for k in keep):
                continue
            keep.append(r)
        keep = [r for r in keep if not any(o is not r and len(o["g"]) < len(r["g"])
                                           and re.search(r"(?<!\w)" + re.escape(o["g"]) + r"(?!\w)", r["g"]) for o in keep)]
        keep.sort(key=lambda r: (-r["ratio"] * min(r["with"], 20), r["g"]))
        path, cfg, match, known, _rm = rules[(rs, sym)]
        for r in keep[:MAX_PER_COMPANY]:
            g = r["g"]
            out.append({"id": f"{rs}:{sym}:{g}", "set": rs, "symbol": sym, "name": cfg.get("name") or sym,
                        "term": d["surf"][g].most_common(1)[0][0], "headlines_with_company": r["with"],
                        "headlines_total": r["total"], "co_mention": r["ratio"], "sites": len(d["doms"][g]),
                        "stories": stories(d["tl"][g], g),
                        "first_seen": d["first"][g], "examples": d["ex"][g], "company_headlines": d["n_titles"],
                        "current_keywords": match[:20]})
    res = {"created": now_iso(), "days": days, "headlines": len(titles), "candidates": out}
    print(f"mine: {len(out)} candidates from {len({c['id'].rsplit(':', 1)[0] for c in out})} companies ({time.time() - t0:.0f}s)", flush=True)
    return res


JUDGE_PROMPT = """You maintain the keyword list used to find news about one listed company.

Company: {name} (ticker {symbol})
Keywords it already has: {known}

Candidate: "{term}"
This week's headlines that contain the candidate and name the company:
{headlines}

Is "{term}" a name that, in the news, refers to this company or to something it owns or runs today: one of its
products, brands, services or subsidiaries, or one of its current executives?
Answer "none" for: a person who is not its executive (actor, athlete, reporter, analyst, politician), another
company (a rival, partner, customer, supplier, or a company it is only bidding for), a place, an event, a
number or amount, a generic technical term (GPU, OLED, SSD), or a fragment of a longer name.

Reply with JSON only: {{"kind": "product|brand|service|subsidiary|executive|none", "reason": "<at most 12 words>"}}"""


def _ask(c: dict) -> dict | None:
    import requests
    prompt = JUDGE_PROMPT.format(name=c["name"], symbol=c["symbol"], known=", ".join(c["current_keywords"][:15]) or "-",
                                 term=c["term"], headlines="\n".join(f"- {e['title'][:160]}" for e in c["examples"]))
    msgs = [{"role": "user", "content": prompt},
            {"role": "assistant", "content": "<think>\n\n</think>\n{" if "qwen" in MODEL.lower() else "{"}]
    for _ in range(2):
        try:
            r = requests.post(f"{API}/chat/completions", timeout=120,
                              json={"model": MODEL, "messages": msgs, "max_tokens": 80, "temperature": 0.0, "stream": False})
            r.raise_for_status()
            raw = r.json()["choices"][0]["message"]["content"].strip()
            raw = raw if raw.startswith("{") else "{" + raw
            m = re.search(r"\{[^{}]*\"kind\"[^{}]*\}", raw, re.S)
            if m:
                d = json.loads(m.group())
                kind = str(d.get("kind", "none")).lower().strip()
                return {"kind": kind if kind in KINDS else "none", "reason": str(d.get("reason", ""))[:200]}
        except Exception as e:  # noqa: BLE001
            print(f"judge {c['id']}: {type(e).__name__}: {e}", flush=True)
    return None


def _weak_word(term: str) -> bool:
    import generate_company_rules as g
    return g._weak(term)


def judge(res: dict, deadline: float | None = None) -> dict:
    """Adds kind / reason / tier to every candidate; any not reached before `deadline` stays tier "unjudged"."""
    cands = res["candidates"]
    t0 = time.time()
    with ThreadPoolExecutor(WORKERS) as ex:
        futs = []
        for c in cands:
            futs.append(ex.submit(lambda c=c: None if deadline and time.time() > deadline else _ask(c)))
        for c, f in zip(cands, futs):
            v = f.result()
            c["model"] = MODEL
            if v is None:
                c.update(kind=None, reason="", tier="unjudged")
                continue
            c.update(v)
            weak = _weak_word(c["term"])
            c["ordinary_word"] = weak
            # 2026-11-04 dry run: "11 PC", "18 Pro", "9 Pro" (fragments of Windows 11 PC, iPhone 18 Pro) and
            # reasons that undo the kind ("a former COO", "a discontinued Kindle", "a product category",
            # "being acquired") were judged as products / executives
            caveat = c["term"].split()[0].isdigit() or bool(DOUBT_RE.search(c["reason"]))
            c["caveat"] = caveat
            c["tier"] = ("model_none" if v["kind"] == "none" else
                         "add" if c["co_mention"] >= STRONG and not weak and not caveat else "weak")
    tiers = Counter(c["tier"] for c in cands)
    print(f"judge: {len(cands)} in {time.time() - t0:.0f}s: {dict(tiers)}", flush=True)
    res["judged"] = now_iso()
    return res


# ---------------------------------------------------------------- collateral check (2026-11-19)
# The user (2026-11-19): terms should go into the rules without a review where that is safe, and the ones
# that would also pull in news about something else must be recognised. The week's co-mention is measured
# while the term is in the company's news; these checks look at the rest of the time:
#   history     the GKG headlines of the HISTORY_MONTHS before the week: of those containing the term, the
#               share that name the company (today's keywords and the rule engine, as in mine())
#   neighbour   in those that do not name it, the word most often next to the term ("Galaxy" before
#               "A20") and the other tracked company most often named
#   flood       those headlines per month: what the term would bring in on its own
#   Claude      other common meanings in the news, one `claude -p` call per CLAUDE_BATCH terms
# A candidate with any risk waits for the user; auto_decide() may apply the rest.
HISTORY_MONTHS = int(os.getenv("KEYWORD_SCREEN_MONTHS", "24"))
HIST_CAP = 3000
RISK_CO, RISK_MIN_HIST = 0.5, 20
NEIGHBOUR_SHARE, OTHER_SHARE, MIN_EXTRA = 0.4, 0.2, 8
FLOOD_PER_MONTH = 300
CLAUDE_BATCH = 40
CLAUDE_TIMEOUT_S = 900
SCREEN_TIERS = ("add", "weak")


def history_titles(terms: list, before: str, months: int = HISTORY_MONTHS) -> dict:
    """{term in lower case: [{title, url, ts, raw}]}: distinct GKG headlines containing the term in the
    `months` months before `before` (YYYYMMDDHHMMSS), newest first, at most HIST_CAP per term."""
    import pyarrow.compute as pc
    import pyarrow.dataset as ds
    import backfill_symbol_news as bf
    if not terms:
        return {}
    prx = _kw_rx(terms).replace("(?<![\\w])", "\\b").replace("(?![\\w])", "\\b")      # RE2: no lookbehind
    pats = {t.lower(): re.compile(r"(?<!\w)" + re.escape(t) + r"(?!\w)", re.I) for t in terms}
    out: dict = {k: {} for k in pats}
    end = datetime.strptime(before[:8], "%Y%m%d")
    for m in sorted({(end - timedelta(days=30 * i)).strftime("%Y-%m") for i in range(months + 1)}, reverse=True):
        p = PARQUET_DIR / f"month={m}"
        if not p.exists():
            continue
        t = ds.dataset(str(p), format="parquet").to_table(columns=["ts", "url", "raw"])
        t = t.filter(pc.less(t["ts"], before))
        t = t.filter(pc.match_substring_regex(t["raw"], prx))
        for r in t.to_pylist():
            title = bf.gkg_title(r["raw"]).replace("’", "'").strip()
            if len(title) < 12 or bf.SPAM_RE.search(title):
                continue
            key = re.sub(r"[^a-z0-9]+", " ", title.lower()).strip()
            for k, pat in pats.items():
                if len(out[k]) < HIST_CAP and key not in out[k] and pat.search(title):
                    out[k][key] = {"title": title, "url": r["url"], "ts": str(r["ts"]), "raw": r["raw"]}
    return {k: list(v.values()) for k, v in out.items()}


def _company_test(cfg: dict, match: list, rm, sym: str):
    """mine()'s test of a headline naming the company: today's keywords, then the rule engine unless a
    proof name (a distinctive primary keyword, the identity words) is in the headline."""
    rx = re.compile(_kw_rx(match)[4:], re.I) if match else None
    rule = rm.rules.get(sym)
    use_rule = rule is not None and type(rule).__name__ != "NoRule"
    if use_rule:
        _no_slm(rule)
    amb = (cfg.get("special_rules") or {}).get("ambiguous") or {}
    proof = [k for k in cfg.get("primary_keywords") or [] if isinstance(k, str) and len(k) > 3 and not _weak_word(k)
             and k.upper() != k] + [k for k in (amb.get("strict_validation_keywords") or []) if isinstance(amb, dict) and isinstance(k, str)]
    prx = re.compile(_kw_rx(proof)[4:], re.I) if proof else None

    def names(e: dict) -> bool:
        if rx is None or not rx.search(e["title"]):
            return False
        if not use_rule or (prx is not None and prx.search(e["title"])):
            return True
        return bool(rm.should_include(sym, {"title": e["title"], "content": e["raw"], "url": e["url"], "date": e["ts"][:8]}))
    return names


def _neighbours(titles: list, term: str) -> Counter:
    """Words right before / after the term, once per headline ("before galaxy", "after jet")."""
    tt = [w.lower() for w in _TOK.findall(term)]
    out: Counter = Counter()
    for title in titles:
        toks = [w.lower() for w in _TOK.findall(title)]
        seen = set()
        for i in range(len(toks) - len(tt) + 1):
            if toks[i:i + len(tt)] != tt:
                continue
            for side, j in (("before", i - 1), ("after", i + len(tt))):
                if 0 <= j < len(toks) and toks[j] not in STOP and not toks[j].isdigit() and len(toks[j]) > 1:
                    seen.add(f"{side} {toks[j]}")
        out.update(seen)
    return out


CLAUDE_SYSTEM = "You check candidate keywords for a company news collector. Answer only in the given JSON schema."
CLAUDE_PROMPT = """Each candidate below is a name found in this week's headlines about a listed company. If it becomes
a keyword, every news headline that contains it is collected as news about that company.

For each one, say whether that would often collect news about something else: another company's product or
brand, a person with the same name, a place, a vehicle, an event, a sports team, a common phrase, a generic
term, or an older meaning. Judge the name as it appears in English-language news in general, not only this
week. risky = true when a noticeable share of headlines containing the name would not be about this company
or what it owns. other = that other meaning in at most 10 words (empty when not risky).

Candidates (id | company | name | kind | one headline):
{rows}"""
CLAUDE_SCHEMA = {"type": "object", "required": ["items"], "properties": {"items": {"type": "array", "items": {
    "type": "object", "required": ["id", "risky", "other"],
    "properties": {"id": {"type": "string"}, "risky": {"type": "boolean"}, "other": {"type": "string"}}}}}}


def _claude_bin() -> str:
    """As research/news_linkage/judge.py: the standalone CLI, else the one bundled with the VS Code extension."""
    for p in (os.getenv("CLAUDE_BIN"), shutil.which("claude"), str(Path.home() / ".local/bin/claude")):
        if p and Path(p).exists():
            return p
    bundled = sorted(Path.home().glob(".vscode/extensions/anthropic.claude-code-*/resources/native-binary/claude"))
    if bundled:
        return str(bundled[-1])
    raise FileNotFoundError("no claude CLI found")


def claude_check(cands: list) -> dict:
    """{candidate id: {"risky", "other"}} from `claude -p` on the subscription (API keys cleared)."""
    import subprocess
    import tempfile
    env = {k: v for k, v in os.environ.items() if k not in ("ANTHROPIC_API_KEY", "ANTHROPIC_AUTH_TOKEN")}
    out: dict = {}
    for i in range(0, len(cands), CLAUDE_BATCH):
        part = cands[i:i + CLAUDE_BATCH]
        rows = "\n".join(f"{c['id']} | {c['name']} ({c['symbol']}) | {c['term']} | {c.get('kind') or '-'} | "
                         f"{(c.get('examples') or [{}])[0].get('title', '')[:140]}" for c in part)
        cmd = [_claude_bin(), "-p", "--output-format", "json", "--json-schema", json.dumps(CLAUDE_SCHEMA), "--tools", "",
               "--strict-mcp-config", "--no-session-persistence", "--system-prompt", CLAUDE_SYSTEM]
        try:
            with tempfile.TemporaryDirectory(prefix="kw-screen-") as cwd:
                r = subprocess.run(cmd, input=CLAUDE_PROMPT.format(rows=rows), capture_output=True, text=True,
                                   timeout=CLAUDE_TIMEOUT_S, cwd=cwd, env=env)
            res = json.loads(r.stdout) if r.returncode == 0 else {}
            ans = res.get("structured_output") or json.loads(res.get("result") or "{}")
        except Exception as e:  # noqa: BLE001 - an unanswered term keeps no Claude verdict and is not auto-applied
            print(f"claude_check {i}: {type(e).__name__}: {e}", flush=True)
            continue
        ids = {c["id"] for c in part}
        for a in (ans or {}).get("items") or []:
            if a.get("id") in ids:
                out[a["id"]] = {"risky": bool(a.get("risky")), "other": str(a.get("other") or "")[:120]}
    return out


def screen(res: dict, use_claude: bool = True) -> dict:
    """Adds c["screen"] = {history, neighbours, others, flood, claude, risks} to the add / weak candidates."""
    cands = [c for c in res.get("candidates") or [] if c.get("tier") in SCREEN_TIERS]
    if not cands:
        return res
    t0 = time.time()
    rules = load_rules()
    created = datetime.fromisoformat((res.get("created") or now_iso())[:19])
    before = (created - timedelta(days=res.get("days") or DAYS)).strftime("%Y%m%d%H%M%S")
    hist = history_titles(sorted({c["term"] for c in cands}), before)
    print(f"screen: {len(cands)} candidates, {sum(map(len, hist.values()))} past headlines ({time.time() - t0:.0f}s)", flush=True)
    other_kw: dict = defaultdict(set)
    for (_rs, sym), (_p, cfg, _m, _k, _rm) in rules.items():
        for k in cfg.get("primary_keywords") or []:
            if isinstance(k, str) and len(k) > 3 and not _weak_word(k):
                other_kw[k.lower()].add(sym)
    orx = re.compile(_kw_rx(other_kw)[4:], re.I)
    tests: dict = {}
    verdicts = claude_check(cands) if use_claude else {}
    for c in cands:
        key = (c["set"], c["symbol"])
        if key not in rules:
            c["screen"] = {"at": now_iso(), "risks": [{"code": "no_rule", "text": "the company's rule file is missing"}]}
            continue
        if key not in tests:
            _p, cfg, match, _k, rm = rules[key]
            tests[key] = _company_test(cfg, match, rm, c["symbol"])
        rows = hist.get(c["term"].lower(), [])
        flags = [tests[key](e) for e in rows]
        extra = [e for e, f in zip(rows, flags) if not f]
        n, w = len(rows), sum(flags)
        first = min((e["ts"] for e in rows), default=None)
        span = max(1.0, (created - datetime.strptime(first[:8], "%Y%m%d")).days / 30.4) if first else 1.0
        per_month = round(len(extra) / span, 1)
        nb = _neighbours([e["title"] for e in extra], c["term"]).most_common(3)
        oc: Counter = Counter()
        for e in extra:
            oc.update({s for m in orx.findall(e["title"]) for s in other_kw.get(m.lower(), ()) if s != c["symbol"]})
        risks = []
        if n >= RISK_MIN_HIST and w / n < RISK_CO:
            risks.append({"code": "history", "text": f"named the company in {w / n:.0%} of {n} headlines over {HISTORY_MONTHS} months"})
        if len(extra) >= MIN_EXTRA and nb and nb[0][1] / len(extra) >= NEIGHBOUR_SHARE:
            side, word = nb[0][0].split(" ", 1)
            phrase = f"{word} {c['term']}" if side == "before" else f"{c['term']} {word}"
            risks.append({"code": "neighbour", "text": f'other headlines mostly say "{phrase}" ({nb[0][1] / len(extra):.0%})'})
        if len(extra) >= MIN_EXTRA and oc and oc.most_common(1)[0][1] / len(extra) >= OTHER_SHARE:
            s, k = oc.most_common(1)[0]
            risks.append({"code": "other_company", "text": f"{k / len(extra):.0%} of the other headlines are {s} news"})
        if per_month >= FLOOD_PER_MONTH:
            risks.append({"code": "flood", "text": f"{per_month:.0f} headlines a month without the company"})
        v = verdicts.get(c["id"])
        if v and v["risky"]:
            risks.append({"code": "claude", "text": f"Claude: {v['other'] or 'other meanings'}"})
        c["screen"] = {"at": now_iso(), "history_months": HISTORY_MONTHS, "hist_headlines": n, "hist_with_company": w,
                       "hist_co_mention": round(w / n, 3) if n else None, "extra_per_month": per_month,
                       "neighbours": [[k, round(x / len(extra), 2)] for k, x in nb] if extra else [],
                       "others": [[s, round(x / len(extra), 2)] for s, x in oc.most_common(3)] if extra else [],
                       "extra_examples": [{"title": e["title"], "url": e["url"], "ts": e["ts"]} for e in extra[:4]],
                       "claude": v, "risks": risks}
    res["screened"] = now_iso()
    print(f"screen: done in {time.time() - t0:.0f}s; with a risk {sum(bool(c['screen']['risks']) for c in cands)} of {len(cands)}, "
          f"Claude answered {len(verdicts)}", flush=True)
    return res


# ---------------------------------------------------------------- decisions -> rule files
def _read(p: Path, default):
    try:
        return json.loads(p.read_text())
    except (OSError, ValueError):
        return default


def proposals() -> dict:
    """Every judged proposal of every run, newest run winning: {id: proposal}."""
    out = {}
    for p in sorted((OUT / "runs").glob("????-??-??.json")):
        for c in _read(p, {}).get("candidates", []):
            out[c["id"]] = {**c, "run": p.stem}
    return out


def _dump_like(obj: dict, original: str) -> str:
    """JSON in the original file's style (indent, \\u escapes, final newline), so a git diff shows only the change."""
    lines = original.splitlines()
    indent = len(lines[1]) - len(lines[1].lstrip(" ")) if len(lines) > 1 else 2
    text = json.dumps(obj, indent=indent or 2, ensure_ascii="\\u" in original)
    return text + ("\n" if original.endswith("\n") else "")


def apply(dry_run: bool = False) -> list:
    """Makes each rule's auto_keywords match the decisions: added, moved between keyword and weak, or taken out."""
    decisions = _read(OUT / "decisions.json", {})
    props = proposals()
    want: dict = defaultdict(dict)                     # (set, sym) -> {term: entry}
    for pid, d in decisions.items():
        p = props.get(pid)
        if not p or d.get("decision") not in ("add", "weak"):
            continue
        want[(p["set"], p["symbol"])][p["term"]] = {"term": p["term"], "as": "keyword" if d["decision"] == "add" else "weak",
                                                    "kind": p.get("kind"), "first_seen": p.get("first_seen", "")[:8],
                                                    "week": p["run"], "approved": d.get("at"), "id": pid,
                                                    "history": d.get("history") or default_history(p.get("kind"))}
    changes = []
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    syms = set(want) | {(rs, p.stem) for rs, d in RULE_SETS.items() if d.exists() for p in d.glob("*.json")
                        if "auto_keywords" in _read(p, {})}
    for rs, sym in sorted(syms):
        path = RULE_SETS[rs] / f"{sym}.json"
        rule = _read(path, None)
        if rule is None:
            continue
        old = {a["term"]: a for a in rule.get("auto_keywords") or []}
        new = want.get((rs, sym), {})
        if {t: (a["as"], a.get("history")) for t, a in old.items()} == {t: (a["as"], a.get("history")) for t, a in new.items()}:
            continue
        exp = [k for k in rule.get("expansion_keywords") or [] if not (k in old and old[k]["as"] == "keyword")]
        weak = [k for k in rule.get("weak_names_not_counted") or [] if not (k in old and old[k]["as"] == "weak")]
        exp += [t for t, a in new.items() if a["as"] == "keyword" and t not in exp]
        weak += [t for t, a in new.items() if a["as"] == "weak" and t not in weak]
        diff = {"set": rs, "symbol": sym, "added": sorted(set(new) - set(old)), "removed": sorted(set(old) - set(new)),
                "moved": sorted(t for t in set(old) & set(new) if old[t]["as"] != new[t]["as"])}
        changes.append(diff)
        if dry_run:
            continue
        bdir = OUT / "backups" / stamp / rs
        bdir.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, bdir / path.name)
        rule["expansion_keywords"] = exp
        if weak or "weak_names_not_counted" in _read(bdir / path.name, {}):
            rule["weak_names_not_counted"] = weak
        else:
            rule.pop("weak_names_not_counted", None)
        if new:
            rule["auto_keywords"] = sorted(new.values(), key=lambda a: a["term"])
        else:
            rule.pop("auto_keywords", None)
        tmp = path.with_suffix(".json.tmp")
        tmp.write_text(_dump_like(rule, (bdir / path.name).read_text()))
        tmp.rename(path)
    if changes and not dry_run:
        with (OUT / "applied.jsonl").open("a") as f:
            for c in changes:
                f.write(json.dumps({"at": now_iso(), **c}, ensure_ascii=False) + "\n")
    print(json.dumps({"dry_run": dry_run, "changed": changes}, ensure_ascii=False), flush=True)
    return changes


# ---------------------------------------------------------------- automatic decisions and their check (2026-11-19)
# auto_decide: a candidate the model suggests adding (tier "add"), seen in AUTO_MIN_STORIES stories on
# AUTO_MIN_SITES sites, screened with no risk and answered by Claude, that the user has not decided, is
# added with by="auto" and history "first_seen" (no years of past news until audit() confirms it). Only
# when <OUT>/settings.json has "auto_apply": true; otherwise <OUT>/auto_preview.json lists what it would add.
# The user's decision always wins: an entry without by="auto" is never touched.
# audit: for every keyword in the rules, the core articles collected since it went in that contain it but
# none of the company's own names, i.e. the ones it brought in, and the share the relevance model kept. An
# auto-added term is taken out again (reject, by="auto") when AUDIT_MIN_N of them were kept less than
# AUDIT_KEEP_MIN, or when more than FLOOD_PER_WEEK come in a week; after CONFIRM_DAYS it is confirmed, and
# a product / brand / service then gets its full history searched. -> <OUT>/audit.json
AUTO_MIN_STORIES, AUTO_MIN_SITES = 3, 3
AUDIT_MIN_N, AUDIT_KEEP_MIN, AUDIT_KEEP_OK = 10, 0.3, 0.5
CONFIRM_DAYS, FLOOD_PER_WEEK = 14, 200


def settings() -> dict:
    return _read(OUT / "settings.json", {})


def _write_decisions(change) -> dict:
    """change(decisions) under quant_ai's lock (markets/keyword_review.py writes the same file)."""
    import fcntl
    OUT.mkdir(parents=True, exist_ok=True)
    with open(OUT / "decisions.lock", "w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        cur = _read(OUT / "decisions.json", {})
        change(cur)
        tmp = OUT / "decisions.json.tmp"
        tmp.write_text(json.dumps(cur, indent=1, ensure_ascii=False))
        tmp.rename(OUT / "decisions.json")
    return cur


def auto_eligible(p: dict) -> bool:
    s = p.get("screen") or {}
    return (p.get("tier") == "add" and "risks" in s and not s["risks"] and bool(s.get("claude"))
            and (p.get("stories") or 0) >= AUTO_MIN_STORIES and (p.get("sites") or 0) >= AUTO_MIN_SITES)


def auto_decide(dry_run: bool = False) -> list:
    props, decisions = proposals(), _read(OUT / "decisions.json", {})
    picks = sorted(pid for pid, p in props.items() if pid not in decisions and auto_eligible(p))
    on = bool(settings().get("auto_apply")) and not dry_run
    preview = {"at": now_iso(), "auto_apply": on, "ids": picks}
    tmp = OUT / "auto_preview.json.tmp"
    tmp.write_text(json.dumps(preview, indent=1, ensure_ascii=False))
    tmp.rename(OUT / "auto_preview.json")
    if on and picks:
        now = datetime.now().isoformat(timespec="seconds")

        def add(cur):
            for pid in picks:
                cur.setdefault(pid, {"decision": "add", "at": now, "by": "auto", "history": "first_seen"})
        _write_decisions(add)
    print(f"auto_decide: {len(picks)} eligible, {'added' if on else 'preview only'}", flush=True)
    return picks


def _own_names(rs: str, sym: str):
    """The company's names before any refresh term: v2's keywords and short name, the rule's primary keywords."""
    import slm_company_match_v2 as m2
    prim = [k for k in (_read(RULE_SETS[rs] / f"{sym}.json", {}).get("primary_keywords") or []) if isinstance(k, str) and len(k) > 2]
    pats = [p for p in (m2._SCAN_PATTERNS.get(sym), m2._NAME_PATTERNS.get(sym)) if p]
    if prim:
        pats.append(re.compile(r"(?<!\w)(" + "|".join(re.escape(k) for k in sorted(prim, key=len, reverse=True)) + r")(?!\w)"))
    return lambda text: any(p.search(text) for p in pats)


def audit(dry_run: bool = False) -> dict:
    sys.path.insert(0, str(ROOT / "research" / "labeling"))
    from pymongo import MongoClient
    import slm_company_match_v2 as m2
    entries = []
    for rs, d in RULE_SETS.items():
        for p in sorted(d.glob("*.json")) if d.exists() else []:
            for a in _read(p, {}).get("auto_keywords") or []:
                if a.get("as") == "keyword" and a.get("approved"):
                    entries.append((rs, p.stem, a))
    report: dict = {"at": now_iso(), "terms": {}}
    if entries:
        db = MongoClient(os.environ["MONGO_URI"])[m2.DB_NAME]
        since = min(a["approved"][:10].replace("-", "") for _rs, _s, a in entries)
        core = sorted({s for rs, s, _a in entries if rs == "core"})
        docs = defaultdict(list)
        proj = {"symbol": 1, "url": 1, "date": 1, "title": 1,
                "content": {"$substrCP": [{"$ifNull": ["$content", ""]}, 0, m2.CONTENT_CHARS]}}
        for d in db[m2.SOURCE_COLLECTION].aggregate([{"$match": {"date": {"$gte": since}, "symbol": {"$in": core},
                                                                 "origin": {"$ne": m2.UNIVERSE_ORIGIN}}}, {"$project": proj}],
                                                    allowDiskUse=True, batchSize=5000):
            docs[d["symbol"]].append(d)
        kept = {(r["symbol"], r["url"]) for r in db[m2.TARGET_COLLECTION].find(
            {"symbol": {"$in": core}, "date": {"$gte": since}}, {"symbol": 1, "url": 1})}
        decisions = _read(OUT / "decisions.json", {})
        now = datetime.now()
        for rs, sym, a in entries:
            pid = a.get("id") or f"{rs}:{sym}:{a['term'].lower()}"
            age = (now - datetime.fromisoformat(a["approved"][:19])).days
            row = {"set": rs, "symbol": sym, "term": a["term"], "approved": a["approved"], "days": age,
                   "by": (decisions.get(pid) or {}).get("by") or "user"}
            if rs == "core":
                own, pat = _own_names(rs, sym), re.compile(r"(?<!\w)" + re.escape(a["term"]) + r"(?!\w)", re.I)
                day0 = a["approved"][:10].replace("-", "")
                mine_ = [d for d in docs.get(sym, []) if (d.get("date") or "") >= day0
                         and pat.search(f"{d.get('title') or ''} {d.get('content') or ''}")
                         and not own(f"{d.get('title') or ''} {d.get('content') or ''}")]
                n, k = len(mine_), sum((sym, d["url"]) in kept for d in mine_)
                row.update(brought_in=n, kept=k, keep_rate=round(k / n, 3) if n else None,
                           per_week=round(n / max(1, age) * 7, 1),
                           examples=[{"title": d.get("title"), "url": d.get("url"), "kept": (sym, d["url"]) in kept} for d in mine_[:5]])
                if row["by"] == "auto" and ((n >= AUDIT_MIN_N and k / n < AUDIT_KEEP_MIN) or (age >= 1 and row["per_week"] > FLOOD_PER_WEEK)):
                    row["verdict"] = "revert"
                    row["reason"] = (f"{row['per_week']:.0f} articles a week" if row["per_week"] > FLOOD_PER_WEEK
                                     else f"the relevance model kept {k} of {n}")
                elif age >= CONFIRM_DAYS and (n < AUDIT_MIN_N or k / n >= AUDIT_KEEP_OK):
                    row["verdict"] = "confirmed"
                elif n >= AUDIT_MIN_N and k / n < AUDIT_KEEP_MIN:
                    row["verdict"] = "poor"                   # a term the user added: shown, not undone
                else:
                    row["verdict"] = "watching"
            else:
                row["verdict"] = "confirmed" if age >= CONFIRM_DAYS else "watching"   # universe rows: kept at the backfill
            report["terms"][pid] = row
        if not dry_run:
            stamp = datetime.now().isoformat(timespec="seconds")

            def change(cur):
                for pid, row in report["terms"].items():
                    d = cur.get(pid) or {}
                    if d.get("by") != "auto":
                        continue
                    if row["verdict"] == "revert" and d.get("decision") == "add":
                        cur[pid] = {**d, "decision": "reject", "at": stamp, "reason": row["reason"]}
                    elif row["verdict"] == "confirmed" and not d.get("confirmed"):
                        p = proposals().get(pid) or {}
                        cur[pid] = {**d, "confirmed": stamp,
                                    **({"history": "full"} if p.get("kind") in FULL_HISTORY_KINDS else {})}
            if any(r["verdict"] in ("revert", "confirmed") for r in report["terms"].values()):
                _write_decisions(change)
    if not dry_run:
        tmp = OUT / "audit.json.tmp"
        tmp.write_text(json.dumps(report, indent=1, ensure_ascii=False, default=str))
        tmp.rename(OUT / "audit.json")
    print(f"audit: {len(report['terms'])} terms; " + json.dumps(Counter(r["verdict"] for r in report["terms"].values())), flush=True)
    return report


# ---------------------------------------------------------------- past months for added terms
RESCAN_SINCE = os.getenv("KEYWORD_RESCAN_SINCE", "2018-01-01")  # all news from 2018 (2026-11-05)
FETCH_CHUNK = 200
RESCAN_FETCH = os.getenv("KEYWORD_RESCAN_FETCH", "1") == "1"      # 0: core rows kept, not downloaded (tests)


class _bf_settings:
    """backfill_symbol_news reads its rules dir / collection / title gate / work dir from module globals;
    set them for one rule set and put them back afterwards (the universe window process has its own)."""

    def __init__(self, **kw):
        import backfill_symbol_news as bf
        self.bf, self.kw, self.old = bf, kw, {}

    def __enter__(self):
        for k, v in self.kw.items():
            self.old[k] = getattr(self.bf, k)
            setattr(self.bf, k, v)
        return self.bf

    def __exit__(self, *exc):
        for k, v in self.old.items():
            setattr(self.bf, k, v)


# A product's name is in no headline before it exists, so its whole history is safe to scan. A subsidiary
# or an executive is not: "Instacart" in 2021-2025 is Instacart's news, not DoorDash's (deal announced in
# the week of 2026-11-04), and a new CEO's past news is the old employer's. Those start at first sight.
FULL_HISTORY_KINDS = {"product", "brand", "service"}


def default_history(kind: str | None) -> str:
    """"full" (past months from RESCAN_SINCE) or "first_seen"; the user can change it on the Keywords tab."""
    return "full" if kind in FULL_HISTORY_KINDS else "first_seen"


def _rescan_since(a: dict) -> str:
    if (a.get("history") or default_history(a.get("kind"))) == "full":
        return RESCAN_SINCE
    f = (a.get("first_seen") or "") or (a.get("approved") or "").replace("-", "")[:8]
    return f"{f[:4]}-{f[4:6]}-{f[6:8]}" if len(f) >= 8 else datetime.now().strftime("%Y-%m-%d")


def _rescan_pending() -> dict:
    """{(set, since): [(symbol, term, id)]} for terms in the rules as keywords whose past months are not
    scanned yet."""
    done = _read(OUT / "rescan.json", {})
    out: dict = defaultdict(list)
    for rs, d in RULE_SETS.items():
        if not d.exists():
            continue
        for p in sorted(d.glob("*.json")):
            for a in _read(p, {}).get("auto_keywords") or []:
                pid = a.get("id") or f"{rs}:{p.stem}:{a['term'].lower()}"
                since = _rescan_since(a)
                # keyed by id and start: a term switched to the full history later is scanned again
                if a.get("as") == "keyword" and f"{pid}@{since}" not in done:
                    out[(rs, since)].append((p.stem, a["term"], pid))
    return out


def _universe_merge(bf, sym: str, rows: list) -> int:
    """New kept rows of a universe company into the backfill's own state: kept.jsonl, and once indexed,
    the month's label todo list; a month already written is written again by the next label run (its
    labelled headlines are kept in <month>.keys.jsonl, so only the new ones go to the model)."""
    import backfill_universe_news as bu
    kept = bu.WORK / sym / "kept.jsonl"
    have = {json.loads(l)["url"] for l in kept.open()} if kept.exists() else set()
    new = [r for r in rows if r["url"] not in have]
    if not new:
        return 0
    kept.parent.mkdir(parents=True, exist_ok=True)
    with kept.open("a") as f:
        for r in new:
            f.write(json.dumps(r) + "\n")
    todo = bu.LABELS / "todo"
    if (todo / "index.done").exists():
        months = set()
        for r in new:
            title = bf.gkg_title(r["raw"])
            m = f"{str(r['ts'])[:4]}-{str(r['ts'])[4:6]}"
            months.add(m)
            with (todo / f"{m}.jsonl").open("a") as f:
                f.write(json.dumps({"symbol": sym, "url": r["url"], "ts": str(r["ts"]), "title": title,
                                    "key": bu.key_of(sym, title)}) + "\n")
        for m in months:
            final = bu.LABELS / f"{m}.jsonl"
            if final.exists():
                final.unlink()
    return len(new)


def rescan(deadline: float | None = None) -> bool:
    """Past months (from _rescan_since) for every added keyword not rescanned yet, one GKG read per month for
    all of them. core: the collector's rules, then download into news_articles (the nightly company match
    and enrichment take them from there). universe: rules + headline gate, current month left out (the
    backfill leaves it out too), merged into the headline-label backfill. Stops between months, companies
    and download chunks at `deadline`; resumable. False when it stopped early."""
    import hashlib
    pending = _rescan_pending()
    if not pending:
        return True
    state = _read(OUT / "rescan.json", {})
    for (rs, since), items in sorted(pending.items()):
        batch = hashlib.sha1("|".join(sorted(i for _s, _t, i in items)).encode()).hexdigest()[:10]
        work = OUT / "rescan" / rs / f"{since}-{batch}"
        work.mkdir(parents=True, exist_ok=True)
        kws: dict = defaultdict(list)
        for sym, term, _i in items:
            kws[sym].append(term)
        print(f"rescan {rs}: {sum(map(len, kws.values()))} terms of {len(kws)} companies since {since} -> {work}", flush=True)
        uni = rs == "universe"
        settings = dict(WORK_DIR=str(work), RULES_DIR=str(RULE_SETS[rs]) if uni else None,
                        NEWS_COLLECTION="news_articles",   # one store for both since 2026-11-10
                        ORIGIN="universe_backfill" if uni else None,
                        TITLE_GATE=True)                # the core collector checks headlines too since 2026-11-04
        with _bf_settings(**settings) as bf:
            if not bf.scan_many(dict(kws), since, deadline, before_month=datetime.now().strftime("%Y-%m") if uni else "9999-99"):
                return False
            for sym in sorted(kws):
                if deadline and time.time() >= deadline:
                    return False
                cand, kept = work / sym / "candidates.jsonl", work / sym / "kept.jsonl"
                if not cand.exists():
                    cand.touch()
                if not (work / sym / "filter.done").exists():
                    bf.filter_stage(sym, cand, kept, WORKERS)
                    kept.touch()
                    (work / sym / "filter.done").write_text(now_iso())
                rows = [json.loads(l) for l in kept.open()]
                n_new = 0
                if uni:
                    if not (work / sym / "merged.done").exists():
                        n_new = _universe_merge(bf, sym, rows)
                        (work / sym / "merged.done").write_text(str(n_new))
                    n_new = int((work / sym / "merged.done").read_text() or 0)
                else:
                    name = (_read(RULE_SETS[rs] / f"{sym}.json", {}).get("name")) or sym
                    for i in range(0, len(rows) if RESCAN_FETCH else 0, FETCH_CHUNK):
                        if deadline and time.time() >= deadline:
                            return False
                        part = work / sym / f"kept.part{i // FETCH_CHUNK:04d}.jsonl"
                        if not part.exists():
                            part.write_text("".join(json.dumps(r) + "\n" for r in rows[i:i + FETCH_CHUNK]))
                        bf.fetch_stage(sym, name, part, 12)
                    n_new = len(rows)
                cands = sum(1 for _ in cand.open())
                for s2, term, pid in items:
                    if s2 == sym:
                        state[f"{pid}@{since}"] = {"id": pid, "at": now_iso(), "set": rs, "symbol": sym, "term": term, "since": since,
                                      "candidates_for_company_batch": cands, "kept": len(rows), "new_rows": n_new,
                                      "work": str(work / sym)}
                tmp = OUT / "rescan.json.tmp"
                tmp.write_text(json.dumps(state, indent=1, ensure_ascii=False))
                tmp.rename(OUT / "rescan.json")
                print(f"rescan {rs} {sym} {kws[sym]}: {cands} candidates, {len(rows)} kept, {n_new} new", flush=True)
    return True


def status() -> dict:
    props, dec = proposals(), _read(OUT / "decisions.json", {})
    out = {"runs": sorted(p.stem for p in (OUT / "runs").glob("????-??-??.json")) if OUT.exists() else [],
           "proposals": len(props), "tiers": dict(Counter(p.get("tier") for p in props.values())),
           "decided": dict(Counter(d["decision"] for d in dec.values() if d.get("decision"))),
           "pending": sum(1 for k, p in props.items() if k not in dec and p.get("tier") in ("add", "weak")),
           "rescan_pending": {f"{rs} since {since}": len(v) for (rs, since), v in _rescan_pending().items()},
           "rescanned": len(_read(OUT / "rescan.json", {}))}
    print(json.dumps(out), flush=True)
    return out


def _until(argv: list) -> float | None:
    if "--until" not in argv:
        return None
    h, m = map(int, argv[argv.index("--until") + 1].split(":"))
    return datetime.now().replace(hour=h, minute=m, second=0, microsecond=0).timestamp()


def _write_run(res: dict, suffix: str = "") -> Path:
    p = OUT / "runs" / f"{datetime.now():%Y-%m-%d}{suffix}.json"
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(".tmp")
    tmp.write_text(json.dumps(res, indent=1, ensure_ascii=False))
    tmp.rename(p)
    print(f"wrote {p}", flush=True)
    return p


def main(argv: list) -> None:
    stage = argv[0] if argv else "status"
    if stage == "mine":
        _write_run(mine(), ".candidates")
    elif stage == "judge":
        p = OUT / "runs" / f"{datetime.now():%Y-%m-%d}.candidates.json"
        _write_run(judge(_read(p, {"candidates": []}), _until(argv)))
    elif stage == "run":
        res = mine()
        _write_run(res, ".candidates")
        _write_run(judge(res, _until(argv)))
    elif stage == "screen":
        runs = sorted((OUT / "runs").glob("????-??-??.json"))
        p = OUT / "runs" / f"{argv[argv.index('--run') + 1]}.json" if "--run" in argv else (runs[-1] if runs else None)
        if p is None or not p.exists():
            raise SystemExit(f"no run to screen: {p}")
        res = screen(_read(p, {}), use_claude="--no-claude" not in argv)
        tmp = p.with_suffix(".tmp")
        tmp.write_text(json.dumps(res, indent=1, ensure_ascii=False))
        tmp.rename(p)
        print(f"wrote {p}", flush=True)
    elif stage == "auto":
        auto_decide(dry_run=True)
    elif stage == "audit":
        audit(dry_run="--dry-run" in argv)
    elif stage == "apply":
        if "--dry-run" not in argv:
            auto_decide()
        apply(dry_run="--dry-run" in argv)
    elif stage == "rescan":
        rescan(_until(argv))
    elif stage == "status":
        status()
    else:
        raise SystemExit(__doc__)


if __name__ == "__main__":
    main(sys.argv[1:])
