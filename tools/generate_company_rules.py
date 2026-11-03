"""Company rule files for the wider universe, drafted by the local model (PROJECT_PLAN Q.16 news, 2026-11-02).

The 102 hand-written rules in collectors/news/gdelt/company_rules decide which GDELT articles are about
a company: distinctive names, executives and products, and for names that are also common words (Apple,
Target, Seagate ...) the words that must appear. The 499 companies added with the S&P 500 history need
the same before their news can be backfilled. Each is drafted from the company's name and sector by
the local model, checked mechanically (JSON shape, no bare common words as keywords, the ticker not
used alone when it is a word), and written to a separate directory, so the nightly collector does not
start matching them unasked.

Usage: .venv311/bin/python tools/generate_company_rules.py [SYMBOLS...]   (default: universe needs_rules.txt)
"""

from __future__ import annotations

import json
import os
import re
import sys
import time
from pathlib import Path

import requests

OUT = Path(os.getenv("UNIVERSE_RULES_DIR", "/Volumes/Data4T/universe/company_rules"))
NAMES = Path("/Volumes/Data4T/universe/names.json")
LM = os.getenv("LM_STUDIO_URL", "http://localhost:1234/v1").rstrip("/")
MODEL = os.getenv("RULES_LLM", "qwen/qwen3.8-27b")
COMMON = {"inc", "corp", "group", "holdings", "company", "energy", "health", "capital", "financial", "global", "american",
          "international", "united", "general", "first", "national", "systems", "technologies", "services", "solutions",
          "the", "and", "trust", "bank", "partners", "resources", "industries", "brands", "foods", "power", "water", "data"}

EXAMPLE = {"symbol": "STX", "name": "Seagate Technology", "primary_keywords": ["Seagate"],
           "expansion_keywords": ["IronWolf", "Dave Mosley", "Gianluca Romano", "Lyve Cloud"],
           "special_rules": {"ambiguous": {"problem": "Seagate is also a street name and a homebuilder; STX is also a Korean shipbuilder",
                                           "required_keywords": ["Seagate", "IronWolf", "Dave Mosley", "Lyve Cloud"], "min_matches": 1}}}


def names(syms: list) -> dict:
    have = json.loads(NAMES.read_text()) if NAMES.exists() else {}
    import yfinance as yf
    for s in syms:
        if s in have:
            continue
        try:
            info = yf.Ticker(s).info
            have[s] = {"name": info.get("longName") or info.get("shortName") or s, "sector": info.get("sector"),
                       "industry": info.get("industry")}
        except Exception:  # noqa: BLE001 - delisted names often have no info: the ticker stands in
            have[s] = {"name": s, "sector": None, "industry": None}
        NAMES.write_text(json.dumps(have, indent=1))
        time.sleep(0.3)
    return have


def draft(sym: str, info: dict) -> dict:
    prompt = (f"Write a news-matching rule for {info['name']} (ticker {sym}; sector {info.get('sector')}; industry {info.get('industry')}).\n"
              "Articles are matched when they mention a keyword. Give:\n"
              "- primary_keywords: 1-3 names that identify this company in news (its common name, e.g. 'Seagate' not 'Seagate Technology "
              "Holdings plc'; the ticker only if it is not an ordinary word)\n"
              "- expansion_keywords: 4-12 distinctive terms: current and recent CEO / CFO names, flagship products or brands, "
              "well-known subsidiaries. No generic words (like 'revenue', 'energy', 'cloud')\n"
              "- special_rules.ambiguous: only if the company's name is also an ordinary word, a place, a person's name or another "
              "company's name: {problem, required_keywords (terms that prove it is this company), min_matches: 1}\n"
              f"Example for Seagate:\n{json.dumps(EXAMPLE)}\n"
              "Reply with the JSON object only, same keys (symbol, name, primary_keywords, expansion_keywords, special_rules).")
    r = requests.post(f"{LM}/chat/completions", timeout=300, json={
        "model": MODEL, "temperature": 0.1, "max_tokens": 900,
        "messages": [{"role": "user", "content": prompt}, {"role": "assistant", "content": "<think>\n\n</think>\n\n"}]})
    txt = re.sub(r"<think>.*?</think>", "", r.json()["choices"][0]["message"]["content"], flags=re.S)
    return json.loads(txt[txt.find("{"): txt.rfind("}") + 1])


def check(sym: str, rule: dict, info: dict) -> list:
    """Mechanical checks; the list of problems (empty = accepted)."""
    probs = []
    rule["symbol"] = sym
    rule.setdefault("name", info["name"])
    prim = [k for k in rule.get("primary_keywords") or [] if isinstance(k, str) and k.strip()]
    exp = [k for k in rule.get("expansion_keywords") or [] if isinstance(k, str) and k.strip()]
    bad = [k for k in prim + exp if k.lower().strip() in COMMON or len(k.strip()) < 3]
    if bad:
        probs.append(f"common or too short keywords {bad}")
        prim = [k for k in prim if k not in bad]
        exp = [k for k in exp if k not in bad]
    if not prim:
        probs.append("no primary keyword")
    rule["primary_keywords"], rule["expansion_keywords"] = prim, exp
    # The local model invents executives and brands (11-02 test: a made-up Target CFO, DeWalt and Fabletics
    # as Target brands, wrong MercadoLibre executives), and a wrong expansion keyword pulls in another
    # company's news. So only words built on the company's own name are kept: no expansion keywords, and
    # required words only when they contain a primary keyword. Fewer articles, none about someone else.
    if rule["expansion_keywords"]:
        rule["dropped_expansion"] = rule["expansion_keywords"]
    rule["expansion_keywords"] = []
    amb = (rule.get("special_rules") or {}).get("ambiguous")
    if amb:
        roots = [k.lower() for k in prim if k != sym]
        req = [k for k in amb.get("required_keywords") or [] if isinstance(k, str) and any(r in k.lower() for r in roots) and k.lower() not in roots]
        long_names = [k for k in prim if len(k.split()) > 1 or len(k) >= 8]
        amb["required_keywords"] = sorted(set(req + [info["name"]] + long_names))
        amb["min_matches"] = 1
    if amb and not amb.get("required_keywords"):
        probs.append("ambiguous without required keywords")
    if not amb:
        rule["special_rules"] = {}
    # a ticker that is an English word must not match alone
    if any(k == sym and sym.lower() in {"all", "it", "key", "hum", "now", "low", "cost", "main", "fast", "well", "on", "a"} for k in prim):
        probs.append(f"ticker {sym} is a word")
    return probs


# ---------------------------------------------------------------- tighten (2026-11-03)
# tools/validate_universe_rules.py on three sample months found the drafted rules unusable: 265k rows kept
# (6.3M estimated since 2021), CPB / MSTR / FOX / FERG / PGR thousands a month of unrelated news. Causes:
#   - check() put every name of 8+ letters into required_keywords, so the ambiguous word itself
#     (Campbell, Strategy, Progressive, Ferguson) counted as proof, and overlapping names ("Campbell" inside
#     "Campbell's") counted twice, which the engine treats as two identity hits and passes without the SLM
#   - bare tickers as keywords, matched case-insensitively: UPS, ICE, POOL, NOV, DASH, HOOD
#   - yfinance names of tickers reused since: FB is a ProShares ETF, INFO a Harbor ETF, PARA "Parabolic
#     Technologies", LB LandBridge, SATS "Strive"
# tighten() rewrites every rule from its own names without the model: tickers out (unless the ticker is
# the company's everyday name), the words below never count alone, a name that contains another is dropped
# (one mention, one hit). The weak words are left out (kept in weak_names_not_counted for a later pass with
# the SLM on a bigger machine); a company known only by weak names gets an "ambiguous" rule: one hit goes to
# the SLM in the real filter, two different names pass.

# Names that are also ordinary words, surnames, places, other organisations or index / exchange names.
AMBIGUOUS = {
    "abbott", "anthem", "apollo", "ares", "arista", "axon", "baxter", "cadence", "caesars", "campbell", "campbell's",
    "carrier", "casey's", "charter", "chevron", "chubb", "citi", "coherent", "cognizant", "coke", "colgate", "corning",
    "cummins", "darden", "deckers", "delta", "dover", "dupont", "eastman", "eaton", "emerson", "essex", "ferguson",
    "flex", "ford", "gallagher", "grainger", "hartford", "hilton", "hubbell", "jacobs", "leggett", "lennox", "lowe",
    "lowe's", "lumen", "merck", "molina", "mosaic", "nasdaq", "nasdaq-100", "newell", "occidental", "otis", "pmi",
    "progressive", "quanta", "quest", "robinhood", "rollins", "spectrum", "strategy", "strive", "synchrony",
    "tapestry", "travelers", "trimble", "tyson", "ventas", "vulcan", "waters", "the hartford",
    "fox news", "state street", "southern co", "southern company", "public storage",
    "realty income", "principal financial", "old dominion", "jack henry", "iron mountain", "raymond james",
    "best buy", "global payments", "air products", "msci", "cboe",
    # tickers that read as words or other things, even where they are the company's name
    "ups", "ice", "pool", "nov", "fast", "dash", "hood", "tko", "erie", "tel", "fmc", "fis", "cme", "eqt", "aes",
    "ptc", "nrg", "dte", "mgm", "echo", "flex", "key", "well", "wm", "yum", "zion", "block", "square", "xylem",
    "zebra", "whirlpool", "wynn", "berkley", "williams", "penn", "mccormick", "3m", "first solar", "kkr", "lincoln national", "the southern", "waste management", "albemarle", "take two",
    "pdd", "pvh", "expand energy", "costar", "hewlett packard",
}
# Terms that name another company or are not names at all.
JUNK = {"TJX": {"Macy's"}, "HIG": {"Hartford Fire"}, "J": {"Jacobs Data"}, "VMRK": {"Vivint"}, "CZR": {"Caesars Superdome"},
        "WHR": {"KitchenAid", "Maytag"}, "PVH": {"Calvin Klein", "Tommy Hilfiger"}, "IEX": None, "Q": {"Qnity Electronics (Q)"}, "NDAQ": {"Nasdaq-100 ETF"}}
JUNK_RE = {"IEX": re.compile(r"^IDEX XE\d*$")}
# an index, fund or contract named after the company is quoted in every market report, not news of it
PRODUCT_RE = re.compile(r"(?<!\w)(Index|Indices|Futures|Options|ETFs?|Fund)(?!\w)", re.I)
# Not news targets of their own: a reused ticker (the company it was is elsewhere or has no bars in its
# index years) or a second share class of a company that has a rule already.
DROP = {"FB": "META", "INFO": None, "LB": None, "SATS": None, "PARA": None,
        "GOOG": "GOOGL", "FOX": "FOXA", "NWS": "NWSA", "UA": "UAA", "TGT": "TGT"}
# former names since 2021, so the years before a rename are found too
EXTRA = {"SLB": ["Schlumberger"], "CPB": ["Campbell Soup"], "ELV": ["Anthem Inc", "Anthem, Inc."],
         "COR": ["AmerisourceBergen"], "RVTY": ["PerkinElmer"], "CPAY": ["FleetCor"], "EG": ["Everest Re"],
         "XYZ": ["Square Inc", "Square, Inc."], "EXE": ["Chesapeake Energy"], "LUMN": ["CenturyLink"],
         "WBD": ["WarnerMedia", "Discovery Inc", "Discovery, Inc."], "SW": ["WestRock", "Smurfit Kappa"],
         "GEN": ["NortonLifeLock"], "DOC": ["Healthpeak"], "META": [], "BALL": ["Ball Corporation"],
         "ADP": ["ADP"], "NOV": ["National Oilwell Varco"], "TPR": ["Coach Inc", "Kate Spade"], "FFIV": ["F5 Networks", "F5 Inc"],
         "PENN": ["Penn National Gaming", "PENN Entertainment"], "MKC": ["McCormick & Co"],
         "FSLR": ["First Solar Inc", "First Solar, Inc."], "KKR": ["KKR & Co"],
         "LNC": ["Lincoln Financial", "Lincoln National Corp", "Lincoln National Corporation"],
         "SO": ["Southern Company Gas", "Georgia Power"], "HPQ": ["HP Inc"], "CSGP": ["CoStar Group"], "ALB": ["Albemarle Corp"],
         "PDD": ["PDD Holdings"], "PVH": ["PVH Corp"], "EXE": ["Expand Energy Corp", "Chesapeake Energy"], "TTWO": ["Take-Two Interactive"]}
# Written by hand: the 26 the drafting run never reached and the 6 it rejected (2026-11-03).
MANUAL = {
    "CE": ["Celanese"], "KEY": ["KeyCorp", "KeyBank", "Key Bank"], "MMM": ["3M Company", "3M Co", "Post-it", "Scotch-Brite"],
    "NOC": ["Northrop Grumman", "Northrop"], "RSG": ["Republic Services"], "TMUS": ["T-Mobile"],
    "WDC": ["Western Digital"], "WEC": ["WEC Energy"], "WELL": ["Welltower"], "WFC": ["Wells Fargo"],
    "WHR": ["Whirlpool Corp", "Whirlpool Corporation", "Whirlpool"],
    "WM": ["Waste Management"], "WMB": ["Williams Companies", "Williams Cos", "Transco pipeline"],
    "WMT": ["Walmart", "Wal-Mart", "Sam's Club"], "WRB": ["W. R. Berkley", "W.R. Berkley", "WR Berkley", "Berkley"],
    "WSM": ["Williams-Sonoma", "Pottery Barn", "West Elm"], "WST": ["West Pharmaceutical"],
    "WTW": ["Willis Towers Watson", "WTW"], "WU": ["Western Union"], "WY": ["Weyerhaeuser"],
    "WYNN": ["Wynn Resorts", "Wynn Las Vegas", "Wynn Macau", "Wynn"], "XEL": ["Xcel Energy"],
    "XOM": ["ExxonMobil", "Exxon Mobil", "Exxon"], "XRAY": ["Dentsply Sirona", "Dentsply"], "XRX": ["Xerox"],
    "XYL": ["Xylem Inc", "Xylem"], "XYZ": ["Block, Inc.", "Block Inc", "Cash App", "Afterpay", "Square"],
    "YUM": ["Yum! Brands", "Yum Brands", "Taco Bell", "KFC", "Pizza Hut"], "ZBH": ["Zimmer Biomet"],
    "ZBRA": ["Zebra Technologies", "Zebra"], "ZION": ["Zions Bancorporation", "Zions Bank"], "ZTS": ["Zoetis"],
}


def _contains(longer: str, shorter: str) -> bool:
    return longer.lower() != shorter.lower() and re.search(r"(?<!\w)" + re.escape(shorter) + r"(?!\w)", longer, re.I) is not None


_WORDS = None


def _weak(t: str) -> bool:
    """An ambiguous name: listed above, or one ordinary English word (the system dictionary)."""
    global _WORDS
    if _WORDS is None:
        try:
            _WORDS = {w.strip() for w in open("/usr/share/dict/words") if w[:1].islower()}
        except OSError:
            _WORDS = set()
    return t.lower() in AMBIGUOUS or (len(t.split()) == 1 and t.lower() in _WORDS and t.lower() not in DISTINCT)


# rare dictionary words that are, in the news, only the company
DISTINCT = {"assurant", "amentum", "biogen", "copart", "coned", "organon"}


def tighten_rule(sym: str, rule: dict, name: str) -> dict:
    amb = (rule.get("special_rules") or {}).get("ambiguous") or {}
    terms = list(dict.fromkeys([k.strip() for k in (rule.get("primary_keywords") or []) + (amb.get("required_keywords") or [])
                                + EXTRA.get(sym, []) if isinstance(k, str) and k.strip()]))
    if sym in MANUAL:
        terms = list(MANUAL[sym]) + EXTRA.get(sym, [])
    junk = JUNK.get(sym) or set()
    extra = set(EXTRA.get(sym, [])) | set(MANUAL.get(sym, []))
    terms = [t for t in terms if t not in junk and not (sym in JUNK_RE and JUNK_RE[sym].match(t)) and not PRODUCT_RE.search(t)]
    # a bare ticker (exactly the ticker, in capitals: "EBAY" yes, "eBay" no) stays only when the company goes
    # by it ("CVS Health", "AIG") or it was given by hand
    tick = {sym, sym.replace("-", "."), sym.replace("-", "")}
    in_name = sym.upper() in re.findall(r"[A-Z0-9&]+", name or "")
    terms = [t for t in terms if t not in tick or (in_name and len(t) >= 3) or t in extra]
    # GKG writes organisations without punctuation and often without the legal suffix ("ventas inc")
    for t in list(terms):
        bare = re.sub(r"[,.]", "", t).strip()
        short = re.sub(r"[\s,]+(Inc|Incorporated|Corp|Corporation|Co|Company|Ltd|Limited|plc|PLC|NV|N\.V|SA|S\.A|LLC|Holdings|Group)\.?$",
                       "", t, flags=re.I).strip(" ,&")
        for v in (bare, short):
            if v and v not in terms and len(v) >= 4 and not _weak(v) and (v == bare or (len(v.split()) >= 2 and not v.startswith("The "))):
                terms.append(v)
    weak = [t for t in terms if _weak(t)]
    strong = [t for t in terms if t not in weak]
    ids = strong or weak                               # nothing distinctive: the weak names, each checked by the SLM
    ids = [t for t in ids if not any(_contains(t, o) for o in ids if o != t)]   # one mention, one hit
    out = {"symbol": sym, "name": name or rule.get("name") or sym, "primary_keywords": ids, "expansion_keywords": [],
           "tightened": "2026-11-03", "weak_names_not_counted": weak if strong else []}
    if rule.get("dropped_expansion"):
        out["dropped_expansion"] = rule["dropped_expansion"]
    if not strong:                                     # only weak names: each hit is checked by the SLM
        out["special_rules"] = {"ambiguous": {"problem": f"{', '.join(weak)}: also an ordinary word, a surname, a place or another organisation",
                                              "required_keywords": ids, "min_matches": 1}}
    else:
        out["special_rules"] = {}
    return out


def tighten() -> None:
    names = json.loads(NAMES.read_text()) if NAMES.exists() else {}
    syms = sorted({p.stem for p in OUT.glob("*.json")} | set(MANUAL))
    changed, dropped = 0, []
    for s in syms:
        path = OUT / f"{s}.json"
        if s in DROP:
            if path.exists():
                path.rename(path.with_suffix(".dropped"))
            dropped.append(s)
            continue
        rule = json.loads(path.read_text()) if path.exists() else {}
        new = tighten_rule(s, rule, (names.get(s) or {}).get("name") or rule.get("name") or s)
        path.write_text(json.dumps(new, indent=1, ensure_ascii=False))
        changed += 1
    (OUT.parent / "rule_aliases.json").write_text(json.dumps({k: v for k, v in DROP.items() if v and v != k}, indent=1))
    print(f"tightened {changed} rules, dropped {len(dropped)}: {dropped}")


def main(syms: list) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    syms = syms or Path("/Volumes/Data4T/universe/needs_rules.txt").read_text().split()
    info = names(syms)
    log = []
    for i, s in enumerate(syms, 1):
        path = OUT / f"{s}.json"
        if path.exists():
            continue
        t = time.time()
        try:
            rule = draft(s, info[s])
            probs = check(s, rule, info[s])
        except Exception as e:  # noqa: BLE001
            rule, probs = None, [f"{type(e).__name__}: {e}"]
        ok = rule is not None and not any(p.startswith(("no primary", "ambiguous without", "ticker")) or ":" in p[:20] for p in probs)
        if ok:
            path.write_text(json.dumps(rule, indent=1, ensure_ascii=False))
        log.append({"symbol": s, "ok": ok, "problems": probs, "seconds": round(time.time() - t, 1)})
        print(f"PROGRESS rules {i} {len(syms)} {s} {'ok' if ok else 'REJECTED'} {probs} {time.time() - t:.0f}s", flush=True)
    (OUT.parent / "rules_log.json").write_text(json.dumps(log, indent=1))


if __name__ == "__main__":
    if sys.argv[1:2] == ["tighten"]:
        tighten()
    else:
        main(sys.argv[1:])
