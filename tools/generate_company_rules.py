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
    main(sys.argv[1:])
